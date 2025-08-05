import torch
import triton
import triton.language as tl

from hypll.kernels.fc_fwd_kernel import _single_block_fwd


def get_autotune_configs(device=None):
    if device is None:
        device = torch.cuda.current_device()
    cc_major, _ = torch.cuda.get_device_capability(device)
    stage_set = (1,) if cc_major < 8 else (1, 2)  # Turing vs Ampere+

    # (BLOCK_K, BLOCK_M, num_warps)
    tiles = [
        (32, 32, 1),  # tiny rows
        (64, 64, 2),  # mid‑size
        (128, 64, 4),  # long‑K
        (128, 128, 4),  # large K & M
    ]

    configs = []
    for bk, bm, w in tiles:
        for s in stage_set:
            configs.append(
                triton.Config(
                    {"BLOCK_K": bk, "BLOCK_M": bm},
                    num_warps=w,
                    num_stages=s,
                )
            )
    return configs


@triton.jit
def _dnum_dx(
    Z_ptr,  # [K, M]
    XZ_ptr,  # [B, M]
    ZN_ptr,  # [M]      fp32 (‖z_k‖)
    B_ptr,  # [M]      fp32 or dummy when no bias; = exp(2 * cs * r)
    lam,
    dlam_dx,
    cs,  # fp32  (sqrt(curvature))
    m_ids,
    k_ids,
    M,
    K,
    stride_z_k,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    mask_m = m_ids < M
    mask_z = (k_ids[:, None] < K) & mask_m[None, :]
    z_ids = (k_ids * stride_z_k)[:, None] + m_ids[None, :]

    # load values
    zn = tl.load(ZN_ptr + m_ids, mask=mask_m)
    xz = tl.load(XZ_ptr + m_ids, mask=mask_m)
    z = tl.load(Z_ptr + z_ids, mask=mask_z)
    b = tl.zeros((BLOCK_M,), dtype=tl.float32)
    if HAS_BIAS:
        b = tl.load(B_ptr + m_ids, mask=mask_m)

    _sq_p2_1, ed, num = _single_block_fwd(b, lam, zn, xz, cs, HAS_BIAS)

    # waste a few FLOPS when no bias present, negligible
    eb = tl.exp(b)
    ebi = 1 / eb
    _frac = (eb + ebi) / (2 * zn)
    _dp_dx_row = _frac * xz - (eb - ebi)
    dp_dx = dlam_dx[:, None] * _dp_dx_row[None, :] + _frac * lam * z

    _dnum_dx_row = (ed + (1 / ed)) / cs * zn / (_sq_p2_1)
    dnum_dx = _dnum_dx_row[None, :] * dp_dx

    return num, dnum_dx


@triton.autotune(
    configs=get_autotune_configs(),
    key=["K", "M"],
)
@triton.jit
def _poincare_fc_bwd_kernel(
    X_ptr,  # [B, K]   fp32
    Z_ptr,  # [K, M]
    XZ_ptr,  # [B, M]
    ZN_ptr,  # [M]      fp32 (‖z_k‖)
    B_ptr,  # [M]      fp32 or dummy when no bias; = 2 * cs * r
    den_ptr,  # [B]   fp32 – denominator cache
    lam_ptr,  # [B]   fp32 – lambda cache
    dout_ptr,  # [B, M]
    out_ptr,  # [B, K]   fp32 - pointer where to write output derivative
    c,  # fp32  (curvature)
    cs,  # fp32  (sqrt(curvature))
    K: tl.constexpr,
    M: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_xz_b: tl.constexpr,
    stride_z_k: tl.constexpr,
    stride_dout_b: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1)

    # offset pointers
    X_ptr += row * stride_x_b
    out_ptr += row * stride_out_b
    XZ_ptr += row * stride_xz_b
    dout_ptr += row * stride_dout_b

    k_ids = col * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_ids < K

    # load values for the entire kernel execution
    x = tl.load(X_ptr + k_ids, mask=mask_k)
    lam = tl.load(lam_ptr + row)
    den = tl.load(den_ptr + row)
    dlam_dx = c * lam * lam * x

    # pre-compute derivative of den_b w.r.t. x[b, k:k+BLOCK_K]
    dden_dx = tl.zeros_like(x)
    m_ids = tl.arange(0, BLOCK_M)
    for _ in range(0, M, BLOCK_M):
        num, dnum_dx = _dnum_dx(
            Z_ptr,
            XZ_ptr,
            ZN_ptr,
            B_ptr,
            lam,
            dlam_dx,
            cs,
            m_ids,
            k_ids,
            M,
            K,
            stride_z_k,
            HAS_BIAS,
            BLOCK_M,
        )
        dden_dx += tl.sum(num * dnum_dx, axis=1)
        m_ids += BLOCK_M

    # finish den derivative computation
    dden_dx *= c / (den - 1)

    # start computing derivative of loss w.r.t. x
    out = tl.zeros_like(x)
    m_ids = tl.arange(0, BLOCK_M)
    for _ in range(0, M, BLOCK_M):
        num, dnum_dx = _dnum_dx(
            Z_ptr,
            XZ_ptr,
            ZN_ptr,
            B_ptr,
            lam,
            dlam_dx,
            cs,
            m_ids,
            k_ids,
            M,
            K,
            stride_z_k,
            HAS_BIAS,
            BLOCK_M,
        )

        mask_m = m_ids < M
        dout = tl.load(dout_ptr + m_ids, mask=mask_m)

        # _t1 = den * dnum_dx
        # _t2 = dden_dx[:, None] * num[None, :]
        # dy_dx = (t1 - t2) / (den * den)
        # out += tl.sum(dout[None, :] * dy_dx, axis=1)
        # we can reduce the number of operations by first multiplying, then summing:
        _t1 = den * tl.sum(dout[None, :] * dnum_dx, axis=1)  # [K,]
        _t2 = dden_dx * tl.sum(dout * num)  # [K,]
        iden = 1.0 / den
        out += _t1 * iden - _t2 * iden * iden

        m_ids += BLOCK_M

    tl.store(out_ptr + k_ids, out)


def poincare_fc_bwd_triton(dout, x, z, xz, zn, b, den, lam, c, cs, has_bias):
    # TODO: sanity checks
    B, K = x.shape
    _, M = z.shape

    c = c if isinstance(c, float) else c.item()
    cs = cs if isinstance(cs, float) else cs.item()

    out = torch.empty_like(x)
    grid = lambda meta: (B, triton.cdiv(K, meta["BLOCK_K"]))
    _poincare_fc_bwd_kernel[grid](
        x,
        z,
        xz,
        zn,
        b,
        den,
        lam,
        dout,
        out,
        c,
        cs,
        K,
        M,
        x.stride(0),
        out.stride(0),
        xz.stride(0),
        z.stride(0),
        dout.stride(0),
        has_bias,
    )

    return out
