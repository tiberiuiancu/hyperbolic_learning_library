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
        # (64, 64, 2),  # mid‑size
        # (128, 64, 4),  # long‑K
        # (128, 128, 4),  # large K & M
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
):
    mask_m = m_ids < M
    mask_k = k_ids < K
    mask_z = mask_k[:, None] & mask_m[None, :]
    z_ids = (k_ids * stride_z_k)[:, None] + m_ids[None, :]

    # load values
    zn = tl.load(ZN_ptr + m_ids, mask=mask_m, other=1.0)
    xz = tl.load(XZ_ptr + m_ids, mask=mask_m, other=0.0)
    z = tl.load(Z_ptr + z_ids, mask=mask_z, other=0.0)
    b = tl.load(B_ptr + m_ids, mask=mask_m, other=0.0)

    sq_p2_1, eb, ebi, ed, edi, num = _single_block_fwd(b, lam, zn, xz, cs)

    _frac = cs * (eb + ebi) / zn
    dp_dx = 0.5 * (_frac[None, :] * (xz[None, :] * dlam_dx + lam * z) - (eb - ebi) * dlam_dx)

    dnum_dx = (ed + edi) / cs * zn / sq_p2_1 * dp_dx

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
    lam_ptr,  # [B]   fp32 – lambda cache
    den_ptr,  # [B]   fp32 – denominator cache
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
    lam_ptr += row
    den_ptr += row

    k_ids = col * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_ids < K

    # load values for the entire kernel execution
    x = tl.load(X_ptr + k_ids, mask=mask_k, other=0.0)
    lam = tl.load(lam_ptr)
    den = tl.load(den_ptr)
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
        )

        dden_dx += tl.sum(num[None, :] * dnum_dx, axis=1)
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
        )

        mask_m = m_ids < M
        dout = tl.load(dout_ptr + m_ids, mask=mask_m, other=0.0)

        iden = 1 / den
        _t1 = dnum_dx * iden
        _t2 = dden_dx[:, None] * num[None, :] * iden * iden
        dy_dx = _t1 - _t2
        out += tl.sum(dout[None, :] * dy_dx, axis=1)

        m_ids += BLOCK_M

    tl.store(out_ptr + k_ids, out, mask=mask_k)


def poincare_fc_bwd_triton(dout, x, z, xz, zn, b, lam, den, c, cs):
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
        lam,
        den,
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
    )

    return out
