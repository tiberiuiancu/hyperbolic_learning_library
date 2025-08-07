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

    sq_p2_1, _1, eb, ebi, ed, edi, num = _single_block_fwd(b, lam, zn, xz, cs)

    _frac = cs * (eb + ebi) / zn
    dp_dx = 0.5 * (
        _frac[None, :] * (xz[None, :] * dlam_dx[:, None] + lam * z) - (eb - ebi) * dlam_dx[:, None]
    )
    dnum_dd = (ed + edi) / cs
    dd_dp = zn / sq_p2_1
    dnum_dx = dnum_dd * dd_dp * dp_dx

    return num, dnum_dx


@triton.autotune(configs=get_autotune_configs(), key=["K", "M"])
@triton.jit
def _poincare_fc_bwd_dx_kernel(
    x_ptr,  # [B, K]   fp32
    z_ptr,  # [K, M]
    xz_ptr,  # [B, M]
    zn_ptr,  # [M]      fp32 (‖z_k‖)
    b_ptr,  # [M]      fp32 or dummy when no bias; = 2 * cs * r
    lam_ptr,  # [B]   fp32 – lambda cache
    den_ptr,  # [B]   fp32 – denominator cache
    dout_ptr,  # [B, M]
    dx_ptr,  # [B, K]   fp32 - pointer where to write dx
    c,  # fp32  (curvature)
    cs,  # fp32  (sqrt(curvature))
    K: tl.constexpr,
    M: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_dx_b: tl.constexpr,
    stride_xz_b: tl.constexpr,
    stride_z_k: tl.constexpr,
    stride_dout_b: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Each program calculates the derivative of the loss w.r.t.
    a block of size K within one row of x"""
    row = tl.program_id(0)
    col = tl.program_id(1)

    # offset pointers
    x_ptr += row * stride_x_b
    dx_ptr += row * stride_dx_b
    xz_ptr += row * stride_xz_b
    dout_ptr += row * stride_dout_b
    lam_ptr += row
    den_ptr += row

    k_ids = col * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_ids < K

    # load values for the entire kernel execution
    x = tl.load(x_ptr + k_ids, mask=mask_k, other=0.0)
    lam = tl.load(lam_ptr)
    den = tl.load(den_ptr)
    iden = 1 / den
    dlam_dx = c * lam * lam * x

    # pre-compute derivative of den_b w.r.t. x[b, k:k+BLOCK_K]
    dden_dx = tl.zeros_like(x)
    m_ids = tl.arange(0, BLOCK_M)
    for _ in range(0, M, BLOCK_M):
        num, dnum_dx = _dnum_dx(
            z_ptr,
            xz_ptr,
            zn_ptr,
            b_ptr,
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

    dx = tl.zeros_like(x)
    m_ids = tl.arange(0, BLOCK_M)
    for _ in range(0, M, BLOCK_M):
        num, dnum_dx = _dnum_dx(
            z_ptr,
            xz_ptr,
            zn_ptr,
            b_ptr,
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

        # calculate dx
        _t1 = dnum_dx * iden
        _t2 = dden_dx[:, None] * num[None, :] * iden * iden
        dy_dx = _t1 - _t2
        dx += tl.sum(dout[None, :] * dy_dx, axis=1)

        m_ids += BLOCK_M

    tl.store(dx_ptr + k_ids, dx, mask=mask_k)


@triton.autotune(configs=get_autotune_configs(), key=["K", "M"])
@triton.jit
def _poincare_fc_bwd_dz_dr_kernel(
    x_ptr,
    z_ptr,
    b_ptr,
    dz_ptr,
    dr_ptr,
    xz_ptr,
    zn_ptr,
    lam_ptr,
    den_ptr,
    dout_ptr,
    c,
    cs,
    stride_x_b: tl.constexpr,
    stride_z_k: tl.constexpr,
    stride_dz_k: tl.constexpr,
    stride_xz_b: tl.constexpr,
    stride_dout_b: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Each program computes a tile of size BLOCK_K, BLOCK_M of dz, and BLOCK_M of db"""
    row = tl.program_id(0)
    col = tl.program_id(1)

    k_ids = row * BLOCK_K + tl.arange(0, BLOCK_K)
    m_ids = col * BLOCK_M + tl.arange(0, BLOCK_M)
    z_ids = (k_ids * stride_z_k)[:, None] + m_ids[None, :]

    mask_k = k_ids < K
    mask_m = m_ids < M
    mask_z = mask_k[:, None] & mask_m[None, :]

    # load for the entire kernel execution
    z = tl.load(z_ptr + z_ids, mask=mask_z, other=0.0)

    dz_acc = tl.zeros((BLOCK_K, BLOCK_M), dtype=tl.float32)
    dr_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for i in range(B):
        # load values
        x = tl.load(x_ptr + k_ids, mask=mask_k, other=0.0)
        zn = tl.load(zn_ptr + m_ids, mask=mask_m, other=1.0)
        xz = tl.load(xz_ptr + m_ids, mask=mask_m, other=0.0)
        b = tl.load(b_ptr + m_ids, mask=mask_m, other=0.0)
        lam = tl.load(lam_ptr + i)
        den = tl.load(den_ptr + i)
        dout = tl.load(dout_ptr + m_ids, mask=mask_m, other=0.0)

        # calculate values from forward pass
        sq_p2_1, log_p_sq, eb, ebi, ed, edi, num = _single_block_fwd(b, lam, zn, xz, cs)

        # dy/dz
        iden = 1 / den
        iden_1 = 1 / (den - 1)
        izn = 1 / zn
        dnum_dd = (ed + edi) / cs
        dd_dp = zn / sq_p2_1
        dzn_dz = z * izn[None, :]
        dp_dz = (
            0.5
            * cs
            * lam
            * ((eb + ebi) * izn)[None, :]
            * (x[:, None] - dzn_dz * (xz * izn)[None, :])
        )
        dnum_dz = dnum_dd * (log_p_sq * dzn_dz + dd_dp * dp_dz)
        _dy_cache = iden * (1 - c * num * num * iden * iden_1)
        dy_dz = dnum_dz * _dy_cache[None, :]
        dz_acc += dy_dz * dout[None, :]

        # dy/dr
        _dp_cache_1 = cs * lam * xz / zn
        _dp_cache_2 = lam - 1
        dp_dr = cs * (eb * (_dp_cache_1 - _dp_cache_2) - ebi * (_dp_cache_1 + _dp_cache_2))
        dnum_dr = dnum_dd * dd_dp * dp_dr
        dy_dr = dnum_dr * _dy_cache
        dr_acc += dy_dr * dout

        # move pointer to next line for next batch item
        x_ptr += stride_x_b
        xz_ptr += stride_xz_b
        dout_ptr += stride_dout_b

    mask_dz = (k_ids[:, None] < K) & (m_ids[None, :] < M)
    offs_dz = (k_ids[:, None] * stride_dz_k) + (m_ids[None, :])
    mask_dr = m_ids < M
    offs_dr = m_ids
    tl.store(dz_ptr + offs_dz, dz_acc, mask=mask_dz)
    tl.store(dr_ptr + offs_dr, dr_acc, mask=mask_dr)


def poincare_fc_bwd_triton(dout, x, z, xz, zn, b, lam, den, c, cs):
    # TODO: sanity checks
    B, K = x.shape
    _, M = z.shape

    c = c if isinstance(c, float) else c.item()
    cs = cs if isinstance(cs, float) else cs.item()

    dx = torch.empty_like(x)
    dz = torch.empty_like(z)
    dr = torch.empty_like(b)

    grid = lambda meta: (B, triton.cdiv(K, meta["BLOCK_K"]))
    _poincare_fc_bwd_dx_kernel[grid](
        x,
        z,
        xz,
        zn,
        b,
        lam,
        den,
        dout,
        dx,
        c,
        cs,
        K,
        M,
        x.stride(0),
        dx.stride(0),
        xz.stride(0),
        z.stride(0),
        dout.stride(0),
    )

    grid = lambda meta: (triton.cdiv(K, meta["BLOCK_K"]), triton.cdiv(M, meta["BLOCK_M"]))
    _poincare_fc_bwd_dz_dr_kernel[grid](
        x,
        z,
        b,
        dz,
        dr,
        xz,
        zn,
        lam,
        den,
        dout,
        c,
        cs,
        x.stride(0),
        z.stride(0),
        dz.stride(0),
        xz.stride(0),
        dout.stride(0),
        B,
        K,
        M,
    )

    return dx, dz, dr
