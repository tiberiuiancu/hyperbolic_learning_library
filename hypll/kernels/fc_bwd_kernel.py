import torch
import triton
import triton.language as tl

from hypll.kernels.fc_fwd_kernel import single_block_fwd


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


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16}),
        triton.Config({"BLOCK_M": 32}),
        triton.Config({"BLOCK_M": 64}),
        triton.Config({"BLOCK_M": 128}),
        triton.Config({"BLOCK_M": 256}),
    ],
    key=["B", "M"],
)
@triton.jit
def _poincare_fc_bwd_kernel(
    ##### forward pass cache
    Y_ptr,
    XZ_ptr,
    zn_ptr,
    b_ptr,
    lam_ptr,
    den_ptr,
    yn_ptr,
    dout_ptr,
    c,
    cs,
    max_norm,
    # backward pass precomputations
    y_sum_ptr,
    ##### outputs
    T5_ptr,
    T10_ptr,
    T911_norm_ptr,
    T4_norm_ptr,
    dr_ptr,
    ##### strides
    Y_stride_b: tl.constexpr,
    XZ_stride_b: tl.constexpr,
    dout_stride_b: tl.constexpr,
    cslam_T5_stride_b: tl.constexpr,
    T10_stride_b: tl.constexpr,
    ##### dimensions
    M: tl.constexpr,
    B: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # calculate which block we're working on
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    mask_m = offs_m < M

    # offset pointers to correct row
    XZ_ptr += pid_b * XZ_stride_b
    Y_ptr += pid_b * Y_stride_b
    dout_ptr += pid_b * dout_stride_b
    T5_ptr += pid_b * cslam_T5_stride_b
    T10_ptr += pid_b * T10_stride_b

    # perform scalar reads
    yn = tl.load(yn_ptr + pid_b)
    den = tl.load(den_ptr + pid_b)
    y_sum = tl.load(y_sum_ptr + pid_b)
    lam = tl.load(lam_ptr + pid_b)

    # perform BLOCK_M reads
    Y = tl.load(Y_ptr + offs_m, mask=mask_m, other=0.0)
    XZ = tl.load(XZ_ptr + offs_m, mask=mask_m, other=0.0)
    dout = tl.load(dout_ptr + offs_m, mask=mask_m, other=0.0)
    zn = tl.load(zn_ptr + offs_m, mask=mask_m, other=1.0)
    b = tl.load(b_ptr + offs_m, mask=mask_m, other=0.0)

    sq_p2_1, log_p_sq, eb_sum, eb_dif, ed_dif, P, num = single_block_fwd(b, lam, zn, XZ, cs)

    # calculate outputs for dx
    deni = 1 / den
    deni_1 = deni * (1 / (den - 1))
    yni = 1 / yn
    zni = 1 / zn
    if yn > max_norm:
        T1 = dout * max_norm * yni * (1 - Y * Y * yni * yni)
    else:
        T1 = dout
    ed_div = ed_dif / (2 * cs)
    t1 = c * deni_1 * y_sum
    T2 = (T1 * deni - num * t1) * ed_div
    T3 = T2 * zn / sq_p2_1
    eb_div = eb_sum * zni
    T5 = cs * lam * T3 * eb_div
    T4 = c * T3 * (cs * XZ * eb_div - eb_dif)
    T4_norm = tl.sum(T4) * lam * lam

    # write outputs for dx
    tl.store(T5_ptr + offs_m, T5, mask=mask_m)
    tl.atomic_add(T4_norm_ptr + pid_b, T4_norm)

    # calculate outputs for dz
    T6 = T1  # TODO: decrease T counters for conistency, and correct derivatives as well
    T7 = deni * (1 - c * num * num * deni_1)
    T8 = T6 * T7 * ed_div
    T9 = T8 * zni * log_p_sq
    _tmp4 = T8 / sq_p2_1
    _tmp5 = _tmp4 * eb_sum * cs * lam * 0.5
    T10 = _tmp5 * zn * zn
    T11 = _tmp5 * XZ

    # write output for dz
    tl.store(T10_ptr + offs_m, T10, mask=mask_m)
    tl.atomic_add(T911_norm_ptr + offs_m, T9 + T11, mask=mask_m)

    # calculate outputs for dr
    T12 = _tmp4 * zn
    dr_tmp = 2 * cs * ((c * eb_dif * zni + eb_sum) * lam - eb_sum) * T12
    tl.atomic_add(dr_ptr + offs_m, dr_tmp, mask=mask_m)


def poincare_fc_bwd_triton(dout, Y, X, Z, XZ, zn, b, lam, den, yn, max_norm, c, cs):
    # TODO: sanity checks
    B, K = X.shape
    K2, M = Z.shape

    assert K == K2

    c = c if isinstance(c, float) else c.item()
    cs = cs if isinstance(cs, float) else cs.item()

    dX = torch.empty_like(X)
    dZ = torch.empty_like(Z)
    dr = torch.zeros_like(b)

    T4_norm = torch.zeros_like(lam)
    T5 = torch.empty_like(XZ)
    T911_norm = torch.zeros_like(b)
    T10 = torch.empty_like(T5)

    y_sum = Y.sum(dim=1)

    grid = lambda meta: (B, triton.cdiv(M, meta["BLOCK_M"]))
    _poincare_fc_bwd_kernel[grid](
        Y,
        XZ,
        zn,
        b,
        lam,
        den,
        yn,
        dout,
        c,
        cs,
        max_norm,
        y_sum,
        T5,
        T10,
        T911_norm,
        T4_norm,
        dr,
        Y.stride(0),
        XZ.stride(0),
        dout.stride(0),
        T5.stride(0),
        T10.stride(0),
        M,
        B,
    )

    # perform the matrix multiply and addition in one kernel call
    dX = torch.addmm(X * T4_norm[:, None], T5, Z.T)
    dZ = torch.addmm(Z * T911_norm[None, :], X.T, T10)

    return dX, dZ, dr
