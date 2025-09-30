import torch
import triton
import triton.language as tl

from hypll.kernels.fc_fwd_kernel import single_block_fwd


def get_autotune_configs():
    return [
        triton.Config({"BLOCK_M": 16}),
        triton.Config({"BLOCK_M": 32}),
        triton.Config({"BLOCK_M": 64}),
        triton.Config({"BLOCK_M": 128}),
        triton.Config({"BLOCK_M": 256}),
    ]


@triton.autotune(
    configs=get_autotune_configs(),
    key=["B", "M"],
)
@triton.jit
def _poincare_fc_bwd_kernel(
    ##### forward pass cache
    T1_ptr,
    T1_num_ptr,
    XZ_ptr,
    zn_ptr,
    b_ptr,
    lam_ptr,
    den_ptr,
    c,
    cs,
    ##### outputs
    T5_ptr,
    T8_ptr,
    T79_sum_ptr,
    T4_norm_ptr,
    dr_ptr,
    ##### strides
    T1_stride_b: tl.constexpr,
    XZ_stride_b: tl.constexpr,
    T5_stride_b: tl.constexpr,
    T8_stride_b: tl.constexpr,
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
    T1_ptr += pid_b * T1_stride_b
    T5_ptr += pid_b * T5_stride_b
    T8_ptr += pid_b * T8_stride_b

    # perform scalar reads
    den = tl.load(den_ptr + pid_b)
    lam = tl.load(lam_ptr + pid_b)
    T1_num = tl.load(T1_num_ptr + pid_b)

    # perform BLOCK_M reads
    T1 = tl.load(T1_ptr + offs_m, mask=mask_m, other=0.0)
    XZ = tl.load(XZ_ptr + offs_m, mask=mask_m, other=0.0)
    zn = tl.load(zn_ptr + offs_m, mask=mask_m, other=1.0)
    b = tl.load(b_ptr + offs_m, mask=mask_m, other=0.0)

    sq_p2_1, log_p_sq, eb_sum, eb_dif, ed_dif, num = single_block_fwd(b, lam, zn, XZ, cs)

    # calculate outputs for dx
    deni = 1 / den
    deni_1 = deni * (1 / (den - 1))
    zni = 1 / zn

    # T2
    T2 = deni * (T1 - c * num * deni_1 * T1_num)

    # T3
    ed_div = ed_dif / (2 * cs)
    T3 = T2 * ed_div * zn / sq_p2_1

    # T4: multiply by lambda after summation to save compute
    eb_div = eb_sum * zni
    T4 = c * T3 * (cs * XZ * eb_div - eb_dif)
    T4_norm = tl.sum(T4) * lam * lam

    # T5
    T5 = cs * lam * T3 * eb_div

    # write outputs for dx
    tl.atomic_add(T4_norm_ptr + pid_b, T4_norm)
    tl.store(T5_ptr + offs_m, T5, mask=mask_m)

    # calculate outputs for dz
    T6 = T1 * deni * (1 - c * num * num * deni_1) * ed_div
    T7 = T6 * zni * log_p_sq
    _tmp1 = T6 / sq_p2_1
    _tmp2 = _tmp1 * eb_sum * cs * lam * 0.5
    T8 = _tmp2 * zn * zn
    T9 = _tmp2 * XZ

    # write output for dz
    tl.store(T8_ptr + offs_m, T8, mask=mask_m)
    tl.atomic_add(T79_sum_ptr + offs_m, T7 + T9, mask=mask_m)

    # calculate outputs for dr
    T10 = _tmp1 * zn
    dr_tmp = 2 * cs * ((c * eb_dif * zni + eb_sum) * lam - eb_sum) * T10
    tl.atomic_add(dr_ptr + offs_m, dr_tmp, mask=mask_m)


@triton.autotune(
    configs=get_autotune_configs(),
    key=["B", "M"],
)
@triton.jit
def _dL_dY(
    # inputs
    dout_ptr,
    dout_y_sum_ptr,
    num_ptr,
    yn_ptr,
    max_norm,
    # outputs
    T1_ptr,
    T1_num_ptr,
    # strides
    dout_stride_b: tl.constexpr,
    Y_stride_b: tl.constexpr,
    num_stride_b: tl.constexpr,
    T1_stride_b: tl.constexpr,
    # constants
    B: tl.constexpr,
    M: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    dout_ptr += dout_stride_b * pid_b
    num_ptr += num_stride_b * pid_b
    T1_ptr += T1_stride_b * pid_b

    offs_m = tl.arange(0, BLOCK_M) + BLOCK_M * pid_m
    mask_m = offs_m < M

    dout = tl.load(dout_ptr + offs_m, mask=mask_m, other=0.0)
    num = tl.load(num_ptr + offs_m, mask=mask_m, other=0.0)
    yn = tl.load(yn_ptr + pid_b)

    if yn < max_norm:
        T1 = dout
    else:
        dout_y_sum = tl.load(dout_y_sum_ptr + pid_b)
        yni = 1 / yn
        T1 = max_norm * yni * (dout - yni * yni * dout_y_sum)

    tl.atomic_add(T1_num_ptr + pid_b, tl.sum(num * T1))
    tl.store(T1_ptr + offs_m, T1, mask=mask_m)


def poincare_fc_bwd_triton(dout, Y, X, Z, XZ, zn, b, lam, num, den, yn, max_norm, c, cs):
    # TODO: sanity checks
    B, K = X.shape
    K2, M = Z.shape

    assert K == K2

    c = c if isinstance(c, float) else c.item()
    cs = cs if isinstance(cs, float) else cs.item()

    # get dL_dY
    T1 = torch.empty_like(XZ)
    T1_num = torch.zeros_like(lam)
    dout_y_sum = torch.einsum("ij,ij->i", dout, Y)
    grid = lambda meta: (B, triton.cdiv(M, meta["BLOCK_M"]))
    _dL_dY[grid](
        dout,
        dout_y_sum,
        num,
        yn,
        max_norm,
        T1,
        T1_num,
        dout.stride(0),
        Y.stride(0),
        num.stride(0),
        T1.stride(0),
        B,
        M,
    )

    dX = torch.empty_like(X)
    dZ = torch.empty_like(Z)
    dr = torch.zeros_like(b)

    T4_norm = torch.zeros_like(lam)
    T5 = torch.empty_like(XZ)
    T79_norm = torch.zeros_like(b)
    T8 = torch.empty_like(T5)

    _poincare_fc_bwd_kernel[grid](
        T1,
        T1_num,
        XZ,
        zn,
        b,
        lam,
        den,
        c,
        cs,
        T5,
        T8,
        T79_norm,
        T4_norm,
        dr,
        T1.stride(0),
        XZ.stride(0),
        T5.stride(0),
        T8.stride(0),
        M,
        B,
    )

    # perform the matrix multiply and addition in one kernel call
    dX = torch.addmm(X * T4_norm[:, None], T5, Z.T)
    dZ = torch.addmm(Z * T79_norm[None, :], X.T, T8)

    return dX, dZ, dr
