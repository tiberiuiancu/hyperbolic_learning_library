from typing import Tuple
import torch
import triton
import triton.language as tl

from hypll.kernels.fc.fc_fwd_kernel import single_block_fwd
from hypll.kernels.utils import Tensor1D, Tensor2D, validate_tensors
from hypll.utils.memory import gpu_memory_pool


def get_autotune_configs():
    return [
        triton.Config({"BLOCK_M": 4}),
        triton.Config({"BLOCK_M": 8}),
        triton.Config({"BLOCK_M": 16}),
        triton.Config({"BLOCK_M": 32}),
        triton.Config({"BLOCK_M": 64}),
        triton.Config({"BLOCK_M": 128}),
        triton.Config({"BLOCK_M": 256}),
        triton.Config({"BLOCK_M": 512}),
        triton.Config({"BLOCK_M": 1024}),
    ]


@triton.autotune(
    configs=get_autotune_configs(),
    key=["B", "M"],
    reset_to_zero=["T7_sum_ptr", "dr_ptr"],  # we atomically add to these
)
@triton.jit
def _poincare_fc_bwd_kernel(  # 1D grid over B
    # inputs
    dout_ptr,  # [B, M]
    Y_ptr,  # [B, M]   (unprojected y)
    num_ptr,  # [B, M]   (forward 'num' cache) — used only for T1_num
    yn_ptr,  # [B]
    XZ_ptr,  # [B, M]
    zn_ptr,  # [M]
    b_ptr,  # [M]
    lam_ptr,  # [B]
    den_ptr,  # [B]
    # outputs
    T5_ptr,  # [B, M]   (written; used as TEMP in pass 2, final T5 in pass 3)
    T8_ptr,  # [B, M]
    T7_sum_ptr,  # [M]      (atomic add over B)
    T4_sum_ptr,  # [B]
    dr_ptr,  # [M]      (atomic add over B)
    # scalars (runtime, NOT constexpr)
    max_norm,  # fp32
    c,  # fp32
    cs,  # fp32
    # strides
    dout_stride_b,
    Y_stride_b,
    num_stride_b,
    XZ_stride_b,
    T5_stride_b,
    T8_stride_b,
    # dims
    M,
    B,
    BLOCK_M: tl.constexpr,
):
    b_id = tl.program_id(0)

    # row offsets
    dout_row = dout_ptr + b_id * dout_stride_b
    Y_row = Y_ptr + b_id * Y_stride_b
    num_row = num_ptr + b_id * num_stride_b
    XZ_row = XZ_ptr + b_id * XZ_stride_b
    T5_row = T5_ptr + b_id * T5_stride_b
    T8_row = T8_ptr + b_id * T8_stride_b

    lam = tl.load(lam_ptr + b_id)
    den = tl.load(den_ptr + b_id)
    yn = tl.load(yn_ptr + b_id)
    den1 = den * (den - 1)

    # -------------------------
    # Pass 1: dout_y_sum[b]
    # -------------------------
    dout_y_sum = 0.0
    for m0 in range(0, M, BLOCK_M):
        offs = tl.arange(0, BLOCK_M) + m0
        mask = offs < M
        dout = tl.load(dout_row + offs, mask=mask, other=0.0)
        Y = tl.load(Y_row + offs, mask=mask, other=0.0)
        dout_y_sum += tl.sum((dout * Y * mask.to(dout.dtype)).to(tl.float32))

    # -------------------------
    # Pass 2: T1_num[b] and stash T1 in T5_row (as temp)
    #   T1_num = sum_m num[m] * T1[m]
    # -------------------------
    T1_num_acc = 0.0
    for m0 in range(0, M, BLOCK_M):
        offs = tl.arange(0, BLOCK_M) + m0
        mask = offs < M
        dout = tl.load(dout_row + offs, mask=mask, other=0.0)
        Y = tl.load(Y_row + offs, mask=mask, other=0.0)
        num = tl.load(num_row + offs, mask=mask, other=0.0)

        T1 = tl.where(
            yn < max_norm,
            dout,
            max_norm * (dout - Y * (dout_y_sum / (yn * yn))) / yn,
        )
        # stash T1 for reuse
        tl.store(T5_row + offs, T1, mask=mask)

        T1_num_acc += tl.sum((num * T1 * mask.to(T1.dtype)).to(tl.float32))

    T1_num = T1_num_acc

    # -------------------------
    # Pass 3: outputs + reductions
    # -------------------------
    T4_sum_acc = 0.0
    for m0 in range(0, M, BLOCK_M):
        offs = tl.arange(0, BLOCK_M) + m0
        mask = offs < M

        # loads
        XZ = tl.load(XZ_row + offs, mask=mask, other=0.0)
        zn = tl.load(zn_ptr + offs, mask=mask, other=1.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        T1 = tl.load(T5_row + offs, mask=mask, other=0.0)  # previously stashed

        # forward invariants
        sq_p2_1, log_p_sq, eb_sum, eb_dif, ed_sum, num_f = single_block_fwd(b, lam, zn, XZ, cs)
        znc = tl.clamp(zn, 1e-15, 1e30)

        ed_div = ed_sum / (2 * cs)
        eb_div = eb_sum / znc

        # T2, T3
        T2 = (T1 - c * num_f / den1 * T1_num) / den
        T3 = T2 * ed_div * znc / sq_p2_1

        # T4 and row reduction
        T4 = T3 * (cs * XZ * eb_div - eb_dif)
        T4_sum_acc += tl.sum((T4 * mask.to(T4.dtype)).to(tl.float32)) * lam * lam * c

        # T5 -> write (overwrites temp)
        T5 = cs * lam * T3 * eb_div
        tl.store(T5_row + offs, T5, mask=mask)

        # T6 / T7 / T8
        T6 = T1 * (1 - c * num_f * num_f / den1) * ed_div / den
        tmp = eb_sum * cs * lam / sq_p2_1

        T7 = tl.where(zn > 1e-15, T6 * (2 * log_p_sq - tmp * XZ / znc) / znc, 0.0)
        tl.atomic_add(T7_sum_ptr + offs, T7, mask=mask)

        T8 = T6 * tmp
        tl.store(T8_row + offs, T8, mask=mask)

        # dr
        T9 = T6 * cs / znc / sq_p2_1 * (eb_dif * c * lam / znc - eb_sum * (lam - 1))
        tl.atomic_add(dr_ptr + offs, T9, mask=mask)

    # per-row write
    tl.store(T4_sum_ptr + b_id, T4_sum_acc)


@validate_tensors
def poincare_fc_bwd_triton(
    dout: Tensor2D,
    Y: Tensor2D,
    X: Tensor2D,
    Z: Tensor2D,
    XZ: Tensor2D,
    zn: Tensor1D,
    b: Tensor1D,
    lam: Tensor1D,
    num: Tensor2D,
    den: Tensor1D,
    yn: Tensor1D,
    max_norm: float,
    c: float,
    cs: float,
):
    assert Y.shape == dout.shape
    B, K = X.shape
    K2, M = Z.shape
    assert K == K2

    dtype = dout.dtype

    c = c if isinstance(c, float) else c.item()
    cs = cs if isinstance(cs, float) else cs.item()

    # preallocate GPU buffers
    dtype = dout.dtype
    T4_sum = gpu_memory_pool.get_shared("T4_sum", lam.shape, dtype)
    T5 = gpu_memory_pool.get_shared("T5", XZ.shape, dtype)
    T7_sum = gpu_memory_pool.get_shared("T7_sum", b.shape, dtype)
    T8 = gpu_memory_pool.get_shared("T7_sum", XZ.shape, dtype)
    dr = torch.zeros_like(b)
    T4_sum.zero_()
    T7_sum.zero_()

    # 1D launch over B
    grid = (B,)
    _poincare_fc_bwd_kernel[grid](
        # inputs
        dout,
        Y,
        num,
        yn,
        XZ,
        zn,
        b,
        lam,
        den,
        # outputs
        T5,
        T8,
        T7_sum,
        T4_sum,
        dr,
        # consts
        float(max_norm),
        float(c),
        float(cs),
        # strides
        dout.stride(0),
        Y.stride(0),
        num.stride(0),
        XZ.stride(0),
        T5.stride(0),
        T8.stride(0),
        # dims
        M,
        B,
    )

    if X.requires_grad:
        dX = torch.addmm(X * T4_sum[:, None], T5, Z.T)
    else:
        dX = None
    dZ = torch.addmm(Z * T7_sum[None, :], X.T, T8)
    return dX, dZ, dr
