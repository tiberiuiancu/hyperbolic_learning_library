"""
Modified version of the triton tutorial for matmul:
https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
"""

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def get_autotune_config():
    return [
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4},
            num_stages=2,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 4},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 4},
            num_stages=3,
            num_warps=4,
        ),
        # For larger K, still modest tile sizes to avoid register pressure
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 4},
            num_stages=3,
            num_warps=4,
        ),
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def gemm_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    v_ptr,
    o_ptr,
    # whether A and/or B are given transposed
    a_transp: tl.constexpr,
    b_transp: tl.constexpr,
    # Matrix dimensions
    M,
    N,
    K,
    # strides (first index stride, then second index stride)
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_vm,
    stride_vn,
    stride_om,
    stride_on,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Compute O = v * C + A @ B
    A: (M, K) if not transposed, else (K, M) with same stride order (am, ak)
    B: (K, N) if not transposed, else (N, K) with same stride order (bn, bk)
    C, O: (M, N)
    """

    # -------------------- program id mapping --------------------
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -------------------- assumptions (helps codegen) --------------------
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # -------------------- tile offsets --------------------
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    a_inc_k = BLOCK_SIZE_K * stride_ak

    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    b_inc_k = BLOCK_SIZE_K * stride_bk

    # -------------------- main loop --------------------
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask_row = offs_k[None, :] < K - k * BLOCK_SIZE_K
        k_mask_col = offs_k[:, None] < K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=k_mask_row, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask_col, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += a_inc_k
        b_ptrs += b_inc_k

    # -------------------- epilogue: O = v * C + AB --------------------
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    v_ptrs = v_ptr + stride_vm * offs_m[:, None] + stride_vn * offs_n[None, :]
    o_ptrs = o_ptr + stride_om * offs_m[:, None] + stride_on * offs_n[None, :]
    mask_mn = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    c = tl.load(c_ptrs, mask=mask_mn, other=0.0)
    v = tl.load(v_ptrs, mask=mask_mn, other=0.0)
    o = c * v + accumulator
    tl.store(o_ptrs, o.to(c.dtype), mask=mask_mn)


def addmm(v, C, A, B, out=None, a_transp: bool = False, b_transp: bool = False):
    """
    out = v * C + A @ B

    v broadcast: scalar, (M,1), (1,N), (M,N)
    dtypes: A,B,C,v,out on same device; accum in fp32; store to out.dtype (defaults to A.dtype)
    """
    assert A.ndim == B.ndim == C.ndim == 2
    if not a_transp:
        M, K = A.shape
    else:
        K, M = A.shape
    if not b_transp:
        Kb, N = B.shape
    else:
        N, Kb = B.shape
    assert K == Kb and C.shape == (M, N)

    assert A.is_contiguous()
    assert B.is_contiguous()
    assert C.is_contiguous()

    # make broadcast work via zero strides (no materialization)
    if not torch.is_tensor(v):
        v = torch.tensor(v, dtype=C.dtype, device=C.device)
    if v.ndim == 0:
        v_view = v.view(1, 1)
    elif v.shape == (M, 1) or v.shape == (1, N) or v.shape == (M, N):
        v_view = v
    elif v.numel() == 1:
        v_view = v.view(1, 1)
    else:
        raise ValueError("v must be scalar, (M,1), (1,N), or (M,N)")

    assert v_view.is_contiguous()

    # expand ONLY for strides (keeps storage shared/zero-stride where possible)
    v_b = v_view.expand(M, N)

    assert A.device == B.device == C.device == v_b.device
    if out is None:
        out = torch.empty((M, N), device=A.device, dtype=A.dtype)

    def get_strides(x: torch.Tensor, transp: bool):
        s = [x.stride(0), x.stride(1)]
        return list(reversed(s)) if transp else s

    a_strides = get_strides(A, a_transp)
    b_strides = get_strides(B, b_transp)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    gemm_kernel[grid](
        A,
        B,
        C,
        v_b,
        out,
        int(a_transp),
        int(b_transp),
        M,
        N,
        K,
        a_strides[0],
        a_strides[1],
        b_strides[0],
        b_strides[1],
        C.stride(0),
        C.stride(1),
        v_b.stride(0),
        v_b.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out
