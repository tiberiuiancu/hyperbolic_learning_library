import torch
import triton.language as tl
import triton
from hypll.kernels.utils import atanh, tanh


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


@triton.autotune(configs=get_autotune_configs(), key=["B", "M"])
@triton.jit
def _tangent_space_op_fwd_kernel(
    y_ptr,
    v_ptr,
    yn_ptr,
    vn_ptr,
    cs,
    maxnorm,
    out_ptr,
    y_stride_b,
    v_stride_b,
    out_stride_b,
    B,
    M,
    OP: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """each program computes one row"""
    pid_b = tl.program_id(0)
    y_ptr += y_stride_b * pid_b
    out_ptr += out_stride_b * pid_b
    v_ptr += v_stride_b * pid_b

    # first pass: calculate yn and vn
    yn = 0.0
    vn = 0.0
    for m in tl.range(0, M, BLOCK_M):
        offs_m = m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        y = tl.load(y_ptr + offs_m, mask=mask_m, other=0.0)
        yn += tl.sum(y * y)

        if OP == "relu":
            yr = tl.where(y < 0.0, 0.0, y)
            vn += tl.sum(yr * yr)

    yn = tl.sqrt(yn)
    yncs = tl.where(yn < 1e-15, 1e-15, yn) * cs

    if OP == "relu":
        vn = tl.sqrt(vn)
    else:
        vn = yn

    ayncs_yncs = atanh(yncs) / yncs

    vn *= ayncs_yncs
    vncs = tl.where(vn < 1e-15, 1e-15, vn) * cs

    tanh_vncs = tanh(vncs)
    if tanh_vncs > maxnorm:
        tanh_vncs = maxnorm

    tvncs_vncs = tanh_vncs / vncs

    # second pass: calculate out
    for m in tl.range(0, M, BLOCK_M):
        offs_m = m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        y = tl.load(y_ptr + offs_m, mask=mask_m, other=0.0)
        if OP == "relu":
            y = tl.where(y < 0.0, 0.0, y)
        v = y * ayncs_yncs
        out = v * tvncs_vncs
        tl.store(v_ptr + offs_m, v, mask=mask_m)
        tl.store(out_ptr + offs_m, out, mask=mask_m)

    # save vn for bwd pass
    tl.store(yn_ptr + pid_b, yn)
    tl.store(vn_ptr + pid_b, vn)


def tangent_space_op_fwd_triton(
    y: torch.Tensor, c: float, op: str = "relu", return_cache: bool = False
):
    if op != "relu":
        raise NotImplementedError("Currently only ReLU is implemented")

    assert y.ndim == 2  # assume y [B, M]
    B, M = y.shape

    yn = torch.empty((B,), device="cuda", dtype=y.dtype)
    vn = torch.empty((B,), device="cuda", dtype=y.dtype)
    v = torch.empty_like(y)
    out = torch.empty_like(y)

    cs = c**0.5
    eps = 4e-3 if v.dtype == torch.float32 else 1e-5
    maxnorm = (1 - eps) / ((c + 1e-15) ** 0.5) if c > 0.0 else 1e15

    grid = (B,)
    _tangent_space_op_fwd_kernel[grid](
        y, v, yn, vn, cs, maxnorm, out, y.stride(0), v.stride(0), out.stride(0), B, M, op
    )
    cache = (v, yn, vn, cs, maxnorm)
    return (out, cache) if return_cache else out
