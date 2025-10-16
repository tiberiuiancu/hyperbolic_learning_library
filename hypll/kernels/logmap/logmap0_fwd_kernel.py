import torch
import triton.language as tl
import triton
from hypll.kernels.utils import Tensor2D, atanh, validate_tensors


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
def _logmap0_fwd_kernel(
    y_ptr,
    yn_ptr,
    cs,
    out_ptr,
    y_stride_b,
    out_stride_b,
    B,
    M,
    ACTIVATION: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """each program computes a block_m slice in one row"""
    pid_b = tl.program_id(0)
    y_ptr += y_stride_b * pid_b
    out_ptr += out_stride_b * pid_b

    # first pass: calculate yn
    yn = 0.0
    for m in tl.range(0, M, BLOCK_M):
        offs_m = m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        y = tl.load(y_ptr + offs_m, mask=mask_m, other=0.0)
        yn += tl.sum(y * y)

    yn = tl.sqrt(yn)
    yncs = tl.where(yn < 1e-15, 1e-15, yn) * cs

    # second pass: calculate out
    for m in tl.range(0, M, BLOCK_M):
        offs_m = m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        y = tl.load(y_ptr + offs_m, mask=mask_m, other=0.0)
        out = atanh(yncs) * y / yncs
        if ACTIVATION == "relu":
            out = tl.where(out < 0.0, 0.0, out)
        tl.store(out_ptr + offs_m, out, mask=mask_m)

    # save for bwd pass
    tl.store(yn_ptr + pid_b, yn)


@validate_tensors
def logmap0_fwd_triton(y: Tensor2D, c: float, activation: str = "none", return_cache: bool = False):
    B, M = y.shape

    yn = torch.empty((B,), device="cuda", dtype=y.dtype)
    out = torch.empty_like(y)

    cs = c**0.5

    grid = (B,)
    _logmap0_fwd_kernel[grid](y, yn, cs, out, y.stride(0), out.stride(0), B, M, activation)
    cache = (yn, cs)
    return (out, cache) if return_cache else out


def logmap0_ref(y: torch.Tensor, c: torch.Tensor, dim: int = -1, activation: str = "none"):
    y_norm_c_sqrt = y.norm(dim=dim, keepdim=True).clamp_min(1e-15) * c.sqrt()
    out = torch.atanh(y_norm_c_sqrt) * y / y_norm_c_sqrt
    if activation == "relu":
        out = torch.nn.functional.relu(out)
    return out
