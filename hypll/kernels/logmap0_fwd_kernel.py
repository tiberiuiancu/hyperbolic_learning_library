import torch
import triton.language as tl
import triton
from hypll.kernels.utils import atanh


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
    pid_m = tl.program_id(1)

    offs_m = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    mask_m = offs_m < M

    y_ptr += y_stride_b * pid_b
    out_ptr += out_stride_b * pid_b

    y = tl.load(y_ptr + offs_m, mask=mask_m, other=0.0)
    yn = tl.load(yn_ptr + pid_b)
    yncs = tl.where(yn < 1e-15, 1e-15, yn) * cs

    out = atanh(yncs) * y / yncs
    if ACTIVATION == "relu":
        out = tl.where(out > 0.0, out, 0.0)
    tl.store(out_ptr + offs_m, out, mask=mask_m)


def logmap0_fwd_triton(
    y: torch.Tensor, c: float, activation: str = "none", return_cache: bool = False
):
    assert y.ndim == 2  # assume y [B, M]
    B, M = y.shape

    yn = y.norm(dim=-1, keepdim=True)
    out = torch.empty_like(y)

    cs = c**0.5

    grid = lambda meta: (B, triton.cdiv(M, meta["BLOCK_M"]))
    _logmap0_fwd_kernel[grid](y, yn, cs, out, y.stride(0), out.stride(0), B, M, activation)
    cache = (yn, cs)
    return (out, cache) if return_cache else out


def logmap0_ref(y: torch.Tensor, c: torch.Tensor, dim: int = -1):
    y_norm_c_sqrt = y.norm(dim=dim, keepdim=True).clamp_min(1e-15) * c.sqrt()
    return torch.atanh(y_norm_c_sqrt) * y / y_norm_c_sqrt
