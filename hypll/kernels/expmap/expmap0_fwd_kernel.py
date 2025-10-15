import torch
import triton.language as tl
import triton
from hypll.kernels.utils import Tensor2D, tanh, validate_tensors


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
def _expmap0_fwd_kernel(
    v_ptr, vn_ptr, cs, maxnorm, out_ptr, v_stride_b, out_stride_b, B, M, BLOCK_M: tl.constexpr
):
    """each program computes one row of the output"""
    pid_b = tl.program_id(0)

    v_ptr += v_stride_b * pid_b
    out_ptr += out_stride_b * pid_b

    # first pass: calculate vn
    vn = 0.0
    for m in tl.range(0, M, BLOCK_M):
        offs_m = m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        v = tl.load(v_ptr + offs_m, mask=mask_m, other=0.0)
        vn += tl.sum(v * v)
    vn = tl.sqrt(vn)
    vncs = tl.where(vn < 1e-15, 1e-15, vn) * cs
    tanh_vncs = tanh(vncs)
    if tanh_vncs > maxnorm:
        tanh_vncs = maxnorm

    # second pass: calculate out
    for m in tl.range(0, M, BLOCK_M):
        offs_m = m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        v = tl.load(v_ptr + offs_m, mask=mask_m, other=0.0)
        out = tanh_vncs * v / vncs
        tl.store(out_ptr + offs_m, out, mask=mask_m)

    # save vn for bwd pass
    tl.store(vn_ptr + pid_b, vn)


@validate_tensors
def expmap0_fwd_triton(v: Tensor2D, c: float, return_cache: bool = False):
    B, M = v.shape

    vn = torch.empty((B,), device="cuda", dtype=v.dtype)
    out = torch.empty_like(v)
    cs = c**0.5

    eps = 4e-3 if v.dtype == torch.float32 else 1e-5
    maxnorm = (1 - eps) / ((c + 1e-15) ** 0.5) if c > 0.0 else 1e15

    grid = (B,)
    _expmap0_fwd_kernel[grid](v, vn, cs, maxnorm, out, v.stride(0), out.stride(0), B, M)
    cache = (vn, cs, maxnorm)
    return (out, cache) if return_cache else out


def expmap0_ref(v: torch.Tensor, c: torch.Tensor, dim: int = -1, eps: float = -1):
    vncs = v.norm(dim=dim, keepdim=True).clamp_min(1e-15) * c.sqrt()
    tanh_vncs = torch.tanh(vncs)

    if eps < 0:
        if v.dtype == torch.float32:
            eps = 4e-3
        else:
            eps = 1e-5

    c_val = c.item()
    if c > 0:
        maxnorm = (1 - eps) / ((c_val + 1e-15) ** 0.5)
    else:
        maxnorm = 1e15

    cond = tanh_vncs > maxnorm
    return torch.where(cond, v / vncs * maxnorm, tanh_vncs * v / vncs)
