import torch
import triton.language as tl
import triton
from hypll.kernels.utils import tanh


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
    """each program computes a block_m slice in one row"""
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    mask_m = offs_m < M

    v_ptr += v_stride_b * pid_b
    out_ptr += out_stride_b * pid_b

    v = tl.load(v_ptr + offs_m, mask=mask_m, other=0.0)
    vn = tl.load(vn_ptr + pid_b)
    vncs = tl.where(vn < 1e-15, 1e-15, vn) * cs
    tanh_vncs = tanh(vncs)
    if tanh_vncs > maxnorm:
        tanh_vncs = maxnorm

    out = tanh_vncs * v / vncs
    tl.store(out_ptr + offs_m, out, mask=mask_m)


def expmap0_fwd_triton(v: torch.Tensor, c: float, return_cache: bool = False):
    assert v.ndim == 2  # assume y [B, M]
    B, M = v.shape

    vn = v.norm(dim=-1, keepdim=True)
    out = torch.empty_like(v)
    cs = c**0.5

    eps = 4e-3 if v.dtype == torch.float32 else 1e-5
    maxnorm = (1 - eps) / ((c + 1e-15) ** 0.5) if c > 0.0 else 1e15

    grid = lambda meta: (B, triton.cdiv(M, meta["BLOCK_M"]))
    _expmap0_fwd_kernel[grid](v, vn, cs, maxnorm, out, v.stride(0), out.stride(0), B, M)
    cache = (vn, cs)
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
