import torch
import triton
import triton.language as tl
from hypll.kernels.utils import tanh, sech_squared


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
def _expmap0_bwd_kernel(
    dout_ptr,
    v_ptr,
    dout_v_sum_ptr,
    vn_ptr,
    cs,
    maxnorm,
    dv_ptr,
    v_stride_b,
    dout_stride_b,
    dv_stride_b,
    B,
    M,
    BLOCK_M: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    mask_m = offs_m < M

    v_ptr += v_stride_b * pid_b
    dout_ptr += dout_stride_b * pid_b
    dv_ptr += dv_stride_b * pid_b

    v = tl.load(v_ptr + offs_m, mask=mask_m, other=0.0)
    dout = tl.load(dout_ptr + offs_m, mask=mask_m, other=0.0)
    dout_v_sum = tl.load(dout_v_sum_ptr + pid_b)
    vn = tl.load(vn_ptr + pid_b)
    vncs = tl.where(vn < 1e-15, 1e-15, vn) * cs

    tanh_vncs = tanh(vncs)
    term1 = tanh_vncs / vncs * dout
    dtanh = tl.where(tanh_vncs > maxnorm, 0, vncs * sech_squared(vncs))
    term2 = tl.where(vn < 1e-15, 0, cs / vn * (dtanh - tanh_vncs) / (vncs * vncs) * dout_v_sum * v)
    dv = term1 + term2

    tl.store(dv_ptr + offs_m, dv, mask=mask_m)


def expmap0_bwd_triton(
    dout: torch.Tensor, v: torch.Tensor, vn: torch.Tensor, cs: float, maxnorm: float
) -> torch.Tensor:
    assert v.ndim == 2
    B, M = v.shape

    dout_v_sum = (dout * v).sum(dim=-1)
    assert dout_v_sum.shape[0] == v.shape[0]
    dv = torch.zeros_like(v)

    grid = lambda meta: (B, triton.cdiv(M, meta["BLOCK_M"]))
    _expmap0_bwd_kernel[grid](
        dout, v, dout_v_sum, vn, cs, maxnorm, dv, v.stride(0), dout.stride(0), dv.stride(0), B, M
    )

    return dv
