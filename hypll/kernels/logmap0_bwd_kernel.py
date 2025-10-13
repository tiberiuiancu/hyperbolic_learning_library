import torch
import triton
import triton.language as tl
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
def _logmap0_bwd_kernel(
    dout_ptr,
    y_ptr,
    dout_y_sum_ptr,
    yn_ptr,
    cs,
    dy_ptr,
    y_stride_b,
    dout_stride_b,
    dy_stride_b,
    B,
    M,
    ACTIVATION: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    mask_m = offs_m < M

    y_ptr += y_stride_b * pid_b
    dout_ptr += dout_stride_b * pid_b
    dy_ptr += dy_stride_b * pid_b

    y = tl.load(y_ptr + offs_m, mask=mask_m, other=0.0)
    dout = tl.load(dout_ptr + offs_m, mask=mask_m, other=0.0)
    dout_y_sum = tl.load(dout_y_sum_ptr + pid_b)
    yn = tl.load(yn_ptr + pid_b)
    yncs = tl.where(yn < 1e-15, 1e-15, yn) * cs

    ayncs = atanh(yncs)
    yncs_sq = yncs * yncs
    term1 = dout * ayncs / yncs
    term2 = tl.where(
        yn < 1e-15, 0, cs / yn * (yncs / (1 - yncs_sq) - ayncs) / yncs_sq * dout_y_sum * y
    )
    dy = term1 + term2

    if ACTIVATION == "relu":
        # sign(relu(y)) = sign(relu(out)), so we don't need out for bwd pass
        dy = tl.where(y > 0.0, dy, 0.0)

    tl.store(dy_ptr + offs_m, dy, mask=mask_m)


def logmap0_bwd_triton(
    dout: torch.Tensor, y: torch.Tensor, yn: torch.Tensor, cs: float, activation: str = "none"
) -> torch.Tensor:
    assert y.ndim == 2
    B, M = y.shape

    dout_y_sum = (dout * y).sum(dim=-1)
    assert dout_y_sum.shape[0] == y.shape[0]
    dy = torch.zeros_like(y)

    grid = lambda meta: (B, triton.cdiv(M, meta["BLOCK_M"]))
    _logmap0_bwd_kernel[grid](
        dout, y, dout_y_sum, yn, cs, dy, y.stride(0), dout.stride(0), dy.stride(0), B, M, activation
    )

    return dy
