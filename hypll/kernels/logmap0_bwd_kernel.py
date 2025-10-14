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
def _logmap0_bwd_fused_kernel(
    dout_ptr,  # *[B, M]
    y_ptr,  # *[B, M]
    yn_ptr,  # *[B]
    cs,  # scalar (sqrt(c)) or per-batch scalar if you pass it that way
    dy_ptr,  # *[B, M]
    y_stride_b,  # stride for leading dim of y
    dout_stride_b,  # stride for leading dim of dout
    dy_stride_b,  # stride for leading dim of dy
    B,
    M,
    ACTIVATION: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_b = tl.program_id(0)

    # Base pointers for this batch row
    y_row_ptr = y_ptr + pid_b * y_stride_b
    dout_row_ptr = dout_ptr + pid_b * dout_stride_b
    dy_row_ptr = dy_ptr + pid_b * dy_stride_b

    # Load per-batch scalars
    yn = tl.load(yn_ptr + pid_b)
    yn_c = tl.where(yn < 1e-15, 1e-15, yn)  # clamp for stability
    yncs = yn_c * cs
    ayncs = atanh(yncs)
    yncs_sq = yncs * yncs

    # First pass: accumulate masked dot = sum_j (mask_j * dout_j * y_j)
    dot = 0.0
    for start_m in range(0, M, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        y = tl.load(y_row_ptr + offs_m, mask=mask_m, other=0.0)
        dout = tl.load(dout_row_ptr + offs_m, mask=mask_m, other=0.0)

        if ACTIVATION == "relu":
            relu_mask = y > 0.0
            contrib = dout * y * relu_mask
        else:
            contrib = dout * y

        dot += tl.sum(contrib, axis=0)

    # Scalar coefficients (same for all m in this batch row)
    # term1 per-element coeff: ayncs / yncs
    coeff1 = ayncs / yncs

    # term2 scalar coeff in front of y: (cs/yn) * ((yncs/(1-yncs^2) - ayncs) / yncs^2) * dot
    tmp = (yncs / (1.0 - yncs_sq) - ayncs) / yncs_sq
    coeff2 = tl.where(yn < 1e-15, 0.0, cs / yn * tmp) * dot

    # Second pass: write dy = mask * ( coeff1 * dout + coeff2 * y )
    for start_m in range(0, M, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        y = tl.load(y_row_ptr + offs_m, mask=mask_m, other=0.0)
        dout = tl.load(dout_row_ptr + offs_m, mask=mask_m, other=0.0)

        dy = coeff1 * dout + coeff2 * y

        if ACTIVATION == "relu":
            dy = tl.where(y > 0.0, dy, 0.0)

        tl.store(dy_row_ptr + offs_m, dy, mask=mask_m)


def logmap0_bwd_triton(
    dout: torch.Tensor, y: torch.Tensor, yn: torch.Tensor, cs: float, activation: str = "none"
) -> torch.Tensor:
    assert y.ndim == 2 and dout.shape == y.shape
    B, M = y.shape
    dy = torch.empty_like(y)

    grid = (B,)
    _logmap0_bwd_fused_kernel[grid](
        dout,
        y,
        yn,
        cs,
        dy,
        y.stride(0),
        dout.stride(0),
        dy.stride(0),
        B,
        M,
        activation,
    )
    return dy
