import torch
from hypll.kernels.logmap.logmap0_fwd_kernel import logmap0_fwd_triton
from hypll.kernels.logmap.logmap0_bwd_kernel import logmap0_bwd_triton
from hypll.kernels.utils import dim_shift_input, dim_shift_output


class FastLogmap0(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        y: torch.Tensor,
        c: torch.Tensor,
        dim: int = -1,
        activation: str = "none",
    ):
        ctx.shape = y.shape
        y = dim_shift_input(y, dim)

        out, (yn, cs) = logmap0_fwd_triton(y, c.item(), activation, return_cache=True)
        out = dim_shift_output(out, dim, ctx.shape)

        ctx.save_for_backward(y, yn)
        ctx.cs = cs
        ctx.dim = dim
        ctx.activation = activation
        return out

    @staticmethod
    def backward(ctx, dout):
        y, yn = ctx.saved_tensors
        dy = logmap0_bwd_triton(dout, y, yn, ctx.cs, ctx.activation)
        dy = dim_shift_output(dy, ctx.dim, ctx.shape)
        return dy, None, None, None
