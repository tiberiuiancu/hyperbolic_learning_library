import torch
from hypll.kernels.logmap0_fwd_kernel import logmap0_fwd_triton
from hypll.kernels.logmap0_bwd_kernel import logmap0_bwd_triton


class FastLogmap0(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        y: torch.Tensor,
        c: torch.Tensor,
        dim: int = -1,
        activation: str = "none",
    ):
        if dim == y.ndim - 1:
            dim = -1

        # TODO: handle y.ndim > 2
        if dim != -1:
            y = y.movedim(source=dim, destination=-1)
        out, (yn, cs) = logmap0_fwd_triton(y, c.item(), activation, return_cache=True)
        ctx.save_for_backward(y, yn)
        ctx.cs = cs
        ctx.dim = dim
        ctx.activation = activation
        return out.movedim(source=-1, destination=dim)

    @staticmethod
    def backward(ctx, dout):
        y, yn = ctx.saved_tensors
        dy = logmap0_bwd_triton(dout, y, yn, ctx.cs, ctx.activation)
        if ctx.dim != -1:
            dy = dy.movedim(source=-1, destination=ctx.dim)
        return dy, None, None, None
