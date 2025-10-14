import torch
from hypll.kernels.expmap.expmap0_bwd_kernel import expmap0_bwd_triton
from hypll.kernels.logmap.logmap0_bwd_kernel import logmap0_bwd_triton
from hypll.kernels.tangent_space_ops.tangent_space_op_fwd_kernel import tangent_space_op_fwd_triton


class FastTangentSpaceOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        y: torch.Tensor,
        c: torch.Tensor,
        dim: int = -1,
        op: str = "relu",
    ):
        if dim == y.ndim - 1:
            dim = -1

        # TODO: handle y.ndim > 2
        if dim != -1:
            y = y.movedim(source=dim, destination=-1)
        out, (v, yn, vn, cs, maxnorm) = tangent_space_op_fwd_triton(
            y, c.item(), op, return_cache=True
        )
        ctx.save_for_backward(y.detach(), v, yn, vn)
        ctx.cs = cs
        ctx.maxnorm = maxnorm
        ctx.dim = dim
        ctx.op = op
        return out.movedim(source=-1, destination=dim)

    @staticmethod
    def backward(ctx, dout):
        y, v, yn, vn = ctx.saved_tensors
        dv = expmap0_bwd_triton(dout, v, vn, ctx.cs, ctx.maxnorm)
        dy = logmap0_bwd_triton(dv, y, yn, ctx.cs, ctx.op)
        if ctx.dim != -1:
            dy = dy.movedim(source=-1, destination=ctx.dim)
        return dy, None, None, None
