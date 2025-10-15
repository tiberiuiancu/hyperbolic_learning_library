import torch
from hypll.kernels.expmap.expmap0_bwd_kernel import expmap0_bwd_triton
from hypll.kernels.logmap.logmap0_bwd_kernel import logmap0_bwd_triton
from hypll.kernels.tangent_space_ops.tangent_space_op_fwd_kernel import tangent_space_op_fwd_triton
from hypll.kernels.utils import dim_shift_input, dim_shift_output


class FastTangentSpaceOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        y: torch.Tensor,
        c: torch.Tensor,
        dim: int = -1,
        op: str = "relu",
    ):
        ctx.shape = y.shape
        y = dim_shift_input(y, dim)

        out, (v, yn, vn, cs, maxnorm) = tangent_space_op_fwd_triton(
            y, c.item(), op, return_cache=True
        )
        out = dim_shift_output(out, dim, ctx.shape)

        ctx.save_for_backward(y.detach(), v, yn, vn)
        ctx.cs = cs
        ctx.maxnorm = maxnorm
        ctx.dim = dim
        ctx.op = op

        return out

    @staticmethod
    def backward(ctx, dout):
        y, v, yn, vn = ctx.saved_tensors
        dv = expmap0_bwd_triton(dout, v, vn, ctx.cs, ctx.maxnorm)
        dy = logmap0_bwd_triton(dv, y, yn, ctx.cs, ctx.op)
        dy = dim_shift_output(dy, ctx.dim, ctx.shape)
        return dy, None, None, None
