import torch
from hypll.kernels.expmap.expmap0_fwd_kernel import expmap0_fwd_triton
from hypll.kernels.expmap.expmap0_bwd_kernel import expmap0_bwd_triton
from hypll.kernels.utils import dim_shift_input, dim_shift_output


class FastExpmap0(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        v: torch.Tensor,
        c: torch.Tensor,
        dim: int = -1,
    ):
        ctx.shape = v.shape
        v = dim_shift_input(v, dim)

        out, (vn, cs, maxnorm) = expmap0_fwd_triton(v, c.item(), return_cache=True)
        ctx.save_for_backward(v, vn)
        ctx.cs = cs
        ctx.maxnorm = maxnorm
        ctx.dim = dim
        return dim_shift_output(out, dim, ctx.shape)

    @staticmethod
    def backward(ctx, dout):
        v, vn = ctx.saved_tensors
        dv = expmap0_bwd_triton(dout, v, vn, ctx.cs, ctx.maxnorm)
        dv = dim_shift_output(dv, ctx.dim, ctx.shape)
        return dv, None, None
