import torch
from hypll.kernels.expmap.expmap0_fwd_kernel import expmap0_fwd_triton
from hypll.kernels.expmap.expmap0_bwd_kernel import expmap0_bwd_triton


class FastExpmap0(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        v: torch.Tensor,
        c: torch.Tensor,
        dim: int = -1,
    ):
        if dim == v.ndim - 1:
            dim = -1

        # TODO: handle v.ndim > 2
        if dim != -1:
            v = v.movedim(source=dim, destination=-1)
        out, (vn, cs, maxnorm) = expmap0_fwd_triton(v, c.item(), return_cache=True)
        ctx.save_for_backward(v, vn)
        ctx.cs = cs
        ctx.maxnorm = maxnorm
        ctx.dim = dim
        return out.movedim(source=-1, destination=dim)

    @staticmethod
    def backward(ctx, dout):
        v, vn = ctx.saved_tensors
        dv = expmap0_bwd_triton(dout, v, vn, ctx.cs, ctx.maxnorm)
        if ctx.dim != -1:
            dv = dv.movedim(source=-1, destination=ctx.dim)
        return dv, None, None
