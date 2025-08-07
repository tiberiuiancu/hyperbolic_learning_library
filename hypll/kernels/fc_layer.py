import torch
from hypll.kernels.fc_fwd_kernel import poincare_fc_fwd_triton
from hypll.kernels.fc_bwd_kernel import poincare_fc_bwd_triton


class PoincareFCLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, z, r=None, c=1.0):
        out, (x, z, xz, zn, b, lam, den, c, cs) = poincare_fc_fwd_triton(
            x, z, r, c, return_cache=True
        )
        ctx.save_for_backward(x, z, xz, zn, b, lam, den)
        ctx.c = c
        ctx.cs = cs
        ctx.has_bias = r is not None
        return out

    @staticmethod
    def backward(ctx, dout):
        x, z, xz, zn, b, lam, den = ctx.saved_tensors
        dx, dz, _ = poincare_fc_bwd_triton(dout, x, z, xz, zn, b, lam, den, ctx.c, ctx.cs)
        # TODO: implement dr
        dr = torch.zeros_like(b) if ctx.has_bias else None
        return dx, dz, dr, None
