import torch
from hypll.kernels.fc_fwd_kernel import poincare_fc_fwd_triton
from hypll.kernels.fc_bwd_kernel import poincare_fc_bwd_triton


class FastPoincareFC(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, z: torch.Tensor, r: torch.Tensor = None, c: float = 1.0, dim: int = -1
    ):
        x = x.movedim(source=dim, destination=-1)
        out, (x, z, xz, zn, b, lam, den, c, cs) = poincare_fc_fwd_triton(
            x, z, r, c, return_cache=True
        )
        ctx.save_for_backward(x, z, xz, zn, b, lam, den)
        ctx.c = c
        ctx.cs = cs
        ctx.has_bias = r is not None
        return out.movedim(source=-1, destination=dim)

    @staticmethod
    def backward(ctx, dout):
        x, z, xz, zn, b, lam, den = ctx.saved_tensors
        dx, dz, dr = poincare_fc_bwd_triton(dout, x, z, xz, zn, b, lam, den, ctx.c, ctx.cs)
        return dx, dz, dr, None
