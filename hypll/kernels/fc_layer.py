import torch
from hypll.kernels.fc_fwd_kernel import poincare_fc_project_fwd_triton
from hypll.kernels.fc_bwd_kernel import poincare_fc_bwd_triton


class FastPoincareFC(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, z: torch.Tensor, r: torch.Tensor = None, c: float = 1.0, dim: int = -1
    ):
        x = x.movedim(source=dim, destination=-1)
        y, (x, z, xz, zn, b, lam, den, yn, mn, c, cs) = poincare_fc_project_fwd_triton(
            x, z, r, c, return_cache=True
        )
        ctx.save_for_backward(y, x, z, xz, zn, b, lam, den, yn, mn)
        ctx.c = c
        ctx.cs = cs
        ctx.dim = dim
        ctx.has_bias = r is not None
        return y.movedim(source=-1, destination=dim)

    @staticmethod
    def backward(ctx, dout):
        y, x, z, xz, zn, b, lam, den, yn, mn = ctx.saved_tensors
        dx, dz, dr = poincare_fc_bwd_triton(
            dout, y, x, z, xz, zn, b, lam, den, yn, mn, ctx.c, ctx.cs
        )

        dx = dx.movedim(source=-1, destination=ctx.dim)
        dr = dr if ctx.has_bias else None

        return dx, dz, dr, None, None
