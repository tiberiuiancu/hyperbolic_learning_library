import torch
from hypll.kernels.fc_fwd_kernel import poincare_fc_project_fwd_triton
from hypll.kernels.fc_bwd_kernel import poincare_fc_bwd_triton


class FastPoincareFC(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, z: torch.Tensor, r: torch.Tensor = None, c: float = 1.0, dim: int = -1
    ):
        x = x.movedim(source=dim, destination=-1)
        py, (y, x, z, xz, zn, b, lam, num, den, yn, max_norm, c, cs) = (
            poincare_fc_project_fwd_triton(x, z, r, c, return_cache=True)
        )
        ctx.save_for_backward(y, x, z, xz, zn, b, lam, num, den, yn)
        ctx.c = c
        ctx.cs = cs
        ctx.max_norm = max_norm
        ctx.dim = dim
        ctx.has_bias = r is not None
        return py.movedim(source=-1, destination=dim)

    @staticmethod
    def backward(ctx, dout):
        y, x, z, xz, zn, b, lam, num, den, yn = ctx.saved_tensors
        dx, dz, dr = poincare_fc_bwd_triton(
            dout, y, x, z, xz, zn, b, lam, num, den, yn, ctx.max_norm, ctx.c, ctx.cs
        )

        dx = dx.movedim(source=-1, destination=ctx.dim)
        dr = dr if ctx.has_bias else None

        return dx, dz, dr, None, None
