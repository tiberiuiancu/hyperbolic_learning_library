import torch
from hypll.kernels.fc.fc_fwd_kernel import poincare_fc_project_fwd_triton
from hypll.kernels.fc.fc_bwd_kernel import poincare_fc_bwd_triton
from hypll.kernels.utils import dim_shift_input, dim_shift_output


class FastPoincareFC(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        z: torch.Tensor,
        r: torch.Tensor = None,
        c: float = 1.0,
        dim: int = -1,
    ):
        ctx.shape = x.shape
        x = dim_shift_input(x, dim)

        py, (y, x, z, xz, zn, b, lam, num, den, yn, max_norm, c, cs) = (
            poincare_fc_project_fwd_triton(x, z, r, c, return_cache=True)
        )
        py = dim_shift_output(py, dim, ctx.shape)

        ctx.save_for_backward(y, x, z, xz, zn, b, lam, num, den, yn)
        ctx.c = c
        ctx.cs = cs
        ctx.max_norm = max_norm
        ctx.dim = dim
        ctx.has_bias = r is not None
        return py

    @staticmethod
    def backward(ctx, dout):
        y, x, z, xz, zn, b, lam, num, den, yn = ctx.saved_tensors
        dx, dz, dr = poincare_fc_bwd_triton(
            dout, y, x, z, xz, zn, b, lam, num, den, yn, ctx.max_norm, ctx.c, ctx.cs
        )

        dx = dim_shift_output(dx, ctx.dim, ctx.shape)
        dr = dr if ctx.has_bias else None

        return dx, dz, dr, None, None
