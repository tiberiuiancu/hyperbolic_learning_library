from typing import Iterable
import torch
import triton
import triton.language as tl


@triton.jit
def atanh(x):
    return 0.5 * tl.log((1 + x) / (1 - x))


@triton.jit
def tanh(x):
    e2m = tl.exp(-2 * tl.abs(x))
    t = 1 - 2 / (1 + 1 / e2m)
    return tl.where(x > 0, t, -t)


@triton.jit
def sech_squared(x):
    t = tanh(x)
    return 1 - t * t


def dim_shift_input(x: torch.Tensor, dim: int) -> torch.Tensor:
    ndim = x.ndim
    if ndim < 2:
        raise ValueError("Input must have at least 2 dimensions")

    dim = ndim + dim if dim < 0 else dim

    # man dim is assumed to be last
    if dim != ndim - 1:
        x = x.movedim(dim, -1)

    # kernels need 2D input
    if ndim > 2:
        x = x.flatten(0, -2)

    return x.contiguous()


def dim_shift_output(y: torch.Tensor, dim: int, shape: Iterable) -> torch.Tensor:
    ndim = len(shape)
    if ndim < 2:
        raise ValueError("Input must have at least 2 dimensions")

    dim = ndim + dim if dim < 0 else dim

    first_dims = list(shape)
    first_dims.pop(dim)

    if ndim > 2:
        y = y.reshape(*first_dims, -1)
    if dim != ndim - 1:
        y = y.movedim(-1, dim)

    return y
