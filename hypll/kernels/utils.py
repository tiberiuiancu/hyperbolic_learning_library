import inspect
from typing import Iterable
import torch
import triton
import triton.language as tl
from typing import Annotated


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
    if y is None:
        return

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


class TensorSpec:
    def __init__(self, ndim=None):
        self.ndim = ndim


Tensor1D = Annotated[torch.Tensor, TensorSpec(ndim=1)]
Tensor2D = Annotated[torch.Tensor, TensorSpec(ndim=2)]


def validate_tensors(func):
    sig = inspect.signature(func)
    specs = {
        name: p.annotation.__metadata__[0]
        for name, p in sig.parameters.items()
        if hasattr(p.annotation, "__metadata__")
    }

    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, x in bound.arguments.items():
            spec = specs.get(name)
            if isinstance(spec, TensorSpec) and isinstance(x, torch.Tensor):
                assert x.ndim == spec.ndim, f"{name} must have {spec.ndim} dims, got {x.ndim}"
                assert x.is_cuda, f"{name} must be on the GPU"
                assert x.is_contiguous(), f"{name} must be contiguous"
        return func(*args, **kwargs)

    return wrapper
