import pytest
import torch

from hypll.kernels.utils import (
    Tensor1D,
    Tensor2D,
    dim_shift_input,
    dim_shift_output,
    validate_tensors,
)
from tests.kernels.utils import requires_cuda


@pytest.mark.parametrize("shape", [[8, 16], [8, 16, 32], [8, 16, 32, 64]])
@pytest.mark.parametrize("dim", [0, 1, 2, 3, -1, -2, -3])
def test_dim_shift(shape, dim):
    torch.manual_seed(0)
    if dim >= len(shape):
        return
    x = torch.rand(shape)
    shx = dim_shift_input(x, dim)
    assert shx.ndim == 2
    shy = dim_shift_output(shx, dim, shape)
    assert (x == shy).all()


@requires_cuda
def test_tensor_spec():
    @validate_tensors
    def f1(x: Tensor1D):
        return 0

    @validate_tensors
    def f2(x: Tensor2D):
        return 0

    x1d = torch.zeros(10, device="cuda")
    x2d = torch.zeros((10, 10), device="cuda")

    # should not error
    f1(x1d)
    f2(x2d)

    with pytest.raises(AssertionError):
        f1(x2d)

    with pytest.raises(AssertionError):
        f2(x1d)

    x1d = torch.zeros(10, device="cpu")
    with pytest.raises(AssertionError):
        f1(x1d)
