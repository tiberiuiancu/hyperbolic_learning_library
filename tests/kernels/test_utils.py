import pytest
import torch

from hypll.kernels.utils import dim_shift_input, dim_shift_output


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
