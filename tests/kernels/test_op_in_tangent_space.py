import pytest
import torch

from hypll.manifolds.poincare_ball.curvature import Curvature
from hypll.manifolds.poincare_ball.manifold import PoincareBall
from hypll.tensors.manifold_tensor import ManifoldTensor
from hypll.utils.layer_utils import op_in_tangent_space
from tests.kernels.utils import assert_allclose, safe_rand, requires_cuda


B, M = 4, 128


@requires_cuda
@pytest.mark.parametrize("use_triton_backend", [True, False])
@pytest.mark.parametrize("op", [torch.nn.functional.relu, lambda x: x])
def test_op_in_tangent_space(use_triton_backend, op):
    torch.manual_seed(0)
    manifold = PoincareBall(Curvature(0.1), use_triton_backend=use_triton_backend)

    data = safe_rand((B, M))
    x = ManifoldTensor(data, manifold)

    ref = op_in_tangent_space(op, manifold, x)
    man = manifold.op_in_tangent_space(op, x)
    assert_allclose(ref.tensor, man.tensor, equal_nan=True)
