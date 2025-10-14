import pytest
import torch

from hypll.kernels.tangent_space_op_layer import FastTangentSpaceOp
from hypll.manifolds.poincare_ball.curvature import Curvature
from hypll.manifolds.poincare_ball.manifold import PoincareBall
from hypll.tensors.manifold_tensor import ManifoldTensor
from hypll.utils.layer_utils import op_in_tangent_space
from tests.kernels.utils import assert_allclose, safe_rand, requires_cuda


B, M = 4, 128
EPS = 1e-3
ATOL = 1e-3
RTOL = 1
NONDET_TOL = 1e-3


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


@requires_cuda
@pytest.mark.parametrize("op", ["relu"])
def test_bwd(op):
    torch.manual_seed(3)
    c = torch.tensor(0.1, dtype=torch.float32, requires_grad=False).cuda()
    y = safe_rand(B, M, grad=True)
    dim = -1
    inputs = (y, c, dim, op)

    assert torch.autograd.gradcheck(
        FastTangentSpaceOp.apply,
        inputs,
        eps=EPS,
        atol=ATOL,
        rtol=RTOL,
        nondet_tol=NONDET_TOL,
    ), "Gradcheck failed"
