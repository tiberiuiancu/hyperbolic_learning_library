import torch
from hypll.kernels.expmap0_fwd_kernel import expmap0_fwd_triton, expmap0_ref
from hypll.kernels.expmap0_layer import FastExpmap0
from tests.kernels.utils import assert_allclose, safe_rand
from hypll.manifolds.poincare_ball.math.diffgeom import expmap0


B, M = 4, 128

ATOL = 1e-3
RTOL = 1
EPS = 1e-3
NONDET_TOL = 1e16


def test_fwd_ref():
    v = safe_rand((B, M))
    c = torch.tensor(0.1, dtype=torch.float32)
    assert_allclose(expmap0(v, c), expmap0_ref(v, c))


def test_fwd():
    v = safe_rand((B, M))
    c = torch.tensor(0.1, dtype=torch.float32)
    assert_allclose(expmap0_ref(v, c), expmap0_fwd_triton(v, c.item()), equal_nan=True)


def test_bwd():
    torch.manual_seed(1)

    c = torch.tensor(0.1, dtype=torch.float32, requires_grad=False).cuda()
    y = safe_rand(B, M, grad=True)
    dim = -1
    inputs = (y, c, dim)

    assert torch.autograd.gradcheck(
        FastExpmap0.apply,
        inputs,
        eps=EPS,
        atol=ATOL,
        rtol=RTOL,
        nondet_tol=NONDET_TOL,
    ), "Gradcheck failed for Poincare FC layer"
