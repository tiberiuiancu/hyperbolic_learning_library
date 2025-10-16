import pytest
import torch
from hypll.kernels.logmap.logmap0_fwd_kernel import logmap0_fwd_triton, logmap0_ref
from hypll.kernels.logmap.logmap0_layer import FastLogmap0
from tests.kernels.utils import assert_allclose, safe_rand, requires_cuda


B, M = 4, 128

ATOL = 1e-3
RTOL = 1
EPS = 1e-3
NONDET_TOL = 1e16


@requires_cuda
@pytest.mark.parametrize("activation", ["none", "relu"])
def test_fwd(activation):
    torch.manual_seed(0)
    y = safe_rand((B, M))
    c = torch.tensor(0.1, dtype=torch.float32)

    assert_allclose(
        logmap0_ref(y, c, -1, activation),
        logmap0_fwd_triton(y, c.item(), activation),
        equal_nan=True,
    )

    assert_allclose(
        logmap0_ref(y, c, -1, activation),
        FastLogmap0.apply(y, c, -1, activation),
        equal_nan=True,
    )


@requires_cuda
@pytest.mark.parametrize("activation", ["none", "relu"])
def test_bwd(activation):
    torch.manual_seed(0)
    c = torch.tensor(0.1, dtype=torch.float32, requires_grad=False).cuda()
    y = safe_rand(B, M, grad=True)
    dim = -1
    inputs = (y, c, dim, activation)

    assert torch.autograd.gradcheck(
        FastLogmap0.apply,
        inputs,
        eps=EPS,
        atol=ATOL,
        rtol=RTOL,
        nondet_tol=NONDET_TOL,
    ), "Gradcheck failed"
