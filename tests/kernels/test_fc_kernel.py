import math
import torch, pytest

from hypll.kernels.fc_fwd_kernel import (
    poincare_fc_fwd_ref,
    poincare_fc_fwd_triton,
)
from hypll.manifolds.poincare_ball.math.linalg import poincare_fully_connected
from hypll.kernels.fc_layer import PoincareFCLayer

import pytest

RTOL = 1
ATOL = 1e-3
NONDET_TOL = 1e16
EPS = 1e-3

B, K, M = 16, 128, 64


def _rand(*shape, grad: bool = False):
    return (torch.randn(*shape, dtype=torch.float32, requires_grad=grad).cuda() * 0.1).clamp(-1, 1)


def assert_allclose(x, y, message: str = ""):
    assert type(x) == type(y), message
    if isinstance(x, float):
        assert isinstance(y, float) and abs(x - y) < ATOL, message
    elif not isinstance(x, torch.Tensor):
        assert x == y, message
    else:
        assert torch.allclose(
            x, y, rtol=RTOL, atol=ATOL
        ), f"{message} | max diff: {(x - y).abs().max().item()}"


@pytest.mark.parametrize("bias_flag", [True, False])
def test_fwd_ref(bias_flag):
    c = torch.tensor(0.1, dtype=torch.float32).cuda()
    x = _rand(B, K)
    z = _rand(K, M)
    r = _rand(M) if bias_flag else None

    y = poincare_fully_connected(x, z, r, c)
    y_ref = poincare_fc_fwd_ref(x, z, r, c)

    assert assert_allclose(y, y_ref)


@pytest.mark.parametrize("bias_flag", [True, False])
def test_fwd(bias_flag: bool):
    c = torch.tensor(0.1, dtype=torch.float32).cuda()
    x = _rand(B, K)
    z = _rand(K, M)
    r = _rand(M) if bias_flag else None

    y, cache = poincare_fc_fwd_ref(x, z, r, c, return_cache=True)
    y_triton, cache_triton = poincare_fc_fwd_triton(x, z, r, c, return_cache=True)
    cache_names = ["x", "z", "xz", "zn", "b", "lam", "den", "c", "cs", "has_bias"]

    for n, c, ct in zip(cache_names, cache, cache_triton):
        assert_allclose(c, ct, n)

    assert_allclose(y, y_triton)


@pytest.mark.parametrize("bias_flag", [True, False])
def test_bwd(bias_flag: bool):
    torch.manual_seed(0)

    c = torch.tensor(0.1, dtype=torch.float32, requires_grad=False).cuda()
    x = _rand(B, K, grad=True)
    z = _rand(K, M, grad=True)
    r = _rand(M, grad=True) if bias_flag else None
    inputs = (x, z, r, c)

    assert torch.autograd.gradcheck(
        PoincareFCLayer.apply,
        inputs,
        eps=EPS,
        atol=ATOL,
        rtol=RTOL,
        nondet_tol=NONDET_TOL,
    ), "Gradcheck failed for Poincare FC layer"
