import torch, pytest

from hypll.kernels.fc_kernel import (
    poincare_fc_fwd_ref,
    poincare_fully_connected_triton,
    PoincareFCLayer,
)
from hypll.manifolds.poincare_ball.math.linalg import poincare_fully_connected
import pytest

RTOL = 1e-3
ATOL = 1e-3
NONDET_TOL = 1e16
EPS = 1e-3


def _rand(*shape, grad: bool = False):
    return (torch.randn(*shape, dtype=torch.float32, requires_grad=grad).cuda() * 0.1).clamp(-1, 1)


@pytest.mark.parametrize("bias_flag", [True, False])
def test_fwd_reference(bias_flag: bool):
    """Run a simple correctness test."""
    B, K, M = 256, 128, 64
    c = torch.tensor(0.1, dtype=torch.float32).cuda()
    x = _rand(B, K)
    z = _rand(K, M)
    r = _rand(M) if bias_flag else None

    y = poincare_fully_connected(x, z, r, c)
    y_ref, cache_ref = poincare_fc_fwd_ref(x, z, r, c)
    y_triton, cache = poincare_fully_connected_triton(x, z, r, c)

    cache_vars = ["num", "v", "inner", "lam", "den", "two_cs_r"]
    for ca_r, ca, cv in zip(cache_ref, cache, cache_vars):
        if ca_r is None:
            continue
        assert torch.allclose(
            ca, ca_r, atol=ATOL, rtol=RTOL
        ), f"{cv} does not match; max diff: {(ca_r - ca).abs().max().item()}"

    assert torch.allclose(y, y_ref, atol=ATOL, rtol=RTOL), (y - y_ref).abs().max().item()
    assert torch.allclose(y, y_triton, atol=ATOL, rtol=RTOL), (y - y_triton).abs().max().item()


@pytest.mark.parametrize("bias_flag", [False, True])
def test_poincare_fc_autograd_backward(bias_flag: bool):
    torch.manual_seed(0)
    B, K, M = 8, 16, 32
    c = torch.tensor(0.1, dtype=torch.float32, requires_grad=False).cuda()

    x = _rand(B, K, grad=True)
    z = _rand(K, M, grad=True)
    r = _rand(M, grad=True) if bias_flag else None

    inputs = (x, z, r, c) if bias_flag else (x, z, None, c)
    inputs = tuple(inp for inp in inputs if inp is not None)

    assert torch.autograd.gradcheck(
        PoincareFCLayer.apply,
        inputs,
        eps=EPS,
        atol=ATOL,
        rtol=RTOL,
        nondet_tol=NONDET_TOL,
    ), "Gradcheck failed for Poincare FC layer"
