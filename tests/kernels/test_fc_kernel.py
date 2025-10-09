from unittest.mock import patch
import torch, pytest

from hypll.kernels.fc_fwd_kernel import (
    poincare_fc_fwd_project_ref,
    poincare_fc_project_fwd_triton,
)
from hypll.manifolds.poincare_ball.curvature import Curvature
from hypll.manifolds.poincare_ball.manifold import PoincareBall
from hypll.manifolds.poincare_ball.math.diffgeom import project
from hypll.manifolds.poincare_ball.math.linalg import poincare_fully_connected
from hypll.kernels.fc_layer import FastPoincareFC

import hypll.nn as hnn

import pytest

from hypll.tensors.tangent_tensor import TangentTensor

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
        assert abs(x - y) < ATOL, message
    elif not isinstance(x, torch.Tensor):
        assert x == y, message
    else:
        assert x.shape == y.shape, f"{message} shape mismatch: {x.shape}, {y.shape}"
        assert torch.allclose(
            x, y, atol=ATOL
        ), f"{message} | max diff: {(x - y).abs().max().item()}"


@pytest.mark.parametrize("bias_flag", [True, False])
def test_fwd_ref(bias_flag):
    c = torch.tensor(0.1, dtype=torch.float32).cuda()

    x = _rand(B, K)
    z = _rand(K, M)
    r = _rand(M) if bias_flag else None

    y = poincare_fully_connected(x, z, r, c)
    y = project(y, c)

    y_ref = poincare_fc_fwd_project_ref(x, z, r, c)

    assert_allclose(y, y_ref)


@pytest.mark.parametrize("bias_flag", [True, False])
def test_fwd(bias_flag: bool):
    c = torch.tensor(0.1, dtype=torch.float32).cuda()
    x = _rand(B, K)
    z = _rand(K, M)
    r = _rand(M) if bias_flag else None

    out, cache = poincare_fc_fwd_project_ref(x, z, r, c, return_cache=True)
    out_trit, cache_triton = poincare_fc_project_fwd_triton(x, z, r, c, return_cache=True)
    cache_names = ["y", "x", "z", "xz", "zn", "b", "lam", "num", "den", "yn", "max_norm", "c", "cs"]
    scalars = ["max_norm", "c", "cs"]

    for n, c, ct in zip(cache_names, cache, cache_triton):
        if n in scalars:
            assert type(c) == type(ct) == float, f"{n}: {c} | {ct}"
        assert_allclose(c, ct, n)

    assert_allclose(out, out_trit)


@pytest.mark.parametrize("bias_flag", [True, False])
def test_bwd(bias_flag: bool):
    torch.manual_seed(0)

    c = torch.tensor(0.1, dtype=torch.float32, requires_grad=False).cuda()
    x = _rand(B, K, grad=True)
    z = _rand(K, M, grad=True)
    r = _rand(M, grad=True) if bias_flag else None
    dim = -1
    inputs = (x, z, r, c, dim)

    assert torch.autograd.gradcheck(
        FastPoincareFC.apply,
        inputs,
        eps=EPS,
        atol=ATOL,
        rtol=RTOL,
        nondet_tol=NONDET_TOL,
    ), "Gradcheck failed for Poincare FC layer"


def test_conv():
    torch.manual_seed(0)
    manifold = PoincareBall(Curvature(0.1), use_triton_backend=False)
    conv = hnn.HConvolution2d(3, 8, 3, manifold, padding=1).cuda()
    conv.train(False)

    torch.manual_seed(0)
    manifold2 = PoincareBall(Curvature(0.1), use_triton_backend=True)
    conv2 = hnn.HConvolution2d(3, 8, 3, manifold2, padding=1).cuda()
    conv2.train(False)
    assert_allclose(conv.weights.tensor, conv2.weights.tensor, "weights")

    x = torch.rand((64, 3, 32, 32), dtype=torch.float32, device="cuda")
    tangents = TangentTensor(data=x, man_dim=1, manifold=manifold)
    manifold_inputs = manifold.expmap(tangents)
    out = conv(manifold_inputs)

    tangents2 = TangentTensor(data=x, man_dim=1, manifold=manifold2)
    manifold_inputs2 = manifold2.expmap(tangents2)
    assert_allclose(tangents.tensor, tangents2.tensor, "tangents")

    out2 = conv2(manifold_inputs2)
    assert_allclose(out.tensor, out2.tensor, "conv output")
