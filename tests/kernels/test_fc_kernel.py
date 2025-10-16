import torch, pytest

from hypll.kernels.fc.fc_fwd_kernel import (
    poincare_fc_fwd_project_ref,
    poincare_fc_project_fwd_triton,
)
from hypll.manifolds.poincare_ball.curvature import Curvature
from hypll.manifolds.poincare_ball.manifold import PoincareBall
from hypll.manifolds.poincare_ball.math.diffgeom import project
from hypll.manifolds.poincare_ball.math.linalg import poincare_fully_connected
from hypll.kernels import FastPoincareFC

import hypll.nn as hnn

import pytest

from hypll.tensors.tangent_tensor import TangentTensor
from tests.kernels.utils import assert_allclose, safe_rand, requires_cuda

RTOL = 1
ATOL = 1e-3
NONDET_TOL = 1e16
EPS = 1e-3

B, K, M = 16, 128, 64


@requires_cuda
@pytest.mark.parametrize("bias_flag", [True, False])
def test_fwd_ref(bias_flag):
    c = torch.tensor(0.1, dtype=torch.float32).cuda()

    x = safe_rand(B, K)
    z = safe_rand(K, M)
    r = safe_rand(M) if bias_flag else None

    y = poincare_fully_connected(x, z, r, c)
    y = project(y, c)

    y_ref = poincare_fc_fwd_project_ref(x, z, r, c)

    assert_allclose(y, y_ref)


@requires_cuda
@pytest.mark.parametrize("bias_flag", [True, False])
def test_fwd(bias_flag: bool):
    c = torch.tensor(0.1, dtype=torch.float32).cuda()
    x = safe_rand(B, K)
    z = safe_rand(K, M)
    r = safe_rand(M) if bias_flag else None

    out, cache = poincare_fc_fwd_project_ref(x, z, r, c, return_cache=True)
    out_trit, cache_triton = poincare_fc_project_fwd_triton(x, z, r, c, return_cache=True)
    cache_names = ["y", "x", "z", "xz", "zn", "b", "lam", "num", "den", "yn", "max_norm", "c", "cs"]
    scalars = ["max_norm", "c", "cs"]

    for n, c, ct in zip(cache_names, cache, cache_triton):
        if n in scalars:
            assert type(c) == type(ct) == float, f"{n}: {c} | {ct}"
        assert_allclose(c, ct, n)

    assert_allclose(out, out_trit)


@requires_cuda
@pytest.mark.parametrize("bias_flag", [True, False])
def test_bwd(bias_flag: bool):
    torch.manual_seed(0)

    c = torch.tensor(0.1, dtype=torch.float32, requires_grad=False).cuda()
    x = safe_rand(B, K, grad=True)
    z = safe_rand(K, M, grad=True)
    r = safe_rand(M, grad=True) if bias_flag else None
    dim = -1
    inputs = (x, z, r, c, dim)

    assert torch.autograd.gradcheck(
        FastPoincareFC.apply,
        inputs,
        eps=EPS,
        atol=ATOL,
        rtol=RTOL,
        nondet_tol=NONDET_TOL,
    ), "Gradcheck failed"


@requires_cuda
@pytest.mark.parametrize("kernel_size", [1, 3])
@pytest.mark.parametrize("in_channels", [1, 3])
@pytest.mark.parametrize("out_channels", [4, 8])
@pytest.mark.parametrize("padding", [0, 1])
def test_conv(kernel_size, in_channels, out_channels, padding):
    torch.manual_seed(0)
    manifold = PoincareBall(Curvature(0.1), use_triton_backend=False)
    conv = hnn.HConvolution2d(
        in_channels, out_channels, kernel_size, manifold, padding=padding
    ).cuda()
    conv.train(False)

    torch.manual_seed(0)
    manifold2 = PoincareBall(Curvature(0.1), use_triton_backend=True)
    conv2 = hnn.HConvolution2d(
        in_channels, out_channels, kernel_size, manifold2, padding=padding
    ).cuda()
    conv2.train(False)
    assert_allclose(conv.weights.tensor, conv2.weights.tensor, "weights")

    batch_size = 8
    height = 16
    width = 16
    x = torch.rand((batch_size, in_channels, height, width), dtype=torch.float32, device="cuda")
    tangents = TangentTensor(data=x, man_dim=1, manifold=manifold)
    manifold_inputs = manifold.expmap(tangents)
    out = conv(manifold_inputs)

    tangents2 = TangentTensor(data=x, man_dim=1, manifold=manifold2)
    manifold_inputs2 = manifold2.expmap(tangents2)
    assert_allclose(tangents.tensor, tangents2.tensor, "tangents")

    out2 = conv2(manifold_inputs2)
    assert_allclose(out.tensor, out2.tensor, "conv output")
