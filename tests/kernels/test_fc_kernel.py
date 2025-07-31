import torch, pytest

from hypll.kernels.fc_kernel import (
    poincare_fc_bwd_ref,
    poincare_fc_fwd_ref,
    poincare_fully_connected_triton,
)
from hypll.manifolds.poincare_ball.math.linalg import poincare_fully_connected
import pytest

RTOL = 1e-6
ATOL = 1e-6


def test_fwd_reference():
    """Run a simple correctness test."""
    B, K, M = 256, 128, 64
    c = torch.tensor(0.1, dtype=torch.float32).cuda()
    x = torch.randn(B, K, dtype=torch.float32).cuda()
    z = torch.randn(K, M, dtype=torch.float32).cuda()
    r = torch.randn(M, dtype=torch.float32).cuda()

    y = poincare_fully_connected(x, z, r, c)
    y_ref = poincare_fc_fwd_ref(x, z, r, c)
    y_triton = poincare_fully_connected_triton(x, z, r, c)

    assert torch.allclose(y, y_ref, atol=1e-3), (y - y_ref).abs().max().item()
    assert torch.allclose(y, y_triton, atol=1e-3), (y - y_triton).abs().max().item()


@pytest.mark.parametrize("bias_flag", [False, True])
def test_poincare_fc_backward_matches_autograd(bias_flag: bool):
    torch.manual_seed(0)
    B, D, K = 4, 6, 5  # batch, in-dim, out-dim
    c = 0.1

    x = torch.randn(B, D, requires_grad=True)
    z = torch.randn(D, K, requires_grad=True)
    bias = torch.randn(K, requires_grad=True) if bias_flag else None

    out = poincare_fc_fwd_ref(x, z, bias, c)
    dout = torch.randn_like(out)

    # analytic gradients (detach to avoid autograd tracking)
    dx, dz, dbias = poincare_fc_bwd_ref(
        dout.detach(), x.detach(), z.detach(), bias.detach() if bias_flag else None, c
    )

    # autograd gradients
    out.backward(dout)

    assert torch.allclose(
        x.grad, dx, rtol=RTOL, atol=ATOL
    ), f"max diff: {torch.max(torch.abs(x.grad - dx))}"
    assert torch.allclose(
        z.grad, dz, rtol=RTOL, atol=ATOL
    ), f"max diff: {torch.max(torch.abs(z.grad - dz))}"
    if bias_flag:
        assert torch.allclose(
            bias.grad, dbias, rtol=RTOL, atol=ATOL
        ), f"max diff: {torch.max(torch.abs(bias.grad - dbias))}"
