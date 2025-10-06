import torch
import pytest
from hypll.kernels.gemm import addmm


@pytest.mark.parametrize("v_shape", ["scalar", "1,1", "row", "col", "full"])
@pytest.mark.parametrize("a_transp", [False, True])
@pytest.mark.parametrize("b_transp", [False, True])
def test_addmm_fp32_correctness(v_shape, a_transp, b_transp):
    torch.manual_seed(0)
    device = "cuda"
    M, K, N = 64, 96, 80

    A = torch.randn(M, K, device=device, dtype=torch.float32)
    B = torch.randn(K, N, device=device, dtype=torch.float32)
    C = torch.randn(M, N, device=device, dtype=torch.float32)

    # scalar / broadcast shapes for v
    if v_shape == "scalar":
        v = torch.tensor(0.5, device=device, dtype=torch.float32)
    elif v_shape == "1,1":
        v = torch.randn(1, 1, device=device, dtype=torch.float32)
    elif v_shape == "row":
        v = torch.randn(M, 1, device=device, dtype=torch.float32)
    elif v_shape == "col":
        v = torch.randn(1, N, device=device, dtype=torch.float32)
    elif v_shape == "full":
        v = torch.randn(M, N, device=device, dtype=torch.float32)

    ref = v * C + A @ B

    # possibly transpose A or B before passing
    A_in = A.t().contiguous() if a_transp else A
    B_in = B.t().contiguous() if b_transp else B

    # call kernel
    out = addmm(v, C, A_in, B_in, a_transp=a_transp, b_transp=b_transp)

    assert torch.allclose(
        out, ref, atol=1e-4, rtol=1e-4
    ), f"Mismatch for v_shape={v_shape}, a_transp={a_transp}, b_transp={b_transp}"
