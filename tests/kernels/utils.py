import torch


def assert_allclose(x, y, message: str = "", atol: float = 1e-3, equal_nan: bool = False):
    assert type(x) == type(y), f"{type(x)}, {type(y)}, {message}"
    if isinstance(x, float):
        assert abs(x - y) < atol, message
    elif not isinstance(x, torch.Tensor):
        assert x == y, message
    else:
        assert x.shape == y.shape, f"{message} shape mismatch: {x.shape}, {y.shape}"
        assert torch.allclose(
            x, y, atol=atol, equal_nan=equal_nan
        ), f"{message} | max diff: {(x - y).abs().max().item()}"


def safe_rand(*shape, grad: bool = False):
    return (torch.randn(*shape, dtype=torch.float32, requires_grad=grad).cuda() * 0.1).clamp(-1, 1)
