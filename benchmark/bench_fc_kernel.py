import os

import torch
from hypll.manifolds.poincare_ball.curvature import Curvature
from hypll.manifolds.poincare_ball.manifold import PoincareBall
import triton

from hypll.kernels.fc_layer import FastPoincareFC
import hypll.nn as hnn


def build_layers(M, K, c, dtype, device):
    manifold = PoincareBall(c=Curvature(0.1, requires_grad=True))

    # fast triton implentation
    triton_layer = FastPoincareFC(M, K, c=c, dtype=dtype, device=device)

    _get_torch_layer = lambda: hnn.HLinear(M, K, manifold=manifold).to(device=device, dtype=dtype)

    # torch implementation in hnn
    torch_layer = _get_torch_layer()

    # compiled hnn
    compiled_layer = torch.compile(_get_torch_layer())

    # euclidean nn.Linear
    eu_layer = torch.nn.Linear(M, K, bias=True, device=device, dtype=dtype)

    return triton_layer, torch_layer, compiled_layer, eu_layer


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["B", "M", "K"],
        x_vals=[128 * i for i in range(1, 17)],  # B = M = K = 128, 256, ..., 2048
        line_arg="provider",
        line_vals=["triton", "torch", "compile", "matmul"],
        line_names=["Triton", "PyTorch", "Compiled PyTorch", "Euclidean"],
        styles=[("red", "-"), ("blue", "--"), ("green", "-."), ("orange", ":")],
        ylabel="TFLOPS",
        plot_name="poincare_fc-performance",
        args={"c": 0.1},
    )
)
def bench(B, M, K, provider: str, c: float):
    device = "cuda"
    dtype = torch.float32
    torch.manual_seed(0)

    global manifold, hnn

    x = torch.randn(B, M, device=device, dtype=dtype, requires_grad=True)

    triton_layer, torch_layer, compiled_layer, eu_layer = build_layers(
        M, K, c, dtype, device, manifold
    )

    if provider == "triton":
        layer = triton_layer
    elif provider == "torch":
        layer = torch_layer
    elif provider == "compile":
        layer = compiled_layer
    elif provider == "euclidean":
        layer = eu_layer
    else:
        raise ValueError(f"Unknown provider: {provider}")

    layer.train()

    # FLOPs: forward matmul ~2*B*M*K, backward wrt input another ~2*B*M*K,
    # backward wrt weight another ~2*B*M*K. So ~6*B*M*K total.
    # TODO: calculate FLOPS properly
    flop = 6.0 * B * M * K

    # Timed run: forward + backward
    def run():
        y = layer(x)
        loss = y.sum()
        loss.backward()
        x.grad = None  # reset for next iteration

    ms = triton.do_bench(run)
    tflops = flop / ms / 1e9
    return tflops


if __name__ == "__main__":
    # Run benchmarks and save plots
    os.makedirs("plots", exist_ok=True)
    bench.run(show_plots=False, print_data=True, save_path="./plots/poincare_fc_performance")
