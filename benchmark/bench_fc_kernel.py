import os

import torch
from hypll.manifolds.poincare_ball.curvature import Curvature
from hypll.manifolds.poincare_ball.manifold import PoincareBall
import triton

import hypll.nn as hnn
from hypll.tensors.tangent_tensor import TangentTensor

# configs for B, M, K
default_b, default_m, default_k = 128, 2048, 2048
b_sweep = [2**i for i in range(12)]
m_sweep = [2**i for i in range(7, 15)]
k_sweep = [2**i for i in range(7, 15)]


def get_bench_kwargs():
    return dict(
        line_arg="provider",
        line_vals=["triton", "torch", "compile", "euclidean"],
        line_names=[
            "Triton",
            "PyTorch",
            "Compiled PyTorch",
            "Euclidean",
        ],
        styles=[
            ("red", "-"),
            ("blue", "--"),
            ("green", "-."),
            ("orange", ":"),
        ],
        ylabel="Execution Time (ms)",
        args={"c": 0.1},
        x_log=True,
    )


def build_layer(M, K, c, dtype, device, config, compiled: bool = False):
    manifold = None
    if config == "euclidean":
        layer = torch.nn.Linear(M, K, bias=True, device=device, dtype=dtype)
    else:
        manifold = PoincareBall(Curvature(c))
        layer = hnn.HLinear(M, K, manifold=manifold).to(device=device, dtype=dtype)

    if compiled:
        layer = torch.compile(layer)

    return layer, manifold


def bench(B, M, K, c, provider):
    device = "cuda"
    dtype = torch.float32
    torch.manual_seed(0)

    compiled = False
    if provider == "compile":
        provider = "torch"
        compiled = True
    layer, manifold = build_layer(M, K, c, torch.float32, "cuda", provider, compiled)
    layer.train()

    def run():
        x = torch.randn(B, M, device=device, dtype=dtype, requires_grad=True)
        if provider != "euclidean":
            tangents = TangentTensor(data=x, manifold=manifold)
            x = manifold.expmap(tangents)
        y = layer(x)
        if provider != "euclidean":
            y = y.tensor
        y.sum().backward()

    ms = triton.testing.do_bench(run)
    return ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=b_sweep,
        plot_name=f"poincare_fc_performance_b",
        **get_bench_kwargs(),
    )
)
def bench_b(batch_size, provider: str, c: float):
    return bench(batch_size, default_m, default_k, c, provider)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["input_size"],
        x_vals=m_sweep,
        plot_name=f"poincare_fc_performance_m",
        **get_bench_kwargs(),
    )
)
def bench_m(input_size, provider: str, c: float):
    return bench(default_b, input_size, default_k, c, provider)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["hidden_size"],
        x_vals=k_sweep,
        plot_name=f"poincare_fc_performance_k",
        **get_bench_kwargs(),
    )
)
def bench_k(hidden_size, provider: str, c: float):
    return bench(default_b, default_m, hidden_size, c, provider)


if __name__ == "__main__":
    # Run benchmarks and save plots
    os.makedirs("plots", exist_ok=True)
    run_kwargs = dict(
        show_plots=False, print_data=True, save_path="./plots/poincare_fc_performance"
    )
    bench_b.run(**run_kwargs)
    bench_m.run(**run_kwargs)
    bench_k.run(**run_kwargs)
