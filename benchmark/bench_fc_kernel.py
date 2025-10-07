import os

import torch
from hypll.manifolds.poincare_ball.curvature import Curvature
from hypll.manifolds.poincare_ball.manifold import PoincareBall
import triton

import hypll.nn as hnn
from hypll.tensors.tangent_tensor import TangentTensor

# configs for B, M, K
configs = [
    # Small
    [8, 256, 256],
    [16, 512, 512],
    [32, 512, 512],
    # Medium
    [64, 1024, 1024],
    [128, 1024, 2048],
    [128, 2048, 1024],
    [256, 2048, 2048],
    # Large
    [512, 4096, 2048],
    [512, 2048, 4096],
    [1024, 4096, 4096],
    [1024, 8192, 4096],
]

configs = [
    # --- Small models (edge devices, simple MLPs, tabular data) ---
    [32, 128, 256],
    [64, 256, 512],
    [128, 512, 1024],
    # --- Medium models (typical academic or production MLPs/transformers) ---
    [256, 1024, 2048],
    [512, 2048, 4096],
    [512, 4096, 8192],
    [1024, 4096, 16384],
    [2048, 8192, 16384],
]


def build_layer(M, K, c, dtype, device, config, compiled: bool = False):
    manifold = None
    if config == "euclidean":
        layer = torch.nn.Linear(M, K, bias=True, device=device, dtype=dtype)
    else:
        manifold = PoincareBall(Curvature(c), backend=config)
        layer = hnn.HLinear(M, K, manifold=manifold).to(device=device, dtype=dtype)

    if compiled:
        layer = torch.compile(layer)

    return layer, manifold


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["cfg_id"],
        x_vals=list(range(len(configs))),
        line_arg="provider",
        line_vals=["triton", "triton-own", "triton-own-transp", "torch", "compile", "euclidean"],
        line_names=[
            "Triton",
            "Triton own gemm",
            "Triton own gemm transp",
            "PyTorch",
            "Compiled PyTorch",
            "Euclidean",
        ],
        styles=[
            ("red", "-"),
            ("red", "--"),
            ("red", "-."),
            ("blue", "--"),
            ("green", "-."),
            ("orange", ":"),
        ],
        ylabel="Execution Time (ms)",
        plot_name="poincare_fc-performance",
        args={"c": 0.1},
    )
)
def bench(cfg_id, provider: str, c: float):
    device = "cuda"
    dtype = torch.float32
    torch.manual_seed(0)

    B, M, K = configs[cfg_id]
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


if __name__ == "__main__":
    # Run benchmarks and save plots
    os.makedirs("plots", exist_ok=True)
    bench.run(show_plots=False, print_data=True, save_path="./plots/poincare_fc_performance")
