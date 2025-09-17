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
    [4, 128, 128],
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
    # Extra large (stress test)
    [2048, 8192, 8192],
    [4096, 8192, 4096],
    [4096, 16384, 8192],
]


def build_layers(M, K, c, dtype, device):
    manifold_triton = PoincareBall(Curvature(c), use_triton_backend=True)
    manifold_default = PoincareBall(Curvature(c), use_triton_backend=False)

    # fast triton implentation
    _get_hyperbolic_layer = lambda manifold: hnn.HLinear(M, K, manifold=manifold).to(
        device=device, dtype=dtype
    )

    triton_layer = _get_hyperbolic_layer(manifold_triton)

    # torch implementation in hnn
    torch_layer = _get_hyperbolic_layer(manifold_default)

    # compiled hnn
    compiled_layer = torch.compile(_get_hyperbolic_layer(manifold_default))

    # euclidean nn.Linear
    eu_layer = torch.nn.Linear(M, K, bias=True, device=device, dtype=dtype)

    return triton_layer, torch_layer, compiled_layer, eu_layer, manifold_triton, manifold_default


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["cfg_id"],
        x_vals=[i for i in range(8)],
        line_arg="provider",
        line_vals=["triton", "torch", "compile", "euclidean"],
        line_names=["Triton", "PyTorch", "Compiled PyTorch", "Euclidean"],
        styles=[("red", "-"), ("blue", "--"), ("green", "-."), ("orange", ":")],
        ylabel="TFLOPS",
        plot_name="poincare_fc-performance",
        args={"c": 0.1},
    )
)
def bench(cfg_id, provider: str, c: float):
    device = "cuda"
    dtype = torch.float32
    torch.manual_seed(0)

    B, M, K = configs[cfg_id]
    triton_layer, torch_layer, compiled_layer, eu_layer, manifold_triton, manifold_default = (
        build_layers(M, K, c, dtype, device)
    )

    if provider == "triton":
        layer = triton_layer
        manifold = manifold_triton
    elif provider == "torch":
        layer = torch_layer
        manifold = manifold_default
    elif provider == "compile":
        layer = compiled_layer
        manifold = manifold_default
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
        x = torch.randn(B, M, device=device, dtype=dtype, requires_grad=True)
        if provider != "euclidean":
            tangents = TangentTensor(data=x, manifold=manifold)
            x = manifold.expmap(tangents)

        y = layer(x)

        if provider != "euclidean":
            y = y.tensor

        loss = y.sum()
        loss.backward()
        x.grad = None  # reset for next iteration

    ms = triton.testing.do_bench(run)
    # tflops = flop / ms / 1e9
    return ms


if __name__ == "__main__":
    # Run benchmarks and save plots
    os.makedirs("plots", exist_ok=True)
    bench.run(show_plots=False, print_data=True, save_path="./plots/poincare_fc_performance")
