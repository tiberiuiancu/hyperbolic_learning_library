import os

import torch
import triton

from hypll.kernels.fc_kernel import poincare_fc_fwd_ref, poincare_fully_connected_triton


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
def bench(B, M, K, c, provider):
    B, M, K = int(B), int(M), int(K)
    torch.manual_seed(42)
    device = "cuda"
    x = torch.randn(B, K, device=device, dtype=torch.float32)
    z = torch.randn(K, M, device=device, dtype=torch.float32)
    r = torch.randn(M, device=device, dtype=torch.float32)

    if provider == "triton":
        fn = lambda: poincare_fully_connected_triton(x, z, r, c)
    elif provider == "torch":
        fn = lambda: poincare_fc_fwd_ref(x, z, r, c)
    elif provider == "compile":
        compiled = torch.compile(poincare_fc_fwd_ref)
        fn = lambda: compiled(x, z, r, c)
    elif provider == "matmul":
        fn = lambda: x @ z
    else:
        raise ValueError(provider)

    ms = triton.testing.do_bench(fn)
    # ms is milliseconds, so convert to seconds for TFLOPS calculation
    seconds = ms / 1e3
    tflops = (2 * B * M * K) / (seconds * 1e12)
    return tflops


if __name__ == "__main__":
    # Run benchmarks and save plots
    os.makedirs("plots", exist_ok=True)
    bench.run(show_plots=False, print_data=True, save_path="./plots/poincare_fc_performance_B")
