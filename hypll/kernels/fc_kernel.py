import os
import math
import triton.testing

from hypll.manifolds.poincare_ball.math.linalg import poincare_fully_connected

import torch
import triton
import triton.language as tl


def get_autotune_configs(device=None):
    if device is None:
        device = torch.cuda.current_device()
    cc_major, _ = torch.cuda.get_device_capability(device)
    stage_set = (1,) if cc_major < 8 else (1, 2)  # Turing vs Ampere+

    # (BLOCK_K, BLOCK_M, num_warps)
    tiles = [
        (32, 32, 1),  # tiny rows
        (64, 64, 2),  # mid‑size
        (128, 64, 4),  # long‑K
        (128, 128, 4),  # large K & M
    ]

    configs = []
    for bk, bm, w in tiles:
        for s in stage_set:
            configs.append(
                triton.Config(
                    {"BLOCK_K": bk, "BLOCK_M": bm},
                    num_warps=w,
                    num_stages=s,
                )
            )
    return configs


@triton.jit
def sinh(x):
    e = tl.exp(x)
    ei = 1.0 / e
    return 0.5 * (e - ei)


@triton.jit
def asinh(x):
    return tl.log(x + tl.sqrt(x * x + 1.0))


@triton.jit
def cosh(x):
    e = tl.exp(x)
    ei = 1.0 / e
    return 0.5 * (e + ei)


@triton.autotune(
    configs=get_autotune_configs(),
    key=["K", "M"],
)
@triton.jit
def _poincare_fc_fwd_kernel(
    X_ptr,  # [B, K]   fp32
    ZN_ptr,  # [M]      fp32 (‖z_k‖)
    XZ_ptr,  # [B, M]   fp32 – X @ Z
    R_ptr,  # [M]      fp32 or dummy when no bias
    numerator_ptr,  # [B, M]   fp32 – output numerator
    denominator_ptr,  # [B, M]   fp32 – output denominator
    c,  # fp32  (curvature)
    c_sqrt,  # fp32  (sqrt(curvature))
    K: tl.constexpr,
    M: tl.constexpr,
    stride_x_batch: tl.constexpr,
    stride_xz_batch: tl.constexpr,
    stride_numerator_batch: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Each program instance handles one output row (batch element)."""

    row = tl.program_id(0)

    # 1. compute lambda
    offs_k = tl.arange(0, BLOCK_K)
    x_row_ptr = X_ptr + row * stride_x_batch
    norm_x_sq = tl.zeros([1], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_ids = k + offs_k
        mask_k = k_ids < K
        xk = tl.load(x_row_ptr + k_ids, mask=mask_k, other=0.0)
        norm_x_sq += tl.sum(xk * xk, axis=0)
    lam = 2.0 / (1.0 - c * norm_x_sq)  # shape [1

    # compute numerator and accumulate denominator
    denominator_acc = 0.0
    for m in range(0, M, BLOCK_M):
        m_ids = tl.arange(0, BLOCK_M) + m
        mask_m = m_ids < M
        zn = tl.load(ZN_ptr + m_ids, mask=mask_m, other=1.0)
        xz = tl.load(XZ_ptr + row * stride_xz_batch + m_ids, mask=mask_m, other=0.0)

        inner = c_sqrt * lam / zn * xz
        if HAS_BIAS:
            two_cs_r = 2.0 * c_sqrt * tl.load(R_ptr + m_ids, mask=mask_m, other=0.0)
            inner = inner * cosh(two_cs_r) - (lam - 1) * sinh(two_cs_r)

        v = 2.0 * zn * asinh(inner)
        numerator = sinh(v) / c_sqrt

        # write numerator
        tl.store(numerator_ptr + row * stride_numerator_batch + m_ids, numerator, mask=mask_m)

        # accumulate in denominator
        denominator_acc += tl.sum(numerator * numerator)

    denominator_acc = 1.0 + tl.sqrt(1.0 + c * denominator_acc)

    # write denominator
    tl.store(denominator_ptr + row, denominator_acc)


def poincare_fully_connected_triton(x, z, r=None, c=1.0):
    """
    Host function to call the Triton Poincare fully connected kernel.
    Args:
        x: [B, K] input tensor (fp32, CUDA)
        z: [K, M] weight tensor (fp32, CUDA)
        r: [M] bias tensor or None (fp32, CUDA)
        c: curvature (float or tensor)
    Returns:
        Output tensor [B, M] (fp32, CUDA)
    """
    assert x.is_cuda and z.is_cuda, "Input tensors must be on CUDA"
    B, K = x.shape
    K2, M = z.shape
    assert K == K2, "Dimension mismatch"
    dtype = x.dtype

    # Compute required intermediates
    c_val = float(c) if not torch.is_tensor(c) else float(c.item())
    c_sqrt = math.sqrt(c_val)
    zn = z.norm(dim=0).clamp_min(1e-15)  # [M]
    xz = x @ z  # [B, M]

    # Allocate output tensors
    numerator = torch.empty((B, M), device=x.device, dtype=dtype)
    denominator = torch.empty((B,), device=x.device, dtype=dtype)

    # Prepare bias
    has_bias = r is not None
    if not has_bias:
        r = torch.empty((M,), device=x.device, dtype=dtype)  # dummy

    # Launch Triton kernel
    grid = (B,)
    _poincare_fc_fwd_kernel[grid](
        x,
        zn,
        xz,
        r,
        numerator,
        denominator,
        c_val,
        c_sqrt,
        K,
        M,
        x.stride(0),
        xz.stride(0),
        numerator.stride(0),
        HAS_BIAS=has_bias,
    )

    # Final output: numerator / denominator[:, None]
    return numerator / denominator[:, None]


def poincare_fc_ref(x, z, bias, c):
    c_sqrt = math.sqrt(c)
    lam = 2 / (1 - c * x.pow(2).sum(-1, keepdim=True))
    z_norm = z.norm(dim=0).clamp_min(1e-15)

    inner = c_sqrt * lam / z_norm * (x @ z)
    if bias is not None:
        two_cs_r = 2.0 * c_sqrt * bias
        inner = inner * torch.cosh(two_cs_r) - (lam - 1) * torch.sinh(two_cs_r)

    v = 2 * z_norm * torch.asinh(inner)
    y = torch.sinh(v) / c_sqrt
    numerator = y
    denominator = 1 + torch.sqrt(1 + c * y.pow(2).sum(-1, keepdim=True))  # shape [B, 1]
    return numerator / denominator


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
        fn = lambda: poincare_fc_ref(x, z, r, c)
    elif provider == "compile":
        compiled = torch.compile(poincare_fc_ref)
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


def test_reference():
    """Run a simple correctness test."""
    B, K, M = 256, 128, 64
    c = torch.tensor(0.1, dtype=torch.float32).cuda()
    x = torch.randn(B, K, dtype=torch.float32).cuda()
    z = torch.randn(K, M, dtype=torch.float32).cuda()
    r = torch.randn(M, dtype=torch.float32).cuda()

    y = poincare_fully_connected(x, z, r, c)
    y_ref = poincare_fc_ref(x, z, r, c)
    y_triton = poincare_fully_connected_triton(x, z, r, c)

    assert torch.allclose(y, y_ref, atol=1e-3), (y - y_ref).abs().max().item()
    assert torch.allclose(y, y_triton, atol=1e-3), (y - y_triton).abs().max().item()


if __name__ == "__main__":
    test_reference()

    # Run benchmarks and save plots
    os.makedirs("plots", exist_ok=True)
    bench.run(show_plots=False, print_data=True, save_path="./plots/poincare_fc_performance_B")
