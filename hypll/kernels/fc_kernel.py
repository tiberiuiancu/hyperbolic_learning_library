import math
import torch
import triton
import triton.language as tl

from hypll.manifolds.poincare_ball.math.linalg import poincare_fully_connected

# -------------------------------------------------------------
#  Autotuning configs
# -------------------------------------------------------------


def get_autotune_configs(device=None):
    if device is None:
        device = torch.cuda.current_device()
    cc_major, _ = torch.cuda.get_device_capability(device)
    stage_set = (1,) if cc_major < 8 else (1, 2)

    # (BLOCK_K, BLOCK_M, num_warps)
    tiles = [
        (32, 32, 1),
        (64, 64, 2),
        (128, 64, 4),
        (128, 128, 4),
    ]
    cfgs = []
    for bk, bm, w in tiles:
        for s in stage_set:
            cfgs.append(triton.Config({"BLOCK_K": bk, "BLOCK_M": bm}, num_warps=w, num_stages=s))
    return cfgs


# -------------------------------------------------------------
#  Hyperbolic helpers
# -------------------------------------------------------------


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


# -------------------------------------------------------------
#  Forward kernel – 2‑D launch grid
# -------------------------------------------------------------


@triton.autotune(configs=get_autotune_configs(), key=["K", "M"])
@triton.jit
def _poincare_fc_fwd_kernel_opt(
    X_ptr,  # [B, K]   fp32
    INV_ZN_ptr,  # [M]      fp32 (1 / ‖z_k‖)
    XZ_ptr,  # [B, M]   fp32 – x @ z
    R_ptr,  # [M]      fp32 bias (or dummy)
    NUM_ptr,  # [B, M]   fp32 – numerator
    DEN_PTR,  # [B]      fp32 – per‑row accumulator
    c,  # fp32 curvature
    two_cs,  # fp32 = 2·sqrt(c)
    K: tl.constexpr,
    M: tl.constexpr,
    stride_x_batch: tl.constexpr,
    stride_xz_batch: tl.constexpr,
    stride_num_batch: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Each program handles a tile (row, BLOCK_M columns)."""

    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    # Column indices for this program
    col_start = pid_col * BLOCK_M
    offs_m = tl.arange(0, BLOCK_M)
    m_ids = col_start + offs_m
    mask_m = m_ids < M

    # ---------------------------------------------------------
    #  λ(x): double‑buffered accumulation over K
    # ---------------------------------------------------------
    offs_k = tl.arange(0, BLOCK_K)
    x_row_ptr = X_ptr + pid_row * stride_x_batch

    # First tile
    xk = tl.load(x_row_ptr + offs_k, mask=offs_k < K, other=0.0)
    norm_x_sq = tl.sum(xk * xk, axis=0)

    for k in range(BLOCK_K, K, BLOCK_K):
        xk = tl.load(x_row_ptr + k + offs_k, mask=k + offs_k < K, other=0.0)
        norm_x_sq += tl.sum(xk * xk, axis=0)

    lam = 2.0 / (1.0 - c * norm_x_sq)  # scalar per row

    # ---------------------------------------------------------
    #  Column‑tile operands (cache‑hinted global loads)
    # ---------------------------------------------------------
    xz = tl.load(
        XZ_ptr + pid_row * stride_xz_batch + m_ids, mask=mask_m, other=0.0, cache_modifier=".ca"
    )
    inv_zn = tl.load(INV_ZN_ptr + m_ids, mask=mask_m, other=0.0, cache_modifier=".ca")

    # inv_zn == 0 never happens thanks to host clamp, but avoid div‑by‑0 anyway
    zn = 1.0 / tl.where(inv_zn == 0.0, 1e-15, inv_zn)

    inner = (lam * (two_cs * 0.5)) * (xz * inv_zn)  # sqrt(c)·λ/‖z‖·⟨x,z⟩

    if HAS_BIAS:
        bias = tl.load(R_ptr + m_ids, mask=mask_m, other=0.0, cache_modifier=".ca")
        two_cs_r = bias * two_cs
        inner = inner * cosh(two_cs_r) - (lam - 1.0) * sinh(two_cs_r)

    v = 2.0 * zn * asinh(inner)
    numerator = sinh(v) / (two_cs * 0.5)  # divide by sqrt(c)

    # Write numerator tile
    tl.store(NUM_ptr + pid_row * stride_num_batch + m_ids, numerator, mask=mask_m)

    # Atomic reduction for denominator
    denom_partial = tl.sum(numerator * numerator, axis=0)
    tl.atomic_add(DEN_PTR + pid_row, denom_partial)


# -------------------------------------------------------------
#  Host wrapper
# -------------------------------------------------------------


def poincare_fully_connected_triton(x, z, r=None, c=1.0):
    """Optimised Poincaré fully‑connected using Triton.
    Implements fixes (1,2,4,5,7).
    Args:
        x : [B, K] fp32 CUDA
        z : [K, M] fp32 CUDA
        r : [M]    fp32 CUDA bias or None
        c : float  curvature
    Returns:
        y : [B, M] fp32 CUDA
    """
    assert x.is_cuda and z.is_cuda, "inputs must be CUDA tensors"
    B, K = x.shape
    K2, M = z.shape
    assert K == K2, "dimension mismatch"

    dtype = x.dtype
    c_val = float(c) if not torch.is_tensor(c) else float(c.item())
    two_cs = 2.0 * math.sqrt(c_val)

    # Host‑side pre‑computations
    zn = z.norm(dim=0).clamp_min(1e-15)
    inv_zn = 1.0 / zn
    xz = x @ z

    numerator = torch.empty((B, M), device=x.device, dtype=dtype)
    denom_sq = torch.zeros((B,), device=x.device, dtype=dtype)

    has_bias = r is not None
    if not has_bias:
        r = torch.empty((M,), device=x.device, dtype=dtype)

    # Launch grid: (rows, column‑tiles)
    BLOCK_M_MAX = 128  # keep in sync with autotune shapes
    grid = (B, (M + BLOCK_M_MAX - 1) // BLOCK_M_MAX)

    _poincare_fc_fwd_kernel_opt[grid](
        x,
        inv_zn,
        xz,
        r,
        numerator,
        denom_sq,
        c_val,
        two_cs,
        K,
        M,
        x.stride(0),
        xz.stride(0),
        numerator.stride(0),
        HAS_BIAS=has_bias,
    )

    denominator = 1.0 + torch.sqrt(1.0 + c_val * denom_sq)
    return numerator / denominator[:, None]


# -------------------------------------------------------
#  Reference fallback (PyTorch) – useful for correctness + timing
# -------------------------------------------------------


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
        x_names=["B"],  # varying batch size
        x_vals=[32, 64, 256, 512, 1024, 2048],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("red", "-"), ("blue", "--")],
        ylabel="TFLOPS",
        plot_name="poincare_fc-performance-batch",
        args={
            "K": 1024,
            "M": 1024,
            "c": 0.1,
        },
    )
)
def bench_B(B, K, M, c, provider):
    torch.manual_seed(42)
    device = "cuda"
    x = torch.randn(B, K, device=device, dtype=torch.float32)
    z = torch.randn(K, M, device=device, dtype=torch.float32)
    r = torch.randn(M, device=device, dtype=torch.float32)

    if provider == "triton":
        fn = lambda: poincare_fully_connected_triton(x, z, r, c)
    elif provider == "torch":
        fn = lambda: poincare_fc_ref(x, z, r, c)
    else:
        raise ValueError(provider)

    ms = triton.testing.do_bench(fn) * 1e3
    return ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],  # varying output features
        x_vals=[128, 256, 512, 1024, 2048, 4096, 8192],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("red", "-"), ("blue", "--")],
        ylabel="TFLOPS",
        plot_name="poincare_fc-performance-M",
        args={
            "B": 256,
            "K": 1024,
            "c": 0.1,
        },
    )
)
def bench_M(B, K, M, c, provider):
    torch.manual_seed(42)
    device = "cuda"
    x = torch.randn(B, K, device=device, dtype=torch.float32)
    z = torch.randn(K, M, device=device, dtype=torch.float32)
    r = torch.randn(M, device=device, dtype=torch.float32)

    if provider == "triton":
        fn = lambda: poincare_fully_connected_triton(x, z, r, c)
    elif provider == "torch":
        fn = lambda: poincare_fc_ref(x, z, r, c)
    else:
        raise ValueError(provider)

    ms = triton.testing.do_bench(fn) * 1e3
    return ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["K"],  # varying input features
        x_vals=[128, 256, 512, 1024, 2048, 4096, 8192],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("red", "-"), ("blue", "--")],
        ylabel="TFLOPS",
        plot_name="poincare_fc-performance-K",
        args={
            "B": 256,
            "M": 512,
            "c": 0.1,
        },
    )
)
def bench_K(B, K, M, c, provider):
    torch.manual_seed(42)
    device = "cuda"
    x = torch.randn(B, K, device=device, dtype=torch.float32)
    z = torch.randn(K, M, device=device, dtype=torch.float32)
    r = torch.randn(M, device=device, dtype=torch.float32)

    if provider == "triton":
        fn = lambda: poincare_fully_connected_triton(x, z, r, c)
    elif provider == "torch":
        fn = lambda: poincare_fc_ref(x, z, r, c)
    else:
        raise ValueError(provider)

    ms = triton.testing.do_bench(fn) * 1e3
    return ms


def test_reference():
    """Run a simple correctness test."""
    B, K, M = 256, 128, 64
    c = torch.tensor(0.1, dtype=torch.float32)
    x = torch.randn(B, K, dtype=torch.float32)
    z = torch.randn(K, M, dtype=torch.float32)
    r = torch.randn(M, dtype=torch.float32)

    y = poincare_fully_connected(x, z, r, c)
    y_ref = poincare_fc_ref(x, z, r, c)

    assert torch.allclose(y, y_ref, atol=1e-5, equal_nan=True), (y - y_ref).abs().max().item()


if __name__ == "__main__":
    import os

    os.makedirs("plots", exist_ok=True)
    bench_B.run(show_plots=False, print_data=True, save_path="./plots/poincare_fc_performance_B")
    bench_M.run(show_plots=False, print_data=True, save_path="./plots/poincare_fc_performance_M")
    bench_K.run(show_plots=False, print_data=True, save_path="./plots/poincare_fc_performance_K")
