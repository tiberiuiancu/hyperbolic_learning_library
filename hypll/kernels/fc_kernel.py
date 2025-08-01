import math

import torch
import triton
import triton.language as tl


def get_autotune_configs_fwd(device=None):
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
    configs=get_autotune_configs_fwd(),
    key=["K", "M"],
)
@triton.jit
def _poincare_fc_fwd_kernel(
    X_ptr,  # [B, K]   fp32
    ZN_ptr,  # [M]      fp32 (‖z_k‖)
    XZ_ptr,  # [B, M]   fp32 – X @ Z
    R_ptr,  # [M]      fp32 or dummy when no bias
    numerator_ptr,  # [B, M]   fp32 – output numerator
    denominator_ptr,  # [B]   fp32 – output denominator
    v_ptr,  # [B, M]   fp32 – cache for v
    inner_ptr,  # [B, M]   fp32 – cache for inner
    lam_ptr,  # [B]   fp32 – cache for lambda
    twocsr_ptr,  # [M]   fp32 – cache for 2*c_sqrt*bias or dummy
    c,  # fp32  (curvature)
    c_sqrt,  # fp32  (sqrt(curvature))
    K: tl.constexpr,
    M: tl.constexpr,
    stride_x_batch: tl.constexpr,
    stride_xz_batch: tl.constexpr,
    stride_numerator_batch: tl.constexpr,
    stride_v_batch: tl.constexpr,
    stride_inner_batch: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Each program instance handles one output row (batch element)."""

    row = tl.program_id(0)

    X_ptr += row * stride_x_batch

    # 1. compute lambda
    offs_k = tl.arange(0, BLOCK_K)
    norm_x_sq = 0.0
    for k in range(0, K, BLOCK_K):
        mask_k = offs_k < K
        xk = tl.load(X_ptr + offs_k, mask=mask_k, other=0.0)
        norm_x_sq += tl.sum(xk * xk, axis=0)
        offs_k += BLOCK_K
    lam = 2.0 / (1.0 - c * norm_x_sq)  # shape [1]
    tl.store(lam_ptr + row, lam)

    # compute numerator and accumulate denominator
    den_acc = 0.0
    for m in range(0, M, BLOCK_M):
        m_ids = tl.arange(0, BLOCK_M) + m
        mask_m = m_ids < M
        zn = tl.load(ZN_ptr + m_ids, mask=mask_m, other=1.0)
        xz = tl.load(XZ_ptr + row * stride_xz_batch + m_ids, mask=mask_m, other=0.0)

        if HAS_BIAS:
            r = tl.load(R_ptr + m_ids, mask=mask_m, other=0.0)
            two_cs_r = 2.0 * c_sqrt * r
            tl.store(twocsr_ptr + m_ids, two_cs_r, mask=mask_m)
            inner = c_sqrt * lam / zn * xz
            inner = inner * cosh(two_cs_r) - (lam - 1) * sinh(two_cs_r)
        else:
            inner = c_sqrt * lam / zn * xz

        v = 2.0 * zn * asinh(inner)
        num = sinh(v) / c_sqrt

        den_acc += c * tl.sum(num * num)

        # write caches
        tl.store(numerator_ptr + row * stride_numerator_batch + m_ids, num, mask=mask_m)
        tl.store(v_ptr + row * stride_v_batch + m_ids, v, mask=mask_m)
        tl.store(inner_ptr + row * stride_inner_batch + m_ids, inner, mask=mask_m)

    # accumulate in denominator
    den = 1.0 + tl.sqrt(1.0 + den_acc)

    # write denominator
    tl.store(denominator_ptr + row, den)


def poincare_fully_connected_triton(x, z, r=None, c=1.0):
    """
    Host function to call the Triton Poincare fully connected kernel.
    Args:
        x: [B, K] input tensor (fp32, CUDA)
        z: [K, M] weight tensor (fp32, CUDA)
        r: [M] bias tensor or None (fp32, CUDA)
        c: curvature (tensor or float)
    Returns:
        Tuple: (output [B, M], numerator [B, M], denominator [B], v [B, M], inner [B, M], lam [B], twocsr [M])
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

    # Allocate output tensors and caches
    num = torch.empty((B, M), device=x.device, dtype=dtype)
    den = torch.empty((B,), device=x.device, dtype=dtype)
    v = torch.empty((B, M), device=x.device, dtype=dtype)
    inner = torch.empty((B, M), device=x.device, dtype=dtype)
    lam = torch.empty((B,), device=x.device, dtype=dtype)
    twocsr = torch.empty((M,), device=x.device, dtype=dtype)

    # Prepare bias
    has_bias = r is not None
    if not has_bias:
        r = torch.empty((M,), device=x.device, dtype=dtype)  # dummy

    # Launch Triton kernel, passing pointers to all outputs
    grid = (B,)
    _poincare_fc_fwd_kernel[grid](
        x,
        zn,
        xz,
        r,
        num,
        den,
        v,
        inner,
        lam,
        twocsr,
        c_val,
        c_sqrt,
        K,
        M,
        x.stride(0),
        xz.stride(0),
        num.stride(0),
        v.stride(0),
        inner.stride(0),
        has_bias,
    )

    out = num / den[:, None]
    return out, (num, v, inner, lam, den, twocsr)


def poincare_fc_fwd_ref(x, z, bias, c):
    """
    For my own reference; it's the same as the hypll implementation, but in a single function.
    Args:
        x: [B, K] input tensor (fp32)
        z: [K, M] weight tensor (fp32)
        bias: [M] bias tensor or None (fp32)
        c: curvature (float)
    Returns:
        Output tensor [B, M] (fp32)
    """
    c_sqrt = math.sqrt(c)
    lam = 2 / (1 - c * x.pow(2).sum(-1, keepdim=True))  # shape [B, 1]
    z_norm = z.norm(dim=0).clamp_min(1e-15)  # shape [M]

    inner_tmp = c_sqrt * lam / z_norm * (x @ z)  # shape [B, M]
    if bias is not None:
        two_cs_r = 2.0 * c_sqrt * bias
        inner = inner_tmp * torch.cosh(two_cs_r) - (lam - 1) * torch.sinh(two_cs_r)
    else:
        inner = inner_tmp

    v = 2 * z_norm * torch.asinh(inner)
    num = torch.sinh(v) / c_sqrt
    den = 1 + torch.sqrt(1 + c * num.pow(2).sum(-1, keepdim=True))  # shape [B, 1]
    out = num / den
    return out, (num, v, inner, lam.squeeze(), den, two_cs_r)


# TODO: pass sinh_twocsr as a parameter to avoid recomputing it
@triton.jit
def _dnum_dx(
    x,  # [BLOCK_K] - loaded slice of input
    v,  # [BLOCK_M]
    z,  # [BLOCK_K, BLOCK_M]
    inner,  # [BLOCK_M]
    twocsr,  # [BLOCK_M]
    lam,  # scalar
    c,  # scalar
    HAS_BIAS,
):
    # broadcast all vectors
    x = x[:, None]
    v = v[None, :]
    inner = inner[None, :]
    twocsr = twocsr[None, :]

    inner_sq = inner * inner
    x_sq = x * x

    y = cosh(v) * 2 / tl.sqrt(1 + inner_sq) * z * lam * (1 + c * lam * x_sq)
    if HAS_BIAS:
        sinh_twocsr = sinh(twocsr)
        y = y * cosh(twocsr) - sinh_twocsr * c * lam * lam * x + sinh_twocsr

    return y


def get_autotune_configs_bwd(device=None):
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


@triton.autotune(
    configs=get_autotune_configs_bwd(),
    key=["K", "M"],
)
@triton.jit
def _poincare_fc_bwd_dx_kernel(
    x_ptr,  # [B, K] - input
    z_ptr,  # [K, M] - weights
    dout_ptr,  # [B, M] - output gradient
    num_ptr,  # [B, M] - cache of "num" from fwd pass
    v_ptr,  # [B, M] - cache of "v" from fwd pass
    inner_ptr,  # [B, M] - cache of "inner" from fwd pass
    lam_ptr,  # [B] - cache of lambda from fwd pass
    den_ptr,  # [B] - cache of den from fwd pass
    twocsr_ptr,  # [M] - 2 * c_sqrt * bias, or dummy if no bias
    c,  # curvature, scalar
    out_ptr,  # [K] - pointer to write output to
    x_stride_b: tl.constexpr,
    z_stride_k: tl.constexpr,
    dout_stride_b: tl.constexpr,
    num_stride_b: tl.constexpr,
    v_stride_b: tl.constexpr,
    inner_stride_b: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    B: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Each program instance handles BLOCK_K elements of a single input row (batch element)"""

    b = tl.program_id(0)
    k = tl.program_id(1) * BLOCK_K

    # offset pointers to current row
    x_ptr += b * x_stride_b
    dout_ptr += b * dout_stride_b
    num_ptr += b * num_stride_b
    v_ptr += b * v_stride_b
    inner_ptr += b * inner_stride_b

    # load row of x in memory for the entire kernel execution
    offs_k = k + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    x = tl.load(x_ptr + offs_k, mask=mask_k)

    # load lambda and denominator value of current row
    lam = tl.load(lam_ptr + b, mask=b < B)
    den = tl.load(den_ptr + b, mask=b < B)

    ###### STAGE 1: precompute sum for derivative of den w.r.t. x
    offs_m = tl.arange(0, BLOCK_M)
    sum_buf = tl.zeros([BLOCK_K], dtype=tl.float32)
    for m in range(0, M, BLOCK_M):
        # calculate z offsets
        offs_z = offs_k[:, None] * z_stride_k + offs_m[None, :]
        mask_z = (offs_k[:, None] < K) & (offs_m[None, :] < M)

        # load data
        mask_m = offs_m < M
        num = tl.load(num_ptr + offs_m, mask=mask_m)
        v = tl.load(v_ptr + offs_m, mask=mask_m)
        inner = tl.load(inner_ptr + offs_m, mask=mask_m)
        z = tl.load(z_ptr + offs_z, mask=mask_z)
        twocsr = num
        if HAS_BIAS:
            twocsr = tl.load(twocsr_ptr + offs_m, mask=mask_m)

        # calculate derivative of num w.r.t. x
        d = _dnum_dx(x, v, z, inner, twocsr, lam, c, HAS_BIAS)  # shape [BLOCK_K, BLOCK_M]

        sum_buf += tl.sum(d * num, axis=1)

        # increment offsets
        offs_m += BLOCK_M

    ###### STAGE 2: compute the sum over dout M dimension
    # cache some values
    den_inv = 1 / den
    c_den_1 = c / (den - 1)

    offs_m = tl.arange(0, BLOCK_M)
    out_buf = tl.zeros([BLOCK_K], dtype=tl.float32)
    for m in range(0, M, BLOCK_M):
        # calculate z offsets
        offs_z = offs_k[:, None] * z_stride_k + offs_m[None, :]
        mask_z = (offs_k[:, None] < K) & (offs_m[None, :] < M)

        # load data
        mask_m = offs_m < M
        dout = tl.load(dout_ptr + offs_m, mask=mask_m)
        num = tl.load(num_ptr + offs_m, mask=mask_m)
        v = tl.load(v_ptr + offs_m, mask=mask_m)
        inner = tl.load(inner_ptr + offs_m, mask=mask_m)
        z = tl.load(z_ptr + offs_z, mask=mask_z)
        twocsr = num
        if HAS_BIAS:
            twocsr = tl.load(twocsr_ptr + offs_m)

        d = _dnum_dx(x, v, z, inner, twocsr, lam, c, HAS_BIAS)
        dy_dx = den_inv * d - c_den_1 * num[None, :] * sum_buf[:, None]  # shape [BLOCK_K, BLOCK_M]

        # aggregate sum over M dim, multiply by dout
        out_buf += tl.sum(dy_dx * dout, axis=1)

    tl.store(out_ptr + offs_k, out_buf, mask=mask_k)


def poincare_fc_bwd_dx_triton(x, z, dout, num, v, inner, lam, den, twocsr, c, has_bias):
    """
    Host function to call the Triton backward kernel for input gradients.
    Args:
        x: [B, K] input tensor (fp32, CUDA)
        z: [K, M] weight tensor (fp32, CUDA)
        dout: [B, M] output gradient (fp32, CUDA)
        num: [B, M] numerator cache from fwd (fp32, CUDA)
        v: [B, M] v cache from fwd (fp32, CUDA)
        inner: [B, M] inner cache from fwd (fp32, CUDA)
        lam: [B] lambda cache from fwd (fp32, CUDA)
        den: [B] denominator cache from fwd (fp32, CUDA)
        twocsr: [M] 2*c_sqrt*bias or dummy (fp32, CUDA)
        c: curvature (float)
        has_bias: bool
    Returns:
        dx: [B, K] input gradient (fp32, CUDA)
    """
    B, K = x.shape
    K2, _ = z.shape
    assert K == K2
    dx = torch.empty_like(x)

    c_val = float(c) if not torch.is_tensor(c) else float(c.item())

    def grid(meta):
        return (B, triton.cdiv(K, meta["BLOCK_K"]))

    _poincare_fc_bwd_dx_kernel[grid](
        x,
        z,
        dout,
        num,
        v,
        inner,
        lam,
        den,
        twocsr,
        c_val,
        dx,
        x.stride(0),
        z.stride(0),
        dout.stride(0),
        num.stride(0),
        v.stride(0),
        inner.stride(0),
        K,
        z.shape[1],
        x.shape[0],
        HAS_BIAS=has_bias,
    )
    return dx


class PoincareFCLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, z, r=None, c=1.0):
        # Call the Triton forward kernel
        out, (num, v, inner, lam, den, twocsr) = poincare_fully_connected_triton(x, z, r, c)
        ctx.save_for_backward(x, z, num, v, inner, lam, den, twocsr)
        ctx.has_bias = r is not None
        ctx.c = c
        return out

    @staticmethod
    def backward(ctx, dout):
        x, z, num, v, inner, lam, den, twocsr = ctx.saved_tensors
        c = ctx.c
        has_bias = ctx.has_bias
        # Compute input gradient
        dx = poincare_fc_bwd_dx_triton(x, z, dout, num, v, inner, lam, den, twocsr, c, has_bias)
        # TODO: implement dz and dbias
        dz = torch.empty((z.shape[0], dout.shape[1]), device=x.device, dtype=x.dtype)  # dummy
        dbias = torch.empty((dout.shape[1],), device=x.device, dtype=x.dtype) if has_bias else None
        return dx, dz, dbias, None
