import math
from typing import Optional

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


@triton.autotune(
    configs=get_autotune_configs_fwd(),
    key=["K", "M"],
)
@triton.jit
def _poincare_fc_fwd_kernel(
    X_ptr,  # [B, K]   fp32
    ZN_ptr,  # [M]      fp32 (‖z_k‖)
    XZ_ptr,  # [B, M]   fp32 – X @ Z
    B_ptr,  # [M]      fp32 or dummy when no bias; = 2 * cs * r
    num_ptr,  # [B, M]   fp32 – output numerator
    den_ptr,  # [B]   fp32 – output denominator
    lam_ptr,  # [B]   fp32 – cache for lambda
    c,  # fp32  (curvature)
    cs,  # fp32  (sqrt(curvature))
    K: tl.constexpr,
    M: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_xz_b: tl.constexpr,
    stride_num_b: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Each program instance handles one output row (batch element)."""

    row = tl.program_id(0)

    X_ptr += row * stride_x_b

    # 1. compute lambda
    offs_k = tl.arange(0, BLOCK_K)
    norm_x_sq = 0.0
    for _ in range(0, K, BLOCK_K):
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
        zn = tl.load(ZN_ptr + m_ids, mask=mask_m, other=0.0)
        xz = tl.load(XZ_ptr + row * stride_xz_b + m_ids, mask=mask_m, other=0.0)

        p = cs * lam / zn * xz
        if HAS_BIAS:
            b = tl.load(B_ptr + m_ids, mask=mask_m, other=0.0)
            eb = tl.exp(b)
            ebi = 1 / eb
            p = 0.5 * ((eb + ebi) * p - (eb - ebi) * (lam - 1))

        d = 2.0 * zn * tl.log(p + tl.sqrt(p * p + 1))
        ed = tl.exp(d)
        num = (ed - 1 / ed) / (2 * cs)

        den_acc += tl.sum(num * num)

        tl.store(num_ptr + row * stride_num_b + m_ids, num, mask=mask_m)

    # accumulate in denominator
    den = 1.0 + tl.sqrt(1.0 + c * den_acc)

    # write denominator
    tl.store(den_ptr + row, den)


def poincare_fully_connected_triton(x, z, r=None, c=1.0, return_cache: bool = False):
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
    cs = math.sqrt(c_val)
    zn = z.norm(dim=0).clamp_min(1e-15)  # [M]
    xz = x @ z  # [B, M]

    # Allocate output tensors and caches
    num = torch.empty((B, M), device=x.device, dtype=dtype)
    den = torch.empty((B,), device=x.device, dtype=dtype)
    lam = torch.empty((B,), device=x.device, dtype=dtype)

    # Prepare bias
    has_bias = r is not None
    b = torch.empty((M,), device=x.device, dtype=dtype)  # dummy
    if has_bias:
        b = 2 * cs * r

    # Launch Triton kernel, passing pointers to all outputs
    grid = (B,)
    _poincare_fc_fwd_kernel[grid](
        x,
        zn,
        xz,
        b,
        num,
        den,
        lam,
        c_val,
        cs,
        K,
        M,
        x.stride(0),
        xz.stride(0),
        num.stride(0),
        has_bias,
    )

    out = num / den[:, None]

    if return_cache:
        return out, (num, lam, den)
    return out


def poincare_fc_fwd_ref(
    x: torch.Tensor,
    z: torch.Tensor,
    bias: Optional[torch.Tensor],
    c: torch.Tensor,
    dim: int = -1,
    return_cache: bool = False,
) -> torch.Tensor:
    x = x.movedim(dim, -1)  # features last
    cs = c.sqrt()
    lam = 2 / (1 - c * x.pow(2).sum(-1, keepdim=True))
    zn = z.norm(dim=0).clamp_min(1e-15)

    p = (cs * lam / zn) * torch.matmul(x, z)

    if bias is not None:
        b = 2 * cs * bias
        eb = torch.exp(b)
        ebi = 1 / eb
        p = 0.5 * ((eb + ebi) * p - (eb - ebi) * (lam - 1))

    d = 2 * zn * torch.log(p + torch.sqrt(p * p + 1))  # scaled distance
    ed = torch.exp(d)
    num = (ed - 1 / ed) / (2 * cs)  # sinh(d)/cs
    den = 1 + torch.sqrt(1 + c * num.pow(2).sum(-1, keepdim=True))

    y = (num / den).movedim(-1, dim)

    if return_cache:
        return y, (num, lam, den)

    return y


# class PoincareFCLayer(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, z, r=None, c=1.0):
#         out, (num, v, inner, lam, den, twocsr) = poincare_fully_connected_triton(x, z, r, c)
#         ctx.save_for_backward(x, z, num, v, inner, lam, den, twocsr)
#         ctx.has_bias = r is not None
#         ctx.c = c
#         return out

#     @staticmethod
#     def backward(ctx, dout):
#         x, z, num, v, inner, lam, den, twocsr = ctx.saved_tensors
#         c = ctx.c
#         has_bias = ctx.has_bias
#         dx = poincare_fc_bwd_dx_triton(x, z, dout, num, v, inner, lam, den, twocsr, c, has_bias)
#         # TODO: implement dz and dbias
#         dz = torch.empty((z.shape[0], dout.shape[1]), device=x.device, dtype=x.dtype)  # dummy
#         dbias = torch.empty((dout.shape[1],), device=x.device, dtype=x.dtype) if has_bias else None
#         return dx, dz, dbias, None
