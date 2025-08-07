import math
from typing import Optional

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
def _single_block_fwd(
    b,
    lam,
    zn,
    xz,
    cs,
):
    p0 = cs * lam / zn * xz
    eb = tl.exp(b)
    ebi = 1 / eb
    p = 0.5 * ((eb + ebi) * p0 - (eb - ebi) * (lam - 1))

    sq_p2_1 = tl.sqrt(p * p + 1)
    log_p_sq = tl.log(p + sq_p2_1)
    d = 2.0 * zn * log_p_sq
    ed = tl.exp(d)
    edi = 1 / ed
    num = 0.5 * (ed - edi) / cs

    return sq_p2_1, log_p_sq, eb, ebi, ed, edi, num


@triton.autotune(
    configs=get_autotune_configs(),
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
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Each program instance handles one output row (batch element)."""

    row = tl.program_id(0)

    # offset pointers
    X_ptr += row * stride_x_b
    num_ptr += row * stride_num_b
    XZ_ptr += row * stride_xz_b
    lam_ptr += row

    # 1. compute lambda
    offs_k = tl.arange(0, BLOCK_K)
    norm_x_sq = 0.0
    for _ in range(0, K, BLOCK_K):
        mask_k = offs_k < K
        xk = tl.load(X_ptr + offs_k, mask=mask_k, other=0.0)
        norm_x_sq += tl.sum(xk * xk)
        offs_k += BLOCK_K
    lam = 2.0 / (1.0 - c * norm_x_sq)  # shape [1]
    tl.store(lam_ptr, lam)

    # compute numerator and accumulate denominator
    den_acc = 0.0
    for m in range(0, M, BLOCK_M):
        m_ids = tl.arange(0, BLOCK_M) + m
        mask_m = m_ids < M
        zn = tl.load(ZN_ptr + m_ids, mask=mask_m, other=1.0)
        xz = tl.load(XZ_ptr + m_ids, mask=mask_m, other=0.0)
        b = tl.load(B_ptr + m_ids, mask=mask_m, other=0.0)

        # we only need num
        _1, _2, _3, _4, _5, _6, num = _single_block_fwd(b, lam, zn, xz, cs)

        den_acc += tl.sum(num * num)

        tl.store(num_ptr + m_ids, num, mask=mask_m)

    # calculate denominator value
    den = 1.0 + tl.sqrt(1.0 + c * den_acc)

    # write denominator
    tl.store(den_ptr + row, den)


def poincare_fc_fwd_triton(x, z, r=None, c=1.0, return_cache: bool = False):
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

    # Compute required intermediates
    c_val = float(c) if not torch.is_tensor(c) else float(c.item())
    cs = math.sqrt(c_val)
    zn = z.norm(dim=0).clamp_min(1e-15)  # [M]
    xz = x @ z  # [B, M]

    # Allocate output tensors and caches
    num = torch.empty((B, M), dtype=torch.float32).cuda()
    den = torch.empty((B,), dtype=torch.float32).cuda()
    lam = torch.empty((B,), dtype=torch.float32).cuda()

    # Prepare bias
    if r is None:
        r = torch.zeros((M,)).cuda()  # dummy
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
    )

    out = num / den[:, None]

    if return_cache:
        return out, (x, z, xz, zn, b, lam, den, c, cs)
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

    xz = torch.matmul(x, z)
    p = (cs * lam / zn) * xz

    if bias is None:
        bias = torch.zeros(z.shape[1], dtype=torch.float32).cuda()
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
        return y, (
            x,
            z,
            xz,
            zn,
            b,
            lam.squeeze(),
            den.squeeze(),
            c,
            cs.item(),
        )

    return y
