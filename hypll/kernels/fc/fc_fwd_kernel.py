import math
from typing import Optional

import torch
import triton
import triton.language as tl

from hypll.kernels.utils import Tensor1D, Tensor2D, validate_tensors


def get_autotune_configs(device=None):
    if device is None:
        device = torch.cuda.current_device()
    cc_major, _ = torch.cuda.get_device_capability(device)
    stage_set = (1,) if cc_major < 8 else (1, 2)  # Turing vs Ampere+

    # (BLOCK_K, BLOCK_M, num_warps)
    tiles = [
        # Small tiles (good for small problems / latency bound)
        (16, 16, 1),
        (32, 16, 2),
        (16, 64, 2),
        # Balanced medium tiles
        (32, 32, 2),
        (64, 32, 2),
        (64, 64, 2),
        (64, 64, 4),
        # Rectangular, long-K or long-M
        (128, 32, 4),  # tall
        (32, 128, 4),  # wide
        (128, 64, 4),
        (64, 128, 4),
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
def single_block_fwd(
    b,
    lam,
    zn,
    xz,
    cs,
):
    znc = tl.clamp(zn, 1e-15, 1e30)
    eb = tl.exp(b)
    ebi = 1 / eb
    eb_sum = eb + ebi
    eb_dif = eb - ebi
    P = 0.5 * (eb_sum * (cs * lam / znc * xz) - eb_dif * (lam - 1))
    sq_p2_1 = tl.sqrt(P * P + 1)
    log_p_sq = tl.log(P + sq_p2_1)
    D = 2.0 * znc * log_p_sq
    ed = tl.exp(D)
    edi = 1 / ed
    ed_dif = ed - edi
    ed_sum = ed + edi
    num = 0.5 * ed_dif / cs
    return sq_p2_1, log_p_sq, eb_sum, eb_dif, ed_sum, num


@triton.autotune(
    configs=get_autotune_configs(),
    key=["K", "M"],
)
@triton.jit
def _poincare_fc_fwd_kernel(  # fused: fwd + project
    X_ptr,  # [B, K]
    ZN_ptr,  # [M]
    XZ_ptr,  # [B, M]
    B_ptr,  # [M]
    num_ptr,  # [B, M]  (kept for cache)
    den_ptr,  # [B]
    yn_ptr,  # [B]
    y_ptr,  # [B, M]
    py_ptr,  # [B, M]
    lam_ptr,  # [B]
    c,  # fp32
    cs,  # fp32
    max_norm,  # fp32
    K,
    M,
    stride_x_b,
    stride_xz_b,
    stride_num_b,
    stride_y_b,
    stride_py_b,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row = tl.program_id(0)

    # row offsets
    X_ptr += row * stride_x_b
    XZ_ptr += row * stride_xz_b
    num_ptr += row * stride_num_b
    y_ptr += row * stride_y_b
    py_ptr += row * stride_py_b
    lam_ptr += row
    den_ptr += row
    yn_ptr += row

    # 1) lambda(x)
    offs_k = tl.arange(0, BLOCK_K)
    norm_x_sq = 0.0
    for _ in range(0, K, BLOCK_K):
        mask_k = offs_k < K
        xk = tl.load(X_ptr + offs_k, mask=mask_k, other=0.0)
        norm_x_sq += tl.sum(xk * xk)
        offs_k += BLOCK_K
    lam = 2.0 / (1.0 - c * norm_x_sq)
    tl.store(lam_ptr, lam)

    # 2) compute num and accumulate denominator
    den_acc = 0.0
    for m in range(0, M, BLOCK_M):
        m_ids = tl.arange(0, BLOCK_M) + m
        mask_m = m_ids < M
        zn = tl.load(ZN_ptr + m_ids, mask=mask_m, other=1.0)
        xz = tl.load(XZ_ptr + m_ids, mask=mask_m, other=0.0)
        b = tl.load(B_ptr + m_ids, mask=mask_m, other=0.0)

        _1, _2, _3, _4, _5, num = single_block_fwd(b, lam, zn, xz, cs)
        den_acc += tl.sum(num * num)

        tl.store(num_ptr + m_ids, num, mask=mask_m)

    # 3) den and ||y||
    den = 1.0 + tl.sqrt(1.0 + c * den_acc)
    y_norm = tl.sqrt(den_acc / (den * den))
    y_norm = tl.clamp(y_norm, 1e-15, 1e15)

    tl.store(den_ptr, den)
    tl.store(yn_ptr, y_norm)

    # 4) project in-place using stored num
    scale = tl.where(y_norm < max_norm, 1.0, max_norm / y_norm)

    for m in range(0, M, BLOCK_M):
        m_ids = tl.arange(0, BLOCK_M) + m
        mask_m = m_ids < M
        num = tl.load(num_ptr + m_ids, mask=mask_m, other=0.0)
        y = num / den
        py = y * scale
        tl.store(y_ptr + m_ids, y, mask=mask_m)
        tl.store(py_ptr + m_ids, py, mask=mask_m)


@validate_tensors
def poincare_fc_project_fwd_triton(
    x: Tensor2D, z: Tensor2D, r: Tensor1D = None, c=1.0, eps: float = -1, return_cache: bool = False
):
    assert x.is_cuda and z.is_cuda
    B, K = x.shape
    K2, M = z.shape
    assert K == K2

    if not x.is_contiguous():
        x = x.contiguous()

    c_val = float(c) if not torch.is_tensor(c) else float(c.item())
    cs = math.sqrt(c_val)
    zn = z.norm(dim=0)  # [M]
    xz = x @ z  # [B, M]

    dtype, device = x.dtype, x.device
    num = torch.empty((B, M), dtype=dtype, device=device)
    den = torch.empty((B,), dtype=dtype, device=device)
    yn = torch.empty((B,), dtype=dtype, device=device)
    lam = torch.empty((B,), dtype=dtype, device=device)

    if r is None:
        b = torch.zeros((M,), device=device, dtype=dtype)
    else:
        b = (2 * cs * r).to(device=device, dtype=dtype)

    if eps < 0:
        eps = 4e-3 if x.dtype == torch.float32 else 1e-5

    max_norm = 1e15
    if c_val > 0:
        max_norm = (1 - eps) / ((c_val + 1e-15) ** 0.5)

    y = torch.empty_like(num)
    py = torch.empty_like(num)

    grid = (B,)
    _poincare_fc_fwd_kernel[grid](
        x,
        zn,
        xz,
        b,
        num,
        den,
        yn,
        y,
        py,
        lam,
        float(c_val),
        float(cs),
        float(max_norm),
        K,
        M,
        x.stride(0),
        xz.stride(0),
        num.stride(0),
        y.stride(0),
        py.stride(0),
    )

    if return_cache:
        return py, (y, x, z, xz, zn, b, lam, num, den, yn, max_norm, c_val, cs)
    return py


def poincare_fc_fwd_project_ref(
    x: torch.Tensor,
    z: torch.Tensor,
    bias: Optional[torch.Tensor],
    c: torch.Tensor,
    dim: int = -1,
    eps: float = -1.0,
    return_cache: bool = False,
) -> torch.Tensor:
    x = x.movedim(dim, -1)  # features last
    if bias is None:
        bias = torch.zeros(z.shape[1], dtype=x.dtype, device=x.device)

    cs = c.sqrt()
    lam = 2 / (1 - c * x.pow(2).sum(-1, keepdim=True))
    zn = z.norm(dim=0).clamp_min(1e-15)

    xz = torch.matmul(x, z)
    p = (cs * lam / zn) * xz

    b = 2 * cs * bias
    eb = torch.exp(b)
    ebi = 1 / eb
    p = 0.5 * ((eb + ebi) * p - (eb - ebi) * (lam - 1))

    d = 2 * zn * torch.log(p + torch.sqrt(p * p + 1))
    ed = torch.exp(d)
    num = (ed - 1 / ed) / (2 * cs)  # sinh(d)/cs
    den = 1 + torch.sqrt(1 + c * num.pow(2).sum(-1, keepdim=True))

    y = (num / den).movedim(-1, dim)

    # final projection
    if eps < 0:
        if y.dtype == torch.float32:
            eps = 4e-3
        else:
            eps = 1e-5

    max_norm = (1 - eps) / ((c + 1e-15) ** 0.5)
    max_norm_tensor = torch.where(c.gt(0), max_norm, c.new_full((), 1e15))

    yn = y.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    mn = torch.where(yn > max_norm_tensor, max_norm_tensor, yn)
    py = y / yn * mn

    if return_cache:
        return py, (
            y,
            x,
            z,
            xz,
            zn,
            b,
            lam.squeeze(),
            num,
            den.squeeze(),
            yn.squeeze(),
            max_norm.item(),
            c.item(),
            cs.item(),
        )

    return py
