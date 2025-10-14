import torch
import triton
import triton.language as tl
from hypll.kernels.utils import tanh, sech_squared


def get_autotune_configs():
    return [
        triton.Config({"BLOCK_M": 4}),
        triton.Config({"BLOCK_M": 8}),
        triton.Config({"BLOCK_M": 16}),
        triton.Config({"BLOCK_M": 32}),
        triton.Config({"BLOCK_M": 64}),
        triton.Config({"BLOCK_M": 128}),
        triton.Config({"BLOCK_M": 256}),
        triton.Config({"BLOCK_M": 512}),
        triton.Config({"BLOCK_M": 1024}),
    ]


@triton.autotune(configs=get_autotune_configs(), key=["B", "M"])
@triton.jit(do_not_specialize=range(9), do_not_specialize_on_alignment=range(12))
def _expmap0_bwd_kernel(
    dout_ptr: "*fp32",
    v_ptr: "*fp32",
    vn_ptr: "*fp32",
    cs: "fp32",
    maxnorm: "fp32",
    dv_ptr: "*fp32",
    v_stride_b: "int32",
    dout_stride_b: "int32",
    dv_stride_b: "int32",
    B: tl.constexpr,
    M: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # Each program handles one batch item
    pid_b = tl.program_id(0)

    # Batch pointers
    v_row_ptr = v_ptr + pid_b * v_stride_b
    dout_row_ptr = dout_ptr + pid_b * dout_stride_b
    dv_row_ptr = dv_ptr + pid_b * dv_stride_b

    # Load per-batch scalars
    vn = tl.load(vn_ptr + pid_b)
    vn_safe = tl.where(vn < 1e-15, 1e-15, vn)
    vncs = vn_safe * cs

    # Pass 1: compute sum(dout * v)
    acc = tl.zeros((), dtype=tl.float32)
    for start in tl.range(0, M, BLOCK_M):
        offs = start + tl.arange(0, BLOCK_M)
        mask = offs < M
        v_blk = tl.load(v_row_ptr + offs, mask=mask, other=0.0)
        dout_blk = tl.load(dout_row_ptr + offs, mask=mask, other=0.0)
        prod = (v_blk * dout_blk).to(tl.float32)
        acc += tl.sum(prod, axis=0)
    dout_v_sum = acc  # scalar

    t_vncs = tanh(vncs)
    dtanh = tl.where(t_vncs > maxnorm, 0.0, vncs * sech_squared(vncs))  # scalar

    # Pass 2: compute dv
    for start in tl.range(0, M, BLOCK_M):
        offs = start + tl.arange(0, BLOCK_M)
        mask = offs < M

        v_blk = tl.load(v_row_ptr + offs, mask=mask, other=0.0)
        dout_blk = tl.load(dout_row_ptr + offs, mask=mask, other=0.0)

        term1 = (t_vncs / vncs) * dout_blk
        # avoid division by zero handled via vn_safe above
        term2_num = (dtanh - t_vncs) * dout_v_sum * v_blk
        term2_den = (vncs * vncs) * vn_safe
        term2 = tl.where(vn < 1e-15, 0.0, (cs / vn_safe) * (term2_num / term2_den))

        dv_blk = term1 + term2
        tl.store(dv_row_ptr + offs, dv_blk, mask=mask)


def expmap0_bwd_triton(
    dout: torch.Tensor, v: torch.Tensor, vn: torch.Tensor, cs: float, maxnorm: float
) -> torch.Tensor:
    assert v.ndim == 2 and dout.shape == v.shape and vn.ndim == 1 and vn.shape[0] == v.shape[0]
    B, M = v.shape
    dv = torch.empty_like(v)

    grid = (B,)
    _expmap0_bwd_kernel[grid](
        dout,
        v,
        vn,
        cs,
        maxnorm,
        dv,
        v.stride(0),
        dout.stride(0),
        dv.stride(0),
        B,
        M,
    )
    return dv
