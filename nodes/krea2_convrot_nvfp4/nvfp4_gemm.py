"""HSWQ-owned NVFP4 GEMM helpers (ConvRot NVFP4).

ComfyUI / comfy_kitchen do **not** ship ConvRot×NVFP4 load+forward.
The Linear hot path lives in ``nvfp4_forward._tc_forward_pooled`` →
``nvfp4_runtime.scaled_mm_nvfp4_pooled`` (raw ``_C.cublas_gemm_blockwise_fp4``
with pre-validated shapes; weight stays packed). Never torch native
``F.scaled_mm`` FP4 / registry dispatch — path A sticky-poisons SM120.

This module owns:
  - NVFP4 unpack / dequant (FP4 E2M1 + block scales)
  - one-shot weight bake (replace QT Parameter → dense float; free packed) —
    TC-failure fallback only, never the hot path
  - float GEMM ``a @ b.T`` (+ optional bias) for residual QT×QT addmm edges

Runtime patches under ``benchmark/krea2_nvfp4`` must use these entry points.
"""
from __future__ import annotations

from typing import Optional

# FP4 E2M1 decode LUT (same values as kitchen float_utils / eager dequant).
_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
_E2M1_LUT_CACHE: dict = {}


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def from_blocked(blocked_matrix, num_rows: int, num_cols: int):
    """Reverse cuBLAS 32×4×4 block-scale swizzle → (num_rows, num_cols)."""
    n_row_blocks = _ceil_div(num_rows, 128)
    n_col_blocks = _ceil_div(num_cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    step1 = blocked_matrix.reshape(-1, 32, 16)
    step2 = step1.reshape(-1, 32, 4, 4).transpose(1, 2)
    step3 = step2.reshape(n_row_blocks, n_col_blocks, 4, 32, 4)
    step4 = step3.reshape(n_row_blocks, n_col_blocks, 128, 4)
    step5 = step4.permute(0, 2, 1, 3)
    unblocked = step5.reshape(padded_rows, padded_cols)
    return unblocked[:num_rows, :num_cols]


def clear_cuda_sticky_error() -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
    except Exception:
        pass


def dequantize_nvfp4(
    qx,
    per_tensor_scale,
    block_scales,
    output_type=None,
    *,
    hi_first: bool = True,
):
    """Unpack NVFP4 uint8×2 + block scales → dense float tensor (HSWQ-owned)."""
    import torch

    if output_type is None:
        output_type = torch.bfloat16

    key = (str(qx.device), output_type)
    lut = _E2M1_LUT_CACHE.get(key)
    if lut is None:
        lut = torch.tensor(
            _E2M1_VALUES, device=qx.device, dtype=output_type
        ).unsqueeze(1)
        _E2M1_LUT_CACHE[key] = lut

    lo = qx & 0x0F
    hi = qx >> 4
    if hi_first:
        out = torch.stack([hi, lo], dim=-1).view(*qx.shape[:-1], -1)
    else:
        out = torch.stack([lo, hi], dim=-1).view(*qx.shape[:-1], -1)
    out = torch.nn.functional.embedding(out.int(), lut).squeeze(-1)

    orig_shape = out.shape
    block_size = 16
    out = out.reshape(orig_shape[0], -1, block_size)
    num_blocks_per_row = orig_shape[1] // block_size
    block_scales_unswizzled = from_blocked(
        block_scales, num_rows=orig_shape[0], num_cols=num_blocks_per_row
    )
    if not isinstance(per_tensor_scale, torch.Tensor):
        per_tensor_scale = torch.tensor(
            per_tensor_scale, device=qx.device, dtype=torch.float32
        )
    if per_tensor_scale.device != qx.device or per_tensor_scale.dtype != output_type:
        per_tensor_scale = per_tensor_scale.to(device=qx.device, dtype=output_type)
    total_scale = per_tensor_scale * block_scales_unswizzled.to(output_type)
    data_dequantized = out * total_scale.unsqueeze(-1)
    return data_dequantized.view(orig_shape).to(output_type)


def hswq_scaled_mm_nvfp4(
    a_qdata,
    w_qdata,
    *,
    tensor_scale_a,
    tensor_scale_b,
    block_scale_a,
    block_scale_b,
    bias=None,
    out_dtype=None,
    alpha: Optional[object] = None,
    orig_m: Optional[int] = None,
    orig_n: Optional[int] = None,
    out=None,
):
    """HSWQ ConvRot-NVFP4 GEMM: dequant both sides → ``a @ w.T`` (+ bias).

    Never calls ``comfy_kitchen`` ``scaled_mm_nvfp4`` / CUBLAS blockwise FP4.
    ``alpha`` is ignored (scales live inside dequant).
    """
    import torch

    _ = alpha
    if out_dtype is None:
        out_dtype = torch.bfloat16

    if isinstance(tensor_scale_a, torch.nn.Parameter):
        tensor_scale_a = tensor_scale_a.data
    if isinstance(tensor_scale_b, torch.nn.Parameter):
        tensor_scale_b = tensor_scale_b.data
    if isinstance(a_qdata, torch.nn.Parameter):
        a_qdata = a_qdata.data
    if isinstance(w_qdata, torch.nn.Parameter):
        w_qdata = w_qdata.data
    if isinstance(block_scale_a, torch.nn.Parameter):
        block_scale_a = block_scale_a.data
    if isinstance(block_scale_b, torch.nn.Parameter):
        block_scale_b = block_scale_b.data

    a_dq = dequantize_nvfp4(
        a_qdata, tensor_scale_a, block_scale_a, output_type=out_dtype
    )
    w_dq = dequantize_nvfp4(
        w_qdata, tensor_scale_b, block_scale_b, output_type=out_dtype
    )
    result = torch.mm(a_dq, w_dq.t())

    bias_arg = bias
    if bias is None or (isinstance(bias, torch.Tensor) and bias.numel() == 0):
        bias_arg = None
    elif isinstance(bias, torch.nn.Parameter):
        bias_arg = bias.data
    if bias_arg is not None:
        result = result + bias_arg.to(dtype=result.dtype, device=result.device)

    if orig_m is None:
        orig_m = int(a_qdata.shape[0])
    if orig_n is None:
        orig_n = int(w_qdata.shape[0])
    if result.shape[0] != orig_m or result.shape[1] != orig_n:
        result = result[:orig_m, :orig_n]

    if out is not None:
        if out.shape != result.shape or out.dtype != result.dtype or out.device != result.device:
            raise ValueError("out buffer shape/dtype/device mismatch")
        out.copy_(result)
        return out
    return result


def bake_nvfp4_weight_inplace(module, weight_qt, out_dtype):
    """One-shot NVFP4 → dense float bake; drop packed QT (no dual VRAM).

    Prior ``_hswq_nvfp4_w_dequant`` cache kept packed ``_qdata`` **and** a full
    BF16/FP16 matrix → peak VRAM ≈ BF16 (or worse) while still paying ConvRot +
    dequant cost. Bake replaces ``module.weight`` with a plain Parameter and
    releases the QuantizedTensor so only one residency remains.
    """
    import torch
    from comfy_kitchen.tensor.base import QuantizedTensor
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout

    cur = module.weight
    if isinstance(cur, torch.nn.Parameter):
        cur_data = cur.data
    else:
        cur_data = cur
    if not isinstance(cur_data, QuantizedTensor):
        # Already baked; align dtype for this forward if needed.
        if cur_data.dtype != out_dtype:
            return cur_data.to(dtype=out_dtype)
        return cur_data

    # Prefer layout-owned dequant (CUDA path when kitchen provides it); HSWQ
    # LUT unpack is the fallback if dequantize is unavailable.
    try:
        w_f = weight_qt.dequantize()
        if not isinstance(w_f, torch.Tensor):
            raise TypeError("dequantize did not return a Tensor")
        w_f = w_f.to(dtype=out_dtype)
    except Exception:
        w_qdata, scale_b, block_scale_b = TensorCoreNVFP4Layout.get_plain_tensors(
            weight_qt
        )
        orig_n = int(weight_qt._params.orig_shape[0])
        orig_k = int(weight_qt._params.orig_shape[1])
        w_f = dequantize_nvfp4(w_qdata, scale_b, block_scale_b, output_type=out_dtype)
        if w_f.shape[0] != orig_n or w_f.shape[1] != orig_k:
            w_f = w_f[:orig_n, :orig_k].contiguous()

    w_f = w_f.detach().contiguous()
    module.weight = torch.nn.Parameter(w_f, requires_grad=False)
    if hasattr(module, "_hswq_nvfp4_w_dequant"):
        try:
            delattr(module, "_hswq_nvfp4_w_dequant")
        except Exception:
            module._hswq_nvfp4_w_dequant = None
    return module.weight.data


def dequantize_weight_cached(module, weight_qt, out_dtype):
    """Backward-compatible alias → ``bake_nvfp4_weight_inplace`` (no dual cache)."""
    return bake_nvfp4_weight_inplace(module, weight_qt, out_dtype)
