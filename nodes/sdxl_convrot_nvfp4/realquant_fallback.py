# Case C real-quant fallback: NVFP4 -> INT8 one-way requantization
# (bit-width preserving relative to INT8; applies ONLY when the FP4 TC path
# refuses a layer and the stock fallback would bake to dense float)
#
# ABSOLUTE CONSTRAINT (Owner): LoRA bake must not break.
# Verified against loader source (see md/2026-10-01_case_c_lora_safety_review.md):
# - rank-decomposed residual path (_add_krea2_lora_residual) adds AFTER forward,
#   basis-invariant -> survives weight-format swap untouched
# - requantize bake path (DoRA/mid/strength!=1) goes dequantize() (layout-
#   agnostic) -> unrotate -> re-rotate -> requantize_from_float (base QT impl
#   calls layout quantize; TensorWiseINT8Layout.quantize exists)
# - therefore: swap module.weight NVFP4-QT -> INT8-QT, touch NOTHING else
#   (no patcher/backup/patches/residual attrs), one-way, tagged, once.
import os

import torch
import torch.nn as nn

_ENV = "HSWQ_REALQUANT_FALLBACK"


def realquant_fallback_enabled() -> bool:
    return os.environ.get(_ENV, "").strip().lower() in ("1", "true", "on", "enable")


def _int8_qt_from_float(w_f32: torch.Tensor, QuantizedTensor, layout_cls):
    """Quantize a float (out, in) matrix to a TensorWiseINT8Layout QT.

    Scale convention matches comfy_quant int8_tensorwise packs: per-tensor
    scale = max|w| / 127 (+eps), int8 = round-half-away(x / scale).
    """
    scale = (w_f32.abs().amax() / 127.0).clamp_min(1e-12).reshape(1)
    x = w_f32 / scale
    x_int8 = (torch.sign(x) * torch.floor(torch.abs(x) + 0.5)).clamp(-127, 127).to(torch.int8)
    # TensorCoreLayout classes take (qdata, layout_cls_str, params); params are
    # layout-specific dataclasses. We mirror how comfy ops construct them via
    # layout.quantize() to avoid param-shape drift across comfy versions.
    qt = layout_cls.quantize(w_f32.to(torch.float32))
    return qt


def nvfp4_to_int8_weight(module, weight_qt):
    """One-way NVFP4 QT -> INT8 QT conversion for a Linear module.

    Returns the new QT, or None when conversion is not possible (caller keeps
    the stock bake fallback). Touches ONLY module.weight.
    """
    try:
        from comfy.quant_ops import QuantizedTensor
        from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
    except ImportError:
        return None

    # dequant via layout (CUDA path when available), then INT8 quantize via
    # the int8 layout's own quantize() (version-safe construction).
    try:
        w_f = weight_qt.dequantize()
        if not isinstance(w_f, torch.Tensor):
            return None
        w_f = w_f.to(torch.float32)
        qt = _int8_qt_from_float(w_f, QuantizedTensor, TensorWiseINT8Layout)
        return qt
    except Exception:
        return None


def apply_realquant_fallback(module, weight_qt) -> bool:
    """Replace module.weight (NVFP4 QT) with an INT8 QT. One-way, once.

    Returns True when the swap happened. Absolutely no other state is touched:
    - _hswq_krea2_lora_res / _hswq_krea2_lora_res_gpu kept (residual add is
      basis-invariant post-forward)
    - _hswq_nvfp4_convrot* flags kept (LoRA convert/set_weight wraps rely on
      them for unrotate/re-rotate; layout-agnostic dequant keeps them working)
    - input_scale kept (state_dict stability)
    - NOT registered in _hswq_int8_baked_keys (this is a format swap, not a
      LoRA bake; patches_uuid invalidation semantics stay stock)
    """
    if getattr(module, "_hswq_realquant_backend", None) == "int8_fallback":
        return False  # already converted; never double-convert
    qt = nvfp4_to_int8_weight(module, weight_qt)
    if qt is None:
        return False
    module.weight = nn.Parameter(qt, requires_grad=False)
    module._hswq_realquant_backend = "int8_fallback"
    return True


def patched_tc_forward_pooled(stock_tc_forward_pooled):
    """Wrap _tc_forward_pooled: on failure (None), apply Case-C instead of bake.

    The stock fallback in nvfp4_forward.forward_nvfp4 (bake + F.linear) runs
    when this returns None. By converting to INT8 first and succeeding, the
    downstream stock fallback never triggers the bake... EXCEPT the INT8 path
    itself needs a linear op: we return the int8 QT path through the module's
    normal mixed-precision linear (comfy ops) which supports TensorWiseINT8
    via kitchen. Simplest correct integration: convert weight, then let the
    caller run the INT8 linear (int8_linear) itself.
    """

    def wrapper(module, input_2d, weight_qt, bias, act_scale, out_dtype):
        out = stock_tc_forward_pooled(module, input_2d, weight_qt, bias, act_scale, out_dtype)
        if out is not None:
            return out
        if not realquant_fallback_enabled():
            return None
        # One-way conversion; if it succeeds, signal the caller by returning
        # a sentinel the caller understands: we instead perform the INT8
        # linear here to keep the fallback self-contained.
        if apply_realquant_fallback(module, weight_qt):
            w = module.weight
            if isinstance(w, torch.nn.Parameter):
                w = w.data
            b = bias
            try:
                b = b.dequantize() if hasattr(bias, "dequantize") else bias
            except Exception:
                b = bias
            try:
                import comfy.ops as _ops
                # kitchen int8 linear handles TensorWiseINT8Layout QT weights
                out = torch.nn.functional.linear(
                    input_2d,
                    w.dequantize() if hasattr(w, "dequantize") else w,
                    b,
                )
                # NOTE: this is the same math as the stock bake fallback but the
                # weight stays INT8-resident; per-call dequant is only the tensor
                # we hand to F.linear. (Full int8_linear GEMM integration is a
                # follow-up once quantize-scale parity is validated on GPU.)
                return out
            except Exception:
                return None
        return None

    return wrapper
