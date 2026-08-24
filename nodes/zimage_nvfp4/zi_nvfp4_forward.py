"""
Z Image hybrid ConvRot LoRA bake + NVFP4 Linear forward helpers (branch-only).

Owns the ZI delta that must not live in ``nodes/nvfp4`` (SDXL TC product):
  - per-kind LoRA bake counters / EVIDENCE (NVFP4 + INT8 protect)
  - ``_hswq_int8_convrot`` bake arm / Params.convrot clear
  - ``attach_nvfp4_linear_lora_bake`` ver >= 8 (hybrid)

Hadamard lives in ``zi_nvfp4_hadamard`` (Distorch-safe cache). Runtime stays under ``nodes/nvfp4``.
"""
from __future__ import annotations

import logging
import os

from .zi_nvfp4_hadamard import (
    build_hadamard,
    rotate_weight_linear,
    unrotate_weight_linear,
)
from ..nvfp4.nvfp4_runtime import (
    ensure_act_scale,
    clear_nvfp4_cudagraphs,
    nvfp4_quant_mm_cudagraph,
    quantize_nvfp4_act_pooled,
    rotate_last_dim_pooled,
    scaled_mm_nvfp4_pooled,
    _GRAPH_MAX_M,
)

logger = logging.getLogger(__name__)

# Counters for bench / diagnostics (reset per run if needed)
_TC_HITS = 0
_DEQUANT_FALLBACKS = 0
_TC_FLOPS = 0
# Per-kind totals (always incremented) + per-kind sample log caps.
# Shared max of 8 hid all int8_protect samples (nvfp4 filled the quota first).
_LORA_CONVERT_TOTAL = {"nvfp4": 0, "int8_protect": 0}
_LORA_SET_TOTAL = {"nvfp4": 0, "int8_protect": 0}
_LORA_CONVERT_LOGGED = {"nvfp4": 0, "int8_protect": 0}
_LORA_SET_LOGGED = {"nvfp4": 0, "int8_protect": 0}
_LORA_KIND_LOG_MAX = 4
# Bump when convert_weight / set_weight ConvRot LoRA bake changes.
# v2: also unrotate/re-rotate INT8 protect ConvRot (``_hswq_int8_convrot``).
# Hybrid ZI packs = ConvRot NVFP4 + ConvRot INT8 protect — both need bake basis.
# v3: bake-time fallback if load arm missed and Params.convrot still True on INT8 QT.
# v4: (reverted) bake-only with Params.convrot — WRONG: kitchen dequant already
#     unrotates when Params.convrot=True → double unrotate → LoRA dead.
# v5: Conv2d twin — arm flag + clear Params; after set_weight keep Params=False
#     (noise was requant restoring Params while parity still rotated).
# v6: per-kind LoRA bake counters + EVIDENCE log (int8_protect must be visible).
# v7: pass-delta EVIDENCE only (no stale OK spam on empty re-bake / VAE load).
# v8: peer NVFP4_LORA_BAKE_* verdict + sample_nvfp4_keys (same weight as INT8).
_NVFP4_LORA_BAKE_VER = 8


def reset_nvfp4_lora_log_counters() -> None:
    for d in (
        _LORA_CONVERT_TOTAL,
        _LORA_SET_TOTAL,
        _LORA_CONVERT_LOGGED,
        _LORA_SET_LOGGED,
    ):
        for k in d:
            d[k] = 0


def reset_nvfp4_forward_stats() -> None:
    global _TC_HITS, _DEQUANT_FALLBACKS, _TC_FLOPS
    _TC_HITS = 0
    _DEQUANT_FALLBACKS = 0
    _TC_FLOPS = 0


def _lora_bake_kind(module) -> str:
    if getattr(module, "_hswq_nvfp4_convrot", False):
        return "nvfp4"
    return "int8_protect"


def nvfp4_lora_bake_counters() -> dict:
    """Totals for convert unrotate / set re-rotate by ConvRot kind."""
    return {
        "convert_unrotate_nvfp4": int(_LORA_CONVERT_TOTAL.get("nvfp4", 0)),
        "convert_unrotate_int8_protect": int(_LORA_CONVERT_TOTAL.get("int8_protect", 0)),
        "set_rerotate_nvfp4": int(_LORA_SET_TOTAL.get("nvfp4", 0)),
        "set_rerotate_int8_protect": int(_LORA_SET_TOTAL.get("int8_protect", 0)),
    }


def snapshot_nvfp4_lora_bake_counters() -> dict:
    """Copy of totals for pass-delta EVIDENCE (before bake → after bake)."""
    return dict(nvfp4_lora_bake_counters())


def _counter_delta(before: dict | None, after: dict | None) -> dict:
    b = before or {}
    a = after or nvfp4_lora_bake_counters()
    keys = (
        "convert_unrotate_nvfp4",
        "convert_unrotate_int8_protect",
        "set_rerotate_nvfp4",
        "set_rerotate_int8_protect",
    )
    return {k: int(a.get(k, 0)) - int(b.get(k, 0)) for k in keys}


def _lora_bake_side_verdict(prefix: str, baked: int, convert_n: int, set_n: int) -> str:
    """Peer verdict for one ConvRot kind (NVFP4 or INT8 protect)."""
    match = convert_n == set_n == int(baked)
    if int(baked) > 0 and match and convert_n > 0:
        return f"{prefix}_OK"
    if int(baked) > 0 and not match:
        return f"{prefix}_MISMATCH"
    if int(baked) == 0 and convert_n == 0:
        return f"{prefix}_N/A"
    return f"{prefix}_MISSING"


def _fmt_sample_keys(label: str, keys: list | None) -> str:
    if not keys:
        return ""
    shown = ", ".join(str(k) for k in keys[:3])
    return f" {label}=[{shown}]"


def log_nvfp4_lora_bake_evidence(
    tag: str = "",
    *,
    before: dict | None = None,
    nvfp4_baked: int = 0,
    int8_baked: int = 0,
    sample_nvfp4_keys: list | None = None,
    sample_int8_keys: list | None = None,
    force: bool = False,
) -> str | None:
    """Emit pass-scoped EVIDENCE only when this bake pass actually ran hooks.

    NVFP4 and INT8 protect are peer sides (same verdict shape + key samples).
    Returns the message if emitted, else None (silent skip for empty re-bake).
    """
    after = nvfp4_lora_bake_counters()
    d = _counter_delta(before, after)
    i8c = d["convert_unrotate_int8_protect"]
    i8s = d["set_rerotate_int8_protect"]
    nvc = d["convert_unrotate_nvfp4"]
    nvs = d["set_rerotate_nvfp4"]
    this_pass_hooks = (i8c + i8s + nvc + nvs) > 0
    this_pass_layer = (int(nvfp4_baked) + int(int8_baked)) > 0
    if not force and not this_pass_hooks and not this_pass_layer:
        return None

    nv_verdict = _lora_bake_side_verdict(
        "NVFP4_LORA_BAKE", int(nvfp4_baked), nvc, nvs
    )
    i8_verdict = _lora_bake_side_verdict(
        "INT8_PROTECT_LORA_BAKE", int(int8_baked), i8c, i8s
    )

    suffix = f" ({tag})" if tag else ""
    nv_samples = _fmt_sample_keys("sample_nvfp4_keys", sample_nvfp4_keys)
    i8_samples = _fmt_sample_keys("sample_int8_keys", sample_int8_keys)
    msg = (
        f"[HSWQ ConvRot LoRA] EVIDENCE{suffix}: {nv_verdict} {i8_verdict} "
        f"this_pass | "
        f"nvfp4 convert_unrotate={nvc} set_rerotate={nvs} "
        f"nvfp4_baked={int(nvfp4_baked)}{nv_samples} | "
        f"int8_protect convert_unrotate={i8c} set_rerotate={i8s} "
        f"int8_baked={int(int8_baked)}{i8_samples} | "
        f"session_total nv_c/s="
        f"{after['convert_unrotate_nvfp4']}/"
        f"{after['set_rerotate_nvfp4']} "
        f"int8_c/s="
        f"{after['convert_unrotate_int8_protect']}/"
        f"{after['set_rerotate_int8_protect']}"
    )
    # Single emit path (caller may also _console — prefer logger+print once here).
    logger.info(msg)
    print(msg, flush=True)
    return msg


def _clear_int8_qt_params_convrot(module) -> bool:
    """Force Params.convrot=False on INT8 QT (must stay False after requant)."""
    import dataclasses

    try:
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return False
    w = getattr(module, "weight", None)
    qt = w if isinstance(w, QuantizedTensor) else getattr(w, "data", None)
    if qt is None or not isinstance(qt, QuantizedTensor):
        return False
    params = getattr(qt, "_params", None)
    if params is None or not bool(getattr(params, "convrot", False)):
        return False
    new_params = dataclasses.replace(params, convrot=False)
    try:
        object.__setattr__(qt, "_params", new_params)
        return True
    except Exception:
        pass
    try:
        qt._params = new_params
        return True
    except Exception:
        return False


def _linear_convrot_lora_groupsize(module) -> int | None:
    """Groupsize for offline ConvRot Linear LoRA bake, or None if not ConvRot.

    Hybrid Z Image packs:
      - NVFP4: ``_hswq_nvfp4_convrot`` (Params cleared; parity rotates).
      - INT8 protect: ``_hswq_int8_convrot`` (Params cleared; parity rotates).
        Kitchen dequant with Params.convrot=True already unrotates — bake must
        see Params=False so convert unrotates rotated-basis float once.
    """
    if getattr(module, "_hswq_nvfp4_convrot", False):
        return int(getattr(module, "_hswq_nvfp4_convrot_groupsize", 256) or 256)
    if getattr(module, "_hswq_int8_convrot", False):
        return int(getattr(module, "_hswq_int8_convrot_groupsize", 256) or 256)
    # Late arm if load missed: Params.convrot still True on INT8 QT.
    try:
        from comfy.quant_ops import QuantizedTensor

        w = getattr(module, "weight", None)
        qt = w if isinstance(w, QuantizedTensor) else getattr(w, "data", None)
        if qt is None or not isinstance(qt, QuantizedTensor):
            return None
        layout_cls = getattr(qt, "_layout_cls", None) or ""
        if isinstance(layout_cls, type):
            layout_cls = getattr(layout_cls, "__name__", "") or ""
        if str(layout_cls) != "TensorWiseINT8Layout":
            return None
        params = getattr(qt, "_params", None)
        if params is None or not bool(getattr(params, "convrot", False)):
            return None
        gs = int(getattr(params, "convrot_groupsize", 256) or 256)
        module._hswq_int8_convrot = True
        module._hswq_int8_convrot_groupsize = gs
        _clear_int8_qt_params_convrot(module)
        return gs
    except Exception:
        return None


def nvfp4_forward_stats() -> dict:
    return {
        "scaled_mm_hits": _TC_HITS,
        "dequant_fallbacks": _DEQUANT_FALLBACKS,
        "tc_flops": _TC_FLOPS,
    }


def _slice_nvfp4_mm_out(result, orig_m: int, orig_n: int):
    if result.shape[0] != orig_m or result.shape[1] != orig_n:
        return result[:orig_m, :orig_n]
    return result


def scaled_mm_nvfp4_linear(input_qt, weight_qt, bias):
    """Kitchen / tritant NVFP4 linear (QT path; used as fallback)."""
    global _TC_HITS, _DEQUANT_FALLBACKS
    import torch
    import torch.nn.functional as F
    import comfy_kitchen as ck
    from comfy_kitchen.tensor.base import QuantizedTensor
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout

    if not (
        isinstance(input_qt, QuantizedTensor)
        and isinstance(weight_qt, QuantizedTensor)
        and input_qt._layout_cls == "TensorCoreNVFP4Layout"
        and weight_qt._layout_cls == "TensorCoreNVFP4Layout"
    ):
        _DEQUANT_FALLBACKS += 1
        return F.linear(input_qt, weight_qt, bias)
    if input_qt._qdata.dim() != 2:
        _DEQUANT_FALLBACKS += 1
        return F.linear(input_qt, weight_qt, bias)
    if getattr(input_qt._params, "transposed", False) or getattr(
        weight_qt._params, "transposed", False
    ):
        _DEQUANT_FALLBACKS += 1
        return F.linear(input_qt, weight_qt, bias)

    if isinstance(bias, QuantizedTensor):
        bias = bias.dequantize()

    a_qdata, scale_a, block_scale_a = TensorCoreNVFP4Layout.get_plain_tensors(input_qt)
    w_qdata, scale_b, block_scale_b = TensorCoreNVFP4Layout.get_plain_tensors(weight_qt)
    out_dtype = input_qt._params.orig_dtype
    try:
        result = ck.scaled_mm_nvfp4(
            a_qdata,
            w_qdata,
            tensor_scale_a=scale_a,
            tensor_scale_b=scale_b,
            block_scale_a=block_scale_a,
            block_scale_b=block_scale_b,
            bias=bias,
            out_dtype=out_dtype,
        )
        orig_m = input_qt._params.orig_shape[0]
        orig_n = weight_qt._params.orig_shape[0]  # (out, in)
        _TC_HITS += 1
        return _slice_nvfp4_mm_out(result, orig_m, orig_n)
    except (RuntimeError, TypeError) as e:
        logger.warning("[HSWQ NVFP4] scaled_mm_nvfp4 failed: %s — F.linear dequant", e)
        _DEQUANT_FALLBACKS += 1
        return F.linear(input_qt, weight_qt, bias)


def _plain_weight_cached(module, weight_qt):
    """Cache get_plain_tensors on the module (weight QT identity stable after load)."""
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout

    cached = getattr(module, "_hswq_nvfp4_w_plain", None)
    if cached is not None and cached[0] is weight_qt._qdata:
        return cached[1], cached[2], cached[3], cached[4]
    w_qdata, scale_b, block_scale_b = TensorCoreNVFP4Layout.get_plain_tensors(weight_qt)
    orig_n = int(weight_qt._params.orig_shape[0])
    module._hswq_nvfp4_w_plain = (
        weight_qt._qdata,
        w_qdata,
        scale_b,
        block_scale_b,
        orig_n,
    )
    return w_qdata, scale_b, block_scale_b, orig_n


def _tc_forward_pooled(module, input_2d, weight_qt, bias, act_scale, out_dtype):
    """Act float → pooled NVFP4 quant → pooled cuBLAS mm (no QT alloc).

    Prefers CUDA Graph (quantize+mm) after first capture per shape/weight; falls
    back to eager pooled kernels if capture/replay fails.
    """
    global _TC_HITS, _DEQUANT_FALLBACKS, _TC_FLOPS
    import torch
    from comfy_kitchen.tensor.base import QuantizedTensor
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout

    if not (
        isinstance(weight_qt, QuantizedTensor)
        and weight_qt._layout_cls == "TensorCoreNVFP4Layout"
    ):
        _DEQUANT_FALLBACKS += 1
        return None
    if getattr(weight_qt._params, "transposed", False):
        _DEQUANT_FALLBACKS += 1
        return None

    if isinstance(bias, QuantizedTensor):
        bias = bias.dequantize()

    orig_m, orig_k = int(input_2d.shape[0]), int(input_2d.shape[1])
    needs_padding = TensorCoreNVFP4Layout.get_padded_shape((orig_m, orig_k)) != (
        orig_m,
        orig_k,
    )

    scale_a = ensure_act_scale(input_2d, act_scale)
    try:
        w_qdata, scale_b, block_scale_b, orig_n = _plain_weight_cached(module, weight_qt)
        flops = orig_m * orig_k * orig_n * 2

        # Calib input_scale and placeholder ones are static — always cache
        # alpha. Recomputing scale_a*scale_b every Linear (~18k/sample) was
        # pure waste on FULL ConvRot (every layer has input_scale).
        cached_alpha = getattr(module, "_hswq_nvfp4_alpha", None)
        if cached_alpha is None:
            alpha = scale_a * scale_b
            if alpha.dtype != torch.float32:
                alpha = alpha.to(dtype=torch.float32)
            if alpha.dim() == 0:
                alpha = alpha.reshape(1)
            module._hswq_nvfp4_alpha = alpha
        else:
            alpha = cached_alpha

        # CUDA Graph is OFF by default: shape-shared replay copies full weight
        # every call and was slower than eager (13.05s vs ~11.8s). Opt-in:
        # HSWQ_NVFP4_CUDAGRAPH=1
        use_cg = (
            os.environ.get("HSWQ_NVFP4_CUDAGRAPH", "").strip() == "1"
            and orig_m <= _GRAPH_MAX_M
            and not getattr(module, "_hswq_nvfp4_no_cudagraph", False)
        )
        if use_cg:
            try:
                result = nvfp4_quant_mm_cudagraph(
                    input_2d,
                    w_qdata=w_qdata,
                    weight_scale=scale_b,
                    block_scale_w=block_scale_b,
                    scale_a=scale_a,
                    bias=bias,
                    out_dtype=out_dtype,
                    alpha=alpha,
                    pad_16x=needs_padding,
                    orig_n=orig_n,
                )
                _TC_HITS += 1
                _TC_FLOPS += flops
                return result
            except torch.cuda.OutOfMemoryError:
                clear_nvfp4_cudagraphs()
                torch.cuda.empty_cache()
                logger.warning(
                    "[HSWQ NVFP4] CUDA Graph OOM — cache cleared; eager pooled"
                )
            except (RuntimeError, TypeError, ValueError) as e:
                if "out of memory" in str(e).lower():
                    clear_nvfp4_cudagraphs()
                    torch.cuda.empty_cache()
                    logger.warning(
                        "[HSWQ NVFP4] CUDA Graph OOM (%s); eager pooled", e
                    )
                else:
                    module._hswq_nvfp4_no_cudagraph = True
                    logger.warning(
                        "[HSWQ NVFP4] CUDA Graph disabled for module (%s); eager pooled",
                        e,
                    )

        a_qdata, block_scale_a, _pr, _pc = quantize_nvfp4_act_pooled(
            input_2d, scale_a, pad_16x=needs_padding
        )
        result = scaled_mm_nvfp4_pooled(
            a_qdata,
            w_qdata,
            tensor_scale_a=scale_a,
            tensor_scale_b=scale_b,
            block_scale_a=block_scale_a,
            block_scale_b=block_scale_b,
            bias=bias,
            out_dtype=out_dtype,
            alpha=alpha,
            orig_m=orig_m,
            orig_n=orig_n,
        )
        _TC_HITS += 1
        _TC_FLOPS += flops
        return result
    except (RuntimeError, TypeError, ValueError) as e:
        logger.warning("[HSWQ NVFP4] pooled TC path failed: %s", e)
        _DEQUANT_FALLBACKS += 1
        return None


def make_nvfp4_linear_forward(stock_forward):
    """
    Return a Linear.forward replacement.

    For modules flagged ``_hswq_nvfp4`` (set at load), run the HSWQ TC path.
    All other layers keep stock_forward unchanged.
    """
    import torch
    import comfy.model_management
    from comfy.ops import cast_bias_weight, run_every_op, uncast_bias_weight

    def forward_nvfp4(self, input, *args, **kwargs):
        if not getattr(self, "_hswq_nvfp4", False) or getattr(self, "_full_precision_mm", False):
            return stock_forward(self, input, *args, **kwargs)

        # Training / forced cast: fall back to stock
        if input.requires_grad or getattr(self, "comfy_force_cast_weights", False):
            return stock_forward(self, input, *args, **kwargs)
        # LoRA weight_function: stay on HSWQ path (act ConvRot + cast_bias_weight
        # with want_requant). Stock forward would skip act rotate → ConvRot break.

        run_every_op()
        input_shape = input.shape
        compute_dtype = input.dtype

        # 1) Reshape ≥3D → 2D first (same last-dim math; cheaper than rotating ND)
        reshaped_nd = input.ndim >= 3
        input_2d = input.reshape(-1, input_shape[-1]) if reshaped_nd else input
        if input_2d.ndim != 2:
            return stock_forward(self, input, *args, **kwargs)

        # 2) FULL ConvRot: dense Hadamard GEMM (gs=256 butterfly is ~15x slower)
        if getattr(self, "_hswq_nvfp4_convrot", False):
            gs = int(getattr(self, "_hswq_nvfp4_convrot_groupsize", 256) or 256)
            h = getattr(self, "_hswq_nvfp4_H", None)
            if h is None or h.device != input_2d.device or h.dtype != input_2d.dtype:
                h = build_hadamard(gs, device=input_2d.device, dtype=input_2d.dtype)
                self._hswq_nvfp4_H = h
            input_2d = rotate_last_dim_pooled(input_2d, h, gs)

        # 3) Weight / bias: skip cast_bias_weight when already on-device QT
        #    (cast+sync every Linear was a major share of NVFP4 > FP16 wall time).
        #    Always cast when LoRA weight/bias_function present (need bake apply).
        offload_stream = None
        weight = self.weight
        if isinstance(weight, torch.nn.Parameter):
            weight = weight.data
        bias = self.bias.data if self.bias is not None else None
        has_wf = len(getattr(self, "weight_function", []) or []) or len(
            getattr(self, "bias_function", []) or []
        )
        need_cast = weight.device != input_2d.device or (
            bias is not None and bias.device != input_2d.device
        )
        if has_wf or need_cast or hasattr(self, "_v"):
            weight, bias, offload_stream = cast_bias_weight(
                self,
                input_2d,
                offloadable=True,
                compute_dtype=compute_dtype,
                want_requant=True,
            )

        scale = getattr(self, "input_scale", None)
        if scale is not None:
            if isinstance(scale, torch.nn.Parameter):
                scale = scale.data
            if scale.device != input.device:
                scale = comfy.model_management.cast_to_device(scale, input.device, None)

        layout = getattr(self, "layout_type", None)
        if layout is None:
            if offload_stream is not None:
                uncast_bias_weight(self, weight, bias, offload_stream)
            return stock_forward(self, input, *args, **kwargs)

        # 4) Pooled Tensor Core path (no QuantizedTensor.from_float alloc)
        out_2d = _tc_forward_pooled(
            self, input_2d, weight, bias, scale, compute_dtype
        )
        if out_2d is None:
            # Fallback: stock QT path
            from comfy.quant_ops import QuantizedTensor

            q_input = QuantizedTensor.from_float(input_2d, layout, scale=scale)
            out_2d = scaled_mm_nvfp4_linear(q_input, weight, bias)

        # 5) Restore rank with logical out_features (never QT storage shape[0])
        if reshaped_nd:
            out = out_2d.reshape((*input_shape[:-1], int(self.out_features)))
        else:
            out = out_2d

        if offload_stream is not None:
            uncast_bias_weight(self, weight, bias, offload_stream)
        return out

    forward_nvfp4._hswq_nvfp4_full_forward = True  # type: ignore[attr-defined]
    return forward_nvfp4


def make_nvfp4_linear_convert_weight(stock_convert_weight):
    """Wrap Linear.convert_weight: dequant then unrotate ConvRot weights for LoRA bake.

    Handles ConvRot NVFP4 **and** ConvRot INT8 protect (hybrid Z Image packs).
    Clear INT8 Params.convrot **before** stock dequant — kitchen already
    unrotates when Params.convrot=True (would double-unrotate with bake).
    """
    import torch
    from comfy.quant_ops import QuantizedTensor

    def convert_weight(self, weight, inplace=False, **kwargs):
        # Arm / clear Params before dequant (Conv2d twin).
        gs = _linear_convrot_lora_groupsize(self)
        if callable(stock_convert_weight):
            out = stock_convert_weight(self, weight, inplace=inplace, **kwargs)
        elif isinstance(weight, QuantizedTensor):
            out = weight.dequantize()
        else:
            out = weight
        if gs is None:
            gs = _linear_convrot_lora_groupsize(self)
        if gs is not None and out is not None and getattr(out, "ndim", 0) == 2:
            h = build_hadamard(gs, device="cpu", dtype=torch.float32)
            out = unrotate_weight_linear(out, h, gs)
            kind = _lora_bake_kind(self)
            _LORA_CONVERT_TOTAL[kind] = int(_LORA_CONVERT_TOTAL.get(kind, 0)) + 1
            if int(_LORA_CONVERT_LOGGED.get(kind, 0)) < _LORA_KIND_LOG_MAX:
                _LORA_CONVERT_LOGGED[kind] = int(_LORA_CONVERT_LOGGED.get(kind, 0)) + 1
                logger.info(
                    "[HSWQ ConvRot LoRA] Linear.convert_weight #%s (%s): unrotate "
                    "gs=%s in=%s/%s -> out=%s/%s",
                    _LORA_CONVERT_TOTAL[kind],
                    kind,
                    gs,
                    type(weight).__name__,
                    getattr(weight, "dtype", None),
                    type(out).__name__,
                    getattr(out, "dtype", None),
                )
        return out

    convert_weight._hswq_nvfp4_lora_bake_ver = _NVFP4_LORA_BAKE_VER  # type: ignore[attr-defined]
    convert_weight._hswq_nvfp4_lora_bake_stock = stock_convert_weight  # type: ignore[attr-defined]
    return convert_weight


def make_nvfp4_linear_set_weight(stock_set_weight):
    """Wrap Linear.set_weight: re-rotate ConvRot float weights before requant.

    Handles ConvRot NVFP4 **and** ConvRot INT8 protect (hybrid Z Image packs).
    After INT8 requant, force Params.convrot=False (parity rotates acts;
    kitchen must not also rotate).
    """
    import torch

    def set_weight(
        self,
        weight,
        inplace_update=False,
        seed=None,
        return_weight=False,
        **kwargs,
    ):
        gs = _linear_convrot_lora_groupsize(self)
        if gs is not None and getattr(weight, "ndim", 0) == 2:
            h = build_hadamard(gs, device="cpu", dtype=torch.float32)
            weight = rotate_weight_linear(weight, h, gs)
            kind = _lora_bake_kind(self)
            _LORA_SET_TOTAL[kind] = int(_LORA_SET_TOTAL.get(kind, 0)) + 1
            if int(_LORA_SET_LOGGED.get(kind, 0)) < _LORA_KIND_LOG_MAX:
                _LORA_SET_LOGGED[kind] = int(_LORA_SET_LOGGED.get(kind, 0)) + 1
                logger.info(
                    "[HSWQ ConvRot LoRA] Linear.set_weight #%s (%s): re-rotate "
                    "gs=%s shape=%s layout=%s",
                    _LORA_SET_TOTAL[kind],
                    kind,
                    gs,
                    tuple(weight.shape) if hasattr(weight, "shape") else "?",
                    getattr(self, "layout_type", None),
                )
        out = stock_set_weight(
            self,
            weight,
            inplace_update=inplace_update,
            seed=seed,
            return_weight=return_weight,
            **kwargs,
        )
        # Requant may restore Params.convrot — keep cleared for INT8 protect.
        if getattr(self, "_hswq_int8_convrot", False):
            _clear_int8_qt_params_convrot(self)
        return out

    set_weight._hswq_nvfp4_lora_bake_ver = _NVFP4_LORA_BAKE_VER  # type: ignore[attr-defined]
    set_weight._hswq_nvfp4_lora_bake_stock = stock_set_weight  # type: ignore[attr-defined]
    return set_weight


def _peel_lora_bake_wrap(fn):
    """Unwrap nested HSWQ convert/set wraps to true stock.

    After #2 split, ``nodes/nvfp4`` (3.3.0) may attach VER=1 first; ZI must not
    wrap that as stock (double unrotate / re-rotate → dead LoRA). Same as
    3.3.4 single-module attach: one hybrid wrap over stock only.
    """
    cur = fn
    for _ in range(8):
        if not callable(cur):
            return cur
        if int(getattr(cur, "_hswq_nvfp4_lora_bake_ver", 0) or 0) <= 0:
            return cur
        stock = getattr(cur, "_hswq_nvfp4_lora_bake_stock", None)
        if stock is not None and stock is not cur:
            cur = stock
            continue
        closure = getattr(cur, "__closure__", None)
        code = getattr(cur, "__code__", None)
        if closure is None or code is None:
            return cur
        names = code.co_freevars
        nxt = None
        for i, name in enumerate(names):
            if name in ("stock_convert_weight", "stock_set_weight"):
                nxt = closure[i].cell_contents
                break
        if nxt is None or nxt is cur:
            return cur
        cur = nxt
    return cur


def attach_nvfp4_linear_lora_bake(Lin) -> bool:
    """Ensure MixedPrecision Linear has hybrid ConvRot LoRA wraps (one layer).

    Peels any prior HSWQ bake wrap (e.g. SDXL ``nodes/nvfp4`` VER=1) so ZI
    hybrid VER never nests — nesting double-unrotates NVFP4 and kills LoRA.
    """
    applied = False
    cvt = getattr(Lin, "convert_weight", None)
    if callable(cvt):
        ver = int(getattr(cvt, "_hswq_nvfp4_lora_bake_ver", 0) or 0)
        if ver != _NVFP4_LORA_BAKE_VER:
            stock = _peel_lora_bake_wrap(cvt) if ver > 0 else cvt
            Lin.convert_weight = make_nvfp4_linear_convert_weight(stock)
            applied = True
    sw = getattr(Lin, "set_weight", None)
    if callable(sw):
        ver = int(getattr(sw, "_hswq_nvfp4_lora_bake_ver", 0) or 0)
        if ver != _NVFP4_LORA_BAKE_VER:
            stock = _peel_lora_bake_wrap(sw) if ver > 0 else sw
            Lin.set_weight = make_nvfp4_linear_set_weight(stock)
            applied = True
    return applied
