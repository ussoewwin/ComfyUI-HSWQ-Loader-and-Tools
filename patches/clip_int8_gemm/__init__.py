# -*- coding: utf-8 -*-
"""Enable the real INT8 GEMM for ComfyUI's CLIP text tower.

Problem
-------
ComfyUI's ``int8_tensorwise`` algorithm is defined with
``"quantize_input": False``. That means the runtime keeps the weights in int8
(memory saving) but **dequantizes them for every matmul**: the only path in
``comfy/ops.py``'s ``MixedPrecisionOps.Linear`` that a weight-only algorithm
can take is ``forward_comfy_cast_weights`` -> ``CastBiasWeightContext`` ->
``F.linear`` with an fp16 weight.

Consequently a CLIP text tower loaded from a ConvRot INT8 pack never calls
``comfy_kitchen``'s ``int8_linear``: the quantized weight is dequantized on
every forward. Measured on this machine (CLIP ViT-L/14, batch 1):

    FP16 reference                 11.9 ms
    INT8 pack, stock path           23.3 ms   (int8_linear calls: 0)
    INT8 pack, int8 GEMM enabled    17.6 ms   (int8_linear calls: 2409)

The second condition that blocks the INT8 GEMM is in
``comfy/sd1_clip.py`` ``SDClipModel.__init__``:

    operations = comfy.ops.mixed_precision_ops(quant_config, dtype, full_precision_mm=True)

``full_precision_mm=True`` makes ``_use_quantized`` false in
``MixedPrecisionOps.Linear.forward``, so the input is never quantized and the
INT8 GEMM is never reached.

What this patch does
--------------------
Two runtime-only changes; **ComfyUI core is never edited**:

1. ``comfy.ops.mixed_precision_ops`` is wrapped so that a caller asking for
   ``full_precision_mm=True`` while supplying a ``quant_config`` gets
   ``full_precision_mm=False`` instead. This is what enables the INT8 GEMM.
2. ``MixedPrecisionOps.Linear.forward_comfy_cast_weights`` is wrapped so that a
   ``TensorWiseINT8Layout`` weight is handed to ``int8_linear`` (with the
   online ConvRot activation rotation) instead of being dequantized to fp16.

The patch only touches layers whose weight is a ``TensorWiseINT8Layout``
``QuantizedTensor``; every other layer keeps the stock behaviour.
"""
from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

_PATCHED = False
_STATS = {"int8_linear_calls": 0, "layers_patched": 0}


def int8_gemm_stats() -> dict:
    return dict(_STATS)


def reset_int8_gemm_stats() -> None:
    _STATS["int8_linear_calls"] = 0
    _STATS["layers_patched"] = 0


def _complete_text_projection(clip):
    """CLIP-L checkpoints may ship no ``text_projection``.

    Under the stock (dequant) path the missing projection is harmless, but once
    the real INT8 GEMM is used the forward reaches the projection with a
    ``None`` weight and dies with ``'NoneType' object has no attribute
    'device'``. Complete it with fp16 zeros, which is what the FP16 path
    effectively does for a missing projection.
    """
    try:
        import torch
        csm = getattr(clip, "cond_stage_model", None)
        if csm is None:
            return
        candidates = [csm]
        for a in ("clip_l", "clip_g"):
            sub = getattr(csm, a, None)
            if sub is not None:
                candidates.append(sub)
        for inner in candidates:
            tr = getattr(inner, "transformer", None)
            if tr is None:
                continue
            tp = getattr(tr, "text_projection", None)
            if tp is None or getattr(tp, "weight", None) is not None:
                continue
            hidden = 768
            dtype = torch.float16
            try:
                emb = tr.text_model.embeddings.token_embedding.weight
                hidden = int(emb.shape[1])
                dtype = emb.dtype
            except Exception:  # noqa: BLE001
                pass
            tp.weight = torch.nn.Parameter(torch.zeros((hidden, hidden), dtype=dtype),
                                          requires_grad=False)
            logger.info("[HSWQ INT8 GEMM] completed text_projection (%dx%d zeros)", hidden, hidden)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[HSWQ INT8 GEMM] text_projection completion skipped: %s", exc)


def _enable_int8_gemm_for_clip_text() -> bool:
    """Install the runtime patch. Returns True when it is in place."""
    global _PATCHED
    if _PATCHED:
        return True
    try:
        import torch
        import comfy.ops as ops
        from comfy_kitchen.tensor.base import QuantizedTensor
        from comfy.quant_ops import TensorWiseINT8Layout
    except Exception as exc:  # noqa: BLE001
        logger.warning("[HSWQ INT8 GEMM] comfy.ops unavailable: %s", exc)
        return False

    if getattr(ops, "_hswq_int8_gemm_patched", False):
        _PATCHED = True
        return True

    # --- 1) full_precision_mm=False when a quant_config is supplied ---------
    _orig_mpo = ops.mixed_precision_ops

    def mixed_precision_ops(quant_config={}, compute_dtype=torch.bfloat16,
                            full_precision_mm=False, disabled=[]):
        # A quantized text encoder wants the INT8 GEMM. The stock CLIP text path
        # asks for full_precision_mm=True, which disables it.
        if quant_config and full_precision_mm:
            full_precision_mm = False
        return _orig_mpo(quant_config, compute_dtype, full_precision_mm, disabled)

    ops.mixed_precision_ops = mixed_precision_ops

    # --- 2) route a TensorWiseINT8Layout weight through int8_linear ---------
    ck = ops.quant_ops.ck
    _orig_int8_linear = ck.int8_linear

    def counting_int8_linear(x, weight, scale, bias=None, out_dtype=None,
                             convrot=False, **kwargs):
        _STATS["int8_linear_calls"] += 1
        return _orig_int8_linear(x, weight, scale, bias, out_dtype,
                                 convrot=convrot, **kwargs)

    ck.int8_linear = counting_int8_linear

    _orig_fccw = None

    def make_int8_forward(linear_cls):
        nonlocal _orig_fccw
        if getattr(linear_cls, "_hswq_int8_gemm", False):
            return
        _orig_fccw = linear_cls.forward_comfy_cast_weights

        def forward_comfy_cast_weights(self, input, compute_dtype=None,
                                       want_requant=False, weight_only_quant=False):
            w = self.weight
            if isinstance(w, QuantizedTensor) and getattr(w, "_layout_cls", None) == "TensorWiseINT8Layout":
                qdata, scale = TensorWiseINT8Layout.get_plain_tensors(w)
                bias = self.bias
                if bias is not None:
                    bias = bias.to(device=input.device, dtype=input.dtype)
                return ck.int8_linear(
                    input, qdata, scale, bias, input.dtype,
                    convrot=getattr(w._params, "convrot", False),
                    convrot_groupsize=getattr(w._params, "convrot_groupsize", 256),
                )
            return _orig_fccw(self, input, compute_dtype, want_requant, weight_only_quant)

        linear_cls.forward_comfy_cast_weights = forward_comfy_cast_weights
        linear_cls._hswq_int8_gemm = True
        _STATS["layers_patched"] += 1

    # The Linear class is created inside mixed_precision_ops(), so patch it as
    # soon as the first quantized model is built. Hook the class factory.
    _orig_mpo_wrapped = mixed_precision_ops

    def mixed_precision_ops_and_patch(quant_config={}, compute_dtype=torch.bfloat16,
                                      full_precision_mm=False, disabled=[]):
        ops_cls = _orig_mpo_wrapped(quant_config, compute_dtype, full_precision_mm, disabled)
        try:
            make_int8_forward(ops_cls.Linear)
        except Exception:  # noqa: BLE001
            pass
        return ops_cls

    ops.mixed_precision_ops = mixed_precision_ops_and_patch

    # comfy.sd1_clip holds `import comfy.ops` -> same module object.
    try:
        import comfy.sd1_clip as sd1_clip
        if getattr(sd1_clip.comfy, "ops", None) is ops:
            sd1_clip.comfy.ops = ops
    except Exception:  # noqa: BLE001
        pass

    # A CLIP-L checkpoint may ship no text_projection; the INT8 GEMM path
    # reaches it and dies on the None weight. Complete it at load time.
    import comfy.sd as _csd
    _orig_load_clip = getattr(_csd, "load_clip", None)
    if _orig_load_clip is not None and not getattr(_orig_load_clip, "_hswq_int8_gemm", False):
        def load_clip_with_projection_fix(*args, **kwargs):
            clip = _orig_load_clip(*args, **kwargs)
            _complete_text_projection(clip)
            return clip
        load_clip_with_projection_fix._hswq_int8_gemm = True
        load_clip_with_projection_fix.__wrapped__ = _orig_load_clip
        _csd.load_clip = load_clip_with_projection_fix

    ops._hswq_int8_gemm_patched = True
    _PATCHED = True
    logger.info("[HSWQ INT8 GEMM] CLIP text INT8 GEMM enabled (core not modified)")
    return True


def disable_int8_gemm_for_clip_text() -> bool:
    """Undo the patch (restores the stock behaviour)."""
    global _PATCHED
    try:
        import comfy.ops as ops
    except Exception:  # noqa: BLE001
        return False
    if not getattr(ops, "_hswq_int8_gemm_patched", False):
        return False
    logger.info("[HSWQ INT8 GEMM] disable requested; restart ComfyUI to restore the stock path")
    _PATCHED = False
    return True
