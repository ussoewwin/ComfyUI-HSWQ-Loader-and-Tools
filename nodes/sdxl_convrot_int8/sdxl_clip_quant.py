# -*- coding: utf-8 -*-
"""SDXL-only CLIP ConvRot INT8 metadata injection.

**Scope: SDXL ConvRot INT8 only (nodes/sdxl_convrot_int8/).** This module exists
so the CLIP quantization-metadata fix does not live in any shared module:
``patches/comfy_quant_int8.py`` keeps its original code, and only the SDXL load
branch calls :func:`arm_sdxl_clip_quant_metadata`.

Why this is needed
------------------
ComfyUI builds a text encoder with ``comfy.ops.manual_cast`` unless
``model_options["quantization_metadata"]`` is set (``SDClipModel.__init__`` in
``comfy/sd1_clip.py``). The SDXL text-encoder load path in ``comfy.sd`` never
sets it, so an HSWQ SDXL pack whose CLIP Linear layers are stored as
``int8`` + ``.weight_scale`` + ``.comfy_quant`` loads those tensors as raw fp16
(absmax ~127). The conditioning then collapses (cos ~0.10) or NaNs - the same
failure mode fixed for SD1.5 in
``nodes/sd1.5 int8/comfy_quant_int8.py`` (``_patch_sd15_clip_quant_metadata``).

What it does
------------
Wraps ``comfy.sd.load_state_dict_guess_config`` and, when the state dict carries
SDXL CLIP ``comfy_quant`` layers, passes a CLIP-restricted ``quantization_metadata``
through ``te_model_options`` so ``SDXLClipModel`` builds both ``clip_l`` and
``clip_g`` with ``mixed_precision_ops``. The metadata layer keys are remapped to
the internal CLIP module names that ``SDXL.process_clip_state_dict`` produces:

    conditioner.embedders.0.transformer.*  ->  clip_l.transformer.*
    conditioner.embedders.1.model.*        ->  clip_g.transformer.*

Unlike SD1.5, SDXL's CLIP stores ``text_projection`` / ``logit_scale`` inside the
checkpoint, so no optional-param completion is required.

Separation guarantees
---------------------
* no import-time side effects (importing this module changes nothing),
* the wrapper refuses to act unless the state dict actually carries
  ``conditioner.embedders.*`` ``comfy_quant`` Linear layers (SD1.5 / SD3 / FLUX /
  Z Image / Krea2 / Qwen never match this shape),
* only the SDXL-guarded load branch loads this file, by path, without registering
  it in ``sys.modules``.
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

_MARKER = "_hswq_sdxl_clip_meta"


def _console(msg: str) -> None:
    print(msg, flush=True)


def _remap_sdxl_clip_layer(key: str):
    """Map a pack CLIP layer key to its internal SDXL CLIP module name.

    Returns None when the key is not an SDXL CLIP layer.
    """
    if key.startswith("conditioner.embedders.0.transformer."):
        return "clip_l." + key[len("conditioner.embedders.0."):]
    if key.startswith("conditioner.embedders.1.model."):
        # conditioner.embedders.1.model.* -> clip_g.transformer.*
        # (SDXL.process_clip_state_dict: model. -> clip_g., then
        #  clip_text_transformers_convert clip_g. -> clip_g.transformer.)
        return "clip_g.transformer." + key[len("conditioner.embedders.1.model."):]
    return None


def _decode_comfy_quant_conf(raw):
    """Decode a .comfy_quant uint8 tensor into a conf dict (None if unusable)."""
    try:
        if hasattr(raw, "numpy"):
            return json.loads(raw.numpy().tobytes())
        if isinstance(raw, (bytes, bytearray)):
            return json.loads(bytes(raw))
        if isinstance(raw, str):
            return json.loads(raw)
    except Exception:
        return None
    return None


def _collect_sdxl_clip_layers(state_dict, metadata):
    """Build {internal_clip_layer: conf} from .comfy_quant tensors or metadata."""
    layers = {}
    # 1. File-embedded .comfy_quant tensors (authoritative for our packs).
    for k in list(state_dict.keys()):
        if not (k.startswith("conditioner.embedders.") and k.endswith(".comfy_quant")):
            continue
        base = k[: -len(".comfy_quant")]
        conf = _decode_comfy_quant_conf(state_dict[k])
        if conf is None:
            continue
        internal = _remap_sdxl_clip_layer(base)
        if internal is not None:
            layers[internal] = conf
    # 2. Metadata fallback (packs that omit the per-layer tensor).
    if not layers and isinstance(metadata, dict) and "_quantization_metadata" in metadata:
        try:
            qm = json.loads(metadata["_quantization_metadata"])
            for k, v in qm.get("layers", {}).items():
                internal = _remap_sdxl_clip_layer(k)
                if internal is not None:
                    layers[internal] = v
        except (TypeError, json.JSONDecodeError):
            pass
    return layers


def _complete_sdxl_clip_optional_params(out):
    """SDXL CLIP-L stores no ``text_projection`` in the checkpoint
    (SDXL.process_clip_state_dict_for_saving pops
    ``clip_l.transformer.text_projection.weight``). The FP16 path builds it as an
    unused all-zeros Linear; under mixed_precision_ops the same Linear is a
    quantized ops.Linear whose weight stays None (the pack has no
    ``comfy_quant`` for it), so the INT8 forward crashes on the None weight
    (``'NoneType' object has no attribute 'device'``). Complete it with a plain
    fp16 zeros weight, exactly like the FP16 path.
    """
    try:
        clip = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
        if clip is None:
            return
        csm = getattr(clip, "cond_stage_model", None)
        inner = getattr(csm, "clip_l", None)
        if inner is None:
            return
        import torch as _t
        tr = getattr(inner, "transformer", None)
        tp = getattr(tr, "text_projection", None) if tr is not None else None
        if tp is not None and getattr(tp, "weight", None) is None:
            hidden = 768
            try:
                emb = tr.text_model.embeddings.token_embedding.weight
                hidden = int(emb.shape[1])
                dtype = emb.dtype
            except Exception:
                dtype = _t.float16
            tp.weight = _t.nn.Parameter(
                _t.zeros((hidden, hidden), dtype=dtype), requires_grad=False
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[HSWQ INT8][SDXL CLIP] optional-param completion skipped: %s", exc)


def arm_sdxl_clip_quant_metadata() -> bool:
    """Install the SDXL CLIP quantization-metadata wrapper. Idempotent."""
    try:
        import comfy.sd as _sd
    except ImportError:
        return False

    original = getattr(_sd, "load_state_dict_guess_config", None)
    if original is None:
        return False
    if getattr(original, _MARKER, False):
        return True

    def load_state_dict_guess_config_sdxl_clip(sd, *args, **kwargs):
        try:
            has_clip_quant = any(
                k.startswith("conditioner.embedders.") and k.endswith(".comfy_quant")
                for k in sd.keys()
            )
        except Exception:
            has_clip_quant = False
        if not has_clip_quant:
            return original(sd, *args, **kwargs)

        clip_layers = _collect_sdxl_clip_layers(sd, kwargs.get("metadata"))
        if not clip_layers:
            return original(sd, *args, **kwargs)

        te_model_options = dict(kwargs.get("te_model_options") or {})
        te_model_options["quantization_metadata"] = {
            "format_version": "1.0",
            "layers": clip_layers,
        }
        kwargs["te_model_options"] = te_model_options
        _console(
            f"[HSWQ INT8][SDXL CLIP] injecting ConvRot INT8 metadata for "
            f"{len(clip_layers)} text-encoder layers"
        )
        out = original(sd, *args, **kwargs)
        _complete_sdxl_clip_optional_params(out)
        return out

    load_state_dict_guess_config_sdxl_clip.__dict__[_MARKER] = True
    load_state_dict_guess_config_sdxl_clip.__wrapped__ = original
    _sd.load_state_dict_guess_config = load_state_dict_guess_config_sdxl_clip
    return True


def disarm_sdxl_clip_quant_metadata() -> bool:
    """Restore the original loader (mainly for tests)."""
    try:
        import comfy.sd as _sd
    except ImportError:
        return False
    cur = getattr(_sd, "load_state_dict_guess_config", None)
    if cur is None or not getattr(cur, _MARKER, False):
        return False
    orig = getattr(cur, "__wrapped__", None)
    if orig is None:
        return False
    _sd.load_state_dict_guess_config = orig
    return True
