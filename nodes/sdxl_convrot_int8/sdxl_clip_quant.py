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

    CLIP-L: ``conditioner.embedders.0.transformer.*`` -> ``clip_l.transformer.*``
    (a plain prefix swap, matching SDXL.process_clip_state_dict).

    CLIP-G: ``conditioner.embedders.1.model.*`` is first ``model.`` -> ``clip_g.``
    then run through ``comfy.utils.transformers_convert`` which restructures the
    OpenAI CLIP text-transformer layout into the HuggingFace one. The internal
    module names are therefore NOT a plain prefix swap:
        transformer.resblocks.N.ln_1            -> transformer.text_model.encoder.layers.N.layer_norm1
        transformer.resblocks.N.ln_2            -> transformer.text_model.encoder.layers.N.layer_norm2
        transformer.resblocks.N.mlp.c_fc        -> transformer.text_model.encoder.layers.N.mlp.fc1
        transformer.resblocks.N.mlp.c_proj      -> transformer.text_model.encoder.layers.N.mlp.fc2
        transformer.resblocks.N.attn.out_proj   -> transformer.text_model.encoder.layers.N.self_attn.out_proj
        transformer.resblocks.N.attn.in_proj_*  -> transformer.text_model.encoder.layers.N.self_attn.{q,k,v}_proj (split)
        positional_embedding                    -> transformer.text_model.embeddings.position_embedding.weight
        token_embedding.weight                  -> transformer.text_model.embeddings.token_embedding.weight
        ln_final                                -> transformer.text_model.final_layer_norm
        text_projection                         -> transformer.text_projection

    Returns None when the key is not an SDXL CLIP layer.
    """
    if key.startswith("conditioner.embedders.0.transformer."):
        return "clip_l." + key[len("conditioner.embedders.0."):]
    if key.startswith("conditioner.embedders.1.model."):
        rest = key[len("conditioner.embedders.1.model."):]
        return _clip_g_internal_name(rest)
    return None


def _clip_g_internal_name(rest: str):
    """Translate a CLIP-G (OpenAI CLIP) key tail to its internal HF-style name.

    ``rest`` is the part after ``conditioner.embedders.1.model.`` (e.g.
    ``transformer.resblocks.5.mlp.c_proj`` or ``token_embedding.weight``).
    Returns ``clip_g.<internal>`` or None if it cannot be mapped.
    """
    # whole-model renames (transformers_convert keys_to_replace)
    whole = {
        "positional_embedding": "transformer.text_model.embeddings.position_embedding.weight",
        "token_embedding.weight": "transformer.text_model.embeddings.token_embedding.weight",
        "ln_final.weight": "transformer.text_model.final_layer_norm.weight",
        "ln_final.bias": "transformer.text_model.final_layer_norm.bias",
        "text_projection": "transformer.text_projection.weight",
    }
    if rest in whole:
        return "clip_g." + whole[rest]

    # per-resblock renames (module base form, no trailing .weight/.bias)
    import re as _re

    m = _re.match(
        r"transformer\.resblocks\.(\d+)\.(ln_1|ln_2|mlp\.c_fc|mlp\.c_proj|attn\.out_proj)$",
        rest,
    )
    if m:
        idx, sub = m.group(1), m.group(2)
        sub_map = {
            "ln_1": "layer_norm1",
            "ln_2": "layer_norm2",
            "mlp.c_fc": "mlp.fc1",
            "mlp.c_proj": "mlp.fc2",
            "attn.out_proj": "self_attn.out_proj",
        }
        return (
            f"clip_g.transformer.text_model.encoder.layers.{idx}.{sub_map[sub]}"
        )

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


def _remap_clipg_companion_keys(sd) -> int:
    """Move CLIP-G companion tensors (.comfy_quant / .weight_scale) to the same
    HuggingFace layout that ``comfy.utils.transformers_convert`` gives the .weight
    keys.

    transformers_convert only rewrites ``transformer.resblocks.N.*.weight/bias``
    (and a few whole-model keys); it leaves ``.comfy_quant`` / ``.weight_scale``
    at the OpenAI-CLIP path. The quantized ops module then looks for
    ``clip_g.transformer.text_model.encoder.layers.N.mlp.fc1.comfy_quant`` and
    finds nothing, so the int8 weight loads as raw fp16 and the conditioning
    collapses. Remap the companions here (in place) to match.

    Returns the number of keys moved.
    """
    import re as _re

    sub_map = {
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "mlp.c_fc": "mlp.fc1",
        "mlp.c_proj": "mlp.fc2",
        "attn.out_proj": "self_attn.out_proj",
    }
    whole = {
        "token_embedding": "transformer.text_model.embeddings.token_embedding",
        "positional_embedding": "transformer.text_model.embeddings.position_embedding",
        "ln_final": "transformer.text_model.final_layer_norm",
    }
    prefix = "conditioner.embedders.1.model."
    moved = 0
    for k in list(sd.keys()):
        if not k.startswith(prefix):
            continue
        if not (k.endswith(".comfy_quant") or k.endswith(".weight_scale")):
            continue
        tail = k[len(prefix):]
        suffix = ".comfy_quant" if tail.endswith(".comfy_quant") else ".weight_scale"
        base = tail[: -len(suffix)]

        m = _re.match(
            r"transformer\.resblocks\.(\d+)\.(ln_1|ln_2|mlp\.c_fc|mlp\.c_proj|attn\.out_proj)$",
            base,
        )
        if m:
            idx, sub = m.group(1), m.group(2)
            new_base = (
                f"transformer.text_model.encoder.layers.{idx}.{sub_map[sub]}"
            )
        elif base in whole:
            new_base = whole[base]
        else:
            continue
        sd[prefix + new_base + suffix] = sd.pop(k)
        moved += 1
    return moved


def arm_sdxl_clip_quant_metadata() -> bool:
    """Install the SDXL CLIP quantization-metadata wrapper. Idempotent."""
    try:
        import comfy.sd as _sd
    except ImportError:
        return False

    original = getattr(_sd, "load_state_dict_guess_config", None)
    if original is None:
        return False
    _cur = original
    for _ in range(8):
        if _cur is None:
            break
        if getattr(_cur, _MARKER, False):
            return True
        _cur = getattr(_cur, "__wrapped__", None)

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

        # CLIP-G companions must be moved to the HF layout before the weight-only
        # transformers_convert runs (see _remap_clipg_companion_keys).
        moved = _remap_clipg_companion_keys(sd)
        if moved:
            _console(
                f"[HSWQ INT8][SDXL CLIP] remapped {moved} CLIP-G companion tensors "
                f"to the HF layout"
            )

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
