"""Flux1 ConvRot W4A8 comfy_quant helpers (``asym_w4a8_int8``).

Owned by ``nodes/flux1 convrot w4a8``. The pack layout is the one ComfyUI core
already decodes (measured, ``comfy/ops.py`` grouped-int8 branch):

  <layer>.weight            int8    [N, K/2]   nibble packed, low = even column
  <layer>.weight_s_rel      fp8e4m3 [N, K/16]  per-group scale
  <layer>.weight_s_channel  float32 [N]        per-out-channel scale
  <layer>.weight_codebook   float32 [16]       Lloyd-Max table (absent when unused)
  <layer>.comfy_quant       uint8 JSON {"format": "asym_w4a8_int8",
                                        "group_size": 16,
                                        "convrot_groupsize": 256}

``w6a8_int8`` shares the layout class with 6-bit codes; it is a different being
and is deliberately NOT accepted here.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

W4A8_FORMAT = "asym_w4a8_int8"
W4A8_BITS = 4
W4A8_GROUP_SIZE = 16
W4A8_CONVROT_GROUPSIZE = 256
# Logical K is recovered from the packed K by the code width.
_PACK_DIVISOR = 8 // W4A8_BITS  # 2


def decode_comfy_quant_conf(raw: Any) -> Optional[dict]:
    """Decode a comfy_quant marker into a dict layer config."""
    import torch

    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if torch.is_tensor(raw):
        conf = json.loads(raw.detach().cpu().numpy().tobytes())
    elif isinstance(raw, (bytes, bytearray, memoryview)):
        conf = json.loads(bytes(raw))
    elif isinstance(raw, str):
        conf = raw
    else:
        conf = raw

    while isinstance(conf, str):
        try:
            parsed = json.loads(conf)
        except (TypeError, json.JSONDecodeError):
            return {"format": conf}
        if parsed is conf:
            return {"format": conf}
        conf = parsed

    if isinstance(conf, dict):
        return conf
    raise TypeError(
        f"comfy_quant config must be a dict or format string, got {type(conf).__name__}"
    )


def comfy_quant_key_for_weight(weight_key: str) -> str:
    if weight_key.endswith(".weight"):
        return weight_key[: -len("weight")] + "comfy_quant"
    if weight_key.endswith("weight"):
        return weight_key[: -len("weight")] + "comfy_quant"
    return weight_key + ".comfy_quant"


def is_w4a8_conf(conf: Optional[dict]) -> bool:
    return isinstance(conf, dict) and conf.get("format") == W4A8_FORMAT


def quant_params_from_conf(conf: Optional[dict]) -> dict:
    """Return {group_size, convrot_groupsize} with the pack defaults filled in."""
    if not is_w4a8_conf(conf):
        return {"group_size": W4A8_GROUP_SIZE,
                "convrot_groupsize": W4A8_CONVROT_GROUPSIZE}
    params_conf = conf.get("params", {})
    if not isinstance(params_conf, dict):
        params_conf = {}
    return {
        "group_size": int(
            conf.get("group_size", params_conf.get("group_size", W4A8_GROUP_SIZE))
            or W4A8_GROUP_SIZE
        ),
        "convrot_groupsize": int(
            conf.get(
                "convrot_groupsize",
                params_conf.get("convrot_groupsize", W4A8_CONVROT_GROUPSIZE),
            )
            or W4A8_CONVROT_GROUPSIZE
        ),
    }


def convrot_flags_from_conf(conf: Optional[dict]) -> tuple[bool, int]:
    """W4A8 packs are always ConvRot-rotated; return (True, groupsize)."""
    if not is_w4a8_conf(conf):
        return False, W4A8_CONVROT_GROUPSIZE
    return True, quant_params_from_conf(conf)["convrot_groupsize"]


def logical_linear_in_features(state_dict: dict, weight_key: str) -> int:
    """Logical in_features of a W4A8 Linear weight (packed K -> logical K)."""
    import torch

    weight = state_dict[weight_key]
    if not torch.is_tensor(weight) or weight.ndim < 2:
        raise ValueError(
            f"{weight_key}: expected 2D+ tensor, got {type(weight)} "
            f"ndim={getattr(weight, 'ndim', None)}"
        )
    conf = decode_comfy_quant_conf(
        state_dict.get(comfy_quant_key_for_weight(weight_key))
    )
    if is_w4a8_conf(conf) and weight.ndim == 2:
        return int(weight.shape[1]) * _PACK_DIVISOR
    return int(weight.shape[1])


# Shape-critical Linear keys: ComfyUI's FLUX detector reads shape[1] of these to
# size the architecture (in_channels / context_in_dim / vec_in_dim). A packed
# (K/2) weight there would build the module with half K, so the Flux1 converter
# keeps them in their base dtype. If a pack ever carries them packed, stop with
# a clear error instead of loading a silently wrong config.
_SHAPE_CRITICAL_REL_KEYS = (
    "img_in.weight",
    "txt_in.weight",
    "vector_in.in_layer.weight",
)


def find_packed_shape_critical_key(state_dict, key_prefix: str = "") -> Optional[str]:
    """Return the first shape-critical key that is W4A8-packed, or None."""
    for rel in _SHAPE_CRITICAL_REL_KEYS:
        key = f"{key_prefix}{rel}"
        weight = state_dict.get(key)
        if weight is None:
            continue
        conf = decode_comfy_quant_conf(
            state_dict.get(comfy_quant_key_for_weight(key))
        )
        if is_w4a8_conf(conf):
            return key
    return None


def checkpoint_looks_like_comfy_quant_w4a8(state_dict_or_path) -> bool:
    """True if the checkpoint carries at least one asym_w4a8_int8 marker."""
    import torch

    if isinstance(state_dict_or_path, (str, os.PathLike)):
        return _probe_path_comfy_quant_w4a8(str(state_dict_or_path))

    for key, value in state_dict_or_path.items():
        if not key.endswith(".comfy_quant"):
            continue
        if not torch.is_tensor(value):
            continue
        if is_w4a8_conf(decode_comfy_quant_conf(value)):
            return True
    return False


def _probe_path_comfy_quant_w4a8(path: str) -> bool:
    try:
        from safetensors import safe_open
    except ImportError:
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            for ck in [k for k in f.keys() if k.endswith(".comfy_quant")][:64]:
                if is_w4a8_conf(decode_comfy_quant_conf(f.get_tensor(ck))):
                    return True
    except Exception as e:  # noqa: BLE001 - probe must never raise
        logger.debug("W4A8 probe failed for %s: %s", path, e)
        return False
    return False
