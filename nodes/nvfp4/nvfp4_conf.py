"""NVFP4 comfy_quant config helpers (HSWQ-owned; never edit ComfyUI-master)."""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Packed E2M1 nibble pairs along K: storage_K = logical_padded_K // 2
_NVFP4_PACK_FACTOR = 2


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


def is_nvfp4_conf(conf: Optional[dict]) -> bool:
    return isinstance(conf, dict) and conf.get("format") == "nvfp4"


def convrot_flags_from_conf(conf: Optional[dict]) -> tuple[bool, int]:
    """Return (enabled, groupsize) from an nvfp4 comfy_quant dict."""
    if not is_nvfp4_conf(conf):
        return False, 256
    if not bool(conf.get("convrot", False)):
        return False, 256
    params_conf = conf.get("params", {})
    if not isinstance(params_conf, dict):
        params_conf = {}
    gs = int(conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256)) or 256)
    return True, gs


def logical_linear_in_features(state_dict: dict, weight_key: str) -> int:
    """Return logical in_features for a Linear weight (expand packed NVFP4 K)."""
    import torch

    weight = state_dict[weight_key]
    if not torch.is_tensor(weight) or weight.ndim < 2:
        raise ValueError(
            f"{weight_key}: expected 2D+ tensor, got {type(weight)} "
            f"ndim={getattr(weight, 'ndim', None)}"
        )

    packed_in = int(weight.shape[1])
    cq_key = comfy_quant_key_for_weight(weight_key)
    conf = decode_comfy_quant_conf(state_dict.get(cq_key))
    if is_nvfp4_conf(conf) and weight.ndim == 2:
        return packed_in * _NVFP4_PACK_FACTOR
    return packed_in


def checkpoint_looks_like_comfy_quant_nvfp4(state_dict_or_path) -> bool:
    """True if checkpoint has at least one nvfp4 comfy_quant marker."""
    import torch

    if isinstance(state_dict_or_path, (str, os.PathLike)):
        return _probe_path_comfy_quant_nvfp4(str(state_dict_or_path))

    state_dict = state_dict_or_path
    for key, value in state_dict.items():
        if not key.endswith(".comfy_quant"):
            continue
        if not torch.is_tensor(value):
            continue
        conf = decode_comfy_quant_conf(value)
        if is_nvfp4_conf(conf):
            return True
    return False


def _probe_path_comfy_quant_nvfp4(path: str) -> bool:
    try:
        from safetensors import safe_open
    except ImportError:
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            comfy_keys = [k for k in keys if k.endswith(".comfy_quant")]
            for ck in comfy_keys[:64]:
                conf = decode_comfy_quant_conf(f.get_tensor(ck))
                if is_nvfp4_conf(conf):
                    return True
    except Exception as e:
        logger.debug("NVFP4 probe failed for %s: %s", path, e)
        return False
    return False


def fix_unet_config_packed_dims(unet_config: dict, state_dict: dict, key_prefix: str) -> dict:
    """Rewrite context_dim / adm_in_channels using logical NVFP4 in_features."""
    if not isinstance(unet_config, dict):
        return unet_config

    y_input = f"{key_prefix}label_emb.0.0.weight"
    if y_input in state_dict and unet_config.get("adm_in_channels") is not None:
        try:
            unet_config["adm_in_channels"] = logical_linear_in_features(state_dict, y_input)
        except Exception as e:
            logger.warning("[HSWQ NVFP4] adm_in_channels fix skipped: %s", e)

    if unet_config.get("context_dim") is not None:
        attn_k = None
        suffix = "attn2.to_k.weight"
        for k in state_dict.keys():
            if k.startswith(key_prefix) and k.endswith(suffix):
                attn_k = k
                break
        if attn_k is not None:
            try:
                unet_config["context_dim"] = logical_linear_in_features(state_dict, attn_k)
            except Exception as e:
                logger.warning("[HSWQ NVFP4] context_dim fix skipped: %s", e)

    return unet_config
