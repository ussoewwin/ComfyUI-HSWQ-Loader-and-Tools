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
            # Kitchen "plain NVFP4" (native_convert_nvfp4.py) has no per-layer
            # .comfy_quant tensors; it stores a top-level _quantization_metadata
            # header whose layer entries carry {"format": "nvfp4"}.
            meta = f.metadata() or {}
            raw = meta.get("_quantization_metadata")
            if raw:
                try:
                    layers = json.loads(raw).get("layers", {})
                except (TypeError, ValueError):
                    layers = {}
                for v in layers.values():
                    if is_nvfp4_conf(decode_comfy_quant_conf(v)):
                        return True
    except Exception as e:
        logger.debug("NVFP4 probe failed for %s: %s", path, e)
        return False
    return False


def _nvfp4_logical_linear_shape(state_dict: dict, weight_key: str):
    """Return (out_features, in_features) for an NVFP4 Linear weight, or None.

    Prefers ``orig_shape`` / ``in_features`` written by auto_int8_nvfp4_hybrid.py
    into comfy_quant. Packed ``weight.shape`` alone is not the logical shape.
    """
    import torch

    if weight_key not in state_dict:
        return None
    cq_key = comfy_quant_key_for_weight(weight_key)
    conf = decode_comfy_quant_conf(state_dict.get(cq_key))
    if not is_nvfp4_conf(conf):
        # Hybrid may leave weight_scale_2 without injecting cq yet; still NVFP4.
        base = weight_key[: -len(".weight")] if weight_key.endswith(".weight") else weight_key
        if f"{base}.weight_scale_2" not in state_dict:
            return None
        conf = conf if isinstance(conf, dict) else {}

    if isinstance(conf, dict):
        orig = conf.get("orig_shape")
        if isinstance(orig, (list, tuple)) and len(orig) >= 2:
            return int(orig[0]), int(orig[1])
        if conf.get("in_features") is not None and conf.get("out_features") is not None:
            return int(conf["out_features"]), int(conf["in_features"])
        if conf.get("in_features") is not None:
            weight = state_dict[weight_key]
            if torch.is_tensor(weight) and weight.ndim >= 1:
                # out_features unknown; caller may only need in_features
                return None, int(conf["in_features"])
    return None


def fix_unet_config_packed_dims(unet_config: dict, state_dict: dict, key_prefix: str) -> dict:
    """Rewrite context_dim / adm_in_channels / Krea2 txtlayers for NVFP4 packed K."""
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

    # Krea2: stock detect sets txtlayers = projector.weight.shape[1] (packed K
    # after hybrid NVFP4). Restore logical in_features from comfy_quant
    # orig_shape, else KREA2_TAP_LAYERS length (matches CLIP fused width).
    if unet_config.get("txtlayers") is not None:
        proj = f"{key_prefix}txtfusion.projector.weight"
        if proj in state_dict:
            try:
                logical = _nvfp4_logical_linear_shape(state_dict, proj)
                if logical is not None and logical[1] is not None:
                    unet_config["txtlayers"] = int(logical[1])
                else:
                    cq = decode_comfy_quant_conf(
                        state_dict.get(comfy_quant_key_for_weight(proj))
                    )
                    base = proj[: -len(".weight")]
                    is_proj_nvfp4 = is_nvfp4_conf(cq) or (
                        f"{base}.weight_scale_2" in state_dict
                    )
                    if is_proj_nvfp4:
                        from comfy.text_encoders.krea2 import KREA2_TAP_LAYERS

                        unet_config["txtlayers"] = len(KREA2_TAP_LAYERS)
                        logger.info(
                            "[HSWQ NVFP4] txtlayers <- len(KREA2_TAP_LAYERS)=%s "
                            "(no orig_shape on projector)",
                            unet_config["txtlayers"],
                        )
            except Exception as e:
                logger.warning("[HSWQ NVFP4] txtlayers fix skipped: %s", e)

    return unet_config
