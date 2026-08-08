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
    """Return logical in_features for a Linear weight.

    NVFP4 storage K is packed (and often 16-padded). Never guess
    ``packed_shape[1] * 2`` — that recovers padded K, not logical in_features
    (e.g. logical 12 → pad 16 → pack 8 → *2 = 16 ≠ 12). Require
    ``orig_shape`` / ``in_features`` on comfy_quant (or refuse).
    """
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
        orig = conf.get("orig_shape") if isinstance(conf, dict) else None
        if orig is not None and len(orig) >= 2:
            return int(orig[1])
        if conf.get("in_features") is not None:
            return int(conf["in_features"])
        raise ValueError(
            f"{weight_key}: nvfp4 packed weight but comfy_quant lacks "
            f"orig_shape/in_features; refuse packed_K*{_NVFP4_PACK_FACTOR} guess "
            f"(packed_K={packed_in})"
        )
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
    """Rewrite context_dim / adm_in_channels using logical NVFP4 in_features.

    Fail-closed: if the target weight is nvfp4 and logical in_features cannot
    be resolved from comfy_quant, raise (do not keep packed/padded K).
    """
    if not isinstance(unet_config, dict):
        return unet_config

    y_input = f"{key_prefix}label_emb.0.0.weight"
    if y_input in state_dict and unet_config.get("adm_in_channels") is not None:
        unet_config["adm_in_channels"] = logical_linear_in_features(state_dict, y_input)

    if unet_config.get("context_dim") is not None:
        attn_k = None
        suffix = "attn2.to_k.weight"
        for k in state_dict.keys():
            if k.startswith(key_prefix) and k.endswith(suffix):
                attn_k = k
                break
        if attn_k is not None:
            unet_config["context_dim"] = logical_linear_in_features(state_dict, attn_k)

    return unet_config


# ---------------------------------------------------------------------------
# Blackwell GPU capability detection (SDXL NVFP4 product path only)
# ---------------------------------------------------------------------------
_GPU_CC: tuple | None = None


def _get_gpu_cc() -> tuple:
    """Return (major, minor) compute capability, cached."""
    global _GPU_CC
    if _GPU_CC is None:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            _GPU_CC = torch.cuda.get_device_capability()
        else:
            _GPU_CC = (0, 0)
    return _GPU_CC


def is_blackwell_gpu() -> bool:
    """True if GPU is Blackwell class (SM >= 100): B200, RTX 5090, etc.

    Called only from nodes/nvfp4 product-TC path guards.  Z Image and INT8
    never reach this code.
    """
    major, _ = _get_gpu_cc()
    return major >= 10


def is_blackwell_datacenter() -> bool:
    """True if GPU is SM100 datacenter Blackwell (TMA/TMEM available)."""
    major, minor = _get_gpu_cc()
    return major == 10 and minor == 0


def is_blackwell_consumer() -> bool:
    """True if GPU is SM120/SM121 consumer Blackwell (RTX 50x0 series)."""
    major, _ = _get_gpu_cc()
    return major == 12


def is_nvfp4_cudagraph_enabled() -> bool:
    """Return whether CUDA Graph / Tensor Boost execution is active.

    Evaluates HSWQ_NVFP4_TENSORBOOST and HSWQ_NVFP4_CUDAGRAPH environment variables.
    Returns True if set to '1' / 'true' / 'on' / 'enable'.
    Returns False otherwise.
    """
    import os

    for env_key in ("HSWQ_NVFP4_CUDAGRAPH", "HSWQ_NVFP4_TENSORBOOST"):
        val = os.environ.get(env_key, "").strip().lower()
        if val in ("1", "true", "on", "enable", "enabled"):
            return True
        if val in ("0", "false", "off", "disable", "disabled"):
            return False

    return False

