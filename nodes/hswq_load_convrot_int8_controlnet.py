"""HSWQ Load ConvRot INT8 ControlNet Model.

ComfyUI's stock ControlNet path (``controlnet_load_state_dict``) calls
``control_model.load_state_dict(sd, strict=False)`` directly instead of
routing through the quant-aware ``set_state_dict`` used by UNet/CLIP, so a
ConvRot INT8 (``int8_tensorwise`` + ``comfy_quant``) ControlNet checkpoint
fails to load there ("Load ControlNet Model failed").

This node performs the reverse transform ourselves:

1. Load the checkpoint and find every layer carrying ``<layer>.comfy_quant``
   with ``format == "int8_tensorwise"`` (optionally ``convrot: true``).
2. Dequantize:  ``W ≈ q * scale`` (per-out-channel scale, [out,1]).
3. If ConvRot: undo the Hadamard rotation ``W_rot @ H`` so the weight is back
   in plain float space (matches ``native_convert_int8`` / ComfyUI kitchen).
4. Drop ``weight_scale`` / ``comfy_quant`` keys and hand the float
   state_dict to the stock ``comfy.controlnet.load_controlnet_state_dict``
   (Qwen Image Fun ControlNet auto-detected and loaded as usual).

The output is a normal CONTROL_NET object; use it with the stock
"Apply ControlNet" node.

Known limit: dequantized weights are FP32/BF16 in VRAM, so this loader trades
VRAM savings (the whole point of INT8 storage) for compatibility. Runtime
speed/VRAM equal a plain FP16 ControlNet after load.
"""
from __future__ import annotations

import json
import logging
import math
import os

import torch

import comfy.controlnet
import comfy.utils

from .utils import get_filename_list, get_full_path_or_raise

logger = logging.getLogger(__name__)

# Path lookup helpers
_CN_DIR = "controlnet"


# --------------------------------------------------------------------------
# Hadamard helpers (mirror native_convert_int8.py in this repo)
# --------------------------------------------------------------------------

_HADAMARD_CACHE: dict[int, torch.Tensor] = {}


def build_hadamard(size: int) -> torch.Tensor:
    """Normalized regular Hadamard matrix (power of 4)."""
    cached = _HADAMARD_CACHE.get(size)
    if cached is not None:
        return cached
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1],
         [1, 1, -1, 1],
         [1, -1, 1, 1],
         [-1, 1, 1, 1]],
        dtype=torch.float32,
    )
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    h = h / (size ** 0.5)
    _HADAMARD_CACHE[size] = h
    return h


def unrotate_weight(weight: torch.Tensor, h: torch.Tensor, group_size: int) -> torch.Tensor:
    """Inverse of the ConvRot rotation: W = W_rot @ H (H orthogonal)."""
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features {in_features} not divisible by group_size {group_size}"
        )
    group_count = in_features // group_size
    weight_grouped = weight.view(out_features, group_count, group_size)
    h = h.to(dtype=weight.dtype, device=weight.device)
    return torch.matmul(weight_grouped, h).reshape(weight.shape)


# --------------------------------------------------------------------------
# Dequantization
# --------------------------------------------------------------------------

def _decode_comfy_quant(raw: torch.Tensor) -> dict:
    return json.loads(raw.numpy().tobytes())


def dequantize_state_dict(sd: dict) -> dict:
    """Convert an INT8 (ConvRot) state_dict into plain float weights.

    Returns a NEW dict where quantized weights are replaced by dequantized
    float tensors and ``weight_scale`` / ``comfy_quant`` keys are removed.
    Non-quantized entries are copied as-is.
    """
    out = {}
    quant_keys = set()
    for key in sd.keys():
        if key.endswith(".comfy_quant"):
            quant_keys.add(key)

    # First pass: collect layer metadata
    layer_conf: dict[str, dict] = {}
    for key in quant_keys:
        try:
            layer_conf[key[: -len(".comfy_quant")]] = _decode_comfy_quant(sd[key])
        except Exception:  # noqa: BLE001
            continue

    converted = 0
    for key, tensor in sd.items():
        if key.endswith(".comfy_quant") or key.endswith(".weight_scale"):
            continue  # dropped

        if key.endswith(".weight") and key[: -len(".weight")] in layer_conf:
            conf = layer_conf[key[: -len(".weight")]]
            fmt = conf.get("format")
            if fmt != "int8_tensorwise":
                out[key] = tensor
                continue
            scale_key = key[: -len(".weight")] + ".weight_scale"
            scale = sd.get(scale_key)
            if scale is None:
                logger.warning("[HSWQ INT8 CN] No weight_scale for %s; keep as-is", key)
                out[key] = tensor
                continue

            w = tensor.float() * scale.to(tensor.device).float()
            if conf.get("convrot"):
                gs = int(conf.get("convrot_groupsize", 256))
                w = unrotate_weight(w, build_hadamard(gs), gs)
            # Restore storage dtype: original checkpoints were bf16.
            out[key] = w.to(torch.bfloat16)
            converted += 1
        else:
            out[key] = tensor

    logger.info("[HSWQ INT8 CN] Dequantized %d ConvRot INT8 layers", converted)
    return out


# --------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------

class HSWQLoadConvRotINT8ControlNet:
    """Load a ConvRot INT8 ControlNet via dequant + stock ControlNet loader."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_net_name": (
                    get_filename_list(_CN_DIR),
                    {"tooltip": "ConvRot INT8 ControlNet model (.safetensors)"},
                ),
            },
        }

    RETURN_TYPES = ("CONTROL_NET",)
    RETURN_NAMES = ("control_net",)
    FUNCTION = "load_controlnet"
    CATEGORY = "HSWQ-ussoewwin"
    TITLE = "HSWQ Load ConvRot INT8 ControlNet Model"

    def load_controlnet(self, control_net_name: str, **kwargs):
        ckpt_path = get_full_path_or_raise(_CN_DIR, control_net_name)

        sd = comfy.utils.load_torch_file(ckpt_path, safe_load=True)

        # Check if it actually is an INT8/ConvRot pack; if not, pass through.
        needs_dequant = any(
            k.endswith(".comfy_quant") and "int8_tensorwise" in _safe_decode(sd[k])
            for k in sd.keys()
            if k.endswith(".comfy_quant")
        )
        if needs_dequant:
            sd = dequantize_state_dict(sd)
            logger.info(
                "[HSWQ INT8 CN] Loaded %s via INT8 dequant path", control_net_name
            )
        else:
            logger.info(
                "[HSWQ INT8 CN] No INT8 layers detected, loading %s directly",
                control_net_name,
            )

        control = comfy.controlnet.load_controlnet_state_dict(sd)
        if control is None:
            raise RuntimeError(
                f"[HSWQ INT8 CN] Failed to detect controlnet in {control_net_name}"
            )
        return (control,)


def _safe_decode(raw) -> str:
    try:
        return raw.numpy().tobytes().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""