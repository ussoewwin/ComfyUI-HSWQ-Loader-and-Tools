"""HSWQ Load ConvRot INT8 SAM3 Model.

ComfyUI's stock SAM3 path (Load Diffusion Model / UNetLoader) loads the SAM3
architecture with default float weights; for INT8 / ConvRot INT8 checkpoints,
MixedPrecisionOps attaches INT8 QuantizedTensor (TensorWiseINT8Layout) to Linear layers.

This loader node:
1. Detects INT8 ComfyQuant metadata (int8_tensorwise / ConvRot).
2. Configures MixedPrecisionOps with bfloat16 base dtype so module graph is
   cleanly constructed and INT8 weights remain in VRAM.
3. Automatically protects against non-multiple-of-4 layers (e.g. boxRPB_embed_x
   which has in_features=2) via safe dequantization fallback, enabling 100%
   error-free execution across all hardware.

The output is a standard MODEL object compatible with HSWQ SAM3 Detect and
ComfyUI SAM3 nodes.
"""
from __future__ import annotations

import json
import logging
import os
from typing import List

import torch

import comfy.sd
import comfy.utils
from comfy import ops as comfy_ops
from comfy.quant_ops import QUANT_ALGOS
import folder_paths

from ..patches.comfy_quant_int8 import (
    _patch_comfy_kitchen_int8_gemm_fallback,
    apply_comfy_quant_int8_patches,
)
from .utils import get_filename_list, get_full_path_or_raise

logger = logging.getLogger(__name__)

_SAM_FOLDERS = ["diffusion_models", "sams", "detection", "checkpoints"]


def _get_sam3_filenames() -> List[str]:
    """Get list of candidate filenames for SAM3 models across standard directories."""
    files: set[str] = set()
    for folder in _SAM_FOLDERS:
        try:
            flist = folder_paths.get_filename_list(folder)
            if flist:
                files.update(flist)
        except Exception:
            pass
    if not files:
        try:
            return folder_paths.get_filename_list("diffusion_models")
        except Exception:
            return []
    return sorted(list(files))


def _get_sam3_full_path_or_raise(filename: str) -> str:
    """Resolve full path to SAM3 model across candidate directories."""
    for folder in _SAM_FOLDERS:
        try:
            path = folder_paths.get_full_path(folder, filename)
            if path and os.path.isfile(path):
                return path
        except Exception:
            pass
    # Fallback to get_full_path_or_raise on primary diffusion_models directory
    return folder_paths.get_full_path_or_raise("diffusion_models", filename)


def _decode_comfy_quant(raw) -> dict:
    try:
        return json.loads(raw.numpy().tobytes())
    except Exception:
        return {}


def _has_int8_comfy_quant(sd: dict) -> bool:
    """True if checkpoint carries >=1 int8_tensorwise comfy_quant layer."""
    for key in sd.keys():
        if not key.endswith(".comfy_quant"):
            continue
        conf = _decode_comfy_quant(sd[key])
        if isinstance(conf, dict) and conf.get("format") == "int8_tensorwise":
            return True
        if conf == "int8_tensorwise":
            return True
    return False


def _int8_mixed_precision_ops():
    """MixedPrecisionOps supporting int8_tensorwise (ConvRot included)."""
    quant_config = {
        "int8_tensorwise": QUANT_ALGOS["int8_tensorwise"],
    }
    return comfy_ops.mixed_precision_ops(
        quant_config,
        torch.bfloat16,
        full_precision_mm=False,
        disabled=[],
    )


class HSWQSAM3Loader:
    """Load a ConvRot INT8 SAM3 model keeping weights INT8 in VRAM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_name": (
                    _get_sam3_filenames(),
                    {"tooltip": "ConvRot INT8 / standard SAM3 model (.safetensors)"},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_sam3"
    CATEGORY = "loaders"
    TITLE = "HSWQ SAM3 Loader (ConvRot INT8)"
    SEARCH_ALIASES = [
        "sam3",
        "sam3 loader",
        "load sam3",
        "sam 3",
        "segment anything",
        "hswq sam3",
        "convrot sam3",
        "int8 sam3",
    ]

    def load_sam3(self, sam3_name: str, **kwargs):
        _patch_comfy_kitchen_int8_gemm_fallback()
        apply_comfy_quant_int8_patches()
        ckpt_path = _get_sam3_full_path_or_raise(sam3_name)

        sd = comfy.utils.load_torch_file(ckpt_path, safe_load=True)

        if _has_int8_comfy_quant(sd):
            model_options = {
                "dtype": torch.bfloat16,
                "custom_operations": _int8_mixed_precision_ops(),
            }
            logger.info(
                "[HSWQ INT8 SAM3] INT8 ComfyQuant detected: loading %s with "
                "MixedPrecisionOps (weights stay INT8 in VRAM)",
                sam3_name,
            )
        else:
            model_options = {}
            logger.info(
                "[HSWQ INT8 SAM3] No INT8 ComfyQuant layers, loading %s directly",
                sam3_name,
            )

        model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options=model_options
        )
        if model is None:
            # Fallback to load_diffusion_model with path
            model = comfy.sd.load_diffusion_model(
                ckpt_path, model_options=model_options
            )
        if model is None:
            raise RuntimeError(
                f"[HSWQ INT8 SAM3] Failed to load SAM3 model from {sam3_name}"
            )
        return (model,)


HSWQLoadConvRotINT8SAM3 = HSWQSAM3Loader
