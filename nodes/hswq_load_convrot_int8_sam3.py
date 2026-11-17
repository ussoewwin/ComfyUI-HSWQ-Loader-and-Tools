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

The folder to load from is given in ``sam3_dir`` (required). The model is
loaded from that folder only - an empty folder is rejected, there is no
fallback to another folder. ``sam3_name`` stays a dropdown, and its options
follow ``sam3_dir``.

The output is a standard MODEL object plus the CLIP text encoder that
comes with the checkpoint, compatible with HSWQ SAM3 Detect and ComfyUI SAM3
nodes.
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

from ..hswq.sam3_paths import (
    default_sam3_dir,
    list_sam3_files,
    resolve_sam3_path,
)
from ..patches.comfy_quant_int8 import (
    _patch_comfy_kitchen_int8_gemm_fallback,
    apply_comfy_quant_int8_patches,
)
from .utils import get_filename_list, get_full_path_or_raise

logger = logging.getLogger(__name__)

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
                # COMBO: the dropdown of SAM3 model files. Its options follow
                # sam3_dir (route /hswq/sam3/list).
                "sam3_name": (
                    list_sam3_files(""),
                    {"tooltip": "ConvRot INT8 / standard SAM3 model (.safetensors)"},
                ),
                "sam3_dir": (
                    "STRING",
                    {
                        "default": default_sam3_dir(),
                        "multiline": False,
                        "tooltip": "Folder the SAM3 model is loaded from.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
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

    @classmethod
    def VALIDATE_INPUTS(s, sam3_name, sam3_dir=""):
        """Resolve the path and report a missing file before the run."""
        try:
            path = resolve_sam3_path(sam3_name, sam3_dir)
        except Exception as exc:
            return str(exc)
        if not os.path.isfile(path):
            return "SAM3: model file not found: %s" % path
        return True

    def load_sam3(self, sam3_name: str, sam3_dir: str = "", **kwargs):
        _patch_comfy_kitchen_int8_gemm_fallback()
        apply_comfy_quant_int8_patches()
        ckpt_path = resolve_sam3_path(sam3_name, sam3_dir)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError("SAM3: model file not found: %s" % ckpt_path)

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

        model, clip = None, None
        try:
            # The checkpoint carries the SAM3 text encoder (language_backbone),
            # so the model and the CLIP come out of a single load.
            model, clip, _vae, _clip_vision = comfy.sd.load_state_dict_guess_config(
                sd,
                output_vae=False,
                output_clip=True,
                output_clipvision=False,
                output_model=True,
                model_options=model_options,
            )
        except Exception as exc:
            logger.warning(
                "[HSWQ INT8 SAM3] model+CLIP load failed (%s); "
                "falling back to the model-only path",
                exc,
            )
        if model is None:
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
        return (model, clip)


HSWQLoadConvRotINT8SAM3 = HSWQSAM3Loader
