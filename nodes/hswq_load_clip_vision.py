"""HSWQ Load CLIP Vision.

Port of ComfyUI's stock "Load CLIP Vision" node (nodes.py ``CLIPVisionLoader``)
into the HSWQ loader package, with the ``HSWQ`` name prefix. Float packs (the
default ``models/clip_vision`` content) load byte-identically to the stock node:

- Widget: ``clip_name`` from the ``models/clip_vision`` folder list.
- Loads the file through ``comfy.clip_vision.load`` (the stock key-layout
  auto-detection: OpenAI ``visual.`` prefix conversion, ViT-g/H, SigLIP/SigLIP2,
  ViT-L-336 (+LLaVA projector), ViT-L, DINOv2 giant/large, DINOv3).
- Returns the ``CLIP_VISION`` object (a ``comfy.clip_vision.ClipVisionModel``:
  model + ``CoreModelPatcher`` on the text-encoder device) for downstream
  ``CLIP Vision Encode`` / IPAdapter / Style Model consumers.

ConvRot INT8 packs take a second path. ``comfy.clip_vision`` builds every tower
with ``comfy.ops.manual_cast``, which never calls ``_load_quantized_module``, so a
``comfy_quant`` layer would keep its int8 codes in a float ``Parameter`` and the
tower would run on those codes. When a file declares ConvRot INT8 layers, the load
goes through ``hswq.clip_vision_convrot_int8`` instead, which installs the
quantised op set on the declared layers only. Files without ``comfy_quant`` use
``comfy.clip_vision.load`` exactly as the stock node does.
"""
from __future__ import annotations

import logging

import folder_paths

# Absolute imports cannot reach this package's ``hswq`` subpackage: ComfyUI loads
# custom nodes under a synthetic module name and does not add the folder to
# ``sys.path`` (``prestartup_script.py`` also forbids inserting it). Use a
# package-relative import, same convention as ``__init__.py``
# (``from .hswq.zimage_fp8... import``).
try:
    from ..hswq import clip_vision_convrot_int8 as _cv_convrot_int8
except ImportError:  # direct-script runs (unit tests) with the repo root on sys.path
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
    from hswq import clip_vision_convrot_int8 as _cv_convrot_int8

clip_vision_convrot_int8 = _cv_convrot_int8

logger = logging.getLogger(__name__)


class HSWQLoadClipVision:
    """Load a CLIP Vision model from models/clip_vision (stock-compatible)."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name": (
                    folder_paths.get_filename_list("clip_vision"),
                ),
            }
        }

    RETURN_TYPES = ("CLIP_VISION",)
    FUNCTION = "load_clip"
    CATEGORY = "loaders"
    TITLE = "HSWQ Load CLIP Vision"

    def load_clip(self, clip_name):
        clip_path = folder_paths.get_full_path_or_raise("clip_vision", clip_name)
        clip_vision = clip_vision_convrot_int8.load(clip_path)
        if clip_vision is None:
            raise RuntimeError(
                "ERROR: clip vision file is invalid and does not contain a "
                "valid vision model."
            )
        return (clip_vision,)
