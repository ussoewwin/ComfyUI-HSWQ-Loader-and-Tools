"""HSWQ Load ConvRot INT8 ControlNet Model.

ComfyUI's stock ControlNet path (``controlnet_load_state_dict``) builds the
Qwen Image Fun ControlNet with ``unet_dtype = weight_dtype(sd)``; for an INT8
checkpoint that is ``torch.int8``, so ``operations.Linear(..., dtype=int8)``
crashes with "Only Tensors of floating point and complex dtype can require
gradients". It also never passes a quant-aware ops class, so even with a float
dtype the ``comfy_quant`` metadata would be ignored.

This node fixes both:

1. Force the model architecture dtype to a float type (BF16 on Ampere+,
   FP16 on Turing/older GPUs) so the module graph can be constructed.
2. Pass ``custom_operations = mixed_precision_ops({"int8_tensorwise":
   QUANT_ALGOS["int8_tensorwise"]}, compute_dtype)`` so every Linear in the graph is a
   MixedPrecisionOps Linear.  Its ``_load_from_state_dict`` consumes
   ``<layer>.comfy_quant`` / ``<layer>.weight_scale`` and attaches an INT8
   ``QuantizedTensor`` (TensorWiseINT8Layout) to the module.

Result: weights stay INT8 in VRAM (TensorWiseINT8Layout), forward uses the
comfy-kitchen ``int8_linear`` kernel (with ``convrot`` online activation
rotation for ConvRot layers). This is the same native INT8 path ComfyUI uses
for UNet/CLIP.

The output is a normal CONTROL_NET object; use it with the stock
"Apply ControlNet" node.

FLUX ControlNet note: ``comfy.controlnet.load_controlnet_state_dict`` has no
detection branch for the comfy-native ``ControlNetFlux`` layout
(``img_in`` / ``pos_embed_input.weight`` / ``controlnet_blocks.*``) - it only
knows the diffusers, InstantX (``controlnet_x_embedder.weight``) and
mistoline variants.  ConvRot INT8 FLUX ControlNets ship in exactly that
comfy-native layout, so we detect it here and build ``ControlNetFlux``
directly (mirroring ``load_controlnet_flux_instantx``).
"""
from __future__ import annotations

import json
import logging
import math
import os

import torch

import comfy.controlnet
import comfy.model_management
import comfy.utils
from comfy import ops as comfy_ops
from comfy.quant_ops import QUANT_ALGOS

from .utils import get_filename_list, get_full_path_or_raise

logger = logging.getLogger(__name__)

_CN_DIR = "controlnet"


def _decode_comfy_quant(raw) -> dict:
    try:
        return json.loads(raw.numpy().tobytes())
    except Exception:  # noqa: BLE001
        return {}


def _has_int8_comfy_quant(sd: dict) -> bool:
    """True if the checkpoint carries >=1 int8_tensorwise comfy_quant layer."""
    for key in sd.keys():
        if not key.endswith(".comfy_quant"):
            continue
        conf = _decode_comfy_quant(sd[key])
        if conf.get("format") == "int8_tensorwise":
            return True
    return False


def _get_default_compute_dtype(device: torch.device | None = None) -> torch.dtype:
    """Select BF16 on modern GPUs (Ampere/Ada/Blackwell) or FP16 on Turing/older GPUs."""
    if device is None:
        try:
            device = comfy.model_management.get_torch_device()
        except Exception:  # noqa: BLE001
            device = None
    if comfy.model_management.should_use_bf16(device=device):
        return torch.bfloat16
    return torch.float16


def _int8_mixed_precision_ops(compute_dtype: torch.dtype | None = None):
    """MixedPrecisionOps supporting int8_tensorwise (ConvRot included).

    ``pick_operations`` only returns MixedPrecisionOps when
    ``model_config.quant_config`` is set; the ControlNet loader has no model
    config here, so we build the ops class explicitly.
    """
    if compute_dtype is None:
        compute_dtype = _get_default_compute_dtype()
    quant_config = {
        "int8_tensorwise": QUANT_ALGOS["int8_tensorwise"],
    }
    return comfy_ops.mixed_precision_ops(
        quant_config,
        compute_dtype,
        full_precision_mm=False,
        disabled=[],
    )


def _looks_like_comfy_flux_controlnet(sd: dict) -> bool:
    """True for the comfy-native ``ControlNetFlux`` layout.

    Discriminators (all must hold):

    - ``img_in.weight`` (comfy naming; InstantX/diffusers use
      ``x_embedder`` / ``controlnet_x_embedder``)
    - ``pos_embed_input.weight`` (plain Linear; SD3/mmdit use
      ``pos_embed_input.proj.weight``)
    - ``controlnet_blocks.0.weight``
    - none of the classic-SD or InstantX markers
    """
    return (
        "img_in.weight" in sd
        and "controlnet_blocks.0.weight" in sd
        and "pos_embed_input.weight" in sd
        and "controlnet_x_embedder.weight" not in sd
        and "control_model.zero_convs.0.0.weight" not in sd
        and "zero_convs.0.0.weight" not in sd
    )


def _load_comfy_flux_controlnet(sd: dict, model_options: dict):
    """Build a ``ControlNetFlux`` from a comfy-native FLUX ControlNet sd.

    Mirrors ``comfy.controlnet.load_controlnet_flux_instantx``: latent-input
    ControlNet with the FLUX latent format and y/guidance extra conds.  The
    union mode embedder is only created when the sd actually carries
    ``controlnet_mode_embedder.weight`` (ConvRot INT8 union-pro models don't).
    """
    (
        model_config,
        operations,
        load_device,
        unet_dtype,
        manual_cast_dtype,
        offload_device,
    ) = comfy.controlnet.controlnet_config(sd, model_options=model_options)

    # pos_embed_input is Linear(patch*patch*latent_channels -> hidden);
    # patch size is 2, so latent channels = in_features // 4.
    control_latent_channels = sd["pos_embed_input.weight"].shape[1] // 4

    control_model = comfy.ldm.flux.controlnet.ControlNetFlux(
        latent_input=True,
        num_union_modes=0,
        control_latent_channels=control_latent_channels,
        operations=operations,
        device=offload_device,
        dtype=unet_dtype,
        **model_config.unet_config,
    )
    control_model = comfy.controlnet.controlnet_load_state_dict(
        control_model, sd
    )

    latent_format = comfy.latent_formats.Flux()
    control = comfy.controlnet.ControlNet(
        control_model,
        compression_ratio=1,
        latent_format=latent_format,
        concat_mask=False,
        load_device=load_device,
        manual_cast_dtype=manual_cast_dtype,
        extra_conds=["y", "guidance"],
    )
    return control


class HSWQControlNetLoader:
    """Load a ConvRot INT8 ControlNet keeping weights INT8 in VRAM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_net_name": (
                    get_filename_list(_CN_DIR),
                    {"tooltip": "ConvRot INT8 / standard ControlNet model (.safetensors)"},
                ),
            },
        }

    RETURN_TYPES = ("CONTROL_NET",)
    RETURN_NAMES = ("control_net",)
    FUNCTION = "load_controlnet"
    CATEGORY = "loaders"
    TITLE = "HSWQ ControlNet Loader (ConvRot INT8)"
    SEARCH_ALIASES = [
        "controlnet",
        "control net",
        "cn",
        "load controlnet",
        "controlnet loader",
        "hswq controlnet",
        "convrot controlnet",
        "int8 controlnet",
    ]

    def load_controlnet(self, control_net_name: str, **kwargs):
        ckpt_path = get_full_path_or_raise(_CN_DIR, control_net_name)

        sd = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
        compute_dtype = _get_default_compute_dtype()

        if _looks_like_comfy_flux_controlnet(sd):
            # comfy.controlnet.load_controlnet_state_dict cannot detect the
            # comfy-native ControlNetFlux layout -> build it directly.
            model_options = {}
            if _has_int8_comfy_quant(sd):
                model_options = {
                    "dtype": compute_dtype,
                    "custom_operations": _int8_mixed_precision_ops(compute_dtype=compute_dtype),
                }
                logger.info(
                    "[HSWQ INT8 CN] INT8 ComfyQuant detected: loading %s with "
                    "MixedPrecisionOps (%s, weights stay INT8 in VRAM)",
                    control_net_name,
                    compute_dtype,
                )
            else:
                logger.info(
                    "[HSWQ INT8 CN] No INT8 ComfyQuant layers, loading %s directly",
                    control_net_name,
                )
            control = _load_comfy_flux_controlnet(sd, model_options)
            if control is None:
                raise RuntimeError(
                    f"[HSWQ INT8 CN] Failed to detect controlnet in {control_net_name}"
                )
            return (control,)

        if _has_int8_comfy_quant(sd):
            model_options = {
                # Build the module graph in float; quantized weights are
                # attached by MixedPrecisionOps during state_dict load.
                "dtype": compute_dtype,
                "custom_operations": _int8_mixed_precision_ops(compute_dtype=compute_dtype),
            }
            logger.info(
                "[HSWQ INT8 CN] INT8 ComfyQuant detected: loading %s with "
                "MixedPrecisionOps (%s, weights stay INT8 in VRAM)",
                control_net_name,
                compute_dtype,
            )
        else:
            model_options = {}
            logger.info(
                "[HSWQ INT8 CN] No INT8 ComfyQuant layers, loading %s directly",
                control_net_name,
            )

        control = comfy.controlnet.load_controlnet_state_dict(
            sd, model_options=model_options
        )
        if control is None:
            raise RuntimeError(
                f"[HSWQ INT8 CN] Failed to detect controlnet in {control_net_name}"
            )
        return (control,)


HSWQLoadConvRotINT8ControlNet = HSWQControlNetLoader
