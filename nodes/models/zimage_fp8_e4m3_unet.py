from __future__ import annotations

import logging
import os

import torch
import comfy.utils
from comfy import sd as comfy_sd

from ...nodes.utils import get_filename_list, get_full_path_or_raise


log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class HSWQZImageFP8E4M3UNetLoader:
    """
    UNet loader for Zimage FP8 E4M3 (HSWQ).

    - No dependency on Nunchaku classes/modules
    - Passes state_dict straight to ComfyUI `sd.load_diffusion_model_state_dict`
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    get_filename_list("diffusion_models"),
                    {
                        "tooltip": (
                            "HSWQ UNet model for Zimage FP8 E4M3 "
                            "(.safetensors / .ckpt readable by ComfyUI UNet loader)"
                        )
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "HSWQ-ussoewwin"
    TITLE = "HSWQ ConvRot INT8/ConvRot NVFP4 UNet Loader"

    def load_model(
        self,
        model_name: str,
        **kwargs,
    ):
        # Resolve path under diffusion_models
        model_path = get_full_path_or_raise("diffusion_models", model_name)

        # Load state_dict + metadata (do not mutate contents)
        sd, metadata = comfy.utils.load_torch_file(
            model_path,
            return_metadata=True,
        )

        # Options for ComfyUI core UNet loader
        model_options = {
            # Assume FP8 E4M3
            "dtype": torch.float8_e4m3fn,
            # Flag so comfy.ops.pick_operations() selects fp8_ops
            "fp8_optimizations": True,
        }

        logger.info(
            "HSWQ Zimage FP8 E4M3 UNet: loading via comfy.sd.load_diffusion_model_state_dict (%s)",
            model_path,
        )

        # Pass state_dict to the standard UNet loader. No Nunchaku involvement.
        model = comfy_sd.load_diffusion_model_state_dict(
            sd,
            model_options=model_options,
            metadata=metadata,
        )

        return (model,)

