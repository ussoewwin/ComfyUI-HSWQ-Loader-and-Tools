"""HSWQ Ultimate SD Upscale node. Uses usdu_bundle."""

import logging
import torch
import comfy
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy import from usdu_bundle
usdu = None
tensor_to_pil = None
pil_to_tensor = None
StableDiffusionProcessing = None
shared = None
UpscalerData = None
MODES = {}
SEAM_FIX_MODES = {}

def _ensure_imports():
    global usdu, tensor_to_pil, pil_to_tensor, StableDiffusionProcessing, shared, UpscalerData, MODES, SEAM_FIX_MODES
    if usdu is None:
        try:
            import sys
            import os
            
            current_file = os.path.realpath(__file__)
            nodes_dir = os.path.dirname(current_file)
            loader_dir = os.path.dirname(nodes_dir)
            usdu_bundle = os.path.join(loader_dir, "usdu_bundle")
            if not os.path.isdir(usdu_bundle):
                raise ImportError("usdu_bundle not found at %s" % usdu_bundle)
            original_sys_path = sys.path.copy()
            if usdu_bundle not in sys.path:
                sys.path.insert(0, usdu_bundle)
            
            modules_used = ["modules", "modules.devices", "modules.images", "modules.processing", "modules.scripts", "modules.shared", "modules.upscaler"]
            original_imported_modules = {m: sys.modules.pop(m) for m in modules_used if m in sys.modules}
            try:
                from usdu_patch import usdu  # type: ignore
                from modules.processing import StableDiffusionProcessing  # type: ignore
                import modules.shared as shared  # type: ignore
                from modules.upscaler import UpscalerData  # type: ignore
                import usdu_utils as _usdu_utils  # type: ignore
                tensor_to_pil = _usdu_utils.tensor_to_pil
                pil_to_tensor = _usdu_utils.pil_to_tensor
            finally:
                sys.path = original_sys_path
                sys.modules.update(original_imported_modules)
            
            MODES = {
                "Linear": usdu.USDUMode.LINEAR,
                "Chess": usdu.USDUMode.CHESS,
                "None": usdu.USDUMode.NONE,
            }
            
            SEAM_FIX_MODES = {
                "None": usdu.USDUSFMode.NONE,
                "Band Pass": usdu.USDUSFMode.BAND_PASS,
                "Half Tile": usdu.USDUSFMode.HALF_TILE,
                "Half Tile + Intersections": usdu.USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS,
            }
        except ImportError as e:
            logger.error("Failed to import usdu_bundle: %s", e)
            raise

MAX_RESOLUTION = 8192


def _upscale_by_options():
    return ["Auto"] + [f"{round(i * 0.05, 2):.2f}" for i in range(1, 81)]


def _to_fp32_image(image: torch.Tensor) -> torch.Tensor:
    """
    Convert image tensor to FP32 and ensure proper color range for Nunchaku SDXL.
    
    Nunchaku SDXL VAE decode output may be in a different color space or range.
    This function normalizes the input to ensure upscaler receives correct values.
    """
    t = image
    if torch.is_tensor(t):
        t = t.to(dtype=torch.float32)
        
        # Always normalize to maximize dynamic range
        # This fixes the issue where Nunchaku SDXL VAE outputs compressed range [0.15, 0.85]
        min_val = t.min().item()
        max_val = t.max().item()
        
        # Normalize to [0,1] range if there's a valid range
        if max_val > min_val:
            t = (t - min_val) / (max_val - min_val)
        else:
            t = torch.zeros_like(t)
        
        # Ensure values are in [0,1] range
        t = torch.clamp(t, 0.0, 1.0)
        t = t.contiguous()
    return t


def USDU_base_inputs():
    _ensure_imports()
    required = [
        ("image", ("IMAGE", {"tooltip": "The image to upscale."})),
        # Sampling Params
        ("model", ("MODEL", {"tooltip": "The model to use for image-to-image."})),
        ("positive", ("CONDITIONING", {"tooltip": "The positive conditioning for each tile."})),
        ("negative", ("CONDITIONING", {"tooltip": "The negative conditioning for each tile."})),
        ("vae", ("VAE", {"tooltip": "The VAE model to use for tiles."})),
        ("upscale_by", (_upscale_by_options(), {"default": "Auto", "tooltip": "Choose 'Auto' to calculate the scale from target vertical pixels, or select a fixed magnification."})),
        ("target_height", ("INT", {"default": 4320, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Target output height in pixels. Used only when upscale_by is 'Auto'."})),
        ("seed", ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "The seed to use for image-to-image."})),
        ("steps", ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1, "tooltip": "The number of steps to use for each tile."})),
        ("cfg", ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "tooltip": "The CFG scale to use for each tile."})),
        ("sampler_name", (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The sampler to use for each tile."})),
        ("scheduler", (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler to use for each tile."})),
        ("denoise", ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The denoising strength to use for each tile."})),
        # Upscale Params
        ("upscale_model", ("UPSCALE_MODEL", {"tooltip": "The upscaler model for upscaling the image."})),
        ("mode_type", (list(MODES.keys()), {"tooltip": "The tiling order to use for the redraw step."})),
        ("tile_width", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of each tile."})),
        ("tile_height", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The height of each tile."})),
        ("mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1, "tooltip": "The blur radius for the mask."})),
        ("tile_padding", ("INT", {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The padding to apply between tiles."})),
        # Seam fix params
        ("seam_fix_mode", (list(SEAM_FIX_MODES.keys()), {"tooltip": "The seam fix mode to use."})),
        ("seam_fix_denoise", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The denoising strength to use for the seam fix."})),
        ("seam_fix_width", ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of the bands used for the Band Pass seam fix mode."})),
        ("seam_fix_mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1, "tooltip": "The blur radius for the seam fix mask."})),
        ("seam_fix_padding", ("INT", {"default": 16, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The padding to apply for the seam fix tiles."})),
        # Misc
        ("force_uniform_tiles", ("BOOLEAN", {"default": True, "tooltip": "Force all tiles to be the same as the set tile size, even when tiles could be smaller. This can help prevent the model from working with irregular tile sizes."})),
        ("tiled_decode", ("BOOLEAN", {"default": False, "tooltip": "Whether to use tiled decoding when decoding tiles."})),
        ("batch_size", ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1, "tooltip": "The number of tiles to process in a batch. Higher values can reduce processing time but use more VRAM. If you get tensor size mismatch with FP8/FP4 (quantized) models, set this to 1."})),
        ("tensor_boost", ("BOOLEAN", {"default": False, "tooltip": "Enable Blackwell Per-Weight CUDA Graph Tensor Boost during USDU tile upscaling. ON raises VRAM by several GB; RTX 5090 32GB+ recommended. Keep OFF for tiled upscale."})),
    ]

    optional = []
    return required, optional


def prepare_inputs(required: list, optional: list | None = None):
    inputs: dict = {}
    if required:
        inputs["required"] = {}
        for name, t in required:
            inputs["required"][name] = t
    if optional:
        inputs["optional"] = {}
        for name, t in optional:
            inputs["optional"][name] = t
    return inputs


def remove_input(inputs: list, input_name: str):
    for i, (n, _) in enumerate(inputs):
        if n == input_name:
            del inputs[i]
            break


def rename_input(inputs: list, old_name: str, new_name: str):
    for i, (n, t) in enumerate(inputs):
        if n == old_name:
            inputs[i] = (new_name, t)
            break


class HSWQUltimateSDUpscale:
    @classmethod
    def INPUT_TYPES(s):
        try:
            _ensure_imports()
            required, optional = USDU_base_inputs()
            return prepare_inputs(required, optional)
        except Exception as e:
            logger.error(f"Failed to initialize HSWQUltimateSDUpscale INPUT_TYPES: {e}", exc_info=True)
            # Provide fallback with default modes to allow node registration even if imports fail
            # Fallback INPUT_TYPES when usdu_bundle import fails
            fallback_modes = ["Linear", "Chess", "None"] if not MODES else list(MODES.keys())
            fallback_seam_modes = ["None", "Band Pass", "Half Tile", "Half Tile + Intersections"] if not SEAM_FIX_MODES else list(SEAM_FIX_MODES.keys())
            
            required = [
                ("image", ("IMAGE", {"tooltip": "The image to upscale."})),
                ("model", ("MODEL", {"tooltip": "The model to use for image-to-image."})),
                ("positive", ("CONDITIONING", {"tooltip": "The positive conditioning for each tile."})),
                ("negative", ("CONDITIONING", {"tooltip": "The negative conditioning for each tile."})),
                ("vae", ("VAE", {"tooltip": "The VAE model to use for tiles."})),
                ("upscale_by", (_upscale_by_options(), {"default": "Auto", "tooltip": "Choose 'Auto' to calculate the scale from target vertical pixels, or select a fixed magnification."})),
                ("target_height", ("INT", {"default": 4320, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Target output height in pixels. Used only when upscale_by is 'Auto'."})),
                ("seed", ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "The seed to use for image-to-image."})),
                ("steps", ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1, "tooltip": "The number of steps to use for each tile."})),
                ("cfg", ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "tooltip": "The CFG scale to use for each tile."})),
                ("sampler_name", (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The sampler to use for each tile."})),
                ("scheduler", (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler to use for each tile."})),
                ("denoise", ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The denoising strength to use for each tile."})),
                ("upscale_model", ("UPSCALE_MODEL", {"tooltip": "The upscaler model for upscaling the image."})),
                ("mode_type", (fallback_modes, {"tooltip": "The tiling order to use for the redraw step."})),
                ("tile_width", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of each tile."})),
                ("tile_height", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The height of each tile."})),
                ("mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1, "tooltip": "The blur radius for the mask."})),
                ("tile_padding", ("INT", {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The padding to apply between tiles."})),
                ("seam_fix_mode", (fallback_seam_modes, {"tooltip": "The seam fix mode to use."})),
                ("seam_fix_denoise", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The denoising strength to use for the seam fix."})),
                ("seam_fix_width", ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of the bands used for the Band Pass seam fix mode."})),
                ("seam_fix_mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1, "tooltip": "The blur radius for the seam fix mask."})),
                ("seam_fix_padding", ("INT", {"default": 16, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The padding to apply for the seam fix tiles."})),
                ("force_uniform_tiles", ("BOOLEAN", {"default": True, "tooltip": "Force all tiles to be the same as the set tile size, even when tiles could be smaller. This can help prevent the model from working with irregular tile sizes."})),
                ("tiled_decode", ("BOOLEAN", {"default": False, "tooltip": "Whether to use tiled decoding when decoding tiles."})),
                ("batch_size", ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1, "tooltip": "The number of tiles to process in a batch. Higher values can reduce processing time but use more VRAM. If you get tensor size mismatch with FP8/FP4 (quantized) models, set this to 1."})),
                ("tensor_boost", ("BOOLEAN", {"default": False, "tooltip": "Enable Blackwell Per-Weight CUDA Graph Tensor Boost during USDU tile upscaling. ON raises VRAM by several GB; RTX 5090 32GB+ recommended. Keep OFF for tiled upscale."})),
            ]
            optional = []
            return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"
    OUTPUT_TOOLTIPS = ("The final upscaled image.",)
    DESCRIPTION = "Upscales an image and runs image-to-image on tiles from the input image."
    TITLE = "HSWQ Ultimate SD Upscale"

    def upscale(
        self,
        image,
        model,
        positive,
        negative,
        vae,
        upscale_by,
        target_height,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        upscale_model,
        mode_type,
        tile_width,
        tile_height,
        mask_blur,
        tile_padding,
        seam_fix_mode,
        seam_fix_denoise,
        seam_fix_mask_blur,
        seam_fix_width,
        seam_fix_padding,
        force_uniform_tiles,
        tiled_decode,
        batch_size=1,
        custom_sampler=None,
        custom_sigmas=None,
        tensor_boost=False,
        **kwargs,
    ):
        _ensure_imports()

        # Configure Tensor Boost toggle for USDU tile upscaling
        import os
        if isinstance(tensor_boost, bool):
            tb_enabled = tensor_boost
        else:
            tb_str = str(tensor_boost).strip().lower() if tensor_boost is not None else ""
            tb_enabled = tb_str in ("1", "true", "on", "enable", "enabled")

        if tb_enabled:
            os.environ["HSWQ_NVFP4_TENSORBOOST"] = "1"
        else:
            os.environ["HSWQ_NVFP4_TENSORBOOST"] = "0"
            try:
                from .nvfp4.nvfp4_runtime import clear_nvfp4_cudagraphs
                clear_nvfp4_cudagraphs()
            except Exception:
                pass

        # Normalize color range first to get correct input dimensions
        image = _to_fp32_image(image)
        init_img = tensor_to_pil(image, 0)

        # Resolve upscale_by: explicit numeric value takes precedence over Auto
        init_w = float(init_img.width)
        init_h = float(init_img.height)
        if upscale_by == "Auto":
            # Honor target_height: no 4.0 combo cap (fixed modes still cap at 4.0).
            scale = float(target_height) / init_h
            max_scale = min(MAX_RESOLUTION / init_w, MAX_RESOLUTION / init_h)
            if scale > max_scale:
                logger.warning(
                    "Auto upscale: target_height=%d needs scale %.3f but max is %.3f "
                    "(capped by MAX_RESOLUTION=%d).",
                    target_height, scale, max_scale, MAX_RESOLUTION,
                )
            scale = max(0.05, min(scale, max_scale))
        else:
            scale = float(upscale_by)
            scale = max(0.05, min(4.0, scale))

        # Store params
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.mask_blur = mask_blur
        self.tile_padding = tile_padding
        self.seam_fix_width = seam_fix_width
        self.seam_fix_denoise = seam_fix_denoise
        self.seam_fix_padding = seam_fix_padding
        self.seam_fix_mode = seam_fix_mode
        self.mode_type = mode_type
        self.upscale_by = scale
        self.seam_fix_mask_blur = seam_fix_mask_blur

        #
        # Set up A1111 patches
        #

        # Upscaler
        shared.sd_upscalers[0] = UpscalerData()
        shared.actual_upscaler = upscale_model

        # Set the batch of images
        shared.batch = [tensor_to_pil(image, i) for i in range(len(image))]
        # Set batch_as_tensor
        shared.batch_as_tensor = image

        # Processing
        sdprocessing = StableDiffusionProcessing(
            shared.batch[0],
            model,
            positive,
            negative,
            vae,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
            scale,
            force_uniform_tiles,
            tiled_decode,
            tile_width,
            tile_height,
            MODES[self.mode_type],
            SEAM_FIX_MODES[self.seam_fix_mode],
            custom_sampler,
            custom_sigmas,
            batch_size,
        )

        # Disable logging
        logger = logging.getLogger()
        old_level = logger.getEffectiveLevel()
        logger.setLevel(logging.CRITICAL + 1)
        try:
            # Running the script
            script = usdu.Script()
            _ = script.run(
                p=sdprocessing,
                _=None,
                tile_width=self.tile_width,
                tile_height=self.tile_height,
                mask_blur=self.mask_blur,
                padding=self.tile_padding,
                seams_fix_width=self.seam_fix_width,
                seams_fix_denoise=self.seam_fix_denoise,
                seams_fix_padding=self.seam_fix_padding,
                upscaler_index=0,
                save_upscaled_image=False,
                redraw_mode=MODES[self.mode_type],
                save_seams_fix_image=False,
                seams_fix_mask_blur=self.seam_fix_mask_blur,
                seams_fix_type=SEAM_FIX_MODES[self.seam_fix_mode],
                target_size_type=2,
                custom_width=None,
                custom_height=None,
                custom_scale=self.upscale_by,
            )

            # Return the resulting images
            images = [pil_to_tensor(img) for img in shared.batch]
            tensor = torch.cat(images, dim=0)
            return (tensor,)
        finally:
            # Restore the original logging level
            logger.setLevel(old_level)

