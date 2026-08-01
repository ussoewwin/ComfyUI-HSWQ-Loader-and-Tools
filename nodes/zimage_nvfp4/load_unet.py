"""Z Image / ZIT UNet load for product ConvRot NVFP4.

Patch order:
  1) apply_comfy_quant_nvfp4_patches()
  2) apply_nvfp4_comfy_parity()  — stock Comfy GEMM + online act rotate
  3) require_convrot_parity_forward()

Then ComfyUI load_diffusion_model. SDXL TC product path is not used here.
"""
from __future__ import annotations


def apply_nvfp4_patches() -> None:
    """Arm Z Image ConvRot NVFP4 inference (stock GEMM + act rotate)."""
    from ..nvfp4.comfy_quant_nvfp4 import apply_comfy_quant_nvfp4_patches
    from .nvfp4_comfy_parity import apply_nvfp4_comfy_parity
    from .require_parity import require_convrot_parity_forward

    apply_comfy_quant_nvfp4_patches()
    if not apply_nvfp4_comfy_parity():
        raise RuntimeError(
            "Z Image NVFP4: ComfyUI MixedPrecision path failed to apply "
            "(TC Linear.forward must be off; act-rotate required)"
        )
    require_convrot_parity_forward()
    print(
        "  [HSWQ NVFP4] Z Image: stock Comfy GEMM + online act rotate (x @ H)",
        flush=True,
    )


def load_unet_nvfp4_weight_dtype(unet_name, weight_dtype):
    """Load Z Image / ZIT UNet with ConvRot NVFP4 (product path)."""
    import folder_paths
    import comfy.sd

    apply_nvfp4_patches()
    unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
    print(
        f"[HSWQ NVFP4] Loading UNet (Z Image ConvRot NVFP4): "
        f"{unet_name} (weight_dtype={weight_dtype})",
        flush=True,
    )
    model = comfy.sd.load_diffusion_model(unet_path, model_options={})
    return (model,)
