"""Z Image / ZIT UNet load for product ConvRot NVFP4.

Delegates to the *saved* product ``load_unet_nvfp4_weight_dtype`` (set by
``prestartup_script.py`` before rebind), or imports the product function when
prestartup did not run.

Product path (must not omit any of these):

  - stock Comfy GEMM + online act rotate (not TC Linear.forward)
  - INT8 protect patches (mixed packs e.g. int8protect60)
  - ConvRot Linear LoRA bake
  - disable_dynamic=True (full ModelPatcher; not AIMDO DynamicVRAM)

SDXL TC product path is not used here.
"""
from __future__ import annotations

# Set by prestartup_script before rebinding comfy_quant_nvfp4.load_unet_*.
_PRODUCT_LOAD_UNET = None


def apply_nvfp4_patches() -> None:
    """Arm Z Image ConvRot NVFP4 inference (same stack as product UNet load)."""
    from ..nvfp4.comfy_quant_nvfp4 import (
        _patch_convert_old_quants_nvfp4_kitchen_prefix,
        apply_comfy_quant_nvfp4_patches,
    )
    from ..nvfp4.nvfp4_comfy_parity import (
        apply_nvfp4_comfy_parity,
        require_convrot_parity_forward,
    )
    from ...patches.comfy_quant_int8 import apply_comfy_quant_int8_patches

    apply_comfy_quant_nvfp4_patches()
    _patch_convert_old_quants_nvfp4_kitchen_prefix()
    if not apply_nvfp4_comfy_parity():
        raise RuntimeError(
            "Z Image NVFP4: ComfyUI MixedPrecision path failed to apply "
            "(TC Linear.forward must be off; act-rotate required)"
        )
    apply_comfy_quant_int8_patches()
    _patch_convert_old_quants_nvfp4_kitchen_prefix()
    if not apply_nvfp4_comfy_parity():
        raise RuntimeError(
            "Z Image NVFP4: comfy_parity lost after INT8 patches"
        )
    require_convrot_parity_forward()
    print(
        "  [HSWQ NVFP4] Z Image: stock Comfy GEMM + act rotate + int8 protect "
        "+ ConvRot Linear LoRA bake",
        flush=True,
    )


def load_unet_nvfp4_weight_dtype(unet_name, weight_dtype):
    """Load Z Image / ZIT UNet — identical to product ConvRot NVFP4 UNet load."""
    fn = _PRODUCT_LOAD_UNET
    if fn is None:
        # Prestartup not applied: call product module attribute.
        # Prefer function object from module dict if already rebound to us.
        import inspect

        from ..nvfp4 import comfy_quant_nvfp4 as cq

        cand = cq.load_unet_nvfp4_weight_dtype
        if cand is load_unet_nvfp4_weight_dtype or (
            inspect.unwrap(cand) is load_unet_nvfp4_weight_dtype
        ):
            raise RuntimeError(
                "[HSWQ NVFP4] product load_unet missing "
                "(prestartup did not save _PRODUCT_LOAD_UNET)"
            )
        fn = cand
    return fn(unet_name, weight_dtype)
