# Krea2 ConvRot NVFP4 runtime pieces still shared by the Krea2 ConvRot INT8
# load path (patches/comfy_quant_int8.py imports apply_comfy_quant_nvfp4_patches
# + nvfp4_lora_bake + nvfp4_forward from here). The NVFP4-only loader
# (load_unet.py / nvfp4_comfy_parity.py) has been removed.
# Never edit ComfyUI-master; all logic lives under this package.

from .int8_lora_bake import (
    install_krea2_nvfp4_lora_bake,
    uninstall_krea2_nvfp4_lora_bake,
)

__all__ = [
    "install_krea2_nvfp4_lora_bake",
    "uninstall_krea2_nvfp4_lora_bake",
]