# Krea2 ConvRot INT8 runtime (INT8-only).
# patches/comfy_quant_int8.py imports apply_comfy_quant_int8_patches +
# install_krea2_int8_lora_bake (+ reset counters) from this package for the
# Krea2 ConvRot INT8 load. The retired Krea2 ConvRot NVFP4 loader/runtime
# (NVFP4 load routing, addmm handler, kitchen stub repair, Hadamard rotate
# wraps, packed-K detection) has been removed.
# Never edit ComfyUI-master; all logic lives under this package.

from .comfy_quant_int8_krea2 import apply_comfy_quant_int8_patches
from .int8_lora_bake import (
    install_krea2_int8_lora_bake,
    uninstall_krea2_int8_lora_bake,
)

__all__ = [
    "apply_comfy_quant_int8_patches",
    "install_krea2_int8_lora_bake",
    "uninstall_krea2_int8_lora_bake",
]