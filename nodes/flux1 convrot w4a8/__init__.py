"""Flux1 ConvRot W4A8 — core owns detect/load/GEMM, this package owns LoRA.

Public surface (no side effects on import; node wiring calls ``install_*``):
  apply_w4a8_patches      arm the residual forward + Dynamic bake
  load_unet_w4a8_weight_dtype  load a pack through comfy.sd
  install_flux1_w4a8_unet_dispatch  route a loader node's weight_dtype
  purge_w4a8_patches      peel everything (purge path)
"""

from .flux1_comfy_quant_w4a8 import (
    apply_comfy_quant_w4a8_patches,
    is_armed,
    uninstall_comfy_quant_w4a8_patches,
)
from .flux1_w4a8_forward import (
    clear_w4a8_residuals,
    log_w4a8_lora_bake_evidence,
    w4a8_forward_stats,
)
from .flux1_w4a8_load_unet import (
    W4A8_WEIGHT_DTYPE,
    apply_w4a8_patches,
    install_flux1_w4a8_unet_dispatch,
    load_unet_w4a8_weight_dtype,
    purge_w4a8_patches,
)
from .w4a8_lora_bake import (
    bake_w4a8_patches_on_patcher,
    install_w4a8_lora_bake,
    w4a8_lora_bake_counters,
)

__all__ = [
    "W4A8_WEIGHT_DTYPE",
    "apply_comfy_quant_w4a8_patches",
    "apply_w4a8_patches",
    "bake_w4a8_patches_on_patcher",
    "clear_w4a8_residuals",
    "install_flux1_w4a8_unet_dispatch",
    "install_w4a8_lora_bake",
    "is_armed",
    "load_unet_w4a8_weight_dtype",
    "log_w4a8_lora_bake_evidence",
    "purge_w4a8_patches",
    "uninstall_comfy_quant_w4a8_patches",
    "w4a8_forward_stats",
    "w4a8_lora_bake_counters",
]
