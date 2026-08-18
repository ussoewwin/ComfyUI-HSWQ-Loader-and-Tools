"""Krea2 DiT UNet load - ConvRot NVFP4 (TC) + INT8 ConvRot (ComfyUI core).

Krea2 ConvRot NVFP4 is **not** the Z Image parity path and **not** the SDXL
Checkpoint Loader path. It uses the Krea2 package-local TC forward
(``nodes/krea2_convrot_nvfp4/comfy_quant_nvfp4.apply_comfy_quant_nvfp4_patches``)
plus the shared INT8 ConvRot (``patches/comfy_quant_int8``).

All logic lives under ``nodes/krea2_convrot_nvfp4``. Does not edit
``nodes/nvfp4`` (SDXL TC) or ``nodes/zimage_nvfp4`` (Z Image parity).
"""
from __future__ import annotations

import logging
import sys

# Krea2 UNet dropdown ONLY - never share SDXL "ConvRot NVFP4" (Checkpoint
# Loader) or Z Image "Z Image ConvRot NVFP4" (separate being / separate stack).
KREA2_NVFP4_WEIGHT_DTYPE = "Krea2 ConvRot NVFP4"

_DISPATCH_INSTALLED = False
_INSTALL_HOOKED = False

logger = logging.getLogger(__name__)


def load_unet_nvfp4_weight_dtype(unet_name, weight_dtype):
    """Load Krea2 DiT UNet with ConvRot NVFP4 (TC) + INT8 ConvRot (core)."""
    import folder_paths
    import comfy.sd

    from .comfy_quant_nvfp4 import apply_comfy_quant_nvfp4_patches
    from .nvfp4_forward import reset_nvfp4_forward_stats, reset_nvfp4_lora_log_counters
    from .nvfp4_lora_bake import (
        install_krea2_nvfp4_lora_bake,
        reset_krea2_nvfp4_lora_bake_log_counters,
    )
    from ...patches.comfy_quant_int8 import (
        _int8_quant_conv_scope,
        apply_comfy_quant_int8_patches,
        reset_int8_lora_log_counters,
        summarize_int8_lora_capability,
    )

    unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
    if not apply_comfy_quant_nvfp4_patches():
        raise RuntimeError(
            "[HSWQ NVFP4] Krea2 UNet requires NVFP4 detect/load/TC forward "
            "(krea2_convrot_nvfp4.comfy_quant_nvfp4.apply_comfy_quant_nvfp4_patches)"
        )
    # Mixed pack: Linear=nvfp4 TC, INT8 protect = ComfyUI core ConvRot path.
    apply_comfy_quant_int8_patches()
    # After INT8 Dynamic bake wrap: force Krea2 ConvRot NVFP4 LoRA bake
    # outermost (NVFP4 pass + INT8 leftover pass), ZI-style two-pass bake.
    if not install_krea2_nvfp4_lora_bake(force=True):
        raise RuntimeError(
            "[HSWQ NVFP4] Krea2 UNet requires Dynamic ConvRot NVFP4 LoRA bake "
            "(nvfp4_lora_bake.install_krea2_nvfp4_lora_bake)"
        )
    reset_int8_lora_log_counters()
    reset_nvfp4_forward_stats()
    reset_nvfp4_lora_log_counters()
    reset_krea2_nvfp4_lora_bake_log_counters()
    logging.info(
        "[HSWQ NVFP4] Loading Krea2 UNet (ConvRot NVFP4 TC + INT8 ConvRot core): "
        "%s (weight_dtype=%s)",
        unet_name,
        weight_dtype,
    )
    print(
        f"[HSWQ NVFP4] Loading Krea2 UNet (ConvRot NVFP4 / TC): {unet_name}",
        flush=True,
    )
    with _int8_quant_conv_scope():
        model = comfy.sd.load_diffusion_model(unet_path, model_options={})
    # Stamp: only packs loaded here may fire the Krea2 NVFP4 LoRA bake hook
    # (shared _hswq_nvfp4_convrot flags alone must not - that would mix ZI /
    # SDXL models into this bake). ``model`` is the ModelPatcher; stamp BOTH
    # the patcher and the inner BaseModel - LoRA nodes clone the patcher and
    # only the inner-model stamp survives the clone (shared object, ZI-style).
    model._hswq_krea2_nvfp4_pack = True
    inner_model = getattr(model, "model", None)
    if inner_model is not None:
        inner_model._hswq_krea2_nvfp4_pack = True
    summarize_int8_lora_capability(model)
    return (model,)


def install_krea2_nvfp4_unet_dispatch(node_class_mappings=None) -> bool:
    """Wrap HSWQFP8E4M3UNetLoader for weight_dtype "Krea2 ConvRot NVFP4".

    Must run *after* ``install_int8_option_dispatch``: mixed NVFP4 packs also
    contain ``int8_tensorwise`` layers, so INT8-only auto-detect would otherwise
    steal the load without NVFP4 Linear patches. INT8 ConvRot stays core.
    """
    global _DISPATCH_INSTALLED
    if node_class_mappings is None:
        wrapped_any = False
        for _n, mod in list(sys.modules.items()):
            mappings = getattr(mod, "NODE_CLASS_MAPPINGS", None)
            if isinstance(mappings, dict) and install_krea2_nvfp4_unet_dispatch(mappings):
                wrapped_any = True
        return wrapped_any

    if not isinstance(node_class_mappings, dict):
        return False

    from .nvfp4_conf import checkpoint_looks_like_comfy_quant_nvfp4

    unet_cls = node_class_mappings.get("HSWQFP8E4M3UNetLoader")
    if unet_cls is None:
        return False
    if getattr(unet_cls, "_hswq_krea2_nvfp4_dispatch", False):
        _DISPATCH_INSTALLED = True
        return True

    _fp8 = frozenset({"fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"})
    _prev = unet_cls.load_unet

    def load_unet(self, unet_name, weight_dtype):
        if weight_dtype in _fp8:
            return _prev(self, unet_name, weight_dtype)
        if weight_dtype == KREA2_NVFP4_WEIGHT_DTYPE:
            return load_unet_nvfp4_weight_dtype(unet_name, weight_dtype)
        import folder_paths

        if weight_dtype == "default":
            unet_path = folder_paths.get_full_path_or_raise(
                "diffusion_models", unet_name
            )
            if checkpoint_looks_like_comfy_quant_nvfp4(unet_path):
                return load_unet_nvfp4_weight_dtype(unet_name, weight_dtype)
        # Never treat SDXL "ConvRot NVFP4" / Z Image "Z Image ConvRot NVFP4" as
        # Krea2 - each is a different being. int8_tensorwise / other fall through
        # to INT8 dispatch / original (core ConvRot).
        return _prev(self, unet_name, weight_dtype)

    unet_cls.load_unet = load_unet
    unet_cls._hswq_krea2_nvfp4_dispatch = True  # type: ignore[attr-defined]
    _DISPATCH_INSTALLED = True
    print(
        f"[HSWQ NVFP4] Krea2 UNet dispatch: {KREA2_NVFP4_WEIGHT_DTYPE!r} "
        "-> nodes.krea2_convrot_nvfp4 (TC; not SDXL / Z Image ConvRot NVFP4)",
        flush=True,
    )
    return True


def _hook_nvfp4_install_for_unet_dispatch() -> None:
    """When package ``__init__`` runs SDXL NVFP4 install, also wrap Krea2 UNet."""
    global _INSTALL_HOOKED
    if _INSTALL_HOOKED:
        return
    for name, mod in list(sys.modules.items()):
        if not (
            name.endswith("nodes.nvfp4.comfy_quant_nvfp4")
            or name.endswith(".comfy_quant_nvfp4")
            or name == "comfy_quant_nvfp4"
        ):
            continue
        prev = getattr(mod, "install_nvfp4_option_dispatch", None)
        if prev is None or getattr(prev, "_hswq_krea2_unet_hook", False):
            continue

        def install_nvfp4_option_dispatch(node_class_mappings, _prev=prev):
            ok = _prev(node_class_mappings)
            install_krea2_nvfp4_unet_dispatch(node_class_mappings)
            return ok

        install_nvfp4_option_dispatch._hswq_krea2_unet_hook = True  # type: ignore[attr-defined]
        mod.install_nvfp4_option_dispatch = install_nvfp4_option_dispatch
        _INSTALL_HOOKED = True
        return


# Import-time: hook SDXL install so UNet wrap runs after INT8.
_hook_nvfp4_install_for_unet_dispatch()
install_krea2_nvfp4_unet_dispatch()