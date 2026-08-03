"""Z Image / ZIT UNet load — ConvRot NVFP4 (parity) + INT8 ConvRot (ComfyUI core).

Z Image ConvRot NVFP4 is **not** the SDXL TC Linear.forward path.
``hswq/benchmark/zi_convrot_nvfp4_bench.py`` ``require_convrot_parity_forward``:
TC wrap (``_hswq_nvfp4_full_forward``) destroys SSIM; need stock GEMM + online
act rotate (``_hswq_nvfp4_convrot_parity``) via ``apply_nvfp4_comfy_parity``.

  - Arm detect/load/LoRA bake with ``zi_comfy_quant_nvfp4.apply_comfy_quant_nvfp4_patches``,
    then **replace** Linear.forward with comfy_parity (not stacked double-rotate).
  - INT8 ConvRot: ComfyUI core / kitchen as-is. ``apply_comfy_quant_int8_patches``
    only for int8_tensorwise load.

All logic under ``nodes/zimage_nvfp4``. Does not edit ``nodes/nvfp4`` (SDXL TC).
"""
from __future__ import annotations

import logging
import sys

# ZI/Krea UNet dropdown ONLY — never share the SDXL Checkpoint Loader string.
# SDXL uses nodes/nvfp4 NVFP4_WEIGHT_DTYPE == "ConvRot NVFP4" (separate being).
ZI_NVFP4_WEIGHT_DTYPE = "Z Image ConvRot NVFP4"

_DISPATCH_INSTALLED = False
_INSTALL_HOOKED = False

logger = logging.getLogger(__name__)


def apply_nvfp4_patches() -> None:
    """Arm Z Image ConvRot NVFP4 (parity) + INT8 load (core ConvRot)."""
    from .zi_comfy_quant_nvfp4 import apply_comfy_quant_nvfp4_patches
    from ...patches.comfy_quant_int8 import apply_comfy_quant_int8_patches
    from .nvfp4_comfy_parity import (
        apply_nvfp4_comfy_parity,
        require_convrot_parity_forward,
    )
    from .nvfp4_lora_bake import install_zimage_nvfp4_lora_bake

    if not apply_comfy_quant_nvfp4_patches():
        raise RuntimeError(
            "[HSWQ NVFP4] Z Image: apply_comfy_quant_nvfp4_patches failed "
            "(detect/load/LoRA bake required; see nodes/zimage_nvfp4/zi_comfy_quant_nvfp4)"
        )
    # Replace TC Linear.forward with stock GEMM + act rotate (not double-rotate).
    if not apply_nvfp4_comfy_parity():
        raise RuntimeError(
            "[HSWQ NVFP4] Z Image: apply_nvfp4_comfy_parity failed "
            "(stock GEMM + act rotate required; TC destroys SSIM)"
        )
    require_convrot_parity_forward()
    # INT8 tensorwise load only — ConvRot INT8 remains ComfyUI core / kitchen.
    apply_comfy_quant_int8_patches()
    # After INT8 Dynamic bake wrap: force ConvRot NVFP4 LoRA bake outermost.
    if not install_zimage_nvfp4_lora_bake(force=True):
        raise RuntimeError(
            "[HSWQ NVFP4] Z Image: install_zimage_nvfp4_lora_bake failed "
            "(Dynamic ConvRot NVFP4 LoRA bake required)"
        )
    print(
        "  [HSWQ NVFP4] Z Image: ConvRot NVFP4 (comfy_parity) + INT8 ConvRot "
        "+ Dynamic NVFP4 LoRA bake",
        flush=True,
    )


def _ensure_dynamic_load_bake_wrap() -> None:
    """Re-arm ZI NVFP4 bake wrap if MultiGPU/INT8 overwrote Dynamic.load or load_models_gpu."""
    from .nvfp4_lora_bake import (
        _BAKE_HOOK_VER,
        install_load_models_gpu_bake_hook,
        install_zimage_nvfp4_lora_bake,
    )

    try:
        import comfy.model_management as mm
        import comfy.model_patcher as mp
    except ImportError:
        return
    Dynamic = getattr(mp, "ModelPatcherDynamic", None)
    need_force = True
    if Dynamic is not None:
        cur = getattr(Dynamic, "load", None)
        if (
            cur is not None
            and getattr(cur, "_hswq_zi_nvfp4_lora_bake", False)
            and getattr(cur, "_hswq_zi_nvfp4_lora_bake_ver", 0) >= _BAKE_HOOK_VER
        ):
            need_force = False
    if need_force:
        install_zimage_nvfp4_lora_bake(force=True)
    gpu = getattr(mm, "load_models_gpu", None)
    if (
        gpu is None
        or not getattr(gpu, "_hswq_zi_nvfp4_gpu_bake", False)
        or getattr(gpu, "_hswq_zi_nvfp4_gpu_bake_ver", 0) < _BAKE_HOOK_VER
    ):
        install_load_models_gpu_bake_hook(force=True)
    else:
        install_load_models_gpu_bake_hook(force=False)


def load_unet_nvfp4_weight_dtype(unet_name, weight_dtype):
    """Load Z Image / ZIT UNet with ConvRot NVFP4 parity (not SDXL TC forward)."""
    import folder_paths
    import comfy.sd

    from .zi_comfy_quant_nvfp4 import apply_comfy_quant_nvfp4_patches
    from .zi_nvfp4_forward import reset_nvfp4_lora_log_counters
    from ...patches.comfy_quant_int8 import (
        _int8_quant_conv_scope,
        apply_comfy_quant_int8_patches,
        reset_int8_lora_log_counters,
        summarize_int8_lora_capability,
    )
    from .nvfp4_comfy_parity import (
        apply_nvfp4_comfy_parity,
        require_convrot_parity_forward,
    )
    from .nvfp4_lora_bake import (
        install_zimage_nvfp4_lora_bake,
        reset_zimage_nvfp4_lora_bake_log_counters,
    )

    unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
    if not apply_comfy_quant_nvfp4_patches():
        raise RuntimeError(
            "[HSWQ NVFP4] Z Image UNet requires NVFP4 detect/load/LoRA bake "
            "(zi_comfy_quant_nvfp4.apply_comfy_quant_nvfp4_patches)"
        )
    if not apply_nvfp4_comfy_parity():
        raise RuntimeError(
            "[HSWQ NVFP4] Z Image UNet requires comfy_parity "
            "(stock GEMM + act rotate; not HSWQ TC Linear.forward)"
        )
    require_convrot_parity_forward()
    # Mixed pack: Linear=nvfp4 parity, INT8 = ComfyUI core ConvRot path.
    apply_comfy_quant_int8_patches()
    if not install_zimage_nvfp4_lora_bake(force=True):
        raise RuntimeError(
            "[HSWQ NVFP4] Z Image UNet requires Dynamic ConvRot NVFP4 LoRA bake"
        )
    _ensure_dynamic_load_bake_wrap()
    reset_int8_lora_log_counters()
    reset_nvfp4_lora_log_counters()
    reset_zimage_nvfp4_lora_bake_log_counters()
    logging.info(
        "[HSWQ NVFP4] Loading UNet (ConvRot NVFP4 comfy_parity + INT8 ConvRot ComfyUI core): "
        "%s (weight_dtype=%s)",
        unet_name,
        weight_dtype,
    )
    print(
        f"[HSWQ NVFP4] Loading UNet (ConvRot NVFP4 / comfy_parity): {unet_name}",
        flush=True,
    )
    with _int8_quant_conv_scope():
        model = comfy.sd.load_diffusion_model(unet_path, model_options={})
    summarize_int8_lora_capability(model)
    return (model,)


def _attach_to_comfy_quant_module() -> None:
    """Expose this loader on comfy_quant_nvfp4 so prestartup can bind it."""
    for name, mod in list(sys.modules.items()):
        if not (
            name.endswith("nodes.nvfp4.comfy_quant_nvfp4")
            or name.endswith(".comfy_quant_nvfp4")
            or name == "comfy_quant_nvfp4"
        ):
            continue
        cur = getattr(mod, "load_unet_nvfp4_weight_dtype", None)
        if cur is None or cur is load_unet_nvfp4_weight_dtype:
            mod.load_unet_nvfp4_weight_dtype = load_unet_nvfp4_weight_dtype


def install_zimage_nvfp4_unet_dispatch(node_class_mappings=None) -> bool:
    """Wrap HSWQFP8E4M3UNetLoader for weight_dtype ConvRot NVFP4.

    Must run *after* ``install_int8_option_dispatch``: mixed NVFP4 packs also
    contain ``int8_tensorwise`` layers, so INT8-only auto-detect would otherwise
    steal the load without NVFP4 Linear patches. INT8 ConvRot stays core.
    """
    global _DISPATCH_INSTALLED
    if node_class_mappings is None:
        wrapped_any = False
        for _n, mod in list(sys.modules.items()):
            mappings = getattr(mod, "NODE_CLASS_MAPPINGS", None)
            if isinstance(mappings, dict) and install_zimage_nvfp4_unet_dispatch(mappings):
                wrapped_any = True
        return wrapped_any

    if not isinstance(node_class_mappings, dict):
        return False

    from ..nvfp4.nvfp4_conf import checkpoint_looks_like_comfy_quant_nvfp4

    unet_cls = node_class_mappings.get("HSWQFP8E4M3UNetLoader")
    if unet_cls is None:
        return False
    if getattr(unet_cls, "_hswq_zi_nvfp4_dispatch", False):
        _DISPATCH_INSTALLED = True
        return True

    _fp8 = frozenset({"fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"})
    _prev = unet_cls.load_unet

    def load_unet(self, unet_name, weight_dtype):
        _ensure_dynamic_load_bake_wrap()
        if weight_dtype in _fp8:
            return _prev(self, unet_name, weight_dtype)
        if weight_dtype == ZI_NVFP4_WEIGHT_DTYPE:
            return load_unet_nvfp4_weight_dtype(unet_name, weight_dtype)
        import folder_paths

        if weight_dtype == "default":
            unet_path = folder_paths.get_full_path_or_raise(
                "diffusion_models", unet_name
            )
            if checkpoint_looks_like_comfy_quant_nvfp4(unet_path):
                return load_unet_nvfp4_weight_dtype(unet_name, weight_dtype)
        # Never treat SDXL's "ConvRot NVFP4" string as ZI — different being.
        # int8_tensorwise / other: leave to INT8 dispatch / original (core ConvRot).
        return _prev(self, unet_name, weight_dtype)

    unet_cls.load_unet = load_unet
    unet_cls._hswq_zi_nvfp4_dispatch = True  # type: ignore[attr-defined]
    _DISPATCH_INSTALLED = True
    print(
        f"[HSWQ NVFP4] Z Image UNet dispatch: {ZI_NVFP4_WEIGHT_DTYPE!r} "
        "-> nodes.zimage_nvfp4 (comfy_parity; not SDXL ConvRot NVFP4)",
        flush=True,
    )
    return True


def _hook_nvfp4_install_for_unet_dispatch() -> None:
    """When package ``__init__`` runs SDXL NVFP4 install, also wrap Z Image UNet."""
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
        if prev is None or getattr(prev, "_hswq_zi_unet_hook", False):
            continue

        def install_nvfp4_option_dispatch(node_class_mappings, _prev=prev):
            ok = _prev(node_class_mappings)
            install_zimage_nvfp4_unet_dispatch(node_class_mappings)
            return ok

        install_nvfp4_option_dispatch._hswq_zi_unet_hook = True  # type: ignore[attr-defined]
        mod.install_nvfp4_option_dispatch = install_nvfp4_option_dispatch
        _INSTALL_HOOKED = True
        return


# Import-time: register on comfy_quant; hook SDXL install so UNet wrap runs after INT8.
_attach_to_comfy_quant_module()
_hook_nvfp4_install_for_unet_dispatch()
install_zimage_nvfp4_unet_dispatch()
