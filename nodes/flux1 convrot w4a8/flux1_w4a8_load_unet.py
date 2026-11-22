"""Flux1 / FLUX-family UNet load for ConvRot W4A8 packs (asym_w4a8_int8).

Scope of this package (measured against ComfyUI core):

  - detect + load + GEMM: **core**. ``comfy/ops.py`` grouped-int8 branch reads the
    pack sidecars (``weight_s_rel`` / ``weight_s_channel`` / ``weight_codebook``),
    checks the packed width against ``_orig_shape`` and builds
    ``AsymW4A8Int8Layout``. Nothing is re-implemented here.
  - LoRA on a 4-bit pack: **this package**. Core has no float path that survives
    16 code levels, so patches become forward-time factorised residuals.
  - hybrid packs (W4A8 + ``int8_tensorwise`` protect layers): the INT8 layers
    stay on the existing INT8 stack in ``patches/comfy_quant_int8``.

No node registration happens at import time; the dispatch installer is a plain
function the node wiring calls when it is ready.
"""
from __future__ import annotations

import logging

from .flux1_comfy_quant_w4a8 import (
    apply_comfy_quant_w4a8_patches,
    is_armed,
    uninstall_comfy_quant_w4a8_patches,
)
from .w4a8_conf import (
    checkpoint_looks_like_comfy_quant_w4a8,
    find_packed_shape_critical_key,
)
from .w4a8_lora_bake import (
    install_w4a8_lora_bake,
    reset_w4a8_lora_bake_log_counters,
    uninstall_w4a8_lora_bake,
    w4a8_lora_bake_counters,
)

logger = logging.getLogger(__name__)

W4A8_WEIGHT_DTYPE = "Flux1 ConvRot W4A8"
_DISPATCH_INSTALLED = False

__all__ = [
    "W4A8_WEIGHT_DTYPE",
    "apply_w4a8_patches",
    "load_unet_w4a8_weight_dtype",
    "install_flux1_w4a8_unet_dispatch",
    "purge_w4a8_patches",
]


def _console(msg: str) -> None:
    print(msg, flush=True)
    logger.info(msg)


def _apply_int8_patches_if_present() -> bool:
    """Hybrid packs carry int8_tensorwise protect layers; keep the INT8 stack on."""
    try:
        from ...patches.comfy_quant_int8 import apply_comfy_quant_int8_patches
    except Exception as e:  # noqa: BLE001 - optional stack
        logger.debug("[HSWQ W4A8] INT8 stack unavailable: %s", e)
        return False
    try:
        return bool(apply_comfy_quant_int8_patches())
    except Exception as e:  # noqa: BLE001
        logger.warning("[HSWQ W4A8] apply_comfy_quant_int8_patches failed: %s", e)
        return False


def apply_w4a8_patches(unet_path=None) -> None:
    """Arm the W4A8 forward + LoRA bake. Raises when the pack violates the layout."""
    if unet_path:
        bad = _packed_shape_critical_key_of(unet_path)
        if bad is not None:
            raise RuntimeError(
                f"[HSWQ W4A8] {bad} is 4-bit packed. ComfyUI's FLUX detector reads "
                "shape[1] of img_in / txt_in / vector_in to size the architecture, "
                "so these keys must stay in their base dtype (the Flux1 converter "
                "keeps them out of the pack)"
            )
    # Order matters: ``apply_comfy_quant_int8_patches`` assigns
    # ``ops.mixed_precision_ops`` wholesale (patches/comfy_quant_int8.py), so a
    # W4A8 wrap installed first is replaced and its Linear forward never gets the
    # residual. Hybrid packs carry int8_tensorwise protect layers, so the INT8
    # stack is armed first and the W4A8 wrap is installed on top of it.
    int8_armed = _apply_int8_patches_if_present()
    if not apply_comfy_quant_w4a8_patches():
        raise RuntimeError(
            "[HSWQ W4A8] apply_comfy_quant_w4a8_patches failed "
            "(mixed_precision_ops forward wrap required for LoRA residuals)"
        )
    if not install_w4a8_lora_bake(force=True):
        raise RuntimeError(
            "[HSWQ W4A8] install_w4a8_lora_bake failed "
            "(Dynamic patcher bake required: 4-bit codes round LoRA deltas away)"
        )
    _console(
        "[HSWQ W4A8] ready: core grouped-int8 GEMM + W4A8 LoRA residuals"
        + (" + INT8 protect layers" if int8_armed else "")
    )


def _packed_shape_critical_key_of(unet_path: str):
    """Header-only probe: return a packed shape-critical key, or None."""
    try:
        from safetensors import safe_open
    except ImportError:
        return None
    try:
        with safe_open(unet_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for rel in ("img_in.weight", "txt_in.weight", "vector_in.in_layer.weight"):
                for prefix in ("model.diffusion_model.", "diffusion_model.", ""):
                    key = f"{prefix}{rel}"
                    if key not in keys:
                        continue
                    cq = key[: -len("weight")] + "comfy_quant"
                    if cq in keys:
                        conf = f.get_tensor(cq)
                        if checkpoint_looks_like_comfy_quant_w4a8({cq: conf}):
                            return key
    except Exception as e:  # noqa: BLE001 - probe must never raise
        logger.warning("[HSWQ W4A8] shape-critical probe failed: %s", e)
        return None
    return None


def load_unet_w4a8_weight_dtype(unet_name, weight_dtype, **kwargs):
    """Load a ConvRot W4A8 UNet pack through comfy.sd with the residual arm."""
    import folder_paths
    import comfy.sd

    from .flux1_w4a8_forward import reset_w4a8_forward_stats

    unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
    apply_w4a8_patches(unet_path)
    reset_w4a8_forward_stats()
    reset_w4a8_lora_bake_log_counters()
    _console(f"[HSWQ W4A8] Loading UNet (ConvRot W4A8): {unet_name}")
    model = comfy.sd.load_diffusion_model(unet_path, model_options={})
    try:
        model._hswq_w4a8_pack = True
        if getattr(model, "model", None) is not None:
            model.model._hswq_w4a8_pack = True
    except Exception:  # noqa: BLE001 - tagging is advisory
        pass
    logger.info(
        "[HSWQ W4A8] loaded %s (weight_dtype=%s, bake=%s)",
        unet_name,
        weight_dtype,
        w4a8_lora_bake_counters(),
    )
    return (model,)


def install_flux1_w4a8_unet_dispatch(node_class_mappings=None) -> bool:
    """Wrap a loader class so ``W4A8_WEIGHT_DTYPE`` / W4A8 packs route here.

    Node wiring decides which mapping to pass; nothing runs at import time.
    """
    global _DISPATCH_INSTALLED
    if node_class_mappings is None:
        return False
    if not isinstance(node_class_mappings, dict):
        return False

    target_name = None
    for candidate in ("HSWQFP8E4M3UNetLoader", "UNETLoader"):
        if candidate in node_class_mappings:
            target_name = candidate
            break
    if target_name is None:
        return False

    unet_cls = node_class_mappings[target_name]
    if getattr(unet_cls, "_hswq_w4a8_dispatch", False):
        _DISPATCH_INSTALLED = True
        return True

    prev = unet_cls.load_unet
    fp8 = frozenset({"fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"})

    def load_unet(self, unet_name, weight_dtype, **kwargs):
        if weight_dtype in fp8:
            return prev(self, unet_name, weight_dtype, **kwargs)
        if weight_dtype == W4A8_WEIGHT_DTYPE:
            return load_unet_w4a8_weight_dtype(unet_name, weight_dtype)
        import folder_paths

        if weight_dtype == "default":
            unet_path = folder_paths.get_full_path_or_raise(
                "diffusion_models", unet_name
            )
            if checkpoint_looks_like_comfy_quant_w4a8(unet_path):
                return load_unet_w4a8_weight_dtype(unet_name, weight_dtype)
        return prev(self, unet_name, weight_dtype, **kwargs)

    unet_cls.load_unet = load_unet
    unet_cls._hswq_w4a8_dispatch = True  # type: ignore[attr-defined]
    _DISPATCH_INSTALLED = True
    _console(
        f"[HSWQ W4A8] UNet dispatch on {target_name}: "
        f"{W4A8_WEIGHT_DTYPE!r} / auto-detected W4A8 packs"
    )
    return True


def purge_w4a8_patches() -> bool:
    """Peel the arm + bake hooks (purge path)."""
    uninstall_comfy_quant_w4a8_patches()
    uninstall_w4a8_lora_bake()
    return True
