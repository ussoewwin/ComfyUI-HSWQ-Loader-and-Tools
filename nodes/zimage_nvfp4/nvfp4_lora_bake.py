"""Z Image ConvRot NVFP4 LoRA bake — Dynamic VRAM only (branch under zimage_nvfp4).

Problem (owner A/B + logs):
  Without LoRA, comfy_parity + act_rotate is fine.
  With LoRA, ModelPatcherDynamic attaches LowVramPatch (``180 patches``) and
  never calls Linear.convert_weight / set_weight. Weights stay rotated;
  LoRA deltas land in the wrong space while act_rotate still runs → salt/pepper.

INT8 already bakes via ``patches/comfy_quant_int8.py`` after Dynamic.load, but
that path **skips** NVFP4 QT on purpose (must not requant NVFP4 as INT8).
This module fills that gap for Z Image only — does **not** edit ``nodes/nvfp4``.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_BAKE_HOOK_VER = 1
_STATUS_LOGS = 0
_STATUS_LOG_MAX = 8


def _console(msg: str) -> None:
    print(msg, flush=True)
    logger.info(msg)


def _qt_is_nvfp4(weight, QuantizedTensor) -> bool:
    """True for comfy_quant / kitchen NVFP4 QuantizedTensor."""
    if weight is None:
        return False
    qt = weight
    if not isinstance(weight, QuantizedTensor):
        # Parameter / wrapper
        qt = getattr(weight, "data", weight)
        if not isinstance(qt, QuantizedTensor):
            return False
    layout = getattr(qt, "layout", None)
    if layout is None:
        layout = getattr(qt, "_layout", None)
    name = type(layout).__name__ if layout is not None else ""
    if "NVFP4" in name or "nvfp4" in name.lower():
        return True
    layout_cls = getattr(qt, "_layout_cls", None)
    if isinstance(layout_cls, str) and (
        "NVFP4" in layout_cls or "nvfp4" in layout_cls.lower()
    ):
        return True
    return False


def _module_is_nvfp4_convrot(module) -> bool:
    return bool(
        getattr(module, "_hswq_nvfp4_convrot", False)
        or getattr(module, "_hswq_nvfp4_convrot_parity", False)
    )


def _get_baked_key_set(model) -> set:
    keys = getattr(model, "_hswq_zi_nvfp4_baked_keys", None)
    if keys is None:
        keys = set()
        model._hswq_zi_nvfp4_baked_keys = keys
    return keys


def _model_has_nvfp4_convrot(model) -> bool:
    if model is None:
        return False
    try:
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return False
    for _name, module in model.named_modules():
        if not _module_is_nvfp4_convrot(module):
            continue
        w = getattr(module, "weight", None)
        if _qt_is_nvfp4(w, QuantizedTensor):
            return True
    return False


def bake_nvfp4_convrot_patches_on_dynamic_patcher(patcher, device_to) -> dict:
    """Bake LoRA into ConvRot NVFP4 Linears after ModelPatcherDynamic.load.

    Same VBAR rule as INT8 bake: clear LowVramPatch, call patch_weight_to_device
    (convert_weight unrotate → LoRA → set_weight re-rotate), keep ``_v``.
    """
    stats = {
        "baked": 0,
        "candidates": 0,
        "skipped_no_set": 0,
        "skipped_not_nvfp4": 0,
        "skipped_not_convrot": 0,
        "cleared_already": 0,
    }
    if not getattr(patcher, "patches", None):
        return stats
    try:
        import comfy.model_patcher as mp
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return stats

    already = _get_baked_key_set(patcher.model)
    uuid = getattr(patcher, "patches_uuid", None)
    prev_uuid = getattr(patcher.model, "_hswq_zi_nvfp4_baked_uuid", None)
    if prev_uuid is not None and prev_uuid != uuid:
        already.clear()

    for name, module in patcher.model.named_modules():
        keys_to_bake = []
        for param_key in ("weight", "bias"):
            key = f"{name}.{param_key}"
            if key not in patcher.patches:
                continue
            stats["candidates"] += 1
            if key in already:
                attr = param_key + "_lowvram_function"
                if getattr(module, attr, None) is not None:
                    setattr(module, attr, None)
                try:
                    del patcher.patches[key]
                except KeyError:
                    pass
                stats["cleared_already"] += 1
                continue
            if not _module_is_nvfp4_convrot(module):
                stats["skipped_not_convrot"] += 1
                continue
            weight, set_func, convert_func = mp.get_key_weight(patcher.model, key)
            if weight is None:
                continue
            if not _qt_is_nvfp4(weight, QuantizedTensor):
                stats["skipped_not_nvfp4"] += 1
                continue
            if set_func is None:
                stats["skipped_no_set"] += 1
                _console(
                    f"[HSWQ ZI NVFP4 LoRA] WARN cannot bake {key}: "
                    "NVFP4 QT but no set_weight"
                )
                continue
            keys_to_bake.append((param_key, key))

        if not keys_to_bake:
            continue

        for param_key, _key in keys_to_bake:
            if hasattr(module, param_key + "_lowvram_function"):
                setattr(module, param_key + "_lowvram_function", None)

        for _param_key, key in keys_to_bake:
            patcher.patch_weight_to_device(key, device_to=device_to)
            if key in patcher.backup:
                try:
                    del patcher.backup[key]
                except KeyError:
                    pass
            try:
                del patcher.patches[key]
            except KeyError:
                pass
            already.add(key)
            stats["baked"] += 1

    if stats["baked"] > 0:
        patcher.model._hswq_zi_nvfp4_baked_uuid = uuid

    return stats


def _dump_bake_status(stats: dict, patcher, force: bool = False) -> None:
    global _STATUS_LOGS
    if not force and _STATUS_LOGS >= _STATUS_LOG_MAX:
        return
    _STATUS_LOGS += 1
    left = len(getattr(patcher, "patches", None) or {})
    _console(
        "[HSWQ ZI NVFP4 LoRA] Dynamic.load bake "
        f"#{_STATUS_LOGS}: candidates={stats.get('candidates', 0)} "
        f"baked={stats.get('baked', 0)} "
        f"cleared_already={stats.get('cleared_already', 0)} "
        f"skip_not_convrot={stats.get('skipped_not_convrot', 0)} "
        f"skip_not_nvfp4={stats.get('skipped_not_nvfp4', 0)} "
        f"skip_no_set={stats.get('skipped_no_set', 0)} "
        f"patches_left={left}"
    )


def install_zimage_nvfp4_lora_bake() -> bool:
    """Wrap ModelPatcherDynamic.load to bake ConvRot NVFP4 LoRA (Z Image only)."""
    try:
        import comfy.model_patcher as mp
    except ImportError:
        return False

    Dynamic = getattr(mp, "ModelPatcherDynamic", None)
    if Dynamic is None:
        _console("[HSWQ ZI NVFP4 LoRA] ModelPatcherDynamic missing — bake hook skipped")
        return False
    original = getattr(Dynamic, "load", None)
    if original is None:
        return False
    if getattr(original, "_hswq_zi_nvfp4_lora_bake_ver", 0) >= _BAKE_HOOK_VER:
        return True

    # Chain on whatever is current (INT8 bake wrap or stock Dynamic.load).
    prev_load = original

    def load(
        self,
        device_to=None,
        lowvram_model_memory=0,
        force_patch_weights=False,
        full_load=False,
        dirty=False,
    ):
        result = prev_load(
            self,
            device_to=device_to,
            lowvram_model_memory=lowvram_model_memory,
            force_patch_weights=force_patch_weights,
            full_load=full_load,
            dirty=dirty,
        )
        model = getattr(self, "model", None)
        if model is None:
            return result
        if not _model_has_nvfp4_convrot(model) and not getattr(
            model, "_hswq_zi_nvfp4_baked_keys", None
        ):
            return result
        if not getattr(self, "patches", None) and not getattr(
            model, "_hswq_zi_nvfp4_baked_keys", None
        ):
            return result
        stats = bake_nvfp4_convrot_patches_on_dynamic_patcher(self, device_to=device_to)
        if (
            stats["baked"] > 0
            or stats["candidates"] > 0
            or stats["cleared_already"] > 0
            or stats["skipped_no_set"] > 0
        ):
            _dump_bake_status(stats, self, force=True)
        return result

    load._hswq_zi_nvfp4_lora_bake = True  # type: ignore[attr-defined]
    load._hswq_zi_nvfp4_lora_bake_ver = _BAKE_HOOK_VER  # type: ignore[attr-defined]
    load._hswq_zi_nvfp4_prev_dynamic_load = prev_load  # type: ignore[attr-defined]
    Dynamic.load = load
    _console(
        "[HSWQ ZI NVFP4 LoRA] Dynamic.load bake hook ON "
        "(ConvRot NVFP4 convert_weight unrotate + set_weight re-rotate)"
    )
    return True


def reset_zimage_nvfp4_lora_bake_log_counters() -> None:
    global _STATUS_LOGS
    _STATUS_LOGS = 0
