"""Z Image mixed-pack LoRA bake — Dynamic VRAM only (branch under zimage_nvfp4).

Problem (owner A/B + logs):
  Without LoRA, comfy_parity + act_rotate is fine.
  With LoRA, ModelPatcherDynamic attaches LowVramPatch (``180 patches``).

INT8 Dynamic bake (``patches/comfy_quant_int8.py``) often does **not** fire on this
hybrid pack (no INT8 bake dump in logs), so INT8-protect keys stay as
LowVramPatch. NVFP4 ConvRot bake alone leaves ``patches_left=60`` → broken.

This module (Z Image only):
  1) Bake ConvRot NVFP4 via convert_weight unrotate + set_weight re-rotate
  2) Bake remaining QT patches that have set_weight (INT8 protect etc.)
Does **not** edit ``nodes/nvfp4`` (SDXL).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_BAKE_HOOK_VER = 2
_STATUS_LOGS = 0
_STATUS_LOG_MAX = 12
_SKIP_SAMPLE_LOGS = 0
_SKIP_SAMPLE_MAX = 6


def _console(msg: str) -> None:
    print(msg, flush=True)
    logger.info(msg)


def _qt_payload(weight, QuantizedTensor):
    if weight is None:
        return None
    if isinstance(weight, QuantizedTensor):
        return weight
    data = getattr(weight, "data", None)
    if data is not None and isinstance(data, QuantizedTensor):
        return data
    return None


def _qt_layout_name(qt) -> str:
    if qt is None:
        return ""
    layout = getattr(qt, "layout", None)
    if layout is None:
        layout = getattr(qt, "_layout", None)
    if layout is not None:
        return type(layout).__name__ or ""
    layout_cls = getattr(qt, "_layout_cls", None)
    if isinstance(layout_cls, str):
        return layout_cls
    return ""


def _qt_is_nvfp4(weight, QuantizedTensor) -> bool:
    qt = _qt_payload(weight, QuantizedTensor)
    if qt is None:
        return False
    name = _qt_layout_name(qt)
    return "NVFP4" in name or "nvfp4" in name.lower()


def _qt_is_int8_tensorwise(weight, QuantizedTensor) -> bool:
    """INT8 detect including ``_layout_cls`` string (kitchen / protect packs)."""
    qt = _qt_payload(weight, QuantizedTensor)
    if qt is None:
        return False
    name = _qt_layout_name(qt)
    return "TensorWiseINT8" in name or "int8_tensorwise" in name.lower()


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


def _bake_keys_on_module(patcher, module, keys_to_bake, device_to, already) -> int:
    """Clear LowVramPatch, patch_weight_to_device, drop backup+patches. Keep ``_v``."""
    baked = 0
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
        baked += 1
    return baked


def bake_nvfp4_convrot_patches_on_dynamic_patcher(patcher, device_to) -> dict:
    """Bake LoRA into ConvRot NVFP4 Linears after ModelPatcherDynamic.load."""
    stats = {
        "baked_nvfp4": 0,
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

    global _SKIP_SAMPLE_LOGS
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
                if _SKIP_SAMPLE_LOGS < _SKIP_SAMPLE_MAX:
                    w, _, _ = mp.get_key_weight(patcher.model, key)
                    qt = _qt_payload(w, QuantizedTensor)
                    _SKIP_SAMPLE_LOGS += 1
                    _console(
                        f"[HSWQ ZI NVFP4 LoRA] skip_not_convrot sample "
                        f"#{_SKIP_SAMPLE_LOGS}: {key} layout={_qt_layout_name(qt)!r} "
                        f"convrot={getattr(module, '_hswq_nvfp4_convrot', False)}"
                    )
                continue
            weight, set_func, _convert_func = mp.get_key_weight(patcher.model, key)
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
        stats["baked_nvfp4"] += _bake_keys_on_module(
            patcher, module, keys_to_bake, device_to, already
        )

    if stats["baked_nvfp4"] > 0:
        patcher.model._hswq_zi_nvfp4_baked_uuid = uuid

    return stats


def bake_remaining_quant_patches_on_dynamic_patcher(patcher, device_to) -> dict:
    """Bake leftover QT LoRA (INT8 protect etc.) that INT8 Dynamic bake missed."""
    stats = {
        "baked_int8": 0,
        "baked_other_qt": 0,
        "candidates": 0,
        "skipped_no_set": 0,
        "skipped_not_qt": 0,
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

    for name, module in patcher.model.named_modules():
        keys_to_bake = []
        kind = None
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
            weight, set_func, _convert_func = mp.get_key_weight(patcher.model, key)
            if weight is None:
                continue
            qt = _qt_payload(weight, QuantizedTensor)
            if qt is None:
                stats["skipped_not_qt"] += 1
                continue
            if set_func is None:
                stats["skipped_no_set"] += 1
                _console(
                    f"[HSWQ ZI NVFP4 LoRA] WARN cannot bake leftover {key}: "
                    f"QT layout={_qt_layout_name(qt)!r} but no set_weight"
                )
                continue
            if kind is None:
                if _qt_is_int8_tensorwise(weight, QuantizedTensor):
                    kind = "int8"
                elif _qt_is_nvfp4(weight, QuantizedTensor):
                    kind = "nvfp4"
                else:
                    kind = "other"
            keys_to_bake.append((param_key, key))

        if not keys_to_bake:
            continue
        n = _bake_keys_on_module(patcher, module, keys_to_bake, device_to, already)
        if kind == "int8":
            stats["baked_int8"] += n
        else:
            stats["baked_other_qt"] += n

    if stats["baked_int8"] > 0 or stats["baked_other_qt"] > 0:
        patcher.model._hswq_zi_nvfp4_baked_uuid = uuid
        # Also mark INT8 bake uuid so strip helpers stay coherent if present.
        if stats["baked_int8"] > 0:
            patcher.model._hswq_int8_baked_uuid = uuid

    return stats


def _dump_bake_status(nv_stats: dict, rem_stats: dict, patcher) -> None:
    global _STATUS_LOGS
    if _STATUS_LOGS >= _STATUS_LOG_MAX:
        return
    _STATUS_LOGS += 1
    left = len(getattr(patcher, "patches", None) or {})
    _console(
        "[HSWQ ZI NVFP4 LoRA] Dynamic.load bake "
        f"#{_STATUS_LOGS}: "
        f"nvfp4_baked={nv_stats.get('baked_nvfp4', 0)} "
        f"int8_baked={rem_stats.get('baked_int8', 0)} "
        f"other_qt_baked={rem_stats.get('baked_other_qt', 0)} "
        f"nv_candidates={nv_stats.get('candidates', 0)} "
        f"rem_candidates={rem_stats.get('candidates', 0)} "
        f"skip_not_convrot={nv_stats.get('skipped_not_convrot', 0)} "
        f"patches_left={left}"
    )
    if left > 0:
        _console(
            f"[HSWQ ZI NVFP4 LoRA] WARN patches_left={left} after bake "
            "(LowVramPatch still attached — image may break)"
        )


def install_zimage_nvfp4_lora_bake() -> bool:
    """Wrap ModelPatcherDynamic.load: NVFP4 ConvRot bake + leftover INT8 QT bake."""
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

    # Prefer chaining under an older ver of our wrap; else whatever is current.
    prev_load = getattr(original, "_hswq_zi_nvfp4_prev_dynamic_load", None) or original

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
        nv_stats = bake_nvfp4_convrot_patches_on_dynamic_patcher(
            self, device_to=device_to
        )
        rem_stats = bake_remaining_quant_patches_on_dynamic_patcher(
            self, device_to=device_to
        )
        if (
            nv_stats.get("baked_nvfp4", 0) > 0
            or rem_stats.get("baked_int8", 0) > 0
            or rem_stats.get("baked_other_qt", 0) > 0
            or nv_stats.get("candidates", 0) > 0
            or rem_stats.get("candidates", 0) > 0
        ):
            _dump_bake_status(nv_stats, rem_stats, self)
        return result

    load._hswq_zi_nvfp4_lora_bake = True  # type: ignore[attr-defined]
    load._hswq_zi_nvfp4_lora_bake_ver = _BAKE_HOOK_VER  # type: ignore[attr-defined]
    load._hswq_zi_nvfp4_prev_dynamic_load = prev_load  # type: ignore[attr-defined]
    Dynamic.load = load
    _console(
        "[HSWQ ZI NVFP4 LoRA] Dynamic.load bake hook ON v2 "
        "(NVFP4 ConvRot bake + leftover INT8/QT bake)"
    )
    return True


def reset_zimage_nvfp4_lora_bake_log_counters() -> None:
    global _STATUS_LOGS, _SKIP_SAMPLE_LOGS
    _STATUS_LOGS = 0
    _SKIP_SAMPLE_LOGS = 0
