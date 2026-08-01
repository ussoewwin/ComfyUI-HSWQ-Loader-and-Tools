"""Z Image mixed-pack LoRA bake — Dynamic VRAM only (branch under zimage_nvfp4).

Problem (owner A/B + logs):
  Without LoRA, comfy_parity + act_rotate is fine.
  With LoRA, ModelPatcherDynamic attaches LowVramPatch (``180 patches``).

INT8 Dynamic bake (``patches/comfy_quant_int8.py``) often does **not** fire on this
hybrid pack (no INT8 bake dump in logs), so INT8-protect keys stay as
LowVramPatch. NVFP4 ConvRot bake alone leaves ``patches_left=60`` → broken.

v3 (owner: bake dump missing / LoRA inert):
  - Always log ENTER on Dynamic.load wrap (prove wrap fired)
  - Always dump bake summary (even zeros)
  - Bake by iterating ``patcher.patches`` keys (not only named_modules)
  - Force ZI wrap outermost; also bake after ``load_models_gpu`` (MultiGPU path)
  - leftover INT8/QT bake until patches_left=0

Does **not** edit ``nodes/nvfp4`` (SDXL).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_BAKE_HOOK_VER = 3
_STATUS_LOGS = 0
_STATUS_LOG_MAX = 24
_ENTER_LOGS = 0
_ENTER_LOG_MAX = 24
_SKIP_SAMPLE_LOGS = 0
_SKIP_SAMPLE_MAX = 6
_GPU_BAKE_INSTALLED = False


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


def _resolve_module(model, module_path: str):
    try:
        import comfy.utils as cu

        return cu.get_attr(model, module_path)
    except Exception:
        return None


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


def _iter_patch_weight_keys(patcher):
    """Yield (key, module_path, param_key, module) for weight/bias patches."""
    patches = getattr(patcher, "patches", None) or {}
    model = getattr(patcher, "model", None)
    if model is None or not patches:
        return
    for key in list(patches.keys()):
        if not (key.endswith(".weight") or key.endswith(".bias")):
            continue
        module_path, param_key = key.rsplit(".", 1)
        module = _resolve_module(model, module_path)
        if module is None:
            continue
        yield key, module_path, param_key, module


def bake_nvfp4_convrot_patches_on_dynamic_patcher(patcher, device_to) -> dict:
    """Bake LoRA into ConvRot NVFP4 Linears after ModelPatcherDynamic.load."""
    stats = {
        "baked_nvfp4": 0,
        "candidates": 0,
        "skipped_no_set": 0,
        "skipped_not_nvfp4": 0,
        "skipped_not_convrot": 0,
        "cleared_already": 0,
        "unresolved": 0,
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

    # Group keys by module so LowVramPatch clear happens once per module.
    by_module: dict[str, list] = {}
    modules: dict[str, object] = {}
    for key, module_path, param_key, module in _iter_patch_weight_keys(patcher):
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
        by_module.setdefault(module_path, []).append((param_key, key))
        modules[module_path] = module

    for module_path, keys_to_bake in by_module.items():
        stats["baked_nvfp4"] += _bake_keys_on_module(
            patcher, modules[module_path], keys_to_bake, device_to, already
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

    by_module: dict[str, list] = {}
    modules: dict[str, object] = {}
    kinds: dict[str, str] = {}
    for key, module_path, param_key, module in _iter_patch_weight_keys(patcher):
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
        if module_path not in kinds:
            if _qt_is_int8_tensorwise(weight, QuantizedTensor):
                kinds[module_path] = "int8"
            elif _qt_is_nvfp4(weight, QuantizedTensor):
                kinds[module_path] = "nvfp4"
            else:
                kinds[module_path] = "other"
        by_module.setdefault(module_path, []).append((param_key, key))
        modules[module_path] = module

    for module_path, keys_to_bake in by_module.items():
        n = _bake_keys_on_module(
            patcher, modules[module_path], keys_to_bake, device_to, already
        )
        if kinds.get(module_path) == "int8":
            stats["baked_int8"] += n
        else:
            stats["baked_other_qt"] += n

    if stats["baked_int8"] > 0 or stats["baked_other_qt"] > 0:
        patcher.model._hswq_zi_nvfp4_baked_uuid = uuid
        if stats["baked_int8"] > 0:
            patcher.model._hswq_int8_baked_uuid = uuid

    return stats


def _dump_bake_status(nv_stats: dict, rem_stats: dict, patcher, reason: str) -> None:
    global _STATUS_LOGS
    if _STATUS_LOGS >= _STATUS_LOG_MAX:
        return
    _STATUS_LOGS += 1
    left = len(getattr(patcher, "patches", None) or {})
    _console(
        "[HSWQ ZI NVFP4 LoRA] Dynamic.load bake "
        f"#{_STATUS_LOGS} ({reason}): "
        f"nvfp4_baked={nv_stats.get('baked_nvfp4', 0)} "
        f"int8_baked={rem_stats.get('baked_int8', 0)} "
        f"other_qt_baked={rem_stats.get('baked_other_qt', 0)} "
        f"nv_candidates={nv_stats.get('candidates', 0)} "
        f"rem_candidates={rem_stats.get('candidates', 0)} "
        f"skip_not_convrot={nv_stats.get('skipped_not_convrot', 0)} "
        f"patches_left={left}"
    )
    if left > 0:
        sample = list((getattr(patcher, "patches", None) or {}).keys())[:4]
        _console(
            f"[HSWQ ZI NVFP4 LoRA] WARN patches_left={left} after bake "
            f"sample_keys={sample}"
        )


def run_zimage_nvfp4_lora_bake_on_patcher(patcher, device_to=None, reason: str = "wrap") -> bool:
    """Bake NVFP4 ConvRot + leftover QT if this patcher is a ZI NVFP4 pack with LoRA."""
    model = getattr(patcher, "model", None)
    if model is None:
        return False
    has_flag = _model_has_nvfp4_convrot(model)
    has_baked = bool(getattr(model, "_hswq_zi_nvfp4_baked_keys", None))
    n_patches = len(getattr(patcher, "patches", None) or {})
    if not has_flag and not has_baked:
        return False
    if n_patches == 0 and not has_baked:
        return False
    if device_to is None:
        device_to = getattr(patcher, "load_device", None)
    nv_stats = bake_nvfp4_convrot_patches_on_dynamic_patcher(patcher, device_to=device_to)
    rem_stats = bake_remaining_quant_patches_on_dynamic_patcher(
        patcher, device_to=device_to
    )
    _dump_bake_status(nv_stats, rem_stats, patcher, reason=reason)
    return True


def _unwrap_to_non_zi_load(load_fn):
    """Walk past our ZI wraps so reinstall does not nest ZI→ZI."""
    cur = load_fn
    seen = set()
    while (
        cur is not None
        and id(cur) not in seen
        and getattr(cur, "_hswq_zi_nvfp4_lora_bake", False)
    ):
        seen.add(id(cur))
        nxt = getattr(cur, "_hswq_zi_nvfp4_prev_dynamic_load", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return cur


def install_zimage_nvfp4_lora_bake(force: bool = False) -> bool:
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
    if (
        not force
        and getattr(original, "_hswq_zi_nvfp4_lora_bake", False)
        and getattr(original, "_hswq_zi_nvfp4_lora_bake_ver", 0) >= _BAKE_HOOK_VER
    ):
        install_load_models_gpu_bake_hook(force=False)
        return True

    # Prefer chaining under current outer wrap (INT8 / stock), never nest ZI→ZI.
    prev_load = _unwrap_to_non_zi_load(original)

    def load(
        self,
        device_to=None,
        lowvram_model_memory=0,
        force_patch_weights=False,
        full_load=False,
        dirty=False,
    ):
        global _ENTER_LOGS
        if _ENTER_LOGS < _ENTER_LOG_MAX:
            _ENTER_LOGS += 1
            n_patches = len(getattr(self, "patches", None) or {})
            model = getattr(self, "model", None)
            _console(
                f"[HSWQ ZI NVFP4 LoRA] Dynamic.load ENTER #{_ENTER_LOGS}: "
                f"patches={n_patches} "
                f"nvfp4_convrot={_model_has_nvfp4_convrot(model)} "
                f"model={type(model).__name__ if model is not None else None}"
            )
        result = prev_load(
            self,
            device_to=device_to,
            lowvram_model_memory=lowvram_model_memory,
            force_patch_weights=force_patch_weights,
            full_load=full_load,
            dirty=dirty,
        )
        run_zimage_nvfp4_lora_bake_on_patcher(
            self, device_to=device_to, reason="Dynamic.load"
        )
        return result

    load._hswq_zi_nvfp4_lora_bake = True  # type: ignore[attr-defined]
    load._hswq_zi_nvfp4_lora_bake_ver = _BAKE_HOOK_VER  # type: ignore[attr-defined]
    load._hswq_zi_nvfp4_prev_dynamic_load = prev_load  # type: ignore[attr-defined]
    Dynamic.load = load
    _console(
        f"[HSWQ ZI NVFP4 LoRA] Dynamic.load bake hook ON v{_BAKE_HOOK_VER} "
        "(NVFP4 ConvRot bake + leftover INT8/QT bake + load_models_gpu)"
    )
    install_load_models_gpu_bake_hook(force=True)
    return True


def _unwrap_to_non_zi_load_models_gpu(fn):
    cur = fn
    seen = set()
    while (
        cur is not None
        and id(cur) not in seen
        and getattr(cur, "_hswq_zi_nvfp4_gpu_bake", False)
    ):
        seen.add(id(cur))
        nxt = getattr(cur, "_hswq_zi_nvfp4_prev_load_models_gpu", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return cur


def install_load_models_gpu_bake_hook(force: bool = False) -> bool:
    """After MultiGPU/stock load_models_gpu, bake any remaining ZI NVFP4 LoRA patches."""
    global _GPU_BAKE_INSTALLED
    try:
        import comfy.model_management as mm
    except ImportError:
        return False
    original = mm.load_models_gpu
    if (
        not force
        and getattr(original, "_hswq_zi_nvfp4_gpu_bake", False)
        and getattr(original, "_hswq_zi_nvfp4_gpu_bake_ver", 0) >= _BAKE_HOOK_VER
    ):
        _GPU_BAKE_INSTALLED = True
        return True
    prev = _unwrap_to_non_zi_load_models_gpu(original)

    def load_models_gpu(*args, **kwargs):
        result = prev(*args, **kwargs)
        try:
            for loaded in list(getattr(mm, "current_loaded_models", []) or []):
                patcher = getattr(loaded, "model", None)
                if patcher is None:
                    continue
                try:
                    if not bool(patcher.is_dynamic()):
                        continue
                except Exception:
                    continue
                if not getattr(patcher, "patches", None):
                    # Still try if bake keys exist (LowVram cleared but leftover)
                    if not getattr(
                        getattr(patcher, "model", None),
                        "_hswq_zi_nvfp4_baked_keys",
                        None,
                    ):
                        continue
                run_zimage_nvfp4_lora_bake_on_patcher(
                    patcher,
                    device_to=getattr(patcher, "load_device", None),
                    reason="load_models_gpu",
                )
        except Exception as exc:
            _console(f"[HSWQ ZI NVFP4 LoRA] load_models_gpu bake error: {exc!r}")
        return result

    load_models_gpu._hswq_zi_nvfp4_gpu_bake = True  # type: ignore[attr-defined]
    load_models_gpu._hswq_zi_nvfp4_gpu_bake_ver = _BAKE_HOOK_VER  # type: ignore[attr-defined]
    load_models_gpu._hswq_zi_nvfp4_prev_load_models_gpu = prev  # type: ignore[attr-defined]
    mm.load_models_gpu = load_models_gpu
    _GPU_BAKE_INSTALLED = True
    _console(
        f"[HSWQ ZI NVFP4 LoRA] load_models_gpu bake hook ON v{_BAKE_HOOK_VER}"
    )
    return True


def reset_zimage_nvfp4_lora_bake_log_counters() -> None:
    global _STATUS_LOGS, _SKIP_SAMPLE_LOGS, _ENTER_LOGS
    _STATUS_LOGS = 0
    _SKIP_SAMPLE_LOGS = 0
    _ENTER_LOGS = 0
