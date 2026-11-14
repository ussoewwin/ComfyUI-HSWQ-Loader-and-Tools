"""Z Image mixed-pack LoRA bake — Dynamic VRAM only (branch under zimage_convrot_nvfp4).

Problem (owner A/B + logs):
  Without LoRA, comfy_parity + act_rotate is fine.
  With LoRA, ModelPatcherDynamic attaches LowVramPatch (``180 patches``).

INT8 Dynamic bake (``patches/comfy_quant_int8.py``) often does **not** fire on this
hybrid pack (no INT8 bake dump in logs), so INT8-protect keys stay as
LowVramPatch. NVFP4 ConvRot bake alone leaves ``patches_left=60`` → broken.

v3: ENTER proved wrap fires; bake still silent (``nvfp4_convrot=False``).

v4 (owner: still no good — ENTER patches=180 nvfp4_convrot=False):
  Root cause: kitchen ``QuantizedTensor`` inherits ``torch.Tensor.layout``
  (``torch.strided`` → type name ``\"layout\"``). Old ``_qt_layout_name``
  read ``qt.layout`` first and never saw ``_layout_cls``
  (``TensorCoreNVFP4Layout``), so ``_qt_is_nvfp4`` was always False → gate
  closed → no bake dump → LoRA LowVramPatch left attached.
  Fix: prefer ``_layout_cls`` / ``layout_cls``; gate on ConvRot **flag**
  (do not require QT on ``module.weight`` under Dynamic VRAM).

Does **not** edit ``nodes/sdxl_convrot_nvfp4`` (SDXL legacy).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_BAKE_HOOK_VER = 7
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
    """Kitchen QT layout class name.

    Do **not** use ``qt.layout`` — that is ``torch.Tensor.layout``
    (``torch.strided``), whose type name is literally ``\"layout\"``.
    Real name lives in ``_layout_cls`` (str) / ``layout_cls`` (type).
    """
    if qt is None:
        return ""
    layout_cls = getattr(qt, "_layout_cls", None)
    if isinstance(layout_cls, str) and layout_cls:
        return layout_cls
    layout_cls_t = getattr(qt, "layout_cls", None)
    if layout_cls_t is not None and not isinstance(layout_cls_t, str):
        name = getattr(layout_cls_t, "__name__", "") or ""
        if name:
            return name
    # Legacy object layout (not torch.layout)
    legacy = getattr(qt, "_layout", None)
    if legacy is not None:
        name = type(legacy).__name__ or ""
        if name and name != "layout":
            return name
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


def _nvfp4_convrot_diag(model) -> dict:
    """Count ConvRot-armed modules and how many still expose NVFP4 on ``.weight``."""
    out = {"flagged": 0, "qt_on_weight": 0, "has": False}
    if model is None:
        return out
    try:
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        QuantizedTensor = None
    for _name, module in model.named_modules():
        if not _module_is_nvfp4_convrot(module):
            continue
        out["flagged"] += 1
        if QuantizedTensor is None:
            continue
        w = getattr(module, "weight", None)
        if _qt_is_nvfp4(w, QuantizedTensor):
            out["qt_on_weight"] += 1
    out["has"] = out["flagged"] > 0
    return out


def _model_has_nvfp4_convrot(model) -> bool:
    """True if any module was armed with ConvRot NVFP4 (``_hswq_nvfp4_convrot``).

    Do **not** require QT on ``module.weight``: under Dynamic VRAM / LowVramPatch
    the QT often lives behind ``get_key_weight``, while the flag remains on the
    module (act_rotate still hits). v3 gate required both → always False.
    """
    return bool(_nvfp4_convrot_diag(model)["has"])


def _resolve_module(model, module_path: str):
    try:
        import comfy.utils as cu

        return cu.get_attr(model, module_path)
    except Exception:
        return None


def _extract_zi_lora_residual(patcher, key):
    """Extract rank-decomposed LoRA (up, down, scale, offset) from a patch list.

    Supports both full-layer patches (offset=None) and output-channel slice
    patches (offset=(0, start, length), e.g. Z Image fused attention.qkv.weight
    for to_q / to_k / to_v). Anything else returns None and falls back to the
    requant bake.
    """
    try:
        from comfy.weight_adapter.base import WeightAdapterBase
    except ImportError:
        return None
    patches = getattr(patcher, "patches", {}).get(key)
    if not patches:
        return None
    residual = []
    for p in patches:
        strength = p[0]
        v = p[1]
        strength_model = p[2]
        offset = p[3]
        function = p[4]
        if function is not None:
            return None
        off_tuple = None
        if offset is not None:
            if not (
                isinstance(offset, (tuple, list))
                and len(offset) == 3
                and int(offset[0]) == 0
            ):
                return None
            o_start, o_len = int(offset[1]), int(offset[2])
            if o_start < 0 or o_len <= 0:
                return None
            off_tuple = (o_start, o_len)
        if float(strength_model) != 1.0:
            return None
        if not isinstance(v, WeightAdapterBase):
            return None
        weights = getattr(v, "weights", None)
        if weights is None or len(weights) < 3:
            return None
        mat_up = weights[0]  # lora_B (up)   [out, rank]
        mat_dn = weights[1]  # lora_A (down) [rank, in]
        alpha = weights[2]
        mid = weights[3] if len(weights) > 3 else None
        dora_scale = weights[4] if len(weights) > 4 else None
        reshape = weights[5] if len(weights) > 5 else None
        if mid is not None or dora_scale is not None or reshape is not None:
            return None
        if getattr(mat_up, "ndim", 0) > 2 and all(
            int(s) == 1 for s in mat_up.shape[2:]
        ):
            mat_up = mat_up.reshape(mat_up.shape[0], mat_up.shape[1])
        if getattr(mat_dn, "ndim", 0) > 2 and all(
            int(s) == 1 for s in mat_dn.shape[2:]
        ):
            mat_dn = mat_dn.reshape(mat_dn.shape[0], mat_dn.shape[1])
        if getattr(mat_up, "ndim", 0) != 2 or getattr(mat_dn, "ndim", 0) != 2:
            return None
        if off_tuple is not None and int(mat_up.shape[0]) != off_tuple[1]:
            return None
        if alpha is not None:
            try:
                alpha = alpha / mat_dn.shape[0]
            except Exception:
                alpha = 1.0
        else:
            alpha = 1.0
        scale = float(strength) * float(alpha)
        residual.append((mat_dn, mat_up, scale, off_tuple))
    return residual or None


def _model_has_zi_lora_residuals(model) -> bool:
    if model is None:
        return False
    for _name, module in model.named_modules():
        if hasattr(module, "_hswq_zi_lora_res"):
            return True
    return False


def _clear_zi_stale_lora_residuals(patcher, keep_keys) -> int:
    """Drop residuals this run no longer wants (LoRA removed / swapped)."""
    model = getattr(patcher, "model", None)
    if model is None:
        return 0
    cleared = 0
    for name, module in model.named_modules():
        if not hasattr(module, "_hswq_zi_lora_res"):
            continue
        key = f"{name}.weight"
        if key in keep_keys:
            continue
        try:
            delattr(module, "_hswq_zi_lora_res")
        except Exception:
            pass
        if hasattr(module, "_hswq_zi_lora_res_gpu"):
            try:
                delattr(module, "_hswq_zi_lora_res_gpu")
            except Exception:
                pass
        cleared += 1
    if cleared:
        _console(
            f"[HSWQ ZI NVFP4 LoRA] cleared {cleared} stale LoRA residual(s)"
        )
    return cleared


def _bake_keys_on_module(patcher, module, keys_to_bake, device_to, already) -> tuple:
    """Bake keys as a low-rank residual (exact) with requant fallback.

    NVFP4 requant cannot carry a LoRA delta (4-bit step ~amax/16 vs a
    0.1-0.8% amax delta; measured kept_ratio 0.00-0.19 on the hybrid pack),
    so rank-decomposed weight keys are stored on the module
    (``_hswq_zi_lora_res``) and re-added by the ZI forward. Requant stays only
    as the fallback for patches the residual cannot represent (bias keys,
    strength_model != 1.0, non rank-decomposed adapters).
    """
    baked = 0
    n_residual = 0
    n_requant = 0
    for param_key, _key in keys_to_bake:
        if hasattr(module, param_key + "_lowvram_function"):
            setattr(module, param_key + "_lowvram_function", None)
    for param_key, key in keys_to_bake:
        res = None
        if param_key == "weight":
            res = _extract_zi_lora_residual(patcher, key)
        if res is not None:
            module._hswq_zi_lora_res = res
            module._hswq_zi_lora_res_name = key
            if hasattr(module, "_hswq_zi_lora_res_gpu"):
                try:
                    delattr(module, "_hswq_zi_lora_res_gpu")
                except Exception:
                    pass
            n_residual += 1
        else:
            w_obj = getattr(module, "weight", None)
            target_dev = getattr(w_obj, "device", None) or device_to
            patcher.patch_weight_to_device(key, device_to=target_dev)
            n_requant += 1
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
    return baked, n_residual, n_requant


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
        "residual_nvfp4": 0,
        "requant_nvfp4": 0,
        "candidates": 0,
        "skipped_no_set": 0,
        "skipped_not_nvfp4": 0,
        "skipped_not_convrot": 0,
        "cleared_already": 0,
        "unresolved": 0,
        "sample_nvfp4_keys": [],
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
                params = getattr(qt, "_params", None) if qt is not None else None
                params_convrot = bool(getattr(params, "convrot", False)) if params else False
                _console(
                    f"[HSWQ ZI NVFP4 LoRA] nv_pass_defer_int8_rem sample "
                    f"#{_SKIP_SAMPLE_LOGS}: {key} layout={_qt_layout_name(qt)!r} "
                    f"nvfp4_convrot={getattr(module, '_hswq_nvfp4_convrot', False)} "
                    f"int8_convrot={getattr(module, '_hswq_int8_convrot', False)} "
                    f"params_convrot={params_convrot} "
                    f"(not a failure — baked in INT8 rem pass)"
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
        n, n_res, n_rq = _bake_keys_on_module(
            patcher, modules[module_path], keys_to_bake, device_to, already
        )
        stats["baked_nvfp4"] += n
        stats["residual_nvfp4"] += n_res
        stats["requant_nvfp4"] += n_rq
        if n > 0 and len(stats["sample_nvfp4_keys"]) < 3:
            for _pk, full_key in keys_to_bake:
                if full_key not in stats["sample_nvfp4_keys"]:
                    stats["sample_nvfp4_keys"].append(full_key)
                if len(stats["sample_nvfp4_keys"]) >= 3:
                    break

    if stats["baked_nvfp4"] > 0:
        patcher.model._hswq_zi_nvfp4_baked_uuid = uuid

    return stats


def bake_remaining_quant_patches_on_dynamic_patcher(patcher, device_to) -> dict:
    """Bake leftover QT LoRA (ConvRot INT8 protect etc.) that NVFP4 pass skipped.

    Hybrid packs: NVFP4 ConvRot is baked first; INT8 protect ConvRot Linears
    use ``_hswq_int8_convrot`` + cleared Params (Conv2d twin) via
    ``Linear.convert_weight`` / ``set_weight`` (``_NVFP4_LORA_BAKE_VER`` >= 5).
    """
    stats = {
        "baked_int8": 0,
        "residual_int8": 0,
        "requant_int8": 0,
        "baked_other_qt": 0,
        "candidates": 0,
        "skipped_no_set": 0,
        "skipped_not_qt": 0,
        "cleared_already": 0,
        "sample_int8_keys": [],
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
        n, n_res, n_rq = _bake_keys_on_module(
            patcher, modules[module_path], keys_to_bake, device_to, already
        )
        stats["residual_int8"] += n_res
        stats["requant_int8"] += n_rq
        if kinds.get(module_path) == "int8":
            stats["baked_int8"] += n
            if len(stats["sample_int8_keys"]) < 3:
                for _pk, full_key in keys_to_bake:
                    if full_key not in stats["sample_int8_keys"]:
                        stats["sample_int8_keys"].append(full_key)
                    if len(stats["sample_int8_keys"]) >= 3:
                        break
        else:
            stats["baked_other_qt"] += n

    if stats["baked_int8"] > 0 or stats["baked_other_qt"] > 0:
        patcher.model._hswq_zi_nvfp4_baked_uuid = uuid
        if stats["baked_int8"] > 0:
            patcher.model._hswq_int8_baked_uuid = uuid

    return stats


def _dump_bake_status(
    nv_stats: dict,
    rem_stats: dict,
    patcher,
    reason: str,
    counters_before: dict | None = None,
) -> None:
    global _STATUS_LOGS
    nv_n = int(nv_stats.get("baked_nvfp4", 0) or 0)
    i8 = int(rem_stats.get("baked_int8", 0) or 0)
    # Empty re-bake / VAE: do not spam status or stale EVIDENCE.
    if nv_n == 0 and i8 == 0 and int(rem_stats.get("baked_other_qt", 0) or 0) == 0:
        return
    if _STATUS_LOGS >= _STATUS_LOG_MAX:
        return
    _STATUS_LOGS += 1
    left = len(getattr(patcher, "patches", None) or {})
    skip_i8_in_nv_pass = int(nv_stats.get("skipped_not_convrot", 0) or 0)
    uuid = getattr(patcher, "patches_uuid", None)
    uuid_s = f"{uuid}"[:8] if uuid is not None else "-"
    _console(
        "[HSWQ ZI NVFP4 LoRA] Dynamic.load bake "
        f"#{_STATUS_LOGS} ({reason}): "
        f"nvfp4_baked={nv_n} "
        f"int8_baked={i8} "
        f"other_qt_baked={rem_stats.get('baked_other_qt', 0)} "
        f"nv_candidates={nv_stats.get('candidates', 0)} "
        f"rem_candidates={rem_stats.get('candidates', 0)} "
        f"nv_pass_skip_int8_rem={skip_i8_in_nv_pass} "
        f"(INT8 rem baked separately as int8_baked) "
        f"nvfp4_residual={nv_stats.get('residual_nvfp4', 0)} "
        f"int8_residual={rem_stats.get('residual_int8', 0)} "
        f"patches_left={left} patches_uuid={uuid_s}"
    )
    try:
        from .zi_nvfp4_forward import log_nvfp4_lora_bake_evidence

        log_nvfp4_lora_bake_evidence(
            tag=f"bake#{_STATUS_LOGS}/{reason}",
            before=counters_before,
            nvfp4_baked=nv_n,
            int8_baked=i8,
            sample_nvfp4_keys=list(nv_stats.get("sample_nvfp4_keys") or []),
            sample_int8_keys=list(rem_stats.get("sample_int8_keys") or []),
        )
    except Exception as e:
        _console(f"[HSWQ ConvRot LoRA] EVIDENCE log failed: {e}")
    if left > 0:
        sample = list((getattr(patcher, "patches", None) or {}).keys())[:4]
        _console(
            f"[HSWQ ZI NVFP4 LoRA] WARN patches_left={left} after bake "
            f"sample_keys={sample}"
        )


def _patcher_has_quant_via_keys(patcher) -> bool:
    """True if any LoRA patch key resolves to NVFP4/INT8 QT via get_key_weight."""
    if not getattr(patcher, "patches", None):
        return False
    try:
        import comfy.model_patcher as mp
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return False
    for key, _module_path, _param_key, _module in _iter_patch_weight_keys(patcher):
        weight, _set_func, _convert = mp.get_key_weight(patcher.model, key)
        if weight is None:
            continue
        if _qt_is_nvfp4(weight, QuantizedTensor) or _qt_is_int8_tensorwise(
            weight, QuantizedTensor
        ):
            return True
    return False


def _patcher_is_zi_nvfp4_pack(patcher) -> bool:
    """True only for Z Image NVFP4 / Hybrid packs (never Krea2 INT8 or SDXL)."""
    if patcher is None:
        return False
    model = getattr(patcher, "model", None)
    if model is None:
        return False
    if getattr(patcher, "_hswq_krea2_int8_pack", False) or getattr(
        model, "_hswq_krea2_int8_pack", False
    ):
        return False
    if getattr(patcher, "_hswq_zi_nvfp4_pack", False) or getattr(
        model, "_hswq_zi_nvfp4_pack", False
    ):
        return True
    if bool(getattr(model, "_hswq_zi_nvfp4_baked_keys", None)) or _model_has_zi_lora_residuals(model):
        return True
    return bool(_nvfp4_convrot_diag(model)["has"])


def run_zimage_nvfp4_lora_bake_on_patcher(patcher, device_to=None, reason: str = "wrap") -> bool:
    """Bake NVFP4 ConvRot + leftover QT if this patcher is a ZI NVFP4 pack with LoRA."""
    model = getattr(patcher, "model", None)
    if model is None:
        return False
    if not _patcher_is_zi_nvfp4_pack(patcher):
        return False
    patches = getattr(patcher, "patches", None) or {}
    n_patches = len(patches)
    uuid = getattr(patcher, "patches_uuid", None)
    last = getattr(model, "_hswq_zi_nvfp4_baked_uuid", None)
    if last is not None and uuid == last and n_patches == 0:
        return True
    if n_patches == 0:
        # LoRA removed (or swapped): wipe residuals left by a previous bake on
        # this shared inner model so "LoRA off" really turns it off.
        if last is not None or _model_has_zi_lora_residuals(model):
            _clear_zi_stale_lora_residuals(patcher, set())
            try:
                model._hswq_zi_nvfp4_baked_keys = set()
                model._hswq_zi_nvfp4_baked_uuid = uuid
            except Exception:
                pass
        return False
    if last is not None and uuid != last:
        _clear_zi_stale_lora_residuals(patcher, set())
        try:
            model._hswq_zi_nvfp4_baked_keys = set()
        except Exception:
            pass
    else:
        _clear_zi_stale_lora_residuals(patcher, set(patches.keys()))
    if device_to is None:
        device_to = getattr(patcher, "load_device", None)
    try:
        from .zi_nvfp4_forward import snapshot_nvfp4_lora_bake_counters

        counters_before = snapshot_nvfp4_lora_bake_counters()
    except Exception:
        counters_before = None
    nv_stats = bake_nvfp4_convrot_patches_on_dynamic_patcher(patcher, device_to=device_to)
    rem_stats = bake_remaining_quant_patches_on_dynamic_patcher(
        patcher, device_to=device_to
    )
    _dump_bake_status(
        nv_stats, rem_stats, patcher, reason=reason, counters_before=counters_before
    )
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


def _chain_has_zi_dynamic_load(load_fn) -> bool:
    """True if any ZI wrap remains in prev / ``_hswq_orig_dynamic_load`` chain."""
    cur = load_fn
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if getattr(cur, "_hswq_zi_nvfp4_lora_bake", False):
            return True
        nxt = getattr(cur, "_hswq_zi_nvfp4_prev_dynamic_load", None)
        if nxt is None:
            nxt = getattr(cur, "_hswq_orig_dynamic_load", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return False


def _deep_clean_dynamic_load(load_fn):
    """Peel ZI wraps and discard INT8 wraps that closed over a ZI ``true_orig``.

    Owner log after peel-only uninstall: ``Dynamic.load bake hook OFF`` then still
    ``[HSWQ ZI NVFP4 LoRA] Dynamic.load ENTER … model=SDXL``. Cause: while ZI was
    outermost, INT8 re-patched with ``true_orig = ZI_wrap`` (closure). Peeling the
    outer ZI left INT8→ZI→… so ENTER still fires and SDXL LoRA strength is wrong.

    Returns ``(cleaned_load, discarded_contaminated_int8)``.
    """
    cur = load_fn
    discarded_int8 = False
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if getattr(cur, "_hswq_zi_nvfp4_lora_bake", False):
            nxt = getattr(cur, "_hswq_zi_nvfp4_prev_dynamic_load", None)
            if nxt is None or nxt is cur:
                break
            cur = nxt
            continue
        if getattr(cur, "_hswq_int8_lora_bake", False):
            captured = getattr(cur, "_hswq_orig_dynamic_load", None)
            if _chain_has_zi_dynamic_load(captured):
                # Closure already bound to ZI — attribute rewrite cannot fix it.
                discarded_int8 = True
                cur = _unwrap_to_non_zi_load(captured)
                continue
            break
        break
    return cur, discarded_int8


def uninstall_zimage_nvfp4_lora_bake() -> bool:
    """Remove Z Image Dynamic / load_models_gpu bake hooks (SDXL must not inherit them).

    After Z Image ConvRot NVFP4, these hooks stay on ``ModelPatcherDynamic.load``
    and bake SDXL LoRA with the ZI path (INT8 protect / other_qt) → noise.
    Also discards INT8 Dynamic.load wraps that captured ZI as ``true_orig``
    (logs still show ENTER on SDXL after OFF; LoRA keys apply but strength is weak).
    Call from SDXL loaders before applying the SDXL TC / INT8 stacks.
    """
    global _GPU_BAKE_INSTALLED
    removed = False
    need_int8_repatch = False
    try:
        import comfy.model_patcher as mp
    except ImportError:
        mp = None
    if mp is not None:
        Dynamic = getattr(mp, "ModelPatcherDynamic", None)
        if Dynamic is not None:
            cur = getattr(Dynamic, "load", None)
            cleaned, discarded_int8 = _deep_clean_dynamic_load(cur)
            if (
                cleaned is not cur
                or discarded_int8
                or getattr(cur, "_hswq_zi_nvfp4_lora_bake", False)
            ):
                if cleaned is not None:
                    Dynamic.load = cleaned
                removed = True
                need_int8_repatch = bool(discarded_int8)
                _console(
                    "[HSWQ ZI NVFP4 LoRA] Dynamic.load bake hook OFF "
                    "(restored for SDXL; Z Image path no longer wraps bake)"
                    + (
                        "; discarded INT8 wrap that captured ZI true_orig"
                        if discarded_int8
                        else ""
                    )
                )
    try:
        import comfy.model_management as mm
    except ImportError:
        mm = None
    if mm is not None:
        cur_gpu = getattr(mm, "load_models_gpu", None)
        if getattr(cur_gpu, "_hswq_zi_nvfp4_gpu_bake", False):
            mm.load_models_gpu = _unwrap_to_non_zi_load_models_gpu(cur_gpu)
            _GPU_BAKE_INSTALLED = False
            removed = True
            _console(
                "[HSWQ ZI NVFP4 LoRA] load_models_gpu bake hook OFF "
                "(restored for SDXL)"
            )
    if need_int8_repatch:
        try:
            from ...patches.comfy_quant_int8 import (
                _patch_model_patcher_dynamic_int8_lora_bake,
            )

            _patch_model_patcher_dynamic_int8_lora_bake()
            _console(
                "[HSWQ ZI NVFP4 LoRA] Reinstalled clean INT8 Dynamic.load bake "
                "(after discarding ZI-contaminated wrap)"
            )
        except Exception as e:
            logger.warning(
                "[HSWQ ZI NVFP4 LoRA] INT8 Dynamic.load re-patch after ZI clean "
                "failed: %s",
                e,
            )
    # Also strip legacy ModelPatcher.load / partially_load wraps
    if mp is not None:
        Legacy = getattr(mp, "ModelPatcher", None)
        if Legacy is not None:
            cur_leg = getattr(Legacy, "load", None)
            while getattr(cur_leg, "_hswq_zi_nvfp4_legacy_bake", False):
                nxt = getattr(cur_leg, "_hswq_zi_nvfp4_prev_legacy_load", None)
                if nxt is None or nxt is cur_leg:
                    break
                cur_leg = nxt
            if cur_leg is not None and cur_leg is not getattr(Legacy, "load", None):
                Legacy.load = cur_leg
                removed = True
            cur_pl = getattr(Legacy, "partially_load", None)
            while getattr(cur_pl, "_hswq_zi_nvfp4_legacy_partially_load", False):
                nxt = getattr(cur_pl, "_hswq_zi_nvfp4_prev_legacy_partially_load", None)
                if nxt is None or nxt is cur_pl:
                    break
                cur_pl = nxt
            if cur_pl is not None and cur_pl is not getattr(Legacy, "partially_load", None):
                Legacy.partially_load = cur_pl
                removed = True
    # Also strip VER=8 convert/set left on mp0.Linear by comfy_parity install.
    try:
        import comfy.ops as ops
        from .zi_nvfp4_forward import peel_all_nvfp4_linear_lora_bake

        if peel_all_nvfp4_linear_lora_bake(ops.mixed_precision_ops().Linear):
            removed = True
            _console(
                "[HSWQ ZI NVFP4 LoRA] peeled Linear convert/set bake wraps "
                "(int8_protect) for SDXL"
            )
    except Exception as e:
        logger.warning(
            "[HSWQ ZI NVFP4 LoRA] peel Linear bake wraps failed: %s", e
        )
    return removed


def _zi_legacy_target_keys(patcher) -> set:
    """Keys whose ConvRot NVFP4 / INT8 QT the Z Image bake consumes."""
    if not _patcher_is_zi_nvfp4_pack(patcher):
        return set()
    try:
        import comfy.model_patcher as mp
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return set()
    keys = set()
    for key, _module_path, _param_key, module in list(_iter_patch_weight_keys(patcher)):
        try:
            weight, set_func, _convert = mp.get_key_weight(patcher.model, key)
        except Exception:
            continue
        if weight is None or set_func is None:
            continue
        qt = _qt_payload(weight, QuantizedTensor)
        if qt is None:
            continue
        if _module_is_nvfp4_convrot(module) or _qt_is_nvfp4(
            weight, QuantizedTensor
        ) or _qt_is_int8_tensorwise(weight, QuantizedTensor):
            keys.add(key)
    return keys


def install_zimage_nvfp4_lora_bake_legacy(force: bool = False) -> bool:
    """Bake Z Image NVFP4 ConvRot LoRA on LEGACY ModelPatcher.load & partially_load.

    DisTorch2 (when ``disable_dynamic=True``) replaces ``ModelPatcher.partially_load``
    with ``new_partially_load``, which loops over ``_load_list()`` and calls
    ``self.patch_weight_to_device(weight_key)`` directly without calling
    ``ModelPatcher.load``. Wrapping both ``partially_load`` and ``load`` hides
    the ConvRot keys from ``new_partially_load`` / stock ``load`` and runs the
    low-rank residual bake right after.
    """
    try:
        import comfy.model_patcher as mp
    except ImportError:
        return False
    Legacy = getattr(mp, "ModelPatcher", None)
    if Legacy is None:
        return False
    original = getattr(Legacy, "load", None)
    if original is None:
        return False
    need_load_wrap = (
        force
        or not getattr(original, "_hswq_zi_nvfp4_legacy_bake", False)
        or getattr(original, "_hswq_zi_nvfp4_legacy_bake_ver", 0) < _BAKE_HOOK_VER
    )
    if need_load_wrap:
        prev_load = original
        while getattr(prev_load, "_hswq_zi_nvfp4_legacy_bake", False):
            nxt = getattr(prev_load, "_hswq_zi_nvfp4_prev_legacy_load", None)
            if nxt is None or nxt is prev_load:
                break
            prev_load = nxt

        def load(
            self,
            device_to=None,
            lowvram_model_memory=0,
            force_patch_weights=False,
            full_load=False,
        ):
            hidden = {}
            is_zi = False
            try:
                is_zi = (
                    bool(getattr(getattr(self, "model", None), "_hswq_bake_enabled", True))
                    and _patcher_is_zi_nvfp4_pack(self)
                )
                if is_zi and getattr(self, "patches", None):
                    for key in _zi_legacy_target_keys(self) & set(self.patches.keys()):
                        hidden[key] = self.patches.pop(key)
            except Exception as exc:
                _console(f"[HSWQ ZI NVFP4 LoRA] legacy pre-load scan error: {exc!r}")
            try:
                result = prev_load(
                    self,
                    device_to=device_to,
                    lowvram_model_memory=lowvram_model_memory,
                    force_patch_weights=force_patch_weights,
                    full_load=full_load,
                )
            finally:
                if hidden:
                    self.patches.update(hidden)
            if is_zi and (hidden or not getattr(self, "patches", None)):
                try:
                    run_zimage_nvfp4_lora_bake_on_patcher(
                        self, device_to=device_to, reason="legacy load"
                    )
                except Exception as exc:
                    _console(f"[HSWQ ZI NVFP4 LoRA] legacy load bake error: {exc!r}")
            return result

        load._hswq_zi_nvfp4_legacy_bake = True  # type: ignore[attr-defined]
        load._hswq_zi_nvfp4_legacy_bake_ver = _BAKE_HOOK_VER  # type: ignore[attr-defined]
        load._hswq_zi_nvfp4_prev_legacy_load = prev_load  # type: ignore[attr-defined]
        Legacy.load = load
        _console(
            f"[HSWQ ZI NVFP4 LoRA] legacy ModelPatcher.load bake hook ON "
            f"v{_BAKE_HOOK_VER} (hide ConvRot keys -> family bake)"
        )

    orig_pl = getattr(Legacy, "partially_load", None)
    if orig_pl is not None and (
        force
        or not getattr(orig_pl, "_hswq_zi_nvfp4_legacy_partially_load", False)
        or getattr(orig_pl, "_hswq_zi_nvfp4_legacy_partially_load_ver", 0) < _BAKE_HOOK_VER
    ):
        prev_pl = orig_pl
        while getattr(prev_pl, "_hswq_zi_nvfp4_legacy_partially_load", False):
            nxt = getattr(prev_pl, "_hswq_zi_nvfp4_prev_legacy_partially_load", None)
            if nxt is None or nxt is prev_pl:
                break
            prev_pl = nxt

        def partially_load(
            self,
            device_to,
            extra_memory=0,
            force_patch_weights=False,
            **kwargs,
        ):
            hidden = {}
            is_zi = False
            try:
                is_dyn = bool(getattr(self, "is_dynamic", lambda: False)())
                is_zi = (
                    not is_dyn
                    and bool(getattr(getattr(self, "model", None), "_hswq_bake_enabled", True))
                    and _patcher_is_zi_nvfp4_pack(self)
                )
                if is_zi and getattr(self, "patches", None):
                    for key in _zi_legacy_target_keys(self) & set(self.patches.keys()):
                        hidden[key] = self.patches.pop(key)
            except Exception as exc:
                _console(
                    f"[HSWQ ZI NVFP4 LoRA] legacy partially_load scan error: {exc!r}"
                )
            try:
                result = prev_pl(
                    self,
                    device_to,
                    extra_memory=extra_memory,
                    force_patch_weights=force_patch_weights,
                    **kwargs,
                )
            finally:
                if hidden:
                    self.patches.update(hidden)
            if is_zi and (hidden or not getattr(self, "patches", None)):
                try:
                    run_zimage_nvfp4_lora_bake_on_patcher(
                        self, device_to=device_to, reason="legacy partially_load"
                    )
                except Exception as exc:
                    _console(
                        f"[HSWQ ZI NVFP4 LoRA] legacy partially_load bake error: {exc!r}"
                    )
            return result

        partially_load._hswq_zi_nvfp4_legacy_partially_load = True  # type: ignore[attr-defined]
        partially_load._hswq_zi_nvfp4_legacy_partially_load_ver = _BAKE_HOOK_VER  # type: ignore[attr-defined]
        partially_load._hswq_zi_nvfp4_prev_legacy_partially_load = prev_pl  # type: ignore[attr-defined]
        Legacy.partially_load = partially_load

    return True


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
            diag = _nvfp4_convrot_diag(model)
            _console(
                f"[HSWQ ZI NVFP4 LoRA] Dynamic.load ENTER #{_ENTER_LOGS}: "
                f"patches={n_patches} "
                f"nvfp4_convrot={diag['has']} "
                f"flagged={diag['flagged']} "
                f"qt_on_weight={diag['qt_on_weight']} "
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
        if not bool(getattr(self.model, "_hswq_bake_enabled", True)):
            return result
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
    try:
        install_zimage_nvfp4_lora_bake_legacy(force=force)
    except Exception as e:
        logger.warning(
            "[HSWQ ZI NVFP4 LoRA] legacy load bake hook skipped: %s", e
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
                # Fast skip: no patches AND no baked keys = not our model
                has_patches = bool(getattr(patcher, "patches", None))
                has_baked = bool(getattr(getattr(patcher, "model", None), "_hswq_zi_nvfp4_baked_keys", None))
                if not has_patches and not has_baked:
                    continue
                # Skip non-dynamic models
                try:
                    if not bool(patcher.is_dynamic()):
                        continue
                except Exception:
                    continue
                # Fast skip: check if this is a ZI NVFP4 model via cached diag
                model = getattr(patcher, "model", None)
                if model is not None:
                    diag = _nvfp4_convrot_diag(model)
                    if not diag["has"] and not has_baked:
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
