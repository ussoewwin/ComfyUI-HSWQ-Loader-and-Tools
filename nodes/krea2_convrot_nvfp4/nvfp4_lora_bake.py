"""Krea2 mixed-pack LoRA bake - Dynamic VRAM only (branch under krea2_convrot_nvfp4).

Krea2 hybrid packs (ConvRot NVFP4 Linear + INT8 protect Conv2d/Linear) need
LoRA baked on BOTH quant sides, exactly like Z Image (owner requirement):

  1) NVFP4 pass: ConvRot NVFP4 Linears bake via Linear.convert_weight /
     set_weight (``_NVFP4_LORA_BAKE_VER`` = 2 in krea2 nvfp4_forward:
     dequant -> unrotate -> LoRA -> re-rotate -> requant).
  2) remaining-QT pass: leftover INT8 tensorwise / other QT keys bake via
     their own set_weight (INT8 protect path owned by patches/comfy_quant_int8).

Without this hook, ModelPatcherDynamic attaches LowVramPatch (``180 patches``)
and LoRA strength breaks. Structure mirrors nodes/zimage_nvfp4/nvfp4_lora_bake.py
(ZI v7 field-proven) with Krea2-only stamps so ZI / SDXL never collide:

  - Dynamic.load wrap stamp: ``_hswq_krea2_nvfp4_lora_bake(_ver)``
  - load_models_gpu wrap stamp: ``_hswq_krea2_nvfp4_gpu_bake(_ver)``
  - baked-key registry: ``_hswq_krea2_nvfp4_baked_keys`` / ``_hswq_krea2_nvfp4_baked_uuid``

Does **not** edit ``nodes/nvfp4`` (SDXL), ``nodes/zimage_nvfp4`` (Z Image) or
``patches/comfy_quant_int8``. Runtime only.
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
    """Kitchen QT layout class name.

    Do **not** use ``qt.layout`` - that is ``torch.Tensor.layout``
    (``torch.strided``), whose type name is literally ``"layout"``.
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
    return bool(getattr(module, "_hswq_nvfp4_convrot", False))


def _get_baked_key_set(model) -> set:
    keys = getattr(model, "_hswq_krea2_nvfp4_baked_keys", None)
    if keys is None:
        keys = set()
        model._hswq_krea2_nvfp4_baked_keys = keys
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
    module (act_rotate still hits).
    """
    return bool(_nvfp4_convrot_diag(model)["has"])


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
    prev_uuid = getattr(patcher.model, "_hswq_krea2_nvfp4_baked_uuid", None)
    if prev_uuid is not None and prev_uuid != uuid:
        already.clear()

    # Group keys by module so LowVramPatch clear happens once per module.
    by_module: dict = {}
    modules: dict = {}
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
                    f"[HSWQ Krea2 NVFP4 LoRA] nv_pass_defer_int8_rem sample "
                    f"#{_SKIP_SAMPLE_LOGS}: {key} layout={_qt_layout_name(qt)!r} "
                    f"nvfp4_convrot={getattr(module, '_hswq_nvfp4_convrot', False)} "
                    f"int8_convrot={getattr(module, '_hswq_int8_convrot', False)} "
                    f"params_convrot={params_convrot} "
                    f"(not a failure - baked in INT8 rem pass)"
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
                f"[HSWQ Krea2 NVFP4 LoRA] WARN cannot bake {key}: "
                "NVFP4 QT but no set_weight"
            )
            continue
        by_module.setdefault(module_path, []).append((param_key, key))
        modules[module_path] = module

    for module_path, keys_to_bake in by_module.items():
        n = _bake_keys_on_module(
            patcher, modules[module_path], keys_to_bake, device_to, already
        )
        stats["baked_nvfp4"] += n
        if n > 0 and len(stats["sample_nvfp4_keys"]) < 3:
            for _pk, full_key in keys_to_bake:
                if full_key not in stats["sample_nvfp4_keys"]:
                    stats["sample_nvfp4_keys"].append(full_key)
                if len(stats["sample_nvfp4_keys"]) >= 3:
                    break

    if stats["baked_nvfp4"] > 0:
        patcher.model._hswq_krea2_nvfp4_baked_uuid = uuid

    return stats


def bake_remaining_quant_patches_on_dynamic_patcher(patcher, device_to) -> dict:
    """Bake leftover QT LoRA (ConvRot INT8 protect etc.) that NVFP4 pass skipped.

    Hybrid packs: NVFP4 ConvRot is baked first; INT8 protect ConvRot Linears
    use ``_hswq_int8_convrot`` + cleared Params (Conv2d twin) via
    ``Linear.convert_weight`` / ``set_weight`` (INT8 protect stack).
    """
    stats = {
        "baked_int8": 0,
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

    by_module: dict = {}
    modules: dict = {}
    kinds: dict = {}
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
                f"[HSWQ Krea2 NVFP4 LoRA] WARN cannot bake leftover {key}: "
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
            if len(stats["sample_int8_keys"]) < 3:
                for _pk, full_key in keys_to_bake:
                    if full_key not in stats["sample_int8_keys"]:
                        stats["sample_int8_keys"].append(full_key)
                    if len(stats["sample_int8_keys"]) >= 3:
                        break
        else:
            stats["baked_other_qt"] += n

    if stats["baked_int8"] > 0 or stats["baked_other_qt"] > 0:
        patcher.model._hswq_krea2_nvfp4_baked_uuid = uuid
        if stats["baked_int8"] > 0:
            patcher.model._hswq_int8_baked_uuid = uuid

    return stats


def _dump_bake_status(
    nv_stats: dict,
    rem_stats: dict,
    patcher,
    reason: str,
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
        "[HSWQ Krea2 NVFP4 LoRA] Dynamic.load bake "
        f"#{_STATUS_LOGS} ({reason}): "
        f"nvfp4_baked={nv_n} "
        f"int8_baked={i8} "
        f"other_qt_baked={rem_stats.get('baked_other_qt', 0)} "
        f"nv_candidates={nv_stats.get('candidates', 0)} "
        f"rem_candidates={rem_stats.get('candidates', 0)} "
        f"nv_pass_skip_int8_rem={skip_i8_in_nv_pass} "
        f"(INT8 rem baked separately as int8_baked) "
        f"patches_left={left} patches_uuid={uuid_s}"
    )
    if left > 0:
        sample = list((getattr(patcher, "patches", None) or {}).keys())[:4]
        _console(
            f"[HSWQ Krea2 NVFP4 LoRA] WARN patches_left={left} after bake "
            f"sample_keys={sample}"
        )



def _patcher_is_krea2_nvfp4_pack(patcher) -> bool:
    """True only for packs loaded via the Krea2 ConvRot NVFP4 loader.

    ``_hswq_nvfp4_convrot`` flags are shared with Z Image (same lineage), so
    flags alone would fire this bake on ZI / SDXL models. The loader stamps
    ``_hswq_krea2_nvfp4_pack`` on the ModelPatcher and its inner model at load;
    only that stamp authorizes the bake (never mix with other models).
    """
    if getattr(patcher, "_hswq_krea2_nvfp4_pack", False):
        return True
    model = getattr(patcher, "model", None)
    return bool(getattr(model, "_hswq_krea2_nvfp4_pack", False))


def run_krea2_nvfp4_lora_bake_on_patcher(patcher, device_to=None, reason: str = "wrap") -> bool:
    """Bake NVFP4 ConvRot + leftover QT if this patcher is a Krea2 NVFP4 pack with LoRA."""
    model = getattr(patcher, "model", None)
    if model is None:
        return False
    if not _patcher_is_krea2_nvfp4_pack(patcher):
        return False
    diag = _nvfp4_convrot_diag(model)
    has_flag = bool(diag["has"])
    has_baked = bool(getattr(model, "_hswq_krea2_nvfp4_baked_keys", None))
    n_patches = len(getattr(patcher, "patches", None) or {})
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


def _unwrap_to_non_krea2_load(load_fn):
    """Walk past our Krea2 wraps so reinstall does not nest Krea2->Krea2."""
    cur = load_fn
    seen = set()
    while (
        cur is not None
        and id(cur) not in seen
        and getattr(cur, "_hswq_krea2_nvfp4_lora_bake", False)
    ):
        seen.add(id(cur))
        nxt = getattr(cur, "_hswq_krea2_prev_dynamic_load", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return cur


def _chain_has_krea2_dynamic_load(load_fn) -> bool:
    """True if any Krea2 wrap remains in prev / ``_hswq_orig_dynamic_load`` chain."""
    cur = load_fn
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if getattr(cur, "_hswq_krea2_nvfp4_lora_bake", False):
            return True
        nxt = getattr(cur, "_hswq_krea2_prev_dynamic_load", None)
        if nxt is None:
            nxt = getattr(cur, "_hswq_orig_dynamic_load", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return False


def uninstall_krea2_nvfp4_lora_bake() -> bool:
    """Remove Krea2 Dynamic / load_models_gpu bake hooks (other models must not inherit).

    After Krea2 ConvRot NVFP4, these hooks stay on ``ModelPatcherDynamic.load``
    and would bake other models' LoRA with the Krea2 path -> noise.
    Krea2-only stamps are peeled; ZI / INT8 / stock wraps are untouched.
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
                or getattr(cur, "_hswq_krea2_nvfp4_lora_bake", False)
            ):
                if cleaned is not None:
                    Dynamic.load = cleaned
                removed = True
                need_int8_repatch = bool(discarded_int8)
                _console(
                    "[HSWQ Krea2 NVFP4 LoRA] Dynamic.load bake hook OFF "
                    "(restored; Krea2 path no longer wraps bake)"
                    + (
                        "; discarded INT8 wrap that captured Krea2 true_orig"
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
        if getattr(cur_gpu, "_hswq_krea2_nvfp4_gpu_bake", False):
            mm.load_models_gpu = _unwrap_to_non_krea2_load_models_gpu(cur_gpu)
            _GPU_BAKE_INSTALLED = False
            removed = True
            _console(
                "[HSWQ Krea2 NVFP4 LoRA] load_models_gpu bake hook OFF (restored)"
            )
    if need_int8_repatch:
        try:
            from ...patches.comfy_quant_int8 import (
                _patch_model_patcher_dynamic_int8_lora_bake,
            )

            _patch_model_patcher_dynamic_int8_lora_bake()
            _console(
                "[HSWQ Krea2 NVFP4 LoRA] Reinstalled clean INT8 Dynamic.load bake "
                "(after discarding Krea2-contaminated wrap)"
            )
        except Exception as e:
            logger.warning(
                "[HSWQ Krea2 NVFP4 LoRA] INT8 Dynamic.load re-patch after Krea2 "
                "clean failed: %s",
                e,
            )
    return removed


def _deep_clean_dynamic_load(load_fn):
    """Peel Krea2 wraps; walk past INT8 wraps that closed over a Krea2 ``true_orig``.

    While Krea2 was outermost, INT8 may re-patch with ``true_orig = Krea2 wrap``
    (closure). Peeling the outer Krea2 would leave INT8->Krea2->... so ENTER
    still fires on other models. Discard such contaminated INT8 wraps too, then
    re-patch the clean INT8 Dynamic bake (ZI field lesson: LoRA strength wrong
    when a contaminated wrap survives).

    Returns ``(cleaned_load, discarded_contaminated_int8)``.
    """
    cur = load_fn
    discarded_int8 = False
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if getattr(cur, "_hswq_krea2_nvfp4_lora_bake", False):
            nxt = getattr(cur, "_hswq_krea2_prev_dynamic_load", None)
            if nxt is None or nxt is cur:
                break
            cur = nxt
            continue
        if getattr(cur, "_hswq_int8_lora_bake", False):
            captured = getattr(cur, "_hswq_orig_dynamic_load", None)
            if _chain_has_krea2_dynamic_load(captured):
                # Closure already bound to Krea2 - attribute rewrite cannot fix it.
                discarded_int8 = True
                cur = _unwrap_to_non_krea2_load(captured)
                continue
            break
        break
    return cur, discarded_int8


def install_krea2_nvfp4_lora_bake(force: bool = False) -> bool:
    """Wrap ModelPatcherDynamic.load: NVFP4 ConvRot bake + leftover INT8 QT bake."""
    try:
        import comfy.model_patcher as mp
    except ImportError:
        return False

    Dynamic = getattr(mp, "ModelPatcherDynamic", None)
    if Dynamic is None:
        _console("[HSWQ Krea2 NVFP4 LoRA] ModelPatcherDynamic missing - bake hook skipped")
        return False
    original = getattr(Dynamic, "load", None)
    if original is None:
        return False
    if (
        not force
        and getattr(original, "_hswq_krea2_nvfp4_lora_bake", False)
        and getattr(original, "_hswq_krea2_nvfp4_lora_bake_ver", 0) >= _BAKE_HOOK_VER
    ):
        install_load_models_gpu_bake_hook(force=False)
        return True

    # Cross-branch contract (SDXL does the same): drop the ZI Dynamic bake
    # hook before we take over - ZI's gate keys on the SHARED
    # ``_hswq_nvfp4_convrot`` flags and would also fire (and stamp) Krea2
    # packs. Import-and-call only; never edits nodes/zimage_nvfp4.
    try:
        from ..zimage_nvfp4.nvfp4_lora_bake import uninstall_zimage_nvfp4_lora_bake

        uninstall_zimage_nvfp4_lora_bake()
        # Re-read: uninstall replaced Dynamic.load; chaining to the stale ZI
        # wrap would leave ZI's ENTER + flag-keyed bake firing on Krea2 packs.
        original = getattr(Dynamic, "load", None)
        if original is None:
            return False
    except Exception as e:
        logger.debug("[HSWQ Krea2 NVFP4 LoRA] ZI bake hook uninstall skipped: %s", e)

    # Prefer chaining under current outer wrap (INT8 / stock), never nest
    # Krea2->Krea2.
    prev_load = _unwrap_to_non_krea2_load(original)

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
                f"[HSWQ Krea2 NVFP4 LoRA] Dynamic.load ENTER #{_ENTER_LOGS}: "
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
        run_krea2_nvfp4_lora_bake_on_patcher(
            self, device_to=device_to, reason="Dynamic.load"
        )
        return result

    load._hswq_krea2_nvfp4_lora_bake = True  # type: ignore[attr-defined]
    load._hswq_krea2_nvfp4_lora_bake_ver = _BAKE_HOOK_VER  # type: ignore[attr-defined]
    load._hswq_krea2_prev_dynamic_load = prev_load  # type: ignore[attr-defined]
    Dynamic.load = load
    _console(
        f"[HSWQ Krea2 NVFP4 LoRA] Dynamic.load bake hook ON v{_BAKE_HOOK_VER} "
        "(NVFP4 ConvRot bake + leftover INT8/QT bake + load_models_gpu)"
    )
    try:
        install_load_models_gpu_bake_hook(force=True)
    except Exception as e:
        logger.warning(
            "[HSWQ Krea2 NVFP4 LoRA] load_models_gpu bake hook skipped: %s", e
        )
    return True


def _unwrap_to_non_krea2_load_models_gpu(fn):
    cur = fn
    seen = set()
    while (
        cur is not None
        and id(cur) not in seen
        and getattr(cur, "_hswq_krea2_nvfp4_gpu_bake", False)
    ):
        seen.add(id(cur))
        nxt = getattr(cur, "_hswq_krea2_prev_load_models_gpu", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return cur


def install_load_models_gpu_bake_hook(force: bool = False) -> bool:
    """After stock load_models_gpu, bake any remaining Krea2 NVFP4 LoRA patches."""
    global _GPU_BAKE_INSTALLED
    try:
        import comfy.model_management as mm
    except ImportError:
        return False
    original = mm.load_models_gpu
    if (
        not force
        and getattr(original, "_hswq_krea2_nvfp4_gpu_bake", False)
        and getattr(original, "_hswq_krea2_nvfp4_gpu_bake_ver", 0) >= _BAKE_HOOK_VER
    ):
        _GPU_BAKE_INSTALLED = True
        return True
    prev = _unwrap_to_non_krea2_load_models_gpu(original)

    def load_models_gpu(*args, **kwargs):
        result = prev(*args, **kwargs)
        try:
            for loaded in list(getattr(mm, "current_loaded_models", []) or []):
                patcher = getattr(loaded, "model", None)
                if patcher is None:
                    continue
                # Fast skip: no patches AND no baked keys = not our model
                has_patches = bool(getattr(patcher, "patches", None))
                has_baked = bool(getattr(getattr(patcher, "model", None), "_hswq_krea2_nvfp4_baked_keys", None))
                if not has_patches and not has_baked:
                    continue
                # Skip non-dynamic models
                try:
                    if not bool(patcher.is_dynamic()):
                        continue
                except Exception:
                    continue
                # Fast skip: only Krea2-stamped packs (never ZI / SDXL despite
                # shared _hswq_nvfp4_convrot flags).
                if not _patcher_is_krea2_nvfp4_pack(patcher):
                    continue
                run_krea2_nvfp4_lora_bake_on_patcher(
                    patcher,
                    device_to=getattr(patcher, "load_device", None),
                    reason="load_models_gpu",
                )
        except Exception as exc:
            _console(f"[HSWQ Krea2 NVFP4 LoRA] load_models_gpu bake error: {exc!r}")
        return result

    load_models_gpu._hswq_krea2_nvfp4_gpu_bake = True  # type: ignore[attr-defined]
    load_models_gpu._hswq_krea2_nvfp4_gpu_bake_ver = _BAKE_HOOK_VER  # type: ignore[attr-defined]
    load_models_gpu._hswq_krea2_prev_load_models_gpu = prev  # type: ignore[attr-defined]
    mm.load_models_gpu = load_models_gpu
    _GPU_BAKE_INSTALLED = True
    _console(
        f"[HSWQ Krea2 NVFP4 LoRA] load_models_gpu bake hook ON v{_BAKE_HOOK_VER}"
    )
    return True


def reset_krea2_nvfp4_lora_bake_log_counters() -> None:
    global _STATUS_LOGS, _SKIP_SAMPLE_LOGS, _ENTER_LOGS
    _STATUS_LOGS = 0
    _SKIP_SAMPLE_LOGS = 0
    _ENTER_LOGS = 0
