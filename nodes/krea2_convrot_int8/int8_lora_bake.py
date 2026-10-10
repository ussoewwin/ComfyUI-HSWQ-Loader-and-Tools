"""Krea2 ConvRot INT8 LoRA bake - Dynamic VRAM only (branch under krea2_convrot_int8).

Krea2 ConvRot INT8 packs need LoRA baked as a low-rank residual (INT8 8-bit
requant rounds away small LoRA deltas: step ~amax/127 vs delta ~0.1-0.8% amax).
The remaining-QT pass bakes INT8 tensorwise / other QT keys via their own
set_weight (INT8 stack owned by patches/comfy_quant_int8 +
int8_forward._add_krea2_lora_residual).

Without this hook, ModelPatcherDynamic attaches LowVramPatch (``180 patches``)
and LoRA strength breaks. Structure mirrors nodes/zimage_nvfp4/nvfp4_lora_bake.py
(ZI v7 field-proven) with Krea2-only stamps so ZI / SDXL never collide:

  - Dynamic.load wrap stamp: ``_hswq_krea2_int8_lora_bake(_ver)``
  - load_models_gpu wrap stamp: ``_hswq_krea2_int8_gpu_bake(_ver)``
  - baked-key registry: ``_hswq_krea2_int8_baked_keys`` / ``_hswq_krea2_int8_baked_uuid``

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


def _qt_is_int8_tensorwise(weight, QuantizedTensor) -> bool:
    """INT8 detect including ``_layout_cls`` string (kitchen / protect packs)."""
    qt = _qt_payload(weight, QuantizedTensor)
    if qt is None:
        return False
    name = _qt_layout_name(qt)
    return "TensorWiseINT8" in name or "int8_tensorwise" in name.lower()


def _get_baked_key_set(model) -> set:
    keys = getattr(model, "_hswq_krea2_int8_baked_keys", None)
    if keys is None:
        keys = set()
        model._hswq_krea2_int8_baked_keys = keys
    return keys


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


def _extract_lora_residual(patcher, key):
    """Extract rank-decomposed LoRA (A=down, B=up, scale) from a patch list.

    Small LoRA deltas are ~0.1-0.8% of weight amax vs the ~amax/127 INT8
    requant step, so instead of
    baking into the packed weight we keep the packed QT and store the low-rank
    additive term. VRAM stays packed; the residual is rank x (in+out), ~4% of
    the packed weight. Returns None when any patch is not a simple
    rank-decomposed LoRA (caller falls back to the requantize bake).
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
        if offset is not None or function is not None:
            return None
        if float(strength_model) != 1.0:
            # strength_model scales the BASE weight; residual path keeps the
            # base packed (unscaled), so fall back to the requantize bake.
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
        if alpha is not None:
            try:
                alpha = alpha / mat_dn.shape[0]
            except Exception:
                alpha = 1.0
        else:
            alpha = 1.0
        scale = float(strength) * float(alpha)
        residual.append((mat_dn, mat_up, scale))
    return residual or None


def _bake_residual_keys_on_module(patcher, module, keys_to_bake, device_to, already):
    """Store the low-rank LoRA residual instead of requantizing the QT weight.

    Returns (baked, n_residual, n_requant_fallback).
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
            res = _extract_lora_residual(patcher, key)
        if res is not None:
            module._hswq_krea2_lora_res = res
            if hasattr(module, "_hswq_krea2_lora_res_gpu"):
                delattr(module, "_hswq_krea2_lora_res_gpu")
            n_residual += 1
        else:
            patcher.patch_weight_to_device(key, device_to=device_to)
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


    """Bake leftover QT LoRA (ConvRot INT8 protect etc.).

    INT8 protect ConvRot Linears use ``_hswq_int8_convrot`` + cleared Params (Conv2d twin) via
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
        "baked_residual": 0,
        "int8_residual": 0,
        "baked_requant_fallback": 0,
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
                f"[HSWQ Krea2 INT8 LoRA] WARN cannot bake leftover {key}: "
                f"QT layout={_qt_layout_name(qt)!r} but no set_weight"
            )
            continue
        if module_path not in kinds:
            kinds[module_path] = (
                "int8" if _qt_is_int8_tensorwise(weight, QuantizedTensor) else "other"
            )
        by_module.setdefault(module_path, []).append((param_key, key))
        modules[module_path] = module

    for module_path, keys_to_bake in by_module.items():
        # INT8 8-bit requant ALSO rounds away small LoRA deltas
        # (per-channel step ~amax/127 vs delta ~0.1-0.8% amax), so INT8
        # ConvRot keys take the same low-rank residual path. Non-LoRA /
        # strength_model-scaled patches fall back to requant inside.
        n, n_res, n_rq = _bake_residual_keys_on_module(
            patcher, modules[module_path], keys_to_bake, device_to, already
        )
        if kinds.get(module_path) == "int8":
            stats["baked_int8"] += n
            stats["int8_residual"] = stats.get("int8_residual", 0) + n_res
            if len(stats["sample_int8_keys"]) < 3:
                for _pk, full_key in keys_to_bake:
                    if full_key not in stats["sample_int8_keys"]:
                        stats["sample_int8_keys"].append(full_key)
                    if len(stats["sample_int8_keys"]) >= 3:
                        break
        else:
            stats["baked_other_qt"] += n
            stats["baked_residual"] = stats.get("baked_residual", 0) + n_res
        stats["baked_requant_fallback"] = (
            stats.get("baked_requant_fallback", 0) + n_rq
        )

    if stats["baked_int8"] > 0 or stats["baked_other_qt"] > 0:
        patcher.model._hswq_krea2_int8_baked_uuid = uuid
        if stats["baked_int8"] > 0:
            patcher.model._hswq_int8_baked_uuid = uuid

    return stats


def _dump_bake_status(
    rem_stats: dict,
    patcher,
    reason: str,
) -> None:
    global _STATUS_LOGS
    i8 = int(rem_stats.get("baked_int8", 0) or 0)
    # Empty re-bake / VAE: do not spam status or stale EVIDENCE.
    if i8 == 0 and int(rem_stats.get("baked_other_qt", 0) or 0) == 0:
        return
    if _STATUS_LOGS >= _STATUS_LOG_MAX:
        return
    _STATUS_LOGS += 1
    left = len(getattr(patcher, "patches", None) or {})
    uuid = getattr(patcher, "patches_uuid", None)
    uuid_s = f"{uuid}"[:8] if uuid is not None else "-"
    _console(
        "[HSWQ Krea2 INT8 LoRA] Dynamic.load bake "
        f"#{_STATUS_LOGS} ({reason}): "
        f"int8_baked={i8} "
        f"other_qt_baked={rem_stats.get('baked_other_qt', 0)} "
        f"residual_baked={rem_stats.get('baked_residual', 0)} "
        f"int8_residual={rem_stats.get('int8_residual', 0)} "
        f"requant_fallback={rem_stats.get('baked_requant_fallback', 0)} "
        f"rem_candidates={rem_stats.get('candidates', 0)} "
        f"patches_left={left} patches_uuid={uuid_s}"
    )
    if left > 0:
        sample = list((getattr(patcher, "patches", None) or {}).keys())[:4]
        _console(
            f"[HSWQ Krea2 INT8 LoRA] WARN patches_left={left} after bake "
            f"sample_keys={sample}"
        )



def _patcher_is_krea2_pack(patcher) -> bool:
    """True only for packs loaded via the Krea2 ConvRot INT8 load path.

    The stamp is set only by the Krea2 INT8 load path (patches/comfy_quant_int8);
    shared flags alone would fire this bake on ZI / SDXL models, so only this
    Krea2-specific stamp authorizes it (never mix with other models).
    only that stamp authorizes the bake (never mix with other models).
    """
    if getattr(patcher, "_hswq_krea2_int8_pack", False):
        return True
    model = getattr(patcher, "model", None)
    return bool(getattr(model, "_hswq_krea2_int8_pack", False))


def _clear_stale_lora_residuals(patcher, keep_keys) -> int:
    """Drop residuals from a previous bake that this run no longer wants.

    Dynamic VRAM clones share the inner model: a bake from an earlier queue
    leaves ``_hswq_krea2_lora_res`` on module objects. If the new patcher has
    no patches for those keys (LoRA removed / swapped), the stale residual
    would keep applying forever. Clear everything not in ``keep_keys``.
    """
    model = getattr(patcher, "model", None)
    if model is None:
        return 0
    cleared = 0
    for name, module in model.named_modules():
        if not hasattr(module, "_hswq_krea2_lora_res"):
            continue
        key = f"{name}.weight"
        if key in keep_keys:
            continue
        try:
            delattr(module, "_hswq_krea2_lora_res")
        except Exception:
            pass
        if hasattr(module, "_hswq_krea2_lora_res_gpu"):
            try:
                delattr(module, "_hswq_krea2_lora_res_gpu")
            except Exception:
                pass
        cleared += 1
    if cleared:
        _console(
            f"[HSWQ Krea2 INT8 LoRA] cleared {cleared} stale LoRA residual(s) "
            f"({reason_keep(keep_keys)})"
        )
    return cleared


def reason_keep(keep_keys) -> str:
    return "no patches this run" if not keep_keys else "not in current patches"


def run_krea2_int8_lora_bake_on_patcher(patcher, device_to=None, reason: str = "wrap") -> bool:
    """Bake leftover QT LoRA if this patcher is a Krea2 INT8 pack with LoRA."""
    model = getattr(patcher, "model", None)
    if model is None:
        return False
    if not _patcher_is_krea2_pack(patcher):
        return False
    has_baked = bool(getattr(model, "_hswq_krea2_int8_baked_keys", None))
    patches = getattr(patcher, "patches", None) or {}
    n_patches = len(patches)
    if n_patches == 0:
        # LoRA removed (or none): wipe residuals left by any previous bake on
        # this shared inner model so "LoRA off" really turns it off.
        if has_baked or any(
            hasattr(m, "_hswq_krea2_lora_res")
            for _n, m in model.named_modules()
        ):
            _clear_stale_lora_residuals(patcher, set())
            try:
                model._hswq_krea2_int8_baked_keys = set()
            except Exception:
                pass
        return False
    # Keep only keys this patcher still patches; clear stale residuals first.
    # Idempotency: if this patcher's LoRA run was already baked (same
    # patches_uuid), the only patches left are non-QT (float) layers that the
    # bake intentionally leaves for the stock path. Re-entering here (the
    # load_models_gpu hook fires after Dynamic.load) would clear the residuals
    # we just baked, because _clear_stale_lora_residuals only keeps the
    # leftover patch keys. Skip instead.
    _uuid = getattr(patcher, "patches_uuid", None)
    if _uuid is not None and getattr(
        model, "_hswq_krea2_int8_baked_uuid", None
    ) == _uuid:
        return True
    _clear_stale_lora_residuals(patcher, set(patches.keys()))
    if device_to is None:
        device_to = getattr(patcher, "load_device", None)
    rem_stats = bake_remaining_quant_patches_on_dynamic_patcher(
        patcher, device_to=device_to
    )
    _dump_bake_status(rem_stats, patcher, reason=reason)
    return True


def _unwrap_to_non_krea2_load(load_fn):
    """Walk past our Krea2 wraps so reinstall does not nest Krea2->Krea2."""
    cur = load_fn
    seen = set()
    while (
        cur is not None
        and id(cur) not in seen
        and getattr(cur, "_hswq_krea2_int8_lora_bake", False)
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
        if getattr(cur, "_hswq_krea2_int8_lora_bake", False):
            return True
        nxt = getattr(cur, "_hswq_krea2_prev_dynamic_load", None)
        if nxt is None:
            nxt = getattr(cur, "_hswq_orig_dynamic_load", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return False


def uninstall_krea2_int8_lora_bake() -> bool:
    """Remove Krea2 Dynamic / load_models_gpu bake hooks (other models must not inherit).

    After a Krea2 load, these hooks stay on ``ModelPatcherDynamic.load``
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
                or getattr(cur, "_hswq_krea2_int8_lora_bake", False)
            ):
                if cleaned is not None:
                    Dynamic.load = cleaned
                removed = True
                need_int8_repatch = bool(discarded_int8)
                _console(
                    "[HSWQ Krea2 INT8 LoRA] Dynamic.load bake hook OFF "
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
        if getattr(cur_gpu, "_hswq_krea2_int8_gpu_bake", False):
            mm.load_models_gpu = _unwrap_to_non_krea2_load_models_gpu(cur_gpu)
            _GPU_BAKE_INSTALLED = False
            removed = True
            _console(
                "[HSWQ Krea2 INT8 LoRA] load_models_gpu bake hook OFF (restored)"
            )
    if need_int8_repatch:
        try:
            from ...patches.comfy_quant_int8 import (
                _patch_model_patcher_dynamic_int8_lora_bake,
            )

            _patch_model_patcher_dynamic_int8_lora_bake()
            _console(
                "[HSWQ Krea2 INT8 LoRA] Reinstalled clean INT8 Dynamic.load bake "
                "(after discarding Krea2-contaminated wrap)"
            )
        except Exception as e:
            logger.warning(
                "[HSWQ Krea2 INT8 LoRA] INT8 Dynamic.load re-patch after Krea2 "
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
        if getattr(cur, "_hswq_krea2_int8_lora_bake", False):
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


def install_krea2_int8_lora_bake(force: bool = False) -> bool:
    """Wrap ModelPatcherDynamic.load: Krea2 INT8 QT LoRA bake."""
    try:
        import comfy.model_patcher as mp
    except ImportError:
        return False

    Dynamic = getattr(mp, "ModelPatcherDynamic", None)
    if Dynamic is None:
        _console("[HSWQ Krea2 INT8 LoRA] ModelPatcherDynamic missing - bake hook skipped")
        return False
    original = getattr(Dynamic, "load", None)
    if original is None:
        return False
    if (
        not force
        and getattr(original, "_hswq_krea2_int8_lora_bake", False)
        and getattr(original, "_hswq_krea2_int8_lora_bake_ver", 0) >= _BAKE_HOOK_VER
    ):
        install_load_models_gpu_bake_hook(force=False)
        return True

    # Cross-branch contract (SDXL does the same): drop the ZI Dynamic bake
    # hook before we take over - ZI's gate keys on the SHARED
    # shared ConvRot flags and would also fire (and stamp) Krea2 packs.
    # Import-and-call only; never edits nodes/zimage_nvfp4.
    try:
        from ..zimage_nvfp4.nvfp4_lora_bake import uninstall_zimage_nvfp4_lora_bake

        uninstall_zimage_nvfp4_lora_bake()
        # Re-read: uninstall replaced Dynamic.load; chaining to the stale ZI
        # wrap would leave ZI's ENTER + flag-keyed bake firing on Krea2 packs.
        original = getattr(Dynamic, "load", None)
        if original is None:
            return False
    except Exception as e:
        logger.debug("[HSWQ Krea2 INT8 LoRA] ZI bake hook uninstall skipped: %s", e)

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
            _console(
                f"[HSWQ Krea2 INT8 LoRA] Dynamic.load ENTER #{_ENTER_LOGS}: "
                f"patches={n_patches} "
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
        run_krea2_int8_lora_bake_on_patcher(
            self, device_to=device_to, reason="Dynamic.load"
        )
        return result

    load._hswq_krea2_int8_lora_bake = True  # type: ignore[attr-defined]
    load._hswq_krea2_int8_lora_bake_ver = _BAKE_HOOK_VER  # type: ignore[attr-defined]
    load._hswq_krea2_prev_dynamic_load = prev_load  # type: ignore[attr-defined]
    Dynamic.load = load
    _console(
        f"[HSWQ Krea2 INT8 LoRA] Dynamic.load bake hook ON v{_BAKE_HOOK_VER} "
        "(Krea2 INT8 QT LoRA bake + load_models_gpu)"
    )
    try:
        install_load_models_gpu_bake_hook(force=True)
    except Exception as e:
        logger.warning(
            "[HSWQ Krea2 INT8 LoRA] load_models_gpu bake hook skipped: %s", e
        )
    return True


def _unwrap_to_non_krea2_load_models_gpu(fn):
    cur = fn
    seen = set()
    while (
        cur is not None
        and id(cur) not in seen
        and getattr(cur, "_hswq_krea2_int8_gpu_bake", False)
    ):
        seen.add(id(cur))
        nxt = getattr(cur, "_hswq_krea2_prev_load_models_gpu", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return cur


def install_load_models_gpu_bake_hook(force: bool = False) -> bool:
    """After stock load_models_gpu, bake any remaining Krea2 INT8 LoRA patches."""
    global _GPU_BAKE_INSTALLED
    try:
        import comfy.model_management as mm
    except ImportError:
        return False
    original = mm.load_models_gpu
    if (
        not force
        and getattr(original, "_hswq_krea2_int8_gpu_bake", False)
        and getattr(original, "_hswq_krea2_int8_gpu_bake_ver", 0) >= _BAKE_HOOK_VER
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
                # Only bake patchers that still carry un-consumed patches. The
                # Dynamic.load bake consumes the patches (patches_left=0), so a
                # patcher with no patches here is either already-baked (keep the
                # residuals!) or LoRA-removed (already cleared by Dynamic.load).
                # Re-entering run_krea2_int8_lora_bake_on_patcher here with
                # n_patches==0 would hit its stale-clear branch and wipe the
                # residuals Dynamic.load just baked.
                has_patches = bool(getattr(patcher, "patches", None))
                if not has_patches:
                    continue
                # Skip non-dynamic models
                try:
                    if not bool(patcher.is_dynamic()):
                        continue
                except Exception:
                    continue
                # Fast skip: only Krea2-stamped packs (never ZI / SDXL despite
                # shared ConvRot flags).
                if not _patcher_is_krea2_pack(patcher):
                    continue
                run_krea2_int8_lora_bake_on_patcher(
                    patcher,
                    device_to=getattr(patcher, "load_device", None),
                    reason="load_models_gpu",
                )
        except Exception as exc:
            _console(f"[HSWQ Krea2 INT8 LoRA] load_models_gpu bake error: {exc!r}")
        return result

    load_models_gpu._hswq_krea2_int8_gpu_bake = True  # type: ignore[attr-defined]
    load_models_gpu._hswq_krea2_int8_gpu_bake_ver = _BAKE_HOOK_VER  # type: ignore[attr-defined]
    load_models_gpu._hswq_krea2_prev_load_models_gpu = prev  # type: ignore[attr-defined]
    mm.load_models_gpu = load_models_gpu
    _GPU_BAKE_INSTALLED = True
    _console(
        f"[HSWQ Krea2 INT8 LoRA] load_models_gpu bake hook ON v{_BAKE_HOOK_VER}"
    )
    return True


def reset_krea2_int8_lora_bake_log_counters() -> None:
    global _STATUS_LOGS, _SKIP_SAMPLE_LOGS, _ENTER_LOGS
    _STATUS_LOGS = 0
    _SKIP_SAMPLE_LOGS = 0
    _ENTER_LOGS = 0
