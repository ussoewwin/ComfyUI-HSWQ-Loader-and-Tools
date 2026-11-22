"""Flux1 ConvRot W4A8 — LoRA patches as forward-time residuals (Dynamic load).

Measured ComfyUI behaviour this module works around:

  - ``comfy/lora.py`` contains no ``QuantizedTensor`` handling at all, so a LoRA
    patch aimed at a 4-bit packed weight has no correct float path.
  - ``ModelPatcherDynamic.setup_param`` attaches ``LowVramPatch`` for every key
    with patches whose calculated shape matches the weight, which would push the
    packed weight through float patch math.
  - ``comfy/ops.py`` grouped-int8 load only accepts the pack's own sidecars.

So, for W4A8 keys, this module takes the patches away from the patcher before
the weight is touched and records factorised terms on the module; the forward
wrapper (``flux1_w4a8_forward``) adds them after the stock GEMM. Anything this
module cannot express as plain LoRA factors raises instead of degrading
silently.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_BAKE_HOOK_VER = 1
_DYNAMIC_TAG = "_hswq_w4a8_lora_bake"
_GPU_TAG = "_hswq_w4a8_gpu_bake"
_PACK_TAG = "_hswq_w4a8_pack"

_COUNTERS = {"bakes": 0, "keys_baked": 0, "keys_skipped": 0}


def _console(msg: str) -> None:
    print(msg, flush=True)
    logger.info(msg)


def _module_for_key(model, weight_key: str):
    import comfy.utils

    module_key = (
        weight_key[: -len(".weight")] if weight_key.endswith(".weight") else weight_key
    )
    try:
        return comfy.utils.get_attr(model, module_key)
    except Exception:
        return None


def _extract_w4a8_lora_terms(patcher, key: str):
    """Return [(down, up, scale, offset|None)] for a plain LoRA patch, else None."""
    try:
        from comfy.weight_adapter.base import WeightAdapterBase
    except ImportError:
        return None
    patches = getattr(patcher, "patches", {}).get(key)
    if not patches:
        return None

    terms = []
    for entry in patches:
        strength, adapter, strength_model, offset, function = entry
        if function is not None:
            return None
        if float(strength_model) != 1.0:
            return None
        off_tuple = None
        if offset is not None:
            if not (isinstance(offset, (tuple, list)) and len(offset) == 3
                    and int(offset[0]) == 0):
                return None
            start, length = int(offset[1]), int(offset[2])
            if start < 0 or length <= 0:
                return None
            off_tuple = (start, length)
        if not isinstance(adapter, WeightAdapterBase):
            return None
        if getattr(adapter, "name", "lora") != "lora":
            return None
        weights = getattr(adapter, "weights", None)
        if weights is None or len(weights) < 3:
            return None
        mat_up = weights[0]   # lora_B [out, rank]
        mat_dn = weights[1]   # lora_A [rank, in]
        alpha = weights[2]
        if len(weights) > 3 and any(w is not None for w in weights[3:6]):
            return None
        if getattr(mat_up, "ndim", 0) > 2 and all(int(s) == 1 for s in mat_up.shape[2:]):
            mat_up = mat_up.reshape(mat_up.shape[0], mat_up.shape[1])
        if getattr(mat_dn, "ndim", 0) > 2 and all(int(s) == 1 for s in mat_dn.shape[2:]):
            mat_dn = mat_dn.reshape(mat_dn.shape[0], mat_dn.shape[1])
        if getattr(mat_up, "ndim", 0) != 2 or getattr(mat_dn, "ndim", 0) != 2:
            return None
        if int(mat_up.shape[1]) != int(mat_dn.shape[0]):
            return None
        if off_tuple is not None and int(mat_up.shape[0]) != off_tuple[1]:
            return None
        try:
            alpha_scale = float(alpha) / float(mat_dn.shape[0]) if alpha is not None else 1.0
        except Exception:
            alpha_scale = 1.0
        terms.append((mat_dn, mat_up, float(strength) * alpha_scale, off_tuple))
    return terms or None


def _patcher_is_w4a8_pack(patcher) -> bool:
    model = getattr(patcher, "model", None)
    if model is None:
        return False
    if getattr(model, _PACK_TAG, False):
        return True
    for _name, mod in model.named_modules():
        if getattr(mod, "quant_format", None) == "asym_w4a8_int8":
            try:
                model._hswq_w4a8_pack = True
            except Exception:
                pass
            return True
    return False


def bake_w4a8_patches_on_patcher(patcher) -> dict:
    """Move LoRA patches off W4A8 weights and onto the forward residual path."""
    from .flux1_w4a8_forward import (
        clear_w4a8_residuals,
        log_w4a8_lora_bake_evidence,
        set_w4a8_residual_terms,
    )

    model = getattr(patcher, "model", None)
    if model is None or not _patcher_is_w4a8_pack(patcher):
        return {"baked": 0, "skipped": 0}

    patches = getattr(patcher, "patches", {}) or {}
    wrapper_patches = getattr(patcher, "weight_wrapper_patches", {}) or {}
    keys = [k for k in list(patches.keys()) if k.endswith(".weight")]
    baked_keys = []
    skipped = 0

    for key in keys:
        module = _module_for_key(model, key)
        if module is None or getattr(module, "quant_format", None) != "asym_w4a8_int8":
            skipped += 1
            continue
        if key in wrapper_patches:
            raise RuntimeError(
                f"[HSWQ W4A8] weight_wrapper patches are not supported on a "
                f"4-bit pack (key={key}); the delta would be rounded away"
            )
        terms = _extract_w4a8_lora_terms(patcher, key)
        if terms is None:
            raise RuntimeError(
                f"[HSWQ W4A8] LoRA patch on {key} is not expressible as plain "
                f"LoRA factors (adapter/offset/function); refusing to patch the "
                f"packed 4-bit weight"
            )
        set_w4a8_residual_terms(module, terms, key=key)
        patches.pop(key, None)
        baked_keys.append(key)

    if baked_keys:
        clear_w4a8_residuals(model, keep_keys=set(baked_keys))

    _COUNTERS["bakes"] += 1
    _COUNTERS["keys_baked"] += len(baked_keys)
    _COUNTERS["keys_skipped"] += skipped
    if baked_keys:
        _console(
            f"[HSWQ W4A8 LoRA] baked {len(baked_keys)} W4A8 layers as residuals "
            f"(skipped non-W4A8 keys={skipped})"
        )
        log_w4a8_lora_bake_evidence("after bake")
    return {"baked": len(baked_keys), "skipped": skipped}


def _log_residual_evidence_before_load() -> None:
    """Print the residual counters before the next model load.

    The after-bake evidence line is taken before the first forward, so it can
    only prove that terms were attached. Sampling then runs, and the next model
    load (VAE / safety checker) is the first point where a finished run can be
    reported with its real residual count.
    """
    try:
        from .flux1_w4a8_forward import (
            log_w4a8_lora_bake_evidence,
            w4a8_forward_stats,
        )

        if int(w4a8_forward_stats().get("residual_hits") or 0) <= 0:
            return
        log_w4a8_lora_bake_evidence("before model load")
    except Exception:  # noqa: BLE001 - evidence must never break a load
        logger.debug("[HSWQ W4A8 LoRA] residual evidence skipped", exc_info=True)


def install_w4a8_lora_bake(force: bool = False) -> bool:
    """Wrap ModelPatcherDynamic.load and mm.load_models_gpu with the bake hook."""
    try:
        import comfy.model_management as mm
        import comfy.model_patcher as mp
    except ImportError as e:
        logger.warning("[HSWQ W4A8] comfy import failed for bake install: %s", e)
        return False

    Dynamic = getattr(mp, "ModelPatcherDynamic", None)
    if Dynamic is None:
        logger.warning("[HSWQ W4A8] ModelPatcherDynamic missing; bake not installed")
        return False

    cur = getattr(Dynamic, "load", None)
    if cur is None:
        return False
    if not force and getattr(cur, _DYNAMIC_TAG, False) and getattr(
        cur, _DYNAMIC_TAG + "_ver", 0
    ) >= _BAKE_HOOK_VER:
        return True
    if not getattr(cur, _DYNAMIC_TAG, False):
        orig_load = cur

        def _load_with_bake(self, *args, **kwargs):
            try:
                bake_w4a8_patches_on_patcher(self)
            except Exception:
                # A bake failure must stop the load: a silently un-baked 4-bit
                # pack renders wrong output, which is worse than an error.
                logger.exception("[HSWQ W4A8 LoRA] bake failed; aborting load")
                raise
            return orig_load(self, *args, **kwargs)

        _load_with_bake._hswq_w4a8_prev = orig_load  # type: ignore[attr-defined]
        _load_with_bake._hswq_w4a8_lora_bake = True  # type: ignore[attr-defined]
        _load_with_bake._hswq_w4a8_lora_bake_ver = _BAKE_HOOK_VER  # type: ignore[attr-defined]
        Dynamic.load = _load_with_bake

    gpu = getattr(mm, "load_models_gpu", None)
    if gpu is not None and (
        force
        or not getattr(gpu, _GPU_TAG, False)
        or getattr(gpu, _GPU_TAG + "_ver", 0) < _BAKE_HOOK_VER
    ):
        if not getattr(gpu, _GPU_TAG, False):
            orig_gpu = gpu

            def _load_models_gpu_with_bake(models, *args, **kwargs):
                _log_residual_evidence_before_load()
                patchers = models if isinstance(models, (list, tuple)) else [models]
                for patcher in patchers:
                    try:
                        bake_w4a8_patches_on_patcher(patcher)
                    except Exception:
                        logger.exception(
                            "[HSWQ W4A8 LoRA] bake failed in load_models_gpu"
                        )
                        raise
                return orig_gpu(models, *args, **kwargs)

            _load_models_gpu_with_bake._hswq_w4a8_prev = orig_gpu  # type: ignore[attr-defined]
            _load_models_gpu_with_bake._hswq_w4a8_gpu_bake = True  # type: ignore[attr-defined]
            _load_models_gpu_with_bake._hswq_w4a8_gpu_bake_ver = _BAKE_HOOK_VER  # type: ignore[attr-defined]
            mm.load_models_gpu = _load_models_gpu_with_bake

    _console(
        f"[HSWQ W4A8] LoRA bake installed (ver={_BAKE_HOOK_VER}; "
        "patch -> factorised residual, packed weight untouched)"
    )
    return True


def uninstall_w4a8_lora_bake() -> bool:
    """Peel the bake hooks (purge path)."""
    try:
        import comfy.model_management as mm
        import comfy.model_patcher as mp
    except ImportError:
        return False
    peeled = False
    Dynamic = getattr(mp, "ModelPatcherDynamic", None)
    cur = getattr(Dynamic, "load", None) if Dynamic is not None else None
    if cur is not None and getattr(cur, _DYNAMIC_TAG, False):
        Dynamic.load = cur._hswq_w4a8_prev
        peeled = True
    gpu = getattr(mm, "load_models_gpu", None)
    if gpu is not None and getattr(gpu, _GPU_TAG, False):
        mm.load_models_gpu = gpu._hswq_w4a8_prev
        peeled = True
    return peeled


def reset_w4a8_lora_bake_log_counters() -> None:
    _COUNTERS["bakes"] = 0
    _COUNTERS["keys_baked"] = 0
    _COUNTERS["keys_skipped"] = 0


def w4a8_lora_bake_counters() -> dict:
    return dict(_COUNTERS)
