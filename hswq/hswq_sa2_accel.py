"""SageAttention2 (SA2) acceleration for HSWQ-loaded DiT models.

Scope (2026-09-10, Owner directive): the 4 supported checkpoint kinds loaded
through ``HSWQFP8E4M3UNetLoader`` with ``attention_accel == "sa2"``:

    1. Z Image  ConvRot INT8
    2. Z Image  ConvRot NVFP4
    3. Krea2    ConvRot INT8
    4. Krea2    ConvRot NVFP4

SEPARATION PRINCIPLE (Owner directive, 2026-09-10 - "4 patterns, absolutely
no mixing"): the (architecture x quant-format) matrix is dispatched through
EXPLICIT per-pattern functions. There is one dedicated arm function per
pattern (``_arm_zimage_int8`` / ``_arm_zimage_nvfp4`` / ``_arm_krea2_int8`` /
``_arm_krea2_nvfp4``). Detection is done from the checkpoint itself with the
existing proven probes, cross-checked against the loader option the user
picked. A mismatch NEVER falls through silently: SA2 is refused and the
mismatch is logged.

Pattern -> attention module mapping (each pattern patches ONLY its own module):

    Z Image  (int8 / nvfp4) -> comfy.ldm.lumina.model      (NextDiT)
    Krea2    (int8 / nvfp4) -> comfy.ldm.krea2.model       (SingleStreamDiT)

Layout handling is the validated one (benchmark/zi_traj_compare.py, 2026-09-10):
stock ``attention_pytorch`` semantics, ``skip_reshape=True`` takes q,k,v as
[B,H,N,D]; output needs ``transpose(1,2)`` BEFORE ``reshape`` to [B,N,H*D].
SA2 kernel: ``sageattn`` sm120 auto path = INT8 QK per_warp + FP8 PV
(fp32+fp16 accum). SDPA fallback for mask / head_dim > 256 / kernel errors.
"""

import logging
import threading

logger = logging.getLogger("hswq.sa2")

_STATE = {
    "armed": False,
    "calls": 0,
    "sa2": 0,
    "fb_mask": 0,
    "fb_dim": 0,
    "err": 0,
    "t_sa2_ms": 0.0,
    "t_sdpa_ms": 0.0,
    "patched_modules": (),
    "pattern": None,
}
_LOCK = threading.Lock()

# Pattern -> the ONE arch module whose from-import binding of
# optimized_attention_masked must be re-pointed while armed.
# Explicit per pattern; no shared/global patching.
_PATTERN_ATTENTION_MODULE = {
    "zimage_int8": "comfy.ldm.lumina.model",
    "zimage_nvfp4": "comfy.ldm.lumina.model",
    "krea2_int8": "comfy.ldm.krea2.model",
    "krea2_nvfp4": "comfy.ldm.krea2.model",
}

# Loader weight_dtype options mapped to the pattern they declare.
_DTYPE_PATTERN = {
    "int8_tensorwise": None,  # could be Z Image or Krea2 -> resolved by probe
    "Z Image ConvRot NVFP4": "zimage_nvfp4",
    "Krea2 ConvRot NVFP4": "krea2_nvfp4",
}


def _reset_stats():
    for k in ("calls", "sa2", "fb_mask", "fb_dim", "err"):
        _STATE[k] = 0
    _STATE["t_sa2_ms"] = 0.0
    _STATE["t_sdpa_ms"] = 0.0


def get_stats() -> dict:
    return dict(_STATE)


def get_pattern() -> str | None:
    return _STATE.get("pattern")


def _log_stats():
    s = get_stats()
    logger.info(
        "[HSWQ SA2] attention calls: total=%d sa2=%d fallback(mask)=%d fallback(dim)=%d errors=%d",
        s["calls"], s["sa2"], s["fb_mask"], s["fb_dim"], s["err"],
    )
    logger.info(
        "[HSWQ SA2] attention time: sage2=%.1f ms sdpa_fallback=%.1f ms",
        s["t_sa2_ms"], s["t_sdpa_ms"],
    )


# ---------------------------------------------------------------------------
# Checkpoint probes (existing proven helpers; each pattern uses its own).
# ---------------------------------------------------------------------------

def _probe_is_int8(unet_path: str) -> bool:
    from ..patches.comfy_quant_int8 import checkpoint_looks_like_comfy_quant_int8

    return checkpoint_looks_like_comfy_quant_int8(unet_path)


def _probe_is_nvfp4(unet_path: str) -> bool:
    from ..nodes.nvfp4.nvfp4_conf import checkpoint_looks_like_comfy_quant_nvfp4

    return checkpoint_looks_like_comfy_quant_nvfp4(unet_path)


def _probe_is_krea2(unet_path: str) -> bool:
    from ..patches.comfy_quant_int8 import checkpoint_is_krea2

    return checkpoint_is_krea2(unet_path)


def resolve_pattern(unet_path: str, weight_dtype: str) -> str | None:
    """Resolve the explicit (arch x quant) pattern from the checkpoint itself.

    "Hybrid" NVFP4 checkpoints (Owner's TC/hybrid builds) carry BOTH conf
    formats: int8_tensorwise on most layers + nvfp4 on the TensorCore/GEMM
    layers. The discriminator is therefore: any nvfp4 layer => NVFP4 pattern
    (the int8 layers are that pattern's activation-quantized Linear path, not
    a different pattern). Cross-checks against the loader option; returns None
    on any mismatch.
    """
    has_nvfp4 = _scan_conf_formats(unet_path).get("nvfp4", 0) > 0
    is_int8 = _probe_is_int8(unet_path)
    is_krea2 = _probe_is_krea2(unet_path)

    if has_nvfp4:
        pattern = "krea2_nvfp4" if is_krea2 else "zimage_nvfp4"
    elif is_int8:
        pattern = "krea2_int8" if is_krea2 else "zimage_int8"
    else:
        logger.warning("[HSWQ SA2] checkpoint is neither ConvRot INT8 nor ConvRot NVFP4; refusing: %s", unet_path)
        return None

    # Cross-check against the dtype option the user selected.
    declared = _DTYPE_PATTERN.get(weight_dtype)
    if declared is not None and declared != pattern:
        logger.warning(
            "[HSWQ SA2] dtype option %r declares pattern %r but checkpoint probes as %r; refusing",
            weight_dtype, declared, pattern,
        )
        return None
    if weight_dtype == "int8_tensorwise" and has_nvfp4:
        logger.warning("[HSWQ SA2] dtype int8_tensorwise but checkpoint carries nvfp4 layers; refusing")
        return None

    return pattern


def _scan_conf_formats(unet_path: str) -> dict:
    """Count comfy_quant conf formats present in the checkpoint (hybrid-safe)."""
    import collections
    import json

    from safetensors import safe_open

    cnt = collections.Counter()
    try:
        with safe_open(unet_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if not k.endswith(".comfy_quant"):
                    continue
                raw = f.get_tensor(k)
                try:
                    conf = json.loads(bytes(raw).decode("utf-8"))
                    fmt = conf.get("format")
                    if fmt:
                        cnt[fmt] += 1
                except Exception:
                    continue
    except Exception as e:
        logger.warning("[HSWQ SA2] conf scan failed for %s: %s", unet_path, e)
    return dict(cnt)


# ---------------------------------------------------------------------------
# Shared kernel construction (math-identical for all patterns; the SEPARATION
# lives in arm/dispatch functions below, not in duplicated kernel bodies).
# ---------------------------------------------------------------------------

def _make_attention_sage2():
    import time

    import torch
    from sageattention import sageattn
    from torch.nn.functional import scaled_dot_product_attention as _sdpa

    def attention_sage2(q, k, v, heads, mask=None, attn_precision=None,
                        skip_reshape=False, skip_output_reshape=False, **kw):
        if not _STATE["armed"]:
            return _STATE["_stock_fn"](
                q, k, v, heads, mask=mask, attn_precision=attn_precision,
                skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape,
                **kw
            )

        in_dtype = v.dtype
        if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
            q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)

        if skip_reshape:
            b, _, _, dim_head = q.shape
            qh, kh, vh = q, k, v
        else:
            b, n_q, _ = q.shape
            dim_head = q.shape[-1] // heads
            qh = q.view(b, n_q, heads, dim_head).transpose(1, 2)
            kh = k.view(b, k.shape[1], heads, dim_head).transpose(1, 2)
            vh = v.view(b, v.shape[1], heads, dim_head).transpose(1, 2)

        use_fallback = (mask is not None) or (dim_head > 256)
        _STATE["calls"] += 1
        torch.cuda.synchronize()
        if use_fallback:
            if mask is not None:
                _STATE["fb_mask"] += 1
            else:
                _STATE["fb_dim"] += 1
            t0 = time.perf_counter()
            out = _sdpa(qh, kh, vh, attn_mask=mask, is_causal=False)
            torch.cuda.synchronize()
            _STATE["t_sdpa_ms"] += (time.perf_counter() - t0) * 1000.0
        else:
            try:
                t0 = time.perf_counter()
                out = sageattn(qh, kh, vh, tensor_layout="HND", is_causal=False)
                torch.cuda.synchronize()
                _STATE["t_sa2_ms"] += (time.perf_counter() - t0) * 1000.0
                _STATE["sa2"] += 1
            except Exception as e:
                _STATE["err"] += 1
                logger.warning("[HSWQ SA2] kernel fallback (%s: %s)", type(e).__name__, str(e)[:80])
                out = _sdpa(qh, kh, vh, attn_mask=mask, is_causal=False)

        if skip_output_reshape:
            pass
        else:
            out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out.to(in_dtype)

    return attention_sage2


def _patch_modules(module_names, attention_sage2) -> list:
    import importlib

    import comfy.ldm.modules.attention as comfy_attention

    patched = []
    if _STATE.get("_stock_fn") is None:
        _STATE["_stock_fn"] = comfy_attention.optimized_attention_masked

    for mod_name in module_names:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "optimized_attention_masked"):
                mod.optimized_attention_masked = attention_sage2
                patched.append(mod_name)
        except Exception as e:
            logger.warning("[HSWQ SA2] patch %s failed: %s", mod_name, e)
    return patched


def _wrap_forward(model, pattern: str, patched) -> bool:
    """Arm per-MODEL: wrap this diffusion_model's forward so SA2 is live only
    while THIS model runs. Returns True on success."""
    inner = getattr(model, "model", None)
    dm = getattr(inner, "diffusion_model", None)
    if dm is None:
        logger.warning("[HSWQ SA2] no diffusion_model on MODEL; skip")
        return False

    _STATE["patched_modules"] = tuple(patched)
    _STATE["pattern"] = pattern

    if getattr(dm, "_hswq_sa2_wrapped", False):
        _STATE["armed"] = True
        return True

    original_forward = dm.forward

    def sa2_forward(*args, **kwargs):
        _STATE["armed"] = True
        try:
            return original_forward(*args, **kwargs)
        finally:
            _STATE["armed"] = False

    dm.forward = sa2_forward
    dm._hswq_sa2_wrapped = True
    dm._hswq_sa2_original_forward = original_forward
    _STATE["armed"] = True
    return True


# ---------------------------------------------------------------------------
# EXPLICIT per-pattern arm functions. 4 patterns, 4 functions, zero mixing.
# Each validates the checkpoint kind AND the loaded model class before arming.
# ---------------------------------------------------------------------------

def _expected_class_name_ok(pattern: str, dm) -> bool:
    cls = type(dm).__name__
    mod = type(dm).__module__
    if pattern.startswith("zimage"):
        return "lumina" in mod.lower() and "nextdit" in cls.lower()
    if pattern.startswith("krea2"):
        return "krea2" in mod.lower() and "singlestreamdit" in cls.lower()
    return False


def _arm_pattern(model, unet_path: str, pattern: str) -> bool:
    probes = {
        "zimage_int8": (lambda: _probe_is_int8(unet_path) and not _probe_is_krea2(unet_path), _probe_is_int8),
        "zimage_nvfp4": (lambda: _probe_is_nvfp4(unet_path) and not _probe_is_krea2(unet_path), _probe_is_nvfp4),
        "krea2_int8": (lambda: _probe_is_int8(unet_path) and _probe_is_krea2(unet_path), _probe_is_int8),
        "krea2_nvfp4": (lambda: _probe_is_nvfp4(unet_path) and _probe_is_krea2(unet_path), _probe_is_nvfp4),
    }
    if pattern not in probes:
        logger.warning("[HSWQ SA2] unknown pattern %r; refusing", pattern)
        return False

    check, _ = probes[pattern]
    if not check():
        logger.warning(
            "[HSWQ SA2] checkpoint does not match pattern %s; refusing: %s", pattern, unet_path
        )
        return False

    attention_module = _PATTERN_ATTENTION_MODULE[pattern]
    attention_sage2 = _make_attention_sage2()
    patched = _patch_modules(["comfy.ldm.modules.attention", attention_module], attention_sage2)
    if not patched:
        return False

    inner = getattr(model, "model", None)
    dm = getattr(inner, "diffusion_model", None)
    if dm is not None and not _expected_class_name_ok(pattern, dm):
        _unpatch(patched)
        logger.warning(
            "[HSWQ SA2] loaded model class %s (%s) does not match pattern %s; refusing",
            type(dm).__name__, type(dm).__module__, pattern,
        )
        return False

    return _wrap_forward(model, pattern, patched)


def _arm_zimage_int8(model, unet_path: str) -> bool:
    """Pattern 1: Z Image ConvRot INT8 -> comfy.ldm.lumina.model."""
    return _arm_pattern(model, unet_path, "zimage_int8")


def _arm_zimage_nvfp4(model, unet_path: str) -> bool:
    """Pattern 2: Z Image ConvRot NVFP4 -> comfy.ldm.lumina.model."""
    return _arm_pattern(model, unet_path, "zimage_nvfp4")


def _arm_krea2_int8(model, unet_path: str) -> bool:
    """Pattern 3: Krea2 ConvRot INT8 -> comfy.ldm.krea2.model."""
    return _arm_pattern(model, unet_path, "krea2_int8")


def _arm_krea2_nvfp4(model, unet_path: str) -> bool:
    """Pattern 4: Krea2 ConvRot NVFP4 -> comfy.ldm.krea2.model."""
    return _arm_pattern(model, unet_path, "krea2_nvfp4")


_PATTERN_ARM = {
    "zimage_int8": _arm_zimage_int8,
    "zimage_nvfp4": _arm_zimage_nvfp4,
    "krea2_int8": _arm_krea2_int8,
    "krea2_nvfp4": _arm_krea2_nvfp4,
}


def _unpatch(patched_modules):
    import importlib

    for mod_name in patched_modules:
        try:
            mod = importlib.import_module(mod_name)
            if _STATE.get("_stock_fn") is not None:
                mod.optimized_attention_masked = _STATE["_stock_fn"]
        except Exception as e:
            logger.warning("[HSWQ SA2] unpatch %s failed: %s", mod_name, e)


def sa2_arm_for_model(model, unet_path: str, weight_dtype: str) -> bool:
    """Arm SA2 for THIS MODEL only, via the explicit per-pattern functions.

    ``model`` is a ComfyUI ModelPatcher as returned by the loader node.
    Returns True when SA2 was installed for one of the 4 supported patterns,
    False when the checkpoint/option combination is not supported (the loader
    then just returns the stock model).
    """
    pattern = resolve_pattern(unet_path, weight_dtype)
    if pattern is None:
        return False

    arm_fn = _PATTERN_ARM[pattern]
    with _LOCK:
        _reset_stats()
        ok = arm_fn(model, unet_path)
        if ok:
            logger.info(
                "[HSWQ SA2] armed pattern=%s dtype=%s; patched: %s",
                pattern, weight_dtype, ", ".join(_STATE["patched_modules"]),
            )
        else:
            _STATE["pattern"] = None
            _unpatch(list(_STATE.get("patched_modules") or ()))
            _STATE["patched_modules"] = ()
        return ok


def sa2_report_stats():
    if _STATE["patched_modules"]:
        _log_stats()
