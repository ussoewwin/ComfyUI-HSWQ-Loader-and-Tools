"""SageAttention2 (SA2) acceleration for HSWQ-loaded DiT models.

Scope (2026-09-10, Owner directive): Z Image ConvRot INT8 / NVFP4 and
Krea2 ConvRot INT8 / NVFP4 loaded through ``HSWQFP8E4M3UNetLoader`` with the
``attention_accel`` selector set to ``sa2``.

Design (separation principle - no shared/global monkeypatch):
  * ``sa2_arm_for_model(model)`` wraps the *specific* ``diffusion_model``
    instance's ``forward`` so SA2 is armed only while THAT model runs.
  * The armed attention function patches ONLY the modules bound to the
    architecture that model actually uses (dispatch by class name):
      - Z Image  -> comfy.ldm.lumina.model      (JointAttention, skip_reshape)
      - Krea2    -> comfy.ldm.krea2.model       (skip_reshape=True)
      - Qwen Image -> comfy.ldm.qwen_image.model (skip_reshape=True, reserved)
    Other architectures keep stock attention; other MODEL objects in the same
    graph are never affected because their forwards do not arm the flag.

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
}
_LOCK = threading.Lock()

# Architecture modules whose from-import binding of optimized_attention_masked
# must be re-pointed while armed. Dispatch is explicit per arch (no shared path).
_ARCH_MODULES = {
    "zimage": "comfy.ldm.lumina.model",
    "krea2": "comfy.ldm.krea2.model",
    "qwen_image": "comfy.ldm.qwen_image.model",
}


def _reset_stats():
    for k in ("calls", "sa2", "fb_mask", "fb_dim", "err"):
        _STATE[k] = 0
    _STATE["t_sa2_ms"] = 0.0
    _STATE["t_sdpa_ms"] = 0.0


def get_stats() -> dict:
    return dict(_STATE)


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


def _detect_arch(diffusion_model) -> str | None:
    """Detect architecture by the DiT module's class name (explicit dispatch)."""
    cls_name = type(diffusion_model).__name__.lower()
    mod_name = type(diffusion_model).__module__.lower()
    # Z Image (NextDiT in comfy.ldm.lumina) / Krea2 / Qwen Image - check module path first.
    if "lumina" in mod_name:
        return "zimage"
    if "krea2" in mod_name:
        return "krea2"
    if "qwen_image" in mod_name or "qwenimage" in mod_name:
        return "qwen_image"
    # Fallback by class name only if module path was inconclusive.
    if "nextdit" in cls_name or "lumina" in cls_name:
        return "zimage"
    if "krea" in cls_name:
        return "krea2"
    if "qwen" in cls_name:
        return "qwen_image"
    return None


def _make_attention_sage2():
    import time

    import torch
    from sageattention import sageattn
    from torch.nn.functional import scaled_dot_product_attention as _sdpa

    def attention_sage2(q, k, v, heads, mask=None, attn_precision=None,
                        skip_reshape=False, skip_output_reshape=False, **kw):
        if not _STATE["armed"]:
            # Not armed (model without SA2 running while modules still patched):
            # delegate to the stock backend captured at arm time.
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


def _patch_arch_modules(arch: str, attention_sage2) -> list:
    """Re-point optimized_attention_masked on the core module + the ONE arch
    module that owns this model. Returns the list of patched module names."""
    import importlib

    import comfy.ldm.modules.attention as comfy_attention

    patched = []
    stock = comfy_attention.optimized_attention_masked
    if _STATE.get("_stock_fn") is None:
        _STATE["_stock_fn"] = stock

    names = ["comfy.ldm.modules.attention"]
    target = _ARCH_MODULES.get(arch)
    if target:
        names.append(target)

    for mod_name in names:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "optimized_attention_masked"):
                mod.optimized_attention_masked = attention_sage2
                patched.append(mod_name)
        except Exception as e:
            logger.warning("[HSWQ SA2] patch %s failed: %s", mod_name, e)
    return patched


def _unpatch_arch_modules(patched_modules):
    import importlib

    for mod_name in patched_modules:
        try:
            mod = importlib.import_module(mod_name)
            if _STATE.get("_stock_fn") is not None:
                mod.optimized_attention_masked = _STATE["_stock_fn"]
        except Exception as e:
            logger.warning("[HSWQ SA2] unpatch %s failed: %s", mod_name, e)


def sa2_arm_for_model(model) -> bool:
    """Arm SA2 for THIS MODEL only (wrap its diffusion_model.forward).

    ``model`` is a ComfyUI ModelPatcher as returned by a loader node.
    Returns True when SA2 was installed, False when the architecture is not
    supported (loader then just returns the stock model).
    """
    inner = getattr(model, "model", None)
    dm = getattr(inner, "diffusion_model", None)
    if dm is None:
        logger.warning("[HSWQ SA2] no diffusion_model on MODEL; skip")
        return False

    arch = _detect_arch(dm)
    if arch is None:
        logger.warning(
            "[HSWQ SA2] unsupported architecture %s (%s); SA2 not installed",
            type(dm).__name__, type(dm).__module__,
        )
        return False

    with _LOCK:
        _reset_stats()
        attention_sage2 = _make_attention_sage2()
        patched = _patch_arch_modules(arch, attention_sage2)
        if not patched:
            return False
        _STATE["patched_modules"] = tuple(patched)

        if getattr(dm, "_hswq_sa2_wrapped", False):
            # Already wrapped in a previous queue run; keep the wrapper.
            _STATE["armed"] = True
            logger.info("[HSWQ SA2] armed for %s (%s)", arch, type(dm).__name__)
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
        logger.info(
            "[HSWQ SA2] armed for %s (%s); patched modules: %s",
            arch, type(dm).__name__, ", ".join(patched),
        )
        return True


def sa2_report_stats():
    if _STATE["patched_modules"]:
        _log_stats()
