"""Flux1 ConvRot W4A8 arm — attach the residual add to the mixed-precision Linear.

ComfyUI core owns detect / load / forward for ``asym_w4a8_int8`` (measured:
``comfy/ops.py`` sets ``_orig_shape`` from the Linear ctor, validates the packed
width, and builds ``AsymW4A8Int8Layout.Params``), so this package does NOT
replace ``ops._load_quantized_module``. The only thing core cannot do is add a
LoRA delta that 4-bit codes would round away; that lives in the forward wrap.

Arming order matters: ``comfy.ops.mixed_precision_ops`` is the factory every
model load goes through, so wrapping it once means each freshly built Linear
gets the residual-aware forward.
"""
from __future__ import annotations

import logging

from .flux1_w4a8_forward import attach_w4a8_linear_lora_bake

logger = logging.getLogger(__name__)

_ARM_VER = 1
_ARMED = False


def _console(msg: str) -> None:
    print(msg, flush=True)
    logger.info(msg)


def apply_comfy_quant_w4a8_patches() -> bool:
    """Wrap ``ops.mixed_precision_ops`` so W4A8 Linear forwards add residuals."""
    global _ARMED
    try:
        import comfy.ops as ops
    except Exception as e:  # noqa: BLE001 - comfy may not be importable yet
        logger.warning("[HSWQ W4A8] comfy import failed: %s", e)
        return False

    mp_fn = getattr(ops, "mixed_precision_ops", None)
    if mp_fn is None:
        logger.warning("[HSWQ W4A8] ops.mixed_precision_ops missing")
        return False
    if getattr(mp_fn, "_hswq_w4a8_stack_ver", 0) >= _ARM_VER:
        _ARMED = True
        return True

    prev = getattr(mp_fn, "_hswq_w4a8_orig_mp", None) or mp_fn

    def mixed_precision_ops_w4a8(*args, **kwargs):
        mp = prev(*args, **kwargs)
        attach_w4a8_linear_lora_bake(mp.Linear)
        return mp

    mixed_precision_ops_w4a8._hswq_w4a8_orig_mp = prev  # type: ignore[attr-defined]
    mixed_precision_ops_w4a8._hswq_w4a8_stack_ver = _ARM_VER  # type: ignore[attr-defined]
    ops.mixed_precision_ops = mixed_precision_ops_w4a8
    _ARMED = True
    _console(
        f"[HSWQ W4A8] armed (ver={_ARM_VER}; stock grouped-int8 GEMM kept, "
        "LoRA delta added as a factorised residual)"
    )
    return True


def uninstall_comfy_quant_w4a8_patches() -> bool:
    """Restore the stock factory (purge path)."""
    global _ARMED
    try:
        import comfy.ops as ops
    except Exception:
        return False
    mp_fn = getattr(ops, "mixed_precision_ops", None)
    if mp_fn is None or not getattr(mp_fn, "_hswq_w4a8_stack_ver", 0):
        return False
    ops.mixed_precision_ops = mp_fn._hswq_w4a8_orig_mp
    _ARMED = False
    return True


def is_armed() -> bool:
    return _ARMED
