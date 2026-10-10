"""Krea2 ConvRot INT8 runtime stack (INT8-only).

Installs the Krea2-only `mixed_precision_ops` wrap that runs every Linear
through the stock INT8 forward (comfy_kitchen ``int8_linear`` + online ConvRot
rotation) plus the low-rank LoRA residual re-add (``int8_forward``). The
residual itself is produced by ``int8_lora_bake`` on Dynamic.load.

Removed together with the retired Krea2 ConvRot NVFP4 format: NVFP4 load
routing (``_load_quantized_module`` wrap), packed-K / txtlayers detection
fixes, the addmm handler, and the kitchen stub repair. INT8 packs never
contained NVFP4-armed layers, so those code paths were no-ops for them
(verified: each was gated on ``is_nvfp4_conf`` / ``_hswq_nvfp4_convrot``,
which the INT8 loader never sets).

Shared peel protocol (SDXL / Z Image read these - DO NOT rename):
``_hswq_nvfp4_stack_ver`` (=2), ``_hswq_nvfp4_orig_mp``,
``_hswq_nvfp4_full_forward`` on the wrap + forward, plus the Krea2-only
``_hswq_krea2_stack`` / ``_hswq_krea2_int8`` stamps.

Runtime only - never permanently edit ComfyUI-master.
"""
from __future__ import annotations

import logging

from .int8_forward import (
    int8_forward_stats,
    make_int8_linear_forward,
    reset_int8_forward_stats,
)

logger = logging.getLogger(__name__)
_PATCHES_APPLIED = False

# External peel contract: SDXL/ZI walkers accept the live chain only
# when _hswq_nvfp4_stack_ver >= 2 (nodes/nvfp4 + nodes/zimage_nvfp4 read it).
# This value is part of their protocol - never change it here.
_KREA2_INT8_STACK_VER = 2

__all__ = [
    "apply_comfy_quant_int8_patches",
    "int8_forward_stats",
    "reset_int8_forward_stats",
]


def _console(msg: str) -> None:
    print(msg, flush=True)
    logger.info(msg)


def _closure_named(fn, name):
    """Free-variable lookup (same convention as the other HSWQ stacks)."""
    if fn is None or getattr(fn, "__closure__", None) is None:
        return None
    for n, cell in zip(fn.__code__.co_freevars, fn.__closure__):
        if n == name:
            return cell.cell_contents
    return None


def _unwrap_foreign_forward_to_stock(fwd):
    """Walk foreign HSWQ Linear.forward wraps down to the true stock forward.

    SDXL TC / ZI parity wraps close over ``stock_forward`` and stamp
    ``_hswq_nvfp4_full_forward`` / ``_hswq_nvfp4_convrot_parity``.
    """
    cur = fwd
    seen: set[int] = set()
    for _ in range(8):
        if cur is None or not callable(cur) or id(cur) in seen:
            return None
        seen.add(id(cur))
        is_ours = getattr(cur, "_hswq_krea2_int8", False)
        is_wrap = bool(
            getattr(cur, "_hswq_nvfp4_full_forward", False)
            or getattr(cur, "_hswq_nvfp4_convrot_parity", False)
        )
        if not is_wrap:
            return cur  # stock reached
        nxt = _closure_named(cur, "stock_forward")
        if nxt is None or nxt is cur:
            return None if is_ours else cur
        cur = nxt
    return None


def _mp_base_under_foreign(mp_fn):
    """Peel foreign HSWQ mp wraps (SDXL product / ZI parity / stale ours).

    Returns the factory to chain under (INT8 force-conv wrap or stock), or
    None when the chain cannot be walked safely.
    """
    cur = mp_fn
    seen: set[int] = set()
    for _ in range(12):
        if cur is None or not callable(cur) or id(cur) in seen:
            return None
        seen.add(id(cur))
        if getattr(cur, "_hswq_krea2_stack", False):
            nxt = getattr(cur, "_hswq_nvfp4_orig_mp", None)
            if nxt is None or nxt is cur:
                return None
            cur = nxt
            continue
        if getattr(cur, "_hswq_int8_conv_patched", False):
            return cur  # INT8 base: keep (Conv2d forcing is stack-agnostic)
        foreign = bool(
            getattr(cur, "_hswq_nvfp4_comfy_only", False)
            or getattr(cur, "_hswq_nvfp4_product_tc", False)
            or (int(getattr(cur, "_hswq_nvfp4_stack_ver", 0) or 0) > 0)
        )
        if not foreign:
            return cur  # stock
        nxt = getattr(cur, "_hswq_nvfp4_orig_mp", None)
        if nxt is None:
            nxt = _closure_named(cur, "_cur_mp")  # ZI parity wrap
        if nxt is None:
            nxt = _closure_named(cur, "_orig_mp")  # ZI upgraded wrap closure
        if nxt is None or nxt is cur:
            return None
        cur = nxt
    return None


def apply_comfy_quant_int8_patches() -> bool:
    """Install the Krea2 INT8-only mp stack (stock forward + LoRA residual).

    Idempotent AND coexisting: safe to call before every Krea2 INT8 load no
    matter which stack (SDXL TC / ZI parity / INT8) is currently live.
    """
    global _PATCHES_APPLIED
    try:
        import comfy.ops as ops
    except Exception as e:
        logger.warning("[HSWQ INT8] comfy import failed: %s", e)
        return False

    if not getattr(ops.mixed_precision_ops, "_hswq_krea2_stack", False):
        _orig_mp = _mp_base_under_foreign(ops.mixed_precision_ops)
        if _orig_mp is None:
            logger.error(
                "[HSWQ INT8] krea2: cannot resolve mixed_precision_ops base "
                "under foreign wraps; refusing to load with a foreign stack"
            )
            return False

        def mixed_precision_ops_patched(*args, **kwargs):
            mp = _orig_mp(*args, **kwargs)
            Lin = mp.Linear
            fwd = getattr(Lin, "forward", None)
            if getattr(fwd, "_hswq_krea2_int8", False):
                return mp  # ours already on this freshly-built class
            if fwd is not None and (
                getattr(fwd, "_hswq_nvfp4_full_forward", False)
                or getattr(fwd, "_hswq_nvfp4_convrot_parity", False)
            ):
                # Foreign TC / parity forward ended up under us (chain case):
                # unwrap to true stock, then take over.
                stock = _unwrap_foreign_forward_to_stock(fwd)
                if stock is not None:
                    Lin.forward = make_int8_linear_forward(stock)
                    return mp
                return mp  # cannot unwrap safely; leave inner stack intact
            Lin.forward = make_int8_linear_forward(fwd)
            return mp

        mixed_precision_ops_patched._hswq_nvfp4_full_forward = True  # type: ignore[attr-defined]
        mixed_precision_ops_patched._hswq_nvfp4_stack_ver = _KREA2_INT8_STACK_VER  # type: ignore[attr-defined]
        mixed_precision_ops_patched._hswq_nvfp4_orig_mp = _orig_mp  # type: ignore[attr-defined]
        mixed_precision_ops_patched._hswq_krea2_stack = True  # type: ignore[attr-defined]
        ops.mixed_precision_ops = mixed_precision_ops_patched
        _console(
            "[HSWQ INT8] krea2 mp stack installed "
            "(foreign wraps peeled; stock INT8 forward + LoRA residual)"
        )

    _PATCHES_APPLIED = True
    _console(
        "[HSWQ INT8] krea2 stack ready "
        "(coexisting: SDXL TC / ZI parity / INT8 may re-wire on their loads)"
    )
    return True
