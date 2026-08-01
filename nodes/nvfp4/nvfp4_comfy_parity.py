"""Z Image / ZIT ConvRot NVFP4 — ComfyUI stock GEMM + online act rotate.

Ported from ``hswq/benchmark/nvfp4_comfy_parity.py`` (same math as
``zi_convrot_nvfp4_bench.py``). Product HSWQ Tensor Core Linear.forward
breaks Pixel SSIM on Z Image ConvRot packs; the bench path does not.

Call ``apply_nvfp4_comfy_parity()`` **after** ``apply_comfy_quant_nvfp4_patches()``
for UNet / Z Image loads. SDXL product path keeps TC via
``restore_nvfp4_tc_product_stack()`` before SDXL checkpoint load.

Does not edit ComfyUI-master.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_PARITY_APPLIED = False
_PRODUCT_LOAD: Optional[Callable] = None
_PRODUCT_MP: Optional[Callable] = None


def _console(msg: str) -> None:
    print(msg, flush=True)
    logger.info(msg)


def remember_nvfp4_tc_product_stack(load_fn, mp_fn) -> None:
    """Store SDXL product TC refs (call from apply_comfy_quant_nvfp4_patches only).

    Never overwrite with parity wrappers — SDXL must always be able to restore.
    """
    global _PRODUCT_LOAD, _PRODUCT_MP
    if load_fn is not None and getattr(load_fn, "_hswq_nvfp4_full_load", False):
        if not getattr(load_fn, "_hswq_nvfp4_comfy_only", False):
            _PRODUCT_LOAD = load_fn
    if mp_fn is not None and getattr(mp_fn, "_hswq_nvfp4_stack_ver", 0):
        if not getattr(mp_fn, "_hswq_nvfp4_comfy_only", False):
            _PRODUCT_MP = mp_fn


def is_nvfp4_comfy_parity_active() -> bool:
    return bool(_PARITY_APPLIED)


def _closure_named(fn, name: str):
    try:
        cells = fn.__closure__ or ()
        for n, c in zip(fn.__code__.co_freevars, cells):
            if n == name:
                return c.cell_contents
    except Exception:
        return None
    return None


def _unwrap_stock_forward(forward_fn):
    """Peel HSWQ TC wrappers until stock MixedPrecision Linear.forward."""
    f = forward_fn
    for _ in range(8):
        if not getattr(f, "_hswq_nvfp4_full_forward", False):
            return f
        stock = _closure_named(f, "stock_forward")
        if stock is None:
            return None
        f = stock
    return None


def _make_convrot_parity_forward(stock_forward):
    """Stock MixedPrecision forward + online act rotate for ConvRot NVFP4."""
    from .nvfp4_hadamard import build_hadamard, rotate_last_dim

    def forward_parity(self, input, *args, **kwargs):
        if getattr(self, "_hswq_nvfp4_convrot", False) and not getattr(
            self, "_full_precision_mm", False
        ):
            if not (input.requires_grad or getattr(self, "comfy_force_cast_weights", False)):
                gs = int(getattr(self, "_hswq_nvfp4_convrot_groupsize", 256) or 256)
                h = getattr(self, "_hswq_nvfp4_H", None)
                if h is None or h.device != input.device or h.dtype != input.dtype:
                    h = build_hadamard(gs, device=input.device, dtype=input.dtype)
                    self._hswq_nvfp4_H = h
                input = rotate_last_dim(input, h, gs)
        return stock_forward(self, input, *args, **kwargs)

    forward_parity._hswq_nvfp4_convrot_parity = True  # type: ignore[attr-defined]
    return forward_parity


def _arm_convrot_after_stock_load(module, conf) -> None:
    from .nvfp4_conf import convrot_flags_from_conf, is_nvfp4_conf

    if not is_nvfp4_conf(conf):
        return
    enabled, gs = convrot_flags_from_conf(conf)
    module._hswq_nvfp4_convrot = bool(enabled)
    module._hswq_nvfp4_convrot_groupsize = int(gs)
    # Do not set _hswq_nvfp4 (TC full-forward arm).


def require_convrot_parity_forward() -> None:
    """Fail fast if TC full-forward is still installed (bench guard)."""
    import comfy.ops as ops

    mp = ops.mixed_precision_ops()
    fwd = mp.Linear.forward
    if getattr(fwd, "_hswq_nvfp4_full_forward", False):
        raise RuntimeError(
            "ConvRot NVFP4 parity requires stock Comfy Linear.forward + act rotate; "
            "HSWQ TC full-forward is still installed (_hswq_nvfp4_full_forward)."
        )
    if not getattr(fwd, "_hswq_nvfp4_convrot_parity", False):
        raise RuntimeError(
            "ConvRot NVFP4 parity forward missing "
            "(_hswq_nvfp4_convrot_parity). Call apply_nvfp4_comfy_parity()."
        )


def restore_nvfp4_tc_product_stack() -> bool:
    """Put SDXL product TC load + forward back. No-op if already on TC.

    Z Image parity must never leak into SDXL. Call this from the SDXL loader
    only — do not change SDXL's TC / LoRA bake behavior.
    """
    global _PARITY_APPLIED
    try:
        import comfy.ops as ops
    except Exception as e:
        logger.warning("[HSWQ NVFP4] restore TC stack skipped: %s", e)
        return False

    mp = ops.mixed_precision_ops
    already_tc = (
        getattr(mp, "_hswq_nvfp4_stack_ver", 0)
        and not getattr(mp, "_hswq_nvfp4_comfy_only", False)
        and not getattr(ops._load_quantized_module, "_hswq_nvfp4_comfy_only", False)
    )
    if already_tc and not _PARITY_APPLIED:
        return True

    if _PRODUCT_LOAD is None or _PRODUCT_MP is None:
        if already_tc:
            _PARITY_APPLIED = False
            return True
        logger.warning(
            "[HSWQ NVFP4] restore TC stack: no saved product refs "
            "(SDXL needs apply_comfy_quant_nvfp4_patches first)"
        )
        return False

    ops._load_quantized_module = _PRODUCT_LOAD
    ops.mixed_precision_ops = _PRODUCT_MP
    _PARITY_APPLIED = False
    _console("[HSWQ NVFP4] restored product TC stack (SDXL path; parity off)")
    return True


def apply_nvfp4_comfy_parity() -> bool:
    """Switch NVFP4 Linear path to stock Comfy GEMM + online act rotate.

    Also registers aten.addmm for TensorCoreNVFP4Layout (kitchen gap).
    Saves product TC refs so SDXL can restore later.
    """
    global _PARITY_APPLIED, _PRODUCT_LOAD, _PRODUCT_MP
    try:
        import comfy.ops as ops
        from comfy.quant_ops import QUANT_ALGOS
    except Exception as e:
        logger.warning("[HSWQ NVFP4] comfy_parity import failed: %s", e)
        return False

    from .nvfp4_addmm_patch import register_nvfp4_addmm_handler
    from .nvfp4_conf import decode_comfy_quant_conf, is_nvfp4_conf
    from .nvfp4_forward import attach_nvfp4_linear_lora_bake

    register_nvfp4_addmm_handler()

    if "nvfp4" not in QUANT_ALGOS:
        logger.warning("[HSWQ NVFP4] comfy_parity: nvfp4 not in QUANT_ALGOS")
        return False

    patched_load = ops._load_quantized_module
    # Prefer refs already saved by apply_comfy_quant_nvfp4_patches (TC only).
    remember_nvfp4_tc_product_stack(patched_load, ops.mixed_precision_ops)

    if getattr(patched_load, "_hswq_nvfp4_comfy_only", False):
        # Already on parity load; still ensure forward + LoRA bake.
        _cur_mp = ops.mixed_precision_ops

        def mixed_precision_ops_parity_refresh(*args, **kwargs):
            mp = _cur_mp(*args, **kwargs)
            Lin = mp.Linear
            if getattr(Lin.forward, "_hswq_nvfp4_full_forward", False):
                stock = _unwrap_stock_forward(Lin.forward)
                if stock is not None:
                    Lin.forward = _make_convrot_parity_forward(stock)
            elif not getattr(Lin.forward, "_hswq_nvfp4_convrot_parity", False):
                Lin.forward = _make_convrot_parity_forward(Lin.forward)
            attach_nvfp4_linear_lora_bake(Lin)
            return mp

        mixed_precision_ops_parity_refresh._hswq_nvfp4_comfy_only = True  # type: ignore[attr-defined]
        mixed_precision_ops_parity_refresh._hswq_nvfp4_stack_ver = getattr(
            _cur_mp, "_hswq_nvfp4_stack_ver", 0
        )  # type: ignore[attr-defined]
        ops.mixed_precision_ops = mixed_precision_ops_parity_refresh
        _PARITY_APPLIED = True
        _console(
            "[HSWQ NVFP4] comfy_parity refresh: stock GEMM + act rotate "
            "(Z Image / bench path; ConvRot Linear LoRA bake kept)"
        )
        return True

    orig_load = _closure_named(patched_load, "_orig_load")
    if orig_load is None:
        orig_load = patched_load

    def _load_quantized_module_comfy_only(
        module,
        super_load,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
        load_extra_params=False,
    ):
        conf = decode_comfy_quant_conf(state_dict.get(f"{prefix}comfy_quant"))
        orig_load(
            module,
            super_load,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
            load_extra_params=load_extra_params,
        )
        if is_nvfp4_conf(conf):
            _arm_convrot_after_stock_load(module, conf)

    _load_quantized_module_comfy_only._hswq_nvfp4_comfy_only = True  # type: ignore[attr-defined]
    ops._load_quantized_module = _load_quantized_module_comfy_only

    _cur_mp = ops.mixed_precision_ops

    def mixed_precision_ops_comfy_only(*args, **kwargs):
        mp = _cur_mp(*args, **kwargs)
        Lin = mp.Linear
        if getattr(Lin.forward, "_hswq_nvfp4_full_forward", False):
            stock = _unwrap_stock_forward(Lin.forward)
            if stock is None:
                raise RuntimeError(
                    "Could not unwrap HSWQ TC Linear.forward for ConvRot parity"
                )
            Lin.forward = _make_convrot_parity_forward(stock)
        elif not getattr(Lin.forward, "_hswq_nvfp4_convrot_parity", False):
            Lin.forward = _make_convrot_parity_forward(Lin.forward)
        # Keep ConvRot Linear LoRA bake (convert unrotate / set re-rotate).
        attach_nvfp4_linear_lora_bake(Lin)
        return mp

    mixed_precision_ops_comfy_only._hswq_nvfp4_comfy_only = True  # type: ignore[attr-defined]
    mixed_precision_ops_comfy_only._hswq_nvfp4_stack_ver = getattr(
        _cur_mp, "_hswq_nvfp4_stack_ver", 0
    )  # type: ignore[attr-defined]
    if getattr(_cur_mp, "_hswq_nvfp4_orig_mp", None) is not None:
        mixed_precision_ops_comfy_only._hswq_nvfp4_orig_mp = (  # type: ignore[attr-defined]
            _cur_mp._hswq_nvfp4_orig_mp
        )
    ops.mixed_precision_ops = mixed_precision_ops_comfy_only

    _PARITY_APPLIED = True
    _console(
        "[HSWQ NVFP4] comfy_parity ON: stock MixedPrecision GEMM + online act rotate "
        "(Z Image / zi_convrot_nvfp4_bench path; not HSWQ TC Linear.forward)"
    )
    return True
