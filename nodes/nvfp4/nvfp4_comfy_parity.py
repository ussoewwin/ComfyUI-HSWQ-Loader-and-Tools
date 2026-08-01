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

# Runtime / load diagnostics (console — owner-ordered visibility).
_LOAD_NVFP4_SEEN = 0
_LOAD_CONVROT_ARMED = 0
_LOAD_NVFP4_NO_CONVROT = 0
_LOAD_INT8_CONVROT_ARMED = 0
_ACT_ROTATE_HITS = 0
_ACT_ROTATE_INT8_HITS = 0
_ACT_ROTATE_LOG_EVERY = 32
_ACT_ROTATE_FIRST_N = 4


def _console(msg: str) -> None:
    print(msg, flush=True)
    logger.info(msg)


def reset_nvfp4_parity_load_counters() -> None:
    global _LOAD_NVFP4_SEEN, _LOAD_CONVROT_ARMED, _LOAD_NVFP4_NO_CONVROT
    global _LOAD_INT8_CONVROT_ARMED, _ACT_ROTATE_HITS, _ACT_ROTATE_INT8_HITS
    _LOAD_NVFP4_SEEN = 0
    _LOAD_CONVROT_ARMED = 0
    _LOAD_NVFP4_NO_CONVROT = 0
    _LOAD_INT8_CONVROT_ARMED = 0
    _ACT_ROTATE_HITS = 0
    _ACT_ROTATE_INT8_HITS = 0


def log_nvfp4_parity_load_summary(label: str = "") -> None:
    """Print how many nvfp4 / int8protect ConvRot layers were armed during load."""
    tag = f" ({label})" if label else ""
    _console(
        f"[HSWQ NVFP4][diag] load summary{tag}: "
        f"nvfp4_seen={_LOAD_NVFP4_SEEN} "
        f"convrot_armed={_LOAD_CONVROT_ARMED} "
        f"nvfp4_no_convrot={_LOAD_NVFP4_NO_CONVROT} "
        f"int8_convrot_armed={_LOAD_INT8_CONVROT_ARMED}"
    )
    if _LOAD_NVFP4_SEEN == 0:
        _console(
            "[HSWQ NVFP4][diag] WARNING: zero nvfp4 layers seen during load — "
            "comfy_quant markers may be missing / wrong prefix "
            "(kitchen bare→prefixed remap should have run)"
        )
    elif _LOAD_CONVROT_ARMED == 0:
        _console(
            "[HSWQ NVFP4][diag] WARNING: nvfp4 layers loaded but "
            "convrot_armed=0 — act rotate will never run"
        )
    if _LOAD_INT8_CONVROT_ARMED == 0:
        _console(
            "[HSWQ NVFP4][diag] WARNING: int8protect ConvRot Linear armed=0 — "
            "mixed packs need online act rotate on protect Linears "
            "(offline W@H^T without x@H → bit-crush)"
        )


def summarize_nvfp4_parity_modules(model, max_names: int = 8) -> None:
    """Post-load walk: Linear counts + forward type + sample ConvRot names."""
    import torch.nn as nn

    try:
        import comfy.ops as ops
    except Exception as e:
        _console(f"[HSWQ NVFP4][diag] post-load skipped (ops): {e}")
        return

    # ModelPatcher -> BaseModel -> diffusion_model (same as INT8 summary).
    diffusion = model
    if hasattr(model, "model") and hasattr(model.model, "diffusion_model"):
        diffusion = model.model.diffusion_model
    elif hasattr(model, "diffusion_model"):
        diffusion = model.diffusion_model

    n_linear = 0
    n_convrot = 0
    n_int8_convrot = 0
    n_tc_arm = 0
    names: list[str] = []
    names_i8: list[str] = []
    for name, mod in diffusion.named_modules():
        if not isinstance(mod, nn.Linear) and "Linear" not in type(mod).__name__:
            continue
        n_linear += 1
        if getattr(mod, "_hswq_nvfp4_convrot", False):
            n_convrot += 1
            if len(names) < max_names:
                gs = getattr(mod, "_hswq_nvfp4_convrot_groupsize", "?")
                names.append(f"{name}(gs={gs})")
        if getattr(mod, "_hswq_int8_convrot", False):
            n_int8_convrot += 1
            if len(names_i8) < max_names:
                gs = getattr(mod, "_hswq_int8_convrot_groupsize", "?")
                names_i8.append(f"{name}(gs={gs})")
        if getattr(mod, "_hswq_nvfp4", False):
            n_tc_arm += 1

    fwd = ops.mixed_precision_ops().Linear.forward
    fwd_parity = bool(getattr(fwd, "_hswq_nvfp4_convrot_parity", False))
    fwd_tc = bool(getattr(fwd, "_hswq_nvfp4_full_forward", False))
    load_fn = ops._load_quantized_module
    load_parity = bool(getattr(load_fn, "_hswq_nvfp4_comfy_only", False))
    # INT8 may wrap load outside; peel once for display.
    if not load_parity and getattr(load_fn, "_hswq_int8_decode_patched", False):
        inner = _closure_named(load_fn, "original_load")
        if inner is not None:
            load_parity = bool(getattr(inner, "_hswq_nvfp4_comfy_only", False))
            load_fn = inner
    load_tc = bool(
        getattr(load_fn, "_hswq_nvfp4_full_load", False)
        and not getattr(load_fn, "_hswq_nvfp4_comfy_only", False)
    )

    _console(
        "[HSWQ NVFP4][diag] ===== post-load ====="
    )
    _console(
        f"[HSWQ NVFP4][diag] Linear={n_linear} "
        f"_hswq_nvfp4_convrot={n_convrot} "
        f"_hswq_int8_convrot={n_int8_convrot} "
        f"_hswq_nvfp4(TC arm)={n_tc_arm}"
    )
    _console(
        f"[HSWQ NVFP4][diag] Linear.forward: "
        f"parity={fwd_parity} tc_full={fwd_tc} "
        f"load: parity={load_parity} tc_full={load_tc} "
        f"_PARITY_APPLIED={_PARITY_APPLIED}"
    )
    if names:
        _console(
            "[HSWQ NVFP4][diag] sample NVFP4 ConvRot: "
            + ", ".join(names)
        )
    if names_i8:
        _console(
            "[HSWQ NVFP4][diag] sample INT8 protect ConvRot: "
            + ", ".join(names_i8)
        )
    _console(
        f"[HSWQ NVFP4][diag] act_rotate_hits_so_far="
        f"nvfp4={_ACT_ROTATE_HITS} int8protect={_ACT_ROTATE_INT8_HITS}"
    )
    _console("[HSWQ NVFP4][diag] =====================")


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


def _is_tc_full_load(fn) -> bool:
    """True for product TC load (load_nvfp4_linear_module), not parity stock load."""
    return bool(
        getattr(fn, "_hswq_nvfp4_full_load", False)
        and not getattr(fn, "_hswq_nvfp4_comfy_only", False)
    )


def _parity_load_in_chain(fn) -> bool:
    """True if comfy_parity load wrapper is already somewhere under ``fn``."""
    cur = fn
    seen = set()
    for _ in range(8):
        if cur is None or id(cur) in seen:
            return False
        seen.add(id(cur))
        if getattr(cur, "_hswq_nvfp4_comfy_only", False):
            return True
        if getattr(cur, "_hswq_int8_decode_patched", False):
            cur = _closure_named(cur, "original_load")
            continue
        if _is_tc_full_load(cur):
            cur = _closure_named(cur, "_orig_load")
            continue
        return False
    return False


def _resolve_load_under_tc(patched_load):
    """Callable under TC for parity to close over (stock Comfy or INT8 normalize).

    Peel **only** TC ``load_nvfp4_linear_module``. Keep INT8 decode wrap so
    int8protect layers still normalize ``comfy_quant`` tensors.
    Never return TC itself (ones(1) / ``_hswq_nvfp4`` arm).
    """
    if _is_tc_full_load(patched_load):
        inner = _closure_named(patched_load, "_orig_load")
        if inner is None:
            raise RuntimeError(
                "[HSWQ NVFP4] comfy_parity: TC load has no _orig_load "
                "(cannot recover Comfy / INT8 load under TC)"
            )
        if _is_tc_full_load(inner):
            raise RuntimeError(
                "[HSWQ NVFP4] comfy_parity: nested TC load; refusing"
            )
        return inner
    return patched_load


def _chain_has_int8_protect_in_load(fn) -> bool:
    """True if load chain already arms INT8 protect ConvRot after stock load."""
    cur = fn
    seen = set()
    for _ in range(8):
        if cur is None or id(cur) in seen:
            return False
        seen.add(id(cur))
        if getattr(cur, "_hswq_int8_protect_in_load", False):
            return True
        if getattr(cur, "_hswq_int8_protect_arm_v2", False):
            return True
        if getattr(cur, "_hswq_int8_decode_patched", False):
            cur = _closure_named(cur, "original_load")
            continue
        if _is_tc_full_load(cur):
            cur = _closure_named(cur, "_orig_load")
            continue
        if getattr(cur, "_hswq_nvfp4_comfy_only", False):
            return False
        return False
    return False


def _ensure_int8_protect_arm_overlay() -> None:
    """Hot-refresh: wrap current load so INT8 protect Linears get act-rotate arm.

    No-op when ``_load_quantized_module_comfy_only`` already has
    ``_hswq_int8_protect_in_load`` (fresh install path).
    """
    try:
        import comfy.ops as ops
    except Exception:
        return
    cur = ops._load_quantized_module
    if _chain_has_int8_protect_in_load(cur):
        return
    from .nvfp4_conf import decode_comfy_quant_conf

    def _load_int8_protect_arm_overlay(
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
        out = cur(
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
        _arm_int8_protect_convrot_after_stock_load(module, conf)
        return out

    _load_int8_protect_arm_overlay._hswq_int8_protect_arm_v2 = True  # type: ignore[attr-defined]
    ops._load_quantized_module = _load_int8_protect_arm_overlay
    _console(
        "[HSWQ NVFP4] comfy_parity: INT8 protect ConvRot arm overlay installed "
        "(hot refresh; online act rotate for protect Linears)"
    )


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


def _is_int8_tensorwise_convrot_conf(conf) -> bool:
    """True for INT8 protect Linear layers stamped with ConvRot offline rotate."""
    if not isinstance(conf, dict):
        return False
    fmt = conf.get("format")
    if fmt is not None and str(fmt).lower() != "int8_tensorwise":
        return False
    from .nvfp4_conf import convrot_flags_from_conf

    enabled, _gs = convrot_flags_from_conf(conf)
    return bool(enabled)


def _make_convrot_parity_forward(stock_forward):
    """Stock MixedPrecision forward + online act rotate for ConvRot weights.

    INT8 protect ConvRot Linear only (``_hswq_int8_convrot``). NVFP4 ConvRot
    (``_hswq_nvfp4_convrot``) is left to the kitchen NVFP4 path; rotating
    activations in stock forward double-rotates NVFP4 and destroys the image.

    Convert stores offline ``W @ H^T``. For INT8 protect, online must apply
    ``x @ H`` when Dynamic / dequant / ``F.linear`` would otherwise skip the
    kitchen ``int8_linear(convrot=True)`` rotate.
    """
    from .nvfp4_hadamard import build_hadamard, rotate_last_dim

    def forward_parity(self, input, *args, **kwargs):
        global _ACT_ROTATE_INT8_HITS
        # INT8 protect only: after Dynamic / dequant / F.linear, kitchen
        # int8_linear(convrot=True) does not rotate — apply x @ H here.
        # NVFP4 must NOT rotate here; kitchen NVFP4 path already handles ConvRot.
        i8 = bool(getattr(self, "_hswq_int8_convrot", False))
        if i8:
            _ACT_ROTATE_INT8_HITS += 1
            hit = _ACT_ROTATE_INT8_HITS
            tag = "int8protect"
            gs = int(getattr(self, "_hswq_int8_convrot_groupsize", 256) or 256)
            if hit <= _ACT_ROTATE_FIRST_N or (
                _ACT_ROTATE_LOG_EVERY > 0 and hit % _ACT_ROTATE_LOG_EVERY == 0
            ):
                cls = type(self).__name__
                shape = tuple(getattr(input, "shape", ()))
                _console(
                    f"[HSWQ NVFP4][diag] act_rotate hit#{hit} ({tag}) "
                    f"Linear={cls} gs={gs} x.shape={shape}"
                )
            h = getattr(self, "_hswq_nvfp4_parity_H", None)
            if h is None or h.device != input.device or h.dtype != input.dtype:
                h = build_hadamard(gs, device=input.device, dtype=input.dtype)
                self._hswq_nvfp4_parity_H = h
            input = rotate_last_dim(input, h, gs)
        return stock_forward(self, input, *args, **kwargs)

    forward_parity._hswq_nvfp4_convrot_parity = True  # type: ignore[attr-defined]
    return forward_parity


def _arm_convrot_after_stock_load(module, conf) -> None:
    global _LOAD_NVFP4_SEEN, _LOAD_CONVROT_ARMED, _LOAD_NVFP4_NO_CONVROT
    from .nvfp4_conf import convrot_flags_from_conf, is_nvfp4_conf

    if not is_nvfp4_conf(conf):
        return
    _LOAD_NVFP4_SEEN += 1
    enabled, gs = convrot_flags_from_conf(conf)
    module._hswq_nvfp4_convrot = False
    module._hswq_nvfp4_convrot_groupsize = int(gs)
    try:
        import comfy.quant_ops as quant_ops

        p = getattr(module, "weight", None)
        layout = getattr(p, "layout_params", None) if p is not None else None
        if isinstance(layout, quant_ops.Params) and getattr(layout, "convrot", False):
            layout.convrot = False
    except Exception:
        pass
    if enabled:
        _LOAD_CONVROT_ARMED += 1
        if _LOAD_CONVROT_ARMED <= 4 or _LOAD_CONVROT_ARMED % 40 == 0:
            fmt = conf.get("format")
            top = conf.get("convrot")
            params = conf.get("params") if isinstance(conf.get("params"), dict) else {}
            _console(
                f"[HSWQ NVFP4][diag] arm ConvRot #{_LOAD_CONVROT_ARMED} "
                f"gs={gs} format={fmt} convrot={top!r} "
                f"params.convrot={params.get('convrot')!r}"
            )
    else:
        _LOAD_NVFP4_NO_CONVROT += 1
        if _LOAD_NVFP4_NO_CONVROT <= 4:
            _console(
                f"[HSWQ NVFP4][diag] nvfp4 without convrot "
                f"(#{_LOAD_NVFP4_NO_CONVROT}) keys={list(conf.keys())[:12]}"
            )
    # Do not set _hswq_nvfp4 (TC full-forward arm).


def _arm_int8_protect_convrot_after_stock_load(module, conf) -> None:
    """Arm parity act-rotate for INT8 protect Linear (offline W@H^T).

    Clears ``Params.convrot`` so kitchen QT path does not double-rotate when
    QuantTensor still reaches ``int8_linear``. Same pattern as INT8 Conv2d
    (``_hswq_convrot`` + cleared Params.convrot).
    """
    global _LOAD_INT8_CONVROT_ARMED
    if not _is_int8_tensorwise_convrot_conf(conf):
        return
    from .nvfp4_conf import convrot_flags_from_conf

    _enabled, gs = convrot_flags_from_conf(conf)
    module._hswq_int8_convrot = True
    module._hswq_int8_convrot_groupsize = int(gs)
    try:
        import comfy.quant_ops as quant_ops

        p = getattr(module, "weight", None)
        layout = getattr(p, "layout_params", None) if p is not None else None
        if isinstance(layout, quant_ops.Params) and getattr(layout, "convrot", False):
            layout.convrot = False
    except Exception:
        pass
    _LOAD_INT8_CONVROT_ARMED += 1
    if _LOAD_INT8_CONVROT_ARMED <= 4 or _LOAD_INT8_CONVROT_ARMED % 20 == 0:
        _console(
            f"[HSWQ NVFP4][diag] arm INT8 protect ConvRot "
            f"#{_LOAD_INT8_CONVROT_ARMED} gs={gs}"
        )


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
    # Product Z Image: keep ConvRot Linear LoRA bake (same as SDXL). Do not peel.

    register_nvfp4_addmm_handler()

    if "nvfp4" not in QUANT_ALGOS:
        logger.warning("[HSWQ NVFP4] comfy_parity: nvfp4 not in QUANT_ALGOS")
        return False

    patched_load = ops._load_quantized_module
    # Prefer refs already saved by apply_comfy_quant_nvfp4_patches (TC only).
    remember_nvfp4_tc_product_stack(patched_load, ops.mixed_precision_ops)

    def _refresh_parity_mp() -> None:
        _cur_mp = ops.mixed_precision_ops

        def mixed_precision_ops_parity_refresh(*args, **kwargs):
            mp = _cur_mp(*args, **kwargs)
            Lin = mp.Linear
            attach_nvfp4_linear_lora_bake(Lin)
            if getattr(Lin.forward, "_hswq_nvfp4_full_forward", False):
                stock = _unwrap_stock_forward(Lin.forward)
                if stock is not None:
                    Lin.forward = _make_convrot_parity_forward(stock)
            elif not getattr(Lin.forward, "_hswq_nvfp4_convrot_parity", False):
                Lin.forward = _make_convrot_parity_forward(Lin.forward)
            return mp

        mixed_precision_ops_parity_refresh._hswq_nvfp4_comfy_only = True  # type: ignore[attr-defined]
        mixed_precision_ops_parity_refresh._hswq_nvfp4_stack_ver = getattr(
            _cur_mp, "_hswq_nvfp4_stack_ver", 0
        )  # type: ignore[attr-defined]
        if getattr(_cur_mp, "_hswq_nvfp4_orig_mp", None) is not None:
            mixed_precision_ops_parity_refresh._hswq_nvfp4_orig_mp = (  # type: ignore[attr-defined]
                _cur_mp._hswq_nvfp4_orig_mp
            )
        ops.mixed_precision_ops = mixed_precision_ops_parity_refresh

    # Already on parity load (possibly under INT8 decode wrap): keep load chain.
    if _parity_load_in_chain(patched_load):
        _ensure_int8_protect_arm_overlay()
        _refresh_parity_mp()
        _PARITY_APPLIED = True
        _console(
            "[HSWQ NVFP4] comfy_parity refresh: stock GEMM + act rotate "
            "(NVFP4 + INT8 protect) + ConvRot Linear LoRA bake (Z Image)"
        )
        return True

    orig_load = _resolve_load_under_tc(patched_load)

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
        out = orig_load(
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
        else:
            _arm_int8_protect_convrot_after_stock_load(module, conf)
        return out

    _load_quantized_module_comfy_only._hswq_nvfp4_comfy_only = True  # type: ignore[attr-defined]
    _load_quantized_module_comfy_only._hswq_int8_protect_in_load = True  # type: ignore[attr-defined]
    # Bench marks full_load on the parity wrapper too; keep comfy_only distinct
    # so remember_nvfp4_tc_product_stack never stores this as SDXL TC.
    ops._load_quantized_module = _load_quantized_module_comfy_only

    _cur_mp = ops.mixed_precision_ops

    def mixed_precision_ops_comfy_only(*args, **kwargs):
        mp = _cur_mp(*args, **kwargs)
        Lin = mp.Linear
        attach_nvfp4_linear_lora_bake(Lin)
        if getattr(Lin.forward, "_hswq_nvfp4_full_forward", False):
            stock = _unwrap_stock_forward(Lin.forward)
            if stock is None:
                raise RuntimeError(
                    "Could not unwrap HSWQ TC Linear.forward for ConvRot parity"
                )
            Lin.forward = _make_convrot_parity_forward(stock)
        elif not getattr(Lin.forward, "_hswq_nvfp4_convrot_parity", False):
            Lin.forward = _make_convrot_parity_forward(Lin.forward)
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

    # Prove unwrap once at install; keep LoRA bake attached for product use.
    mp0 = _cur_mp()
    attach_nvfp4_linear_lora_bake(mp0.Linear)
    if getattr(mp0.Linear.forward, "_hswq_nvfp4_full_forward", False):
        stock0 = _unwrap_stock_forward(mp0.Linear.forward)
        if stock0 is None:
            raise RuntimeError(
                "[HSWQ NVFP4] comfy_parity: failed to unwrap Linear.forward "
                "to Comfy stock at install"
            )
        mp0.Linear.forward = _make_convrot_parity_forward(stock0)

    _PARITY_APPLIED = True
    _console(
        "[HSWQ NVFP4] comfy_parity ON: stock MixedPrecision GEMM + online act rotate "
        "(NVFP4 ConvRot + INT8 protect ConvRot) "
        "+ ConvRot Linear LoRA bake (Z Image; not HSWQ TC Linear.forward)"
    )
    return True