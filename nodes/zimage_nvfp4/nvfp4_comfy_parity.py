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


def _console(msg: str) -> None:
    print(msg, flush=True)
    logger.info(msg)


def reset_nvfp4_parity_load_counters() -> None:
    global _LOAD_NVFP4_SEEN, _LOAD_CONVROT_ARMED, _LOAD_NVFP4_NO_CONVROT
    global _LOAD_INT8_CONVROT_ARMED
    _LOAD_NVFP4_SEEN = 0
    _LOAD_CONVROT_ARMED = 0
    _LOAD_NVFP4_NO_CONVROT = 0
    _LOAD_INT8_CONVROT_ARMED = 0


def clear_nvfp4_parity_hadamard_caches(root=None) -> int:
    """Drop parity ``H`` attrs + global Hadamard dicts after Distorch purge.

    Method 3 may ``t.data = empty`` on module ``_hswq_nvfp4_parity_H`` while the
    same tensor remains in ``zi_nvfp4_hadamard._HADAMARD_CACHE``. The next gen then
    gets a dead/garbage ``H`` from the global cache (nbytes==0 rebuild still
    returns the poisoned entry) and quality decays as CUDA reuses the region
    (2nd→3rd→4th gen gradually worse). Distorch Method 2c calls this via
    ``sys.modules``.
    """
    import gc

    import torch

    from .zi_nvfp4_hadamard import clear_hadamard_global_caches

    cleared = 0
    cleared += int(clear_hadamard_global_caches() or 0)

    def _drop_attr(mod, name: str) -> None:
        nonlocal cleared
        if not hasattr(mod, name):
            return
        try:
            delattr(mod, name)
            cleared += 1
        except Exception:
            try:
                setattr(mod, name, None)
                cleared += 1
            except Exception:
                pass

    def _clear_one(mod) -> None:
        if not isinstance(mod, torch.nn.Module):
            return
        _drop_attr(mod, "_hswq_nvfp4_parity_H")
        _drop_attr(mod, "_hswq_nvfp4_H")
        # Z Image Dynamic LoRA bake bookkeeping — Distorch INT8 clear missed these.
        _drop_attr(mod, "_hswq_zi_nvfp4_baked_keys")
        _drop_attr(mod, "_hswq_zi_nvfp4_baked_uuid")

    if root is not None:
        if isinstance(root, torch.nn.Module):
            for m in root.modules():
                _clear_one(m)
            _clear_one(root)
        return cleared

    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.nn.Module):
                _clear_one(obj)
        except Exception:
            continue
    return cleared


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
    _console("[HSWQ NVFP4][diag] =====================")


def remember_nvfp4_tc_product_stack(load_fn, mp_fn) -> None:
    """Store SDXL product TC refs (call from apply_comfy_quant_nvfp4_patches only).

    Never overwrite with parity wrappers — SDXL must always be able to restore.

    Z Image ``zi_comfy_quant_nvfp4`` also stamps ``_hswq_nvfp4_stack_ver`` /
    ``_hswq_nvfp4_full_forward`` without ``_hswq_nvfp4_comfy_only``. Treating that
    as PRODUCT poisoned SDXL INT8 after Z Image (ZI VER=8 ``int8_protect`` bake
    on ConvRot INT8 → LoRA falls off on the 3rd prompt). Require
    ``_hswq_nvfp4_product_tc`` stamped only by ``nodes/nvfp4`` SDXL product.
    """
    global _PRODUCT_LOAD, _PRODUCT_MP
    if load_fn is not None and getattr(load_fn, "_hswq_nvfp4_product_tc", False):
        if not getattr(load_fn, "_hswq_nvfp4_comfy_only", False):
            _PRODUCT_LOAD = load_fn
    if mp_fn is not None and getattr(mp_fn, "_hswq_nvfp4_product_tc", False):
        if not getattr(mp_fn, "_hswq_nvfp4_comfy_only", False):
            _PRODUCT_MP = mp_fn


def _discard_poisoned_product_refs() -> None:
    """Drop PRODUCT refs that are Z Image stack / parity mistaken for SDXL TC."""
    global _PRODUCT_LOAD, _PRODUCT_MP
    if _PRODUCT_MP is not None and not getattr(
        _PRODUCT_MP, "_hswq_nvfp4_product_tc", False
    ):
        logger.warning(
            "[HSWQ NVFP4] discarding poisoned PRODUCT_MP "
            "(not SDXL product_tc — likely Z Image stack)"
        )
        _PRODUCT_MP = None
    if _PRODUCT_LOAD is not None and not getattr(
        _PRODUCT_LOAD, "_hswq_nvfp4_product_tc", False
    ):
        logger.warning(
            "[HSWQ NVFP4] discarding poisoned PRODUCT_LOAD "
            "(not SDXL product_tc — likely Z Image stack)"
        )
        _PRODUCT_LOAD = None


def peel_non_product_nvfp4_ops(ops) -> bool:
    """Peel Z Image / comfy_parity mp+load wrappers down to stock or SDXL product_tc.

    Used when PRODUCT was never saved (INT8-only → Z Image → SDXL) so restore
    cannot reinstate TC, but ZI mp must not keep attaching VER=8 Linear bake.
    """
    changed = False
    cur = getattr(ops, "mixed_precision_ops", None)
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if getattr(cur, "_hswq_nvfp4_product_tc", False):
            if ops.mixed_precision_ops is not cur:
                ops.mixed_precision_ops = cur
                changed = True
            break
        is_foreign = bool(
            getattr(cur, "_hswq_nvfp4_comfy_only", False)
            or (
                getattr(cur, "_hswq_nvfp4_stack_ver", 0)
                and not getattr(cur, "_hswq_nvfp4_product_tc", False)
            )
        )
        if not is_foreign:
            if ops.mixed_precision_ops is not cur:
                ops.mixed_precision_ops = cur
                changed = True
            break
        nxt = getattr(cur, "_hswq_nvfp4_orig_mp", None) or getattr(
            cur, "_hswq_orig_mixed_precision_ops", None
        )
        if nxt is None:
            break
        ops.mixed_precision_ops = nxt
        changed = True
        cur = nxt

    def _is_foreign_int8_protect_load(fn) -> bool:
        """ZI / INT8-protect / parity load wraps that must not survive onto SDXL."""
        return bool(
            getattr(fn, "_hswq_nvfp4_comfy_only", False)
            or getattr(fn, "_hswq_int8_protect_in_load", False)
            or getattr(fn, "_hswq_int8_protect_arm_v2", False)
            or getattr(fn, "_hswq_int8_decode_patched", False)
            or (
                getattr(fn, "_hswq_nvfp4_full_load", False)
                and not getattr(fn, "_hswq_nvfp4_product_tc", False)
            )
        )

    def _next_load_under(fn):
        # arm overlay closes over ``cur``; comfy_only / INT8 use orig*/_orig_load.
        for name in ("cur", "orig_load", "original_load", "_orig_load"):
            nxt = _closure_named(fn, name)
            if nxt is not None:
                return nxt
        return getattr(fn, "_hswq_nvfp4_orig_load", None)

    cur_l = getattr(ops, "_load_quantized_module", None)
    seen_l: set[int] = set()
    while cur_l is not None and id(cur_l) not in seen_l:
        seen_l.add(id(cur_l))
        if getattr(cur_l, "_hswq_nvfp4_product_tc", False):
            under = _next_load_under(cur_l)
            # PRODUCT restored on top of ZI protect arm still calls that arm for
            # int8_tensorwise+convrot (same conf shape as SDXL INT8 ConvRot).
            if under is not None and _is_foreign_int8_protect_load(under):
                ops._load_quantized_module = under
                changed = True
                cur_l = under
                continue
            if ops._load_quantized_module is not cur_l:
                ops._load_quantized_module = cur_l
                changed = True
            break
        if not _is_foreign_int8_protect_load(cur_l):
            if ops._load_quantized_module is not cur_l:
                ops._load_quantized_module = cur_l
                changed = True
            break
        nxt_l = _next_load_under(cur_l)
        if nxt_l is None:
            break
        ops._load_quantized_module = nxt_l
        changed = True
        cur_l = nxt_l
    return changed


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
    """True for SDXL product TC load only (not Z Image / parity).

    Z Image also stamps ``_hswq_nvfp4_full_load`` without ``_hswq_nvfp4_product_tc``.
    Treating that as product TC left ZI VER=8 Linear bake on SDXL INT8 (3rd prompt).
    """
    return bool(
        getattr(fn, "_hswq_nvfp4_full_load", False)
        and getattr(fn, "_hswq_nvfp4_product_tc", False)
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
    from ..nvfp4.nvfp4_conf import decode_comfy_quant_conf

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
    """Peel HSWQ TC *and* ConvRot parity wrappers until Comfy stock forward.

    After Distorch purge, Z Image reload may hit NVFP4 ``upgraded stack`` which
    wraps TC over an already-parity ``Linear.forward``. Refresh then used to
    peel only TC and re-wrap parity on top of parity → double online rotate →
    noise. Always flatten both wrapper kinds before a single parity wrap.
    """
    f = forward_fn
    for _ in range(8):
        if getattr(f, "_hswq_nvfp4_full_forward", False) or getattr(
            f, "_hswq_nvfp4_convrot_parity", False
        ):
            stock = _closure_named(f, "stock_forward")
            if stock is None:
                return None
            f = stock
            continue
        return f
    return None


def _ensure_single_parity_linear_forward(Lin) -> None:
    """Idempotent: one ConvRot parity wrap over true stock MixedPrecision forward."""
    fwd = Lin.forward
    stock = _unwrap_stock_forward(fwd)
    if stock is None:
        raise RuntimeError(
            "[HSWQ NVFP4] comfy_parity: could not unwrap Linear.forward "
            "to Comfy stock (TC/parity chain broken)"
        )
    # Already exactly parity(stock) with no nested wrappers under stock.
    if (
        getattr(fwd, "_hswq_nvfp4_convrot_parity", False)
        and not getattr(fwd, "_hswq_nvfp4_full_forward", False)
        and _closure_named(fwd, "stock_forward") is stock
        and not getattr(stock, "_hswq_nvfp4_full_forward", False)
        and not getattr(stock, "_hswq_nvfp4_convrot_parity", False)
    ):
        return
    Lin.forward = _make_convrot_parity_forward(stock)


def _is_int8_tensorwise_convrot_conf(conf) -> bool:
    """True for INT8 protect Linear layers stamped with ConvRot offline rotate."""
    from .zi_nvfp4_conf import int8_convrot_flags_from_conf

    enabled, _gs = int8_convrot_flags_from_conf(conf)
    return bool(enabled)


def _make_convrot_parity_forward(stock_forward):
    """Stock MixedPrecision forward + online act rotate for ConvRot Linears.

    NVFP4: ``_hswq_nvfp4_convrot`` (Params.convrot cleared at load).
    INT8 protect: ``_hswq_int8_convrot`` (Params.convrot cleared at load —
    same as Conv2d). Kitchen must **not** see Params.convrot=True or
    int8_linear double-rotates with this path.
    """
    from .zi_nvfp4_hadamard import build_hadamard, rotate_last_dim

    def forward_parity(self, input, *args, **kwargs):
        nv = bool(getattr(self, "_hswq_nvfp4_convrot", False))
        i8 = bool(getattr(self, "_hswq_int8_convrot", False))
        if nv or i8:
            if nv:
                gs = int(getattr(self, "_hswq_nvfp4_convrot_groupsize", 256) or 256)
            else:
                gs = int(getattr(self, "_hswq_int8_convrot_groupsize", 256) or 256)
            h = getattr(self, "_hswq_nvfp4_parity_H", None)
            need_rebuild = True
            if h is not None:
                try:
                    from .zi_nvfp4_hadamard import _tensor_storage_ok

                    # Global cache already rejects poisoned H via
                    # _tensor_storage_ok; module-local H must use the same
                    # check. nbytes==0 alone misses emptied shells that still
                    # report device/dtype (Distorch Method 3), so 2nd+ gen
                    # rotates with garbage and quality decays.
                    need_rebuild = (
                        h.device != input.device
                        or h.dtype != input.dtype
                        or (bool(input.is_cuda) and not bool(h.is_cuda))
                        or not _tensor_storage_ok(h)
                    )
                except Exception:
                    need_rebuild = True
            if need_rebuild:
                h = build_hadamard(gs, device=input.device, dtype=input.dtype)
                self._hswq_nvfp4_parity_H = h
            input = rotate_last_dim(input, h, gs)
        return stock_forward(self, input, *args, **kwargs)

    forward_parity._hswq_nvfp4_convrot_parity = True  # type: ignore[attr-defined]
    return forward_parity


def _arm_convrot_after_stock_load(module, conf) -> None:
    global _LOAD_NVFP4_SEEN, _LOAD_CONVROT_ARMED, _LOAD_NVFP4_NO_CONVROT
    from ..nvfp4.nvfp4_conf import convrot_flags_from_conf, is_nvfp4_conf

    if not is_nvfp4_conf(conf):
        return
    _LOAD_NVFP4_SEEN += 1
    enabled, gs = convrot_flags_from_conf(conf)
    module._hswq_nvfp4_convrot = bool(enabled)
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


def _clear_int8_qt_params_convrot(module) -> bool:
    """Force ``Params.convrot=False`` on INT8 QT weight (Conv2d-style).

    ``layout_params`` on Parameter is unreliable for QuantizedTensor; clear
    ``qt._params`` via dataclasses.replace. Leaving Params.convrot=True while
    ``_hswq_int8_convrot`` is set double-rotates acts (parity + kitchen).
    """
    import dataclasses

    try:
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return False
    w = getattr(module, "weight", None)
    qt = w if isinstance(w, QuantizedTensor) else getattr(w, "data", None)
    if qt is None or not isinstance(qt, QuantizedTensor):
        return False
    params = getattr(qt, "_params", None)
    if params is None or not bool(getattr(params, "convrot", False)):
        return False
    new_params = dataclasses.replace(params, convrot=False)
    try:
        object.__setattr__(qt, "_params", new_params)
        return True
    except Exception:
        pass
    try:
        qt._params = new_params
        return True
    except Exception:
        return False


def _arm_int8_protect_convrot_after_stock_load(module, conf) -> None:
    """Arm INT8 protect ConvRot Linear like Conv2d: flag + clear Params.convrot.

    Kitchen ``dequantize_int8_convrot_weight`` already unrotates when
    Params.convrot=True — LoRA bake must see Params=False so convert gets
    rotated-basis float and unrotates once. Online act rotate is parity
    (``_hswq_int8_convrot``). Requant must keep Params.convrot=False
    (see ``zi_nvfp4_forward`` set_weight).
    """
    global _LOAD_INT8_CONVROT_ARMED
    from .zi_nvfp4_conf import int8_convrot_flags_from_conf

    enabled, gs = int8_convrot_flags_from_conf(conf)
    if not enabled:
        return
    module._hswq_int8_convrot = True
    module._hswq_int8_convrot_groupsize = int(gs)
    cleared = _clear_int8_qt_params_convrot(module)
    _LOAD_INT8_CONVROT_ARMED += 1
    if _LOAD_INT8_CONVROT_ARMED <= 4 or _LOAD_INT8_CONVROT_ARMED % 20 == 0:
        _console(
            f"[HSWQ NVFP4][diag] arm INT8 protect ConvRot "
            f"#{_LOAD_INT8_CONVROT_ARMED} gs={gs} params_cleared={cleared}"
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
    """Put SDXL product TC load + forward back; peel Z Image if PRODUCT missing.

    Z Image stamps ``_hswq_nvfp4_stack_ver`` / ``_hswq_nvfp4_full_forward`` without
    ``_hswq_nvfp4_product_tc``. Treating that as ``already_tc`` left ZI VER=8
    ``int8_protect`` Linear bake on SDXL INT8 after Z Image (LoRA falls off on
    the 3rd prompt). Only ``product_tc`` counts as product TC.
    """
    global _PARITY_APPLIED
    try:
        import comfy.ops as ops
    except Exception as e:
        logger.warning("[HSWQ NVFP4] restore TC stack skipped: %s", e)
        return False

    _discard_poisoned_product_refs()

    mp = ops.mixed_precision_ops
    load_fn = ops._load_quantized_module
    already_product = bool(
        getattr(mp, "_hswq_nvfp4_product_tc", False)
        and getattr(load_fn, "_hswq_nvfp4_product_tc", False)
        and not getattr(mp, "_hswq_nvfp4_comfy_only", False)
        and not getattr(load_fn, "_hswq_nvfp4_comfy_only", False)
    )
    if already_product and not _PARITY_APPLIED:
        return True

    if _PRODUCT_LOAD is not None and _PRODUCT_MP is not None:
        ops._load_quantized_module = _PRODUCT_LOAD
        ops.mixed_precision_ops = _PRODUCT_MP
        _PARITY_APPLIED = False
        _console("[HSWQ NVFP4] restored product TC stack (SDXL path; parity off)")
        return True

    # INT8-only → Z Image → SDXL: PRODUCT was never saved. Peel ZI / parity so
    # SDXL INT8 does not keep attaching VER=8 ``[HSWQ ConvRot LoRA] int8_protect``.
    peeled = peel_non_product_nvfp4_ops(ops)
    _PARITY_APPLIED = False
    if peeled:
        if (
            getattr(ops.mixed_precision_ops, "_hswq_nvfp4_product_tc", False)
            and getattr(ops._load_quantized_module, "_hswq_nvfp4_product_tc", False)
        ):
            remember_nvfp4_tc_product_stack(
                ops._load_quantized_module, ops.mixed_precision_ops
            )
            _console(
                "[HSWQ NVFP4] restored product TC stack via peel "
                "(SDXL path; parity off)"
            )
        else:
            _console(
                "[HSWQ NVFP4] peeled non-product NVFP4 ops "
                "(stock/INT8 base; no product_tc PRODUCT — SDXL INT8 LoRA safe)"
            )
        return True

    logger.warning(
        "[HSWQ NVFP4] restore TC stack: no saved product refs "
        "(SDXL needs apply_comfy_quant_nvfp4_patches first)"
    )
    return False


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
    from ..nvfp4.nvfp4_conf import decode_comfy_quant_conf, is_nvfp4_conf
    from .zi_nvfp4_forward import attach_nvfp4_linear_lora_bake
    # Product Z Image: keep ConvRot Linear LoRA bake (same as SDXL). Do not peel.

    register_nvfp4_addmm_handler()

    if "nvfp4" not in QUANT_ALGOS:
        logger.warning("[HSWQ NVFP4] comfy_parity: nvfp4 not in QUANT_ALGOS")
        return False

    patched_load = ops._load_quantized_module
    # Prefer refs already saved by apply_comfy_quant_nvfp4_patches (TC only).
    remember_nvfp4_tc_product_stack(patched_load, ops.mixed_precision_ops)

    def _parity_mp_base(mp_fn):
        """Innermost non-parity ``mixed_precision_ops`` (TC / product stack).

        Refresh used to wrap the previous refresh wrapper every reload, stacking
        ``attach_nvfp4_linear_lora_bake`` / forward ensure. Peel to ``_orig_mp``.
        """
        cur = mp_fn
        seen: set[int] = set()
        while id(cur) not in seen:
            seen.add(id(cur))
            if not getattr(cur, "_hswq_nvfp4_comfy_only", False):
                return cur
            nxt = getattr(cur, "_hswq_nvfp4_orig_mp", None)
            if nxt is None or nxt is cur:
                return cur
            cur = nxt
        return cur

    def _refresh_parity_mp() -> None:
        _cur_mp = ops.mixed_precision_ops
        _base_mp = _parity_mp_base(_cur_mp)

        def mixed_precision_ops_parity_refresh(*args, **kwargs):
            mp = _base_mp(*args, **kwargs)
            Lin = mp.Linear
            attach_nvfp4_linear_lora_bake(Lin)
            _ensure_single_parity_linear_forward(Lin)
            return mp

        mixed_precision_ops_parity_refresh._hswq_nvfp4_comfy_only = True  # type: ignore[attr-defined]
        mixed_precision_ops_parity_refresh._hswq_nvfp4_stack_ver = getattr(
            _base_mp, "_hswq_nvfp4_stack_ver", getattr(_cur_mp, "_hswq_nvfp4_stack_ver", 0)
        )  # type: ignore[attr-defined]
        mixed_precision_ops_parity_refresh._hswq_nvfp4_orig_mp = _base_mp  # type: ignore[attr-defined]
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
    _base_install = _parity_mp_base(_cur_mp)

    def mixed_precision_ops_comfy_only(*args, **kwargs):
        mp = _base_install(*args, **kwargs)
        Lin = mp.Linear
        attach_nvfp4_linear_lora_bake(Lin)
        _ensure_single_parity_linear_forward(Lin)
        return mp

    mixed_precision_ops_comfy_only._hswq_nvfp4_comfy_only = True  # type: ignore[attr-defined]
    mixed_precision_ops_comfy_only._hswq_nvfp4_stack_ver = getattr(
        _base_install, "_hswq_nvfp4_stack_ver", 0
    )  # type: ignore[attr-defined]
    mixed_precision_ops_comfy_only._hswq_nvfp4_orig_mp = _base_install  # type: ignore[attr-defined]
    ops.mixed_precision_ops = mixed_precision_ops_comfy_only

    # Prove unwrap once at install; keep LoRA bake attached for product use.
    mp0 = _base_install()
    attach_nvfp4_linear_lora_bake(mp0.Linear)
    _ensure_single_parity_linear_forward(mp0.Linear)

    _PARITY_APPLIED = True
    _console(
        "[HSWQ NVFP4] comfy_parity ON: stock MixedPrecision GEMM + online act rotate "
        "(NVFP4 ConvRot + INT8 protect ConvRot) "
        "+ ConvRot Linear LoRA bake (Z Image; not HSWQ TC Linear.forward)"
    )
    return True