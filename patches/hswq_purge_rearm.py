"""Reconcile HSWQ install gates after Distorch purge peels the quant stacks.

Background (2026-10-07):
``ComfyUI-DistorchMemoryManager`` ``DisTorchPurgeVRAMV2`` (HSWQ toggle) peels
the HSWQ overlays (ZI bake / parity / Linear wraps, Krea2 stack, INT8 decode
overlay, SDXL product TC wrap) so a *later SDXL* load never inherits Z Image
state. The peel side was written assuming the next load is SDXL; it has no
counterpart for the current primary use case: the NEXT prompt loads Z Image
ConvRot NVFP4 + INT8 ControlNet / Qwen INT8 / Krea2 again in the same process.

The HSWQ ``apply_*`` functions gate on module-level booleans
(``_PATCHES_APPLIED`` / ``_APPLIED`` / ``_GPU_BAKE_INSTALLED``) that the purge
does not know about. After a peel, gate=True makes the next load path skip
re-application (early ``return True``) while the actual wrappers are gone —
the run executes on a half-peeled stack and produces corrupt images
(observed as alternating good/bad prompts in one session).

``hswq_purge_rearm_state()`` never unwraps or wraps anything and never touches
tensors. It only inspects the live function/class stamps on
comfy.ops / comfy.model_detection / comfy.utils / comfy.model_patcher and
resets each family's gate boolean when its layer is ABSENT, so the next load
path re-applies via the legitimate install code.

Branch separation (Owner philosophy 分離・分岐、絶対混ぜるな): ZI /
SDXL-product-NVFP4 / Krea2 / INT8 / bake / parity each have an INDEPENDENT
predicate and reset its OWN global. A peeled ZI layer never resets the SDXL
gate, and vice versa.

Callers: Distorch purge calls this at the end of its peel (after all
unload_all_models / nuclear gc / peel APIs). It is safe to call any number of
times on a healthy stack (no-op).
"""
from __future__ import annotations

import sys


def _console(msg: str) -> None:
    print(f"[HSWQ purge-rearm] {msg}", flush=True)


def _mod_by_dotted_end(dotted: str):
    """Find an imported module by its dotted tail (independent of the
    custom_nodes package spelling): ``nodes.zimage_nvfp4.zi_comfy_quant_nvfp4``
    matches ``custom_nodes.ComfyUI-HSWQ-Loader-and-Tools.nodes.zimage_nvfp4.zi_comfy_quant_nvfp4``."""
    tail = "." + dotted.lstrip(".")
    for name, mod in list(sys.modules.items()):
        if mod is not None and name.endswith(tail):
            return mod
    return None


def _closure_named(fn, name: str):
    try:
        cells = getattr(fn, "__closure__", None) or ()
        code = getattr(fn, "__code__", None)
        if code is None:
            return None
        for n, cell in zip(code.co_freevars, cells):
            if n == name:
                return cell.contents
    except Exception:
        return None
    return None


_LOAD_PREV_NAMES = (
    "_orig_load",
    "original_load",
    "orig_load",
    "cur",
    "_prev",
    "_prev_load",
    "prev_load",
    "base",
)
_MP_PREV_NAMES = ("_orig_mp", "original_mp", "mp_fn", "_cur_mp", "prev_mp", "base")


def _walk_layers(fn, attr: str, closure_names: tuple, limit: int = 24):
    """Yield every layer of a wrapper chain (comfy.ops load / mixed_precision_ops).

    Follows the explicit ``attr`` back-pointer first, then known closure cells.
    Cycle-safe via id-seen set.
    """
    seen: set[int] = set()
    cur = fn
    depth = 0
    while cur is not None and callable(cur) and id(cur) not in seen and depth < limit:
        seen.add(id(cur))
        yield cur
        depth += 1
        try:
            nxt = getattr(cur, attr, None)
        except Exception:
            nxt = None
        if nxt is None or nxt is cur:
            for nm in closure_names:
                nxt = _closure_named(cur, nm)
                if nxt is not None and nxt is not cur:
                    break
        if nxt is cur:
            break
        cur = nxt


def _chain_has(fn, attr: str, closure_names: tuple, predicate) -> bool:
    try:
        for layer in _walk_layers(fn, attr, closure_names):
            try:
                if predicate(layer):
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False


def hswq_purge_rearm_state() -> dict:
    """Peel-aware gate reconciliation. Returns {family: action} for logging.

    families: zimage_nvfp4 / sdxl_nvfp4_product / krea2_nvfp4 / int8_comfy_quant
    / zimage_parity / zimage_bake_flag / krea2_bake_flag
    """
    report: dict = {}
    try:
        import comfy.ops as ops
        import comfy.model_detection as model_detection
        import comfy.utils as comfy_utils
    except Exception as e:  # comfy missing: nothing to reconcile
        _console(f"skipped: comfy import failed ({e})")
        return {"skipped": str(e)}

    load_fn = getattr(ops, "_load_quantized_module", None)
    mp_fn = getattr(ops, "mixed_precision_ops", None)
    detect = getattr(model_detection, "detect_unet_config", None)
    convert_old_quants = getattr(comfy_utils, "convert_old_quants", None)

    # --- independent presence predicates per family (never mixed) ----------
    # ZI ConvRot NVFP4: load wrap with full_load but NOT product_tc and NOT
    # krea2_full_load; detect packed_dims without krea2 txtlayers; mp carrying
    # _hswq_nvfp4_stack_ver (ZI stamps it; SDXL product stamps product_tc too,
    # so exclude the product chain by absence of product_tc on THAT layer).
    zi_load_ok = _chain_has(
        load_fn,
        "_hswq_nvfp4_orig_load",
        _LOAD_PREV_NAMES,
        lambda f: bool(getattr(f, "_hswq_nvfp4_full_load", False))
        and not bool(getattr(f, "_hswq_nvfp4_product_tc", False))
        and not bool(getattr(f, "_hswq_krea2_full_load", False)),
    )
    zi_mp_ok = _chain_has(
        mp_fn,
        "_hswq_nvfp4_orig_mp",
        _MP_PREV_NAMES,
        lambda f: int(getattr(f, "_hswq_nvfp4_stack_ver", 0) or 0) > 0
        and not bool(getattr(f, "_hswq_nvfp4_product_tc", False))
        and not bool(getattr(f, "_hswq_krea2_stack", False)),
    )
    zi_detect_ok = bool(
        getattr(detect, "_hswq_nvfp4_packed_dims", False)
        and not getattr(detect, "_hswq_krea2_txtlayers_fix", False)
    )

    # SDXL ConvRot NVFP4 product: identified ONLY by _hswq_nvfp4_product_tc.
    sdxl_ok = _chain_has(
        load_fn, "_hswq_nvfp4_orig_load", _LOAD_PREV_NAMES,
        lambda f: bool(getattr(f, "_hswq_nvfp4_product_tc", False)),
    ) or _chain_has(
        mp_fn, "_hswq_nvfp4_orig_mp", _MP_PREV_NAMES,
        lambda f: bool(getattr(f, "_hswq_nvfp4_product_tc", False)),
    )

    # Krea2 ConvRot NVFP4: krea2-specific stamps only.
    krea2_ok = (
        _chain_has(
            load_fn, "_hswq_nvfp4_orig_load", _LOAD_PREV_NAMES,
            lambda f: bool(getattr(f, "_hswq_krea2_full_load", False)),
        )
        or _chain_has(
            mp_fn, "_hswq_nvfp4_orig_mp", _MP_PREV_NAMES,
            lambda f: bool(getattr(f, "_hswq_krea2_stack", False)),
        )
        or _chain_has(
            convert_old_quants, "_hswq_krea2_prev_oldquants", ("_prev", "prev", "original"),
            lambda f: bool(getattr(f, "_hswq_krea2_oldquants", False)),
        )
        or bool(getattr(detect, "_hswq_krea2_txtlayers_fix", False))
    )

    # INT8 comfy_quant decode overlay (patches/comfy_quant_int8).
    int8_ok = _chain_has(
        load_fn, "_hswq_nvfp4_orig_load", _LOAD_PREV_NAMES,
        lambda f: bool(getattr(f, "_hswq_int8_decode_patched", False))
        or bool(getattr(f, "_hswq_int8_protect_in_load", False))
        or bool(getattr(f, "_hswq_int8_protect_arm_v2", False)),
    )

    # --- gate resets: one branch per family, own global only ----------------
    def _reset_gate(mod, attr: str, family: str, missing: bool) -> None:
        if mod is None:
            report[family] = "module-not-imported"
            return
        if not getattr(mod, attr, False):
            report[family] = "gate-already-open"
            return
        if not missing:
            report[family] = "layer-present(no-op)"
            return
        try:
            setattr(mod, attr, False)
            report[family] = "RESET"
            _console(f"{family}: peeled layer detected -> {attr}=False (next load re-applies)")
        except Exception as e:
            report[family] = f"reset-failed:{e}"

    _reset_gate(
        _mod_by_dotted_end("nodes.zimage_nvfp4.zi_comfy_quant_nvfp4"),
        "_PATCHES_APPLIED",
        "zimage_nvfp4",
        not (zi_load_ok and zi_mp_ok and zi_detect_ok),
    )
    _reset_gate(
        _mod_by_dotted_end("nodes.nvfp4.comfy_quant_nvfp4"),
        "_PATCHES_APPLIED",
        "sdxl_nvfp4_product",
        not sdxl_ok,
    )
    _reset_gate(
        _mod_by_dotted_end("nodes.krea2_convrot_nvfp4.comfy_quant_nvfp4"),
        "_PATCHES_APPLIED",
        "krea2_nvfp4",
        not krea2_ok,
    )
    _reset_gate(
        _mod_by_dotted_end("patches.comfy_quant_int8"),
        "_PATCHES_APPLIED",
        "int8_comfy_quant",
        not int8_ok,
    )

    # Parity gates: their apply() refreshes idempotently from live stamps
    # (no stale-gate early return), so force them open after a peel to make
    # the next apply_nvfp4_comfy_parity()/krea2 equivalent re-arm the parity
    # chain cleanly instead of trusting a stale bool. Independent globals.
    zi_par = _mod_by_dotted_end("nodes.zimage_nvfp4.nvfp4_comfy_parity")
    if zi_par is not None and getattr(zi_par, "_PARITY_APPLIED", False):
        try:
            zi_par._PARITY_APPLIED = False  # type: ignore[attr-defined]
            report["zimage_parity"] = "RESET(self-heal refresh)"
        except Exception as e:
            report["zimage_parity"] = f"reset-failed:{e}"
    else:
        report["zimage_parity"] = "gate-already-open"


    # Bake installed flags: bookkeeping only (install is stamp-gated per
    # Dynamic.load). Reset when the corresponding bake stamp is absent from
    # the Dynamic.load chain, so nothing believes the hook is still installed.
    try:
        import comfy.model_patcher as _mp
        Dynamic = getattr(_mp, "ModelPatcherDynamic", None)
        cur_load = getattr(Dynamic, "load", None) if Dynamic is not None else None
    except Exception:
        cur_load = None
    dyn_chain: list = []
    seen_ids: set[int] = set()
    c = cur_load
    while c is not None and id(c) not in seen_ids:
        seen_ids.add(id(c))
        dyn_chain.append(c)
        nxt = getattr(c, "_hswq_zi_rearm_guard_prev", None)
        if nxt is None:
            nxt = _closure_named(c, "cur") or _closure_named(c, "original")
        if nxt is c:
            break
        c = nxt

    def _dyn_has(stamp: str) -> bool:
        return any(bool(getattr(f, stamp, False)) for f in dyn_chain)

    for dotted, stamp, family in (
        ("nodes.zimage_nvfp4.nvfp4_lora_bake", "_hswq_zi_nvfp4_lora_bake", "zimage_bake_flag"),
        ("nodes.krea2_convrot_nvfp4.nvfp4_lora_bake", "_hswq_krea2_nvfp4_lora_bake", "krea2_bake_flag"),
    ):
        mod = _mod_by_dotted_end(dotted)
        if mod is None:
            report[family] = "module-not-imported"
            continue
        if getattr(mod, "_GPU_BAKE_INSTALLED", False) and not _dyn_has(stamp):
            try:
                mod._GPU_BAKE_INSTALLED = False  # type: ignore[attr-defined]
                report[family] = "RESET"
            except Exception as e:
                report[family] = f"reset-failed:{e}"
        else:
            report[family] = "consistent(no-op)"

    layer_state = {
        "zi_load": zi_load_ok, "zi_mp": zi_mp_ok, "zi_detect": zi_detect_ok,
        "sdxl_product": sdxl_ok, "krea2": krea2_ok, "int8": int8_ok,
        "zi_bake": _dyn_has("_hswq_zi_nvfp4_lora_bake"),
        "zi_guard": _dyn_has("_hswq_zi_rearm_guard"),
        "krea2_bake": _dyn_has("_hswq_krea2_nvfp4_lora_bake"),
    }
    report["layers"] = layer_state
    _console("reconciled " + ", ".join(f"{k}={v}" for k, v in sorted(report.items()) if k != "layers"))
    return report
