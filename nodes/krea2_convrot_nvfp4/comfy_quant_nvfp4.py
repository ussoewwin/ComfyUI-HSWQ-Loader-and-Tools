"""
ComfyUI runtime monkey-patches for HSWQ comfy_quant NVFP4 (FULL ConvRot).

Runtime only - never permanently edit ComfyUI-master.

Owns (via sibling modules under this package):
  - packed-K UNet detection (logical in_features)
  - full NVFP4 Linear load (scales, QT, ConvRot flags, storage validation)
  - full ConvRot forward (act ConvRot -> pooled act NVFP4 quant -> cuBLAS FP4
    TC GEMM, weight packed resident; bake->float F.linear is fallback only)

Coexistence protocol (SDXL <-> Z Image field-proven; krea2 joins it):
  Every loader re-wires the SHARED globals to its own stack at load time:
  - ``ops.mixed_precision_ops`` / ``ops._load_quantized_module``: peel foreign
    NVFP4 wraps (SDXL product TC / ZI comfy_parity / stale ours) down to the
    INT8 wrap or stock, then install ours fresh. Our stamps follow the shared
    conventions (``_hswq_nvfp4_stack_ver`` / ``_hswq_nvfp4_orig_mp`` /
    ``_hswq_nvfp4_full_load`` + closure ``_orig_load``) so SDXL's
    ``peel_non_product_nvfp4_ops`` and ZI's parity install peel/unwrap US the
    same way they peel each other. Symmetric in both directions.
  - ``model_detection.*`` / ``convert_old_quants``: shared functional stamps
    (``_hswq_nvfp4_packed_dims``) - skip when present (all stacks' fixes are
    equivalent; avoids wrap-chain growth across alternated loads).
"""
from __future__ import annotations

import logging

from .nvfp4_conf import (
    checkpoint_looks_like_comfy_quant_nvfp4,
    decode_comfy_quant_conf,
    fix_unet_config_packed_dims,
    is_nvfp4_conf,
    logical_linear_in_features,
)
from .nvfp4_forward import (
    attach_nvfp4_linear_lora_bake,
    make_nvfp4_linear_forward,
    nvfp4_forward_stats,
    reset_nvfp4_forward_stats,
)
from .nvfp4_load import load_nvfp4_linear_module, peek_nvfp4_conf

logger = logging.getLogger(__name__)
_PATCHES_APPLIED = False

# Align with SDXL (=2) / ZI (=2) so ZI's `_effective_nvfp4_stack_ver` accepts
# the live chain and SDXL/ZI peel logic treats us as a foreign NVFP4 stack
# (stack_ver without _hswq_nvfp4_product_tc) they know how to peel.
_KREA2_NVFP4_STACK_VER = 2

# Re-export for benches / callers
__all__ = [
    "apply_comfy_quant_nvfp4_patches",
    "checkpoint_looks_like_comfy_quant_nvfp4",
    "decode_comfy_quant_conf",
    "is_nvfp4_conf",
    "logical_linear_in_features",
    "nvfp4_forward_stats",
    "reset_nvfp4_forward_stats",
]


def _console(msg: str) -> None:
    print(msg, flush=True)
    logger.info(msg)


def _closure_named(fn, name):
    """Free-variable lookup shared with ZI parity / SDXL peel walkers."""
    if fn is None or getattr(fn, "__closure__", None) is None:
        return None
    for n, cell in zip(fn.__code__.co_freevars, fn.__closure__):
        if n == name:
            return cell.cell_contents
    return None


def _unwrap_foreign_forward_to_stock(fwd):
    """Walk NVFP4-family Linear.forward wraps down to the true stock forward.

    SDXL TC / krea2 TC wraps close over ``stock_forward`` and stamp
    ``_hswq_nvfp4_full_forward``; ZI parity closes over ``stock_forward`` and
    stamps ``_hswq_nvfp4_convrot_parity``. Same closure name in all three.
    """
    cur = fwd
    seen: set[int] = set()
    for _ in range(8):
        if cur is None or not callable(cur) or id(cur) in seen:
            return None
        seen.add(id(cur))
        is_ours = getattr(cur, "_hswq_krea2_tc", False)
        is_nvfp4_wrap = bool(
            getattr(cur, "_hswq_nvfp4_full_forward", False)
            or getattr(cur, "_hswq_nvfp4_convrot_parity", False)
        )
        if not is_nvfp4_wrap:
            return cur  # stock reached
        nxt = _closure_named(cur, "stock_forward")
        if nxt is None or nxt is cur:
            return None if is_ours else cur
        cur = nxt
    return None


def _mp_base_under_foreign(mp_fn):
    """Peel foreign NVFP4 mp wraps (SDXL / ZI parity / stale ours) to INT8/stock.

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


def _load_base_under_foreign(load_fn):
    """Peel foreign NVFP4 load wraps to the INT8 decode wrap or stock.

    Walker names mirror ZI ``_next_load_under`` ("cur"/"orig_load"/
    "original_load"/"_orig_load") so every existing stack's wrap is walkable.
    """
    cur = load_fn
    seen: set[int] = set()
    for _ in range(12):
        if cur is None or not callable(cur) or id(cur) in seen:
            return None
        seen.add(id(cur))
        if getattr(cur, "_hswq_krea2_full_load", False):
            nxt = _closure_named(cur, "_orig_load")
            if nxt is None or nxt is cur:
                return None
            cur = nxt
            continue
        if getattr(cur, "_hswq_int8_decode_patched", False):
            return cur  # INT8 decode: keep (conf tensor normalization, shared)
        foreign = bool(
            getattr(cur, "_hswq_nvfp4_comfy_only", False)
            or getattr(cur, "_hswq_nvfp4_full_load", False)
        )
        if not foreign:
            return cur  # stock
        nxt = None
        for nm in ("_orig_load", "orig_load", "original_load", "cur"):
            nxt = _closure_named(cur, nm)
            if nxt is not None and nxt is not cur:
                break
        if nxt is None:
            return None
        cur = nxt
    return None


def _collapse_int8_over_stale_krea2(ops) -> bool:
    """Collapse ``INT8 wrap -> stale krea2 wrap -> base`` chains.

    INT8's ``mixed_precision_ops_force_conv`` / decode load wrap capture the
    then-outermost function as their base. When that base is a stale krea2
    wrap, alternated loads would grow the chain by two layers per cycle. Peel
    both, re-apply the INT8 stack fresh (import-and-call only - never edits
    ``patches/comfy_quant_int8``), so this apply can install over a clean
    ``[INT8, base]`` prefix. No-op when no stale krea2 hides under INT8.
    """
    try:
        fc = getattr(ops, "mixed_precision_ops", None)
        if fc is None or not getattr(fc, "_hswq_int8_conv_patched", False):
            return False
        # Import FIRST: if this fails we must not have unwired anything yet.
        from ...patches.comfy_quant_int8 import apply_comfy_quant_int8_patches

        node = getattr(fc, "_hswq_orig_mixed_precision_ops", None)
        base = None
        seen: set[int] = set()
        while node is not None and callable(node) and id(node) not in seen:
            seen.add(id(node))
            if getattr(node, "_hswq_krea2_stack", False):
                base = getattr(node, "_hswq_nvfp4_orig_mp", None)
                break
            node = getattr(node, "_hswq_orig_mixed_precision_ops", None) or getattr(
                node, "_hswq_nvfp4_orig_mp", None
            )
        if base is None or not callable(base):
            return False
        ops.mixed_precision_ops = base
        fl = getattr(ops, "_load_quantized_module", None)
        if fl is not None and getattr(fl, "_hswq_int8_decode_patched", False):
            node = _closure_named(fl, "original_load")
            lbase = None
            seen_l: set[int] = set()
            while node is not None and callable(node) and id(node) not in seen_l:
                seen_l.add(id(node))
                if getattr(node, "_hswq_krea2_full_load", False):
                    lbase = _closure_named(node, "_orig_load")
                    break
                nxt = _closure_named(node, "_orig_load")
                if nxt is None:
                    nxt = getattr(node, "_hswq_nvfp4_orig_load", None)
                node = nxt
            if lbase is not None and callable(lbase):
                ops._load_quantized_module = lbase
        apply_comfy_quant_int8_patches()
        _console(
            "[HSWQ NVFP4] krea2: collapsed stale krea2 wraps under INT8 "
            "(INT8 re-applied fresh)"
        )
        return True
    except Exception as e:
        logger.warning("[HSWQ NVFP4] krea2 collapse under INT8 skipped: %s", e)
        return False


def apply_comfy_quant_nvfp4_patches() -> bool:
    """Install NVFP4 detection + full load + full TC Linear forward.

    Idempotent AND coexisting: safe to call before every Krea2 load no matter
    which stack (SDXL TC / ZI parity / INT8) is currently live.
    """
    global _PATCHES_APPLIED
    # Gap fill: always (re)ensure addmm -> HSWQ hswq_scaled_mm_nvfp4 (idempotent).
    from .nvfp4_addmm_patch import register_nvfp4_addmm_handler

    register_nvfp4_addmm_handler()

    # Branch A (stock healthy): no rebind - plain NVFP4 layouts untouched.
    # Branch B (Asym bulk-import stubs only): submodule rebind. No shortcuts.
    from .kitchen_quant_ops_repair import ensure_kitchen_quant_ops

    ensure_kitchen_quant_ops()

    try:
        import comfy.model_detection as model_detection
        import comfy.ops as ops
        import comfy.utils as comfy_utils
    except Exception as e:
        logger.warning("[HSWQ NVFP4] comfy import failed: %s", e)
        return False

    # 0) Trim staleness: INT8 wraps that captured a stale krea2 base.
    _collapse_int8_over_stale_krea2(ops)

    # ------------------------------------------------------------------
    # 1) mixed_precision_ops: peel foreign NVFP4 stacks, install ours fresh.
    #    (THE coexistence fix: never early-return on the shared detect stamp -
    #    SDXL/ZI stamp it too, which used to leave THEIR forward live for a
    #    Krea2 model.)
    # ------------------------------------------------------------------
    if not getattr(ops.mixed_precision_ops, "_hswq_krea2_stack", False):
        _orig_mp = _mp_base_under_foreign(ops.mixed_precision_ops)
        if _orig_mp is None:
            logger.error(
                "[HSWQ NVFP4] krea2: cannot resolve mixed_precision_ops base "
                "under foreign wraps; refusing to load with a foreign stack"
            )
            return False

        def mixed_precision_ops_patched(*args, **kwargs):
            mp = _orig_mp(*args, **kwargs)
            Lin = mp.Linear
            fwd = getattr(Lin, "forward", None)
            if getattr(fwd, "_hswq_krea2_tc", False):
                return mp  # ours already on this freshly-built class
            if fwd is not None and (
                getattr(fwd, "_hswq_nvfp4_full_forward", False)
                or getattr(fwd, "_hswq_nvfp4_convrot_parity", False)
            ):
                # Foreign TC / parity forward ended up under us (chain case):
                # unwrap to true stock, then take over.
                stock = _unwrap_foreign_forward_to_stock(fwd)
                if stock is not None:
                    Lin.forward = make_nvfp4_linear_forward(stock)
                    attach_nvfp4_linear_lora_bake(Lin)
                    return mp
                return mp  # cannot unwrap safely; leave inner stack intact
            Lin.forward = make_nvfp4_linear_forward(fwd)
            attach_nvfp4_linear_lora_bake(Lin)
            return mp

        mixed_precision_ops_patched._hswq_nvfp4_full_forward = True  # type: ignore[attr-defined]
        mixed_precision_ops_patched._hswq_nvfp4_stack_ver = _KREA2_NVFP4_STACK_VER  # type: ignore[attr-defined]
        mixed_precision_ops_patched._hswq_nvfp4_orig_mp = _orig_mp  # type: ignore[attr-defined]
        mixed_precision_ops_patched._hswq_krea2_stack = True  # type: ignore[attr-defined]
        ops.mixed_precision_ops = mixed_precision_ops_patched
        _console(
            "[HSWQ NVFP4] krea2 mp stack installed "
            "(foreign NVFP4 wraps peeled; TC forward + VER=2 LoRA bake)"
        )

    # ------------------------------------------------------------------
    # 2) _load_quantized_module: peel foreign NVFP4 loads, install ours fresh.
    # ------------------------------------------------------------------
    if not getattr(ops._load_quantized_module, "_hswq_krea2_full_load", False):
        _orig_load = _load_base_under_foreign(ops._load_quantized_module)
        if _orig_load is None:
            logger.error(
                "[HSWQ NVFP4] krea2: cannot resolve _load_quantized_module "
                "base under foreign wraps; refusing to load with a foreign stack"
            )
            return False

        def _load_quantized_module_patched(
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
            conf = peek_nvfp4_conf(state_dict, prefix)
            if is_nvfp4_conf(conf):
                load_nvfp4_linear_module(
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
                return
            _orig_load(
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
            # Non-nvfp4 path: leave stock. (INT8 ConvRot etc. stay on stock/int8 patches.)

        _load_quantized_module_patched._hswq_nvfp4_full_load = True  # type: ignore[attr-defined]
        _load_quantized_module_patched._hswq_krea2_full_load = True  # type: ignore[attr-defined]
        ops._load_quantized_module = _load_quantized_module_patched
        _console("[HSWQ NVFP4] krea2 nvfp4 load installed (foreign loads peeled)")

    # ------------------------------------------------------------------
    # 3) Detection family: shared functional stamp. All NVFP4 stacks fix the
    #    same packed-K dims, so skip when ANY of us already stamped it (this
    #    also prevents the wrap chain growing on alternated loads).
    # ------------------------------------------------------------------
    if not getattr(model_detection.detect_unet_config, "_hswq_nvfp4_packed_dims", False):
        _orig_detect = model_detection.detect_unet_config
        _orig_calc = model_detection.calculate_transformer_depth

        def calculate_transformer_depth_patched(prefix, state_dict_keys, state_dict):
            out = _orig_calc(prefix, state_dict_keys, state_dict)
            if out is None:
                return None
            depth, context_dim, use_linear, time_stack, time_stack_cross = out
            k = f"{prefix}1.transformer_blocks.0.attn2.to_k.weight"
            if k in state_dict:
                try:
                    context_dim = logical_linear_in_features(state_dict, k)
                except Exception as e:
                    logger.warning("[HSWQ NVFP4] transformer context_dim fix skipped: %s", e)
            return depth, context_dim, use_linear, time_stack, time_stack_cross

        def detect_unet_config_patched(state_dict, key_prefix, metadata=None):
            unet_config = _orig_detect(state_dict, key_prefix, metadata=metadata)
            if unet_config is None:
                return None
            return fix_unet_config_packed_dims(unet_config, state_dict, key_prefix)

        def model_config_from_unet_patched(
            state_dict, unet_key_prefix, use_base_if_no_match=False, metadata=None
        ):
            import comfy.supported_models_base
            import comfy.utils

            unet_config = model_detection.detect_unet_config(
                state_dict, unet_key_prefix, metadata=metadata
            )
            if unet_config is None:
                return None
            model_config = model_detection.model_config_from_unet_config(
                unet_config, state_dict, unet_key_prefix
            )
            if model_config is None and use_base_if_no_match:
                model_config = comfy.supported_models_base.BASE(unet_config)

            quant_config = comfy.utils.detect_layer_quantization(
                state_dict, unet_key_prefix
            )
            if quant_config:
                if model_config is None:
                    logging.error(
                        "[HSWQ NVFP4] model_config is None with quant_config present "
                        "(packed NVFP4 dims still unmatched?). prefix=%r config=%s",
                        unet_key_prefix,
                        unet_config,
                    )
                    return None
                model_config.quant_config = quant_config
                logging.info("Detected mixed precision quantization")
            return model_config

        model_detection.calculate_transformer_depth = calculate_transformer_depth_patched
        model_detection.detect_unet_config = detect_unet_config_patched
        model_detection.model_config_from_unet = model_config_from_unet_patched

        detect_unet_config_patched._hswq_nvfp4_packed_dims = True  # type: ignore[attr-defined]
        calculate_transformer_depth_patched._hswq_nvfp4_packed_dims = True  # type: ignore[attr-defined]
        model_config_from_unet_patched._hswq_nvfp4_packed_dims = True  # type: ignore[attr-defined]

    # convert_old_quants: own marker (nobody else wraps it) - no chain growth.
    if not getattr(comfy_utils.convert_old_quants, "_hswq_krea2_oldquants", False):
        _orig_convert_old_quants = comfy_utils.convert_old_quants

        def convert_old_quants_patched(state_dict, model_prefix="", metadata={}):
            state_dict, metadata = _orig_convert_old_quants(
                state_dict, model_prefix, metadata=metadata
            )
            # Kitchen "plain NVFP4" stores _quantization_metadata layer keys
            # WITHOUT the diffusion-model prefix; stock injects `.comfy_quant`
            # at bare keys. Move each nvfp4 marker to the full-prefix key so
            # packed-K expansion + load succeed. HSWQ ConvRot files already
            # carry full-prefix markers -> skipped (untouched).
            if model_prefix:
                for k in list(state_dict.keys()):
                    if not k.endswith(".comfy_quant") or k.startswith(model_prefix):
                        continue
                    try:
                        conf = decode_comfy_quant_conf(state_dict[k])
                    except Exception:
                        continue
                    if not is_nvfp4_conf(conf):
                        continue
                    layer = k[: -len(".comfy_quant")]
                    if f"{model_prefix}{layer}.weight" not in state_dict:
                        continue
                    state_dict[f"{model_prefix}{k}"] = state_dict.pop(k)
            return state_dict, metadata

        convert_old_quants_patched._hswq_krea2_oldquants = True  # type: ignore[attr-defined]
        convert_old_quants_patched._hswq_krea2_prev_oldquants = _orig_convert_old_quants  # type: ignore[attr-defined]
        comfy_utils.convert_old_quants = convert_old_quants_patched

    _PATCHES_APPLIED = True
    _console(
        "[HSWQ NVFP4] krea2 stack ready "
        "(coexisting: SDXL TC / ZI parity / INT8 may re-wire on their loads)"
    )
    return True
