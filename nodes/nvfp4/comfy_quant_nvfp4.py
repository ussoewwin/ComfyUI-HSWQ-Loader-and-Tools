"""
ComfyUI runtime monkey-patches for HSWQ comfy_quant NVFP4 (FULL ConvRot).

Runtime only — never permanently edit ComfyUI-master.

Owns (via sibling modules under nodes/nvfp4/):
  - packed-K UNet detection (logical in_features)
  - full NVFP4 Linear load (scales, QT, ConvRot flags, storage validation)
  - full Tensor Core forward (act ConvRot → NVFP4 quant → scaled_mm_nvfp4)
  - ConvRot NVFP4 Linear LoRA bake (convert_weight unrotate → set_weight re-rotate)

This is not an INT8/FP8 “small tweak”: load + forward are HSWQ-owned stacks.
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
    reset_nvfp4_lora_log_counters,
)
from .nvfp4_load import load_nvfp4_linear_module, peek_nvfp4_conf

logger = logging.getLogger(__name__)
_PATCHES_APPLIED = False
# Bump when NVFP4 stack contract changes (forces re-wire of mixed_precision_ops).
_NVFP4_STACK_VER = 2

# Re-export for benches / callers
__all__ = [
    "NVFP4_WEIGHT_DTYPE",
    "apply_comfy_quant_nvfp4_patches",
    "checkpoint_looks_like_comfy_quant_nvfp4",
    "decode_comfy_quant_conf",
    "install_nvfp4_option_dispatch",
    "is_nvfp4_conf",
    "load_checkpoint_sdxl_nvfp4_weight_dtype",
    "load_unet_nvfp4_weight_dtype",
    "logical_linear_in_features",
    "nvfp4_forward_stats",
    "reset_nvfp4_forward_stats",
    "reset_nvfp4_lora_log_counters",
]


def _console(msg: str) -> None:
    print(msg, flush=True)
    logger.info(msg)


def _patcher_needs_nvfp4_force_full_load(patcher) -> bool:
    """True for HSWQ ConvRot NVFP4 UNets that must not stream via lowvram / Pin thrash.

    Owner log: ``Pin error`` × hundreds then
    ``loaded partially; 0.00 MB usable, 0.00 MB loaded, ~5177 MB offloaded``
    immediately before ``act_rotate hit#1``. Sampling with zero loaded weights
    destroys ConvRot NVFP4 quality. Not a side issue — fix residency.
    """
    if patcher is None:
        return False
    if getattr(patcher, "_hswq_nvfp4_force_full_load", False):
        return True
    base = getattr(patcher, "model", None)
    if base is None:
        return False
    if getattr(base, "_hswq_nvfp4_disable_dynamic", False):
        return True
    if getattr(base, "_hswq_nvfp4_force_full_load", False):
        return True
    try:
        for _, mod in base.named_modules():
            if getattr(mod, "_hswq_nvfp4_convrot", False):
                return True
    except Exception:
        return False
    return False


def _force_detach_dynamic_except(keep_patchers=None, device=None) -> int:
    """Detach Dynamic VRAM occupants (VAE staging, leftovers) before NVFP4 full load.

    Same spirit as INT8→Nunchaku handoff: leave ``unpatch_weights=False`` so
    QT / ConvRot stamps stay intact; only free GPU / VBAR / pin occupancy.
    """
    try:
        import comfy.model_management as mm
    except ImportError:
        return 0

    keep_ids = {id(p) for p in (keep_patchers or []) if p is not None}
    unloaded = 0
    i = 0
    while i < len(mm.current_loaded_models):
        lm = mm.current_loaded_models[i]
        patcher = lm.model
        if patcher is None:
            i += 1
            continue
        if id(patcher) in keep_ids:
            i += 1
            continue
        if device is not None and getattr(lm, "device", None) is not None:
            try:
                if str(lm.device) != str(device):
                    i += 1
                    continue
            except Exception:
                pass
        is_dyn = False
        try:
            is_dyn = bool(patcher.is_dynamic())
        except Exception:
            is_dyn = False
        if not is_dyn:
            i += 1
            continue
        # Never detach another ConvRot NVFP4 UNet that is also being kept.
        if _patcher_needs_nvfp4_force_full_load(patcher):
            i += 1
            continue
        try:
            lm.model_unload(unpatch_weights=False)
        except TypeError:
            try:
                patcher.detach(unpatch_all=False)
            except TypeError:
                try:
                    patcher.detach(False)
                except Exception as exc:
                    _console(
                        f"[HSWQ NVFP4] detach(False) failed: {exc!r}"
                    )
            except Exception as exc:
                _console(
                    f"[HSWQ NVFP4] detach(unpatch_all=False) failed: {exc!r}"
                )
            try:
                fin = getattr(lm, "model_finalizer", None)
                if fin is not None:
                    fin.detach()
            except Exception:
                pass
            try:
                lm.model_finalizer = None
                lm.real_model = None
            except Exception:
                pass
        except Exception as exc:
            _console(f"[HSWQ NVFP4] model_unload(False) failed: {exc!r}")
            i += 1
            continue
        mm.current_loaded_models.pop(i)
        unloaded += 1
    if unloaded > 0:
        try:
            mm.soft_empty_cache()
        except Exception:
            pass
    return unloaded


def _patch_load_models_gpu_nvfp4_force_full() -> bool:
    """Force ``force_full_load`` + free Dynamic occupancy for ConvRot NVFP4.

    Must wrap *after* MultiGPU / INT8 handoff when possible (re-install from
    UNet load path). Outer wrapper sets force_full_load before calling through.
    """
    try:
        import comfy.model_management as mm
    except ImportError:
        return False

    original = getattr(mm, "load_models_gpu", None)
    if original is None:
        return False
    # v2 = detach any Dynamic (VAE etc.), not only INT8; re-arm after MultiGPU.
    _VER = 2
    if getattr(original, "_hswq_nvfp4_force_full_ver", 0) >= _VER:
        return True
    # Keep MultiGPU / INT8-handoff wrappers in the chain. Only unwrap *our*
    # older wrap — never jump to ``_hswq_orig_load_models_gpu`` (that skips INT8).
    if getattr(original, "_hswq_nvfp4_force_full", False):
        true_orig = getattr(
            original, "_hswq_nvfp4_orig_load_models_gpu", original
        )
    else:
        true_orig = original

    def load_models_gpu(
        models,
        memory_required=0,
        force_patch_weights=False,
        minimum_memory_required=None,
        force_full_load=False,
    ):
        keep = []
        need_full = False
        device = None
        for m in models or []:
            keep.append(m)
            try:
                for mm_extra in getattr(m, "model_patches_models", lambda: [])() or []:
                    keep.append(mm_extra)
            except Exception:
                pass
            if _patcher_needs_nvfp4_force_full_load(m):
                need_full = True
                if device is None:
                    device = getattr(m, "load_device", None)
        if need_full:
            n = _force_detach_dynamic_except(keep_patchers=keep, device=device)
            n2 = _force_detach_dynamic_except(keep_patchers=keep, device=None)
            try:
                mm.soft_empty_cache()
            except Exception as exc:
                _console(f"[HSWQ NVFP4] soft_empty_cache failed: {exc!r}")
            _console(
                "[HSWQ NVFP4] force_full_load before ConvRot NVFP4 sample "
                f"(Dynamic offload keep-weights={n + n2}; "
                "avoids Pin thrash / 0.00 MB usable partial load)"
            )
            force_full_load = True
        return true_orig(
            models,
            memory_required=memory_required,
            force_patch_weights=force_patch_weights,
            minimum_memory_required=minimum_memory_required,
            force_full_load=force_full_load,
        )

    load_models_gpu._hswq_nvfp4_force_full = True  # type: ignore[attr-defined]
    load_models_gpu._hswq_nvfp4_force_full_ver = _VER  # type: ignore[attr-defined]
    load_models_gpu._hswq_nvfp4_orig_load_models_gpu = true_orig  # type: ignore[attr-defined]
    mm.load_models_gpu = load_models_gpu
    return True


def _callable_chain_has_attr(fn, attr: str) -> bool:
    """True if ``fn`` or a closed-over callable wrapper has ``attr``."""
    cur = fn
    seen = set()
    for _ in range(12):
        if cur is None or not callable(cur) or id(cur) in seen:
            return False
        seen.add(id(cur))
        if getattr(cur, attr, False):
            return True
        closure = getattr(cur, "__closure__", None) or ()
        freevars = getattr(getattr(cur, "__code__", None), "co_freevars", ()) or ()
        next_fn = None
        prefer = (
            "_orig",
            "original",
            "original_convert",
            "_orig_convert_old_quants",
        )
        for name, cell in zip(freevars, closure):
            val = cell.cell_contents
            if not callable(val):
                continue
            if getattr(val, attr, False):
                return True
            if name in prefer:
                next_fn = val
        if next_fn is None:
            for cell in closure:
                val = cell.cell_contents
                if callable(val):
                    next_fn = val
                    break
        cur = next_fn
    return False


def _patch_convert_old_quants_nvfp4_kitchen_prefix() -> bool:
    """Remap bare kitchen ``.comfy_quant`` keys under ``model_prefix``.

    Kitchen Z Image / ZIT packs store ``_quantization_metadata`` layer keys
    **without** ``model.diffusion_model.``. Stock ``convert_old_quants`` then
    injects markers at bare paths (``layers.0.attention.qkv.comfy_quant``) while
    weights live at ``model.diffusion_model.layers.0...weight``. MixedPrecision
    load peeks ``{prefix}comfy_quant`` under the full module prefix → **miss**.

    Must remap **all** formats with a matching prefixed weight — not only NVFP4.
    ``int8protect*`` packs leave INT8 markers bare if we filter ``is_nvfp4_conf``;
    those layers then load as raw tensors while ConvRot NVFP4 layers rotate → morphing.
    HSWQ bench uses empty ``model_prefix`` + key strip so bare markers match; product
    ``load_diffusion_model`` needs this full remap.
    """
    try:
        import comfy.utils as comfy_utils
    except Exception as e:
        logger.warning("[HSWQ NVFP4] convert_old remap skipped: %s", e)
        return False

    current = comfy_utils.convert_old_quants
    # v2: remap every bare .comfy_quant (nvfp4 + int8_tensorwise + …). Re-wrap if
    # only the old nvfp4-only patch is present so a running ComfyUI picks up the fix.
    if _callable_chain_has_attr(current, "_hswq_nvfp4_kitchen_prefix_v2"):
        return True

    _orig = current

    def convert_old_quants_nvfp4_prefix(state_dict, model_prefix="", metadata=None):
        if metadata is None:
            metadata = {}
        state_dict, metadata = _orig(state_dict, model_prefix, metadata=metadata)
        moved = 0
        by_fmt: dict[str, int] = {}
        if model_prefix:
            for k in list(state_dict.keys()):
                if not k.endswith(".comfy_quant") or k.startswith(model_prefix):
                    continue
                try:
                    conf = decode_comfy_quant_conf(state_dict[k])
                except Exception:
                    continue
                if not isinstance(conf, dict):
                    continue
                layer = k[: -len(".comfy_quant")]
                if f"{model_prefix}{layer}.weight" not in state_dict:
                    continue
                state_dict[f"{model_prefix}{k}"] = state_dict.pop(k)
                moved += 1
                fmt = str(conf.get("format") or "?")
                by_fmt[fmt] = by_fmt.get(fmt, 0) + 1
        if moved:
            fmt_bits = ", ".join(f"{f}={n}" for f, n in sorted(by_fmt.items()))
            _console(
                f"[HSWQ NVFP4] convert_old: remapped {moved} bare .comfy_quant "
                f"marker(s) under prefix={model_prefix!r} ({fmt_bits})"
            )
        return state_dict, metadata

    convert_old_quants_nvfp4_prefix._hswq_nvfp4_kitchen_prefix = True  # type: ignore[attr-defined]
    convert_old_quants_nvfp4_prefix._hswq_nvfp4_kitchen_prefix_v2 = True  # type: ignore[attr-defined]
    if getattr(_orig, "_hswq_int8_patched", False) or _callable_chain_has_attr(
        _orig, "_hswq_int8_patched"
    ):
        convert_old_quants_nvfp4_prefix._hswq_int8_patched = True  # type: ignore[attr-defined]
    comfy_utils.convert_old_quants = convert_old_quants_nvfp4_prefix
    _console(
        "[HSWQ NVFP4] convert_old_quants: kitchen bare→prefixed .comfy_quant remap ON "
        "(all formats: nvfp4 + int8 + …)"
    )
    return True


def apply_comfy_quant_nvfp4_patches() -> bool:
    """Install NVFP4 detection + full load + TC Linear forward + ConvRot LoRA bake."""
    global _PATCHES_APPLIED
    try:
        import comfy.model_detection as model_detection
        import comfy.ops as ops
    except Exception as e:
        logger.warning("[HSWQ NVFP4] comfy import failed: %s", e)
        return False

    # Always (re)ensure kitchen prefix remap — early-return paths must not skip it.
    _patch_convert_old_quants_nvfp4_kitchen_prefix()
    # Pin / 0.00 MB usable partial load kills ConvRot NVFP4 — force full GPU.
    _patch_load_models_gpu_nvfp4_force_full()

    mp_fn = getattr(ops, "mixed_precision_ops", None)
    stack_ver = int(getattr(mp_fn, "_hswq_nvfp4_stack_ver", 0) or 0) if mp_fn else 0
    if (
        _PATCHES_APPLIED
        and getattr(model_detection.detect_unet_config, "_hswq_nvfp4_packed_dims", False)
        and stack_ver >= _NVFP4_STACK_VER
    ):
        # Keep SDXL product refs fresh when still on TC (not Z Image parity).
        if not getattr(mp_fn, "_hswq_nvfp4_comfy_only", False):
            from .nvfp4_comfy_parity import remember_nvfp4_tc_product_stack

            remember_nvfp4_tc_product_stack(
                ops._load_quantized_module, ops.mixed_precision_ops
            )
        return True

    # Already patched detect/load but LoRA bake missing: re-wrap mixed_precision_ops only.
    if getattr(model_detection.detect_unet_config, "_hswq_nvfp4_packed_dims", False) and stack_ver < _NVFP4_STACK_VER:
        _orig_mp = getattr(mp_fn, "_hswq_nvfp4_orig_mp", mp_fn)

        def mixed_precision_ops_upgraded(*args, **kwargs):
            mp = _orig_mp(*args, **kwargs)
            Lin = mp.Linear
            if not getattr(Lin.forward, "_hswq_nvfp4_full_forward", False):
                Lin.forward = make_nvfp4_linear_forward(Lin.forward)
            attach_nvfp4_linear_lora_bake(Lin)
            return mp

        mixed_precision_ops_upgraded._hswq_nvfp4_full_forward = True  # type: ignore[attr-defined]
        mixed_precision_ops_upgraded._hswq_nvfp4_stack_ver = _NVFP4_STACK_VER  # type: ignore[attr-defined]
        mixed_precision_ops_upgraded._hswq_nvfp4_orig_mp = _orig_mp  # type: ignore[attr-defined]
        ops.mixed_precision_ops = mixed_precision_ops_upgraded
        _PATCHES_APPLIED = True
        from .nvfp4_comfy_parity import remember_nvfp4_tc_product_stack

        remember_nvfp4_tc_product_stack(
            ops._load_quantized_module, ops.mixed_precision_ops
        )
        _console(
            "[HSWQ NVFP4] upgraded stack ver=%s "
            "(ConvRot Linear LoRA bake: convert_weight unrotate + set_weight re-rotate)"
            % _NVFP4_STACK_VER
        )
        return True

    _orig_detect = model_detection.detect_unet_config
    _orig_calc = model_detection.calculate_transformer_depth
    _orig_load = ops._load_quantized_module
    _orig_mp = ops.mixed_precision_ops

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

    def mixed_precision_ops_patched(*args, **kwargs):
        mp = _orig_mp(*args, **kwargs)
        Lin = mp.Linear
        if not getattr(Lin.forward, "_hswq_nvfp4_full_forward", False):
            Lin.forward = make_nvfp4_linear_forward(Lin.forward)
        attach_nvfp4_linear_lora_bake(Lin)
        return mp

    model_detection.calculate_transformer_depth = calculate_transformer_depth_patched
    model_detection.detect_unet_config = detect_unet_config_patched
    model_detection.model_config_from_unet = model_config_from_unet_patched
    ops._load_quantized_module = _load_quantized_module_patched
    ops.mixed_precision_ops = mixed_precision_ops_patched

    detect_unet_config_patched._hswq_nvfp4_packed_dims = True  # type: ignore[attr-defined]
    calculate_transformer_depth_patched._hswq_nvfp4_packed_dims = True  # type: ignore[attr-defined]
    model_config_from_unet_patched._hswq_nvfp4_packed_dims = True  # type: ignore[attr-defined]
    _load_quantized_module_patched._hswq_nvfp4_full_load = True  # type: ignore[attr-defined]
    mixed_precision_ops_patched._hswq_nvfp4_full_forward = True  # type: ignore[attr-defined]
    mixed_precision_ops_patched._hswq_nvfp4_stack_ver = _NVFP4_STACK_VER  # type: ignore[attr-defined]
    mixed_precision_ops_patched._hswq_nvfp4_orig_mp = _orig_mp  # type: ignore[attr-defined]

    _PATCHES_APPLIED = True
    from .nvfp4_comfy_parity import remember_nvfp4_tc_product_stack

    remember_nvfp4_tc_product_stack(
        ops._load_quantized_module, ops.mixed_precision_ops
    )
    _console(
        "[HSWQ NVFP4] full stack applied "
        "(detect packed K + nvfp4_load + TC forward + ConvRot act + "
        "ConvRot Linear LoRA bake; ComfyUI-master untouched)"
    )
    return True


# UI / dispatch value — must match HSWQ Checkpoint Loader (SDXL) dropdown.
NVFP4_WEIGHT_DTYPE = "ConvRot NVFP4"


def load_checkpoint_sdxl_nvfp4_weight_dtype(ckpt_name, weight_dtype, device=None):
    """Load SDXL checkpoint with HSWQ NVFP4 Linear (+ INT8 Conv2d ConvRot) stack.

    SDXL stays on the product TC path. Never apply Z Image comfy_parity here.
    If a prior Z Image UNet load left parity on ``ops``, restore TC first.
    """
    import sys

    import folder_paths
    import comfy.sd

    # Package root = ComfyUI-nunchaku-unofficial-loader
    pkg = sys.modules[__name__.rsplit(".", 3)[0]]
    get_current_device = pkg.get_current_device
    set_current_device = pkg.set_current_device
    sdxl_logger = pkg.sdxl_logger

    from ...patches.comfy_quant_int8 import (
        _int8_quant_conv_scope,
        apply_comfy_quant_int8_patches,
        reset_int8_lora_log_counters,
        summarize_int8_lora_capability,
    )
    from .nvfp4_comfy_parity import restore_nvfp4_tc_product_stack

    original_device = get_current_device()
    if device is not None:
        set_current_device(device)
    try:
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        apply_comfy_quant_nvfp4_patches()
        # Branch: SDXL = TC product only. Undo any Z Image bench parity on ops.
        restore_nvfp4_tc_product_stack()
        # Mixed pack: Linear=nvfp4, Conv2d=int8_tensorwise (+ ConvRot) — same as bench.
        apply_comfy_quant_int8_patches()
        reset_int8_lora_log_counters()
        reset_nvfp4_lora_log_counters()
        sdxl_logger.info(
            "[SDXL NVFP4] Loading checkpoint via MixedPrecisionOps "
            "(nvfp4 Linear + int8 Conv / ConvRot + ConvRot Linear LoRA bake; TC): "
            "%s (weight_dtype=%s)",
            ckpt_name,
            weight_dtype,
        )
        with _int8_quant_conv_scope():
            out = comfy.sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=False,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                model_options={},
            )
        model, clip, _v = out[:3]
        summarize_int8_lora_capability(model)
        return (model, clip)
    finally:
        set_current_device(original_device)


def load_unet_nvfp4_weight_dtype(unet_name, weight_dtype):
    """Load Z Image / ZIT diffusion UNet with ConvRot NVFP4 (+ INT8 protect).

    Uses the same path as ``hswq/benchmark/zi_convrot_nvfp4_bench.py``:
    NVFP4 detect/load patches, then ``apply_nvfp4_comfy_parity()`` (stock Comfy
    GEMM + online act rotate). Product TC Linear.forward is **not** used here —
    it destroys Pixel SSIM on Z Image ConvRot packs. SDXL still uses TC.

    Mixed kitchen packs (Linear nvfp4 + int8protect) need INT8 patches too.
    ConvRot Linear LoRA bake stays on (same convert_weight / set_weight as SDXL).
    """
    import logging

    import folder_paths
    import comfy.sd

    from ...patches.comfy_quant_int8 import (
        _int8_quant_conv_scope,
        apply_comfy_quant_int8_patches,
        reset_int8_lora_log_counters,
        summarize_int8_lora_capability,
    )
    from .nvfp4_comfy_parity import (
        apply_nvfp4_comfy_parity,
        log_nvfp4_parity_load_summary,
        require_convrot_parity_forward,
        reset_nvfp4_parity_load_counters,
        summarize_nvfp4_parity_modules,
    )

    unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
    apply_comfy_quant_nvfp4_patches()
    # Re-assert kitchen remap after any prior INT8 wrap of convert_old_quants.
    _patch_convert_old_quants_nvfp4_kitchen_prefix()
    if not apply_nvfp4_comfy_parity():
        raise RuntimeError(
            "[HSWQ NVFP4] Z Image UNet requires comfy_parity "
            "(stock GEMM + act rotate; see hswq/benchmark/zi_convrot_nvfp4_bench.py)"
        )
    # INT8 after parity (mixed packs). Re-arm parity then require — INT8 must not
    # leave TC Linear.forward or drop act-rotate.
    apply_comfy_quant_int8_patches()
    _patch_convert_old_quants_nvfp4_kitchen_prefix()
    if not apply_nvfp4_comfy_parity():
        raise RuntimeError(
            "[HSWQ NVFP4] comfy_parity lost after INT8 patches"
        )
    require_convrot_parity_forward()
    reset_int8_lora_log_counters()
    reset_nvfp4_lora_log_counters()
    reset_nvfp4_parity_load_counters()
    logging.info(
        "[HSWQ NVFP4] Loading UNet via Comfy parity "
        "(stock GEMM + act rotate + int8 protect + ConvRot Linear LoRA bake; "
        "disable_dynamic=True — not AIMDO ModelPatcherDynamic): "
        "%s (weight_dtype=%s)",
        unet_name,
        weight_dtype,
    )
    print(
        f"[HSWQ NVFP4] Loading UNet (ConvRot NVFP4 / bench parity, full ModelPatcher): "
        f"{unet_name}",
        flush=True,
    )
    with _int8_quant_conv_scope():
        # AIMDO sets CoreModelPatcher = ModelPatcherDynamic. HSWQ bench loads via
        # stock ModelPatcher (full residency). Dynamic QT streaming breaks ConvRot
        # NVFP4 quality (grain / mush). Force classic patcher for this UNet only.
        model = comfy.sd.load_diffusion_model(
            unet_path, model_options={}, disable_dynamic=True
        )
    try:
        model.model._hswq_nvfp4_disable_dynamic = True
        model.model._hswq_nvfp4_force_full_load = True
    except Exception:
        pass
    try:
        # Stamp the ModelPatcher too — load_models_gpu sees the patcher first.
        model._hswq_nvfp4_force_full_load = True
    except Exception:
        pass
    # Re-arm outermost after INT8 / MultiGPU wraps so sample-time load is full.
    _patch_load_models_gpu_nvfp4_force_full()
    log_nvfp4_parity_load_summary(unet_name)
    summarize_nvfp4_parity_modules(model)
    summarize_int8_lora_capability(model)
    return (model,)

def install_nvfp4_option_dispatch(node_class_mappings) -> bool:
    """Wrap SDXL + Z Image UNet loaders so ConvRot NVFP4 uses nodes/nvfp4 stack.

    Must run *after* ``install_int8_option_dispatch``: NVFP4 checkpoints also
    contain ``int8_tensorwise`` layers (e.g. int8protect), so INT8-only
    auto-detect would otherwise steal the load path without NVFP4 Linear patches
    / ConvRot Linear LoRA bake.
    """
    if not isinstance(node_class_mappings, dict):
        return False

    _FP8_WEIGHT_DTYPES = frozenset({"fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"})
    wrapped = False

    unet_cls = node_class_mappings.get("HSWQFP8E4M3UNetLoader")
    if unet_cls is not None:
        _prev_load_unet = unet_cls.load_unet

        def load_unet(self, unet_name, weight_dtype):
            if weight_dtype in _FP8_WEIGHT_DTYPES:
                return _prev_load_unet(self, unet_name, weight_dtype)
            if weight_dtype == NVFP4_WEIGHT_DTYPE:
                return load_unet_nvfp4_weight_dtype(unet_name, weight_dtype)
            import folder_paths

            if weight_dtype == "default":
                unet_path = folder_paths.get_full_path_or_raise(
                    "diffusion_models", unet_name
                )
                if checkpoint_looks_like_comfy_quant_nvfp4(unet_path):
                    return load_unet_nvfp4_weight_dtype(unet_name, weight_dtype)
            return _prev_load_unet(self, unet_name, weight_dtype)

        unet_cls.load_unet = load_unet
        wrapped = True

    sdxl_cls = node_class_mappings.get("HSWQCheckpointLoaderSDXL")
    if sdxl_cls is not None:
        _prev_load_checkpoint = sdxl_cls.load_checkpoint

        def load_checkpoint(self, ckpt_name, weight_dtype, device=None):
            if weight_dtype in _FP8_WEIGHT_DTYPES:
                return _prev_load_checkpoint(self, ckpt_name, weight_dtype, device=device)
            if weight_dtype == NVFP4_WEIGHT_DTYPE:
                return load_checkpoint_sdxl_nvfp4_weight_dtype(
                    ckpt_name, weight_dtype, device=device
                )
            import folder_paths

            # default (and any non-FP8 path): NVFP4 markers beat INT8-only auto-detect.
            # Mixed packs also have int8_tensorwise Conv layers.
            if weight_dtype == "default":
                ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
                if checkpoint_looks_like_comfy_quant_nvfp4(ckpt_path):
                    return load_checkpoint_sdxl_nvfp4_weight_dtype(
                        ckpt_name, weight_dtype, device=device
                    )
            return _prev_load_checkpoint(self, ckpt_name, weight_dtype, device=device)

        sdxl_cls.load_checkpoint = load_checkpoint
        wrapped = True

    if wrapped:
        _console(
            "[HSWQ NVFP4] install_nvfp4_option_dispatch: "
            f"SDXL/UNet weight_dtype includes {NVFP4_WEIGHT_DTYPE!r}"
        )
    return wrapped
