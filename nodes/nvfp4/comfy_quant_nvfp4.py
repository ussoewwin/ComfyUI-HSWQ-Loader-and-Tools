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
    "logical_linear_in_features",
    "nvfp4_forward_stats",
    "reset_nvfp4_forward_stats",
    "reset_nvfp4_lora_log_counters",
]


def _console(msg: str) -> None:
    print(msg, flush=True)
    logger.info(msg)


def apply_comfy_quant_nvfp4_patches() -> bool:
    """Install NVFP4 detection + full load + TC Linear forward + ConvRot LoRA bake."""
    global _PATCHES_APPLIED
    try:
        import comfy.model_detection as model_detection
        import comfy.ops as ops
    except Exception as e:
        logger.warning("[HSWQ NVFP4] comfy import failed: %s", e)
        return False

    mp_fn = getattr(ops, "mixed_precision_ops", None)
    stack_ver = int(getattr(mp_fn, "_hswq_nvfp4_stack_ver", 0) or 0) if mp_fn else 0
    if (
        _PATCHES_APPLIED
        and getattr(model_detection.detect_unet_config, "_hswq_nvfp4_packed_dims", False)
        and stack_ver >= _NVFP4_STACK_VER
    ):
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
    _console(
        "[HSWQ NVFP4] full stack applied "
        "(detect packed K + nvfp4_load + TC forward + ConvRot act + "
        "ConvRot Linear LoRA bake; ComfyUI-master untouched)"
    )
    return True


# UI / dispatch value — must match HSWQ Checkpoint Loader (SDXL) dropdown.
NVFP4_WEIGHT_DTYPE = "ConvRot NVFP4"


def load_checkpoint_sdxl_nvfp4_weight_dtype(ckpt_name, weight_dtype, device=None):
    """Load SDXL checkpoint with HSWQ NVFP4 Linear (+ INT8 Conv2d ConvRot) stack."""
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

    original_device = get_current_device()
    if device is not None:
        set_current_device(device)
    try:
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        apply_comfy_quant_nvfp4_patches()
        # Mixed pack: Linear=nvfp4, Conv2d=int8_tensorwise (+ ConvRot) — same as bench.
        apply_comfy_quant_int8_patches()
        reset_int8_lora_log_counters()
        reset_nvfp4_lora_log_counters()
        sdxl_logger.info(
            "[SDXL NVFP4] Loading checkpoint via MixedPrecisionOps "
            "(nvfp4 Linear + int8 Conv / ConvRot + ConvRot Linear LoRA bake): "
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


def install_nvfp4_option_dispatch(node_class_mappings) -> bool:
    """Wrap SDXL loader so ConvRot NVFP4 uses nodes/nvfp4 (bench) stack.

    Must run *after* ``install_int8_option_dispatch``: NVFP4 checkpoints also
    contain ``int8_tensorwise`` Conv layers, so INT8-only auto-detect would
    otherwise steal the load path without NVFP4 Linear patches.
    """
    if not isinstance(node_class_mappings, dict):
        return False

    _FP8_WEIGHT_DTYPES = frozenset({"fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"})

    sdxl_cls = node_class_mappings.get("HSWQCheckpointLoaderSDXL")
    if sdxl_cls is None:
        return False

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
    _console(
        "[HSWQ NVFP4] install_nvfp4_option_dispatch: "
        f"SDXL weight_dtype includes {NVFP4_WEIGHT_DTYPE!r}"
    )
    return True
