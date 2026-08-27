"""
Z Image arm for NVFP4 detect/load/LoRA bake — branch-only.

Owns the ZI delta that must not live in ``nodes/nvfp4`` (SDXL TC product):
  - walk stack_ver through INT8 / comfy_parity wraps
  - stamp stack_ver instead of false TC "upgrade" over ConvRot parity
  - never wrap TC Linear.forward over ``_hswq_nvfp4_convrot_parity``

Detect/load helpers stay under ``nodes/nvfp4``. Forward/bake come from
``zi_nvfp4_forward`` (hybrid). Call ``apply_nvfp4_comfy_parity`` after this.
"""
from __future__ import annotations

import logging

from ..nvfp4.nvfp4_conf import (
    fix_unet_config_packed_dims,
    is_nvfp4_conf,
    logical_linear_in_features,
)
from ..nvfp4.nvfp4_load import load_nvfp4_linear_module, peek_nvfp4_conf
from .zi_nvfp4_forward import (
    attach_nvfp4_linear_lora_bake,
    make_nvfp4_linear_forward,
)

logger = logging.getLogger(__name__)
_PATCHES_APPLIED = False
# Same contract bump as SDXL product; ZI reads through wrap chain.
_NVFP4_STACK_VER = 2

__all__ = [
    "apply_comfy_quant_nvfp4_patches",
]


def _console(msg: str) -> None:
    print(msg, flush=True)
    logger.info(msg)


def _effective_nvfp4_stack_ver(mp_fn) -> int:
    """Read stack_ver through INT8 / comfy_parity wraps (attrs may live on inner)."""
    cur = mp_fn
    seen: set[int] = set()
    for _ in range(8):
        if cur is None or id(cur) in seen:
            return 0
        seen.add(id(cur))
        v = int(getattr(cur, "_hswq_nvfp4_stack_ver", 0) or 0)
        if v > 0:
            return v
        if getattr(cur, "_hswq_int8_conv_patched", False):
            cur = getattr(cur, "_hswq_orig_mixed_precision_ops", None)
            continue
        cur = getattr(cur, "_hswq_nvfp4_orig_mp", None)
    return 0


def _mp_chain_has_comfy_only(mp_fn) -> bool:
    cur = mp_fn
    seen: set[int] = set()
    for _ in range(8):
        if cur is None or id(cur) in seen:
            return False
        seen.add(id(cur))
        if getattr(cur, "_hswq_nvfp4_comfy_only", False):
            return True
        if getattr(cur, "_hswq_int8_conv_patched", False):
            cur = getattr(cur, "_hswq_orig_mixed_precision_ops", None)
            continue
        cur = getattr(cur, "_hswq_nvfp4_orig_mp", None)
    return False


def apply_comfy_quant_nvfp4_patches() -> bool:
    """ZI: NVFP4 detect/load + hybrid LoRA bake; skip TC over ConvRot parity."""
    global _PATCHES_APPLIED
    try:
        import comfy.model_detection as model_detection
        import comfy.ops as ops
    except Exception as e:
        logger.warning("[HSWQ NVFP4] comfy import failed: %s", e)
        return False

    mp_fn = getattr(ops, "mixed_precision_ops", None)
    stack_ver = _effective_nvfp4_stack_ver(mp_fn)
    # The HSWQ purge peels the ops._load_quantized_module wrap (drops the
    # _hswq_nvfp4_full_load stamp) while leaving _PATCHES_APPLIED and the
    # detect_unet_config stamp intact. If the load wrap is gone, the arming
    # (arm_nvfp4_module -> _hswq_nvfp4_convrot) never fires on reload, so the
    # 2nd generation bakes NVFP4 layers as other_qt -> noise. Only early-return
    # when the load wrap is still armed; otherwise fall through to re-apply.
    _load_wrap_ok = bool(
        getattr(ops._load_quantized_module, "_hswq_nvfp4_full_load", False)
    )
    if (
        _PATCHES_APPLIED
        and _load_wrap_ok
        and getattr(model_detection.detect_unet_config, "_hswq_nvfp4_packed_dims", False)
        and stack_ver >= _NVFP4_STACK_VER
    ):
        return True

    # Already patched detect/load but LoRA bake missing: re-wrap mixed_precision_ops only.
    if _load_wrap_ok and getattr(model_detection.detect_unet_config, "_hswq_nvfp4_packed_dims", False) and stack_ver < _NVFP4_STACK_VER:
        # Z Image: INT8 wrap used to drop _hswq_nvfp4_stack_ver → false "upgrade"
        # that wrapped TC over ConvRot parity → double online rotate after refresh.
        if _mp_chain_has_comfy_only(mp_fn) or (
            _PATCHES_APPLIED
            and stack_ver == 0
            and getattr(mp_fn, "_hswq_int8_conv_patched", False)
        ):
            try:
                if mp_fn is not None:
                    mp_fn._hswq_nvfp4_stack_ver = _NVFP4_STACK_VER  # type: ignore[attr-defined]
            except Exception:
                pass
            _PATCHES_APPLIED = True
            _console(
                "[HSWQ NVFP4] stack ver stamped "
                "(skip TC upgrade; comfy_parity / INT8 chain intact)"
            )
            return True

        _orig_mp = getattr(mp_fn, "_hswq_nvfp4_orig_mp", None)
        if _orig_mp is None:
            _orig_mp = mp_fn

        def mixed_precision_ops_upgraded(*args, **kwargs):
            mp = _orig_mp(*args, **kwargs)
            Lin = mp.Linear
            # Never wrap TC over ConvRot parity (Z Image double-rotate / noise).
            if getattr(Lin.forward, "_hswq_nvfp4_convrot_parity", False):
                attach_nvfp4_linear_lora_bake(Lin)
                return mp
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
        weight_key = f"{prefix}weight"
        w_tensor = state_dict.get(weight_key)
        has_nvfp4_scale = f"{prefix}weight_scale_2" in state_dict

        # Route to NVFP4 loader only if format is nvfp4 AND NVFP4 scales/packed storage exist
        if is_nvfp4_conf(conf) and (
            has_nvfp4_scale or (w_tensor is not None and getattr(w_tensor, "dtype", None) == torch.uint8)
        ):
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

    def mixed_precision_ops_patched(*args, **kwargs):
        mp = _orig_mp(*args, **kwargs)
        Lin = mp.Linear
        if getattr(Lin.forward, "_hswq_nvfp4_convrot_parity", False):
            attach_nvfp4_linear_lora_bake(Lin)
            return mp
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
        "[HSWQ NVFP4] Z Image stack applied "
        "(detect packed K + nvfp4_load + hybrid LoRA bake; "
        "TC skipped over ConvRot parity; ComfyUI-master untouched)"
    )
    return True
