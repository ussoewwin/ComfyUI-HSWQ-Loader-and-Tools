"""
HSWQ-owned NVFP4 Linear load path.

Stock Comfy ``_load_quantized_module`` loads NVFP4 weights/scales into a
QuantizedTensor, but does **not**:
  - stamp FULL ConvRot flags (Params have no convrot for NVFP4)
  - validate storage shape against logical (out, in)
  - mark the module for the HSWQ Tensor Core forward path

This module owns that load logic entirely. It never edits ComfyUI-master;
callers monkey-patch ``ops._load_quantized_module`` to route nvfp4 here.
"""
from __future__ import annotations

import logging
from typing import Optional

from .nvfp4_conf import convrot_flags_from_conf, decode_comfy_quant_conf, is_nvfp4_conf

logger = logging.getLogger(__name__)


def peek_nvfp4_conf(state_dict, prefix: str) -> Optional[dict]:
    """Read comfy_quant without popping (for routing before stock load)."""
    return decode_comfy_quant_conf(state_dict.get(f"{prefix}comfy_quant"))


def arm_nvfp4_module(module, conf: Optional[dict]) -> None:
    """Attach HSWQ NVFP4 runtime flags after weight QT is in place."""
    if not is_nvfp4_conf(conf):
        return
    import torch

    module._hswq_nvfp4 = True
    enabled, gs = convrot_flags_from_conf(conf)
    module._hswq_nvfp4_convrot = bool(enabled)
    module._hswq_nvfp4_convrot_groupsize = int(gs)
    # Checkpoints often omit input_scale (0 keys in test.safetensors).
    # Placeholder ones(1) + flag → one amax per module then freeze (not every Linear).
    if getattr(module, "input_scale", None) is None:
        device = module.factory_kwargs.get("device", "cpu")
        module.register_parameter(
            "input_scale",
            torch.nn.Parameter(
                torch.ones(1, device=device, dtype=torch.float32),
                requires_grad=False,
            ),
        )
        module._hswq_nvfp4_scale_placeholder = True
    else:
        module._hswq_nvfp4_scale_from_ckpt = True
        module._hswq_nvfp4_scale_placeholder = False
    if enabled:
        logger.debug(
            "[HSWQ NVFP4] ConvRot armed groupsize=%s on %s",
            gs,
            getattr(module, "in_features", "?"),
        )


def validate_nvfp4_weight_storage(module, weight) -> None:
    """Ensure packed uint8 storage matches TensorCoreNVFP4Layout for _orig_shape."""
    import torch
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout

    orig = getattr(module, "_orig_shape", None)
    if orig is None or not torch.is_tensor(weight) or weight.ndim != 2:
        return
    expected = TensorCoreNVFP4Layout.get_storage_shape(tuple(int(x) for x in orig))
    got = tuple(int(x) for x in weight.shape)
    if got != expected:
        raise ValueError(
            f"[HSWQ NVFP4] weight storage shape mismatch: got {got}, "
            f"expected {expected} for orig_shape={tuple(orig)}"
        )


def load_nvfp4_linear_module(
    module,
    super_load,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
    load_extra_params: bool = True,
) -> None:
    """
    Full NVFP4 Linear ``_load_from_state_dict`` body (HSWQ).

    Mirrors stock scale/QT construction, then:
      - validates storage vs logical shape
      - loads input_scale when present
      - arms ConvRot + HSWQ TC forward flags
    """
    import torch
    from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor, get_layout_class

    device = module.factory_kwargs["device"]
    compute_dtype = module.factory_kwargs["dtype"]
    disabled_formats = module._disabled_formats
    layer_name = prefix.rstrip(".")

    weight = state_dict.pop(f"{prefix}weight", None)
    if weight is None:
        logger.warning("Missing weight for layer %s", layer_name)
        module.weight = None
        return
    manually_loaded_keys = [f"{prefix}weight"]

    def pop_scale(name, dtype=None):
        key = f"{prefix}{name}"
        v = state_dict.pop(key, None)
        if v is not None:
            v = v.to(device=device)
            if dtype is not None:
                v = v.view(dtype=dtype)
            manually_loaded_keys.append(key)
        return v

    layer_conf_raw = state_dict.pop(f"{prefix}comfy_quant", None)
    layer_conf = decode_comfy_quant_conf(layer_conf_raw)
    if layer_conf_raw is not None:
        manually_loaded_keys.append(f"{prefix}comfy_quant")

    if not is_nvfp4_conf(layer_conf):
        raise ValueError(
            f"[HSWQ NVFP4] load_nvfp4_linear_module called for non-nvfp4 "
            f"layer {layer_name}: {layer_conf}"
        )

    module.quant_format = "nvfp4"
    module._full_precision_mm_config = bool(layer_conf.get("full_precision_matrix_mult", False))
    if not module._full_precision_mm:
        module._full_precision_mm = module._full_precision_mm_config
    if module.quant_format in disabled_formats:
        module._full_precision_mm = True

    qconfig = QUANT_ALGOS["nvfp4"]
    module.layout_type = qconfig["comfy_tensor_layout"]
    layout_cls = get_layout_class(module.layout_type)

    ts = pop_scale("weight_scale_2")
    bs = pop_scale("weight_scale", torch.float8_e4m3fn)
    if ts is None or bs is None:
        raise ValueError(f"Missing NVFP4 scales for layer {layer_name}")

    # Prefer logical shape from comfy_quant. Detect / Linear ctor may have
    # stamped module._orig_shape from packed weight.shape[1] (or packed*2
    # pad guess); Params must use orig_shape / out×in from sidecar.
    orig_shape = getattr(module, "_orig_shape", None)
    conf_orig = layer_conf.get("orig_shape") if isinstance(layer_conf, dict) else None
    if conf_orig is not None and len(conf_orig) >= 2:
        orig_shape = tuple(int(x) for x in conf_orig)
    elif (
        isinstance(layer_conf, dict)
        and layer_conf.get("out_features") is not None
        and layer_conf.get("in_features") is not None
    ):
        orig_shape = (int(layer_conf["out_features"]), int(layer_conf["in_features"]))
    if orig_shape is None:
        raise ValueError(
            f"[HSWQ NVFP4] Missing orig_shape for quantized layer {layer_name}"
        )
    if hasattr(module, "_orig_shape"):
        module._orig_shape = orig_shape
    if len(orig_shape) >= 2 and hasattr(module, "out_features") and hasattr(module, "in_features"):
        module.out_features = int(orig_shape[0])
        module.in_features = int(orig_shape[1])

    validate_nvfp4_weight_storage(module, weight)

    params = layout_cls.Params(
        scale=ts,
        block_scale=bs,
        orig_dtype=compute_dtype,
        orig_shape=orig_shape,
    )
    module.weight = torch.nn.Parameter(
        QuantizedTensor(
            weight.to(device=device, dtype=qconfig["storage_t"]),
            module.layout_type,
            params,
        ),
        requires_grad=False,
    )

    if load_extra_params:
        for param_name in qconfig["parameters"]:
            if param_name in {"weight_scale", "weight_scale_2"}:
                continue
            param_key = f"{prefix}{param_name}"
            _v = state_dict.pop(param_key, None)
            if _v is None:
                continue
            module.register_parameter(
                param_name, torch.nn.Parameter(_v.to(device=device), requires_grad=False)
            )
            manually_loaded_keys.append(param_key)

    arm_nvfp4_module(module, layer_conf)

    super_load(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    for key in manually_loaded_keys:
        if key in missing_keys:
            missing_keys.remove(key)
