"""Flux1 ConvRot W4A8 — conf peek, storage validation, per-module arm flags.

Deliberately NOT a custom ``_load_quantized_module``: ComfyUI core already loads
``asym_w4a8_int8`` (measured, ``comfy/ops.py`` grouped-int8 branch):

  - validates ``weight.shape[1] * 8 == _orig_shape[1] * bits``
  - pops ``weight_s_rel`` (fp8), ``weight_s_channel``, ``weight_codebook``
  - builds ``AsymW4A8Int8Layout.Params(scale, s_channel, codebook, group_size,
    convrot_groupsize, orig_dtype, orig_shape)``
  - ``QUANT_ALGOS["asym_w4a8_int8"]`` = storage int8 / layout AsymW4A8Int8Layout

So this module only supplies what the loader owns on top of core: reading the
conf before stock load, checking the storage shape against the logical shape,
and stamping per-module W4A8 flags used by the forward / bake.
"""
from __future__ import annotations

import logging
from typing import Optional

from .w4a8_conf import (
    W4A8_BITS,
    convrot_flags_from_conf,
    decode_comfy_quant_conf,
    is_w4a8_conf,
    quant_params_from_conf,
)

logger = logging.getLogger(__name__)


def peek_w4a8_conf(state_dict, prefix: str) -> Optional[dict]:
    """Read comfy_quant without popping (routing decisions before stock load)."""
    return decode_comfy_quant_conf(state_dict.get(f"{prefix}comfy_quant"))


def is_w4a8_layer(module) -> bool:
    """True when this Linear was loaded as an asym_w4a8_int8 pack."""
    if module is None:
        return False
    if getattr(module, "quant_format", None) == "asym_w4a8_int8":
        return True
    return bool(getattr(module, "_hswq_w4a8", False))


def arm_w4a8_module(module, conf: Optional[dict]) -> None:
    """Stamp HSWQ W4A8 flags on a module whose packed weight is in place."""
    if not is_w4a8_conf(conf):
        return
    enabled, gs = convrot_flags_from_conf(conf)
    params = quant_params_from_conf(conf)
    module._hswq_w4a8 = True
    module._hswq_w4a8_convrot = bool(enabled)
    module._hswq_w4a8_convrot_groupsize = int(gs)
    module._hswq_w4a8_group_size = int(params["group_size"])
    logger.debug(
        "[HSWQ W4A8] armed groupsize=%s quant_group=%s on %s",
        gs,
        params["group_size"],
        getattr(module, "in_features", "?"),
    )


def validate_w4a8_weight_storage(module, weight) -> None:
    """Packed storage width must equal logical_K * bits / 8."""
    import torch

    if not torch.is_tensor(weight) or weight.ndim != 2:
        return
    orig = getattr(module, "_orig_shape", None)
    if orig is None or len(orig) < 2:
        return
    width = int(weight.shape[1])
    expected = int(orig[1]) * W4A8_BITS // 8
    if width != expected:
        raise ValueError(
            f"[HSWQ W4A8] packed width mismatch: got {width}, expected "
            f"{expected} for orig_shape={tuple(int(x) for x in orig)} "
            f"({W4A8_BITS} bits)"
        )


def w4a8_weight_params(module) -> Optional[dict]:
    """Return the AsymW4A8Int8Layout params of this module's QT weight, or None."""
    try:
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return None
    weight = getattr(module, "weight", None)
    qt = weight if isinstance(weight, QuantizedTensor) else getattr(weight, "data", None)
    if qt is None or not isinstance(qt, QuantizedTensor):
        return None
    params = getattr(qt, "_params", None)
    if params is None or not hasattr(params, "codebook"):
        return None
    return params


def w4a8_convrot_groupsize(module) -> Optional[int]:
    """ConvRot groupsize of this W4A8 layer (256 in every pack we build)."""
    if not is_w4a8_layer(module):
        return None
    params = w4a8_weight_params(module)
    gs = getattr(params, "convrot_groupsize", None) if params is not None else None
    if gs is None:
        gs = getattr(module, "_hswq_w4a8_convrot_groupsize", None)
    return int(gs) if gs else None
