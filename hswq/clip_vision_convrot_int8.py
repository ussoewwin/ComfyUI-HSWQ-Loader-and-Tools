"""ConvRot INT8 execution for CLIP Vision towers.

``comfy.clip_vision`` builds every tower with ``comfy.ops.manual_cast``. That op set is
a ``disable_weight_init`` derivative and never calls ``_load_quantized_module``, so a
``comfy_quant`` layer inside a CLIP Vision pack arrives as an int8 code block sitting in
a float Parameter: the tower runs on the codes and every GEMM is a float GEMM on values
that mean nothing.

This module gives the CLIP Vision route the same quantised op set the rest of the loader
uses. It is scoped to CLIP Vision only - it changes no shared module, no op class, and no
other model family. Nothing here runs unless a pack actually declares ``comfy_quant``.

How a pack is executed
----------------------
Each declared layer is swapped for ``comfy.ops.mixed_precision_ops``' own ``Linear`` and
its ``{weight, weight_scale, bias, comfy_quant}`` sub-dict is loaded into it. From there
the layer holds a ``QuantizedTensor`` with ``quant_format == "int8_tensorwise"`` and the
ConvRot group rotation is applied inside the kernel, so the GEMM is a real INT8 GEMM
(``comfy_kitchen.backends.cuda.int8_linear``) rather than a dequantise-then-float-matmul.

The float weights the tower was built with are replaced, not wrapped, so a quantised pack
also occupies its own size, not the float size.
"""

from __future__ import annotations

import json
import logging

import torch

import comfy.clip_vision
import comfy.ops
import comfy.utils

logger = logging.getLogger(__name__)

PACK_FORMAT = "int8_tensorwise"
QUANT_KEY_SUFFIX = ".comfy_quant"


def declared_layers(state_dict):
    """Layer names a pack declares as quantised, in pack order."""
    names = []
    for key in state_dict:
        if not key.endswith(QUANT_KEY_SUFFIX):
            continue
        conf = _decode(state_dict[key])
        if conf is None or conf.get("format") != PACK_FORMAT:
            continue
        names.append(key[: -len(QUANT_KEY_SUFFIX)])
    return names


def is_convrot_int8_pack(state_dict):
    """True when the file declares ConvRot INT8 layers."""
    return len(declared_layers(state_dict)) > 0


def _decode(raw):
    try:
        return json.loads(raw.numpy().tobytes().decode("utf-8"))
    except Exception:
        return None


def _layer_state(state_dict, name):
    prefix = name + "."
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def apply_convrot_int8(clip_vision, state_dict):
    """Swap every declared layer for a quantised Linear and load its codes.

    ``clip_vision`` is the object ``comfy.clip_vision`` returns; its ``.model`` is the
    tower. Returns a dict of what happened so the caller can log it rather than guess.
    """
    names = declared_layers(state_dict)
    model = clip_vision.model
    ops = comfy.ops.mixed_precision_ops(quant_config={})

    replaced = 0
    skipped = []
    for name in names:
        parent_path, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        old = getattr(parent, leaf, None)
        if not isinstance(old, torch.nn.Linear):
            skipped.append((name, type(old).__name__))
            continue
        new = ops.Linear(
            old.in_features,
            old.out_features,
            bias=old.bias is not None,
            device=old.weight.device,
            dtype=old.weight.dtype,
        )
        missing, unexpected = new.load_state_dict(_layer_state(state_dict, name), strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "CLIP Vision ConvRot INT8: layer %s did not fit its declared layout "
                "(missing %s, unexpected %s)" % (name, list(missing), list(unexpected))
            )
        setattr(parent, leaf, new)
        replaced += 1

    armed = sum(
        1
        for m in model.modules()
        if getattr(m, "quant_format", None) == PACK_FORMAT
    )
    report = {
        "declared": len(names),
        "replaced": replaced,
        "armed": armed,
        "skipped": skipped,
    }
    if replaced != len(names) or armed != replaced:
        raise RuntimeError(
            "CLIP Vision ConvRot INT8: %d declared, %d replaced, %d armed - refusing to "
            "run a partially quantised tower" % (len(names), replaced, armed)
        )
    return report


def load(path):
    """Load a CLIP Vision file, executing its ConvRot INT8 layers when it declares any.

    A file with no ``comfy_quant`` layer goes through ``comfy.clip_vision.load`` unchanged,
    so the plain route keeps the stock behaviour byte for byte.
    """
    state_dict = comfy.utils.load_torch_file(path)
    if not is_convrot_int8_pack(state_dict):
        return comfy.clip_vision.load(path)

    # load_clipvision_from_sd reads the keys to pick the tower config, then hands each
    # tensor to the model. It mutates the dict it is given, so hand it a shallow copy:
    # the tensors are shared, the key set is not.
    clip_vision = comfy.clip_vision.load_clipvision_from_sd(dict(state_dict))
    if clip_vision is None:
        raise RuntimeError(
            "ERROR: clip vision file is invalid and does not contain a valid vision model."
        )
    report = apply_convrot_int8(clip_vision, state_dict)
    logger.info(
        "[HSWQ CLIP Vision] ConvRot INT8 armed: %d declared, %d replaced, %d armed, %d skipped",
        report["declared"], report["replaced"], report["armed"], len(report["skipped"]),
    )
    for name, kind in report["skipped"]:
        logger.info("[HSWQ CLIP Vision] skipped %s (%s) - not a Linear", name, kind)
    return clip_vision
