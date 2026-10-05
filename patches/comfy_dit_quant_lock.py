"""HSWQ DiT quantized-weight lock guard, migrated out of ComfyUI core.

Background
----------
INT8/NVFP4 comfy_quant DiT packs (HSWQ, SeedVR2, ...) need ComfyUI's generic
weight-casting paths to keep QuantizedTensor weights quantized instead of
silently fp16-expanding them, and must not let layer-config or disabled-format
hints downgrade the authoritative quantized pack. This guard was historically
implemented by editing comfy/ops.py directly, which is silently lost on every
ComfyUI update.

This module installs the identical guard at runtime from inside the HSWQ custom
node (prestartup ``__import__`` hook that fires when comfy.ops is imported).
It transforms the *stock* source of the target comfy.ops functions with exact
anchor replacements and compiles them using comfy.ops' own globals dict, so
behaviour matches the previously working core edit byte-for-byte.

All-or-nothing: if any anchor does not match (ComfyUI changed the function
bodies), NOTHING is installed and a loud error is logged. No partial patches,
no silent fallback -- the fast-disk rule applies here too.

Scope (never touches VAE or plain modules)
------------------------------------------
A module is "locked" when it carries ``quant_format`` (set from the pack's
comfy_quant metadata by _load_quantized_module) with a known quantized
layout_type. This matches the previous core-edit predicate exactly, byte for
byte in behavior. The separate per-tensor ``_dit_quant_locked`` flag (set by
ComfyUI-SeedVR2 *_native_ops) is consumed inside comfy_kitchen tensor/base.py
and is NOT handled here, unchanged from before.
"""
from __future__ import annotations

import inspect
import logging
import textwrap

import torch

logger = logging.getLogger("HSWQ")

_INSTALL_MARK = "_hswq_dit_lock_installed"
_FN_TAG = "_hswq_dit_lock_fn"

_DIT_QUANT_LAYOUTS = (
    "TensorWiseINT8Layout", "TensorCoreNVFP4Layout", "TensorCoreFP8Layout",
    "TensorCoreMXFP8Layout", "TensorCoreConvRotW4A4Layout",
    "TensorCoreAWQW4A16Layout", "TensorCoreSVDQuantW4A4Layout",
    "AsymW4A8Int8Layout",
)

_DIT_LOCKED_DTYPES = tuple(
    d for d in (
        torch.int8, torch.uint8,
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e8m0fnu", None),
    ) if d is not None
)


def _dit_quant_lock(module):
    """Return True when this module must keep its weight quantized.

    Only quantized DiT Linear/MoE layers opt in: either the pack set
    ``quant_format`` on the module (with a known layout) or the weight itself
    carries the SeedVR2-style ``_dit_quant_locked`` flag.
    """
    if module is None:
        return False
    if getattr(module, "quant_format", None) is None:
        return False
    name = getattr(module, "layout_type", None)
    if name is not None and name not in _DIT_QUANT_LAYOUTS:
        return False
    return True


def _dit_dtype_is_locked(dtype):
    return dtype in _DIT_LOCKED_DTYPES


# ---- anchor replacements against the *stock* comfy/ops.py (v0.38.x) ---------
# old -> new pairs are applied with str.replace after asserting the old text
# occurs exactly once in the function source.

_REPLACEMENTS = {
    "resolve_cast_module_with_vbar": [(
        "        if (return_weights and orig.dtype != dtype) or len(fns) > 0:\n"
        "            x = to_dequant(x, dtype)\n",
        "        if (_dit_quant_lock(s) and len(fns) == 0\n"
        "                and isinstance(x, QuantizedTensor)):\n"
        "            # Keep quantized; skip fp16 dequant in the prefetch/post_cast path.\n"
        "            pass\n"
        "        elif (return_weights and orig.dtype != dtype) or len(fns) > 0:\n"
        "            x = to_dequant(x, dtype)\n",
    )],
    "cast_bias_weight": [
        (
            "        materialize_meta_param(s, [\"weight\", \"bias\"])\n"
            "        weight = s.weight.to(dtype=dtype, copy=True)\n"
            "        if isinstance(weight, QuantizedTensor):\n"
            "            weight = weight.dequantize()\n",
            "        materialize_meta_param(s, [\"weight\", \"bias\"])\n"
            "        if _dit_quant_lock(s) and isinstance(s.weight, QuantizedTensor):\n"
            "            # Quantized DiT weight must never be materialized (fp16-expanded) on CPU.\n"
            "            weight = s.weight\n"
            "        else:\n"
            "            weight = s.weight.to(dtype=dtype, copy=True)\n"
            "            if isinstance(weight, QuantizedTensor):\n"
            "                weight = weight.dequantize()\n",
        ),
        (
            "    if weight_has_function or weight.dtype != dtype:\n"
            "        weight = weight.to(dtype=dtype)\n"
            "        if isinstance(weight, QuantizedTensor):\n"
            "            weight = weight.dequantize()\n",
            "    if _dit_quant_lock(s) and isinstance(weight, QuantizedTensor) and len(s.weight_function) == 0:\n"
            "        # Keep the QuantizedTensor as-is (int8/int4/fp8 storage). Do not\n"
            "        # dequantize to fp16 even though weight.dtype (logical) != dtype.\n"
            "        pass\n"
            "    elif weight_has_function or weight.dtype != dtype:\n"
            "        weight = weight.to(dtype=dtype)\n"
            "        if isinstance(weight, QuantizedTensor):\n"
            "            weight = weight.dequantize()\n",
        ),
    ],
    "_load_quantized_module": [(
        "        if not module._full_precision_mm:\n"
        "            module._full_precision_mm = module._full_precision_mm_config\n"
        "        if module.quant_format in disabled_formats:\n"
        "            module._full_precision_mm = True\n",
        "        if _dit_quant_lock(module):\n"
        "            # Quantized DiT: the pack is authoritative. Never let the layer\n"
        "            # config or a disabled-format hint enable full-precision mm.\n"
        "            module._full_precision_mm = False\n"
        "            module._full_precision_mm_config = False\n"
        "        else:\n"
        "            if not module._full_precision_mm:\n"
        "                module._full_precision_mm = module._full_precision_mm_config\n"
        "            if module.quant_format in disabled_formats:\n"
        "                module._full_precision_mm = True\n",
    )],
    "mixed_precision_ops": [(
        "                        if self._full_precision_mm and isinstance(weight, QuantizedTensor):\n"
        "                            weight = weight.dequantize()\n",
        "                        if (self._full_precision_mm and isinstance(weight, QuantizedTensor)\n"
        "                                and not _dit_quant_lock(self)):\n"
        "                            weight = weight.dequantize()\n",
    )],
}

_METHOD_REPLACEMENTS = {
    ("MixedPrecisionOp", "can_use_quantized_matmul"): [(
        "        return (self.quant_format in QUANT_ALGOS\n",
        "        if _dit_quant_lock(self):\n"
        "            # Quantized DiT: force the quantized matmul; never fall back.\n"
        "            return True\n"
        "        return (self.quant_format in QUANT_ALGOS\n",
    )],
}


def _transform(fn, replacements, g):
    """Return a transformed copy of fn compiled against globals g, or raise."""
    src = inspect.getsource(fn).replace("\r\n", "\n")
    if getattr(fn, _FN_TAG, False):
        return None  # already transformed (idempotent)
    for old, new in replacements:
        if old not in src and new in src:
            # The guard text is already in the source (e.g. someone kept the
            # core edit applied). Do not double-apply; keep stock fn.
            return "ALREADY"
        count = src.count(old)
        if count != 1:
            raise RuntimeError(
                f"anchor mismatch in {fn.__name__} (found {count} occurrences); "
                "ComfyUI ops.py changed since this guard was written. "
                "NOT installing any DiT-lock patch."
            )
        src = src.replace(old, new, 1)
    ns = {}
    src_c = textwrap.dedent(src)  # class methods start with a uniform leading indent
    exec(compile(src_c, f"<hswq-dit-lock {fn.__qualname__}>", "exec"), g, ns)
    out = ns[fn.__name__]
    setattr(out, _FN_TAG, True)
    return out


def apply_comfy_dit_quant_lock_patch() -> bool:
    """Install the DiT quantized-weight lock guard into comfy.ops.

    Returns True when installed (or already installed), False on anchor
    mismatch (with a loud error; nothing partial is applied).
    """
    try:
        import comfy.ops as ops
    except ImportError:
        return False
    if getattr(ops, _INSTALL_MARK, False):
        return True
    g = vars(ops)

    # 1) fetch/compile everything into a staging area first (all-or-nothing)
    staged = []
    try:
        for fname, reps in _REPLACEMENTS.items():
            orig = g.get(fname)
            if orig is None:
                raise RuntimeError(f"comfy.ops.{fname} missing; NOT installing.")
            result = _transform(orig, reps, g)
            if result == "ALREADY":
                staged.append((None, None, fname, orig, "fn"))
            elif result is not None:
                staged.append((None, None, fname, result, "fn"))
            else:
                staged.append((None, None, fname, orig, "fn"))
        for (cls_name, mname), reps in _METHOD_REPLACEMENTS.items():
            cls = g.get(cls_name)
            method = getattr(cls, mname, None) if cls is not None else None
            if method is None:
                raise RuntimeError(f"comfy.ops.{cls_name}.{mname} missing; NOT installing.")
            result = _transform(method, reps, g)
            if result not in ("ALREADY", None):
                staged.append((cls_name, None, mname, result, "method"))
    except Exception as exc:
        logger.error(
            "[HSWQ DiT-lock] install aborted, guard NOT applied: %s "
            "(comfy/ops.py structure changed; update patches/comfy_dit_quant_lock.py)",
            exc,
        )
        return False

    # 2) inject helpers, then swap functions (the transformed bodies look
    #    these up in comfy.ops globals at call time)
    g.setdefault("_dit_quant_lock", _dit_quant_lock)
    g.setdefault("_dit_dtype_is_locked", _dit_dtype_is_locked)
    g.setdefault("_DIT_QUANT_LAYOUTS", _DIT_QUANT_LAYOUTS)
    g.setdefault("_DIT_LOCKED_DTYPES", _DIT_LOCKED_DTYPES)

    for cls_name, _unused, name, value, kind in staged:
        if kind == "fn":
            setattr(ops, name, value)
        else:
            setattr(g[cls_name], name, value)

    setattr(ops, _INSTALL_MARK, True)
    logger.info(
        "[HSWQ DiT-lock] guard installed on stock comfy.ops "
        "(resolve_cast_module_with_vbar, cast_bias_weight, _load_quantized_module, "
        "MixedPrecisionOp.can_use_quantized_matmul, mixed_precision_ops)"
    )
    return True
