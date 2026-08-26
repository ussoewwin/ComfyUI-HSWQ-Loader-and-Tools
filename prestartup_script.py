"""Wire Z Image UNet ConvRot NVFP4 without regressing the product loader.

ComfyUI runs this before the custom-node ``__init__.py``. We keep a reference to
the *original* ``comfy_quant_nvfp4.load_unet_nvfp4_weight_dtype`` (INT8 protect +
disable_dynamic + LoRA bake + stock GEMM + act rotate), then optionally rebind
the module attribute to ``nodes.zimage_nvfp4.load_unet`` which *delegates* to that
saved original ? never to the rebound name (that would recurse).

SDXL ``load_checkpoint_sdxl_nvfp4_weight_dtype`` is left unchanged.

Do NOT insert this package root onto ``sys.path``. That shadows ComfyUI's top-level
``nodes`` module and crashes startup with::

    AttributeError: module 'nodes' has no attribute 'init_extra_nodes'
"""
from __future__ import annotations

import builtins
import importlib
import os
import sys

# --- cp932 → UTF-8: apply before any file is read with the locale encoding ---
import importlib.util as _ilu
import os as _os

_utf8_patch_spec = _ilu.spec_from_file_location(
    "win_utf8_patch",
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "win_utf8_patch.py"),
)
_utf8_patch_mod = _ilu.module_from_spec(_utf8_patch_spec)
_utf8_patch_spec.loader.exec_module(_utf8_patch_mod)
del _ilu, _os, _utf8_patch_spec, _utf8_patch_mod

_ROOT = os.path.dirname(os.path.abspath(__file__))

_PATCHED = False
_CTRL_PATCHED = False
_ORIG_IMPORT = builtins.__import__
_PRODUCT_LOAD_UNET = None


def _zimage_load_module():
    """Resolve zimage load only via the already-imported HSWQ package prefix."""
    for name in list(sys.modules):
        if not name.endswith("nodes.nvfp4.comfy_quant_nvfp4"):
            continue
        pkg = name[: -len(".nodes.nvfp4.comfy_quant_nvfp4")]
        if not pkg:
            continue
        return importlib.import_module(f"{pkg}.nodes.zimage_nvfp4.load_unet")
    raise ImportError(
        "comfy_quant_nvfp4 not in sys.modules yet "
        "(cannot import nodes.zimage_nvfp4 without shadowing ComfyUI nodes)"
    )


def _krea2_load_module():
    """Resolve krea2 load only via the already-imported HSWQ package prefix."""
    for name in list(sys.modules):
        if not name.endswith("nodes.nvfp4.comfy_quant_nvfp4"):
            continue
        pkg = name[: -len(".nodes.nvfp4.comfy_quant_nvfp4")]
        if not pkg:
            continue
        return importlib.import_module(f"{pkg}.nodes.krea2_convrot_nvfp4.load_unet")
    raise ImportError(
        "comfy_quant_nvfp4 not in sys.modules yet "
        "(cannot import nodes.krea2_convrot_nvfp4 without shadowing ComfyUI nodes)"
    )

def _try_patch() -> bool:
    global _PATCHED, _PRODUCT_LOAD_UNET
    if _PATCHED:
        return True
    try:
        zl = _zimage_load_module()
    except Exception as e:
        print(f"[HSWQ NVFP4] Z Image load import deferred: {e}", flush=True)
        return False
    try:
        _krea2_load_module()
    except Exception as e:
        print(f"[HSWQ NVFP4] Krea2 load import deferred: {e}", flush=True)
    for name, mod in list(sys.modules.items()):
        if not (
            name.endswith("nodes.nvfp4.comfy_quant_nvfp4")
            or name.endswith(".comfy_quant_nvfp4")
            or name == "comfy_quant_nvfp4"
        ):
            continue
        if not hasattr(mod, "load_unet_nvfp4_weight_dtype"):
            continue
        # Save product implementation *before* rebind (avoid recursion).
        _PRODUCT_LOAD_UNET = mod.load_unet_nvfp4_weight_dtype
        zl._PRODUCT_LOAD_UNET = _PRODUCT_LOAD_UNET
        mod.load_unet_nvfp4_weight_dtype = zl.load_unet_nvfp4_weight_dtype
        _PATCHED = True
        print(
            "[HSWQ NVFP4] UNet ConvRot NVFP4 -> nodes.zimage_nvfp4 "
            "(delegates to saved product: GEMM + act rotate + int8 + LoRA bake + "
            "disable_dynamic)",
            flush=True,
        )
        return True
    return False


def _patch_controlnet_int8() -> bool:
    """Retrofit: stock "Load ControlNet Model" node keeps INT8 weights in VRAM.

    ComfyUI builds the ControlNet from ``weight_dtype(sd)``; for an HSWQ INT8
    checkpoint that is ``torch.int8``, so ``operations.Linear(dtype=int8)``
    crashes and ``comfy_quant``/``weight_scale`` are ignored.

    We wrap ``comfy.controlnet.load_controlnet_state_dict`` and inject
    ``dtype=bf16 + MixedPrecisionOps(int8_tensorwise)`` when the checkpoint
    carries int8_tensorwise markers. Same native QuantizedTensor path as the
    dedicated HSWQ node; ``custom_operations`` already present is respected
    (no double injection).
    """
    global _CTRL_PATCHED
    if _CTRL_PATCHED:
        return True
    try:
        import json

        import torch

        import comfy.controlnet as cn
        from comfy import ops
        from comfy.quant_ops import QUANT_ALGOS
    except Exception as e:  # noqa: BLE001
        print(f"[HSWQ INT8 CN] import deferred: {e}", flush=True)
        return False

    _orig = cn.load_controlnet_state_dict

    def _has_int8(sd) -> bool:
        for key in sd.keys():
            if not key.endswith(".comfy_quant"):
                continue
            try:
                conf = json.loads(sd[key].numpy().tobytes())
            except Exception:  # noqa: BLE001
                continue
            if conf.get("format") == "int8_tensorwise":
                return True
        return False

    def _load_controlnet_state_dict(state_dict, model=None, model_options={}):
        opts = dict(model_options)
        if _has_int8(state_dict) and "custom_operations" not in opts:
            opts["dtype"] = torch.bfloat16
            opts["custom_operations"] = ops.mixed_precision_ops(
                {"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]},
                torch.bfloat16,
                full_precision_mm=False,
                disabled=[],
            )
        return _orig(state_dict, model=model, model_options=opts)

    cn.load_controlnet_state_dict = _load_controlnet_state_dict
    _CTRL_PATCHED = True
    print(
        "[HSWQ INT8 CN] Retrofit armed: stock Load ControlNet node keeps INT8 "
        "weights in VRAM",
        flush=True,
    )
    return True


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    mod = _ORIG_IMPORT(name, globals, locals, fromlist, level)
    if not _CTRL_PATCHED and (
        str(name) == "comfy.controlnet"
        or (str(name) == "comfy" and any(str(x) == "controlnet" for x in (fromlist or ())))
    ):
        _patch_controlnet_int8()
    if not _PATCHED and "comfy_quant_nvfp4" in str(name):
        _try_patch()
    elif not _PATCHED and fromlist:
        if any("comfy_quant_nvfp4" in str(x) for x in fromlist):
            _try_patch()
    return mod


builtins.__import__ = _import
print(
    "[HSWQ NVFP4] prestartup: Z Image ConvRot NVFP4 product path armed",
    flush=True,
)
_try_patch()