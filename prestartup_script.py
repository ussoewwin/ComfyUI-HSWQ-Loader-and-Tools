"""Wire Z Image UNet ConvRot NVFP4 without regressing the product loader.

ComfyUI runs this before the custom-node ``__init__.py``. We keep a reference to
the *original* ``comfy_quant_nvfp4.load_unet_nvfp4_weight_dtype`` (INT8 protect +
disable_dynamic + LoRA bake + stock GEMM + act rotate), then optionally rebind
the module attribute to ``nodes.zimage_nvfp4.load_unet`` which *delegates* to that
saved original — never to the rebound name (that would recurse).

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

_ROOT = os.path.dirname(os.path.abspath(__file__))

_PATCHED = False
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


def _try_patch() -> bool:
    global _PATCHED, _PRODUCT_LOAD_UNET
    if _PATCHED:
        return True
    try:
        zl = _zimage_load_module()
    except Exception as e:
        print(f"[HSWQ NVFP4] Z Image load import deferred: {e}", flush=True)
        return False
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


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    mod = _ORIG_IMPORT(name, globals, locals, fromlist, level)
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
