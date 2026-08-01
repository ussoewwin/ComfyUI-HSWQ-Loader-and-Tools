"""Wire Z Image UNet ConvRot NVFP4 to the product path (no edits to existing files).

ComfyUI runs this before the custom-node ``__init__.py``. When
``comfy_quant_nvfp4`` is imported, replace ``load_unet_nvfp4_weight_dtype`` with
``nodes.zimage_nvfp4.load_unet`` (stock Comfy GEMM + online act rotate).

SDXL ``load_checkpoint_sdxl_nvfp4_weight_dtype`` is left unchanged.
"""
from __future__ import annotations

import builtins
import importlib
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_PATCHED = False
_ORIG_IMPORT = builtins.__import__


def _zimage_load_fn():
    for name in list(sys.modules):
        if not name.endswith("nodes.nvfp4.comfy_quant_nvfp4"):
            continue
        pkg = name[: -len(".nodes.nvfp4.comfy_quant_nvfp4")]
        if not pkg:
            continue
        return importlib.import_module(
            f"{pkg}.nodes.zimage_nvfp4.load_unet"
        ).load_unet_nvfp4_weight_dtype
    return importlib.import_module(
        "nodes.zimage_nvfp4.load_unet"
    ).load_unet_nvfp4_weight_dtype


def _try_patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    try:
        fn = _zimage_load_fn()
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
        mod.load_unet_nvfp4_weight_dtype = fn
        _PATCHED = True
        print(
            "[HSWQ NVFP4] UNet ConvRot NVFP4 -> nodes.zimage_nvfp4 "
            "(stock Comfy GEMM + act rotate)",
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
