"""Runtime patch: run ConvRot INT8 IP-Adapter packs through the INT8 GEMM.

The stock ComfyUI IP-Adapter extension rebuilds every layer as a plain
``nn.Linear`` and assigns ``weight.data = value``. For a ConvRot INT8 pack
that is wrong twice over: the int8 codes would be read as float weights, and
the ``weight_scale`` / ``comfy_quant`` sidecars would not fit a Linear at
all.

This module installs the INT8 payload into the graph the same way the Forge
IP-Adapter runtime does (``sd_forge_ipadapter/lib_ipadapter/convrot_int8.py``):
the codes stay INT8 and are evaluated with the ConvRot INT8 GEMM
(``comfy_kitchen.int8_linear(..., convrot=True)``). Nothing is dequantized in
the runtime path.

How it is applied:
  * ``To_KV`` (module global of ``comfyui_ipadapter_plus.IPAdapterPlus``) is
    replaced by a subclass that builds ``ConvRotInt8Linear`` for the
    quantised bases and plain ``nn.Linear`` for the float remainder.
  * ``IPAdapter`` (same module) is replaced by a subclass whose ``__init__``
    installs the quantised layers into ``image_proj_model`` and loads only the
    float remainder into it.

Both subclasses delegate to the original implementation when the state dict
carries no ConvRot INT8 sidecar, so float packs take the stock path
unchanged.

Scope:
  * No file of ComfyUI and no file of the IP-Adapter extension is modified.
    The patch lives in this package and replaces two module globals at
    runtime, only when a ConvRot INT8 pack is actually loaded.
  * IP-Adapter route only. No other model family is touched.
"""

from __future__ import annotations

import logging
import sys

import torch
from torch import nn

try:
    from ..hswq.ipadapter_convrot_int8 import (
        ConvRotInt8Linear,
        has_convrot_int8,
        install_convrot_linears,
        split_convrot_state_dict,
    )
except ImportError:  # direct-script runs (unit tests) with the repo root on sys.path
    from hswq.ipadapter_convrot_int8 import (
        ConvRotInt8Linear,
        has_convrot_int8,
        install_convrot_linears,
        split_convrot_state_dict,
    )

logger = logging.getLogger(__name__)

_LOG = "[HSWQ IPAdapter ConvRot INT8]"
_MODULE_MARKER = "_hswq_convrot_int8_patched"


def _find_ipadapter_module():
    """Locate the loaded ``comfyui_ipadapter_plus`` module.

    ComfyUI imports custom nodes by filesystem path, so the module name is a
    path, not a package name. The module is identified by its file location
    and by the two globals it must provide.
    """
    for module in list(sys.modules.values()):
        if module is None:
            continue
        path = (getattr(module, "__file__", "") or "").replace("\\", "/")
        if "comfyui_ipadapter_plus" not in path:
            continue
        if not hasattr(module, "IPAdapter") or not hasattr(module, "To_KV"):
            continue
        return module
    return None


def _make_to_kv(base_to_kv):
    """INT8-aware ``To_KV``: ConvRotInt8Linear for quantised bases."""

    class To_KVConvRotInt8(base_to_kv):
        def __init__(self, state_dict):
            if not has_convrot_int8(state_dict):
                super().__init__(state_dict)
                return

            nn.Module.__init__(self)
            self.to_kvs = nn.ModuleDict()

            quantized, remainder = split_convrot_state_dict(state_dict)
            for base, entry in quantized.items():
                name = base.replace(".weight", "").replace(".", "_")
                self.to_kvs[name] = ConvRotInt8Linear.from_weight(
                    entry["weight"], entry["weight_scale"], entry["groupsize"]
                )

            for key, value in remainder.items():
                if not key.endswith(".weight"):
                    continue
                name = key.replace(".weight", "").replace(".", "_")
                self.to_kvs[name] = nn.Linear(value.shape[1], value.shape[0], bias=False)
                self.to_kvs[name].weight.data = value

    return To_KVConvRotInt8


def _make_ipadapter(base_ipadapter, patched_to_kv):
    """INT8-aware ``IPAdapter``: installs the quantised image_proj layers."""

    class IPAdapterConvRotInt8(base_ipadapter):
        def __init__(
            self,
            ipadapter_model,
            cross_attention_dim=1024,
            output_cross_attention_dim=1024,
            clip_embeddings_dim=1024,
            clip_extra_context_tokens=4,
            is_sdxl=False,
            is_plus=False,
            is_full=False,
            is_faceid=False,
            is_portrait_unnorm=False,
        ):
            image_proj = ipadapter_model.get("image_proj") or {}
            if not has_convrot_int8(image_proj):
                super().__init__(
                    ipadapter_model,
                    cross_attention_dim,
                    output_cross_attention_dim,
                    clip_embeddings_dim,
                    clip_extra_context_tokens,
                    is_sdxl,
                    is_plus,
                    is_full,
                    is_faceid,
                    is_portrait_unnorm,
                )
                return

            nn.Module.__init__(self)
            self.clip_embeddings_dim = clip_embeddings_dim
            self.cross_attention_dim = cross_attention_dim
            self.output_cross_attention_dim = output_cross_attention_dim
            self.clip_extra_context_tokens = clip_extra_context_tokens
            self.is_sdxl = is_sdxl
            self.is_full = is_full
            self.is_plus = is_plus
            self.is_portrait_unnorm = is_portrait_unnorm

            if is_faceid and not is_portrait_unnorm:
                self.image_proj_model = self.init_proj_faceid()
            elif is_full:
                self.image_proj_model = self.init_proj_full()
            elif is_plus or is_portrait_unnorm:
                self.image_proj_model = self.init_proj_plus()
            else:
                self.image_proj_model = self.init_proj()

            quantized, remainder = split_convrot_state_dict(image_proj)
            install_convrot_linears(self.image_proj_model, quantized)
            result = self.image_proj_model.load_state_dict(remainder, strict=False)
            allowed_missing = {base + ".weight" for base in quantized}
            bad_missing = [k for k in result.missing_keys if k not in allowed_missing]
            if bad_missing or result.unexpected_keys:
                raise RuntimeError(
                    "%s IP-Adapter image_proj load mismatch: missing=%s unexpected=%s"
                    % (_LOG, bad_missing, result.unexpected_keys)
                )

            self.ip_layers = patched_to_kv(ipadapter_model["ip_adapter"])

    return IPAdapterConvRotInt8


def patch_ipadapter_plus():
    """Install the ConvRot INT8 patch into the IP-Adapter extension.

    Returns True when the patch is active. Raises when the extension is
    present but cannot be patched, and returns False when it is absent (a
    float pack then still loads through the stock path).
    """
    module = _find_ipadapter_module()
    if module is None:
        return False
    if getattr(module, _MODULE_MARKER, False):
        return True

    try:
        base_to_kv = module.To_KV
        base_ipadapter = module.IPAdapter
        patched_to_kv = _make_to_kv(base_to_kv)
        patched_ipadapter = _make_ipadapter(base_ipadapter, patched_to_kv)
        module.To_KV = patched_to_kv
        module.IPAdapter = patched_ipadapter
    except Exception as exc:
        raise RuntimeError(
            "%s failed to install the ConvRot INT8 patch on the IP-Adapter extension: %s"
            % (_LOG, exc)
        ) from exc

    setattr(module, _MODULE_MARKER, True)
    logger.info("%s patch installed on the IP-Adapter extension", _LOG)
    return True
