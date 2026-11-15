"""ConvRot INT8 execution for the ComfyUI IP-Adapter route.

A ConvRot INT8 IP-Adapter pack stores each measured Linear as int8 codes plus
a per-output-channel ``weight_scale`` and a ``comfy_quant`` sidecar:

    <layer>.weight        int8    [out_features, in_features]
    <layer>.weight_scale  float32 [out_features, 1]
    <layer>.comfy_quant   uint8   JSON {"format":"int8_tensorwise",
                                        "convrot":true,
                                        "convrot_groupsize":256}

The stock ComfyUI IP-Adapter path (``comfyui_ipadapter_plus``) rebuilds every
layer as a plain ``nn.Linear`` and assigns ``weight.data = value``. That is
wrong for an INT8 pack in two ways: the int8 codes would be read as float
weights, and the sidecar keys would not fit a Linear at all.

This module keeps the codes INT8 and evaluates them with the ConvRot INT8
GEMM (``comfy_kitchen.int8_linear(..., convrot=True)``), which applies the
online Hadamard activation rotation inside the kernel. Nothing is
dequantized in the runtime path.

The INT8 payload is installed into the graph after the graph is built, by
``install_convrot_linears`` - the same shape the Forge IP-Adapter runtime
uses (``sd_forge_ipadapter/lib_ipadapter/convrot_int8.py``), reimplemented here
so this package stays self-contained.

Separation:
  - IP-Adapter route only. No ComfyUI core file and no
    ``comfyui_ipadapter_plus`` file is modified; the graph is patched at
    runtime from this package (``patches/ipadapter_convrot_int8.py``).
  - Float packs never enter this code: every entry point returns immediately
    unless the state dict actually carries ConvRot INT8 sidecars.
  - A pack that declares a format other than ``int8_tensorwise``, or declares
    INT8 without ConvRot, is refused instead of being executed as float.
"""
from __future__ import annotations

import json
import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

_WEIGHT = ".weight"
_SCALE = ".weight_scale"
_SIDECAR = ".comfy_quant"
_FORMAT = "int8_tensorwise"
_LOG = "[HSWQ IPAdapter ConvRot INT8]"


def parse_sidecar(value) -> dict:
    """Decode a ``comfy_quant`` sidecar tensor / bytes / str into a dict."""
    if value is None:
        return {}
    try:
        if isinstance(value, torch.Tensor):
            return json.loads(value.numpy().tobytes().decode("utf-8"))
        if isinstance(value, (bytes, bytearray)):
            return json.loads(bytes(value).decode("utf-8"))
        if isinstance(value, str):
            return json.loads(value)
    except Exception:
        return {}
    return {}


def has_convrot_int8(state_dict) -> bool:
    """True when the state dict carries ConvRot INT8 sidecars at all."""
    if not isinstance(state_dict, dict):
        return False
    for key in state_dict:
        if not isinstance(key, str) or not key.endswith(_SCALE):
            continue
        weight = state_dict.get(key[: -len(_SCALE)] + _WEIGHT)
        if isinstance(weight, torch.Tensor) and weight.dtype == torch.int8:
            return True
    return False


def unsupported_quant_formats(state_dict) -> dict:
    """Declared ``comfy_quant`` formats this route cannot execute."""
    found: dict[str, int] = {}
    if not isinstance(state_dict, dict):
        return found
    for key in state_dict:
        if not isinstance(key, str) or not key.endswith(_SIDECAR):
            continue
        conf = parse_sidecar(state_dict[key])
        fmt = conf.get("format") if isinstance(conf, dict) else None
        label = fmt if isinstance(fmt, str) and fmt else "<unreadable>"
        if label == _FORMAT:
            continue
        found[label] = found.get(label, 0) + 1
    return found


class ConvRotInt8Linear(nn.Module):
    """``nn.Linear`` replacement that keeps its weight in INT8 ConvRot form.

    ``int8_weight`` is ``[out_features, in_features]`` (int8) and
    ``weight_scale`` is ``[out_features, 1]`` (float32). ``forward`` runs the
    fused ConvRot INT8 GEMM; the activation rotation happens inside the kernel.
    """

    def __init__(self, in_features, out_features, bias=True, convrot_groupsize=256):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.convrot_groupsize = int(convrot_groupsize)
        self.register_buffer("int8_weight", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("weight_scale", torch.empty(0, dtype=torch.float32), persistent=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_weight(
        cls,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        convrot_groupsize: int = 256,
        bias: torch.Tensor | None = None,
    ):
        out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
        module = cls(
            in_features,
            out_features,
            bias=bias is not None,
            convrot_groupsize=convrot_groupsize,
        )
        module.int8_weight = weight.detach().to(torch.int8).contiguous()
        module.weight_scale = weight_scale.detach().to(torch.float32).reshape(-1, 1).contiguous()
        if bias is not None:
            module.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        return module

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        convrot_groupsize: int = 256,
    ):
        """Build from an existing ``nn.Linear``, keeping its bias."""
        bias = linear.bias.detach() if linear.bias is not None else None
        if bias is not None and weight.dtype == torch.int8:
            bias = bias.to(torch.float32)
        return cls.from_weight(weight, weight_scale, convrot_groupsize, bias)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        # ``Module.to(dtype=...)`` casts float buffers; the scale must stay
        # float32 for the INT8 kernel, so coerce it back.
        if self.weight_scale.dtype != torch.float32:
            self.weight_scale = self.weight_scale.to(torch.float32)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.int8_weight.numel() == 0:
            raise RuntimeError("ConvRotInt8Linear used before its INT8 weight was loaded")
        if x.shape[-1] != self.in_features:
            raise RuntimeError(
                "ConvRotInt8Linear expected %d input features, got %d"
                % (self.in_features, x.shape[-1])
            )
        import comfy_kitchen

        bias = None if self.bias is None else self.bias
        return comfy_kitchen.int8_linear(
            x.contiguous(),
            self.int8_weight,
            self.weight_scale,
            bias,
            out_dtype=x.dtype,
            convrot=True,
            convrot_groupsize=self.convrot_groupsize,
        )


def split_convrot_state_dict(state_dict: dict):
    """Split an IP-Adapter state dict into (quantized layers, float remainder).

    Returns ``({base: {"weight", "weight_scale", "groupsize"}}, remaining)``.
    ``remaining`` keeps biases and every non-quantized tensor, so a normal
    ``load_state_dict`` can consume it unchanged.
    """
    if not has_convrot_int8(state_dict):
        return {}, state_dict

    bases = sorted({k[: -len(_SCALE)] for k in state_dict if isinstance(k, str) and k.endswith(_SCALE)})
    quantized = {}
    for base in bases:
        conf = parse_sidecar(state_dict.get(base + _SIDECAR))
        quant_format = conf.get("format", _FORMAT)
        if quant_format != _FORMAT:
            raise RuntimeError(
                "%s unsupported quantization format %r for layer %s" % (_LOG, quant_format, base)
            )
        if not bool(conf.get("convrot", False)):
            raise RuntimeError(
                "%s layer %s is INT8 without ConvRot; this route only implements ConvRot" % (_LOG, base)
            )
        weight = state_dict.get(base + _WEIGHT)
        if weight is None or weight.dtype != torch.int8:
            raise RuntimeError("%s layer %s has no INT8 weight" % (_LOG, base))
        groupsize = int(conf.get("convrot_groupsize", 256) or 256)
        if weight.dim() != 2 or weight.shape[1] % groupsize != 0:
            raise RuntimeError(
                "%s layer %s input features %s are not divisible by the ConvRot groupsize %d"
                % (_LOG, base, tuple(weight.shape), groupsize)
            )
        quantized[base] = {
            "weight": weight,
            "weight_scale": state_dict[base + _SCALE],
            "groupsize": groupsize,
        }

    remaining = {}
    for key, value in state_dict.items():
        if isinstance(key, str):
            if key.endswith(_SCALE) or key.endswith(_SIDECAR):
                continue
            if key.endswith(_WEIGHT) and key[: -len(_WEIGHT)] in quantized:
                continue
        remaining[key] = value
    return quantized, remaining


def _child(parent: nn.Module, part: str) -> nn.Module:
    if isinstance(parent, (nn.Sequential, nn.ModuleList)) and part.isdigit():
        return parent[int(part)]
    if isinstance(parent, nn.ModuleDict) and part in parent:
        return parent[part]
    return getattr(parent, part)


def _resolve_module(root: nn.Module, path: str):
    """Resolve a dotted path to (parent, key, module).

    Numeric parts index into ``ModuleList`` / ``Sequential`` / ``ModuleDict``,
    which ``nn.Module.get_submodule`` cannot do (it only walks attributes).
    The IP-Adapter image_proj graph uses both (``layers.0.1.1`` is a Linear
    inside a ``FeedForward`` Sequential).
    """
    parent = root
    parts = path.split(".")
    for part in parts[:-1]:
        parent = _child(parent, part)
    last = parts[-1]
    if isinstance(parent, (nn.Sequential, nn.ModuleList)) and last.isdigit():
        index = int(last)
        return parent, index, parent[index]
    if isinstance(parent, nn.ModuleDict) and last in parent:
        return parent, last, parent[last]
    return parent, last, getattr(parent, last)


def install_convrot_linears(root: nn.Module, quantized: dict, prefix_map=None):
    """Replace ``root``'s Linear submodules with ``ConvRotInt8Linear``.

    ``quantized`` is the first element returned by ``split_convrot_state_dict``.
    ``prefix_map`` maps a state-dict base name to the module path inside
    ``root`` (used by the IP-Adapter ``To_KV`` graph, whose keys are flattened
    with ``.`` -> ``_``). Returns the list of replaced layer names.
    """
    replaced = []
    for base, entry in quantized.items():
        path = prefix_map(base) if prefix_map else base
        try:
            parent, key, target = _resolve_module(root, path)
        except (AttributeError, IndexError, KeyError):
            raise RuntimeError(
                "%s layer %s does not exist in %s" % (_LOG, path, type(root).__name__)
            )
        if not isinstance(target, nn.Linear):
            raise RuntimeError(
                "%s layer %s is %s, expected nn.Linear" % (_LOG, path, type(target).__name__)
            )
        module = ConvRotInt8Linear.from_linear(
            target, entry["weight"], entry["weight_scale"], entry["groupsize"]
        )
        if isinstance(parent, (nn.Sequential, nn.ModuleList)):
            parent[key] = module
        else:
            setattr(parent, key, module)
        replaced.append(path)
    return replaced
