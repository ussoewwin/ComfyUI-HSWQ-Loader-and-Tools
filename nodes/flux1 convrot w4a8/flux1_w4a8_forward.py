"""Flux1 ConvRot W4A8 — Linear low-rank residual add (the LoRA path).

Why a residual instead of baking into the packed weight: the pack stores 4-bit
codes (16 levels). A LoRA delta is ~0.1-0.8% of amax, i.e. below half a code
step, so requantising a patched weight rounds the delta away. The ConvRot INT8
sibling states the same mechanism at 8 bits
(``nodes/krea2_convrot_int8/int8_lora_bake.py``). The pack weight therefore
stays untouched and the delta is added after the stock GEMM:

    out = stock_forward(x) + sum_terms( ((x @ dn^T) @ up^T) * scale )

Terms stay factorised (down [r, in], up [out, r]) so no dense [out, in] matrix
is materialised: 228 Flux1 layers would otherwise cost gigabytes.

The residual lives in the ORIGINAL basis. The stock forward already returns
``x W^T`` (ConvRot rotates weight and activation together, cancelling), so no
Hadamard rotation applies to dn/up.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_FORWARD_TAG = "_hswq_w4a8_residual_forward"
_TERMS_ATTR = "_hswq_w4a8_residual_terms"
_KEYS_ATTR = "_hswq_w4a8_residual_keys"

_RESIDUAL_HITS = 0
_COUNTERS = {"terms_set": 0, "modules_cleared": 0}


def reset_w4a8_forward_stats() -> None:
    global _RESIDUAL_HITS
    _RESIDUAL_HITS = 0


def w4a8_forward_stats() -> dict:
    return {
        "residual_hits": _RESIDUAL_HITS,
        "terms_set": _COUNTERS["terms_set"],
        "modules_cleared": _COUNTERS["modules_cleared"],
    }


def set_w4a8_residual_terms(module, terms, key: str = "") -> None:
    """Attach factorised LoRA terms [(down, up, scale, offset|None)] to a module."""
    import torch

    if not terms:
        raise ValueError("[HSWQ W4A8] empty residual term list")
    logical = None
    orig = getattr(module, "_orig_shape", None)
    if orig is not None and len(orig) >= 2:
        logical = (int(orig[0]), int(orig[1]))
    normalised = []
    for down, up, scale, offset in terms:
        if not torch.is_tensor(down) or not torch.is_tensor(up):
            raise TypeError("[HSWQ W4A8] residual factors must be tensors")
        down = down.detach().to(dtype=torch.float32).contiguous()
        up = up.detach().to(dtype=torch.float32).contiguous()
        if down.ndim != 2 or up.ndim != 2 or int(up.shape[1]) != int(down.shape[0]):
            raise ValueError(
                f"[HSWQ W4A8] bad LoRA factors {tuple(up.shape)} x {tuple(down.shape)}"
                f" for {key or '?'}"
            )
        if logical is not None and int(down.shape[1]) != logical[1]:
            raise ValueError(
                f"[HSWQ W4A8] down in_features {int(down.shape[1])} != {logical[1]}"
                f" for {key or '?'}"
            )
        normalised.append((down, up, float(scale), offset))
    setattr(module, _TERMS_ATTR, normalised)
    keys = getattr(module, _KEYS_ATTR, None)
    if keys is None:
        keys = []
        setattr(module, _KEYS_ATTR, keys)
    if key and key not in keys:
        keys.append(key)
    _COUNTERS["terms_set"] += 1


def has_w4a8_residual(module) -> bool:
    return bool(getattr(module, _TERMS_ATTR, None))


def clear_w4a8_residuals(model, keep_keys=None) -> int:
    """Drop residual terms this run no longer wants (LoRA removed / swapped)."""
    keep = set(keep_keys or ())
    cleared = 0
    for _name, mod in model.named_modules():
        keys = getattr(mod, _KEYS_ATTR, None)
        if keys is None:
            continue
        if keep and set(keys) <= keep:
            continue
        if getattr(mod, _TERMS_ATTR, None) is not None:
            delattr(mod, _TERMS_ATTR)
        delattr(mod, _KEYS_ATTR)
        cleared += 1
    _COUNTERS["modules_cleared"] += cleared
    return cleared


def add_w4a8_lora_residual(module, inp, out):
    """Add this module's factorised terms to ``out`` (stock GEMM already done).

    Shared by the wrapped Linear forward and by any path that runs the stock
    GEMM itself (the NVFP4 ConvRot parity forward), so the residual math exists
    in exactly one place.
    """
    global _RESIDUAL_HITS
    terms = getattr(module, _TERMS_ATTR, None)
    if not terms:
        return out

    import torch
    import torch.nn.functional as F

    total = None
    x_f = inp.to(dtype=torch.float32)
    for down, up, scale, offset in terms:
        if down.device != x_f.device:
            down = down.to(x_f.device)
            up = up.to(x_f.device)
        delta = F.linear(F.linear(x_f, down), up) * scale
        if offset is not None:
            start, length = offset
            if int(delta.shape[-1]) != int(length):
                raise RuntimeError(
                    "[HSWQ W4A8] sliced patch width "
                    f"{int(delta.shape[-1])} != {int(length)} on "
                    f"{getattr(module, 'seed_key', '?')}"
                )
            dense = torch.zeros(
                (*delta.shape[:-1], int(out.shape[-1])),
                dtype=delta.dtype,
                device=delta.device,
            )
            dense[..., start:start + length] = delta
            delta = dense
        elif int(delta.shape[-1]) != int(out.shape[-1]):
            raise RuntimeError(
                "[HSWQ W4A8] residual out_features "
                f"{int(delta.shape[-1])} != {int(out.shape[-1])} on "
                f"{getattr(module, 'seed_key', '?')}"
            )
        total = delta if total is None else total + delta
    _RESIDUAL_HITS += 1
    return (out.to(dtype=torch.float32) + total).to(dtype=out.dtype)


def make_w4a8_linear_forward(stock_forward):
    """Return a Linear.forward that adds the factorised residual after stock."""
    if getattr(stock_forward, _FORWARD_TAG, False):
        return stock_forward

    def forward(self, input, *args, **kwargs):
        out = stock_forward(self, input, *args, **kwargs)
        return add_w4a8_lora_residual(self, input, out)

    forward._hswq_w4a8_residual_forward = True  # type: ignore[attr-defined]
    forward._hswq_w4a8_stock_forward = stock_forward  # type: ignore[attr-defined]
    return forward


def attach_w4a8_linear_lora_bake(Lin) -> bool:
    """Wrap the mixed-precision Linear.forward with the residual add (idempotent)."""
    if Lin is None or not hasattr(Lin, "forward"):
        return False
    cur = Lin.forward
    if getattr(cur, _FORWARD_TAG, False):
        return True
    Lin.forward = make_w4a8_linear_forward(cur)
    return True


def peel_w4a8_linear_forward(Lin) -> bool:
    """Restore the stock forward (purge path)."""
    cur = getattr(Lin, "forward", None)
    if cur is None or not getattr(cur, _FORWARD_TAG, False):
        return False
    Lin.forward = cur._hswq_w4a8_stock_forward
    return True


def log_w4a8_lora_bake_evidence(tag: str = "") -> str:
    """One line of measured evidence for the bake pass."""
    stats = w4a8_forward_stats()
    suffix = f" ({tag})" if tag else ""
    msg = (
        f"[HSWQ W4A8 LoRA] EVIDENCE{suffix}: terms_set={stats['terms_set']} "
        f"modules_cleared={stats['modules_cleared']} "
        f"residual_hits={stats['residual_hits']}"
    )
    logger.info(msg)
    print(msg, flush=True)
    return msg
