"""Krea2 ConvRot INT8 Linear forward (INT8-only).

Krea2 ConvRot INT8 layers run the stock MixedPrecision forward (comfy_kitchen
``int8_linear`` with online ConvRot activation rotation). The only Krea2-
specific part is the low-rank LoRA residual: INT8 8-bit requant rounds away
small LoRA deltas (step ~amax/127 vs delta ~0.1-0.8% amax), so the baked
residual is re-added on top of the stock forward output.

The convert_weight/set_weight ConvRot wraps that used to live here only ever
fired for ``_hswq_nvfp4_convrot``-armed layers (the retired Krea2 NVFP4 load
path); for INT8 layers they were pass-through, so they are removed - INT8
behaviour is unchanged (the old condition is False on every INT8 instance).

Never edits ComfyUI-master; installed via monkey-patch on MixedPrecision Linear.
"""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_RESIDUAL_APPLIED = 0
_RESIDUAL_LOGGED_NAMES = set()
_RESIDUAL_LOG_MAX = 6
_RESIDUAL_COUNT_LOGGED = 0


def reset_int8_forward_stats() -> None:
    global _RESIDUAL_APPLIED, _RESIDUAL_LOGGED_NAMES, _RESIDUAL_COUNT_LOGGED
    _RESIDUAL_APPLIED = 0
    _RESIDUAL_LOGGED_NAMES = set()
    _RESIDUAL_COUNT_LOGGED = 0


# Back-compat alias (comfy_quant_int8 imports this name).
reset_int8_lora_log_counters = reset_int8_forward_stats


def int8_forward_stats() -> dict:
    return {
        "residual_applied": _RESIDUAL_APPLIED,
        "residual_layers_logged": len(_RESIDUAL_LOGGED_NAMES),
    }


def _add_krea2_lora_residual(module, inp, out):
    """Add the baked low-rank LoRA residual to a stock-path output.

    Residual terms are stored in the ORIGINAL weight basis (LoRA file basis).
    Layer output is always in the original basis (ConvRot rotation is internal
    and Hadamard-orthogonal), so the add is valid on every path (stock forward).
    Cheap: ~2*rank/out_features extra FLOPs.
    """
    global _RESIDUAL_APPLIED, _RESIDUAL_LOGGED_NAMES, _RESIDUAL_COUNT_LOGGED
    res = getattr(module, "_hswq_krea2_lora_res", None)
    if res is None or out is None:
        return out
    _RESIDUAL_APPLIED += 1
    dev = inp.device
    dt = getattr(inp, "dtype", None)
    if dt is None:
        return out
    cache = getattr(module, "_hswq_krea2_lora_res_gpu", None)
    if cache is None or cache[0] != dev or cache[1] != dt:
        cache = (
            dev,
            dt,
            [
                (
                    md.to(device=dev, dtype=dt),
                    mu.to(device=dev, dtype=dt),
                    sc,
                )
                for md, mu, sc in res
            ],
        )
        module._hswq_krea2_lora_res_gpu = cache
    acc = None
    for md, mu, sc in cache[2]:
        if sc == 0.0:
            continue
        term = torch.matmul(torch.matmul(inp, md.t()), mu.t())
        if sc != 1.0:
            term = term * sc
        acc = term if acc is None else acc + term
    if acc is None:
        return out
    _name = getattr(module, "_hswq_krea2_lora_name", None) or "?"
    if (
        len(_RESIDUAL_LOGGED_NAMES) < _RESIDUAL_LOG_MAX
        and _name not in _RESIDUAL_LOGGED_NAMES
    ):
        _RESIDUAL_LOGGED_NAMES.add(_name)
        try:
            _rn = float(acc.float().norm())
            _on = float(out.float().norm())
            _ratio = (_rn / _on) if _on > 0.0 else -1.0
            _sc = cache[2][0][2] if cache[2] else -1.0
            print(
                f"[HSWQ Krea2 INT8 LoRA] RESIDUAL FORWARD-APPLIED "
                f"name={_name} terms={len(cache[2])} scale={_sc} "
                f"|res|/|out|={_ratio:.6f}",
                flush=True,
            )
        except Exception as _e:
            print(f"[HSWQ Krea2 INT8 LoRA] RESIDUAL log error: {_e!r}", flush=True)
    if _RESIDUAL_APPLIED >= _RESIDUAL_COUNT_LOGGED + 256:
        _RESIDUAL_COUNT_LOGGED = _RESIDUAL_APPLIED
        print(
            f"[HSWQ Krea2 INT8 LoRA] RESIDUAL count={_RESIDUAL_APPLIED} "
            f"(distinct_layers_so_far={len(_RESIDUAL_LOGGED_NAMES)})",
            flush=True,
        )
    if acc.shape != out.shape:
        # rank-safe: ND input produced (..., out); 2D produced (m, out)
        acc = acc.reshape(out.shape)
    return out + acc


def make_int8_linear_forward(stock_forward):
    """Return a Linear.forward replacement for Krea2 ConvRot INT8.

    Every layer runs the stock forward (comfy_kitchen int8_linear + online
    ConvRot rotation); only the baked low-rank LoRA residual is re-added.
    """
    def forward_int8(self, input, *args, **kwargs):
        return _add_krea2_lora_residual(
            self, input, stock_forward(self, input, *args, **kwargs)
        )

    # Shared-convention stamps (read by SDXL / Z Image peel walkers and by
    # patches/hswq_purge_rearm): keep these names and the stack_ver contract.
    forward_int8._hswq_nvfp4_full_forward = True  # type: ignore[attr-defined]
    forward_int8._hswq_krea2_int8 = True  # type: ignore[attr-defined]
    return forward_int8
