"""
Krea2 NVFP4 bench — force ComfyUI MixedPrecision path (ops.py) verbatim.

Package-local under benchmark/krea2_nvfp4/ only. Never import benchmark/nvfp4/.

After apply_comfy_quant_nvfp4_patches():
  1) NVFP4 Linear load → Comfy ops._load_quantized_module (no HSWQ arm / ones(1))
  2) Linear.forward → unwrap to stock Comfy ops.py MixedPrecision Linear.forward
  3) NVFP4 convrot (Hadamard-rotated weights in ckpt):
     stock load drops the comfy_quant stamp, so the load wrapper re-arms
     _hswq_nvfp4_convrot(_groupsize) from the stamp, and the parity forward
     applies the REQUIRED online act rotation (x @ H, per group) right before
     stock MixedPrecision F.linear: (x @ H) @ (W @ H^T)^T == x @ W^T.
     Without this, convrot-stamped ckpts measure as pure garbage (SSIM ~0.04).
     Still ComfyUI-only: stock load + stock Linear.forward (ops.py).
     Kitchen lacked aten.addmm for NVFP4 (bias F.linear → full dequant); that gap
     is filled at runtime by nvfp4_addmm_patch (ck.scaled_mm_nvfp4, TensorCore W4A4).

No invented amax / freeze / ensure_act_scale. Inference + load = ComfyUI only.
"""
from __future__ import annotations

import torch

_APPLIED = False


def _closure_named(fn, name: str):
    if fn is None or fn.__closure__ is None:
        return None
    for n, cell in zip(fn.__code__.co_freevars, fn.__closure__):
        if n == name:
            return cell.cell_contents
    return None


def _unwrap_stock_forward(fwd):
    """Extract closed-over stock_forward from make_nvfp4_linear_forward wrap."""
    if not getattr(fwd, "_hswq_nvfp4_full_forward", False):
        return None
    return _closure_named(fwd, "stock_forward")


def _make_convrot_parity_forward(stock_forward):
    """Stock MixedPrecision forward + required online act rotation for convrot ckpts.

    The HSWQ TC forward (disabled here for parity) is the only other place this
    rotation exists. ConvRot weights are stored pre-rotated (W @ H^T); the math
    only closes if activations are rotated too: (x @ H) @ (W @ H^T)^T == x @ W^T.
    Modules without the armed flag pass through untouched (bit-exact stock).
    """
    from .nvfp4_hadamard import rotate_last_dim_fast

    _LOGGED_FIRST_ROT = [False]

    def forward_convrot_parity(self, input, *args, **kwargs):
        if getattr(self, "_hswq_nvfp4_convrot", False):
            gs = int(getattr(self, "_hswq_nvfp4_convrot_groupsize", 256) or 256)
            if not _LOGGED_FIRST_ROT[0]:
                _LOGGED_FIRST_ROT[0] = True
                print(
                    f"  [CONVROT parity forward] First act-rotation triggered: "
                    f"fast O(N log N) float32 butterfly transform "
                    f"(shape={tuple(input.shape)}, dtype={input.dtype}, groupsize={gs}, zero accumulation error)",
                    flush=True,
                )
            input = rotate_last_dim_fast(input, gs)
        return stock_forward(self, input, *args, **kwargs)

    forward_convrot_parity._hswq_nvfp4_convrot_parity = True  # type: ignore[attr-defined]
    return forward_convrot_parity


def apply_nvfp4_comfy_parity() -> bool:
    """Runtime only. Imports stay inside krea2_nvfp4 (never benchmark/nvfp4/)."""
    global _APPLIED
    import logging
    logging.getLogger("comfy_kitchen.tensor.nvfp4").setLevel(logging.ERROR)
    # Stock F.linear(bias=...) → aten.addmm; kitchen NVFP4 had no handler → dequant.
    from .nvfp4_addmm_patch import register_nvfp4_addmm_handler

    register_nvfp4_addmm_handler()

    if _APPLIED:
        return True

    import comfy.ops as ops

    # --- Load: use Comfy ops._load_quantized_module for nvfp4 (no ones(1) arm) ---
    patched_load = ops._load_quantized_module
    orig_load = _closure_named(patched_load, "_orig_load")
    if orig_load is None:
        raise RuntimeError(
            "[BENCH] nvfp4 Comfy parity: could not recover Comfy _orig_load from patch closure"
        )

    def _load_quantized_module_comfy_only(
        module,
        super_load,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
        load_extra_params=False,
    ):
        from .nvfp4_conf import (
            convrot_flags_from_conf,
            is_nvfp4_conf,
        )
        from .nvfp4_load import peek_nvfp4_conf

        # Peek robustly before orig_load pops the stamp.
        conf = peek_nvfp4_conf(state_dict, prefix)
        # Always Comfy stock load — including nvfp4 (input_scale only if in ckpt)
        out = orig_load(
            module,
            super_load,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
            load_extra_params=load_extra_params,
        )
        # Stock load drops the stamp; re-arm convrot flags so the parity forward
        # applies the online act rotation the offline-rotated weights require.
        if is_nvfp4_conf(conf):
            module._hswq_nvfp4 = True
            enabled, gs = convrot_flags_from_conf(conf)
            module._hswq_nvfp4_convrot = bool(enabled)
            module._hswq_nvfp4_convrot_groupsize = int(gs)
        return out

    _load_quantized_module_comfy_only._hswq_nvfp4_full_load = True  # type: ignore[attr-defined]
    ops._load_quantized_module = _load_quantized_module_comfy_only

    # --- Forward: unwrap HSWQ TC wrap → Comfy MixedPrecision Linear.forward ---
    _cur_mp = ops.mixed_precision_ops

    def mixed_precision_ops_comfy_only(*args, **kwargs):
        mp = _cur_mp(*args, **kwargs)
        Lin = mp.Linear
        stock = _unwrap_stock_forward(Lin.forward)
        if stock is None and getattr(Lin.forward, "_hswq_nvfp4_full_forward", False):
            raise RuntimeError(
                "[BENCH] nvfp4 Comfy parity: HSWQ TC wrap still on Linear.forward; "
                "refusing to leave non-Comfy forward (SSIM target >=0.9)"
            )
        if stock is not None:
            Lin.forward = _make_convrot_parity_forward(stock)
        return mp

    ops.mixed_precision_ops = mixed_precision_ops_comfy_only

    # Prove unwrap works once at install time (no silent leave-TC-on)
    mp0 = _cur_mp()
    stock0 = _unwrap_stock_forward(mp0.Linear.forward)
    if stock0 is None and getattr(mp0.Linear.forward, "_hswq_nvfp4_full_forward", False):
        raise RuntimeError(
            "[BENCH] nvfp4 Comfy parity: failed to unwrap Linear.forward to Comfy stock"
        )
    if stock0 is not None:
        mp0.Linear.forward = _make_convrot_parity_forward(stock0)

    _APPLIED = True
    print(
        "[BENCH] nvfp4 ComfyUI-only: load=_load_quantized_module; "
        "Linear.forward=ops.py stock + O(N log N) float32 butterfly act-rotate (zero accumulation error); "
        "NVFP4 addmm->ck.scaled_mm_nvfp4 registered; "
        "SSIM target >=0.9",
        flush=True,
    )
    return True
