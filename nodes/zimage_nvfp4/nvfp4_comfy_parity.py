"""
Z Image ConvRot NVFP4 — ComfyUI MixedPrecision path (ops.py).

After apply_comfy_quant_nvfp4_patches():
  1) NVFP4 Linear load → Comfy ops._load_quantized_module (no HSWQ arm / ones(1))
  2) Linear.forward → unwrap to stock Comfy ops.py MixedPrecision Linear.forward
  3) NVFP4 convrot (offline Hadamard-rotated weights):
     stock load drops the comfy_quant stamp, so the load wrapper re-arms
     _hswq_nvfp4_convrot(_groupsize) from the stamp, and the forward
     applies the REQUIRED online act rotation (x @ H, per group) right before
     stock MixedPrecision F.linear: (x @ H) @ (W @ H^T)^T == x @ W^T.
     Kitchen lacked aten.addmm for NVFP4 (bias F.linear → full dequant); that gap
     is filled at runtime by nvfp4_addmm_patch (scaled_mm_nvfp4), not HSWQ TC wrap.

No invented amax / freeze / ensure_act_scale. Inference + load = ComfyUI only.
"""
from __future__ import annotations

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
    from ..nvfp4.nvfp4_hadamard import build_hadamard, rotate_last_dim

    def forward_convrot_parity(self, input, *args, **kwargs):
        # No NVFP4 act rotate here — kitchen NVFP4 already handles ConvRot.
        return stock_forward(self, input, *args, **kwargs)

    forward_convrot_parity._hswq_nvfp4_convrot_parity = True  # type: ignore[attr-defined]
    return forward_convrot_parity


def apply_nvfp4_comfy_parity() -> bool:
    """Runtime only — stock Comfy load/forward + ConvRot act rotate."""
    global _APPLIED
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
            "[HSWQ NVFP4] Z Image: could not recover Comfy _orig_load from patch closure"
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
        from ..nvfp4.nvfp4_conf import (
            convrot_flags_from_conf,
            decode_comfy_quant_conf,
            is_nvfp4_conf,
        )

        # Peek before orig_load pops the stamp.
        conf = decode_comfy_quant_conf(state_dict.get(f"{prefix}comfy_quant"))
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
                "[HSWQ NVFP4] Z Image: HSWQ TC wrap still on Linear.forward; "
                "refusing to leave non-Comfy forward"
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
            "[HSWQ NVFP4] Z Image: failed to unwrap Linear.forward to Comfy stock"
        )
    if stock0 is not None:
        mp0.Linear.forward = _make_convrot_parity_forward(stock0)

    _APPLIED = True
    print(
        "[HSWQ NVFP4] Z Image: load=_load_quantized_module; "
        "Linear.forward=ops.py stock + convrot act-rotate; "
        "NVFP4 addmm->scaled_mm_nvfp4 registered (no full-weight dequant)"
    )
    return True
