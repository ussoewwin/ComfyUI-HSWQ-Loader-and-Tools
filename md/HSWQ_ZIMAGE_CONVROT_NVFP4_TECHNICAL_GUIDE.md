# HSWQ Z Image ConvRot NVFP4 — Technical Implementation Manual

Date: 2026-08-01  
Repository: `ussoewwin/ComfyUI-HSWQ-Loader-and-Tools`  
Diff baseline (exclusive): `a9d372089c2314bcfa9a1d314a3bf81f0dfde9fb` (`docs:point-zhmd-changelog-v3.3.0-link-to-zh`)  
HEAD at writing: `b8d1144` — `fix: ZI NVFP4 LoRA bake v4 - prefer _layout_cls over Tensor.layout`  
Scope: **Z Image / ZIT UNet ConvRot NVFP4** under `nodes/zimage_nvfp4`. This line does **not** rewrite the SDXL Tensor Core product stack under `nodes/nvfp4`.

This document is organized as:

1. **Summary** — what was implemented and why
2. **Files created or modified**
3. **Full source** of those files (as on disk at HEAD)
4. **Technical meaning** — per-module / per-hook behavior

Style matches other public manuals under `md/` (for example `HSWQ_INT8_AND_LORA_TECHNICAL_GUIDE.md`).

---

## 1. Summary

### 1.1 Goals

Ship a **Z Image / ZIT** load path for HSWQ **ConvRot NVFP4** UNet packs that:

| Pillar | Description |
|--------|-------------|
| **A. Comfy parity forward** | Offline ConvRot weights (`W` rotated offline) require **online act rotate** on stock MixedPrecision GEMM. Product HSWQ Tensor Core `Linear.forward` (`_hswq_nvfp4_full_forward`) destroys Pixel SSIM on Z Image; the bench path (`hswq/benchmark`) does not. |
| **B. Mixed pack** | Typical pack = about 120 NVFP4 ConvRot Linear + about 60 INT8 protect Linear. Arm both; INT8 ConvRot stays ComfyUI core / kitchen. |
| **C. Dynamic VRAM LoRA bake** | MultiGPU `ModelPatcherDynamic` attaches `LowVramPatch`. Bake must run **after** Dynamic.load (outermost wrap), cover NVFP4 ConvRot **and** leftover INT8 QT, and leave `patches_left=0`. |
| **D. Isolation** | All Z Image logic lives under `nodes/zimage_nvfp4`. Do not edit `nodes/nvfp4` for this feature. INT8 Dynamic bake must not treat NVFP4 QT as INT8. |

### 1.2 Why SDXL TC path is wrong for Z Image

SDXL Checkpoint Loader uses `nodes/nvfp4` Tensor Core / `scaled_mm_nvfp4` product forward.  
Z Image ConvRot NVFP4 packs are validated against **comfy_parity** (stock `F.linear` / GEMM + online act rotate).  
Installing TC full-forward on Z Image fails the bench guard `require_convrot_parity_forward` and collapses SSIM.

### 1.3 User-facing entry

| Surface | Behavior |
|---------|----------|
| **HSWQ ConvRot INT8/ConvRot NVFP4 UNet Loader** | `weight_dtype` = `ConvRot NVFP4`, or `default` when safetensors looks like comfy_quant NVFP4. |
| **Dispatch order** | NVFP4 UNet dispatch installs **after** INT8 so mixed packs are not stolen by INT8-only auto-detect. |
| **Compatibility** | **Only** UNet packs quantized with [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization). |

### 1.4 LoRA bake v4 (critical fix)

**Symptom:** Dynamic.load ENTER fired with `patches=180` but `nvfp4_convrot=False` then no bake, so LoRA looked ineffective / LowVramPatch stayed attached.

**Root cause:** Kitchen `QuantizedTensor` inherits `torch.Tensor.layout` (`torch.strided`). Old `_qt_layout_name` read `qt.layout` first; `type(layout).__name__` was literally `"layout"`, so `_qt_is_nvfp4` was always False. v3 also required QT on `module.weight` while Dynamic VRAM often hides QT behind `get_key_weight`.

**Fix (`b8d1144`, bake hook v4):**

- Prefer `_layout_cls` / `layout_cls` over `Tensor.layout`
- Gate bake on ConvRot **module flag** `_hswq_nvfp4_convrot` (do not require QT on `module.weight`)
- ENTER logs: `flagged=` / `qt_on_weight=`
- After NVFP4 bake: leftover INT8/QT bake until `patches_left=0`
- Outermost `ModelPatcherDynamic.load` wrap + `load_models_gpu` bake hook; re-arm if INT8 / MultiGPU overwrites

### 1.5 Successful bake contract (owner A/B)

```text
Dynamic.load bake hook ON v4
ENTER #1: patches=180 nvfp4_convrot=True flagged=120 qt_on_weight=120
Dynamic.load bake #1: nvfp4_baked=120 int8_baked=60 ... skip_not_convrot=60 patches_left=0
[HSWQ LoRA Bake] total=180 requant=180 ... path OK (all requant)
```

`skip_not_convrot` samples for INT8 protect keys are expected: those keys are baked on the leftover INT8/QT path, not the NVFP4 ConvRot path.

### 1.6 Commit chain from baseline

```text
a9d3720  (baseline — no nodes/zimage_nvfp4)
...      feat / parity / product path commits
a72272d  bake ConvRot NVFP4 LoRA after ModelPatcherDynamic.load
1a4be78  bake leftover INT8 QT LoRA after NVFP4
90958c8  re-arm bake wrap when Dynamic.load overwritten
ea37fae  bake v3 ENTER / patches-key / load_models_gpu
b8d1144  bake v4 _layout_cls + flag gate (HEAD)
```

---

## 2. Files created or modified

Relative to `a9d372089c2314bcfa9a1d314a3bf81f0dfde9fb` → `HEAD`:

| Status | Path | Role |
|--------|------|------|
| **A** | `nodes/zimage_nvfp4/__init__.py` | Package exports |
| **A** | `nodes/zimage_nvfp4/load_unet.py` | UNet load + dispatch + force outermost bake wrap |
| **A** | `nodes/zimage_nvfp4/nvfp4_comfy_parity.py` | Stock GEMM + online act rotate; arm NVFP4 + INT8 protect |
| **A** | `nodes/zimage_nvfp4/nvfp4_lora_bake.py` | Dynamic.load / load_models_gpu LoRA bake v4 |
| **A** | `nodes/zimage_nvfp4/nvfp4_addmm_patch.py` | Register `aten.addmm` for `TensorCoreNVFP4Layout` |
| **A** | `nodes/zimage_nvfp4/nvfp4_tc_gate.py` | CC / CUBLAS gate; mute TC spam on non-Blackwell |
| **A** | `nodes/zimage_nvfp4/require_parity.py` | Fail-closed TC vs parity forward check |
| **A** | `prestartup_script.py` | Early wire without shadowing ComfyUI `nodes` |
| **M** | `patches/comfy_quant_int8.py` | INT8 detect/bake: `int8_tensorwise` only (never NVFP4); Dynamic bake ver 6 |
| **M** | `hswq/zimage_fp8_e4m3_unet.py` | Add `ConvRot NVFP4` dtype; rename loader title |
| **M** | `nodes/models/zimage_fp8_e4m3_unet.py` | Loader title sync |
| **M** | `README.md` | Document Z Image ConvRot NVFP4 UNet path |

Not in this delta as rewrite targets: `nodes/nvfp4/**` (SDXL TC remains separate).

---

## 3. Full source

Sources below are the **full file contents at HEAD** for every **Added** path and the **modified** INT8 patch file. For large pre-existing modules that only changed a few lines (`hswq/zimage_fp8_e4m3_unet.py`, `nodes/models/zimage_fp8_e4m3_unet.py`, `README.md`), the exact unified diffs from the baseline are included instead of reprinting thousand-line parents.

### 3.1 Added package `nodes/zimage_nvfp4/`


### `nodes/zimage_nvfp4/__init__.py`

```python
"""Z Image ConvRot NVFP4 — comfy_parity (stock GEMM + act rotate); INT8 = core."""

from .load_unet import (
    apply_nvfp4_patches,
    install_zimage_nvfp4_unet_dispatch,
    load_unet_nvfp4_weight_dtype,
)
from .nvfp4_lora_bake import install_zimage_nvfp4_lora_bake

__all__ = [
    "apply_nvfp4_patches",
    "install_zimage_nvfp4_unet_dispatch",
    "install_zimage_nvfp4_lora_bake",
    "load_unet_nvfp4_weight_dtype",
]
```


### `nodes/zimage_nvfp4/require_parity.py`

```python
"""Fail closed if ConvRot act-rotate forward is not armed."""


def require_convrot_parity_forward() -> None:
    """Fail if Linear.forward is not the ConvRot act-rotate wrapper."""
    import comfy.ops

    lin_fwd = comfy.ops.mixed_precision_ops().Linear.forward
    if getattr(lin_fwd, "_hswq_nvfp4_full_forward", False):
        raise RuntimeError(
            "Z Image ConvRot NVFP4: Linear.forward still has HSWQ TC wrap "
            "(_hswq_nvfp4_full_forward); quality would be destroyed"
        )
    if not getattr(lin_fwd, "_hswq_nvfp4_convrot_parity", False):
        raise RuntimeError(
            "Z Image ConvRot NVFP4: Linear.forward missing "
            "_hswq_nvfp4_convrot_parity (online act rotation required for "
            "offline W@H^T weights)"
        )
```


### `nodes/zimage_nvfp4/nvfp4_tc_gate.py`

```python
"""NVFP4 TensorCore availability gate (shared by addmm patch + TC forward).

cuBLAS NVFP4 GEMM needs compute capability >= 10.0 (Blackwell). Cloud hosts are
often Ada / Hopper / Ampere — every ``scaled_mm_nvfp4`` then raises
``CUBLAS_STATUS_NOT_SUPPORTED`` and kitchen / addmm log WARNING per Linear.

This module:
  1) probes CC once
  2) after first NOT_SUPPORTED (or CC < 10.0), disables further TC attempts
  3) emits a single clear line; mutes kitchen nvfp4 WARNING spam
"""
from __future__ import annotations

import logging

_PROBED = False
_TC_OK: bool | None = None
_DISABLED = False
_WARNED = False
_DISABLE_REASON = ""

_KITCHEN_NVFP4_LOG = "comfy_kitchen.tensor.nvfp4"
_ADDMM_LOG = "nvfp4.nvfp4_addmm_patch"
_FORWARD_LOG = "nvfp4.nvfp4_forward"


def _mute_nvfp4_warning_spam() -> None:
    for name in (_KITCHEN_NVFP4_LOG, _ADDMM_LOG, _FORWARD_LOG):
        logging.getLogger(name).setLevel(logging.ERROR)


def probe_nvfp4_tc_support(device_index: int = 0) -> bool:
    """Return True if GPU CC looks NVFP4-TC capable (kitchen min is (10, 0))."""
    global _PROBED, _TC_OK
    if _PROBED and _TC_OK is not None:
        return bool(_TC_OK)
    _PROBED = True
    try:
        import torch

        if not torch.cuda.is_available():
            _TC_OK = False
            return False
        major, minor = torch.cuda.get_device_capability(device_index)
        # comfy_kitchen CUDA scaled_mm_nvfp4: min_compute_capability=(10, 0)
        _TC_OK = (int(major), int(minor)) >= (10, 0)
        return bool(_TC_OK)
    except Exception:
        _TC_OK = False
        return False


def nvfp4_tc_enabled() -> bool:
    if _DISABLED:
        return False
    return probe_nvfp4_tc_support()


def disable_nvfp4_tc(reason: str, *, announce: bool = True) -> None:
    """Permanent disable for this process; warn once then mute spam loggers."""
    global _DISABLED, _WARNED, _DISABLE_REASON
    _DISABLED = True
    _DISABLE_REASON = str(reason) if reason else "unknown"
    _mute_nvfp4_warning_spam()
    if announce and not _WARNED:
        _WARNED = True
        name = "?"
        cc = "?"
        try:
            import torch

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                major, minor = torch.cuda.get_device_capability(0)
                cc = f"{major}.{minor}"
        except Exception:
            pass
        print(
            f"[HSWQ NVFP4] TensorCore scaled_mm disabled for this run "
            f"(GPU={name}, CC={cc}): {_DISABLE_REASON}. "
            f"Using dequant mm; further CUBLAS/kitchen WARNINGs suppressed.",
            flush=True,
        )


def note_scaled_mm_failure(exc: BaseException) -> bool:
    """If failure is permanent (NOT_SUPPORTED / unsupported), disable TC.

    Returns True if TC is now disabled (caller should dequant without retry storm).
    """
    msg = str(exc)
    permanent = (
        "CUBLAS_STATUS_NOT_SUPPORTED" in msg
        or "NOT_SUPPORTED" in msg
        or "not supported" in msg.lower()
    )
    if permanent:
        disable_nvfp4_tc(msg.split("\n", 1)[0][:240])
        return True
    return _DISABLED


def announce_tc_status_at_register() -> None:
    """One-line status when addmm / full stack is registered (cloud-visible)."""
    ok = probe_nvfp4_tc_support()
    try:
        import torch

        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            cc = f"{major}.{minor}"
        else:
            cc = "n/a"
    except Exception:
        name, cc = "?", "?"
    if ok:
        print(
            f"[HSWQ NVFP4] TC probe: GPU={name} CC={cc} — "
            f"scaled_mm_nvfp4 enabled (min CC 10.0)",
            flush=True,
        )
    else:
        disable_nvfp4_tc(
            f"compute capability {cc} < 10.0 (NVFP4 TensorCore requires Blackwell+)",
            announce=True,
        )
```


### `nodes/zimage_nvfp4/nvfp4_addmm_patch.py`

```python
"""Fill kitchen NVFP4 gap: register aten.addmm for TensorCoreNVFP4Layout.

comfy_kitchen registers addmm for INT8 / MXFP8 / FP8 / SVDQuant / ConvRotW4A4,
but NOT for TensorCoreNVFP4Layout. PyTorch F.linear(bias=...) often decomposes
to aten.addmm.default → unhandled → full dequantize of both operands.

That is why stock MixedPrecision Linear (Comfy ops.py) can look "NVFP4 loaded"
(uint8 packed weights in state_dict) while peak VRAM exceeds FP16: packed
storage stays resident AND dequant materializes FP16 weights every forward.

Runtime-only registration — does not edit ComfyUI-master or site-packages files.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
_REGISTERED = False


def register_nvfp4_addmm_handler() -> bool:
    """Register aten.addmm.default → scaled_mm_nvfp4 (same contract as MXFP8 addmm)."""
    global _REGISTERED
    if _REGISTERED:
        return True

    try:
        import torch
        import comfy_kitchen as ck
        from comfy_kitchen.tensor.base import (
            QuantizedTensor,
            dequantize_args,
            register_layout_op,
            _LAYOUT_DISPATCH_TABLE,
        )
        from comfy_kitchen.tensor.nvfp4 import (
            TensorCoreNVFP4Layout,
            _slice_to_original_shape,
        )
        from .nvfp4_tc_gate import (
            announce_tc_status_at_register,
            note_scaled_mm_failure,
            nvfp4_tc_enabled,
        )
    except Exception as e:
        logger.warning("[HSWQ NVFP4] addmm register skipped (import): %s", e)
        return False

    announce_tc_status_at_register()

    # Already present in a newer kitchen — do not double-register.
    op = torch.ops.aten.addmm.default
    table = _LAYOUT_DISPATCH_TABLE.get(op, {})
    if TensorCoreNVFP4Layout in table:
        _REGISTERED = True
        logger.info("[HSWQ NVFP4] aten.addmm already registered for NVFP4")
        return True

    @register_layout_op(op, TensorCoreNVFP4Layout)
    def _handle_nvfp4_addmm(qt, args, kwargs):
        """NVFP4 addmm: bias + input @ weight.T (F.linear with bias decomposition)."""
        bias, mat1, mat2 = args[0], args[1], args[2]

        if not (isinstance(mat1, QuantizedTensor) and isinstance(mat2, QuantizedTensor)):
            return torch.addmm(*dequantize_args((bias, mat1, mat2)))
        if mat1._qdata.dim() != 2:
            return torch.addmm(*dequantize_args((bias, mat1, mat2)))

        input_transposed = getattr(mat1._params, "transposed", False)
        weight_transposed = getattr(mat2._params, "transposed", False)
        # F.linear → addmm(bias, x, w.t()): weight must be logically transposed.
        if input_transposed or not weight_transposed:
            logger.debug(
                "NVFP4 addmm: unsupported transpose configuration, falling back to dequantize"
            )
            return torch.addmm(*dequantize_args((bias, mat1, mat2)))

        # Cloud Ada/Hopper etc.: skip scaled_mm after first CUBLAS NOT_SUPPORTED
        # (otherwise WARNING floods every Linear every step).
        if not nvfp4_tc_enabled():
            return torch.addmm(*dequantize_args((bias, mat1, mat2)))

        input_qdata, scale_a, block_scale_a = TensorCoreNVFP4Layout.get_plain_tensors(mat1)
        weight_qdata, scale_b, block_scale_b = TensorCoreNVFP4Layout.get_plain_tensors(mat2)
        out_dtype = kwargs.get("out_dtype", mat1._params.orig_dtype)

        try:
            result = ck.scaled_mm_nvfp4(
                input_qdata,
                weight_qdata,
                tensor_scale_a=scale_a,
                tensor_scale_b=scale_b,
                block_scale_a=block_scale_a,
                block_scale_b=block_scale_b,
                bias=bias,
                out_dtype=out_dtype,
            )
            orig_m = mat1._params.orig_shape[0]
            orig_n = mat2._params.orig_shape[1]
            return _slice_to_original_shape(result, orig_m, orig_n)
        except (RuntimeError, TypeError) as e:
            note_scaled_mm_failure(e)
            return torch.addmm(*dequantize_args((bias, mat1, mat2)))

    _REGISTERED = True
    print(
        "[HSWQ NVFP4] registered aten.addmm.default for TensorCoreNVFP4Layout "
        "(stock F.linear+bias -> scaled_mm_nvfp4; was dequant-only)",
        flush=True,
    )
    return True
```


### `nodes/zimage_nvfp4/load_unet.py`

```python
"""Z Image / ZIT UNet load — ConvRot NVFP4 (parity) + INT8 ConvRot (ComfyUI core).

Z Image ConvRot NVFP4 is **not** the SDXL TC Linear.forward path.
``hswq/benchmark/zi_convrot_nvfp4_bench.py`` ``require_convrot_parity_forward``:
TC wrap (``_hswq_nvfp4_full_forward``) destroys SSIM; need stock GEMM + online
act rotate (``_hswq_nvfp4_convrot_parity``) via ``apply_nvfp4_comfy_parity``.

  - Arm SDXL detect/load/LoRA bake with ``apply_comfy_quant_nvfp4_patches``, then
    **replace** Linear.forward with comfy_parity (not stacked double-rotate).
  - INT8 ConvRot: ComfyUI core / kitchen as-is. ``apply_comfy_quant_int8_patches``
    only for int8_tensorwise load.

All logic under ``nodes/zimage_nvfp4``. Does not edit ``nodes/nvfp4`` (SDXL TC).
"""
from __future__ import annotations

import logging
import sys

NVFP4_WEIGHT_DTYPE = "ConvRot NVFP4"

_DISPATCH_INSTALLED = False
_INSTALL_HOOKED = False

logger = logging.getLogger(__name__)


def apply_nvfp4_patches() -> None:
    """Arm Z Image ConvRot NVFP4 (parity) + INT8 load (core ConvRot)."""
    from ..nvfp4.comfy_quant_nvfp4 import apply_comfy_quant_nvfp4_patches
    from ...patches.comfy_quant_int8 import apply_comfy_quant_int8_patches
    from .nvfp4_comfy_parity import (
        apply_nvfp4_comfy_parity,
        require_convrot_parity_forward,
    )
    from .nvfp4_lora_bake import install_zimage_nvfp4_lora_bake

    if not apply_comfy_quant_nvfp4_patches():
        raise RuntimeError(
            "[HSWQ NVFP4] Z Image: apply_comfy_quant_nvfp4_patches failed "
            "(detect/load/LoRA bake required; see nodes/nvfp4)"
        )
    # Replace TC Linear.forward with stock GEMM + act rotate (not double-rotate).
    if not apply_nvfp4_comfy_parity():
        raise RuntimeError(
            "[HSWQ NVFP4] Z Image: apply_nvfp4_comfy_parity failed "
            "(stock GEMM + act rotate required; TC destroys SSIM)"
        )
    require_convrot_parity_forward()
    # INT8 tensorwise load only — ConvRot INT8 remains ComfyUI core / kitchen.
    apply_comfy_quant_int8_patches()
    # After INT8 Dynamic bake wrap: force ConvRot NVFP4 LoRA bake outermost.
    if not install_zimage_nvfp4_lora_bake(force=True):
        raise RuntimeError(
            "[HSWQ NVFP4] Z Image: install_zimage_nvfp4_lora_bake failed "
            "(Dynamic ConvRot NVFP4 LoRA bake required)"
        )
    print(
        "  [HSWQ NVFP4] Z Image: ConvRot NVFP4 (comfy_parity) + INT8 ConvRot "
        "+ Dynamic NVFP4 LoRA bake",
        flush=True,
    )


def _ensure_dynamic_load_bake_wrap() -> None:
    """Re-arm ZI NVFP4 bake wrap if MultiGPU/INT8 overwrote Dynamic.load or load_models_gpu."""
    from .nvfp4_lora_bake import (
        _BAKE_HOOK_VER,
        install_load_models_gpu_bake_hook,
        install_zimage_nvfp4_lora_bake,
    )

    try:
        import comfy.model_management as mm
        import comfy.model_patcher as mp
    except ImportError:
        return
    Dynamic = getattr(mp, "ModelPatcherDynamic", None)
    need_force = True
    if Dynamic is not None:
        cur = getattr(Dynamic, "load", None)
        if (
            cur is not None
            and getattr(cur, "_hswq_zi_nvfp4_lora_bake", False)
            and getattr(cur, "_hswq_zi_nvfp4_lora_bake_ver", 0) >= _BAKE_HOOK_VER
        ):
            need_force = False
    if need_force:
        install_zimage_nvfp4_lora_bake(force=True)
    gpu = getattr(mm, "load_models_gpu", None)
    if (
        gpu is None
        or not getattr(gpu, "_hswq_zi_nvfp4_gpu_bake", False)
        or getattr(gpu, "_hswq_zi_nvfp4_gpu_bake_ver", 0) < _BAKE_HOOK_VER
    ):
        install_load_models_gpu_bake_hook(force=True)
    else:
        install_load_models_gpu_bake_hook(force=False)


def load_unet_nvfp4_weight_dtype(unet_name, weight_dtype):
    """Load Z Image / ZIT UNet with ConvRot NVFP4 parity (not SDXL TC forward)."""
    import folder_paths
    import comfy.sd

    from ..nvfp4.comfy_quant_nvfp4 import (
        apply_comfy_quant_nvfp4_patches,
        reset_nvfp4_lora_log_counters,
    )
    from ...patches.comfy_quant_int8 import (
        _int8_quant_conv_scope,
        apply_comfy_quant_int8_patches,
        reset_int8_lora_log_counters,
        summarize_int8_lora_capability,
    )
    from .nvfp4_comfy_parity import (
        apply_nvfp4_comfy_parity,
        require_convrot_parity_forward,
    )
    from .nvfp4_lora_bake import (
        install_zimage_nvfp4_lora_bake,
        reset_zimage_nvfp4_lora_bake_log_counters,
    )

    unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
    if not apply_comfy_quant_nvfp4_patches():
        raise RuntimeError(
            "[HSWQ NVFP4] Z Image UNet requires NVFP4 detect/load/LoRA bake "
            "(apply_comfy_quant_nvfp4_patches)"
        )
    if not apply_nvfp4_comfy_parity():
        raise RuntimeError(
            "[HSWQ NVFP4] Z Image UNet requires comfy_parity "
            "(stock GEMM + act rotate; not HSWQ TC Linear.forward)"
        )
    require_convrot_parity_forward()
    # Mixed pack: Linear=nvfp4 parity, INT8 = ComfyUI core ConvRot path.
    apply_comfy_quant_int8_patches()
    if not install_zimage_nvfp4_lora_bake(force=True):
        raise RuntimeError(
            "[HSWQ NVFP4] Z Image UNet requires Dynamic ConvRot NVFP4 LoRA bake"
        )
    _ensure_dynamic_load_bake_wrap()
    reset_int8_lora_log_counters()
    reset_nvfp4_lora_log_counters()
    reset_zimage_nvfp4_lora_bake_log_counters()
    logging.info(
        "[HSWQ NVFP4] Loading UNet (ConvRot NVFP4 comfy_parity + INT8 ConvRot ComfyUI core): "
        "%s (weight_dtype=%s)",
        unet_name,
        weight_dtype,
    )
    print(
        f"[HSWQ NVFP4] Loading UNet (ConvRot NVFP4 / comfy_parity): {unet_name}",
        flush=True,
    )
    with _int8_quant_conv_scope():
        model = comfy.sd.load_diffusion_model(unet_path, model_options={})
    summarize_int8_lora_capability(model)
    return (model,)


def _attach_to_comfy_quant_module() -> None:
    """Expose this loader on comfy_quant_nvfp4 so prestartup can bind it."""
    for name, mod in list(sys.modules.items()):
        if not (
            name.endswith("nodes.nvfp4.comfy_quant_nvfp4")
            or name.endswith(".comfy_quant_nvfp4")
            or name == "comfy_quant_nvfp4"
        ):
            continue
        cur = getattr(mod, "load_unet_nvfp4_weight_dtype", None)
        if cur is None or cur is load_unet_nvfp4_weight_dtype:
            mod.load_unet_nvfp4_weight_dtype = load_unet_nvfp4_weight_dtype


def install_zimage_nvfp4_unet_dispatch(node_class_mappings=None) -> bool:
    """Wrap HSWQFP8E4M3UNetLoader for weight_dtype ConvRot NVFP4.

    Must run *after* ``install_int8_option_dispatch``: mixed NVFP4 packs also
    contain ``int8_tensorwise`` layers, so INT8-only auto-detect would otherwise
    steal the load without NVFP4 Linear patches. INT8 ConvRot stays core.
    """
    global _DISPATCH_INSTALLED
    if node_class_mappings is None:
        wrapped_any = False
        for _n, mod in list(sys.modules.items()):
            mappings = getattr(mod, "NODE_CLASS_MAPPINGS", None)
            if isinstance(mappings, dict) and install_zimage_nvfp4_unet_dispatch(mappings):
                wrapped_any = True
        return wrapped_any

    if not isinstance(node_class_mappings, dict):
        return False

    from ..nvfp4.nvfp4_conf import checkpoint_looks_like_comfy_quant_nvfp4

    unet_cls = node_class_mappings.get("HSWQFP8E4M3UNetLoader")
    if unet_cls is None:
        return False
    if getattr(unet_cls, "_hswq_zi_nvfp4_dispatch", False):
        _DISPATCH_INSTALLED = True
        return True

    _fp8 = frozenset({"fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"})
    _prev = unet_cls.load_unet

    def load_unet(self, unet_name, weight_dtype):
        _ensure_dynamic_load_bake_wrap()
        if weight_dtype in _fp8:
            return _prev(self, unet_name, weight_dtype)
        if weight_dtype == NVFP4_WEIGHT_DTYPE:
            return load_unet_nvfp4_weight_dtype(unet_name, weight_dtype)
        import folder_paths

        if weight_dtype == "default":
            unet_path = folder_paths.get_full_path_or_raise(
                "diffusion_models", unet_name
            )
            if checkpoint_looks_like_comfy_quant_nvfp4(unet_path):
                return load_unet_nvfp4_weight_dtype(unet_name, weight_dtype)
        # int8_tensorwise / other: leave to INT8 dispatch / original (core ConvRot).
        return _prev(self, unet_name, weight_dtype)

    unet_cls.load_unet = load_unet
    unet_cls._hswq_zi_nvfp4_dispatch = True  # type: ignore[attr-defined]
    _DISPATCH_INSTALLED = True
    print(
        "[HSWQ NVFP4] Z Image UNet dispatch: ConvRot NVFP4 -> nodes.zimage_nvfp4 "
        "(comfy_parity; INT8 ConvRot = ComfyUI core)",
        flush=True,
    )
    return True


def _hook_nvfp4_install_for_unet_dispatch() -> None:
    """When package ``__init__`` runs SDXL NVFP4 install, also wrap Z Image UNet."""
    global _INSTALL_HOOKED
    if _INSTALL_HOOKED:
        return
    for name, mod in list(sys.modules.items()):
        if not (
            name.endswith("nodes.nvfp4.comfy_quant_nvfp4")
            or name.endswith(".comfy_quant_nvfp4")
            or name == "comfy_quant_nvfp4"
        ):
            continue
        prev = getattr(mod, "install_nvfp4_option_dispatch", None)
        if prev is None or getattr(prev, "_hswq_zi_unet_hook", False):
            continue

        def install_nvfp4_option_dispatch(node_class_mappings, _prev=prev):
            ok = _prev(node_class_mappings)
            install_zimage_nvfp4_unet_dispatch(node_class_mappings)
            return ok

        install_nvfp4_option_dispatch._hswq_zi_unet_hook = True  # type: ignore[attr-defined]
        mod.install_nvfp4_option_dispatch = install_nvfp4_option_dispatch
        _INSTALL_HOOKED = True
        return


# Import-time: register on comfy_quant; hook SDXL install so UNet wrap runs after INT8.
_attach_to_comfy_quant_module()
_hook_nvfp4_install_for_unet_dispatch()
install_zimage_nvfp4_unet_dispatch()
```


### `nodes/zimage_nvfp4/nvfp4_lora_bake.py`

```python
"""Z Image mixed-pack LoRA bake — Dynamic VRAM only (branch under zimage_nvfp4).

Problem (owner A/B + logs):
  Without LoRA, comfy_parity + act_rotate is fine.
  With LoRA, ModelPatcherDynamic attaches LowVramPatch (``180 patches``).

INT8 Dynamic bake (``patches/comfy_quant_int8.py``) often does **not** fire on this
hybrid pack (no INT8 bake dump in logs), so INT8-protect keys stay as
LowVramPatch. NVFP4 ConvRot bake alone leaves ``patches_left=60`` → broken.

v3: ENTER proved wrap fires; bake still silent (``nvfp4_convrot=False``).

v4 (owner: まだ駄目だ — ENTER patches=180 nvfp4_convrot=False):
  Root cause: kitchen ``QuantizedTensor`` inherits ``torch.Tensor.layout``
  (``torch.strided`` → type name ``\"layout\"``). Old ``_qt_layout_name``
  read ``qt.layout`` first and never saw ``_layout_cls``
  (``TensorCoreNVFP4Layout``), so ``_qt_is_nvfp4`` was always False → gate
  closed → no bake dump → LoRA LowVramPatch left attached.
  Fix: prefer ``_layout_cls`` / ``layout_cls``; gate on ConvRot **flag**
  (do not require QT on ``module.weight`` under Dynamic VRAM).

Does **not** edit ``nodes/nvfp4`` (SDXL).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_BAKE_HOOK_VER = 4
_STATUS_LOGS = 0
_STATUS_LOG_MAX = 24
_ENTER_LOGS = 0
_ENTER_LOG_MAX = 24
_SKIP_SAMPLE_LOGS = 0
_SKIP_SAMPLE_MAX = 6
_GPU_BAKE_INSTALLED = False


def _console(msg: str) -> None:
    print(msg, flush=True)
    logger.info(msg)


def _qt_payload(weight, QuantizedTensor):
    if weight is None:
        return None
    if isinstance(weight, QuantizedTensor):
        return weight
    data = getattr(weight, "data", None)
    if data is not None and isinstance(data, QuantizedTensor):
        return data
    return None


def _qt_layout_name(qt) -> str:
    """Kitchen QT layout class name.

    Do **not** use ``qt.layout`` — that is ``torch.Tensor.layout``
    (``torch.strided``), whose type name is literally ``\"layout\"``.
    Real name lives in ``_layout_cls`` (str) / ``layout_cls`` (type).
    """
    if qt is None:
        return ""
    layout_cls = getattr(qt, "_layout_cls", None)
    if isinstance(layout_cls, str) and layout_cls:
        return layout_cls
    layout_cls_t = getattr(qt, "layout_cls", None)
    if layout_cls_t is not None and not isinstance(layout_cls_t, str):
        name = getattr(layout_cls_t, "__name__", "") or ""
        if name:
            return name
    # Legacy object layout (not torch.layout)
    legacy = getattr(qt, "_layout", None)
    if legacy is not None:
        name = type(legacy).__name__ or ""
        if name and name != "layout":
            return name
    return ""


def _qt_is_nvfp4(weight, QuantizedTensor) -> bool:
    qt = _qt_payload(weight, QuantizedTensor)
    if qt is None:
        return False
    name = _qt_layout_name(qt)
    return "NVFP4" in name or "nvfp4" in name.lower()


def _qt_is_int8_tensorwise(weight, QuantizedTensor) -> bool:
    """INT8 detect including ``_layout_cls`` string (kitchen / protect packs)."""
    qt = _qt_payload(weight, QuantizedTensor)
    if qt is None:
        return False
    name = _qt_layout_name(qt)
    return "TensorWiseINT8" in name or "int8_tensorwise" in name.lower()


def _module_is_nvfp4_convrot(module) -> bool:
    return bool(
        getattr(module, "_hswq_nvfp4_convrot", False)
        or getattr(module, "_hswq_nvfp4_convrot_parity", False)
    )


def _get_baked_key_set(model) -> set:
    keys = getattr(model, "_hswq_zi_nvfp4_baked_keys", None)
    if keys is None:
        keys = set()
        model._hswq_zi_nvfp4_baked_keys = keys
    return keys


def _nvfp4_convrot_diag(model) -> dict:
    """Count ConvRot-armed modules and how many still expose NVFP4 on ``.weight``."""
    out = {"flagged": 0, "qt_on_weight": 0, "has": False}
    if model is None:
        return out
    try:
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        QuantizedTensor = None
    for _name, module in model.named_modules():
        if not _module_is_nvfp4_convrot(module):
            continue
        out["flagged"] += 1
        if QuantizedTensor is None:
            continue
        w = getattr(module, "weight", None)
        if _qt_is_nvfp4(w, QuantizedTensor):
            out["qt_on_weight"] += 1
    out["has"] = out["flagged"] > 0
    return out


def _model_has_nvfp4_convrot(model) -> bool:
    """True if any module was armed with ConvRot NVFP4 (``_hswq_nvfp4_convrot``).

    Do **not** require QT on ``module.weight``: under Dynamic VRAM / LowVramPatch
    the QT often lives behind ``get_key_weight``, while the flag remains on the
    module (act_rotate still hits). v3 gate required both → always False.
    """
    return bool(_nvfp4_convrot_diag(model)["has"])


def _resolve_module(model, module_path: str):
    try:
        import comfy.utils as cu

        return cu.get_attr(model, module_path)
    except Exception:
        return None


def _bake_keys_on_module(patcher, module, keys_to_bake, device_to, already) -> int:
    """Clear LowVramPatch, patch_weight_to_device, drop backup+patches. Keep ``_v``."""
    baked = 0
    for param_key, _key in keys_to_bake:
        if hasattr(module, param_key + "_lowvram_function"):
            setattr(module, param_key + "_lowvram_function", None)
    for _param_key, key in keys_to_bake:
        patcher.patch_weight_to_device(key, device_to=device_to)
        if key in patcher.backup:
            try:
                del patcher.backup[key]
            except KeyError:
                pass
        try:
            del patcher.patches[key]
        except KeyError:
            pass
        already.add(key)
        baked += 1
    return baked


def _iter_patch_weight_keys(patcher):
    """Yield (key, module_path, param_key, module) for weight/bias patches."""
    patches = getattr(patcher, "patches", None) or {}
    model = getattr(patcher, "model", None)
    if model is None or not patches:
        return
    for key in list(patches.keys()):
        if not (key.endswith(".weight") or key.endswith(".bias")):
            continue
        module_path, param_key = key.rsplit(".", 1)
        module = _resolve_module(model, module_path)
        if module is None:
            continue
        yield key, module_path, param_key, module


def bake_nvfp4_convrot_patches_on_dynamic_patcher(patcher, device_to) -> dict:
    """Bake LoRA into ConvRot NVFP4 Linears after ModelPatcherDynamic.load."""
    stats = {
        "baked_nvfp4": 0,
        "candidates": 0,
        "skipped_no_set": 0,
        "skipped_not_nvfp4": 0,
        "skipped_not_convrot": 0,
        "cleared_already": 0,
        "unresolved": 0,
    }
    if not getattr(patcher, "patches", None):
        return stats
    try:
        import comfy.model_patcher as mp
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return stats

    global _SKIP_SAMPLE_LOGS
    already = _get_baked_key_set(patcher.model)
    uuid = getattr(patcher, "patches_uuid", None)
    prev_uuid = getattr(patcher.model, "_hswq_zi_nvfp4_baked_uuid", None)
    if prev_uuid is not None and prev_uuid != uuid:
        already.clear()

    # Group keys by module so LowVramPatch clear happens once per module.
    by_module: dict[str, list] = {}
    modules: dict[str, object] = {}
    for key, module_path, param_key, module in _iter_patch_weight_keys(patcher):
        stats["candidates"] += 1
        if key in already:
            attr = param_key + "_lowvram_function"
            if getattr(module, attr, None) is not None:
                setattr(module, attr, None)
            try:
                del patcher.patches[key]
            except KeyError:
                pass
            stats["cleared_already"] += 1
            continue
        if not _module_is_nvfp4_convrot(module):
            stats["skipped_not_convrot"] += 1
            if _SKIP_SAMPLE_LOGS < _SKIP_SAMPLE_MAX:
                w, _, _ = mp.get_key_weight(patcher.model, key)
                qt = _qt_payload(w, QuantizedTensor)
                _SKIP_SAMPLE_LOGS += 1
                _console(
                    f"[HSWQ ZI NVFP4 LoRA] skip_not_convrot sample "
                    f"#{_SKIP_SAMPLE_LOGS}: {key} layout={_qt_layout_name(qt)!r} "
                    f"convrot={getattr(module, '_hswq_nvfp4_convrot', False)}"
                )
            continue
        weight, set_func, _convert_func = mp.get_key_weight(patcher.model, key)
        if weight is None:
            continue
        if not _qt_is_nvfp4(weight, QuantizedTensor):
            stats["skipped_not_nvfp4"] += 1
            continue
        if set_func is None:
            stats["skipped_no_set"] += 1
            _console(
                f"[HSWQ ZI NVFP4 LoRA] WARN cannot bake {key}: "
                "NVFP4 QT but no set_weight"
            )
            continue
        by_module.setdefault(module_path, []).append((param_key, key))
        modules[module_path] = module

    for module_path, keys_to_bake in by_module.items():
        stats["baked_nvfp4"] += _bake_keys_on_module(
            patcher, modules[module_path], keys_to_bake, device_to, already
        )

    if stats["baked_nvfp4"] > 0:
        patcher.model._hswq_zi_nvfp4_baked_uuid = uuid

    return stats


def bake_remaining_quant_patches_on_dynamic_patcher(patcher, device_to) -> dict:
    """Bake leftover QT LoRA (INT8 protect etc.) that INT8 Dynamic bake missed."""
    stats = {
        "baked_int8": 0,
        "baked_other_qt": 0,
        "candidates": 0,
        "skipped_no_set": 0,
        "skipped_not_qt": 0,
        "cleared_already": 0,
    }
    if not getattr(patcher, "patches", None):
        return stats
    try:
        import comfy.model_patcher as mp
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return stats

    already = _get_baked_key_set(patcher.model)
    uuid = getattr(patcher, "patches_uuid", None)

    by_module: dict[str, list] = {}
    modules: dict[str, object] = {}
    kinds: dict[str, str] = {}
    for key, module_path, param_key, module in _iter_patch_weight_keys(patcher):
        stats["candidates"] += 1
        if key in already:
            attr = param_key + "_lowvram_function"
            if getattr(module, attr, None) is not None:
                setattr(module, attr, None)
            try:
                del patcher.patches[key]
            except KeyError:
                pass
            stats["cleared_already"] += 1
            continue
        weight, set_func, _convert_func = mp.get_key_weight(patcher.model, key)
        if weight is None:
            continue
        qt = _qt_payload(weight, QuantizedTensor)
        if qt is None:
            stats["skipped_not_qt"] += 1
            continue
        if set_func is None:
            stats["skipped_no_set"] += 1
            _console(
                f"[HSWQ ZI NVFP4 LoRA] WARN cannot bake leftover {key}: "
                f"QT layout={_qt_layout_name(qt)!r} but no set_weight"
            )
            continue
        if module_path not in kinds:
            if _qt_is_int8_tensorwise(weight, QuantizedTensor):
                kinds[module_path] = "int8"
            elif _qt_is_nvfp4(weight, QuantizedTensor):
                kinds[module_path] = "nvfp4"
            else:
                kinds[module_path] = "other"
        by_module.setdefault(module_path, []).append((param_key, key))
        modules[module_path] = module

    for module_path, keys_to_bake in by_module.items():
        n = _bake_keys_on_module(
            patcher, modules[module_path], keys_to_bake, device_to, already
        )
        if kinds.get(module_path) == "int8":
            stats["baked_int8"] += n
        else:
            stats["baked_other_qt"] += n

    if stats["baked_int8"] > 0 or stats["baked_other_qt"] > 0:
        patcher.model._hswq_zi_nvfp4_baked_uuid = uuid
        if stats["baked_int8"] > 0:
            patcher.model._hswq_int8_baked_uuid = uuid

    return stats


def _dump_bake_status(nv_stats: dict, rem_stats: dict, patcher, reason: str) -> None:
    global _STATUS_LOGS
    if _STATUS_LOGS >= _STATUS_LOG_MAX:
        return
    _STATUS_LOGS += 1
    left = len(getattr(patcher, "patches", None) or {})
    _console(
        "[HSWQ ZI NVFP4 LoRA] Dynamic.load bake "
        f"#{_STATUS_LOGS} ({reason}): "
        f"nvfp4_baked={nv_stats.get('baked_nvfp4', 0)} "
        f"int8_baked={rem_stats.get('baked_int8', 0)} "
        f"other_qt_baked={rem_stats.get('baked_other_qt', 0)} "
        f"nv_candidates={nv_stats.get('candidates', 0)} "
        f"rem_candidates={rem_stats.get('candidates', 0)} "
        f"skip_not_convrot={nv_stats.get('skipped_not_convrot', 0)} "
        f"patches_left={left}"
    )
    if left > 0:
        sample = list((getattr(patcher, "patches", None) or {}).keys())[:4]
        _console(
            f"[HSWQ ZI NVFP4 LoRA] WARN patches_left={left} after bake "
            f"sample_keys={sample}"
        )


def _patcher_has_quant_via_keys(patcher) -> bool:
    """True if any LoRA patch key resolves to NVFP4/INT8 QT via get_key_weight."""
    if not getattr(patcher, "patches", None):
        return False
    try:
        import comfy.model_patcher as mp
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return False
    for key, _module_path, _param_key, _module in _iter_patch_weight_keys(patcher):
        weight, _set_func, _convert = mp.get_key_weight(patcher.model, key)
        if weight is None:
            continue
        if _qt_is_nvfp4(weight, QuantizedTensor) or _qt_is_int8_tensorwise(
            weight, QuantizedTensor
        ):
            return True
    return False


def run_zimage_nvfp4_lora_bake_on_patcher(patcher, device_to=None, reason: str = "wrap") -> bool:
    """Bake NVFP4 ConvRot + leftover QT if this patcher is a ZI NVFP4 pack with LoRA."""
    model = getattr(patcher, "model", None)
    if model is None:
        return False
    diag = _nvfp4_convrot_diag(model)
    has_flag = bool(diag["has"])
    has_baked = bool(getattr(model, "_hswq_zi_nvfp4_baked_keys", None))
    n_patches = len(getattr(patcher, "patches", None) or {})
    if not has_flag and not has_baked:
        # Fallback: patches present and QT visible via get_key_weight
        if n_patches == 0 or not _patcher_has_quant_via_keys(patcher):
            return False
    if n_patches == 0 and not has_baked:
        return False
    if device_to is None:
        device_to = getattr(patcher, "load_device", None)
    nv_stats = bake_nvfp4_convrot_patches_on_dynamic_patcher(patcher, device_to=device_to)
    rem_stats = bake_remaining_quant_patches_on_dynamic_patcher(
        patcher, device_to=device_to
    )
    _dump_bake_status(nv_stats, rem_stats, patcher, reason=reason)
    return True


def _unwrap_to_non_zi_load(load_fn):
    """Walk past our ZI wraps so reinstall does not nest ZI→ZI."""
    cur = load_fn
    seen = set()
    while (
        cur is not None
        and id(cur) not in seen
        and getattr(cur, "_hswq_zi_nvfp4_lora_bake", False)
    ):
        seen.add(id(cur))
        nxt = getattr(cur, "_hswq_zi_nvfp4_prev_dynamic_load", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return cur


def install_zimage_nvfp4_lora_bake(force: bool = False) -> bool:
    """Wrap ModelPatcherDynamic.load: NVFP4 ConvRot bake + leftover INT8 QT bake."""
    try:
        import comfy.model_patcher as mp
    except ImportError:
        return False

    Dynamic = getattr(mp, "ModelPatcherDynamic", None)
    if Dynamic is None:
        _console("[HSWQ ZI NVFP4 LoRA] ModelPatcherDynamic missing — bake hook skipped")
        return False
    original = getattr(Dynamic, "load", None)
    if original is None:
        return False
    if (
        not force
        and getattr(original, "_hswq_zi_nvfp4_lora_bake", False)
        and getattr(original, "_hswq_zi_nvfp4_lora_bake_ver", 0) >= _BAKE_HOOK_VER
    ):
        install_load_models_gpu_bake_hook(force=False)
        return True

    # Prefer chaining under current outer wrap (INT8 / stock), never nest ZI→ZI.
    prev_load = _unwrap_to_non_zi_load(original)

    def load(
        self,
        device_to=None,
        lowvram_model_memory=0,
        force_patch_weights=False,
        full_load=False,
        dirty=False,
    ):
        global _ENTER_LOGS
        if _ENTER_LOGS < _ENTER_LOG_MAX:
            _ENTER_LOGS += 1
            n_patches = len(getattr(self, "patches", None) or {})
            model = getattr(self, "model", None)
            diag = _nvfp4_convrot_diag(model)
            _console(
                f"[HSWQ ZI NVFP4 LoRA] Dynamic.load ENTER #{_ENTER_LOGS}: "
                f"patches={n_patches} "
                f"nvfp4_convrot={diag['has']} "
                f"flagged={diag['flagged']} "
                f"qt_on_weight={diag['qt_on_weight']} "
                f"model={type(model).__name__ if model is not None else None}"
            )
        result = prev_load(
            self,
            device_to=device_to,
            lowvram_model_memory=lowvram_model_memory,
            force_patch_weights=force_patch_weights,
            full_load=full_load,
            dirty=dirty,
        )
        run_zimage_nvfp4_lora_bake_on_patcher(
            self, device_to=device_to, reason="Dynamic.load"
        )
        return result

    load._hswq_zi_nvfp4_lora_bake = True  # type: ignore[attr-defined]
    load._hswq_zi_nvfp4_lora_bake_ver = _BAKE_HOOK_VER  # type: ignore[attr-defined]
    load._hswq_zi_nvfp4_prev_dynamic_load = prev_load  # type: ignore[attr-defined]
    Dynamic.load = load
    _console(
        f"[HSWQ ZI NVFP4 LoRA] Dynamic.load bake hook ON v{_BAKE_HOOK_VER} "
        "(NVFP4 ConvRot bake + leftover INT8/QT bake + load_models_gpu)"
    )
    install_load_models_gpu_bake_hook(force=True)
    return True


def _unwrap_to_non_zi_load_models_gpu(fn):
    cur = fn
    seen = set()
    while (
        cur is not None
        and id(cur) not in seen
        and getattr(cur, "_hswq_zi_nvfp4_gpu_bake", False)
    ):
        seen.add(id(cur))
        nxt = getattr(cur, "_hswq_zi_nvfp4_prev_load_models_gpu", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return cur


def install_load_models_gpu_bake_hook(force: bool = False) -> bool:
    """After MultiGPU/stock load_models_gpu, bake any remaining ZI NVFP4 LoRA patches."""
    global _GPU_BAKE_INSTALLED
    try:
        import comfy.model_management as mm
    except ImportError:
        return False
    original = mm.load_models_gpu
    if (
        not force
        and getattr(original, "_hswq_zi_nvfp4_gpu_bake", False)
        and getattr(original, "_hswq_zi_nvfp4_gpu_bake_ver", 0) >= _BAKE_HOOK_VER
    ):
        _GPU_BAKE_INSTALLED = True
        return True
    prev = _unwrap_to_non_zi_load_models_gpu(original)

    def load_models_gpu(*args, **kwargs):
        result = prev(*args, **kwargs)
        try:
            for loaded in list(getattr(mm, "current_loaded_models", []) or []):
                patcher = getattr(loaded, "model", None)
                if patcher is None:
                    continue
                try:
                    if not bool(patcher.is_dynamic()):
                        continue
                except Exception:
                    continue
                if not getattr(patcher, "patches", None):
                    # Still try if bake keys exist (LowVram cleared but leftover)
                    if not getattr(
                        getattr(patcher, "model", None),
                        "_hswq_zi_nvfp4_baked_keys",
                        None,
                    ):
                        continue
                run_zimage_nvfp4_lora_bake_on_patcher(
                    patcher,
                    device_to=getattr(patcher, "load_device", None),
                    reason="load_models_gpu",
                )
        except Exception as exc:
            _console(f"[HSWQ ZI NVFP4 LoRA] load_models_gpu bake error: {exc!r}")
        return result

    load_models_gpu._hswq_zi_nvfp4_gpu_bake = True  # type: ignore[attr-defined]
    load_models_gpu._hswq_zi_nvfp4_gpu_bake_ver = _BAKE_HOOK_VER  # type: ignore[attr-defined]
    load_models_gpu._hswq_zi_nvfp4_prev_load_models_gpu = prev  # type: ignore[attr-defined]
    mm.load_models_gpu = load_models_gpu
    _GPU_BAKE_INSTALLED = True
    _console(
        f"[HSWQ ZI NVFP4 LoRA] load_models_gpu bake hook ON v{_BAKE_HOOK_VER}"
    )
    return True


def reset_zimage_nvfp4_lora_bake_log_counters() -> None:
    global _STATUS_LOGS, _SKIP_SAMPLE_LOGS, _ENTER_LOGS
    _STATUS_LOGS = 0
    _SKIP_SAMPLE_LOGS = 0
    _ENTER_LOGS = 0
```


### `nodes/zimage_nvfp4/nvfp4_comfy_parity.py`

```python
"""Z Image / ZIT ConvRot NVFP4 — ComfyUI stock GEMM + online act rotate.

Ported from ``hswq/benchmark/nvfp4_comfy_parity.py`` (same math as
``zi_convrot_nvfp4_bench.py``). Product HSWQ Tensor Core Linear.forward
breaks Pixel SSIM on Z Image ConvRot packs; the bench path does not.

Call ``apply_nvfp4_comfy_parity()`` **after** ``apply_comfy_quant_nvfp4_patches()``
for UNet / Z Image loads. SDXL product path keeps TC via
``restore_nvfp4_tc_product_stack()`` before SDXL checkpoint load.

Does not edit ComfyUI-master.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_PARITY_APPLIED = False
_PRODUCT_LOAD: Optional[Callable] = None
_PRODUCT_MP: Optional[Callable] = None

# Runtime / load diagnostics (console — owner-ordered visibility).
_LOAD_NVFP4_SEEN = 0
_LOAD_CONVROT_ARMED = 0
_LOAD_NVFP4_NO_CONVROT = 0
_LOAD_INT8_CONVROT_ARMED = 0
_ACT_ROTATE_HITS = 0
_ACT_ROTATE_INT8_HITS = 0
_ACT_ROTATE_LOG_EVERY = 32
_ACT_ROTATE_FIRST_N = 4


def _console(msg: str) -> None:
    print(msg, flush=True)
    logger.info(msg)


def reset_nvfp4_parity_load_counters() -> None:
    global _LOAD_NVFP4_SEEN, _LOAD_CONVROT_ARMED, _LOAD_NVFP4_NO_CONVROT
    global _LOAD_INT8_CONVROT_ARMED, _ACT_ROTATE_HITS, _ACT_ROTATE_INT8_HITS
    _LOAD_NVFP4_SEEN = 0
    _LOAD_CONVROT_ARMED = 0
    _LOAD_NVFP4_NO_CONVROT = 0
    _LOAD_INT8_CONVROT_ARMED = 0
    _ACT_ROTATE_HITS = 0
    _ACT_ROTATE_INT8_HITS = 0


def log_nvfp4_parity_load_summary(label: str = "") -> None:
    """Print how many nvfp4 / int8protect ConvRot layers were armed during load."""
    tag = f" ({label})" if label else ""
    _console(
        f"[HSWQ NVFP4][diag] load summary{tag}: "
        f"nvfp4_seen={_LOAD_NVFP4_SEEN} "
        f"convrot_armed={_LOAD_CONVROT_ARMED} "
        f"nvfp4_no_convrot={_LOAD_NVFP4_NO_CONVROT} "
        f"int8_convrot_armed={_LOAD_INT8_CONVROT_ARMED}"
    )
    if _LOAD_NVFP4_SEEN == 0:
        _console(
            "[HSWQ NVFP4][diag] WARNING: zero nvfp4 layers seen during load — "
            "comfy_quant markers may be missing / wrong prefix "
            "(kitchen bare→prefixed remap should have run)"
        )
    elif _LOAD_CONVROT_ARMED == 0:
        _console(
            "[HSWQ NVFP4][diag] WARNING: nvfp4 layers loaded but "
            "convrot_armed=0 — act rotate will never run"
        )
    if _LOAD_INT8_CONVROT_ARMED == 0:
        _console(
            "[HSWQ NVFP4][diag] WARNING: int8protect ConvRot Linear armed=0 — "
            "mixed packs need online act rotate on protect Linears "
            "(offline W@H^T without x@H → bit-crush)"
        )


def summarize_nvfp4_parity_modules(model, max_names: int = 8) -> None:
    """Post-load walk: Linear counts + forward type + sample ConvRot names."""
    import torch.nn as nn

    try:
        import comfy.ops as ops
    except Exception as e:
        _console(f"[HSWQ NVFP4][diag] post-load skipped (ops): {e}")
        return

    # ModelPatcher -> BaseModel -> diffusion_model (same as INT8 summary).
    diffusion = model
    if hasattr(model, "model") and hasattr(model.model, "diffusion_model"):
        diffusion = model.model.diffusion_model
    elif hasattr(model, "diffusion_model"):
        diffusion = model.diffusion_model

    n_linear = 0
    n_convrot = 0
    n_int8_convrot = 0
    n_tc_arm = 0
    names: list[str] = []
    names_i8: list[str] = []
    for name, mod in diffusion.named_modules():
        if not isinstance(mod, nn.Linear) and "Linear" not in type(mod).__name__:
            continue
        n_linear += 1
        if getattr(mod, "_hswq_nvfp4_convrot", False):
            n_convrot += 1
            if len(names) < max_names:
                gs = getattr(mod, "_hswq_nvfp4_convrot_groupsize", "?")
                names.append(f"{name}(gs={gs})")
        if getattr(mod, "_hswq_int8_convrot", False):
            n_int8_convrot += 1
            if len(names_i8) < max_names:
                gs = getattr(mod, "_hswq_int8_convrot_groupsize", "?")
                names_i8.append(f"{name}(gs={gs})")
        if getattr(mod, "_hswq_nvfp4", False):
            n_tc_arm += 1

    fwd = ops.mixed_precision_ops().Linear.forward
    fwd_parity = bool(getattr(fwd, "_hswq_nvfp4_convrot_parity", False))
    fwd_tc = bool(getattr(fwd, "_hswq_nvfp4_full_forward", False))
    load_fn = ops._load_quantized_module
    load_parity = bool(getattr(load_fn, "_hswq_nvfp4_comfy_only", False))
    # INT8 may wrap load outside; peel once for display.
    if not load_parity and getattr(load_fn, "_hswq_int8_decode_patched", False):
        inner = _closure_named(load_fn, "original_load")
        if inner is not None:
            load_parity = bool(getattr(inner, "_hswq_nvfp4_comfy_only", False))
            load_fn = inner
    load_tc = bool(
        getattr(load_fn, "_hswq_nvfp4_full_load", False)
        and not getattr(load_fn, "_hswq_nvfp4_comfy_only", False)
    )

    _console(
        "[HSWQ NVFP4][diag] ===== post-load ====="
    )
    _console(
        f"[HSWQ NVFP4][diag] Linear={n_linear} "
        f"_hswq_nvfp4_convrot={n_convrot} "
        f"_hswq_int8_convrot={n_int8_convrot} "
        f"_hswq_nvfp4(TC arm)={n_tc_arm}"
    )
    _console(
        f"[HSWQ NVFP4][diag] Linear.forward: "
        f"parity={fwd_parity} tc_full={fwd_tc} "
        f"load: parity={load_parity} tc_full={load_tc} "
        f"_PARITY_APPLIED={_PARITY_APPLIED}"
    )
    if names:
        _console(
            "[HSWQ NVFP4][diag] sample NVFP4 ConvRot: "
            + ", ".join(names)
        )
    if names_i8:
        _console(
            "[HSWQ NVFP4][diag] sample INT8 protect ConvRot: "
            + ", ".join(names_i8)
        )
    _console(
        f"[HSWQ NVFP4][diag] act_rotate_hits_so_far="
        f"nvfp4={_ACT_ROTATE_HITS} int8protect={_ACT_ROTATE_INT8_HITS}"
    )
    _console("[HSWQ NVFP4][diag] =====================")


def remember_nvfp4_tc_product_stack(load_fn, mp_fn) -> None:
    """Store SDXL product TC refs (call from apply_comfy_quant_nvfp4_patches only).

    Never overwrite with parity wrappers — SDXL must always be able to restore.
    """
    global _PRODUCT_LOAD, _PRODUCT_MP
    if load_fn is not None and getattr(load_fn, "_hswq_nvfp4_full_load", False):
        if not getattr(load_fn, "_hswq_nvfp4_comfy_only", False):
            _PRODUCT_LOAD = load_fn
    if mp_fn is not None and getattr(mp_fn, "_hswq_nvfp4_stack_ver", 0):
        if not getattr(mp_fn, "_hswq_nvfp4_comfy_only", False):
            _PRODUCT_MP = mp_fn


def is_nvfp4_comfy_parity_active() -> bool:
    return bool(_PARITY_APPLIED)


def _closure_named(fn, name: str):
    try:
        cells = fn.__closure__ or ()
        for n, c in zip(fn.__code__.co_freevars, cells):
            if n == name:
                return c.cell_contents
    except Exception:
        return None
    return None


def _is_tc_full_load(fn) -> bool:
    """True for product TC load (load_nvfp4_linear_module), not parity stock load."""
    return bool(
        getattr(fn, "_hswq_nvfp4_full_load", False)
        and not getattr(fn, "_hswq_nvfp4_comfy_only", False)
    )


def _parity_load_in_chain(fn) -> bool:
    """True if comfy_parity load wrapper is already somewhere under ``fn``."""
    cur = fn
    seen = set()
    for _ in range(8):
        if cur is None or id(cur) in seen:
            return False
        seen.add(id(cur))
        if getattr(cur, "_hswq_nvfp4_comfy_only", False):
            return True
        if getattr(cur, "_hswq_int8_decode_patched", False):
            cur = _closure_named(cur, "original_load")
            continue
        if _is_tc_full_load(cur):
            cur = _closure_named(cur, "_orig_load")
            continue
        return False
    return False


def _resolve_load_under_tc(patched_load):
    """Callable under TC for parity to close over (stock Comfy or INT8 normalize).

    Peel **only** TC ``load_nvfp4_linear_module``. Keep INT8 decode wrap so
    int8protect layers still normalize ``comfy_quant`` tensors.
    Never return TC itself (ones(1) / ``_hswq_nvfp4`` arm).
    """
    if _is_tc_full_load(patched_load):
        inner = _closure_named(patched_load, "_orig_load")
        if inner is None:
            raise RuntimeError(
                "[HSWQ NVFP4] comfy_parity: TC load has no _orig_load "
                "(cannot recover Comfy / INT8 load under TC)"
            )
        if _is_tc_full_load(inner):
            raise RuntimeError(
                "[HSWQ NVFP4] comfy_parity: nested TC load; refusing"
            )
        return inner
    return patched_load


def _chain_has_int8_protect_in_load(fn) -> bool:
    """True if load chain already arms INT8 protect ConvRot after stock load."""
    cur = fn
    seen = set()
    for _ in range(8):
        if cur is None or id(cur) in seen:
            return False
        seen.add(id(cur))
        if getattr(cur, "_hswq_int8_protect_in_load", False):
            return True
        if getattr(cur, "_hswq_int8_protect_arm_v2", False):
            return True
        if getattr(cur, "_hswq_int8_decode_patched", False):
            cur = _closure_named(cur, "original_load")
            continue
        if _is_tc_full_load(cur):
            cur = _closure_named(cur, "_orig_load")
            continue
        if getattr(cur, "_hswq_nvfp4_comfy_only", False):
            return False
        return False
    return False


def _ensure_int8_protect_arm_overlay() -> None:
    """Hot-refresh: wrap current load so INT8 protect Linears get act-rotate arm.

    No-op when ``_load_quantized_module_comfy_only`` already has
    ``_hswq_int8_protect_in_load`` (fresh install path).
    """
    try:
        import comfy.ops as ops
    except Exception:
        return
    cur = ops._load_quantized_module
    if _chain_has_int8_protect_in_load(cur):
        return
    from ..nvfp4.nvfp4_conf import decode_comfy_quant_conf

    def _load_int8_protect_arm_overlay(
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
        conf = decode_comfy_quant_conf(state_dict.get(f"{prefix}comfy_quant"))
        out = cur(
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
        _arm_int8_protect_convrot_after_stock_load(module, conf)
        return out

    _load_int8_protect_arm_overlay._hswq_int8_protect_arm_v2 = True  # type: ignore[attr-defined]
    ops._load_quantized_module = _load_int8_protect_arm_overlay
    _console(
        "[HSWQ NVFP4] comfy_parity: INT8 protect ConvRot arm overlay installed "
        "(hot refresh; online act rotate for protect Linears)"
    )


def _unwrap_stock_forward(forward_fn):
    """Peel HSWQ TC wrappers until stock MixedPrecision Linear.forward."""
    f = forward_fn
    for _ in range(8):
        if not getattr(f, "_hswq_nvfp4_full_forward", False):
            return f
        stock = _closure_named(f, "stock_forward")
        if stock is None:
            return None
        f = stock
    return None


def _is_int8_tensorwise_convrot_conf(conf) -> bool:
    """True for INT8 protect Linear layers stamped with ConvRot offline rotate."""
    if not isinstance(conf, dict):
        return False
    fmt = conf.get("format")
    if fmt is not None and str(fmt).lower() != "int8_tensorwise":
        return False
    from ..nvfp4.nvfp4_conf import convrot_flags_from_conf

    enabled, _gs = convrot_flags_from_conf(conf)
    return bool(enabled)


def _make_convrot_parity_forward(stock_forward):
    """Stock MixedPrecision forward + online act rotate for ConvRot weights.

    INT8 protect ConvRot Linear only (``_hswq_int8_convrot``). NVFP4 ConvRot
    (``_hswq_nvfp4_convrot``) is left to the kitchen NVFP4 path; rotating
    activations in stock forward double-rotates NVFP4 and destroys the image.

    Convert stores offline ``W @ H^T``. For INT8 protect, online must apply
    ``x @ H`` when Dynamic / dequant / ``F.linear`` would otherwise skip the
    kitchen ``int8_linear(convrot=True)`` rotate.
    """
    from ..nvfp4.nvfp4_hadamard import build_hadamard, rotate_last_dim

    def forward_parity(self, input, *args, **kwargs):
        global _ACT_ROTATE_HITS, _ACT_ROTATE_INT8_HITS
        nv = bool(getattr(self, "_hswq_nvfp4_convrot", False))
        i8 = bool(getattr(self, "_hswq_int8_convrot", False))
        if nv or i8:
            if nv:
                _ACT_ROTATE_HITS += 1
                hit = _ACT_ROTATE_HITS
                tag = "nvfp4"
                gs = int(getattr(self, "_hswq_nvfp4_convrot_groupsize", 256) or 256)
            else:
                _ACT_ROTATE_INT8_HITS += 1
                hit = _ACT_ROTATE_INT8_HITS
                tag = "int8protect"
                gs = int(getattr(self, "_hswq_int8_convrot_groupsize", 256) or 256)
            if hit <= _ACT_ROTATE_FIRST_N or (
                _ACT_ROTATE_LOG_EVERY > 0 and hit % _ACT_ROTATE_LOG_EVERY == 0
            ):
                cls = type(self).__name__
                shape = tuple(getattr(input, "shape", ()))
                _console(
                    f"[HSWQ NVFP4][diag] act_rotate hit#{hit} ({tag}) "
                    f"Linear={cls} gs={gs} x.shape={shape}"
                )
            h = getattr(self, "_hswq_nvfp4_parity_H", None)
            if h is None or h.device != input.device or h.dtype != input.dtype:
                h = build_hadamard(gs, device=input.device, dtype=input.dtype)
                self._hswq_nvfp4_parity_H = h
            input = rotate_last_dim(input, h, gs)
        return stock_forward(self, input, *args, **kwargs)

    forward_parity._hswq_nvfp4_convrot_parity = True  # type: ignore[attr-defined]
    return forward_parity


def _arm_convrot_after_stock_load(module, conf) -> None:
    global _LOAD_NVFP4_SEEN, _LOAD_CONVROT_ARMED, _LOAD_NVFP4_NO_CONVROT
    from ..nvfp4.nvfp4_conf import convrot_flags_from_conf, is_nvfp4_conf

    if not is_nvfp4_conf(conf):
        return
    _LOAD_NVFP4_SEEN += 1
    enabled, gs = convrot_flags_from_conf(conf)
    module._hswq_nvfp4_convrot = bool(enabled)
    module._hswq_nvfp4_convrot_groupsize = int(gs)
    try:
        import comfy.quant_ops as quant_ops

        p = getattr(module, "weight", None)
        layout = getattr(p, "layout_params", None) if p is not None else None
        if isinstance(layout, quant_ops.Params) and getattr(layout, "convrot", False):
            layout.convrot = False
    except Exception:
        pass
    if enabled:
        _LOAD_CONVROT_ARMED += 1
        if _LOAD_CONVROT_ARMED <= 4 or _LOAD_CONVROT_ARMED % 40 == 0:
            fmt = conf.get("format")
            top = conf.get("convrot")
            params = conf.get("params") if isinstance(conf.get("params"), dict) else {}
            _console(
                f"[HSWQ NVFP4][diag] arm ConvRot #{_LOAD_CONVROT_ARMED} "
                f"gs={gs} format={fmt} convrot={top!r} "
                f"params.convrot={params.get('convrot')!r}"
            )
    else:
        _LOAD_NVFP4_NO_CONVROT += 1
        if _LOAD_NVFP4_NO_CONVROT <= 4:
            _console(
                f"[HSWQ NVFP4][diag] nvfp4 without convrot "
                f"(#{_LOAD_NVFP4_NO_CONVROT}) keys={list(conf.keys())[:12]}"
            )
    # Do not set _hswq_nvfp4 (TC full-forward arm).


def _arm_int8_protect_convrot_after_stock_load(module, conf) -> None:
    """Arm parity act-rotate for INT8 protect Linear (offline W@H^T).

    Clears ``Params.convrot`` so kitchen QT path does not double-rotate when
    QuantTensor still reaches ``int8_linear``. Same pattern as INT8 Conv2d
    (``_hswq_convrot`` + cleared Params.convrot).
    """
    global _LOAD_INT8_CONVROT_ARMED
    if not _is_int8_tensorwise_convrot_conf(conf):
        return
    from ..nvfp4.nvfp4_conf import convrot_flags_from_conf

    _enabled, gs = convrot_flags_from_conf(conf)
    module._hswq_int8_convrot = True
    module._hswq_int8_convrot_groupsize = int(gs)
    try:
        import comfy.quant_ops as quant_ops

        p = getattr(module, "weight", None)
        layout = getattr(p, "layout_params", None) if p is not None else None
        if isinstance(layout, quant_ops.Params) and getattr(layout, "convrot", False):
            layout.convrot = False
    except Exception:
        pass
    _LOAD_INT8_CONVROT_ARMED += 1
    if _LOAD_INT8_CONVROT_ARMED <= 4 or _LOAD_INT8_CONVROT_ARMED % 20 == 0:
        _console(
            f"[HSWQ NVFP4][diag] arm INT8 protect ConvRot "
            f"#{_LOAD_INT8_CONVROT_ARMED} gs={gs}"
        )


def require_convrot_parity_forward() -> None:
    """Fail fast if TC full-forward is still installed (bench guard)."""
    import comfy.ops as ops

    mp = ops.mixed_precision_ops()
    fwd = mp.Linear.forward
    if getattr(fwd, "_hswq_nvfp4_full_forward", False):
        raise RuntimeError(
            "ConvRot NVFP4 parity requires stock Comfy Linear.forward + act rotate; "
            "HSWQ TC full-forward is still installed (_hswq_nvfp4_full_forward)."
        )
    if not getattr(fwd, "_hswq_nvfp4_convrot_parity", False):
        raise RuntimeError(
            "ConvRot NVFP4 parity forward missing "
            "(_hswq_nvfp4_convrot_parity). Call apply_nvfp4_comfy_parity()."
        )


def restore_nvfp4_tc_product_stack() -> bool:
    """Put SDXL product TC load + forward back. No-op if already on TC.

    Z Image parity must never leak into SDXL. Call this from the SDXL loader
    only — do not change SDXL's TC / LoRA bake behavior.
    """
    global _PARITY_APPLIED
    try:
        import comfy.ops as ops
    except Exception as e:
        logger.warning("[HSWQ NVFP4] restore TC stack skipped: %s", e)
        return False

    mp = ops.mixed_precision_ops
    already_tc = (
        getattr(mp, "_hswq_nvfp4_stack_ver", 0)
        and not getattr(mp, "_hswq_nvfp4_comfy_only", False)
        and not getattr(ops._load_quantized_module, "_hswq_nvfp4_comfy_only", False)
    )
    if already_tc and not _PARITY_APPLIED:
        return True

    if _PRODUCT_LOAD is None or _PRODUCT_MP is None:
        if already_tc:
            _PARITY_APPLIED = False
            return True
        logger.warning(
            "[HSWQ NVFP4] restore TC stack: no saved product refs "
            "(SDXL needs apply_comfy_quant_nvfp4_patches first)"
        )
        return False

    ops._load_quantized_module = _PRODUCT_LOAD
    ops.mixed_precision_ops = _PRODUCT_MP
    _PARITY_APPLIED = False
    _console("[HSWQ NVFP4] restored product TC stack (SDXL path; parity off)")
    return True


def apply_nvfp4_comfy_parity() -> bool:
    """Switch NVFP4 Linear path to stock Comfy GEMM + online act rotate.

    Also registers aten.addmm for TensorCoreNVFP4Layout (kitchen gap).
    Saves product TC refs so SDXL can restore later.
    """
    global _PARITY_APPLIED, _PRODUCT_LOAD, _PRODUCT_MP
    try:
        import comfy.ops as ops
        from comfy.quant_ops import QUANT_ALGOS
    except Exception as e:
        logger.warning("[HSWQ NVFP4] comfy_parity import failed: %s", e)
        return False

    from .nvfp4_addmm_patch import register_nvfp4_addmm_handler
    from ..nvfp4.nvfp4_conf import decode_comfy_quant_conf, is_nvfp4_conf
    from ..nvfp4.nvfp4_forward import attach_nvfp4_linear_lora_bake
    # Product Z Image: keep ConvRot Linear LoRA bake (same as SDXL). Do not peel.

    register_nvfp4_addmm_handler()

    if "nvfp4" not in QUANT_ALGOS:
        logger.warning("[HSWQ NVFP4] comfy_parity: nvfp4 not in QUANT_ALGOS")
        return False

    patched_load = ops._load_quantized_module
    # Prefer refs already saved by apply_comfy_quant_nvfp4_patches (TC only).
    remember_nvfp4_tc_product_stack(patched_load, ops.mixed_precision_ops)

    def _refresh_parity_mp() -> None:
        _cur_mp = ops.mixed_precision_ops

        def mixed_precision_ops_parity_refresh(*args, **kwargs):
            mp = _cur_mp(*args, **kwargs)
            Lin = mp.Linear
            attach_nvfp4_linear_lora_bake(Lin)
            if getattr(Lin.forward, "_hswq_nvfp4_full_forward", False):
                stock = _unwrap_stock_forward(Lin.forward)
                if stock is not None:
                    Lin.forward = _make_convrot_parity_forward(stock)
            elif not getattr(Lin.forward, "_hswq_nvfp4_convrot_parity", False):
                Lin.forward = _make_convrot_parity_forward(Lin.forward)
            return mp

        mixed_precision_ops_parity_refresh._hswq_nvfp4_comfy_only = True  # type: ignore[attr-defined]
        mixed_precision_ops_parity_refresh._hswq_nvfp4_stack_ver = getattr(
            _cur_mp, "_hswq_nvfp4_stack_ver", 0
        )  # type: ignore[attr-defined]
        if getattr(_cur_mp, "_hswq_nvfp4_orig_mp", None) is not None:
            mixed_precision_ops_parity_refresh._hswq_nvfp4_orig_mp = (  # type: ignore[attr-defined]
                _cur_mp._hswq_nvfp4_orig_mp
            )
        ops.mixed_precision_ops = mixed_precision_ops_parity_refresh

    # Already on parity load (possibly under INT8 decode wrap): keep load chain.
    if _parity_load_in_chain(patched_load):
        _ensure_int8_protect_arm_overlay()
        _refresh_parity_mp()
        _PARITY_APPLIED = True
        _console(
            "[HSWQ NVFP4] comfy_parity refresh: stock GEMM + act rotate "
            "(NVFP4 + INT8 protect) + ConvRot Linear LoRA bake (Z Image)"
        )
        return True

    orig_load = _resolve_load_under_tc(patched_load)

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
        conf = decode_comfy_quant_conf(state_dict.get(f"{prefix}comfy_quant"))
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
        if is_nvfp4_conf(conf):
            _arm_convrot_after_stock_load(module, conf)
        else:
            _arm_int8_protect_convrot_after_stock_load(module, conf)
        return out

    _load_quantized_module_comfy_only._hswq_nvfp4_comfy_only = True  # type: ignore[attr-defined]
    _load_quantized_module_comfy_only._hswq_int8_protect_in_load = True  # type: ignore[attr-defined]
    # Bench marks full_load on the parity wrapper too; keep comfy_only distinct
    # so remember_nvfp4_tc_product_stack never stores this as SDXL TC.
    ops._load_quantized_module = _load_quantized_module_comfy_only

    _cur_mp = ops.mixed_precision_ops

    def mixed_precision_ops_comfy_only(*args, **kwargs):
        mp = _cur_mp(*args, **kwargs)
        Lin = mp.Linear
        attach_nvfp4_linear_lora_bake(Lin)
        if getattr(Lin.forward, "_hswq_nvfp4_full_forward", False):
            stock = _unwrap_stock_forward(Lin.forward)
            if stock is None:
                raise RuntimeError(
                    "Could not unwrap HSWQ TC Linear.forward for ConvRot parity"
                )
            Lin.forward = _make_convrot_parity_forward(stock)
        elif not getattr(Lin.forward, "_hswq_nvfp4_convrot_parity", False):
            Lin.forward = _make_convrot_parity_forward(Lin.forward)
        return mp

    mixed_precision_ops_comfy_only._hswq_nvfp4_comfy_only = True  # type: ignore[attr-defined]
    mixed_precision_ops_comfy_only._hswq_nvfp4_stack_ver = getattr(
        _cur_mp, "_hswq_nvfp4_stack_ver", 0
    )  # type: ignore[attr-defined]
    if getattr(_cur_mp, "_hswq_nvfp4_orig_mp", None) is not None:
        mixed_precision_ops_comfy_only._hswq_nvfp4_orig_mp = (  # type: ignore[attr-defined]
            _cur_mp._hswq_nvfp4_orig_mp
        )
    ops.mixed_precision_ops = mixed_precision_ops_comfy_only

    # Prove unwrap once at install; keep LoRA bake attached for product use.
    mp0 = _cur_mp()
    attach_nvfp4_linear_lora_bake(mp0.Linear)
    if getattr(mp0.Linear.forward, "_hswq_nvfp4_full_forward", False):
        stock0 = _unwrap_stock_forward(mp0.Linear.forward)
        if stock0 is None:
            raise RuntimeError(
                "[HSWQ NVFP4] comfy_parity: failed to unwrap Linear.forward "
                "to Comfy stock at install"
            )
        mp0.Linear.forward = _make_convrot_parity_forward(stock0)

    _PARITY_APPLIED = True
    _console(
        "[HSWQ NVFP4] comfy_parity ON: stock MixedPrecision GEMM + online act rotate "
        "(NVFP4 ConvRot + INT8 protect ConvRot) "
        "+ ConvRot Linear LoRA bake (Z Image; not HSWQ TC Linear.forward)"
    )
    return True
```

### 3.2 Added `prestartup_script.py`


### `prestartup_script.py`

```python
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
```

### 3.3 Modified `patches/comfy_quant_int8.py` (full file at HEAD)

This file is the shared INT8 load / LoRA bake patch. The Z Image-critical delta is `_qt_is_int8_tensorwise` and its use in `_model_has_int8_quantized_weights` / Dynamic bake (never treat NVFP4 as INT8). Full file follows.


### `patches/comfy_quant_int8.py`

```python
"""
ComfyUI core-safe patches for native comfy_quant INT8 (int8_tensorwise).

Upstream MixedPrecisionOps only quant-loads Linear / Embedding / MoE.
SD UNet INT8 checkpoints also store Conv2d weights as int8 + comfy_quant, which
fails with: Only Tensors of floating point and complex dtype can require gradients.

Also normalizes bare-string / double-encoded comfy_quant JSON some exporters write.

LoRA: native Linear already has convert_weight + set_weight (dequant → bake →
requant, same idea as BobJohnson24/ComfyUI-INT8-Fast). Injected Conv2d must
mirror that set_weight; without it ModelPatcher falls back to rounding into
int8 and LoRA deltas on Conv layers vanish.

Applied from ComfyUI-HSWQ-Loader-and-Tools so ComfyUI core updates do not wipe it.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import threading

logger = logging.getLogger(__name__)
_PATCHES_APPLIED = False

# LoRA bake path logs (rate-limited so console stays readable)
_LORA_CONVERT_LOG_MAX = 0  # quiet; Status dump is enough
_LORA_SET_LOG_MAX = 0
_LORA_PATCHER_LOG_MAX = 0  # per-key bake lines off; Status dump is enough
_lora_convert_logs = 0
_lora_set_logs = 0
_lora_patcher_logs = 0
_lora_patcher_stats = {
    "calls": 0,
    "with_set_func": 0,
    "without_set_func": 0,
    "with_convert_func": 0,
}

# LoRA key attach / skip accounting (last load_lora_for_models call)
_lora_attach_last = {
    "lora_name": "",
    "strength_model": None,
    "strength_clip": None,
    "lora_file_keys": 0,
    "mapped_keys": 0,
    "applied_unet": 0,
    "applied_clip": 0,
    "applied_unet_keys": [],
    "applied_clip_keys": [],
    "not_mapped": [],
    "mapped_but_not_attached": [],
    "add_patches_skipped_unet": [],
}
# One entry per load_lora_for_models call (stacked loaders → multiple entries)
_lora_attach_history = []
# key -> "requant" | "int8_round" recorded during bake
_lora_bake_by_key = {}
# Set by LoraLoader.load_lora wrap (and cleared after attach)
_current_lora_name = None
_current_lora_strength_model = None
_current_lora_strength_clip = None
_lora_shape_skips = []  # list of (lora_name, key, reason)
_LORA_SKIP_PRINT_MAX = 40


def _console(msg: str) -> None:
    """Always visible in ComfyUI console (print + INFO)."""
    print(msg, flush=True)
    logger.info(msg)


def record_lora_shape_skip(lora_name: str, key: str, reason: str) -> None:
    """Called from LoraDiff reshape/numel skip path."""
    _lora_shape_skips.append((str(lora_name), str(key), str(reason)))


def _basename_lora(name: str) -> str:
    if not name:
        return name
    return os.path.basename(str(name).replace("\\", "/"))


# WeightAdapterBase class attrs — NOT filenames (was the lora=lora bug)
_ADAPTER_TYPE_NAMES = frozenset({"lora", "loha", "lokr", "oft", "boft", "glora"})


def _looks_like_lora_filename(s) -> bool:
    if not isinstance(s, str):
        return False
    s = s.strip()
    if not s or s.lower() in _ADAPTER_TYPE_NAMES:
        return False
    low = s.lower()
    if low.endswith((".safetensors", ".pt", ".ckpt", ".bin", ".sft")):
        return True
    if "/" in s or "\\" in s:
        return True
    # Short folder-relative names without extension still count as filenames
    if len(s) >= 2 and not s.startswith("diffusion_model"):
        return True
    return False


def _lora_line(msg: str) -> None:
    """One visible console line (print only — no print+logger twin)."""
    print(msg, flush=True)


def _slot_skip_count(entry: dict) -> int:
    return len(entry.get("not_mapped") or []) + len(
        entry.get("mapped_but_not_attached") or []
    )


def _slot_applied_count(entry: dict) -> int:
    return int(entry.get("applied_unet") or 0) + int(entry.get("applied_clip") or 0)


def _format_lora_slot_line(slot_i: int, entry: dict, include_bake: bool = False) -> str:
    """lora_name / applied_keys / skipped_keys — always present."""
    name = entry.get("lora_name") or "(unknown)"
    sm = entry.get("strength_model")
    sc = entry.get("strength_clip")
    u = int(entry.get("applied_unet") or 0)
    c = int(entry.get("applied_clip") or 0)
    applied = u + c
    skip = _slot_skip_count(entry)
    parts = [
        f"Slot {slot_i}:",
        f"lora_name='{name}'",
        f"applied_keys={applied} (unet={u} clip={c})",
        f"skipped_keys={skip}",
    ]
    if sm is not None:
        parts.append(f"strength_model={sm}")
    if sc is not None:
        parts.append(f"strength_clip={sc}")
    if include_bake:
        verdict, rq, ir, nb = _per_lora_bake_verdict(entry)
        parts.append(f"bake rq={rq} ir={ir} nb={nb}")
        if verdict == "OK_requant":
            parts.append("→ APPLIED ✓")
        elif verdict == "BROKEN_int8_round":
            parts.append("→ BROKEN ✗")
        elif verdict == "N/A_CLIP_only":
            parts.append("→ CLIP_only ✓")
        else:
            parts.append(f"→ {verdict}")
    else:
        if applied > 0:
            parts.append("→ APPLIED ✓")
        else:
            parts.append("→ SKIPPED ✗")
    return f"[HSWQ LoRA Status] {' | '.join(parts)}"


def _log_lora_slot_attach(entry: dict) -> None:
    """Emit one Status line immediately when a LoRA is attached (any loader)."""
    n = len(_lora_attach_history)
    if n == 1:
        _lora_line("[HSWQ LoRA Status] Processing LoRA slot(s):")
    _lora_line(_format_lora_slot_line(n, entry, include_bake=False))
    _lora_line(
        f"[HSWQ LoRA Status]   file_keys={entry.get('lora_file_keys', 0)} "
        f"mapped={entry.get('mapped_keys', 0)} "
        f"not_mapped={len(entry.get('not_mapped') or [])} "
        f"mapped_not_attached={len(entry.get('mapped_but_not_attached') or [])}"
    )


def _set_current_lora_name(name, strength_model=None, strength_clip=None) -> None:
    """Store real filename/UI name; never store adapter type 'lora'."""
    global _current_lora_name, _current_lora_strength_model, _current_lora_strength_clip
    if _looks_like_lora_filename(name):
        _current_lora_name = _basename_lora(name)
    if strength_model is not None:
        _current_lora_strength_model = strength_model
    if strength_clip is not None:
        _current_lora_strength_clip = strength_clip


def _path_is_under_loras_dir(path: str) -> bool:
    """True if path is inside any registered loras/ folder (any loader)."""
    if not path:
        return False
    try:
        import folder_paths

        bases = folder_paths.get_folder_paths("loras") or []
    except Exception:
        bases = []
    norm = os.path.normcase(os.path.abspath(str(path)))
    for base in bases:
        try:
            b = os.path.normcase(os.path.abspath(str(base)))
            if norm == b or norm.startswith(b + os.sep):
                return True
        except Exception:
            continue
    # Fallback when folder list not ready yet
    low = str(path).replace("\\", "/").lower()
    return "/loras/" in low or low.endswith("/loras")


def _resolve_lora_name(loaded_patches=None) -> str:
    """Filename for the LoRA currently being attached (any loader → common hooks)."""
    global _current_lora_name
    if _looks_like_lora_filename(_current_lora_name):
        return _basename_lora(_current_lora_name)

    try:
        import inspect

        # Common local names used by many LoRA loader nodes / helpers
        keys = (
            "lora_name",
            "lora_path",
            "lora",
            "path",
            "filename",
            "file_path",
            "lora_file",
            "name",
        )
        for frame in inspect.stack()[1:24]:
            loc = frame.frame.f_locals
            for key in keys:
                cand = loc.get(key)
                if _looks_like_lora_filename(cand):
                    return _basename_lora(cand)
            # Widget-style dicts: {'lora': '<file>', 'on': True, 'strength': ...}
            for cand in loc.values():
                if not isinstance(cand, dict):
                    continue
                ui = cand.get("lora")
                if _looks_like_lora_filename(ui) and (
                    "strength" in cand or "on" in cand or "strengthTwo" in cand
                ):
                    return _basename_lora(ui)
    except Exception:
        pass

    return f"unknown_lora#{len(_lora_attach_history) + 1}"


def reset_int8_lora_log_counters() -> None:
    global _lora_convert_logs, _lora_set_logs, _lora_patcher_logs
    global _current_lora_name, _current_lora_strength_model, _current_lora_strength_clip
    _lora_convert_logs = 0
    _lora_set_logs = 0
    _lora_patcher_logs = 0
    _lora_patcher_stats.update(
        calls=0, with_set_func=0, without_set_func=0, with_convert_func=0
    )
    _lora_shape_skips.clear()
    _lora_attach_history.clear()
    _lora_bake_by_key.clear()
    _current_lora_name = None
    _current_lora_strength_model = None
    _current_lora_strength_clip = None
    _lora_attach_last.update(
        lora_name="",
        strength_model=None,
        strength_clip=None,
        lora_file_keys=0,
        mapped_keys=0,
        applied_unet=0,
        applied_clip=0,
        applied_unet_keys=[],
        applied_clip_keys=[],
        not_mapped=[],
        mapped_but_not_attached=[],
        add_patches_skipped_unet=[],
    )
    dump_int8_lora_bake_stats._dumped_this_load = False


def summarize_int8_lora_capability(model) -> dict:
    """Scan loaded MODEL / diffusion_model and print LoRA hook readiness."""
    try:
        from comfy.ops import QuantizedTensor
    except ImportError:
        QuantizedTensor = type(None)

    diffusion = model
    # ModelPatcher -> BaseModel -> diffusion_model
    if hasattr(model, "model") and hasattr(model.model, "diffusion_model"):
        diffusion = model.model.diffusion_model
    elif hasattr(model, "diffusion_model"):
        diffusion = model.diffusion_model

    n_lin = n_conv = 0
    lin_set = conv_set = 0
    lin_cvt = conv_cvt = 0
    lin_q = conv_q = 0
    sample_missing = []

    for name, mod in diffusion.named_modules():
        cls = type(mod).__name__
        is_lin = "Linear" in cls
        is_conv = "Conv2d" in cls
        if not is_lin and not is_conv:
            continue
        has_set = callable(getattr(mod, "set_weight", None))
        has_cvt = callable(getattr(mod, "convert_weight", None))
        w = getattr(mod, "weight", None)
        is_q = False
        if QuantizedTensor is not type(None):
            is_q = isinstance(w, QuantizedTensor) or isinstance(
                getattr(w, "data", None), QuantizedTensor
            )
        layout = getattr(mod, "layout_type", None)
        if is_lin:
            n_lin += 1
            lin_set += int(has_set)
            lin_cvt += int(has_cvt)
            lin_q += int(is_q or layout is not None)
        else:
            n_conv += 1
            conv_set += int(has_set)
            conv_cvt += int(has_cvt)
            conv_q += int(is_q or layout is not None)
            if (not has_set or not has_cvt) and len(sample_missing) < 5:
                sample_missing.append(
                    f"{name} set={has_set} convert={has_cvt} layout={layout}"
                )

    _lora_line("[HSWQ INT8 LoRA] ===== load summary =====")
    _lora_line(
        f"[HSWQ INT8 LoRA] Linear: {n_lin}  set_weight={lin_set}  convert_weight={lin_cvt}  quantized/layout={lin_q}"
    )
    _lora_line(
        f"[HSWQ INT8 LoRA] Conv2d: {n_conv}  set_weight={conv_set}  convert_weight={conv_cvt}  quantized/layout={conv_q}"
    )
    if conv_set < n_conv or conv_cvt < n_conv:
        _lora_line(
            "[HSWQ INT8 LoRA] WARN: some Conv2d lack set/convert — LoRA on those layers will round into int8 and die"
        )
        for s in sample_missing:
            _lora_line(f"[HSWQ INT8 LoRA]   missing: {s}")
    else:
        _lora_line(
            "[HSWQ INT8 LoRA] OK: Conv2d has set_weight+convert_weight (dequant -> bake -> requant)"
        )
    _lora_line("[HSWQ INT8 LoRA] =========================")
    return {
        "linear": n_lin,
        "conv2d": n_conv,
        "linear_set_weight": lin_set,
        "conv_set_weight": conv_set,
    }


def decode_comfy_quant_conf(raw):
    """Decode a comfy_quant marker into a dict layer config."""
    import torch

    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if torch.is_tensor(raw):
        conf = json.loads(raw.numpy().tobytes())
    elif isinstance(raw, (bytes, bytearray, memoryview)):
        conf = json.loads(bytes(raw))
    elif isinstance(raw, str):
        conf = raw
    else:
        conf = raw

    while isinstance(conf, str):
        try:
            parsed = json.loads(conf)
        except (TypeError, json.JSONDecodeError):
            return {"format": conf}
        if parsed is conf:
            return {"format": conf}
        conf = parsed

    if isinstance(conf, dict):
        return conf
    raise TypeError(f"comfy_quant config must be a dict or format string, got {type(conf).__name__}")


def checkpoint_looks_like_comfy_quant_int8(state_dict_or_path) -> bool:
    """True if checkpoint has comfy_quant INT8 markers (native MixedPrecisionOps path).

    Accepts a loaded state_dict, or a filesystem path (probes via safetensors without full load).
    """
    import torch

    if isinstance(state_dict_or_path, (str, os.PathLike)):
        return _probe_path_comfy_quant_int8(str(state_dict_or_path))

    state_dict = state_dict_or_path
    has_marker = False
    has_int8 = False
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        if key.endswith(".comfy_quant"):
            has_marker = True
            conf = decode_comfy_quant_conf(value)
            if isinstance(conf, dict) and conf.get("format") == "int8_tensorwise":
                return True
        if key.endswith(".weight") and value.dtype == torch.int8:
            has_int8 = True
    return has_marker and has_int8


def _probe_path_comfy_quant_int8(path: str) -> bool:
    """Lightweight safetensors probe for int8_tensorwise."""
    import torch

    try:
        from safetensors import safe_open
    except ImportError:
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            comfy_keys = [k for k in keys if k.endswith(".comfy_quant")]
            for ck in comfy_keys[:16]:
                conf = decode_comfy_quant_conf(f.get_tensor(ck))
                if isinstance(conf, dict) and conf.get("format") == "int8_tensorwise":
                    return True
            if comfy_keys:
                for k in keys:
                    if not k.endswith(".weight"):
                        continue
                    if f.get_tensor(k).dtype == torch.int8:
                        return True
                    break
            meta = f.metadata() or {}
            if "_quantization_metadata" in meta:
                try:
                    qm = json.loads(meta["_quantization_metadata"])
                    layers = qm.get("layers", {}) if isinstance(qm, dict) else {}
                    for v in layers.values():
                        if isinstance(v, str) and v == "int8_tensorwise":
                            return True
                        if isinstance(v, dict) and v.get("format") == "int8_tensorwise":
                            return True
                except (TypeError, json.JSONDecodeError):
                    pass
    except Exception as e:
        logger.debug("[HSWQ INT8] probe failed for %s: %s", path, e)
        return False
    return False


def _comfy_quant_conf_has_convrot(conf) -> bool:
    if not isinstance(conf, dict):
        return False
    if conf.get("convrot") is True:
        return True
    params = conf.get("params")
    if isinstance(params, dict) and params.get("convrot") is True:
        return True
    return False


def checkpoint_looks_like_comfy_quant_convrot(state_dict_or_path) -> bool:
    """True if checkpoint marks int8_tensorwise layers with ConvRot (Hadamard)."""
    if isinstance(state_dict_or_path, (str, os.PathLike)):
        return _probe_path_comfy_quant_convrot(str(state_dict_or_path))

    state_dict = state_dict_or_path
    import torch

    for key, value in state_dict.items():
        if not key.endswith(".comfy_quant"):
            continue
        if not torch.is_tensor(value) and not isinstance(value, (dict, bytes, bytearray, str)):
            continue
        conf = decode_comfy_quant_conf(value)
        if _comfy_quant_conf_has_convrot(conf):
            return True
    return False


def checkpoint_needs_hswq_int8_conv2d(state_dict_or_path) -> bool:
    """True for SDXL/ZI-style UNets that need HSWQ INT8 Conv2d patches.

    Keyed off architecture (``input_blocks`` / ``middle_block`` / ``output_blocks``),
    not off ConvRot. DiT/Krea2 (``double_blocks`` / ``single_blocks``) returns False
    so ConvRot stock load stays free of our Conv2d inject (VRAM).
    """
    if isinstance(state_dict_or_path, (str, os.PathLike)):
        return _probe_path_needs_hswq_int8_conv2d(str(state_dict_or_path))

    keys = list(state_dict_or_path.keys())
    return _keys_need_hswq_int8_conv2d(keys)


def _keys_need_hswq_int8_conv2d(keys) -> bool:
    sdxl = False
    dit = False
    for k in keys:
        if (
            ".input_blocks." in k
            or ".middle_block." in k
            or ".output_blocks." in k
            or k.startswith("input_blocks.")
            or k.startswith("middle_block.")
            or k.startswith("output_blocks.")
        ):
            sdxl = True
        if (
            ".double_blocks." in k
            or ".single_blocks." in k
            or ".joint_blocks." in k
            or k.startswith("double_blocks.")
            or k.startswith("single_blocks.")
            or k.startswith("joint_blocks.")
        ):
            dit = True
        if sdxl and dit:
            break
    # Prefer SDXL Conv2d path when UNet blocks exist; DiT-only → no inject.
    if sdxl:
        return True
    return False


def _probe_path_needs_hswq_int8_conv2d(path: str) -> bool:
    try:
        from safetensors import safe_open
    except ImportError:
        # Filename heuristics only as last resort.
        base = os.path.basename(path).lower()
        if "krea" in base or "dit" in base:
            return False
        return True
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            return _keys_need_hswq_int8_conv2d(list(f.keys()))
    except Exception as e:
        logger.debug("[HSWQ INT8] SDXL/ZI Conv2d need probe failed for %s: %s", path, e)
        base = os.path.basename(path).lower()
        if "krea" in base or "convrot" in base or "int8convrot" in base:
            return False
        return True


def _probe_path_comfy_quant_convrot(path: str) -> bool:
    """Lightweight safetensors probe for comfy_quant.convrot=true."""
    try:
        from safetensors import safe_open
    except ImportError:
        return "convrot" in os.path.basename(path).lower()
    base = os.path.basename(path).lower()
    name_hint = "convrot" in base or "int8convrot" in base
    comfy_keys = []
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            comfy_keys = [k for k in keys if k.endswith(".comfy_quant")]
            for ck in comfy_keys[:32]:
                conf = decode_comfy_quant_conf(f.get_tensor(ck))
                if _comfy_quant_conf_has_convrot(conf):
                    return True
            meta = f.metadata() or {}
            if "_quantization_metadata" in meta:
                try:
                    qm = json.loads(meta["_quantization_metadata"])
                    layers = qm.get("layers", {}) if isinstance(qm, dict) else {}
                    for v in layers.values():
                        if isinstance(v, dict) and _comfy_quant_conf_has_convrot(v):
                            return True
                except (TypeError, json.JSONDecodeError):
                    pass
    except Exception as e:
        logger.debug("[HSWQ INT8] ConvRot probe failed for %s: %s", path, e)
        return name_hint
    # Filename alone is enough for *Int8Convrot* when markers were stripped/odd.
    return name_hint


def _normalize_comfy_quant_tensor(value):
    import torch

    conf = decode_comfy_quant_conf(value)
    if conf is None:
        return None
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def _patch_convert_old_quants() -> bool:
    try:
        import torch
        import comfy.utils as utils_module
    except ImportError:
        return False

    original = getattr(utils_module, "convert_old_quants", None)
    if original is None or getattr(original, "_hswq_int8_patched", False):
        return False

    def convert_old_quants_pre(state_dict, model_prefix="", metadata=None):
        if metadata is None:
            metadata = {}
        # Normalize string layer configs in metadata before upstream json.dumps(v).
        if isinstance(metadata, dict) and "_quantization_metadata" in metadata:
            try:
                quant_meta = json.loads(metadata["_quantization_metadata"])
            except (TypeError, json.JSONDecodeError):
                quant_meta = None
            if isinstance(quant_meta, dict) and isinstance(quant_meta.get("layers"), dict):
                layers = quant_meta["layers"]
                changed = False
                for k, v in list(layers.items()):
                    if isinstance(v, str):
                        layers[k] = {"format": v}
                        changed = True
                    elif not isinstance(v, dict):
                        raise TypeError(
                            f"quantization layer config for {k} must be dict or format string, got {type(v).__name__}"
                        )
                if changed:
                    metadata = dict(metadata)
                    metadata["_quantization_metadata"] = json.dumps(quant_meta)

        state_dict, metadata = original(state_dict, model_prefix=model_prefix, metadata=metadata)

        # Re-normalize any .comfy_quant tensors (file-embedded or metadata-written).
        for key in list(state_dict.keys()):
            if not key.endswith(".comfy_quant"):
                continue
            normalized = _normalize_comfy_quant_tensor(state_dict[key])
            if normalized is None:
                state_dict.pop(key, None)
            else:
                state_dict[key] = normalized
        return state_dict, metadata

    convert_old_quants_pre._hswq_int8_patched = True
    utils_module.convert_old_quants = convert_old_quants_pre
    return True


def _quant_config_has_int8_tensorwise(quant_config) -> bool:
    """True if MixedPrecisionOps quant_config targets int8_tensorwise layers."""
    if not isinstance(quant_config, dict) or not quant_config:
        return False
    for v in quant_config.values():
        if isinstance(v, dict) and v.get("format") == "int8_tensorwise":
            return True
        if v == "int8_tensorwise":
            return True
    return False


# INT8 Conv2d inject must NOT run for FP MixedPrecisionOps.
# detect_layer_quantization() only returns {"mixed_ops": True} for both INT8 and FP8,
# so we gate Conv2d injection on this load-scoped flag (set only in INT8 load helpers).
_int8_quant_conv_tls = threading.local()


@contextlib.contextmanager
def _int8_quant_conv_scope():
    prev = getattr(_int8_quant_conv_tls, "active", False)
    _int8_quant_conv_tls.active = True
    try:
        yield
    finally:
        _int8_quant_conv_tls.active = prev


def _should_inject_int8_conv(quant_config) -> bool:
    # Only while an HSWQ INT8 UNet/Checkpoint load explicitly opens the scope.
    # Do NOT key off quant_config alone: once mixed_precision_ops is monkeypatched,
    # stock UNETLoader / Krea2 ConvRot loads also build MixedPrecisionOps with
    # int8_tensorwise config — injecting our Conv2d there is wrong for DiT/ConvRot
    # and can inflate VRAM vs stock.
    _ = quant_config
    return bool(getattr(_int8_quant_conv_tls, "active", False))


def _module_path_is_real_nunchaku_package(mod: str) -> bool:
    """True only for real Nunchaku package modules — never this unofficial-loader.

    INT8 Conv2d from this extension lives under a path containing ``nunchaku``;
    a bare ``\"nunchaku\" in path`` false-positive armed VRAM handoff on
    non-SVDQ loads (SDXL INT8 and any other architecture using those Conv2d)
    and destroyed normal generation. Substring match is forbidden.
    """
    mod_l = (mod or "").lower().replace("\\", "/")
    if not mod_l:
        return False
    # This extension / INT8 patch path must never count as SVDQ.
    if (
        "unofficial" in mod_l
        or "comfy_quant_int8" in mod_l
        or "nunchaku-unofficial" in mod_l
        or "nunchaku_unofficial" in mod_l
    ):
        return False
    if mod_l == "nunchaku" or mod_l.startswith("nunchaku."):
        return True
    if ".nunchaku." in mod_l:
        return True
    return False


def _model_is_nunchaku_svdq(model) -> bool:
    """True only when the graph carries real Nunchaku SVDQ modules.

    ComfyUI registers Z-Image as ``Lumina2`` — classname checks for
    ``Nunchaku`` / ``ZImage`` miss that. Any SVDQ / ComfyNunchaku module means
    never run comfy_quant INT8 Dynamic LoRA bake.

    Branch: everything that is not real SVDQ (SDXL, Flux, ZIT, native INT8,
    FP, …) returns False. Module-path checks must not match this
    unofficial-loader package (see ``_module_path_is_real_nunchaku_package``).
    """
    if model is None:
        return False
    roots = [model]
    dm = getattr(model, "diffusion_model", None)
    if dm is not None:
        roots.append(dm)
    inner = getattr(model, "model", None)
    if inner is not None and inner is not model:
        roots.append(inner)
        dm2 = getattr(inner, "diffusion_model", None)
        if dm2 is not None:
            roots.append(dm2)
    seen = set()
    for root in roots:
        rid = id(root)
        if rid in seen:
            continue
        seen.add(rid)
        try:
            named = root.named_modules()
        except Exception:
            continue
        for _, module in named:
            cls_name = type(module).__name__
            if (
                "SVDQ" in cls_name
                or "Nunchaku" in cls_name
                or cls_name.startswith("ComfyNunchaku")
            ):
                return True
            mod = getattr(type(module), "__module__", "") or ""
            if _module_path_is_real_nunchaku_package(mod):
                return True
    return False


def _qt_is_int8_tensorwise(weight, QuantizedTensor) -> bool:
    """True only for comfy_quant ``int8_tensorwise`` QT (not NVFP4 / FP8 / W4A4).

    Mixed kitchen packs (nvfp4 Linear + int8protect) expose both QT layouts.
    Dynamic INT8 LoRA bake must never requant NVFP4 via this path.
    """
    qt = _qt_payload(weight, QuantizedTensor) if weight is not None else None
    if qt is None:
        if isinstance(weight, QuantizedTensor):
            qt = weight
        else:
            return False
    layout = getattr(qt, "layout", None)
    if layout is None:
        layout = getattr(qt, "_layout", None)
    name = type(layout).__name__ if layout is not None else ""
    # Registered layout: TensorWiseINT8Layout / kitchen _CKTensorWiseINT8Layout
    return "TensorWiseINT8" in name


def _model_has_int8_quantized_weights(model) -> bool:
    """True only for native comfy_quant INT8 (int8_tensorwise QuantizedTensor).

    Must NOT treat bare ``torch.int8`` weights as comfy_quant INT8.
    Must NOT treat NVFP4 / FP8 QT as INT8 — that arms Dynamic LoRA bake on
    ConvRot NVFP4 layers and destroys quality.
    Nunchaku SVDQ / Z-Image / Lumina2 modules often use int8 storage; a false
    positive here arms Dynamic.load INT8 LoRA bake and can Abort those paths.
    """
    if _model_is_nunchaku_svdq(model):
        return False
    try:
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return False
    for _, module in model.named_modules():
        cls_name = type(module).__name__
        if "SVDQ" in cls_name or "Nunchaku" in cls_name:
            continue
        w = getattr(module, "weight", None)
        if w is None:
            continue
        if _qt_is_int8_tensorwise(w, QuantizedTensor):
            return True
    return False


def _load_native_convert_int8_helpers():
    """Lazy-load Hadamard / rotate helpers from sibling native_convert_int8.py."""
    import importlib.util

    global _NATIVE_CONVERT_INT8_MOD
    if _NATIVE_CONVERT_INT8_MOD is not None:
        return _NATIVE_CONVERT_INT8_MOD
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "native_convert_int8.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"native_convert_int8.py not found: {path}")
    name = "native_convert_int8_for_hswq_conv2d"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _NATIVE_CONVERT_INT8_MOD = mod
    return mod


_NATIVE_CONVERT_INT8_MOD = None


def _qt_payload(weight, QuantizedTensor):
    """Unwrap Parameter → QuantizedTensor if needed."""
    if weight is None:
        return None
    if isinstance(weight, QuantizedTensor):
        return weight
    data = getattr(weight, "data", None)
    if isinstance(data, QuantizedTensor):
        return data
    return None


def _arm_hswq_conv2d_convrot(module, QuantizedTensor):
    """Full ConvRot on Conv2d: keep online rotate on module; clear kitchen Params.convrot.

    Kitchen dequantize_int8_convrot_* is 2D-only. Stamping Params.convrot=True on
    4D weights and calling .dequantize() crashes. Weights stay in rotated basis;
    forward rotates NCHW activations; LoRA convert_weight unrotates to float space.
    """
    import dataclasses

    import torch

    qt = _qt_payload(getattr(module, "weight", None), QuantizedTensor)
    if qt is None:
        return
    params = getattr(qt, "_params", None)
    qdata = getattr(qt, "_qdata", None)
    if params is None or qdata is None:
        return
    if getattr(qdata, "ndim", None) != 4:
        return
    if not bool(getattr(params, "convrot", False)):
        return

    gs = int(getattr(params, "convrot_groupsize", 256) or 256)
    module._hswq_convrot = True
    module._hswq_convrot_groupsize = gs
    new_params = dataclasses.replace(params, convrot=False)
    # Prefer in-place params swap. Reconstructing QT needs layout *string*
    # (_layout_cls), not a layout object — wrong arg → empty AssertionError.
    try:
        object.__setattr__(qt, "_params", new_params)
        return
    except Exception:
        pass
    try:
        qt._params = new_params
        return
    except Exception:
        pass
    layout_cls = getattr(qt, "_layout_cls", None)
    if not isinstance(layout_cls, str):
        layout_cls = getattr(module, "layout_type", None)
    if not isinstance(layout_cls, str):
        return
    new_qt = type(qt)(qdata, layout_cls, new_params)
    module.weight = torch.nn.Parameter(new_qt, requires_grad=False)


def _make_quantized_conv2d(ops_module, MixedPrecisionOps, disabled):
    """Build MixedPrecisionOps.Conv2d class using current comfy.ops helpers."""
    import torch

    CastWeightBiasOp = ops_module.CastWeightBiasOp
    QuantizedTensor = ops_module.QuantizedTensor
    cast_bias_weight = ops_module.cast_bias_weight
    uncast_bias_weight = ops_module.uncast_bias_weight
    run_every_op = ops_module.run_every_op
    _load_quantized_module = ops_module._load_quantized_module
    _quantized_weight_state_dict = ops_module._quantized_weight_state_dict
    _quantized_apply = ops_module._quantized_apply

    class Conv2d(torch.nn.Module, CastWeightBiasOp):
        _disabled_formats = disabled
        _hswq_quant_conv2d = True

        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
            device=None,
            dtype=None,
        ):
            super().__init__()
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            if isinstance(stride, int):
                stride = (stride, stride)
            if isinstance(padding, int):
                padding = (padding, padding)
            if isinstance(dilation, int):
                dilation = (dilation, dilation)

            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.groups = groups
            self.padding_mode = padding_mode
            self.factory_kwargs = {"device": device, "dtype": MixedPrecisionOps._compute_dtype}
            self._orig_shape = (out_channels, in_channels // groups, kernel_size[0], kernel_size[1])

            if bias:
                self.bias = torch.nn.Parameter(
                    torch.empty(out_channels, **self.factory_kwargs), requires_grad=False
                )
            else:
                self.register_parameter("bias", None)

            self.weight = None
            self.quant_format = None
            self.layout_type = None
            self._full_precision_mm = MixedPrecisionOps._full_precision_mm
            self._full_precision_mm_config = False
            self._hswq_convrot = False
            self._hswq_convrot_groupsize = 256

        def reset_parameters(self):
            return None

        def _load_from_state_dict(self, *args):
            _load_quantized_module(self, super()._load_from_state_dict, *args, load_extra_params=False)
            _arm_hswq_conv2d_convrot(self, QuantizedTensor)

        def state_dict(self, *args, destination=None, prefix="", **kwargs):
            sd = destination if destination is not None else {}
            sd = _quantized_weight_state_dict(self, sd, prefix)
            # Re-stamp ConvRot on export (Params.convrot cleared for safe 4D dequant).
            if getattr(self, "_hswq_convrot", False):
                cq_key = f"{prefix}comfy_quant"
                conf = {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": int(
                        getattr(self, "_hswq_convrot_groupsize", 256) or 256
                    ),
                }
                sd[cq_key] = torch.tensor(
                    list(json.dumps(conf, separators=(",", ":")).encode("utf-8")),
                    dtype=torch.uint8,
                )
            return sd

        def _conv_forward(self, input, weight, bias):
            if self.padding_mode != "zeros":
                return torch.nn.functional.conv2d(
                    torch.nn.functional.pad(
                        input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                    ),
                    weight,
                    bias,
                    self.stride,
                    (0, 0),
                    self.dilation,
                    self.groups,
                )
            return torch.nn.functional.conv2d(
                input, weight, bias, self.stride, self.padding, self.dilation, self.groups
            )

        def forward_comfy_cast_weights(self, input):
            # Mirror MixedPrecision Linear: when weight is QuantizedTensor and
            # Dynamic VRAM uses weight_lowvram_function, want_requant=True so
            # post_cast dequant → LoRA → requant (want_requant=False left QT
            # in the resident path after the first step and killed LoRA).
            if getattr(self, "_hswq_convrot", False):
                nc = _load_native_convert_int8_helpers()
                gs = int(getattr(self, "_hswq_convrot_groupsize", 256) or 256)
                h = nc.build_hadamard(gs, device="cpu", dtype=torch.float32)
                input = nc.rotate_activation_nchw(input, h, gs)
            want_requant = isinstance(getattr(self, "weight", None), QuantizedTensor)
            weight, bias, offload_stream = cast_bias_weight(
                self,
                input,
                offloadable=True,
                compute_dtype=getattr(input, "dtype", None),
                want_requant=want_requant,
            )
            x = self._conv_forward(input, weight, bias)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return x

        def forward(self, input, *args, **kwargs):
            run_every_op()
            return self.forward_comfy_cast_weights(input)

        def convert_weight(self, weight, inplace=False, **kwargs):
            # Same contract as MixedPrecisionOps.Linear: LoRA / ModelPatcher
            # dequant → calculate_weight → set_weight (see ComfyUI-INT8-Fast bake path).
            # ConvRot weights are stored rotated; unrotate to original float basis for LoRA.
            # LowVRAM may re-materialize QT with Params.convrot still True — clear
            # before dequantize (kitchen ConvRot dequant is 2D-only).
            global _lora_convert_logs
            if isinstance(weight, QuantizedTensor):
                _arm_hswq_conv2d_convrot(self, QuantizedTensor)
                qt = _qt_payload(weight, QuantizedTensor)
                if qt is not None:
                    params = getattr(qt, "_params", None)
                    qdata = getattr(qt, "_qdata", None)
                    if (
                        params is not None
                        and qdata is not None
                        and getattr(qdata, "ndim", None) == 4
                        and bool(getattr(params, "convrot", False))
                    ):
                        import dataclasses

                        gs = int(getattr(params, "convrot_groupsize", 256) or 256)
                        self._hswq_convrot = True
                        self._hswq_convrot_groupsize = gs
                        new_params = dataclasses.replace(params, convrot=False)
                        try:
                            object.__setattr__(qt, "_params", new_params)
                        except Exception:
                            qt._params = new_params
                out = weight.dequantize()
            else:
                out = weight
            if getattr(self, "_hswq_convrot", False) and out is not None and getattr(out, "ndim", 0) == 4:
                nc = _load_native_convert_int8_helpers()
                gs = int(getattr(self, "_hswq_convrot_groupsize", 256) or 256)
                h = nc.build_hadamard(gs, device="cpu", dtype=torch.float32)
                out = nc.unrotate_weight_conv2d(out, h, gs)
            if _lora_convert_logs < _LORA_CONVERT_LOG_MAX:
                _lora_convert_logs += 1
                wdtype = getattr(weight, "dtype", None)
                odtype = getattr(out, "dtype", None)
                _console(
                    f"[HSWQ INT8 LoRA] Conv2d.convert_weight #{_lora_convert_logs}: "
                    f"in={type(weight).__name__}/{wdtype} -> out={type(out).__name__}/{odtype} "
                    f"layout={getattr(self, 'layout_type', None)} "
                    f"convrot={getattr(self, '_hswq_convrot', False)}"
                )
            return out

        def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
            # Mirror MixedPrecisionOps.Linear.set_weight so Conv2d LoRA bake
            # does not fall through to stochastic_rounding(..., int8), which
            # destroys float LoRA deltas (INT8-Fast: normal LoRA loader works).
            # ConvRot: convert_weight returned unrotated float; re-rotate before requant.
            global _lora_set_logs
            layout = getattr(self, "layout_type", None)
            path = "requant" if layout is not None else "cast_only"
            if getattr(self, "_hswq_convrot", False) and getattr(weight, "ndim", 0) == 4:
                nc = _load_native_convert_int8_helpers()
                gs = int(getattr(self, "_hswq_convrot_groupsize", 256) or 256)
                h = nc.build_hadamard(gs, device="cpu", dtype=torch.float32)
                weight = nc.rotate_weight_conv2d(weight, h, gs)
            if _lora_set_logs < _LORA_SET_LOG_MAX:
                _lora_set_logs += 1
                _console(
                    f"[HSWQ INT8 LoRA] Conv2d.set_weight #{_lora_set_logs}: "
                    f"path={path} float_in={getattr(weight, 'dtype', None)} "
                    f"shape={tuple(weight.shape) if hasattr(weight, 'shape') else '?'} "
                    f"seed={seed} layout={layout} "
                    f"convrot={getattr(self, '_hswq_convrot', False)}"
                )
            if layout is not None:
                weight = self.weight.requantize_from_float(
                    weight,
                    scale="recalculate",
                    stochastic_rounding=seed,
                    inplace_ops=True,
                ).to(self.weight.dtype)
            else:
                weight = weight.to(self.weight.dtype)
            if return_weight:
                return weight

            assert inplace_update is False
            self.weight = torch.nn.Parameter(weight, requires_grad=False)

        def _apply(self, fn, recurse=True):
            return _quantized_apply(self, fn, recurse)

        @property
        def _reversed_padding_repeated_twice(self):
            return tuple(x for x in reversed(self.padding) for _ in range(2))

    return Conv2d


def _patch_ops_decode_and_conv() -> bool:
    try:
        import comfy.ops as ops_module
    except ImportError:
        return False

    ops_module._decode_comfy_quant_conf = decode_comfy_quant_conf

    original_load = getattr(ops_module, "_load_quantized_module", None)
    if original_load is None:
        return False

    if not getattr(original_load, "_hswq_int8_decode_patched", False):

        def _load_quantized_module_patched(
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
            key = f"{prefix}comfy_quant"
            if key in state_dict:
                normalized = _normalize_comfy_quant_tensor(state_dict[key])
                if normalized is None:
                    state_dict.pop(key, None)
                else:
                    state_dict[key] = normalized
            return original_load(
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

        _load_quantized_module_patched._hswq_int8_decode_patched = True
        ops_module._load_quantized_module = _load_quantized_module_patched

    # Also normalize Embedding's direct json.loads path by wrapping Embedding._load_from_state_dict
    # is covered if convert_old_quants + file markers are normalized; keep load wrapper as safety.

    original_mp = getattr(ops_module, "mixed_precision_ops", None)
    if original_mp is None or not callable(original_mp):
        return False
    _OPS_PATCH_VER = 4  # Conv2d full ConvRot: online act rotate + safe 4D dequant
    true_orig = getattr(original_mp, "_hswq_orig_mixed_precision_ops", original_mp)
    if (
        getattr(original_mp, "_hswq_int8_ops_ver", 0) >= _OPS_PATCH_VER
        and getattr(original_mp, "_hswq_int8_conv_patched", False)
    ):
        return True

    def mixed_precision_ops_force_conv(
        quant_config=None, compute_dtype=None, full_precision_mm=False, disabled=None
    ):
        if quant_config is None:
            quant_config = {}
        if compute_dtype is None:
            import torch

            compute_dtype = torch.bfloat16
        if disabled is None:
            disabled = []
        result = true_orig(
            quant_config=quant_config,
            compute_dtype=compute_dtype,
            full_precision_mm=full_precision_mm,
            disabled=disabled,
        )
        # Inject Quantized Conv2d only during HSWQ INT8 load scope
        # (_int8_quant_conv_scope). Never from quant_config alone — that would
        # also hit stock UNETLoader / Krea2 ConvRot MixedPrecision builds.
        if _should_inject_int8_conv(quant_config):
            result.Conv2d = _make_quantized_conv2d(ops_module, result, disabled)
        return result

    mixed_precision_ops_force_conv._hswq_orig_mixed_precision_ops = true_orig
    mixed_precision_ops_force_conv._hswq_int8_conv_patched = True
    mixed_precision_ops_force_conv._hswq_int8_ops_ver = _OPS_PATCH_VER
    ops_module.mixed_precision_ops = mixed_precision_ops_force_conv
    return True


def _patch_lowvram_patch_float_intermediate() -> bool:
    """Fix LowVramPatch intermediate_dtype for comfy_quant QuantizedTensor only.

    Upstream LowVramPatch passes intermediate_dtype=weight.dtype. When the
    weight is still a QuantizedTensor (int8 storage), LoRA matmul casts to
    int8 and either errors or silently produces a no-op delta — same bug as
    BobJohnson24/ComfyUI-INT8-Fast#76.

    Must NOT divert bare ``torch.int8`` tensors. Nunchaku SVDQ / Lumina2 use
    int8 storage; grabbing them here corrupts fused CUDA (Abort in
    ``_forward_silu_gating``) even when VRAM handoff already freed GPU memory.
    """
    try:
        import torch
        import comfy.lora
        import comfy.model_patcher as mp
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return False

    LowVramPatch = getattr(mp, "LowVramPatch", None)
    if LowVramPatch is None:
        return False
    original = getattr(LowVramPatch, "__call__", None)
    _LV_VER = 3
    if original is None or getattr(original, "_hswq_int8_lora_dtype_ver", 0) >= _LV_VER:
        return getattr(original, "_hswq_int8_lora_dtype", False)
    true_orig = getattr(original, "_hswq_orig_lowvram_call", original)

    def __call__(self, weight):
        # QuantizedTensor only. Bare int8 / float / None → upstream unchanged.
        if weight is None or not isinstance(weight, QuantizedTensor):
            return true_orig(self, weight)
        patches = (
            self.prepared_patches
            if self.prepared_patches is not None
            else self.patches[self.key]
        )
        w = weight.dequantize()
        dtype = getattr(w, "dtype", None)
        if dtype is not None and hasattr(dtype, "is_floating_point") and dtype.is_floating_point:
            idtype = dtype
        else:
            idtype = torch.float32
        return comfy.lora.calculate_weight(patches, w, self.key, intermediate_dtype=idtype)

    __call__._hswq_int8_lora_dtype = True
    __call__._hswq_int8_lora_dtype_ver = _LV_VER
    __call__._hswq_orig_lowvram_call = true_orig
    LowVramPatch.__call__ = __call__
    return True


def _get_baked_key_set(model) -> set:
    s = getattr(model, "_hswq_int8_baked_keys", None)
    if s is None:
        s = set()
        model._hswq_int8_baked_keys = s
    return s


def _maybe_invalidate_baked_keys(patcher) -> None:
    """If patches_uuid changed (new LoRA), allow those keys to be baked again."""
    model = patcher.model
    baked_uuid = getattr(model, "_hswq_int8_baked_uuid", None)
    cur = getattr(patcher, "patches_uuid", None)
    if baked_uuid is None or cur is None:
        return
    if baked_uuid != cur and patcher.patches:
        _get_baked_key_set(model).clear()
        model._hswq_int8_baked_uuid = None


def _strip_lowvram_for_baked_keys(patcher) -> int:
    """Dynamic.load re-attaches LowVramPatch; clear it for already-baked keys.

    Shared modules keep their VBAR ``_v`` across loads. Re-attaching LoRA on
    top of baked INT8 weights would double-apply; clearing lowvram avoids that.
    """
    _maybe_invalidate_baked_keys(patcher)
    baked = getattr(patcher.model, "_hswq_int8_baked_keys", None)
    if not baked:
        return 0
    cleared = 0
    for name, module in patcher.model.named_modules():
        for param_key in ("weight", "bias"):
            key = f"{name}.{param_key}"
            if key not in baked:
                continue
            attr = param_key + "_lowvram_function"
            if getattr(module, attr, None) is not None:
                setattr(module, attr, None)
                cleared += 1
            # Drop from this patcher's dict so later loads do not re-attach
            if key in patcher.patches:
                try:
                    del patcher.patches[key]
                except KeyError:
                    pass
    return cleared


def _bake_int8_patches_on_dynamic_patcher(patcher, device_to) -> int:
    """Bake LoRA into INT8 modules after ModelPatcherDynamic.load.

    Dynamic VRAM attaches LowVramPatch on weight_lowvram_function and asserts
    force_patch_weights=False. For comfy_quant INT8 that path often leaves
    LoRA attached in the patcher dict but visually inert (keys count OK,
    bake logs absent). We bake via convert_weight/set_weight (requant).

    Critical VBAR rule (2nd-gen FaceDetailer OOM):
      ModelVBAR.alloc is a bump allocator (offset only grows). Deleting
      module._v after bake makes the next load call alloc() again → VBAR OOM.
      Keep ``_v``. Clear LowVramPatch, bake, then pop patches + drop the
      pre-bake backup entry so restore_loaded_backups does not undo bake.
    """
    if _model_is_nunchaku_svdq(getattr(patcher, "model", None)):
        return 0
    if not getattr(patcher, "patches", None):
        return 0
    try:
        import comfy.model_patcher as mp
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return 0

    _maybe_invalidate_baked_keys(patcher)
    already = _get_baked_key_set(patcher.model)
    baked = 0
    for name, module in patcher.model.named_modules():
        keys_to_bake = []
        for param_key in ("weight", "bias"):
            key = f"{name}.{param_key}"
            if key not in patcher.patches:
                continue
            if key in already:
                # Already baked under this patches_uuid; clear re-attached LowVramPatch
                attr = param_key + "_lowvram_function"
                if getattr(module, attr, None) is not None:
                    setattr(module, attr, None)
                try:
                    del patcher.patches[key]
                except KeyError:
                    pass
                continue
            weight, set_func, convert_func = mp.get_key_weight(patcher.model, key)
            if weight is None:
                continue
            # Bake only int8_tensorwise QT — never bare int8 (Nunchaku), never NVFP4/FP8.
            if not _qt_is_int8_tensorwise(weight, QuantizedTensor):
                continue
            if set_func is None:
                _console(
                    f"[HSWQ INT8 LoRA] WARN cannot bake {key}: "
                    "QuantizedTensor but no set_weight (int8_round risk)"
                )
                continue
            keys_to_bake.append((param_key, key))

        if not keys_to_bake:
            continue

        # Clear LowVramPatch so bake uses Parameter + set_weight, not lazy patch.
        # Do NOT unpin/delete module._v — that causes 2nd-load VBAR OOM.
        for param_key, _key in keys_to_bake:
            if hasattr(module, param_key + "_lowvram_function"):
                setattr(module, param_key + "_lowvram_function", None)

        for _param_key, key in keys_to_bake:
            patcher.patch_weight_to_device(key, device_to=device_to)
            # Drop pre-bake backup so the next Dynamic.load restore keeps baked weights
            if key in patcher.backup:
                try:
                    del patcher.backup[key]
                except KeyError:
                    pass
            try:
                del patcher.patches[key]
            except KeyError:
                pass
            already.add(key)
            baked += 1

    if baked > 0:
        patcher.model._hswq_int8_baked_uuid = getattr(patcher, "patches_uuid", None)

    return baked


def _patch_model_patcher_dynamic_int8_lora_bake() -> bool:
    """After ModelPatcherDynamic.load, bake INT8 LoRA via set_weight."""
    try:
        import comfy.model_patcher as mp
    except ImportError:
        return False

    Dynamic = getattr(mp, "ModelPatcherDynamic", None)
    if Dynamic is None:
        return False
    original = getattr(Dynamic, "load", None)
    if original is None:
        return False
    _DYN_VER = 6
    if getattr(original, "_hswq_int8_lora_bake_ver", 0) >= _DYN_VER:
        return True
    true_orig = getattr(original, "_hswq_orig_dynamic_load", original)

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False, dirty=False):
        result = true_orig(
            self,
            device_to=device_to,
            lowvram_model_memory=lowvram_model_memory,
            force_patch_weights=force_patch_weights,
            full_load=full_load,
            dirty=dirty,
        )
        # INT8 LoRA bake only — never touch Nunchaku SVDQ (class is often Lumina2).
        if _model_is_nunchaku_svdq(self.model):
            return result
        if not _model_has_int8_quantized_weights(self.model) and not getattr(
            self.model, "_hswq_int8_baked_keys", None
        ):
            return result
        # Load re-attaches LowVramPatch for any keys still in patches / clones
        _strip_lowvram_for_baked_keys(self)
        if self.patches:
            n = _bake_int8_patches_on_dynamic_patcher(self, device_to=device_to)
            if n > 0 or _lora_attach_history or (_lora_attach_last.get("mapped_keys") or 0) > 0:
                dump_int8_lora_bake_stats(force=True)
        elif _lora_attach_history or (_lora_attach_last.get("mapped_keys") or 0) > 0:
            # Patches already consumed by a prior bake; still emit Status once if needed
            dump_int8_lora_bake_stats(force=False)
        return result

    load._hswq_int8_lora_bake = True
    load._hswq_int8_lora_bake_ver = _DYN_VER
    load._hswq_orig_dynamic_load = true_orig
    Dynamic.load = load
    return True


def _force_detach_int8_dynamic_models(device=None, keep_patchers=None) -> int:
    """Offload INT8 Dynamic VRAM (VBAR + hostbufs) without destroying QT weights.

    free_memory often stops after partially_unload and leaves HostBuffers, so
    Nunchaku sees ``0.00 MB usable`` and Aborts. We must fully offload INT8
    Dynamic models before SVDQ load.

    Critical: use ``unpatch_weights=False`` / ``detach(unpatch_all=False)``.
    ``unpatch_all=True`` unpatches INT8 QuantizedTensor + baked LoRA and causes
    black / noise on the next normal SDXL INT8 KSampler.
    """
    try:
        import comfy.model_management as mm
    except ImportError:
        return 0

    keep_ids = {id(p) for p in (keep_patchers or []) if p is not None}
    unloaded = 0
    i = 0
    while i < len(mm.current_loaded_models):
        lm = mm.current_loaded_models[i]
        patcher = lm.model
        if patcher is None:
            i += 1
            continue
        if id(patcher) in keep_ids:
            i += 1
            continue
        if device is not None and getattr(lm, "device", None) is not None:
            try:
                if str(lm.device) != str(device):
                    i += 1
                    continue
            except Exception:
                pass
        is_dyn = False
        try:
            is_dyn = bool(patcher.is_dynamic())
        except Exception:
            is_dyn = False
        if not is_dyn:
            i += 1
            continue
        base = getattr(patcher, "model", None)
        if base is None or not _model_has_int8_quantized_weights(base):
            i += 1
            continue
        # Preserve QT + baked LoRA; only free GPU / VBAR occupancy.
        try:
            lm.model_unload(unpatch_weights=False)
        except TypeError:
            try:
                patcher.detach(unpatch_all=False)
            except TypeError:
                try:
                    patcher.detach(False)
                except Exception as exc:
                    _console(f"[HSWQ INT8→Nunchaku] detach(False) failed: {exc!r}")
            except Exception as exc:
                _console(f"[HSWQ INT8→Nunchaku] detach(unpatch_all=False) failed: {exc!r}")
            try:
                fin = getattr(lm, "model_finalizer", None)
                if fin is not None:
                    fin.detach()
            except Exception:
                pass
            try:
                lm.model_finalizer = None
                lm.real_model = None
            except Exception:
                pass
        except Exception as exc:
            _console(f"[HSWQ INT8→Nunchaku] model_unload(False) failed: {exc!r}")
            i += 1
            continue
        mm.current_loaded_models.pop(i)
        unloaded += 1
    if unloaded > 0:
        try:
            mm.soft_empty_cache()
        except Exception:
            pass
    return unloaded


def _patch_load_models_gpu_int8_nunchaku_handoff() -> bool:
    """Before Nunchaku SVDQ load, offload INT8 Dynamic VRAM without unpatch."""
    try:
        import comfy.model_management as mm
    except ImportError:
        return False

    original = getattr(mm, "load_models_gpu", None)
    if original is None:
        return False
    # v10 = handoff arms ONLY for real Nunchaku SVDQ. All other loads
    # (SDXL / Flux / ZIT / native INT8 / FP / …) pass through untouched.
    # free_memory → model_unload(unpatch_weights=True) kills non-SVDQ INT8.
    _VER = 10
    if getattr(original, "_hswq_int8_nunchaku_handoff_ver", 0) >= _VER:
        return True
    true_orig = getattr(original, "_hswq_orig_load_models_gpu", original)

    def load_models_gpu(
        models,
        memory_required=0,
        force_patch_weights=False,
        minimum_memory_required=None,
        force_full_load=False,
    ):
        keep = []
        need_handoff = False
        device = None
        for m in models or []:
            keep.append(m)
            for mm_extra in getattr(m, "model_patches_models", lambda: [])() or []:
                keep.append(mm_extra)
            base = getattr(m, "model", None)
            # Branch A: native comfy_quant INT8 (any architecture) — never handoff.
            if base is not None and _model_has_int8_quantized_weights(base):
                continue
            # Branch B: only real Nunchaku SVDQ on the BaseModel arms handoff.
            # Do not probe the ModelPatcher itself (false positives).
            if base is not None and _model_is_nunchaku_svdq(base):
                need_handoff = True
                if device is None:
                    device = getattr(m, "load_device", None)
        if need_handoff:
            n = _force_detach_int8_dynamic_models(device=device, keep_patchers=keep)
            # Second pass: any INT8 Dynamic still listed (missed first pass) —
            # never leave them for free_memory(unpatch=True).
            n2 = _force_detach_int8_dynamic_models(device=None, keep_patchers=keep)
            try:
                mm.soft_empty_cache()
            except Exception as exc:
                _console(f"[HSWQ INT8→Nunchaku] soft_empty_cache failed: {exc!r}")
            _console(
                f"[HSWQ INT8→Nunchaku] VRAM handoff before SVDQ load "
                f"(INT8 Dynamic offload keep-weights={n + n2}, no free_memory unpatch)"
            )
        return true_orig(
            models,
            memory_required=memory_required,
            force_patch_weights=force_patch_weights,
            minimum_memory_required=minimum_memory_required,
            # Full load after handoff avoids Nunchaku 0.00 MB usable Abort.
            # Non-SVDQ loads never set need_handoff (branches A/B above).
            force_full_load=True if need_handoff else force_full_load,
        )

    load_models_gpu._hswq_int8_nunchaku_handoff = True
    load_models_gpu._hswq_int8_nunchaku_handoff_ver = _VER
    load_models_gpu._hswq_orig_load_models_gpu = true_orig
    mm.load_models_gpu = load_models_gpu
    return True


def _patch_model_patcher_lora_logs() -> bool:
    """Log whether LoRA bake uses set_weight (requant) or int8_round fallback."""
    try:
        import comfy.model_patcher as mp
    except ImportError:
        return False

    original = getattr(mp.ModelPatcher, "patch_weight_to_device", None)
    if original is None or getattr(original, "_hswq_int8_lora_log", False):
        return getattr(original, "_hswq_int8_lora_log", False)

    def patch_weight_to_device_logged(self, key, device_to=None, inplace_update=False, return_weight=False, force_cast=False):
        global _lora_patcher_logs
        weight, set_func, convert_func = mp.get_key_weight(self.model, key)
        if key in self.patches:
            _lora_patcher_stats["calls"] += 1
            if set_func is not None:
                _lora_patcher_stats["with_set_func"] += 1
            else:
                _lora_patcher_stats["without_set_func"] += 1
            if convert_func is not None:
                _lora_patcher_stats["with_convert_func"] += 1

            path = "requant" if set_func is not None else "int8_round"
            _lora_bake_by_key[key] = path
            if _lora_patcher_logs < _LORA_PATCHER_LOG_MAX:
                _lora_patcher_logs += 1
                wdtype = getattr(weight, "dtype", None)
                warn = ""
                if set_func is None and wdtype is not None and str(wdtype) in ("torch.int8", "int8"):
                    warn = "  << BROKEN for INT8 (LoRA delta will be destroyed)"
                owners = [
                    e["lora_name"]
                    for e in _lora_attach_history
                    if key in (e.get("applied_unet_keys") or [])
                ]
                owner_s = ",".join(owners[:3]) if owners else "-"
                if len(owners) > 3:
                    owner_s += f"+{len(owners) - 3}"
                _console(
                    f"[HSWQ INT8 LoRA] bake #{_lora_patcher_logs}: key={key} "
                    f"path={path} lora={owner_s} weight_dtype={wdtype} "
                    f"convert={'yes' if convert_func else 'no'} "
                    f"set={'yes' if set_func else 'no'}{warn}"
                )
            # After stacked UNet keys are baked, dump per-LoRA summary once
            target = sum(int(e.get("applied_unet") or 0) for e in _lora_attach_history)
            if target <= 0:
                target = int(_lora_attach_last.get("applied_unet") or 0)
            # Unique baked keys may be less than sum (shared keys across LoRAs)
            unique_target = len(
                {
                    k
                    for e in _lora_attach_history
                    for k in (e.get("applied_unet_keys") or [])
                }
            ) or target
            if (
                unique_target > 0
                and _lora_patcher_stats["calls"] >= unique_target
                and not getattr(dump_int8_lora_bake_stats, "_dumped_this_load", False)
            ):
                # Do NOT set the flag before dump (that made dump a no-op).
                dump_int8_lora_bake_stats(force=False)


        return original(
            self,
            key,
            device_to=device_to,
            inplace_update=inplace_update,
            return_weight=return_weight,
            force_cast=force_cast,
        )

    patch_weight_to_device_logged._hswq_int8_lora_log = True
    mp.ModelPatcher.patch_weight_to_device = patch_weight_to_device_logged
    return True


def _per_lora_bake_verdict(entry: dict) -> tuple[str, int, int, int]:
    """Return (verdict, requant, int8_round, not_baked) for one LoRA attach entry."""
    unet_keys = entry.get("applied_unet_keys") or []
    clip_n = int(entry.get("applied_clip") or 0)
    unet_n = int(entry.get("applied_unet") or 0)
    if unet_n == 0 and clip_n > 0:
        return ("N/A_CLIP_only", 0, 0, 0)
    if unet_n == 0:
        return ("SKIP_no_keys", 0, 0, 0)
    requant = 0
    int8_round = 0
    not_baked = 0
    for k in unet_keys:
        path = _lora_bake_by_key.get(k)
        if path == "requant":
            requant += 1
        elif path == "int8_round":
            int8_round += 1
        else:
            not_baked += 1
    if int8_round > 0:
        return ("BROKEN_int8_round", requant, int8_round, not_baked)
    if requant == 0 and not_baked == unet_n:
        return ("WARN_not_baked_yet", requant, int8_round, not_baked)
    if requant > 0 and int8_round == 0:
        return ("OK_requant", requant, int8_round, not_baked)
    return ("PARTIAL", requant, int8_round, not_baked)


def dump_int8_lora_bake_stats(force: bool = False) -> None:
    """Full Status dump: lora_name / applied_keys / skipped_keys (+ bake if any)."""
    if not force and getattr(dump_int8_lora_bake_stats, "_dumped_this_load", False):
        return
    dump_int8_lora_bake_stats._dumped_this_load = True

    history = list(_lora_attach_history) if _lora_attach_history else []
    if not history and (_lora_attach_last.get("mapped_keys") or 0) > 0:
        history = [dict(_lora_attach_last)]

    n = len(history)
    _lora_line(f"[HSWQ LoRA Status] ===== bake summary ({n} slot(s)) =====")
    if not history:
        _lora_line(
            "[HSWQ LoRA Status] Slot -: | lora_name='(none)' | applied_keys=0 | skipped_keys=0 | → SKIPPED ✗"
        )
    ok_n = 0
    for i, a in enumerate(history, 1):
        line = _format_lora_slot_line(i, a, include_bake=True)
        _lora_line(line)
        verdict, _rq, _ir, _nb = _per_lora_bake_verdict(a)
        if verdict in ("OK_requant", "N/A_CLIP_only") or _slot_applied_count(a) > 0:
            if verdict != "BROKEN_int8_round":
                ok_n += 1
    _lora_line(
        f"[HSWQ LoRA Status] Summary: {ok_n}/{n} LoRA(s) with applied keys"
    )

    s = _lora_patcher_stats
    if s["calls"] == 0:
        _lora_line("[HSWQ LoRA Bake] not yet (model not on GPU)")
        return
    _lora_line(
        f"[HSWQ LoRA Bake] total={s['calls']} requant={s['with_set_func']} "
        f"int8_round={s['without_set_func']} shape_skip={len(_lora_shape_skips)}"
    )
    if s["without_set_func"] > 0:
        _lora_line(
            "[HSWQ LoRA Bake] WARNING: int8_round used — those layers are broken"
        )
    else:
        _lora_line("[HSWQ LoRA Bake] path OK (all requant)")
    if _lora_shape_skips:
        for name, key, reason in _lora_shape_skips[:_LORA_SKIP_PRINT_MAX]:
            _lora_line(
                f"[HSWQ LoRA Bake] shape_skip | '{name}' | {key} | {reason}"
            )


def _patch_lora_loader_name_context() -> bool:
    """Capture name from nodes.LoraLoader when any node calls it."""
    try:
        import nodes as nodes_mod
    except ImportError:
        return False

    LoraLoader = getattr(nodes_mod, "LoraLoader", None)
    if LoraLoader is None:
        return False
    original = getattr(LoraLoader, "load_lora", None)
    if original is None:
        return False
    _NAME_VER = 6
    if getattr(original, "_hswq_lora_name_ctx_ver", 0) >= _NAME_VER:
        return True
    true_orig = getattr(original, "_hswq_orig_load_lora", original)

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        global _current_lora_name, _current_lora_strength_model, _current_lora_strength_clip
        prev = (
            _current_lora_name,
            _current_lora_strength_model,
            _current_lora_strength_clip,
        )
        _set_current_lora_name(lora_name, strength_model, strength_clip)
        try:
            return true_orig(self, model, clip, lora_name, strength_model, strength_clip)
        finally:
            (
                _current_lora_name,
                _current_lora_strength_model,
                _current_lora_strength_clip,
            ) = prev

    load_lora._hswq_lora_name_ctx = True
    load_lora._hswq_lora_name_ctx_ver = _NAME_VER
    load_lora._hswq_orig_load_lora = true_orig
    LoraLoader.load_lora = load_lora
    return True


def _patch_loras_folder_path_name() -> bool:
    """Any loader that resolves folder_paths 'loras' → capture filename."""
    try:
        import folder_paths
    except ImportError:
        return False

    _PATH_VER = 3
    ok = False

    for fname in ("get_full_path", "get_full_path_or_raise"):
        original = getattr(folder_paths, fname, None)
        if original is None:
            continue
        if getattr(original, "_hswq_lora_path_name_ver", 0) >= _PATH_VER:
            ok = True
            continue
        true_orig = getattr(original, "_hswq_orig_get_full_path", original)

        def _make(orig):
            def wrapped(folder_name, filename):
                if folder_name == "loras":
                    _set_current_lora_name(filename)
                return orig(folder_name, filename)

            wrapped._hswq_lora_path_name_ver = _PATH_VER
            wrapped._hswq_orig_get_full_path = orig
            return wrapped

        setattr(folder_paths, fname, _make(true_orig))
        ok = True
    return ok


def _patch_load_torch_file_lora_name() -> bool:
    """Any loader that load_torch_file(lora_path) → capture basename."""
    try:
        import comfy.utils as utils_mod
    except ImportError:
        return False
    original = getattr(utils_mod, "load_torch_file", None)
    if original is None:
        return False
    _TORCH_VER = 1
    if getattr(original, "_hswq_lora_torch_name_ver", 0) >= _TORCH_VER:
        return True
    true_orig = getattr(original, "_hswq_orig_load_torch_file", original)

    def load_torch_file(ckpt, *args, **kwargs):
        if isinstance(ckpt, (str, os.PathLike)):
            p = str(ckpt)
            if _path_is_under_loras_dir(p):
                _set_current_lora_name(p)
        return true_orig(ckpt, *args, **kwargs)

    load_torch_file._hswq_lora_torch_name_ver = _TORCH_VER
    load_torch_file._hswq_orig_load_torch_file = true_orig
    utils_mod.load_torch_file = load_torch_file
    return True


def _patch_load_lora_key_counts() -> bool:
    """Wrap load_lora + load_lora_for_models for applied/skipped key counts."""
    try:
        import comfy.lora as lora_mod
        import comfy.sd as sd_mod
        import comfy.weight_adapter as weight_adapter
    except ImportError:
        return False

    orig_load_lora = getattr(lora_mod, "load_lora", None)
    orig_for_models = getattr(sd_mod, "load_lora_for_models", None)
    if orig_load_lora is None or orig_for_models is None:
        return False

    _KEY_VER = 6
    if getattr(orig_for_models, "_hswq_lora_key_count_ver", 0) >= _KEY_VER:
        _patch_lora_loader_name_context()
        _patch_loras_folder_path_name()
        _patch_load_torch_file_lora_name()
        return True

    if getattr(orig_for_models, "_hswq_lora_key_count", False):
        orig_for_models = getattr(
            orig_for_models, "_hswq_orig_for_models", orig_for_models
        )
    if getattr(orig_load_lora, "_hswq_lora_key_count", False):
        orig_load_lora = getattr(orig_load_lora, "_hswq_orig_load_lora", orig_load_lora)

    _ctx = {"patch_dict": {}, "not_mapped": [], "file_keys": 0}

    def load_lora_counted(lora, to_load, log_missing=True):
        patch_dict = {}
        loaded_keys = set()
        for x in to_load:
            alpha_name = "{}.alpha".format(x)
            alpha = None
            if alpha_name in lora.keys():
                alpha = lora[alpha_name].item()
                loaded_keys.add(alpha_name)

            dora_scale_name = "{}.dora_scale".format(x)
            dora_scale = None
            if dora_scale_name in lora.keys():
                dora_scale = lora[dora_scale_name]
                loaded_keys.add(dora_scale_name)

            for adapter_cls in weight_adapter.adapters:
                adapter = adapter_cls.load(x, lora, alpha, dora_scale, loaded_keys)
                if adapter is not None:
                    patch_dict[to_load[x]] = adapter
                    loaded_keys.update(adapter.loaded_keys)
                    continue

            w_norm_name = "{}.w_norm".format(x)
            b_norm_name = "{}.b_norm".format(x)
            w_norm = lora.get(w_norm_name, None)
            b_norm = lora.get(b_norm_name, None)

            if w_norm is not None:
                loaded_keys.add(w_norm_name)
                patch_dict[to_load[x]] = ("diff", (w_norm,))
                if b_norm is not None:
                    loaded_keys.add(b_norm_name)
                    patch_dict["{}.bias".format(to_load[x][: -len(".weight")])] = (
                        "diff",
                        (b_norm,),
                    )

            diff_name = "{}.diff".format(x)
            diff_weight = lora.get(diff_name, None)
            if diff_weight is not None:
                patch_dict[to_load[x]] = ("diff", (diff_weight,))
                loaded_keys.add(diff_name)

            diff_bias_name = "{}.diff_b".format(x)
            diff_bias = lora.get(diff_bias_name, None)
            if diff_bias is not None:
                patch_dict["{}.bias".format(to_load[x][: -len(".weight")])] = (
                    "diff",
                    (diff_bias,),
                )
                loaded_keys.add(diff_bias_name)

            set_weight_name = "{}.set_weight".format(x)
            set_weight = lora.get(set_weight_name, None)
            if set_weight is not None:
                patch_dict[to_load[x]] = ("set", (set_weight,))
                loaded_keys.add(set_weight_name)

        not_mapped = [x for x in lora.keys() if x not in loaded_keys]
        _ctx["patch_dict"] = patch_dict
        _ctx["not_mapped"] = not_mapped
        _ctx["file_keys"] = len(lora) if hasattr(lora, "keys") else 0

        if log_missing:
            for x in not_mapped:
                logging.warning("lora key not loaded: {}".format(x))

        return patch_dict

    def load_lora_for_models_counted(
        model, clip, lora, strength_model, strength_clip, lora_metadata=None
    ):
        new_model, new_clip = orig_for_models(
            model, clip, lora, strength_model, strength_clip, lora_metadata
        )
        loaded = _ctx.get("patch_dict") or {}
        not_mapped = list(_ctx.get("not_mapped") or [])
        file_key_count = int(_ctx.get("file_keys") or 0)
        lora_name = _resolve_lora_name(loaded)

        unet_keys = set(new_model.patches.keys()) if new_model is not None else set()
        if new_clip is not None and hasattr(new_clip, "patcher"):
            clip_keys = set(new_clip.patcher.patches.keys())
        else:
            clip_keys = set()

        applied_unet_keys = []
        applied_clip_keys = []
        mapped_but_not = []
        add_patches_miss = []
        for x in loaded:
            key = x if isinstance(x, str) else x[0]
            in_u = key in unet_keys
            in_c = key in clip_keys
            if in_u:
                applied_unet_keys.append(key)
            if in_c:
                applied_clip_keys.append(key)
            if not in_u and not in_c:
                mapped_but_not.append(x)
                add_patches_miss.append(x)

        applied_unet = len(applied_unet_keys)
        applied_clip = len(applied_clip_keys)

        entry = {
            "lora_name": lora_name,
            "strength_model": strength_model,
            "strength_clip": strength_clip,
            "lora_file_keys": file_key_count,
            "mapped_keys": len(loaded),
            "applied_unet": applied_unet,
            "applied_clip": applied_clip,
            "applied_unet_keys": list(applied_unet_keys),
            "applied_clip_keys": list(applied_clip_keys),
            "not_mapped": sorted(str(x) for x in not_mapped),
            "mapped_but_not_attached": list(mapped_but_not),
            "add_patches_skipped_unet": list(add_patches_miss),
        }
        _lora_attach_last.update(entry)
        _lora_attach_history.append(dict(entry))
        _log_lora_slot_attach(entry)
        return (new_model, new_clip)

    load_lora_counted._hswq_lora_key_count = True
    load_lora_counted._hswq_orig_load_lora = orig_load_lora
    load_lora_for_models_counted._hswq_lora_key_count = True
    load_lora_for_models_counted._hswq_lora_key_count_ver = _KEY_VER
    load_lora_for_models_counted._hswq_orig_for_models = orig_for_models
    lora_mod.load_lora = load_lora_counted
    sd_mod.load_lora_for_models = load_lora_for_models_counted
    _patch_lora_loader_name_context()
    _patch_loras_folder_path_name()
    _patch_load_torch_file_lora_name()
    return True


def _patch_controllora_int8_dequant() -> bool:
    """Dequantize borrowed base-UNet quantized weights in ControlLora.pre_run.

    LoRA-type ControlNets (``lora_controlnet`` marker, e.g. anytest) build a
    control_model that BORROWS the base UNet's own weights via
    ``diffusion_model.state_dict()`` and injects them with ``set_attr_param``.
    The control_model uses ``ControlLoraOps`` (plain float ops); its forward
    calls ``comfy.ops.cast_bias_weight``, which cannot reconstruct a quantized
    weight without its scale.

    Root cause (confirmed from logs + comfy/ops.py):
    ``MixedPrecisionOps.state_dict`` (``_quantized_weight_state_dict``) does NOT
    emit ``QuantizedTensor`` objects. It FLATTENS each quantized ``weight`` into
    separate tensors:
      * ``X.weight``        -> raw int8 qdata      (torch.int8)
      * ``X.weight_scale``  -> per-tensor scale    (torch.float32)
      * ``X.comfy_quant``   -> JSON metadata       (torch.uint8)
      * ``X.input_scale`` / ``X.weight_scale_2`` -> extra params (fp8/nvfp4)
    So ``ControlLora.pre_run`` injects the RAW int8 ``X.weight`` (no scale) into
    the float control_model, and forward feeds int8 straight into
    ``F.linear`` / ``conv2d`` -> NaN / black output. FP8 avoids this only
    because its dtype differs from the compute dtype.

    Fix: wrap ``diffusion_model.state_dict`` during ``ControlLora.pre_run`` and
    return a DEQUANTIZED state dict: for every module whose ``.weight`` is a
    ``QuantizedTensor``, replace ``X.weight`` with ``weight.dequantize()`` (a
    real float tensor) and drop the now-meaningless sidecar keys
    (``X.weight_scale``, ``X.weight_scale_2``, ``X.comfy_quant``,
    ``X.input_scale``). All non-quant weights, biases and buffers pass through
    unchanged. Full-weight ControlNets (Canny) never enter
    ``ControlLora.pre_run`` and are unaffected; the real anytest LoRA weights
    (``.up`` / ``.down``) are plain fp16 and are not touched.
    """
    try:
        import comfy.controlnet as cn
        import comfy.utils
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return False

    ControlLora = getattr(cn, "ControlLora", None)
    if ControlLora is None:
        return False
    original = getattr(ControlLora, "pre_run", None)
    _CL_VER = 2
    if original is None or getattr(original, "_hswq_int8_controllora_ver", 0) >= _CL_VER:
        return getattr(original, "_hswq_int8_controllora", False)
    true_orig = getattr(original, "_hswq_orig_controllora_pre_run", original)

    def _dequantized_state_dict(diffusion_model, orig_sd):
        """Return diffusion_model.state_dict() with quantized weights turned
        back into float tensors and their scale/metadata sidecars removed.

        ``orig_sd`` is the ORIGINAL bound ``state_dict`` method captured before
        we replaced ``diffusion_model.state_dict``. It MUST be used here instead
        of ``diffusion_model.state_dict()`` to avoid re-entering our wrapper
        (which caused ``RecursionError: maximum recursion depth exceeded``)."""
        full = orig_sd()

        # Collect the state-dict prefix of every quantized weight.
        quant_weight_keys = {}
        for name, module in diffusion_model.named_modules():
            w = getattr(module, "weight", None)
            if isinstance(w, QuantizedTensor):
                key = (name + "." if name else "") + "weight"
                quant_weight_keys[key] = w

        out = {}
        n_dequant = 0
        n_drop = 0
        for k, v in full.items():
            replaced = False
            dropped = False
            for wk, qt in quant_weight_keys.items():
                if k == wk:
                    # raw int8 qdata -> real float weight
                    try:
                        out[k] = qt.dequantize()
                        n_dequant += 1
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "[HSWQ INT8] ControlLora: dequantize failed for %s: %s",
                            k, e,
                        )
                        out[k] = v
                    replaced = True
                    break
                base = wk[: -len("weight")]  # "X."
                if (
                    (k.startswith(wk) and k != wk)      # X.weight_scale / weight_scale_2
                    or k == base + "comfy_quant"        # uint8 JSON metadata
                    or k == base + "input_scale"        # fp8 extra param
                ):
                    dropped = True
                    break
            if replaced:
                continue
            if dropped:
                n_drop += 1
                continue
            out[k] = v

        print(
            f"[HSWQ INT8][ControlLora] dequantized state_dict: "
            f"weights dequantized(int8->float)={n_dequant}, "
            f"sidecar keys dropped(scale/comfy_quant/input_scale)={n_drop}, "
            f"total keys out={len(out)}",
            flush=True,
        )
        return out

    def pre_run(self, model, percent_to_timestep_function):
        diffusion_model = getattr(model, "diffusion_model", None)
        patched = False
        orig_sd = None
        if diffusion_model is not None:
            orig_sd = diffusion_model.state_dict

            def dequant_state_dict(*a, **kw):
                # Only intercept the argument-less borrow call ControlLora makes;
                # fall back to the original for any keyword/destination usage.
                if a or kw:
                    return orig_sd(*a, **kw)
                return _dequantized_state_dict(diffusion_model, orig_sd)

            print(
                "[HSWQ INT8][ControlLora] pre_run ENTER "
                "(LoRA-type ControlNet / lora_controlnet path) "
                "-> wrapping diffusion_model.state_dict for INT8 base-weight dequant",
                flush=True,
            )
            diffusion_model.state_dict = dequant_state_dict
            patched = True
        else:
            print(
                "[HSWQ INT8][ControlLora] pre_run ENTER but model has no "
                "diffusion_model; running unpatched",
                flush=True,
            )

        try:
            result = true_orig(self, model, percent_to_timestep_function)
        finally:
            if patched:
                # Remove the instance-level override so the class method is used again.
                try:
                    del diffusion_model.state_dict
                except AttributeError:
                    diffusion_model.state_dict = orig_sd

        print(
            "[HSWQ INT8][ControlLora] pre_run EXIT (base weights injected as float)",
            flush=True,
        )
        logger.info(
            "[HSWQ INT8] ControlLora: injected dequantized base UNet weights "
            "(anytest / lora_controlnet black-output fix)"
        )
        return result

    pre_run._hswq_int8_controllora = True
    pre_run._hswq_int8_controllora_ver = _CL_VER
    pre_run._hswq_orig_controllora_pre_run = true_orig
    ControlLora.pre_run = pre_run
    print(
        "[HSWQ INT8][ControlLora] pre_run patch INSTALLED "
        "(v%d): borrowed INT8 base weights dequantized via state_dict wrap "
        "for LoRA-type ControlNet (anytest fix)" % _CL_VER,
        flush=True,
    )
    return True


def apply_comfy_quant_int8_patches() -> bool:
    """Install INT8 comfy_quant patches once. Returns True if applied (or already applied)."""
    global _PATCHES_APPLIED
    ok_keys = _patch_load_lora_key_counts()
    ok_name = _patch_lora_loader_name_context()
    ok_path = _patch_loras_folder_path_name()
    ok_torch = _patch_load_torch_file_lora_name()
    ok_lowvram = _patch_lowvram_patch_float_intermediate()
    ok_dyn_bake = _patch_model_patcher_dynamic_int8_lora_bake()
    ok_handoff = _patch_load_models_gpu_int8_nunchaku_handoff()
    ok_controllora = _patch_controllora_int8_dequant()
    # Re-apply ops when patch version bumps (e.g. Conv2d inject gate change).
    ok_ops = _patch_ops_decode_and_conv()
    if _PATCHES_APPLIED:
        return True
    ok_utils = _patch_convert_old_quants()
    ok_lora_log = _patch_model_patcher_lora_logs()
    if ok_ops:
        _PATCHES_APPLIED = True
        _console(
            "[HSWQ INT8] comfy_quant patches applied "
            f"(Conv2d quant load + decode"
            f"{' + convert_old_quants' if ok_utils else ''}"
            f"{' + LoRA bake logs' if ok_lora_log else ''}"
            f"{' + LoRA key counts' if ok_keys else ''}"
            f"{' + LoRA name' if ok_name or ok_path or ok_torch else ''}"
            f"{' + LowVramPatch float dtype' if ok_lowvram else ''}"
            f"{' + Dynamic INT8 LoRA bake' if ok_dyn_bake else ''}"
            f"{' + INT8→Nunchaku VRAM handoff' if ok_handoff else ''}"
            f"{' + ControlLora INT8 dequant' if ok_controllora else ''})"
        )
        return True
    logger.warning(
        "[HSWQ INT8] Failed to apply comfy_quant patches (ops=%s utils=%s)",
        ok_ops,
        ok_utils,
    )
    return False


def load_unet_hswq_weight_dtype(unet_name, weight_dtype):
    import logging
    import torch
    import folder_paths
    import comfy.sd

    # INT8 Conv2d patches: SDXL/ZI UNet (architecture), even if Linear has ConvRot.
    # Krea2/DiT ConvRot: stock-equivalent load — Conv2d inject inflates VRAM vs stock.
    unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
    is_convrot = checkpoint_looks_like_comfy_quant_convrot(unet_path)
    is_int8 = weight_dtype == "int8_tensorwise" or checkpoint_looks_like_comfy_quant_int8(unet_path)
    needs_conv2d = checkpoint_needs_hswq_int8_conv2d(unet_path)

    if is_int8 and is_convrot and not needs_conv2d:
        model_options = {}
        logging.info(
            "[HSWQ INT8] DiT/Krea2 ConvRot — stock-equivalent load "
            "(no INT8 Conv2d patches): %s",
            unet_name,
        )
        print(
            f"[HSWQ INT8] ConvRot DiT/Krea2 stock-equivalent load: {unet_name}",
            flush=True,
        )
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
    elif is_int8:
        apply_comfy_quant_int8_patches()
        model_options = {}
        reset_int8_lora_log_counters()
        if is_convrot and needs_conv2d:
            logging.info(
                "[HSWQ INT8] SDXL/ZI + ConvRot FULL — MixedPrecision + INT8 Conv2d "
                "(Linear: kitchen online; Conv2d: HSWQ online act rotate): %s",
                unet_name,
            )
            print(
                f"[HSWQ INT8] SDXL/ZI ConvRot FULL (Linear+Conv2d) load: {unet_name}",
                flush=True,
            )
        else:
            logging.info(
                "[HSWQ INT8] Loading UNet via MixedPrecisionOps (int8_tensorwise / comfy_quant)"
            )
            print(f"[HSWQ INT8] Loading UNet: {unet_name}", flush=True)
        with _int8_quant_conv_scope():
            model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        summarize_int8_lora_capability(model)
    else:
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)

    return (model,)


def load_checkpoint_sdxl_hswq_weight_dtype(ckpt_name, weight_dtype, device=None):
    import sys
    import torch
    import folder_paths
    import comfy.sd

    pkg = sys.modules[__name__.rsplit(".", 2)[0]]
    get_current_device = pkg.get_current_device
    set_current_device = pkg.set_current_device
    sdxl_logger = pkg.sdxl_logger

    original_device = get_current_device()
    if device is not None:
        set_current_device(device)
    try:
        # INT8 Conv2d + comfy_quant decode only when checkpoint is INT8.
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        # Auto-detect native comfy_quant INT8; do not force float8 dtype over int8 weights.
        is_int8 = weight_dtype == "int8_tensorwise" or checkpoint_looks_like_comfy_quant_int8(ckpt_path)

        model_options = {}
        if is_int8:
            apply_comfy_quant_int8_patches()
            reset_int8_lora_log_counters()
            sdxl_logger.info(
                "[SDXL INT8] Loading checkpoint via MixedPrecisionOps "
                "(int8_tensorwise / comfy_quant): %s",
                ckpt_name,
            )
            with _int8_quant_conv_scope():
                out = comfy.sd.load_checkpoint_guess_config(
                    ckpt_path,
                    output_vae=False,
                    output_clip=True,
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    model_options=model_options,
                )
            model, clip, _v = out[:3]
            summarize_int8_lora_capability(model)
            return (model, clip)

        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=False,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            model_options=model_options,
        )
        model, clip, _v = out[:3]
        return (model, clip)
    finally:
        set_current_device(original_device)


def install_int8_option_dispatch(node_class_mappings) -> bool:
    if not isinstance(node_class_mappings, dict):
        return False

    # Do NOT apply INT8 patches at node registration / import.
    # Patches install only inside load_unet_hswq_weight_dtype /
    # load_checkpoint_sdxl_hswq_weight_dtype when INT8 is actually loaded.

    _FP8_WEIGHT_DTYPES = frozenset({"fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"})

    unet_cls = node_class_mappings.get("HSWQFP8E4M3UNetLoader")
    if unet_cls is not None:
        _orig_load_unet = unet_cls.load_unet

        def load_unet(self, unet_name, weight_dtype):
            # Explicit FP8 choices stay on the original FP loader body — never INT8 helper.
            if weight_dtype in _FP8_WEIGHT_DTYPES:
                return _orig_load_unet(self, unet_name, weight_dtype)
            if weight_dtype == "int8_tensorwise":
                return load_unet_hswq_weight_dtype(unet_name, weight_dtype)
            # default: auto-detect INT8 checkpoints only; otherwise original FP path.
            import folder_paths

            unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
            if checkpoint_looks_like_comfy_quant_int8(unet_path):
                return load_unet_hswq_weight_dtype(unet_name, weight_dtype)
            return _orig_load_unet(self, unet_name, weight_dtype)

        unet_cls.load_unet = load_unet

    sdxl_cls = node_class_mappings.get("HSWQCheckpointLoaderSDXL")
    if sdxl_cls is not None:
        _orig_load_checkpoint = sdxl_cls.load_checkpoint

        def load_checkpoint(self, ckpt_name, weight_dtype, device=None):
            if weight_dtype in _FP8_WEIGHT_DTYPES:
                return _orig_load_checkpoint(self, ckpt_name, weight_dtype, device=device)
            if weight_dtype == "int8_tensorwise":
                return load_checkpoint_sdxl_hswq_weight_dtype(
                    ckpt_name, weight_dtype, device=device
                )
            import folder_paths

            ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
            if checkpoint_looks_like_comfy_quant_int8(ckpt_path):
                return load_checkpoint_sdxl_hswq_weight_dtype(
                    ckpt_name, weight_dtype, device=device
                )
            return _orig_load_checkpoint(self, ckpt_name, weight_dtype, device=device)

        sdxl_cls.load_checkpoint = load_checkpoint

    return True
```

### 3.4 Modified UNet loader snippets (unified diff from baseline)

```diff
diff --git a/hswq/zimage_fp8_e4m3_unet.py b/hswq/zimage_fp8_e4m3_unet.py index 80e7393..9107e84 100644 --- a/hswq/zimage_fp8_e4m3_unet.py +++ b/hswq/zimage_fp8_e4m3_unet.py @@ -953,13 +953,20 @@ class HSWQFP8E4M3UNetLoader:      @classmethod      def INPUT_TYPES(s):          return {"required": { "unet_name": (folder_paths.get_filename_list("diffusion_models"), ), -                              "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "int8_tensorwise"],), +                              "weight_dtype": ([ +                                  "default", +                                  "fp8_e4m3fn", +                                  "fp8_e4m3fn_fast", +                                  "fp8_e5m2", +                                  "int8_tensorwise", +                                  "ConvRot NVFP4", +                              ],),                               }}      RETURN_TYPES = ("MODEL",)      FUNCTION = "load_unet"        CATEGORY = "advanced/loaders" -    TITLE = "HSWQ FP8 E4M3/INT8 UNet Loader" +    TITLE = "HSWQ ConvRot INT8/ConvRot NVFP4 UNet Loader"        def load_unet(self, unet_name, weight_dtype):          model_options = {} diff --git a/nodes/models/zimage_fp8_e4m3_unet.py b/nodes/models/zimage_fp8_e4m3_unet.py index 3ad99c8..a2b09cd 100644 --- a/nodes/models/zimage_fp8_e4m3_unet.py +++ b/nodes/models/zimage_fp8_e4m3_unet.py @@ -45,7 +45,7 @@ class HSWQZImageFP8E4M3UNetLoader:      RETURN_TYPES = ("MODEL",)      FUNCTION = "load_model"      CATEGORY = "HSWQ-ussoewwin" -    TITLE = "HSWQ FP8 E4M3/INT8 UNet Loader" +    TITLE = "HSWQ ConvRot INT8/ConvRot NVFP4 UNet Loader"        def load_model(          self,
```

### 3.5 Modified `README.md` (unified diff from baseline)

```diff
diff --git a/README.md b/README.md index 166d7a1..70adba3 100644 --- a/README.md +++ b/README.md @@ -15,14 +15,15 @@    This custom node pack loads and runs **[Hybrid-Sensitivity-Weighted-Quantization (HSWQ)](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)** packs and related ComfyUI-compatible quantized SDXL / Z Image weights.   -HSWQ is a high-fidelity quantization line for diffusion UNets. Current public HSWQ work focuses on **ConvRot INT8** and **ConvRot NVFP4** for **SDXL** (sensitivity / importance analysis, DualMonitor + weighted-histogram FP16 protection, then FULL ConvRot on the remainder). It is **not** a keep-ratio percentage scheme: keep ratio is fixed at **0 (r0)**; FP16 layers are chosen by automatic analysis under a fixed MiB budget. +HSWQ is a high-fidelity quantization line for diffusion UNets. Current public HSWQ work focuses on **ConvRot INT8** and **ConvRot NVFP4** for **SDXL**, plus **ConvRot NVFP4** for **Z Image / ZIT** UNets (sensitivity / importance analysis, DualMonitor + weighted-histogram FP16 protection, then FULL ConvRot on the remainder). It is **not** a keep-ratio percentage scheme: keep ratio is fixed at **0 (r0)**; FP16 layers are chosen by automatic analysis under a fixed MiB budget.    | Path | Role in this repo |  | :--- | :--- |  | **HSWQ ConvRot INT8 (SDXL V3.1)** | ComfyUI `int8_tensorwise` packs; load via **HSWQ Checkpoint Loader (SDXL)** (`weight_dtype`: `int8_tensorwise` / INT8 auto-detect). **Supported only for models quantized with [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization).** |  | **HSWQ ConvRot NVFP4 (SDXL)** | ComfyUI `nvfp4` packs (Linear竊誰VFP4, Conv2d竊棚NT8 + ConvRot); load via the **same** **HSWQ Checkpoint Loader (SDXL)** (`weight_dtype`: `ConvRot NVFP4`, or `default` with NVFP4 auto-detect). **Supported only for models quantized with [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization).** | +| **HSWQ ConvRot NVFP4 (Z Image / ZIT)** | ComfyUI `nvfp4` UNet packs (often Linear NVFP4 + INT8 protect); load via **HSWQ ConvRot INT8/ConvRot NVFP4 UNet Loader** (`weight_dtype`: `ConvRot NVFP4`, or `default` with NVFP4 auto-detect). Uses the bench-matched **Comfy parity** path (stock GEMM + online act rotate), not the SDXL Tensor Core product path. **Supported only for models quantized with [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization).** |  | **FP8 (E4M3)** | HSWQ **FP8 development has ended** (technical docs remain upstream). Loaders here may still accept existing FP8 weights where ComfyUI supports them | -| **Z Image 8-bit** | HSWQ-specific Z Image INT8 development / publication **ended**. Prefer **native ConvRot INT8** for Z Image (typically SSIM > 0.99). HSWQ INT8 continues for **SDXL** | +| **Z Image 8-bit** | HSWQ-specific Z Image INT8 development / publication **ended**. Prefer **native ConvRot INT8** for Z Image (typically SSIM > 0.99). HSWQ INT8 continues for **SDXL**. **Z Image ConvRot NVFP4** is supported via the UNet loader above |    Upstream HSWQ targets (reference): ConvRot INT8 SSIM about **0.94窶・.98**, ConvRot NVFP4 about **0.95**, with roughly **30窶・0%** smaller files than FP16 while keeping standard ComfyUI loader compatibility.   @@ -140,13 +141,25 @@ ComfyUI output node that saves images to your ComfyUI **output** folder as **PNG  - **Category**: `image` (output node; no return socket)  - **Output path**: Uses ComfyUI's standard output directory via `folder_paths.get_output_directory()`   -### HSWQ FP8 E4M3/INT8 UNet Loader +### HSWQ ConvRot INT8/ConvRot NVFP4 UNet Loader   -<img src="png/hswqunet.png?v=3" alt="HSWQ FP8 E4M3/INT8 UNet Loader" width="400"> +<img src="png/hswqunet.png?v=3" alt="HSWQ ConvRot INT8/ConvRot NVFP4 UNet Loader" width="400">   -Standard ComfyUI UNet loader wrapper that loads FP8 and INT8 diffusion models (**general FP8 and INT8**, not limited to HSWQ-only weights). Loads the UNet (MODEL) from FP8 / INT8 checkpoints like the standard UNet loader (HSWQ FP8 E4M3, Scaled FP8, and native comfy_quant / `int8_tensorwise` when selected or auto-detected). +Standard ComfyUI UNet loader wrapper for diffusion models under `diffusion_models` (**Z Image / ZIT** and other UNet packs). Loads **MODEL** with FP8, INT8, and **ConvRot NVFP4** weight dtypes.   -This loader does **not** ship an in-node Triton accelerate toggle. INT8 Linear speed is left to **ComfyUI + `comfy_kitchen`** (`int8_linear`: cuda 竊・triton 竊・eager). UI inputs are UNet name / weight dtype only; this extension keeps INT8 **load compatibility** patches (Conv2d / LoRA / ControlLora / handoff), not a separate Triton accelerate widget. +**Z Image / ZIT ConvRot NVFP4** is **supported only for models quantized with [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)**. Other third-party ConvRot NVFP4 UNet packs are out of scope. + +- **General FP8 / INT8**: Same idea as the stock UNet loader (HSWQ FP8 E4M3, Scaled FP8, and native comfy_quant / `int8_tensorwise` when selected or auto-detected). Not limited to HSWQ-only weights for those modes. +- **ConvRot NVFP4 (Z Image / ZIT)**: Select `weight_dtype` = **`ConvRot NVFP4`**, or leave **`default`** when the UNet safetensors has comfy_quant / HSWQ `nvfp4` markers. Routes to this extension窶冱 UNet NVFP4 stack under `nodes/nvfp4/` with the **Comfy parity** path used by `hswq/benchmark` (stock MixedPrecision GEMM + online act rotate; ConvRot Linear LoRA bake kept). **Do not** expect the SDXL Checkpoint Loader窶冱 Tensor Core product path here 窶・SDXL NVFP4 stays on the Checkpoint Loader; Z Image NVFP4 stays on this UNet loader. **Supported only for models quantized with [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization).** +- **INT8 / NVFP4 auto-detect**: INT8-looking packs use the INT8 path; NVFP4-looking packs use the ConvRot NVFP4 path when `weight_dtype` is `default` (NVFP4 dispatch is installed after INT8 so mixed packs are not stolen by INT8-only detect). + +**Inputs**: `unet_name`, `weight_dtype` (`default` / FP8 options / `int8_tensorwise` / `ConvRot NVFP4`). + +This loader does **not** ship an in-node Triton accelerate toggle. INT8 Linear speed is left to **ComfyUI + `comfy_kitchen`** (`int8_linear`: cuda 竊・triton 竊・eager). This extension keeps INT8 **load compatibility** patches (Conv2d / LoRA / ControlLora / handoff) and the **NVFP4** UNet patches under `nodes/nvfp4/`. + +- **Z Image / ZIT ConvRot NVFP4 compatibility**: **Only** UNet packs quantized with [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization) + +**VRAM purge**: For **ConvRot NVFP4** (and HSWQ INT8) UNet loads, place **General Purge VRAM V2** from [ComfyUI-DistorchMemoryManager](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager) at the end of the workflow with **`HSWQ`** on 窶・same reason as the SDXL Checkpoint Loader section.    ### HSWQ Batched Detailer (SEGS)  
```

---

## 4. Technical meaning

### 4.1 Architecture (Z Image vs SDXL)

```text
UNet Loader (ConvRot NVFP4)
  -> apply_comfy_quant_nvfp4_patches()     # detect / load / Linear LoRA bake helpers (shared)
  -> apply_nvfp4_comfy_parity()            # REPLACE TC Linear.forward with stock GEMM + act rotate
  -> require_convrot_parity_forward()      # fail if TC wrap still present
  -> apply_comfy_quant_int8_patches()      # mixed pack INT8 protect load
  -> install_zimage_nvfp4_lora_bake(force) # outermost Dynamic.load + load_models_gpu bake
```

SDXL Checkpoint Loader keeps / restores the **TC product stack**. Z Image must call parity **after** SDXL NVFP4 patches and must not leave TC forward armed.

### 4.2 `nvfp4_comfy_parity.py`

- Ports bench math: online act rotate before stock Linear when `_hswq_nvfp4_convrot` (or INT8 protect ConvRot) is armed.
- Clears kitchen `Params.convrot` on weight so kitchen does not **also** rotate (double-rotate).
- Registers `aten.addmm` for NVFP4 (kitchen gap) via `nvfp4_addmm_patch`.
- Keeps `attach_nvfp4_linear_lora_bake` (convert_weight unrotate then set_weight re-rotate) for ConvRot Linear LoRA.
- `restore_nvfp4_tc_product_stack()` exists so SDXL can put TC back; Z Image load must not skip the fail-closed parity check.

### 4.3 `nvfp4_lora_bake.py` (Dynamic VRAM)

| Piece | Meaning |
|-------|---------|
| `_qt_layout_name` | Read `_layout_cls` / `layout_cls` first — **never** `qt.layout` (`torch.strided`). |
| `_model_has_nvfp4_convrot` | True if any module has `_hswq_nvfp4_convrot` flag (Dynamic-safe). |
| `bake_nvfp4_convrot_patches_on_dynamic_patcher` | For ConvRot-flagged keys with NVFP4 QT: clear LowVramPatch, `patch_weight_to_device`, drop `patches` + `backup`; **keep** `_v` (VBAR). |
| `bake_remaining_quant_patches_on_dynamic_patcher` | Bake leftover INT8 / other QT that INT8 Dynamic bake missed. |
| `install_zimage_nvfp4_lora_bake` | Outermost wrap of `ModelPatcherDynamic.load`; `force=True` after INT8 wrap. |
| `install_load_models_gpu_bake_hook` | Second belt after MultiGPU `load_models_gpu`. |
| `_ensure_dynamic_load_bake_wrap` (in `load_unet.py`) | Re-arm if MultiGPU/INT8 overwrote hooks. |

### 4.4 `patches/comfy_quant_int8.py` change

Before: `isinstance(weight, QuantizedTensor)` treated **any** QT (including NVFP4) as INT8, which armed INT8 Dynamic bake on ConvRot NVFP4 layers and destroyed quality.

After: `_qt_is_int8_tensorwise` requires layout name containing `TensorWiseINT8`. Dynamic INT8 bake version bumped to **6**.

### 4.5 `prestartup_script.py`

ComfyUI runs this before custom-node `__init__.py`. It must **not** put the package root on `sys.path` (that shadows ComfyUI `nodes` and yields `AttributeError: module 'nodes' has no attribute 'init_extra_nodes'`).  
It saves the product `load_unet_nvfp4_weight_dtype` then rebinds to `nodes.zimage_nvfp4.load_unet` without recursion.

### 4.6 `nvfp4_addmm_patch` / `nvfp4_tc_gate`

Stock `F.linear` with bias often decomposes to `aten.addmm`. Kitchen registered addmm for INT8/FP8/etc. but not `TensorCoreNVFP4Layout`, causing silent dequant of both operands (VRAM spike + quality risk).  
Registration is runtime-only. On GPU CC less than 10.0 (or first `CUBLAS_STATUS_NOT_SUPPORTED`), `nvfp4_tc_gate` disables scaled_mm retries and mutes WARNING spam; dequant mm remains.

### 4.7 What this extension does **not** do

- Does not edit ComfyUI-master
- Does not change SDXL Checkpoint Loader TC semantics as the Z Image default
- Does not claim third-party ConvRot NVFP4 UNet packs are supported — HSWQ-quantized packs only
- Does not delete `module._v` after bake (same VBAR discipline as INT8 Dynamic bake)

### 4.8 Operational checklist

1. Place HSWQ ConvRot NVFP4 UNet under `diffusion_models`.
2. Loader: `weight_dtype` = `ConvRot NVFP4` (or `default` with NVFP4 markers).
3. LoRA then MultiGPU / Dynamic VRAM then sampler as usual.
4. Console must show bake hook **v4**, `nvfp4_convrot=True`, `patches_left=0`.
5. End workflow with **General Purge VRAM V2** (`HSWQ` on) from ComfyUI-DistorchMemoryManager when using HSWQ NVFP4 / INT8.

---

End of manual. Primary implementation commits for LoRA bake: `a72272d`, `1a4be78`, `90958c8`, `ea37fae`, `b8d1144`.
