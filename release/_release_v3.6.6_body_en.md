<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/zhmd/v3.6.6.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

# ComfyUI-HSWQ-Loader-and-Tools v3.6.6

## Overview

**The CLIP text tower now runs the real INT8 GEMM.** ComfyUI's `int8_tensorwise` algorithm is declared with `quantize_input: False`, which makes it a **weight-only** format: the weight stays int8 in memory, but it is dequantized for every matmul, so the actual INT8 GEMM (`comfy_kitchen`'s `int8_linear`) is never reached. A ConvRot INT8 CLIP pack therefore paid the memory saving of an int8 weight while paying the compute cost of an fp16 GEMM - the text encode measured **1.9x slower than the FP16 baseline**.

This release wires the INT8 GEMM into the CLIP text path. ComfyUI core is **not modified**: a runtime arm is installed at import time by the loader package.

Measured on CLIP ViT-L/14 (RTX 5060 Ti, batch 1, median of 11 CUDA-event timed rounds, reproduced across two independent runs): **INT8 stock path 27.77 / 26.37 ms with 0 `int8_linear` calls** -> **INT8 with the arm 19.53 / 19.74 ms with 73 `int8_linear` calls** (-29.7 % against the stock path), against an **FP16 reference of 14.05 / 14.56 ms**.

---

## Technical Architecture & Implementation Details

### 1. Root Cause: A Weight-Only Algorithm and a Disabled Quantized Matmul
Two separate mechanisms kept the INT8 GEMM out of the CLIP text tower:
- **The quantized matmul was disabled**: `comfy/sd1_clip.py` builds the text encoder with `mixed_precision_ops(quant_config, dtype, full_precision_mm=True)`. `full_precision_mm=True` makes `_use_quantized` false in `MixedPrecisionOps.Linear.forward`, so the activation is never quantized either.
- **The weight-only branch dequantizes**: even with the quantized matmul enabled, the weight-only path of `forward_comfy_cast_weights` dequantizes the weight before `F.linear`. The result is an fp16 GEMM executed with an int8-sized weight - the saving is in memory only, never in compute.

### 2. The Runtime Arm (`patches/clip_int8_gemm/arm.py`)
The arm is installed at import time from the package `__init__.py`. It touches two call sites and nothing else:
- **Wraps `comfy.ops.mixed_precision_ops`**: a caller that supplies a `quant_config` gets `full_precision_mm=False`, so the quantized matmul is used.
- **Wraps `MixedPrecisionOps.Linear.forward_comfy_cast_weights`**: a `TensorWiseINT8Layout` weight is handed to `int8_linear` with the online ConvRot activation rotation instead of being dequantized.
- **Scope**: only `TensorWiseINT8Layout` layers take the new path. Every other layout keeps the stock behaviour, and the wrap is idempotent - a second installation is a no-op.

### 3. Missing `text_projection` Completion
A CLIP-L checkpoint may ship no `text_projection`. Under the stock (dequant) path that is harmless, but the INT8 GEMM reaches the projection with a `None` weight and the forward dies with `'NoneType' object has no attribute 'device'`. The arm completes it with fp16 zeros at load time, and the fault was reproduced and fixed before this release.

### 4. Verification Across Every Shipped CLIP Pack
The INT8 GEMM is confirmed active for every shipped CLIP pack, measured by counting `int8_linear` invocations during a single encode:

| CLIP pack | `int8_linear` calls per encode |
|---|---|
| `clip_l_hswq_convrot_int8` | 70 |
| `ViT-L-14-BEST-smooth-GmP-HF-format_hswq_convrot_int8` | 64 |
| `Long-ViT-L-14-BEST-GmP-smooth-ft-full_hswq_convrot_int8` | 73 |

---

## Benchmark Measurements

CLIP ViT-L/14, RTX 5060 Ti, batch 1, median of 11 CUDA-event timed rounds. The FP16 and INT8 numbers were taken in alternating rounds inside the same process so that GPU clock drift affects both arms equally, and the whole measurement was repeated in a second independent run (the two values per cell are run 1 / run 2):

| Metric | FP16 | INT8 (stock path) | INT8 (with the arm) |
|---|---|---|---|
| **Encode latency** | 14.05 / 14.56 ms | 27.77 / 26.37 ms | **19.53 / 19.74 ms** |
| **`int8_linear` calls / encode** | - | **0** | **73** |
| **Relative to FP16** | 1.00x | **1.9x slower** | **1.35x slower** |
| **Gain from the arm** | - | - | **-29.7 %** |

Fidelity, measured on a fixed 256-prompt acceptance set (seed 42) against the FP16 baseline:

| Metric | INT8 (stock path) | INT8 (with the arm) |
|---|---|---|
| **cos mean** | 0.999792 | 0.999712 |
| **cos min** | 0.999641 | 0.999548 |
| **cos p5** | 0.999711 | 0.999616 |

The small change is the expected cost of the dynamic activation quantization; both variants stay far above the acceptance threshold. The pack keeps its ConvRot INT8 weight layout, so the size is unchanged: **file 1631.80 MB -> 413.20 MB**, resident weights **235.90 MB -> 118.48 MB**.

---

## Backward Compatibility

- **CLIP ConvRot INT8 packs**: the arm is installed for every quantized CLIP load, so existing packs gain the INT8 GEMM without being regenerated.
- **ComfyUI core**: not modified. The arm lives entirely in `patches/clip_int8_gemm/` inside this package and is active only for `TensorWiseINT8Layout` weights.
- **Other model families** (SDXL / SD1.5 / Z Image / Krea2 / SAM3): their load paths are unaffected; the arm only intercepts the CLIP text-tower `Linear` classes.
- **Non-quantized CLIP**: the stock path is untouched, `full_precision_mm` is only changed when a `quant_config` is supplied.

---

## File Changes Summary

| File | Change |
| :--- | :--- |
| `patches/clip_int8_gemm/arm.py` | Added - runtime arm that enables the real INT8 GEMM for the CLIP text tower and completes a missing `text_projection` |
| `patches/clip_int8_gemm/__init__.py` | Added - package entry, re-exports the arm API |
| `__init__.py` | Modified - installs the arm at import time; version bumped to `3.6.6` |
| `changelog.md` / `zhmd/CHANGELOG.md` | Modified - recorded v3.6.6 release entries |
| `pyproject.toml` | Modified - bumped version to `3.6.6` |
| `zhmd/v3.6.6.md` | Added - Chinese release specification document (linked from the EN/中文 switcher above) |
