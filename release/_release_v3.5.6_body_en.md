# v3.5.6 — HSWQ INT8 Linear: CPU text-encoder offload dtype fix

## Summary

- **Fixed**: **INT8 Linear dtype mismatch on the CPU text-encoder path** — with `comfy_kitchen` **0.2.36** (ComfyUI 2026-09-30), `TensorWiseINT8Layout.dequantize()` now restores the dequantized tensor to **`params.orig_dtype`** (bfloat16). The HSWQ INT8 **unaligned-GEMM fallback** in `patches/comfy_quant_int8.py` then passed a **bfloat16 weight** together with the **float32 activation** to `torch.nn.functional.linear()`, raising `self and mat2 must have the same dtype, but got Float and BFloat16`. Triggered whenever the CLIP/text encoder runs on **CPU** (e.g. `CLIPLoaderMultiGPU` with `device=cpu` — reproduced on a Qwen-Image ConvRot INT8 text encoder). The fallback now casts weight/bias to the resolved `out_dtype` (activation dtype) before `F.linear`. **CPU-only regression**: the CUDA aligned path uses the real `int8_linear` kernel and was never affected; INT8 weights still stay INT8 in VRAM, and only the fallback path dequantizes.

## Details

Target: `patches/comfy_quant_int8.py` — comfy_kitchen INT8 unaligned GEMM fallback.

### Symptom
```
[ERROR] !!! Exception during processing !!!
self and mat2 must have the same dtype, but got Float and BFloat16
```

### Root cause
`comfy_kitchen` 0.2.36 made `dequantize()` return `params.orig_dtype` (bf16); the HSWQ unaligned-GEMM fallback forwarded that bf16 weight with a float32 CPU activation into `F.linear`.

### Fix
New helper `_dq_linear_dtype_safe(args, kwargs)` dequantizes and casts weight/bias to `out_dtype` (or the activation dtype). Applied to all four fallback call sites.

### Verification
Unit reproduction (bf16 TensorWiseINT8Layout weight × float32 activation): fails before, computes correctly after. `ast.parse` clean.
