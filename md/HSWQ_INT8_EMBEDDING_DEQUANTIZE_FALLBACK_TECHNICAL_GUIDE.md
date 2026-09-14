# HSWQ INT8 Embedding Dequant Fallback — Technical Guide

Repository: `ussoewwin/ComfyUI-HSWQ-Loader-and-Tools`
Fix commit: `1eb5e9f` — `fix(int8): dequantize_embedding fallback for already-dequantized (fp16) tables`
Reproduced on: Krea2 text encoder (`Krea2TEModel_`, Qwen3-VL-4B, ConvRot INT8), image-conditioned encode

A crash in ComfyUI core's INT8 embedding path was fixed from inside this loader package.
The failing code lives in ComfyUI (`comfy/ops.py`) and `comfy_kitchen`; the repair is a
runtime patch armed by the loader, so it survives ComfyUI updates and needs no workflow change.

This guide answers the five required sections:

1. The error content
2. The root cause
3. Files created or modified
4. The code created or modified
5. The meaning of that code

---

## 1. The error content

### 1.1 Console

```
[INFO] Requested to load Krea2TEModel_
[INFO] Model Krea2TEModel_ prepared for dynamic VRAM loading. 4239MB Staged. 0 patches attached. Force pre-loaded 249 weights: 627 KB.
[ERROR] !!! Exception during processing !!! No backend can handle 'dequantize_int8_embedding': eager: q: dtype torch.float16 not in {torch.int8}
...
[INFO] Prompt executed in 6.47 seconds
```

### 1.2 Traceback (relevant frames)

```
  File "...\ComfyUI\custom_nodes\comfyui-krea2edit\__init__.py", line 421, in encode
    return (clip.encode_from_tokens_scheduled(tokens),)
  File "...\ComfyUI\comfy\sd.py", line 411, in encode_from_tokens
    o = self.cond_stage_model.encode_token_weights(tokens)
  File "...\ComfyUI\comfy\text_encoders\krea2.py", line 44, in encode_token_weights
    out, pooled, extra = super().encode_token_weights(token_weight_pairs)
  File "...\ComfyUI\comfy\sd1_clip.py", line 228, in process_tokens
    emb, extra = self.transformer.preprocess_embed(emb, device=device)
  File "...\ComfyUI\comfy\text_encoders\qwen3vl.py", line 66, in preprocess_embed
    merged, deepstack = self.visual(image.to(device, dtype=torch.float32), grid)
  File "...\ComfyUI\comfy\text_encoders\qwen35.py", line 648, in forward
    pos_embeds = self.fast_pos_embed_interpolate(grid_thw).to(x.device)
  File "...\ComfyUI\comfy\text_encoders\qwen35.py", line 631, in fast_pos_embed_interpolate
    pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
  File "...\ComfyUI\comfy\ops.py", line 1646, in forward_comfy_cast_weights
    x = get_layout_class(self.layout_type).dequantize_embedding(qdata, params, input)
  File "...\comfy_kitchen\tensor\int8.py", line 188, in dequantize_embedding
    result = torch.ops.comfy_kitchen.dequantize_int8_embedding(
  File "...\comfy_kitchen\registry.py", line 233, in get_capable_backend
    raise NoCapableBackendError(func_name, failures)
comfy_kitchen.exceptions.NoCapableBackendError: No backend can handle 'dequantize_int8_embedding': eager: q: dtype torch.float16 not in {torch.int8}
```

### 1.3 Trigger

An **image-conditioned Krea2 text-encode** (the Krea2 edit node feeds an image into the
Krea2 TE). `Qwen3VLVisionModel.fast_pos_embed_interpolate` gathers rows out of
`visual.pos_embed`, and in the INT8 checkpoint that table is a ConvRot INT8 embedding.
The gather reaches ComfyUI's quantized `Embedding.forward`, which calls the
`comfy_kitchen` INT8 embedding op with an argument the op's backend refuses.

The failure is deterministic for this checkpoint + dynamic-VRAM loading: it aborts the
prompt right after the text encoder is staged.

---

## 2. The root cause

### 2.1 The checkpoint side (not the problem)

`models/clip/Qwen3_VL_4B_Thinking_abliterated_convrot_int8.safetensors` stores the
positional embedding exactly as expected:

```
model.visual.pos_embed.weight        I8   [2304, 1024]
model.visual.pos_embed.weight_scale  F32  [2304, 1]
model.visual.pos_embed.comfy_quant   U8   {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}
```

The int8 weight, per-row scale and ConvRot marker are correct. The defect is **not** in the
file and **not** in the quantization.

### 2.2 The forward contract

`comfy/ops.py`, `MixedPrecisionOps.Embedding.forward_comfy_cast_weights` (lines 1630-1647):

```python
def forward_comfy_cast_weights(self, input, out_dtype=None):
    weight = self.weight

    # Optimized path: lookup in fp8/int8, dequantize only the selected rows.
    if isinstance(weight, QuantizedTensor) and len(self.weight_function) == 0:
        with CastBiasWeightContext(self, device=input.device, dtype=weight.dtype, offloadable=True) as (qdata, _bias):
            if isinstance(qdata, QuantizedTensor):
                params = qdata._params
                scale = params.scale
                qdata = qdata._qdata            # <-- raw INT8 storage
            else:
                params = weight._params
                scale = None

            # int8: per-row scale possible ConvRot, so let the layout do the gather
            if self.quant_format == "int8_tensorwise":
                x = get_layout_class(self.layout_type).dequantize_embedding(qdata, params, input)
                return x if out_dtype is None else x.to(dtype=out_dtype)
            ...
```

The `int8_tensorwise` branch **assumes `qdata` is raw INT8 storage**. Everything depends on
what `CastBiasWeightContext(...)` returns.

### 2.3 What the cast actually returns

`comfy/ops.py`, `cast_bias_weight()` — the CPU branch (reached when the module is a
dynamic-VRAM module and the target device is CPU):

```python
if hasattr(s, "_v") and comfy.model_management.is_device_cpu(device):
    materialize_meta_param(s, ["weight", "bias"])
    weight = s.weight.to(dtype=dtype, copy=True)
    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()
    bias = s.bias.to(dtype=bias_dtype, copy=True) if s.bias is not None else None
    return format_return((weight, bias, (None, None, None)), offloadable)
```

`QuantizedTensor.to(dtype=..., copy=True)` keeps the wrapper and the int8 storage
(`_handle_to` / `_handle_to_copy` in `comfy_kitchen/tensor/base.py` only touch `orig_dtype`
metadata). But the very next line calls `.dequantize()`, which returns a **plain fp16
tensor that is already scale-applied and ConvRot un-rotated**.

So the tuple handed to the Embedding forward is a *dequantized table*, not int8 storage —
while the `int8_tensorwise` branch still routes it through the INT8 gather.

### 2.4 Why the input is on CPU

`comfy/text_encoders/qwen35.py`, `fast_pos_embed_interpolate` derives its device from the
module itself:

```python
device = self.pos_embed.weight.device          # line 599
...
pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]   # line 631
```

Under dynamic-VRAM loading the weight is resident on **CPU** at that moment (the console
line `prepared for dynamic VRAM loading` is the marker), so `idx_tensor` and the embedding
input are CPU, and `cast_bias_weight` takes the CPU branch of §2.3. Because the branch
requires `hasattr(s, "_v")` (the aimdo/vbar handle), this only fires with dynamic VRAM
enabled — which is the normal launch configuration.

### 2.5 Why only `int8_tensorwise` crashes

For fp8 formats the same `else` branch does
`torch.nn.functional.embedding(input, qdata, ...)` and never calls the kitchen op, so an
already-dequantized table still works. Only `int8_tensorwise` delegates to
`TensorWiseINT8Layout.dequantize_embedding` → `torch.ops.comfy_kitchen.dequantize_int8_embedding`,
whose `eager` backend declares `q: dtype in {torch.int8}`. An fp16 `q` therefore raises
`NoCapableBackendError`.

Note the language-model embedding `model.language_model.embed_tokens` is also
`int8_tensorwise` + ConvRot, so the same defect applies there; the visual tap simply
triggers first.

### 2.6 Reproduction (isolated, `comfy_kitchen` only)

Using only `comfy_kitchen` (no ComfyUI graph, no GPU):

1. Build a ConvRot INT8 table with `TensorWiseINT8Layout` (`is_weight=True,
   per_channel=True, convrot=True, convrot_groupsize=256`).
2. Genuine INT8 path — `dequantize_embedding(qt._qdata, params, idx)` → **OK** (matches `qt.dequantize()[idx]`).
3. Already-dequantized path — `dequantize_embedding(fp16_table, params, idx)` →
   **`NoCapableBackendError: ... q: dtype torch.float16 not in {torch.int8}`**, byte-identical
   to the reported error.

This isolates the defect to the two-argument-dtype assumption, independent of ComfyUI.

---

## 3. Files created or modified

| File | Status | Role |
|------|--------|------|
| `patches/comfy_quant_int8.py` | **Modified** | Adds a safe `dequantize_embedding` fallback inside the existing `_patch_comfy_kitchen_int8_gemm_fallback()` installer. |
| `__init__.py` | Unchanged | Already calls `_patch_comfy_kitchen_int8_gemm_fallback()` at package import (≈ line 150), which now also arms the embedding fallback. |
| `md/HSWQ_INT8_EMBEDDING_DEQUANTIZE_FALLBACK_TECHNICAL_GUIDE.md` | **New** | This guide. |

No new runtime module was required: the fix is one additive block inside an existing
patch installer, so the activation path and the arming point are unchanged.

Activation requires a **ComfyUI restart** (the patch is applied at custom-node import).
No workflow edit is needed.

---

## 4. The code created or modified

### 4.1 Diff (`patches/comfy_quant_int8.py`, commit `1eb5e9f`)

```diff
@@ -2425,6 +2425,36 @@ def _patch_comfy_kitchen_int8_gemm_fallback() -> bool:
             return w_float.to(output_dtype)
 
         TensorWiseINT8Layout.dequantize = _safe_dequantize
+        # 3. dequantize_embedding fallback. comfy.ops.cast_bias_weight() has a
+        # CPU branch (dynamic VRAM: hasattr(s, "_v") and the target device is
+        # CPU) that returns an ALREADY-dequantized plain tensor:
+        #     weight = s.weight.to(dtype=dtype, copy=True)
+        #     if isinstance(weight, QuantizedTensor):
+        #         weight = weight.dequantize()
+        # The Embedding forward still routes that table through the INT8 layout
+        # gather, so the kitchen op rejects the fp16 ``q``:
+        #     NoCapableBackendError: dequantize_int8_embedding:
+        #         eager: q: dtype torch.float16 not in {torch.int8}
+        # Keep the genuine INT8 gather untouched (delegate to the kitchen op);
+        # for a non-INT8 table the rows are already scale-applied and ConvRot
+        # un-rotated, so gather them directly.
+        try:
+            _orig_dequantize_embedding = TensorWiseINT8Layout.dequantize_embedding
+            if not getattr(_orig_dequantize_embedding.__func__, "_hswq_safe_embed", False):
+                def _safe_dequantize_embedding(cls, qdata, params, indices):
+                    if qdata.dtype in (torch.int8, torch.uint8):
+                        return _orig_dequantize_embedding.__func__(cls, qdata, params, indices)
+                    rows = torch.nn.functional.embedding(indices, qdata)
+                    out_dtype = getattr(params, "orig_dtype", None) or rows.dtype
+                    return rows.to(dtype=out_dtype)
+                _safe_dequantize_embedding = classmethod(_safe_dequantize_embedding)
+                _safe_dequantize_embedding.__func__._hswq_safe_embed = True
+                _safe_dequantize_embedding.__func__._hswq_orig_embed = _orig_dequantize_embedding
+                TensorWiseINT8Layout.dequantize_embedding = _safe_dequantize_embedding
+                applied.append("TensorWiseINT8Layout.dequantize_embedding fallback")
+        except Exception as e:
+            logger.debug("[HSWQ INT8] dequantize_embedding patch failed: %s", e)
+
         _LAYOUT_DISPATCH_TABLE.setdefault(torch.ops.aten.linear.default, {})[TensorWiseINT8Layout] = _safe_handle_int8_linear_tensorwise
         _LAYOUT_DISPATCH_TABLE.setdefault(torch.ops.aten.mm.default, {})[TensorWiseINT8Layout] = _safe_handle_int8_mm_tensorwise
         _LAYOUT_DISPATCH_TABLE.setdefault(torch.ops.aten.addmm.default, {})[TensorWiseINT8Layout] = _safe_handle_int8_addmm_tensorwise
```

### 4.2 The installed replacement (effective behaviour)

```python
classmethod
def _safe_dequantize_embedding(cls, qdata, params, indices):
    if qdata.dtype in (torch.int8, torch.uint8):
        # genuine INT8 storage -> original kitchen gather (unchanged)
        return _orig_dequantize_embedding.__func__(cls, qdata, params, indices)
    # already-dequantized table (scale + ConvRot already applied) -> gather rows only
    rows = torch.nn.functional.embedding(indices, qdata)
    out_dtype = getattr(params, "orig_dtype", None) or rows.dtype
    return rows.to(dtype=out_dtype)
```

`TensorWiseINT8Layout.dequantize_embedding` is replaced by this classmethod. The original
is captured first and kept on `_hswq_orig_embed`; a guard flag `_hswq_safe_embed` keeps the
install idempotent.

### 4.3 Verification performed

| Check | Result |
|-------|--------|
| `py_compile` of `patches/comfy_quant_int8.py` | OK |
| Isolated repro (genuine int8) | OK, matches `dequantize()[idx]` |
| Isolated repro (already-dequantized fp16) | Fails with the exact reported error (pre-patch) |
| Real module: arm patch, genuine int8 path | OK, matches reference |
| Real module: arm patch, fp16 table path | OK, matches reference |
| Re-arm idempotency (no wrapper stacking) | OK |

---

## 5. The meaning of the code

### 5.1 Delegate when it is real INT8, gather when it is not

The fallback splits on the **storage dtype**:

* `int8` / `uint8` → the call is delegated to the original `dequantize_embedding`, i.e. the
  same `comfy_kitchen` op as before. The normal GPU path keeps its exact behaviour and
  performance; nothing is re-implemented.
* anything else (the dequantized fp16 table) → the rows are gathered with
  `torch.nn.functional.embedding` and **no scale and no ConvRot are applied again**.

That last point is what makes it correct rather than merely non-crashing: the fp16 tensor
comes out of `QuantizedTensor.dequantize()`, which already applies the per-row scale and
un-rotates the ConvRot groups. Re-applying `params.scale` or the Hadamard un-rotation would
silently produce wrong embeddings (a wrong-but-silent result is worse than the crash).

### 5.2 Scope: a generic robustness fallback, not a family-specific hack

The patch is armed globally at import, but it is **not** Krea2-specific. It applies to any
`int8_tensorwise` embedding that reaches the CPU cast branch — the Krea2/Qwen3-VL
`visual.pos_embed` (observed), the language-model `embed_tokens`, and any other INT8
embedding table. For the genuine INT8 case the behaviour is byte-for-byte the original, so
non-affected families (SDXL, Z Image, SAM3, …) cannot change behaviour: their int8
embeddings still take the delegated path, and models without INT8 embeddings never reach
this code.

### 5.3 Consistency with the existing patch design

This mirrors the two precedents already in the same installer:

* `_safe_dequantize` replaced `TensorWiseINT8Layout.dequantize` with a safe, numerically
  equivalent implementation.
* the SAM3 unaligned-GEMM fallback replaced the INT8 linear/mm/addmm handlers so a
  non-multiple-of-4 shape dequantizes to float instead of crashing.

The embedding fallback is the third member of that family: it replaces one layout entry
point with a version that degrades gracefully instead of raising.

### 5.4 Fail-soft and idempotent

* The whole block is wrapped in `try/except`; a failure logs at `debug` and leaves the
  original op in place — the patch can never make the loader worse off.
* `_hswq_safe_embed` prevents double wrapping when the installer re-runs (e.g. version
  bump); `_hswq_orig_embed` keeps the unpatched original reachable for auditing.

### 5.5 Effect on the reported failure

With dynamic VRAM active and the Krea2 TE residing on CPU, the embedding gather now returns
the correct rows instead of aborting the prompt. The INT8 tensors still stay INT8 on the GPU
path; only the already-dequantized CPU table is gathered directly. No model file, no
quantization and no workflow had to change.

---

## 6. Summary

* **Error**: `NoCapableBackendError: No backend can handle 'dequantize_int8_embedding':
  eager: q: dtype torch.float16 not in {torch.int8}`, on Krea2 TE image-encode.
* **Root cause**: ComfyUI's `cast_bias_weight` CPU branch (dynamic VRAM) hands the Embedding
  forward an already-dequantized fp16 table; the `int8_tensorwise` branch still calls the
  INT8 kitchen gather, whose backend requires int8 `q`.
* **Fix**: `patches/comfy_quant_int8.py` — `TensorWiseINT8Layout.dequantize_embedding` gets a
  safe classmethod that delegates to the original for int8/uint8 and gathers rows directly
  for an already-dequantized table. Armed at import via the existing
  `_patch_comfy_kitchen_int8_gemm_fallback()`.
* **Verified**: isolated repro of the exact error; both paths correct after the patch;
  idempotent re-arm.
* **Compatibility**: normal INT8 GPU path unchanged; not family-specific; requires a ComfyUI
  restart to take effect.
