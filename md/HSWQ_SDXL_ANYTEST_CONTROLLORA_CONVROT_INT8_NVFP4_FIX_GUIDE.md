# SDXL anytest ControlLoRA on ConvRot INT8 / Hybrid ConvRot NVFP4 Bases - Complete Fix Guide (v2)

**Applies to:** `ComfyUI-HSWQ-Loader-and-Tools` (HSWQ custom node pack)
**Fix commits (pushed to `ussoewwin/ComfyUI-HSWQ-Loader-and-Tools` `main`):**
- `c60bb0b` - ControlLora borrowed-weight dequant wrapper v3 (4D Conv2d ConvRot un-rotation)
- `152c1dc` - **`HSWQCheckpointLoaderSDXL` now routes `int8_tensorwise` through the INT8-aware loader** + Hadamard device fix
**Affected file:** `CN-anytest_v4-marged_am_dim256.safetensors` (SDXL / Illustrious-style **LoRA-type** ControlNet, 738.5 MB, FP16)

---

## 1. Symptoms

Two rounds of symptoms were reported for this LoRA-type ControlNet on HSWQ-quantized SDXL bases:

1. **(fixed by `c60bb0b`)** The control had *no visible effect at all* - no errors, the pipeline ran, but the image ignored the hint entirely.
2. **(fixed by `152c1dc`)** After `c60bb0b` the control visibly worked (structure followed the hint) but the output **locked onto the lineart**: the result stayed black-and-white, **no coloring happened**, and the **strength slider appeared completely dead**.

Decisive observation from the user's real workflow (same base, same hint, strength 1.0):

| ControlNet file | Type | User's output |
|---|---|---|
| `CN-anytest_v4-marged_am_dim256.safetensors` | LoRA-type (`ControlLora`) | near-grayscale (sat ~1.3/255) - lineart lock |
| `CN-anytest_v4-marged_pn_dim256.safetensors` | LoRA-type (`ControlLora`) | near-grayscale (sat ~1.3/255) |
| `CN-anytest4_illustrious2_A_convrot_int8.safetensors` | full-type (`ControlNet`) | correctly colored (sat ~59/255) |

The failure is therefore specific to the **LoRA-type path combined with the checkpoint loader node** - not to the control file itself.

---

## 2. Root cause

### 2.1 `HSWQCheckpointLoaderSDXL` ignored `weight_dtype="int8_tensorwise"` (primary cause of symptom 2)

The node's `load_checkpoint()` called `comfy.sd.load_checkpoint_guess_config(...)` directly. For `int8_tensorwise` it passed `model_options={}` - only the fp8 options ever set a dtype - so **nothing applied the INT8 Conv2d load path**.

ConvRot INT8 checkpoints (e.g. `JANKUTrainedChenkinNoobai_v777_hswq_r32_1off_convrot_int8.safetensors`) store their quantized **Conv2d** layers in the sidecar format:

| Key in checkpoint | Content |
|---|---|
| `X.weight` | raw int8 qdata |
| `X.weight_scale` | per-tensor scale (e.g. shape `(320,1,1,1)`) |
| `X.comfy_quant` | JSON: `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":64}` |

Without the INT8 Conv2d load scope (`_int8_quant_conv_scope()` in `patches/comfy_quant_int8.py`) those layers are left as **raw qdata reinterpreted as fp16 (absmax ~127, std ~30)** in the loaded model. Verified on the user's actual base:

- **base UNet forward alone -> `nan=65536`** (the base itself is broken through this load path)
- **ControlLora borrowed weights** for those convs are +-127 garbage, so the control output **explodes**: `[731, 123352, 183752, NaN]` (vs a sane full-type ControlNet at `[325, 616, 944, 1414]`)

An exploded control signal (even at strength 0.1, `0.1 x 183k` is far above a sane `~1k`) **overrides the model completely**: the output becomes a copy of the hint (B&W lineart), the model can no longer add color, and the strength slider has no visible effect. That is exactly symptom 2.

The INT8-aware loader already existed: `load_checkpoint_sdxl_hswq_weight_dtype()` (same file), which wraps the load in `_int8_quant_conv_scope()` and applies the INT8 Conv2d decode. The node simply never used it.

> Note: this also explains why a repro with plain `load_diffusion_model` / `load_checkpoint_guess_config` showed exploding controls on **both** `waiIllustrious` and `JANKU` bases, while the earlier "sane" results were obtained through the dedicated `load_unet_hswq_weight_dtype` path (which does set the scope).

### 2.2 ControlLora borrowed-weight dequant (fixed by `c60bb0b`, still required)

Recap of the earlier fix (see commit `c60bb0b` for full detail): `ControlLora.pre_run` borrows the base UNet's `state_dict()` and injects it into a **float** `ControlLoraOps` control model.

- **Bug A:** comfy-kitchen's `dequantize_int8_convrot_weight_dtype` is 2D-only - 4D Conv2d raises `NoCapableBackendError`, and the old wrapper fell back to **raw qdata (+-127)**.
- **Bug B:** HSWQ-armed Conv2d (`_hswq_convrot=True`) keeps weights in the **rotated basis** - `qt.dequantize()` succeeds but returns `W_rot`, which is wrong for the float control model (it does not rotate activations).

The v3 wrapper (`_dequantized_state_dict`) now collects `(qt, module)` pairs, dequantizes with `_manual_qt_dequant` (qdata x scale), and **un-rotates** 4D Conv2d weights when the module is armed (`_unrotate_conv2d` mirrors `native_convert_int8.rotate_weight_conv2d` exactly). This is still necessary for any base whose weights are `QuantizedTensor`-wrapped (the `load_unet_hswq_weight_dtype` path).

### 2.3 Hadamard device mismatch (fixed by `152c1dc`)

`_unrotate_conv2d` built the Hadamard matrix on **CPU** while the qdata could be on **CUDA**, causing `RuntimeError: Expected all tensors to be on the same device` inside `ControlLora.pre_run` when the sampler invokes it. `_regular_hadamard` now takes the qdata's device.

---

## 3. Files modified

| Commit | File | Change |
|---|---|---|
| `c60bb0b` | `patches/comfy_quant_int8.py` | `_patch_controllora_int8_dequant` v3: module-aware `_dequantized_state_dict`, `_regular_hadamard` / `_unrotate_conv2d` / `_manual_qt_dequant`, raw-sidecar fallback; `_CL_VER` 2 -> 3 |
| `c60bb0b` | `__init__.py` | Install the ControlLora dequant wrapper **unconditionally at startup** |
| `152c1dc` | `__init__.py` | `HSWQCheckpointLoaderSDXL` delegates `int8_tensorwise` (or auto-detected comfy_quant INT8) to `load_checkpoint_sdxl_hswq_weight_dtype` |
| `152c1dc` | `patches/comfy_quant_int8.py` | `_regular_hadamard(size, device=None)` - device-aware Hadamard |

### Key code - `__init__.py` (node fix, `152c1dc`)

```python
def _checkpoint_looks_int8(ckpt_path):
    try:
        from .patches.comfy_quant_int8 import checkpoint_looks_like_comfy_quant_int8
        return bool(checkpoint_looks_like_comfy_quant_int8(ckpt_path))
    except Exception:
        return False

# inside HSWQCheckpointLoaderSDXL.load_checkpoint:
ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
# INT8 (incl. ConvRot) checkpoints MUST go through the INT8-aware loader:
# plain load_checkpoint_guess_config leaves Conv2d comfy_quant layers as RAW
# int8 qdata (absmax ~127, scale dropped), which breaks the base forward (NaN)
# and poisons the ControlLora borrowed weights (control output explodes ->
# lineart lock / no coloring / strength dead).
if weight_dtype == "int8_tensorwise" or _checkpoint_looks_int8(ckpt_path):
    from .patches.comfy_quant_int8 import load_checkpoint_sdxl_hswq_weight_dtype
    return load_checkpoint_sdxl_hswq_weight_dtype(ckpt_name, weight_dtype, device=None)
```

### Key code - `patches/comfy_quant_int8.py` (Hadamard device fix, `152c1dc`)

```python
def _regular_hadamard(size, device=None):
    h4 = _torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=_torch.float32,
        device=device,
    )
    h = h4
    while h.shape[0] < size:
        h = _torch.kron(h, h4)
    return h / (size ** 0.5)

# in _unrotate_conv2d:
h = _regular_hadamard(gs, w.device)   # was: _regular_hadamard(gs)  -> CPU/CUDA mismatch
```

---

## 4. Verification

All checks run on the user's actual setup: `JANKUTrainedChenkinNoobai_v777_hswq_r32_1off_convrot_int8.safetensors` + `CN-anytest_v4-marged_am_dim256.safetensors`, 16 steps euler/simple, cfg 5.0, denoise 1.0.

### Control-signal sanity (single forward)

| | `input_blocks.1.0.in_layers.2` weight | control output norms (first 4) |
|---|---|---|
| Node path (before fix) | absmax **127.0** (raw qdata) | **[731, 123,352, 183,752, NaN]** |
| Node path (after fix) | absmax **0.50** (dequantized) | **[720, 1,229, 1,415, 1,553]** |
| Reference: full-type ControlNet | - | [325, 616, 944, 1,414] |

Base UNet forward alone: **NaN before fix -> std ~0.97, no NaN after fix**.

### End-to-end generation (user's exact workflow, strength 1.0)

| Run | Saturation (mean HSV-S /255) | Structure adherence (L1 vs hint) |
|---|---|---|
| no control | 110.6 | 0.472 (hint ignored) |
| **am_dim256 (after fix)** | **73.5 - colored** | **0.299 (follows lineart)** |
| full-type ControlNet | 136.2 | 0.554 |

Visual check of the am_dim256 output (image recognition): *"a color image (not a B&W line drawing) - an anime-style girl with long black hair in a sailor uniform sits in a colorful flower field."* Before the fix the same workflow produced a near-grayscale copy of the hint (sat ~1.3).

---

## 5. Usage notes

1. **Restart ComfyUI after updating** the node pack - a long-running server keeps the old (broken) code in memory.
2. Load the INT8 checkpoint with **`HSWQ Checkpoint Loader (SDXL)`** and `weight_dtype = int8_tensorwise` - it is now routed through the INT8-aware loader (Conv2d decode + scope). Unet-only files keep using `load_unet_hswq_weight_dtype` / `UNETLoader`, which already set the scope.
3. The strength slider works normally again (the control magnitude is sane, so 0.0-1.0 scales the visible effect).
4. If a LoRA-type ControlNet still looks "locked" after the update, check that the base itself was loaded through one of the INT8-aware paths above (a plain stock `load_checkpoint_guess_config` reproduces the bug).
5. The full-type (non-LoRA) ControlNet path (`CN-anytest4_illustrious2_A_convrot_int8`) was never affected and continues to work unchanged.
