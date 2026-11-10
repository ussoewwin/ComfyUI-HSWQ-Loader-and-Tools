# HSWQ INT8 - SDXL ConvRot INT8 Black Images on the Stock CheckpointLoaderSimple after ComfyUI v0.44

This document explains the **black-image failure** that occurs when an SDXL
**ConvRot INT8** checkpoint (a `comfy_quant` `int8_tensorwise` pack such as
`perfectionRealisticILXL_80_hswq_r32_1on_convrot_int8.safetensors`) is loaded
through the **stock ComfyUI CheckpointLoaderSimple node** after the
**ComfyUI v0.44.0 update**, why it happens, how it is fixed, and the exact code
that implements the fix.

Ordinary FP16 checkpoints of the same workflows are **not** affected. The same
setup worked before the update, so the failure is specific to the v0.44 core
combined with the stock loader path.

Fix commit: `00bc498` - "Fix SDXL ConvRot INT8 black images on stock
CheckpointLoaderSimple (ComfyUI v0.44)".

---

## 1. What the failure looked like

### 1.1 The symptom

With an SDXL ConvRot INT8 checkpoint, sampling completes normally (no errors,
the workflow reports success), but the decoded image is **completely black**
(uniform black frame). The same workflow with an FP16 checkpoint of the same
base model produces a normal image. The failure showed up right after the
ComfyUI update to **v0.44.0**.

### 1.2 The observed values (measured, real probe against the failing model)

A single `apply_model` forward (sigma=10, y=zeros, 64x64 latent, fixed seed)
was reproduced outside ComfyUI and instrumented layer by layer:

| layer | measured output |
|---|---|
| `input_blocks.2.0` conv1 (INT8 path) | mean -86.6, **std 858.6**, max 3984 (finite, but ~100x too large) |
| `input_blocks.2.0` out_conv | mean 38.7, **std 475.0**, max 2530 |
| `input_blocks.3.0.op` (`Conv2d` INT8) | **out = inf** (first inf in the model), input itself finite |
| `input_blocks.4.0.in_layers.0` (`GroupNorm`) | **out = NaN (all 327,680 elements)** |
| final DiT output | **all 163,840 elements NaN** |

The CLIP text encoder was verified healthy at the same time (cond shape
(1,77,2048), mean -0.0347, std 1.1752, nan 0), so the failure is **DiT-side
only**. inf entering GroupNorm turns into NaN, and NaN propagates through the
rest of the UNet, the sampler, and the VAE decode - a uniform black frame.

### 1.3 The weight measurement that pinned the cause

The same layer (`model.diffusion_model.input_blocks.3.0.op`) was compared
between what the disk contains and what the loader handed to the forward:

| source | dtype | std | range |
|---|---|---|---|
| on-disk pack (`op.weight`, I8) | int8 | raw | min -127 / max 127 |
| correct dequant (int8 x `weight_scale`) | fp32 | **0.02229** | -0.145 .. 0.145 |
| **what the loader produced** | **fp16** | **29.255** | **-127 .. 127** |

Loader produced / correct dequant ratio: **mean 1314.6x, per-channel 989x to
1563x** (about `1/weight_scale_i` per output channel, matching
`weight_scale` mean 7.75e-4). The int8 values had **not** been multiplied by
the weight scale - they flowed straight into the forward through fp16, so the
weights were roughly **1300x too large**. That is exactly the scale error
needed to overflow fp16 activations within a couple of blocks.

---

## 2. The root cause

### 2.1 ComfyUI's own MixedPrecisionOps has never had a Conv2d class

`comfy/ops.py` defines `mixed_precision_ops()` as a class factory. In both
v0.42.5 (measured via git show) and v0.44.0, the returned `MixedPrecisionOps`
class contains **only `Linear`, `MoEExperts` and `Embedding`**. There is
**no `Conv2d`** in stock. The `Conv2d` used in those cases falls through to
`manual_cast.Conv2d`, which for a `QuantizedTensor` weight only
**dequantizes** inside `CastBiasWeightContext` - it cannot apply the ConvRot
activation rotation, and it does not carry any of the loader's
quantization-aware load path either.

### 2.2 The loader's HSWQ INT8 Conv2d is injected through a scope, not always

`patches/comfy_quant_int8.py` provides the quantized Conv2d
(`_make_quantized_conv2d`) and injects it into the ops class produced by
`mixed_precision_ops` **only while the INT8 conv scope is open**:

```python
def _should_inject_int8_conv(quant_config) -> bool:
    _ = quant_config
    return bool(getattr(_int8_quant_conv_tls, "active", False))
```

The scope (`_int8_quant_conv_scope()`) is opened by the loader-specific
checkpoint/unet loader paths
(`load_checkpoint_sdxl_hswq_weight_dtype` / `load_unet_hswq_weight_dtype`).
**The stock `CheckpointLoaderSimple` node therefore never opens the scope**,
because it goes through `comfy.sd.load_checkpoint_guess_config` directly.

### 2.3 What actually broke at v0.44

Loading an SDXL ConvRot INT8 pack via `CheckpointLoaderSimple` (scope closed)
does the following on v0.44:

1. `model_config.quant_config` is detected (`{'mixed_ops': True}`), so
   `pick_operations` returns `mixed_precision_ops(...)`.
2. `_should_inject_int8_conv` returns **False** (scope closed), so
   **no HSWQ Conv2d is injected**.
3. `Conv2d` becomes stock `manual_cast.Conv2d`. Its `_load_from_state_dict`
   is the stock loader (not `_load_quantized_module`), so the ComfyUI loader
   stores the raw int8 tensor **as fp16 without the weight_scale** - exactly
   the 1300x inflation measured above.
4. The forward therefore multiplies activations by a weights tensor of std
   29 (instead of 0.022), activations reach inf within the network
   (measured at `input_blocks.3.0.op`), and everything downstream becomes
   NaN - a black image.

### 2.4 Why it does not trigger on other architectures

- Krea2, FLUX/DiT, MOSS do not carry SDXL/SD1.5-style block keys
  (`input_blocks`/`middle_block`/`output_blocks`), so detection excludes them
  (via `checkpoint_needs_hswq_int8_conv2d`).
- The FP16 checkpoints of the same workflows never reach a quantized path.

---

## 3. How the fix works (overview)

### 3.1 Detection is key-space only (no model-family guessing)

`_keys_need_hswq_int8_conv2d(keys)` is the single source of truth. It returns
True only when the checkpoint carries SDXL/SD1.5-style block keys
(`input_blocks`, `middle_block`, `output_blocks`) and no DiT block keys.
DiT-only checkpoints (Krea2, FLUX, MOSS) return False.

### 3.2 A wrapper around `comfy.sd.load_checkpoint_guess_config`

`_patch_checkpoint_loader_simple_int8_scope()` replaces
`comfy.sd.load_checkpoint_guess_config` with a wrapper that:

1. Opens the on-disk safetensors with `safe_open` at CPU, reads the key list.
2. Calls `_keys_need_hswq_int8_conv2d(keys)`.
3. If the checkpoint is an SDXL/SD1.5-style INT8 pack, wraps the original
   call in `_int8_quant_conv_scope()`; otherwise calls the original directly.
4. Inside the scope, `mixed_precision_ops` is re-built through the loader's
   patched `mixed_precision_ops_force_conv`, which now injects the HSWQ
   quantized Conv2d (`_make_quantized_conv2d`).
5. `Conv2d._load_from_state_dict` inside the HSWQ Conv2d routes through
   `_load_quantized_module`, building a genuine
   `QuantizedTensor`/`TensorWiseINT8Layout` weight with the scale and the
   ConvRot metadata intact, then arms the ConvRot handling via
   `_arm_hswq_conv2d_convrot`.

### 3.3 Installation timing

`_hswq_ensure_loader_scope_at_import()` is executed **at import time** so the
wrapper is in place before any checkpoint is loaded.

### 3.4 Non-goals (explicitly unaffected paths)

- Krea2 / FLUX / MOSS / SD1.5-style non-INT8 workflows: scope never opens.
- CLIP and VAE: the wrapper only changes `Conv2d` behavior for quantized
  weights; the stock Linear path inside MixedPrecisionOps continues to
  dequantize as before. CLIP INT8 GEMM (v3.6.6) is a separate, unchanged arm.
- ComfyUI core is **not modified**.

### 3.5 A second, unrelated crash fixed in the same commit

During probing (bare processes without ComfyUI's UI), the loader crashed with:

```
AttributeError: type object 'PromptServer' has no attribute 'instance'
```

ComfyUI no longer defines `PromptServer.instance` at class definition time
(it is created by the server at startup). Reading
`server.PromptServer.instance` directly raises AttributeError instead of
returning None. `model_management_mgpu.py` had 3 such reads. The fix wraps
them all in `getattr(server.PromptServer, "instance", None)`.

---

## 4. Files added / modified

| file | change | summary |
|---|---|---|
| `patches/comfy_quant_int8.py` | **modified** | +3 patches: `_patch_checkpoint_loader_simple_int8_scope()` (new), `_hswq_ensure_loader_scope_at_import()` (new, import side effect), registration inside `apply_comfy_quant_int8_patches()` |
| `model_management_mgpu.py` | **modified** | 3 reads of `server.PromptServer.instance` converted to `getattr(..., None)` |

Commit: `00bc498` - 2 files changed, 82 insertions(+), 6 deletions(-).
Live sync: repo and live copies validated equal in LF-normalized SHA-256.

---

## 5. The full new / modified code

### 5.1 `patches/comfy_quant_int8.py` - new patch function

```python
def _patch_checkpoint_loader_simple_int8_scope() -> bool:
    """Stock CheckpointLoaderSimple path (v0.40+): comfy.sd.load_checkpoint_guess_config
    never opens the INT8 conv scope, so an SDXL INT8 pack loads its conv weights as
    raw int8 values cast to fp16 (v0.44 stock MixedPrecisionOps has no Conv2d class
    and no ConvRot handling) - the model then produces inf/NaN (black images).

    Wrap load_checkpoint_guess_config: when the checkpoint is an SDXL/SD1.5-style
    INT8 pack (block keys present), open _int8_quant_conv_scope() around the load
    so mixed_precision_ops injects the HSWQ quantized Conv2d. Every other path
    (Krea2, DiT, MOSS) is untouched - detection is scope-wide and any other
    checkpoint that lacks the SDXL/SD1.5 block keys bypasses the wrapper logic.
    """
    try:
        import comfy.sd as sd_module
        import comfy.utils as utils_module
    except ImportError:
        return False
    original = getattr(sd_module, "load_checkpoint_guess_config", None)
    if original is None or getattr(original, "_hswq_int8_conv_scope_patched", False):
        return True

    def load_checkpoint_guess_config_scope(*args, **kwargs):
        open_scope = False
        try:
            ckpt_paths = args[0] if args else None
            if isinstance(ckpt_paths, (str, os.PathLike)):
                path = os.fspath(ckpt_paths)
            elif isinstance(ckpt_paths, (list, tuple)) and ckpt_paths:
                path = os.fspath(ckpt_paths[0])
            else:
                path = None
            if path is not None:
                from safetensors import safe_open
                with safe_open(path, framework="pt", device="cpu") as f:
                    keys = list(f.keys())
                open_scope = _keys_need_hswq_int8_conv2d(keys)
        except Exception:
            open_scope = False
        if not open_scope:
            return original(*args, **kwargs)
        with _int8_quant_conv_scope():
            return original(*args, **kwargs)

    load_checkpoint_guess_config_scope._hswq_int8_conv_scope_patched = True
    sd_module.load_checkpoint_guess_config = load_checkpoint_guess_config_scope
    return True
```

### 5.2 `patches/comfy_quant_int8.py` - import-time installation

```python
def _hswq_ensure_loader_scope_at_import() -> None:
    """Import-time install of the stock CheckpointLoaderSimple INT8 conv scope.

    Without this, an SDXL INT8 checkpoint loaded via the stock CheckpointLoaderSimple
    node goes through comfy's pick_operations -> mixed_precision_ops with no Conv2d
    class injected (scope closed). v0.44's stock MixedPrecisionOps has no Conv2d, so
    the int8 weights load as raw int8 values cast to fp16 and the model converts the
    forward to inf/NaN (black images). Scoped explicitly to SDXL/SD1.5-style INT8
    packs; Krea2/DiT/MOSS load paths are untouched.
    """
    try:
        _patch_checkpoint_loader_simple_int8_scope()
    except Exception:
        pass


_hswq_ensure_loader_scope_at_import()
```

### 5.3 `patches/comfy_quant_int8.py` - registration inside `apply_comfy_quant_int8_patches()`

```python
    ok_ops = _patch_ops_decode_and_conv()
    # Stock CheckpointLoaderSimple path: open the INT8 conv scope around
    # load_checkpoint_guess_config when the checkpoint is an SDXL/SD1.5 INT8 pack
    # (v0.44 stock MixedPrecisionOps has no Conv2d, so without this the conv
    # weights load as raw int8 lambda casts to fp16 -> inf/NaN black images).
    try:
        _patch_checkpoint_loader_simple_int8_scope()
    except Exception as e:
        logger.debug("[HSWQ INT8] checkpoint loader scope patch failed: %s", e)
    if _PATCHES_APPLIED:
        return True
    ok_utils = _patch_convert_old_quants()
```

### 5.4 `model_management_mgpu.py` - PromptServer safety (3 sites)

```python
    prompt_server = getattr(server.PromptServer, "instance", None)
```

Placed at the three previously-direct-attribute sites
(`trigger_executor_cache_reset` L124, `check_cpu_memory_threshold` L146,
`force_full_system_cleanup` L174).

---

## 6. What the code means (line-by-line intent)

### 6.1 The new patch function

- `_patch_checkpoint_loader_simple_int8_scope()` imports the target modules
  and early-returns if ComfyUI is not loaded. It is guarded by the
  `_hswq_int8_conv_scope_patched` attribute, so re-invocation is a no-op
  (modules are re-imported aggressively by ComfyUI during hot reloads).
- The wrapper `load_checkpoint_guess_config_scope`:
  - extracts the first checkpoint path from positional args (both the
    single-path form and the list form supported by ComfyUI),
  - `safe_open`s the disk file at CPU (cheap mmap read) and reads the key list,
  - calls `_keys_need_hswq_int8_conv2d(keys)` - the same detection the loader
    uses everywhere else. **Nothing else about the key format is parsed**:
    the check is intentionally limited to block-key shape, so non-SDXL INT8
    files simply bypass the wrapper,
  - if not matched: **original behavior**, untouched - this preserves all
    other load paths including Krea2 / DiT / MOSS,
  - if matched: the original call runs inside `_int8_quant_conv_scope()`.
- The scope is what flips `_should_inject_int8_conv` to True, so the next
  `mixed_precision_ops()` call (issued by `pick_operations` during the UNet
  construction) now includes the HSWQ quantized Conv2d, which replaces the
  stock manual_cast Conv2d for this model.
- Result: the Conv2d of the INT8 checkpoint now goes through the loader's
  `_load_from_state_dict` -> `_load_quantized_module`, building a real
  `QuantizedTensor` with a proper `weight_scale` and ConvRot metadata; the
  forward rotates activations online via `rotate_activation_nchw` from the
  loader's own NCWH helper.

### 6.2 The import-time installer

`_hswq_ensure_loader_scope_at_import()` is intentionally **silent**: the
scope patch either installs or no-ops; a failure is logged at debug level.
It is invoked at the bottom of the module so that as soon as the loader
package is imported, the wrapper is live even if no HSWQ checkpoint loader
node was ever instantiated.

### 6.3 The `PromptServer` safety

`server.PromptServer.instance` is now an attribute created by the server at
startup. Reading it directly without checking whether the attribute exists
raises `AttributeError` instead of a benign `None`
(`type object 'PromptServer' has no attribute 'instance'`). Using
`getattr(server.PromptServer, "instance", None)` restores the old
"not yet initialized" behavior.

### 6.4 Why this cannot leak to other paths

- The wrapper is a **pass-through** for every checkpoint that is not an
  SDXL/SD1.5-style INT8 pack: the key list does not match, so the original
  function is called directly without touching anything.
- The scope is opened **only around** `load_checkpoint_guess_config`
  (`with` block) - never left open across calls. Even if a call crashes
  inside, the TLS flag resets via the existing `_int8_quant_conv_scope()`
  context manager.
- The Conv2d injection happens inside `mixed_precision_ops_force_conv` and
  only at the **class-factory** level. It does not touch the stock
  `Linear` path or any non-INT8 behavior.

### 6.5 The measured "before / after"

| quantity | before the fix | after the fix |
|---|---|---|
| `op.weight` type after loading | fp16 Parameter (int8 raw values) | QuantizedTensor / TensorWiseINT8Layout |
| `_hswq_convrot` on that layer | absent | True |
| Conv2d class used | stock manual_cast.Conv2d (module `comfy.ops`) | HSWQ Conv2d (module `ComfyUI-HSWQ-Loader-and-Tools.patches.comfy_quant_int8`) |
| forward output | all NaN (163,840 / 163,840) | mean -0.0344, std 0.7010, min -1.755, max 2.322, nan 0, inf 0 |

---

## 7. Result

With the patch active:

- Loading the same SDXL ConvRot INT8 checkpoint through the **stock
  CheckpointLoaderSimple** on ComfyUI v0.44 produces a healthy
  `QuantizedTensor` (`TensorWiseINT8Layout`, `_hswq_convrot=True`) for the
  Conv2d layers, driven by the loader's HSWQ Conv2d forward.
- A measured single `apply_model` forward returns
  `mean -0.0344, std 0.701, nan 0, inf 0` - normal distribution, versus the
  pre-fix **all-NaN** output.
- The first-inf layer `input_blocks.3.0.op` is now healthy (its weight
  dequantization std is 0.022 and the forward no longer overflows).
- FP16 checkpoints, Krea2, FLUX/DiT, and MOSS workflows are unchanged.
- The interface and behavior of every other loader is preserved (separation
  maintained - ComfyUI core is not modified).

Committed as `00bc498` (main) and live-synced
(LF-normalized SHA-256 repo/live validated equal).

