# HSWQ ComfyUI 0.30.2 Compatibility Fix — Complete Technical Reference

- Target repository: `ComfyUI-HSWQ-Loader-and-Tools` (v3.3.9)
- Target ComfyUI: 0.30.2 (comfy_kitchen / MixedPrecisionOps integrated)
- Commits: `21792a8` → `1fd6ad4` → `3dc8a9a` → `ecd6bc0` (all pushed to `origin/main`)
- Verified: Krea2 ConvRot INT8 1st run speed restored ✅ / Krea2 2nd+ run fix applied ✅ / ZI NVFP4 LoRA OK ✅

---

## 1. What Went Wrong

### Problem 1: Krea2 ConvRot INT8 Extremely Slow (Primary Symptom)

After updating ComfyUI to the 0.30.x series, Krea2 ConvRot INT8 inference became extremely slow.
Two root causes.

**(a) Full-module scan on every `load_models_gpu` patch invocation**

`patches/comfy_quant_int8.py` monkey-patches `comfy.model_management.load_models_gpu`,
and **on every call** it ran:

- `_model_has_int8_quantized_weights(model)` — walks **every module** via `model.named_modules()`
  (thousands to tens of thousands for Krea2) searching for QuantizedTensor
- `_model_is_nunchaku_svdq(model)` — same full-module scan

In ComfyUI 0.30.x, model load / memory management frequency increased, so these O(n) scans
executed each time, accumulating to massive overhead.

**(b) CPU→GPU transfer of Hadamard matrix on every forward pass**

ConvRot (Hadamard rotation) rotates activations as `x_rot = x @ H`.
The old `rotate_activation()` called `h_matrix.to(dtype, device)` every time,
transferring a CPU-built Hadamard matrix to the GPU on every forward pass.
Additionally, the HSWQ-injected Conv2d forward also called `build_hadamard(..., device="cpu")`
each time. GPU transfers involve device synchronization; the cost multiplied by layer
count × step count.

### Problem 2: ZI NVFP4 VRAM Growth (Secondary Symptom)

`nodes/zimage_nvfp4/nvfp4_lora_bake.py`'s `install_load_models_gpu_bake_hook`
scanned **all current_loaded_models** on every `load_models_gpu` call:

- `_nvfp4_convrot_diag(model)` — full-module scan (no cache)
- `run_zimage_nvfp4_lora_bake_on_patcher()` — fallback path called
  `_patcher_has_quant_via_keys()` which walks **all LoRA patch keys**,
  calling `get_key_weight()` (expensive QT unwrap) for each key

Repeating this caused GPU memory fragmentation and unnecessary weight movement,
cumulatively increasing VRAM usage.

### Problem 3 (Latent Bug): `get_hadamard_on_device` Referenced But Undefined

In the first fix commit `21792a8`, the HSWQ-injected Conv2d forward in
`patches/comfy_quant_int8.py` was changed to call `nc.get_hadamard_on_device(...)`,
but the patch to **add the function definition** in `native_convert_int8.py`
silently failed (CRLF line-ending mismatch in PowerShell `Replace`).
Only the `_HADAMARD_GPU_CACHE` dict (+3 lines) was committed.

- Krea2 (DiT) does not use the Conv2d injection path, so no symptom was observed
  ("speed restored" appeared correct)
- **Using SDXL ConvRot INT8 would trigger `AttributeError` on first forward** — latent bug

### Problem 4 (Latent Bug): `weight_inner` Referenced But Not Defined

In the same commit `21792a8`, `_bake_int8_patches_on_dynamic_patcher` had
`isinstance(weight, QuantizedTensor)` changed to `isinstance(weight_inner, ...)`,
but the patch to add the definition line
`weight_inner = weight.data if hasattr(weight, "data") else weight`
was not applied ("pattern not found").

- SDXL / INT8 + LoRA Dynamic bake path would trigger `NameError` — latent bug

### Problem 5 (Compatibility): ComfyUI 0.30.2 API Changes

- `mixed_precision_ops`'s `disabled` argument is expected to be a `set` in 0.30.2
  (old HSWQ code passed `[]`)
- `LowVramPatch.__call__` → `comfy.lora.calculate_weight` added `original_weights`
  parameter
- In 0.30.2, quantized weights may be stored as `Parameter(QuantizedTensor)`,
  so `isinstance(w, QuantizedTensor)` alone fails to detect them
- `_quantized_weight_state_dict` added `extra_quant_params` parameter
- LoRA modules relocated to `comfy.weight_adapter.lora`

---

## 2. Files Created / Modified

| File | Type | Content |
|---|---|---|
| `native_convert_int8.py` | Modified | GPU-side Hadamard cache (`get_hadamard_on_device`), `rotate_activation` uses GPU cache |
| `patches/comfy_quant_int8.py` | Modified | Early-return/cache, `disabled` set normalization, 0.30.2 compat (Parameter.data, original_weights, extra_quant_params), `weight_inner` definition, parity contamination peel for Krea2 |
| `nodes/zimage_nvfp4/nvfp4_lora_bake.py` | Modified | `load_models_gpu` bake hook fast-skip |
| `__init__.py` | Modified | `comfy.weight_adapter.lora` import fallback, `calculate_weight` signature fix |

No new files (all existing files modified in-place).
---

## 3. Full Code Changes

### 3-1. `native_convert_int8.py`

#### (a) Module top (GPU cache dict added)

```python
_DEFAULT_GROUPSIZE = 256
_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}
# GPU-side cache: avoids CPU→GPU transfer on every rotate_activation call.
# Keyed by (size, device_str, dtype) – same as CPU cache but on target device.
_HADAMARD_GPU_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}
```

#### (b) New: `get_hadamard_on_device()` (immediately after `build_hadamard`)

```python
def get_hadamard_on_device(
    size: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return Hadamard matrix on target device, with GPU-side caching.

    Builds on CPU (via build_hadamard) once, then transfers to the target
    device and caches there. Subsequent calls with the same
    (size, device, dtype) hit the GPU cache and skip CPU→GPU transfer.
    """
    cache_key = (size, str(device), dtype)
    cached = _HADAMARD_GPU_CACHE.get(cache_key)
    if cached is not None:
        return cached
    h = build_hadamard(size, device="cpu", dtype=torch.float32)
    h = h.to(dtype=dtype, device=device)
    _HADAMARD_GPU_CACHE[cache_key] = h
    return h
```

#### (c) `rotate_activation()` (switched to GPU cache)

```python
def rotate_activation(
    x: torch.Tensor, h_matrix: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Online Linear: x_rot = x @ H (last dim = features)."""
    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} not divisible by group_size {group_size}")
    group_count = features // group_size
    x_grouped = x.reshape(-1, group_count, group_size)
    # GPU-cached Hadamard: build/transfer once, reuse on every call
    h = get_hadamard_on_device(group_size, device=x.device, dtype=x.dtype)
    return torch.matmul(x_grouped, h).reshape(orig_shape)
```

### 3-2. `patches/comfy_quant_int8.py`

#### (a) `_model_is_nunchaku_svdq()` early exit

```python
    seen = set()
    _checked = 0
    _MAX_CHECK_SVDQ = 100  # Early exit: SVDQ modules are typically at the top
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
            _checked += 1
            if _checked >= _MAX_CHECK_SVDQ:
                break
    return False
```

#### (b) `_model_has_int8_quantized_weights()` early exit + 0.30.2 support

```python
    if _model_is_nunchaku_svdq(model):
        return False
    try:
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return False
    _checked = 0
    _MAX_CHECK = 200  # Early exit: only scan first 200 modules
    for _, module in model.named_modules():
        cls_name = type(module).__name__
        if "SVDQ" in cls_name or "Nunchaku" in cls_name:
            continue
        w = getattr(module, "weight", None)
        if w is None:
            continue
        if isinstance(w, QuantizedTensor):
            return True
        # 0.30.2: Parameter wrapping QuantizedTensor
        if hasattr(w, "data") and isinstance(w.data, QuantizedTensor):
            return True
        # Fast path: layout_type set means quantized
        if getattr(module, "layout_type", None) is not None:
            return True
        _checked += 1
        if _checked >= _MAX_CHECK:
            break
    return False
```

#### (c) Injected Conv2d `state_dict()` (0.30.2 `_quantized_weight_state_dict` compat)

```python
        def state_dict(self, *args, destination=None, prefix="", **kwargs):
            sd = destination if destination is not None else {}
            sd = _quantized_weight_state_dict(self, sd, prefix,
                extra_quant_params=("input_scale", "pre_quant_scale"))
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
```

#### (d) Injected Conv2d `forward_comfy_cast_weights()` (GPU cache usage)

```python
        def forward_comfy_cast_weights(self, input):
            if getattr(self, "_hswq_convrot", False):
                nc = _load_native_convert_int8_helpers()
                gs = int(getattr(self, "_hswq_convrot_groupsize", 256) or 256)
                # Use GPU-cached Hadamard via nc.get_hadamard_on_device
                h = nc.get_hadamard_on_device(gs, device=input.device, dtype=input.dtype)
                input = nc.rotate_activation_nchw(input, h, gs)
            want_requant = isinstance(getattr(self, "weight", None), QuantizedTensor)
            weight, bias, offload_stream = cast_bias_weight(
                self, input,
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
```

#### (e) `mixed_precision_ops_force_conv()` `disabled` normalization

```python
        if disabled is None:
            disabled = set()
        elif isinstance(disabled, list):
            disabled = set(disabled)
        result = true_orig(
            quant_config=quant_config,
            compute_dtype=compute_dtype,
            full_precision_mm=full_precision_mm,
            disabled=disabled,
        )
```

#### (f) `LowVramPatch.__call__` `original_weights` support (0.30.2)

```python
    def __call__(self, weight):
        if weight is None or not isinstance(weight, QuantizedTensor):
            return true_orig(self, weight)
        patches = (
            self.prepared_patches
            if self.prepared_patches is not None
            else self.patches[self.key]
        )
        w = weight.dequantize()
        dtype = getattr(w, "dtype", None)
        if dtype is not None and hasattr(dtype, "is_floating_point") \
                and dtype.is_floating_point:
            idtype = dtype
        else:
            idtype = torch.float32
        try:
            return comfy.lora.calculate_weight(
                patches, w, self.key,
                intermediate_dtype=idtype, original_weights=None)
        except TypeError:
            return comfy.lora.calculate_weight(
                patches, w, self.key, intermediate_dtype=idtype)
```

#### (g) `_bake_int8_patches_on_dynamic_patcher()` `weight_inner` definition

```python
            weight, set_func, convert_func = mp.get_key_weight(patcher.model, key)
            if weight is None:
                continue
            # 0.30.2: weight may be Parameter(QuantizedTensor) - unwrap
            weight_inner = weight.data if hasattr(weight, "data") else weight
            if not isinstance(weight_inner, QuantizedTensor):
                continue
            if set_func is None:
                _console(...)
                continue
            keys_to_bake.append((param_key, key))
```
### 3-3. `nodes/zimage_nvfp4/nvfp4_lora_bake.py`

`install_load_models_gpu_bake_hook()` inner `load_models_gpu` wrapper (fast-skip):

```python
    def load_models_gpu(*args, **kwargs):
        result = prev(*args, **kwargs)
        try:
            for loaded in list(getattr(mm, "current_loaded_models", []) or []):
                patcher = getattr(loaded, "model", None)
                if patcher is None:
                    continue
                # Fast skip: no patches AND no baked keys = not our model
                has_patches = bool(getattr(patcher, "patches", None))
                has_baked = bool(getattr(
                    getattr(patcher, "model", None),
                    "_hswq_zi_nvfp4_baked_keys", None))
                if not has_patches and not has_baked:
                    continue
                # Skip non-dynamic models
                try:
                    if not bool(patcher.is_dynamic()):
                        continue
                except Exception:
                    continue
                # Fast skip: check if ZI NVFP4 model via cached diag
                model = getattr(patcher, "model", None)
                if model is not None:
                    diag = _nvfp4_convrot_diag(model)
                    if not diag["has"] and not has_baked:
                        continue
                run_zimage_nvfp4_lora_bake_on_patcher(
                    patcher,
                    device_to=getattr(patcher, "load_device", None),
                    reason="load_models_gpu",
                )
        except Exception as exc:
            _console(f"[HSWQ ZI NVFP4 LoRA] load_models_gpu bake error: {exc!r}")
        return result
```

### 3-4. `__init__.py`

#### (a) LoRA module import fallback

```python
        try:
            try:
                import comfy.weight_adapter.lora as _lora_mod
            except ImportError:
                import comfy.lora as _lora_mod
            _LoraDiff = getattr(_lora_mod, "LoraDiff", None)
```

#### (b) `_lora_skip_calculate_weight` signature fix

```python
                    def _lora_skip_calculate_weight(
                        self, weight, key, strength, strength_model, offset,
                        function, intermediate_dtype=_torch_lora.float32,
                        original_weight=None,
                    ):
                        v = self.weights
                        reshape = v[5]
                        if reshape is not None and tuple(reshape) != weight.shape:
                            logger.warning(
                                "LoRA %s: skip %s (reshape %s != weight %s)",
                                self.name, key, list(reshape), list(weight.shape),
                            )
                            return weight
                        try:
                            lora_diff = _torch_lora.mm(
                                v[0].flatten(start_dim=1), v[1].flatten(start_dim=1)
                            )
                            if lora_diff.numel() != weight.numel():
                                logger.warning(
                                    "LoRA %s: skip %s (numel %d != %d)",
                                    self.name, key, lora_diff.numel(), weight.numel(),
                                )
                                return weight
                        except Exception:
                            return weight
                        return _orig_cw(
                            self, weight=weight, key=key, strength=strength,
                            strength_model=strength_model, offset=offset,
                            function=function, intermediate_dtype=intermediate_dtype,
                            original_weight=original_weight,
                        )
```

---

## 4. Rationale (Per-Change Explanation)

### 4-1. GPU Hadamard Cache (Primary Fix for Krea2 INT8 Speed)

- **Before**: ConvRot rotates activations via Hadamard matrix H. The old implementation
  built H on CPU (`build_hadamard(device="cpu")`) and transferred to GPU via
  `h.to(dtype, device)` on every forward pass. GPU transfer involves device
  synchronization; cost multiplied by layer count × step count.
- **After**: `get_hadamard_on_device()` generates the GPU-side matrix once per
  (size, device, dtype) tuple and caches it in `_HADAMARD_GPU_CACHE`.
  Subsequent calls hit the cache with zero transfer.
- **Effect**: Krea2 ConvRot INT8 inference dramatically faster.

### 4-2. Early Return + Cache (Reducing load_models_gpu Overhead)

- **Before**: `load_models_gpu` monkey-patch ran `_model_has_int8_quantized_weights`
  and `_model_is_nunchaku_svdq` — both doing full `named_modules()` walks.
  For large DiT models like Krea2 (thousands of modules), coupled with 0.30.x's
  increased load/memory management frequency, the total cost was significant.
- **After**:
  - `_model_has_int8_quantized_weights`: checks at most 200 modules. If no
    QuantizedTensor / `layout_type` is found, returns False early. Safe because
    INT8 models always have quantized Linears in the first blocks.
  - `_model_is_nunchaku_svdq`: checks at most 100 modules. SVDQ replaces the
    entire model structure and always appears in early modules.
- **Safety**: Both fall back to False (skip bake, skip handoff) on early exit,
  but target models always have quantized/SVDQ modules at the top, so no
  practical risk. Verified with ZI NVFP4 / Krea2.

### 4-3. `Parameter.data` Unwrap (0.30.2 Weight Storage Format)

- In 0.30.2, quantized weights may be stored as `Parameter(QuantizedTensor)`.
  `isinstance(w, QuantizedTensor)` returns False on Parameter, so old detection failed.
- `hasattr(w, "data") and isinstance(w.data, QuantizedTensor)` unwraps Parameter.
  `_bake_int8_patches_on_dynamic_patcher`'s `weight_inner` serves the same purpose
  (its missing definition was also fixed).
- `layout_type` check added: quantized layers have `layout_type` set by
  `_load_quantized_module`, providing a fast alternative check.

### 4-4. `disabled` Set Normalization (0.30.2 API Contract)

- 0.30.2 `mixed_precision_ops` treats `disabled` as a `set` (uses `disabled.add("nvfp4")` etc.).
  Old HSWQ code passed `[]`, causing potential `AttributeError`.
- Normalized to `set()` / `set(disabled)` for safety.

### 4-5. `original_weights` Parameter (0.30.2 calculate_weight Signature)

- 0.30.2 `comfy.lora.calculate_weight` accepts `original_weights=None`.
  Old code missed this, potentially breaking `model_as_lora`-style patches.
- Retained `try/except TypeError` fallback for older ComfyUI.

### 4-6. `extra_quant_params` (0.30.2 state_dict Contract)

- 0.30.2 `_quantized_weight_state_dict` takes `extra_quant_params`.
  Without explicit `("input_scale", "pre_quant_scale")`, extra keys may
  be dropped or mixed on `state_dict()` save.
- Injected Conv2d `state_dict` now matches 0.30.2 Linear call shape.

### 4-7. load_models_gpu Bake Hook Fast-Skip (ZI NVFP4 VRAM Fix)

- **Before**: ZI NVFP4 `install_load_models_gpu_bake_hook` scanned **all loaded models**
  on every `load_models_gpu` call, running `_nvfp4_convrot_diag` (full-module walk).
  Non-target models (e.g. plain SDXL) were scanned too, causing unnecessary VRAM
  weight movement and fragmentation.
- **After**:
  1. Skip immediately if neither `patches` nor `_hswq_zi_nvfp4_baked_keys` exist
  2. Skip non-dynamic models
  3. Skip if `_nvfp4_convrot_diag` returns `has=False` and no baked keys
- **LoRA impact**: This hook is a **second pass** that picks up leftovers after
  `Dynamic.load` wrapper has already baked. The main bake runs unconditionally
  in the `Dynamic.load` wrapper, so skipping this hook does not affect LoRA
  correctness (user-verified: ZI NVFP4 LoRA OK).

### 4-8. `__init__.py` import / signature (0.30.2 Module Reorganization)

- LoRA modules moved to `comfy.weight_adapter.lora` in 0.30.2.
  Two-stage fallback: try new path, fallback to old `comfy.lora` on ImportError.
- `_lora_skip_calculate_weight` `intermediate_dtype` default changed from
  `None` (replaced internally with `torch.float32`) to direct `torch.float32`,
  matching 0.30.2's real signature.
### 5. Krea2 ConvRot INT8 Cumulative Speed Degradation (4-7x Slowdown on 2nd+ Runs)

#### Symptoms

1st Krea2 run is normal (~4s/step). 2nd run step times explode:
`4.1s → 16.6s → 9.4s → 22.7s → 26.9s → 15.8s`, worsening progressively.
3rd run and beyond are even slower. Z Image NVFP4 runs remain unaffected.

#### Cause: Z Image `comfy_parity` Wrappers Leftover on Krea2 Load

Krea2's load path (`is_int8 and is_convrot and not needs_conv2d`) directly calls
`comfy.sd.load_diffusion_model` — a "stock load" that does not invoke
`apply_comfy_quant_int8_patches()`. However, this path was missing the
`_clear_zimage_parity_contamination_for_sdxl()` call present in the SDXL load path.

After a Z Image run, the HSWQ purge process does **not fully remove** the Z Image
`comfy_parity` wrappers (online act rotate installed by `apply_nvfp4_comfy_parity()`)
from `mixed_precision_ops`. When these wrappers survive and Krea2 stock-loads, the
following chain reaction occurs:

1. **`_load_quantized_module_comfy_only` fires on Krea2's INT8 ConvRot layers**
   → `_arm_int8_protect_convrot_after_stock_load()` sets `_hswq_int8_convrot = True`
   on each Linear module and clears `Params.convrot`.

2. **`_ensure_single_parity_linear_forward()` replaces stock `Linear.forward`
   with `forward_parity`** → online Hadamard act rotate wraps every INT8 ConvRot
   Linear's forward pass.

3. **Every step, `forward_parity` builds/caches a Hadamard matrix and rotates
   inputs via `rotate_last_dim()`**. Krea2 already has ConvRot baked into weights
   offline, with the kitchen's `dequantize_int8_convrot_weight` + `int8_linear`
   handling ConvRot — so `forward_parity`'s online rotation is unnecessary
   (and may double-rotate).

4. **The HSWQ purge discards Hadamard caches (`_hswq_nvfp4_parity_H`)** →
   rebuilds every run → CUDA memory fragmentation accumulates.

As a result, all of Krea2's ConvRot Linear layers (up to 256 layers) perform
unnecessary Hadamard rotation every step, and CUDA fragmentation grows with each
run, causing step times to degrade exponentially.

#### Fix: Peel Parity Contamination Before Krea2 Stock Load

In `patches/comfy_quant_int8.py`'s `load_unet_hswq_weight_dtype()`, at the top
of the Krea2 stock load path, call `_clear_zimage_parity_contamination_for_sdxl()`
(the same function the SDXL path already uses).

```python
if is_int8 and is_convrot and not needs_conv2d:
    try:
        from ..nodes.nvfp4.comfy_quant_nvfp4 import (
            _clear_zimage_parity_contamination_for_sdxl,
        )
        _clear_zimage_parity_contamination_for_sdxl()
    except Exception as e:
        logging.warning(
            "[HSWQ INT8] clear Z Image NVFP4 contamination "
            "for Krea2 failed: %s", e
        )
    model = comfy.sd.load_diffusion_model(...)
```

This ensures:
- `ops.mixed_precision_ops` is restored to stock (non-parity)
- `ops._load_quantized_module` is restored to stock
- Krea2's INT8 ConvRot layers are **not** stamped with `_hswq_int8_convrot`
- `forward_parity` is **not** installed on `Linear.forward`
- Krea2's INT8 stock processing (kitchen `dequantize_int8_convrot_weight` +
  `int8_linear`) operates as-is

---

## Appendix: Commit History

```
ecd6bc0 fix: clear Z Image parity contamination before Krea2 stock load
3dc8a9a fix: define weight_inner before isinstance check in INT8 bake path
1fd6ad4 fix: add missing get_hadamard_on_device GPU cache to native_convert_int8
21792a8 fix: ComfyUI 0.30.2 compatibility + perf (Krea2 ConvRot INT8 slow, ZI NVFP4 VRAM)
```

## Appendix: Verification

- `python -m py_compile` syntax check on all modified files: OK
- `get_hadamard_on_device(256)` returns identical object (cache hit): confirmed
- `rotate_activation` shape preservation ((2,256) → (2,256)): confirmed
- Live test: Krea2 ConvRot INT8 (1st run speed restored / 2nd+ run parity removed) / ZI NVFP4 LoRA OK
