# HSWQ tcon NVFP4 2nd-Generation Noise Issue — Complete Guide

**Date:** 2026-08-27
**Baseline commit:** `1156f00` (ComfyUI-HSWQ-Loader-and-Tools / docs: add v3.4.4 Chinese release notes)
**Repositories covered:**
- ComfyUI-HSWQ-Loader-and-Tools (`1156f00` → `fdc60bc`)
- ComfyUI-DistorchMemoryManager (`62c7cbd` → `9854848`)

---

## ① What Was the Problem

### Symptom

In workflows using tcon (Z Image TC/W4A4) NVFP4 models, **the 1st generation is normal**, but **the 2nd generation after a DisTorch HSWQ purge is completely noise-corrupted**.

### Log Evidence (bake result comparison)

| Generation | Bake result | State |
|------|-----------|------|
| 1st | `nvfp4_baked=86 int8_baked=94` / `NVFP4_LORA_BAKE_OK` | Normal |
| 2nd | `nvfp4_baked=0 other_qt_baked=83` / `NVFP4_LORA_BAKE_N/A` | NVFP4 layers misclassified as `other_qt`, baked without ConvRot → noise |

`nvfp4_baked=0` in the 2nd-generation log means **not a single NVFP4 ConvRot layer was LoRA-baked**. NVFP4 layers are stored with weights rotated by a Hadamard transform (ConvRot); baking a LoRA requires the sequence "unrotate → apply LoRA → re-rotate". When that sequence is skipped, raw LoRA deltas are added onto rotated weights, corrupting them and producing noise.

---

## ② Root Cause

### Causal Chain (4 steps)

1. **The purge peels HSWQ's load wrap**
   DisTorch's HSWQ purge calls `uninstall_zimage_nvfp4_lora_bake()`, which peels the wrap on `ops._load_quantized_module` (the wrapper that arms NVFP4 flags when HSWQ loads a quantized module). This drops the `_hswq_nvfp4_full_load` stamp.

2. **`apply_comfy_quant_nvfp4_patches()` early-returns**
   This function decided "already applied" from `_PATCHES_APPLIED` and `stack_ver` alone, without checking whether the `_load_quantized_module` wrap was **actually still in place**. After a purge it therefore kept early-returning and never re-applied the wrap.

3. **NVFP4 flags are never armed on reload**
   Without the `_load_quantized_module` wrap, `arm_nvfp4_module()` is not called when the model reloads, so the `_hswq_nvfp4_convrot` flag is never set on the individual Linear modules.

4. **The bake function cannot detect NVFP4 and mis-bakes**
   During LoRA bake, `_module_is_nvfp4_convrot()` checks the flag. With the flag missing, NVFP4 layers are not detected, so they are handled as `other_qt` (other quantization); the ConvRot unrotate/re-rotate is skipped, the weights are corrupted → noise.

### Deeper Background

- **ComfyUI caches the loader node's output (the MODEL object).** Even though the purge removes the model from `current_loaded_models`, the loader node itself is not re-executed. As a result, `load_unet` never runs between "purge → next generation", so the hooks were never re-installed.
- The purge's `uninstall_zimage_nvfp4_lora_bake()` peels the `Dynamic.load` wrap — this is correct design ("clean up ZI hooks before switching to SDXL"). The real problem was that **there was no re-arm mechanism after peeling**.

---

## ③ Files Added / Modified

### ComfyUI-HSWQ-Loader-and-Tools (`1156f00` → `fdc60bc`)

| File | Commit | Content |
|---------|---------|------|
| `nodes/zimage_nvfp4/load_unet.py` | `d97bb5b` | Added `_install_permanent_dynamic_load_guard()` — a permanent guard the purge cannot peel, auto-re-arming the `Dynamic.load` bake hook |
| `nodes/zimage_nvfp4/zi_comfy_quant_nvfp4.py` | `fdc60bc` | Added the `_load_wrap_ok` condition to the early-return in `apply_comfy_quant_nvfp4_patches()` — if the purge peeled the load wrap, fall through to a full re-apply |

### ComfyUI-DistorchMemoryManager (`62c7cbd` → `9854848`)

| File | Commit | Content |
|---------|---------|------|
| `nodes/purge_vram.py` (primary — `__init__.py` prefers this one) | `9854848` | After the HSWQ purge completes, set the `unload_models` + `free_memory` queue flags so ComfyUI drops the cached loader-node output → the loader re-runs on the next prompt and the TC stack is rebuilt |

Note: `purge_vram.py` (root, legacy fallback) received the same fix in `2936341`, but `__init__.py` prefers `nodes/purge_vram.py`, so that is the primary file.

---

## ④ Full Code Added / Modified (no omissions)

### 4-1. `nodes/zimage_nvfp4/load_unet.py` (added in `d97bb5b`)

Function added immediately after `_ensure_dynamic_load_bake_wrap()`:

```python
def _install_permanent_dynamic_load_guard() -> None:
    """Install an outer ModelPatcherDynamic.load guard the purge cannot peel.

    The purge uninstall_zimage_nvfp4_lora_bake walks the chain of wraps stamped
    ``_hswq_zi_nvfp4_lora_bake`` and restores the unwrapped Dynamic.load. After that,
    nothing re-installs the bake hook because ComfyUI caches the loader-node output
    and never re-runs load_unet. The 2nd generation then runs without the NVFP4
    ConvRot LoRA bake and produces noise.

    This guard is a separate, permanent wrap (NOT stamped ``_hswq_zi_nvfp4_lora_bake``),
    so the purge deep-clean walks past it. On every Dynamic.load it ensures the bake
    hook is installed via ``_ensure_dynamic_load_bake_wrap()`` (a no-op when the hook
    is already armed at the current version).
    """
    try:
        import comfy.model_patcher as mp
    except ImportError:
        return
    Dynamic = getattr(mp, "ModelPatcherDynamic", None)
    if Dynamic is None:
        return
    cur = getattr(Dynamic, "load", None)
    if cur is None or getattr(cur, "_hswq_zi_rearm_guard", False):
        return

    def _guarded_load(self, *args, **kwargs):
        try:
            _ensure_dynamic_load_bake_wrap()
        except Exception:
            pass
        return cur(self, *args, **kwargs)

    _guarded_load._hswq_zi_rearm_guard = True  # type: ignore[attr-defined]
    _guarded_load._hswq_zi_rearm_guard_prev = cur  # type: ignore[attr-defined]
    Dynamic.load = _guarded_load
```

Call sites added (2 places):

```python
# Inside load_unet_nvfp4_weight_dtype() (right after install_zimage_nvfp4_lora_bake)
    _ensure_dynamic_load_bake_wrap()
    _install_permanent_dynamic_load_guard()   # ← added
    reset_int8_lora_log_counters()
    reset_nvfp4_lora_log_counters()
    reset_zimage_nvfp4_lora_bake_log_counters()
```

```python
# Inside the load_unet wrapper in install_zimage_nvfp4_unet_dispatch()
    def load_unet(self, unet_name, weight_dtype):
        _ensure_dynamic_load_bake_wrap()
        _install_permanent_dynamic_load_guard()   # ← added
        if weight_dtype in _fp8:
            return _prev(self, unet_name, weight_dtype)
        if weight_dtype == ZI_NVFP4_WEIGHT_DTYPE:
            return load_unet_nvfp4_weight_dtype(unet_name, weight_dtype)
```

### 4-2. `nodes/zimage_nvfp4/zi_comfy_quant_nvfp4.py` (fixed in `fdc60bc`)

```python
    mp_fn = getattr(ops, "mixed_precision_ops", None)
    stack_ver = _effective_nvfp4_stack_ver(mp_fn)
    # The HSWQ purge peels the ops._load_quantized_module wrap (drops the
    # _hswq_nvfp4_full_load stamp) while leaving _PATCHES_APPLIED and the
    # detect_unet_config stamp intact. If the load wrap is gone, the arming
    # (arm_nvfp4_module -> _hswq_nvfp4_convrot) never fires on reload, so the
    # 2nd generation bakes NVFP4 layers as other_qt -> noise. Only early-return
    # when the load wrap is still armed; otherwise fall through to re-apply.
    _load_wrap_ok = bool(
        getattr(ops._load_quantized_module, "_hswq_nvfp4_full_load", False)
    )
    if (
        _PATCHES_APPLIED
        and _load_wrap_ok
        and getattr(model_detection.detect_unet_config, "_hswq_nvfp4_packed_dims", False)
        and stack_ver >= _NVFP4_STACK_VER
    ):
        return True

    # Already patched detect/load but LoRA bake missing: re-wrap mixed_precision_ops only.
    if _load_wrap_ok and getattr(model_detection.detect_unet_config, "_hswq_nvfp4_packed_dims", False) and stack_ver < _NVFP4_STACK_VER:
        # Z Image: INT8 wrap used to drop _hswq_nvfp4_stack_ver → false "upgrade"
        # that wrapped TC over ConvRot parity → double online rotate after refresh.
        if _mp_chain_has_comfy_only(mp_fn) or (
            _PATCHES_APPLIED
            and stack_ver == 0
            and getattr(mp_fn, "_hswq_int8_conv_patched", False)
        ):
            try:
                if mp_fn is not None:
                    mp_fn._hswq_nvfp4_stack_ver = _NVFP4_STACK_VER  # type: ignore[attr-defined]
            except Exception:
                pass
            _PATCHES_APPLIED = True
            _console(
                "[HSWQ NVFP4] stack ver stamped "
                "(skip TC upgrade; comfy_parity / INT8 chain intact)"
            )
            return True

        _orig_mp = getattr(mp_fn, "_hswq_nvfp4_orig_mp", None)
        if _orig_mp is None:
            _orig_mp = mp_fn

        def mixed_precision_ops_upgraded(*args, **kwargs):
            mp = _orig_mp(*args, **kwargs)
            Lin = mp.Linear
            # Never wrap TC over ConvRot parity (Z Image double-rotate / noise).
            if getattr(Lin.forward, "_hswq_nvfp4_convrot_parity", False):
                attach_nvfp4_linear_lora_bake(Lin)
                return mp
            if not getattr(Lin.forward, "_hswq_nvfp4_full_forward", False):
                Lin.forward = make_nvfp4_linear_forward(Lin.forward)
            attach_nvfp4_linear_lora_bake(Lin)
            return mp

        mixed_precision_ops_upgraded._hswq_nvfp4_full_forward = True  # type: ignore[attr-defined]
        mixed_precision_ops_upgraded._hswq_nvfp4_stack_ver = _NVFP4_STACK_VER  # type: ignore[attr-defined]
        mixed_precision_ops_upgraded._hswq_nvfp4_orig_mp = _orig_mp  # type: ignore[attr-defined]
        ops.mixed_precision_ops = mixed_precision_ops_upgraded
        _PATCHES_APPLIED = True
        _console(
            "[HSWQ NVFP4] upgraded stack ver=%s "
            "(ConvRot Linear LoRA bake: convert_weight unrotate + set_weight re-rotate)"
            % _NVFP4_STACK_VER
        )
        return True
```

Note: the changes are exactly 3 points — "added the `_load_wrap_ok` definition", "added `and _load_wrap_ok` to the first `if`", "prepended `_load_wrap_ok and ` to the second `if`". The full re-apply path below (`_orig_detect` and beyond) is unchanged.

### 4-3. `nodes/purge_vram.py` (added in `9854848`)

Inserted right after the `HSWQ INT8/NVFP4: Done ? cleared ...` print (just before `except Exception as e:`):

```python
                # tcon NVFP4 support: after the full HSWQ reset, force ComfyUI to re-run
                # the loader node on the next prompt. The purge unloads the model from
                # current_loaded_models, but ComfyUI still caches the loader node output
                # (the MODEL object). Without re-running the loader, the TC (W4A4) stack
                # patches applied in load_unet are not re-installed and the 2nd generation
                # produces noise. Setting the unload_models + free_memory queue flags makes
                # the executor drop cached outputs so the loader re-arms the TC stack.
                try:
                    import server as _srv
                    _ps = getattr(_srv.PromptServer, "instance", None)
                    if _ps is not None and getattr(_ps, "prompt_queue", None) is not None:
                        _pq = _ps.prompt_queue
                        if not getattr(_pq, "currently_running", False):
                            _pq.set_flag("unload_models", True)
                            _pq.set_flag("free_memory", True)
                            print("HSWQ INT8/NVFP4: tcon NVFP4 - queued model unload/cache reset for next prompt (loader will re-arm TC stack)")
                except Exception as _e:
                    print(f"HSWQ INT8/NVFP4: tcon cache-reset flag skipped: {_e}")
```

---

## ⑤ What the Code Means

### 5-1. `_install_permanent_dynamic_load_guard()` (load_unet.py)

**Purpose:** install a "last line of defense" guard on the outside of `ModelPatcherDynamic.load` that the purge cannot peel.

- It is stamped `_hswq_zi_rearm_guard`, but **NOT stamped `_hswq_zi_nvfp4_lora_bake`**. The purge's `_deep_clean_dynamic_load()` walks only wraps stamped `_hswq_zi_nvfp4_lora_bake` and peels those, so this guard passes through untouched.
- `_guarded_load` runs `_ensure_dynamic_load_bake_wrap()` first on every `Dynamic.load` call. That function is "no-op if the bake hook is already armed, re-install if it was peeled", so calling it every time costs almost nothing.
- Then it calls the original `cur` (the bare `Dynamic.load` after peeling). The order is "guard → re-arm check → actual load", so **Dynamic.load can never run with the bake hook peeled**.
- `_guarded_load._hswq_zi_rearm_guard_prev = cur` merely keeps the original function around for future debugging.

### 5-2. `_load_wrap_ok` guard (zi_comfy_quant_nvfp4.py)

**Purpose:** decide "patches applied" not from a flag alone, but from whether the wrap is **actually alive**.

- `_load_wrap_ok = bool(getattr(ops._load_quantized_module, "_hswq_nvfp4_full_load", False))` checks whether `_load_quantized_module` is still HSWQ's wrap (stamped `_hswq_nvfp4_full_load`).
- When the purge peels the wrap, `_hswq_nvfp4_full_load` disappears and `_load_wrap_ok` becomes `False`.
- Adding `and _load_wrap_ok` to the first `if` (the early return) means **no early return when the wrap was peeled** — execution falls through to the full re-apply path (`_orig_detect` and beyond), which re-wraps `_load_quantized_module`.
- `_load_wrap_ok` was also added to the second `if` (the `stack_ver < _NVFP4_STACK_VER` re-wrap path). That path re-wraps only `mixed_precision_ops` and then early-returns, without re-wrapping `_load_quantized_module`; so when the wrap is peeled, this path must also be skipped and the full re-apply must run.
- Result: on the 2nd load, `arm_nvfp4_module()` runs again and `_hswq_nvfp4_convrot` is re-set on all modules → the bake function correctly detects the NVFP4 layers → `nvfp4_baked=86` is restored.

### 5-3. Post-purge cache reset (nodes/purge_vram.py)

**Purpose:** tell ComfyUI "re-run the loader node" after a purge.

- The purge removes every model from `current_loaded_models`, but the **MODEL object stays in ComfyUI's loader-node output cache**. If nothing is done, the next generation reuses the cached MODEL and `load_unet` never runs again (i.e. the TC stack is never rebuilt).
- `prompt_queue.set_flag("unload_models", True)` and `set_flag("free_memory", True)` set flags telling the ComfyUI executor to "unload all models before the next prompt and drop cached node outputs".
- This makes the loader node re-execute on the next generation, running the full chain `load_unet_nvfp4_weight_dtype()` → `apply_comfy_quant_nvfp4_patches()` → `install_zimage_nvfp4_lora_bake()` → `_install_permanent_dynamic_load_guard()`, which fully re-arms the TC (W4A4) stack.
- The `currently_running` check prevents setting the flags mid-prompt and causing misbehavior. The `except` absorbs environment differences such as a not-yet-initialized server.

---

## Overall Picture (3-layer defense)

| Layer | File | Role |
|----|---------|------|
| 1 | `nodes/purge_vram.py` | Reset the ComfyUI cache after purge, **guaranteeing the loader re-runs** |
| 2 | `zi_comfy_quant_nvfp4.py` | The re-run loader **always performs a full re-apply** (detects peeled wrap) |
| 3 | `load_unet.py` | From then on, **auto re-arms the bake hook on every Dynamic.load** |

The 1st generation remains normal as before. With these 3 layers, the 2nd and later generations after a purge are also guaranteed to perform the correct NVFP4 ConvRot LoRA bake.
