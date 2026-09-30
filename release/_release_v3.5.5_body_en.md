<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/zhmd/v3.5.5.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

# HSWQ INT8 Linear — comfy_kitchen 0.2.36 Compatibility Fix (Complete)

Target: `patches/comfy_quant_int8.py` — the comfy_kitchen INT8 unaligned-GEMM fallback
Base commit: `ea46d3d` (v3.5.4 release note)
Scope: `ea46d3d..HEAD`

---

## ① Symptom

A workflow that had been running stopped with:

```
[ERROR] !!! Exception during processing !!!
_patch_comfy_kitchen_int8_gemm_fallback.<locals>._safe_cuda_int8_linear()
    got an unexpected keyword argument 'input_act_weight'
```

The traceback ends inside torch-inductor compiled code at the `torch.ops.comfy_kitchen.int8_linear` call:

```
File ".../comfy_ops.py", line 59, in call
    buf0 = torch.ops.comfy_kitchen.int8_linear.default(arg0_1, arg1_1, ...,
        convrot_groupsize=256, input_act=None, input_act_weight=None,
        input_act_eps=0.0, residual=None, residual_scale=None)
...
TypeError: _patch_comfy_kitchen_int8_gemm_fallback.<locals>._safe_cuda_int8_linear()
    got an unexpected keyword argument 'input_act_weight'
```

The path in the reported case was **Z Image / Lumina2 ConvRot NVFP4 LoRA + USDU** (ComfyUI-HSWQ-Loader-and-Tools `nunchaku_usdu` → `ultimate_sd_upscale` → `comfy.sample`).

---

## ② Root cause

### ②-1. `comfy_kitchen` 0.2.36 changed the `int8_linear` signature

`comfy_kitchen` **0.2.36** (installed 2026-09-30) extended both the op and the CUDA backend:

```python
# comfy_kitchen/backends/cuda/__init__.py (0.2.36)
def int8_linear(
    x, weight, weight_scale, bias=None, out_dtype=None,
    convrot=False, convrot_groupsize=256,
    input_act=None,
    input_act_weight=None,      # <-- new
    input_act_eps=0.0,          # <-- new
    residual=None,              # <-- new
    residual_scale=None,        # <-- new
) -> torch.Tensor:
```

`torch.ops.comfy_kitchen.int8_linear` gained the same four parameters, and ComfyUI core
`comfy/ops.py` (`linear_input_act`) now passes them through.

### ②-2. HSWQ's fallback wrapper still declared the old 8-argument signature

`_patch_comfy_kitchen_int8_gemm_fallback()` in `patches/comfy_quant_int8.py` replaces
`comfy_kitchen.backends.cuda.int8_linear` with a guarded wrapper (the fallback that
dequantizes non-multiple-of-4 K/N layers to float instead of crashing `cublas_gemm_int8`).
That wrapper was written against the old signature and forwarded a fixed keyword set:

```python
def _safe_cuda_int8_linear(x, weight, weight_scale, bias=None, out_dtype=None,
                           convrot=False, convrot_groupsize=256, input_act=None):
    ...
    return orig_cuda_int8_linear(x=..., input_act=input_act)  # 'input_act_weight' unknown
```

The moment ComfyUI core passed `input_act_weight=...`, the wrapper raised `TypeError` before the real kernel was ever reached. Because the op is wrapped by `torch.compile`, the error surfaced at the inductor call site.

### ②-3. Why only the INT8 Linear path broke

`int8_linear` is the only comfy_kitchen op whose signature changed in 0.2.36, and HSWQ's
fallback wrapper is the only place that re-declares that signature. All other HSWQ
comfy_kitchen interactions survived:

- **NVFP4 / W4A4 paths** call `ck.scaled_mm_nvfp4(...)` / `ck.quantize_nvfp4(...)` with **keyword
  arguments**, so extra parameters do not break them.
- **mm / addmm INT8 dispatchers** keep their own `except Exception -> dequantize` fallback, so a
  wrapper `TypeError` on the Linear path does not take them down.

---

## ③ Fix

`patches/comfy_quant_int8.py` — `_patch_comfy_kitchen_int8_gemm_fallback()`, step 1
(`comfy_kitchen.backends.cuda.int8_linear` patch):

1. **Accept the new arguments** — `input_act_weight`, `input_act_eps`, `residual`,
   `residual_scale` — plus `**extra_kwargs` so a future parameter addition cannot raise
   `TypeError` again.
2. **Forward only what the installed op declares** — the wrapper introspects the real
   signature once:

   ```python
   _orig_param_names = set(inspect.signature(orig_cuda_int8_linear).parameters)
   ```

   and only adds a kwarg to the call when its name is present. Old `comfy_kitchen` (no such
   parameters) and new `comfy_kitchen` (0.2.36+) therefore both work from the same code.
3. **Keep the dequantized fallback faithful** — the unaligned / non-CUDA fallback now applies
   `input_act` and `residual` the same way the real op does (via `ck_cuda._apply_input_act` /
   `ck_cuda._apply_residual`, guarded so an unavailable helper simply leaves the tensor
   unchanged). The fallback was folded into one `_dequantized_linear()` closure, so the aligned
   and exception paths share it.

Behaviour is otherwise unchanged: **INT8 checkpoints stay INT8 in VRAM**; only the unaligned /
non-CUDA path dequantizes to float, exactly as before.

---

## ④ Verification performed

- **Op-level (real 0.2.36)**: called `torch.ops.comfy_kitchen.int8_linear` with the full new
  argument set and with minimal kwargs — **both succeed**.
- **Wrapper-level (stubbed CUDA backend)**: the patched wrapper forwards
  `input_act` / `input_act_weight` / `input_act_eps` / `residual` / `residual_scale` correctly,
  ignores an unknown future kwarg without raising, and still handles an old-style call.
- **Scope sweep**: all ~45 `comfy_kitchen` op schemas enumerated — `int8_linear` is the only
  signature change; every `comfy_kitchen.tensor` / `comfy_kitchen.backends.cuda` symbol HSWQ
  imports still exists in 0.2.36; HSWQ patch modules import cleanly against 0.2.36.

---

## ⑤ Files changed

| File | Change |
| --- | --- |
| `patches/comfy_quant_int8.py` | INT8 GEMM fallback wrapper follows the `int8_linear` signature (accept new args + `**extra_kwargs`, forward declared params only, faithful `input_act` / `residual` in the dequant fallback) |
| `__init__.py` / `pyproject.toml` | version `3.5.4` -> `3.5.5` |
| `changelog.md` (EN) / `zhmd/CHANGELOG.md` (ZH) | v3.5.5 entries |
| `md/2026-09-30_hswq_speed_vram_optimization_plan.md` | speed / VRAM optimization plan note (docs) |

---

## ⑥ Upgrade notes

- `comfy_kitchen >= 0.2.36` is what exposed the bug; the fix is backward compatible and needs
  **no** change on the `comfy_kitchen` side.
- Restart ComfyUI after updating so the patched fallback is installed at startup.
- No checkpoint re-conversion is required.
