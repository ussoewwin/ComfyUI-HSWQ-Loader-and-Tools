<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/zhmd/v3.6.0.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

# ComfyUI-HSWQ-Loader-and-Tools v3.6.0

## Overview

**Krea2 ConvRot NVFP4 support has ended.** The `Krea2 ConvRot NVFP4` `weight_dtype` option was removed from the UNet loaders, and the Krea2 NVFP4 load/forward runtime was deleted: 3 pure-deleted files (783 lines) + 8 NVFP4-only modules (1,984 lines) removed, and 3 retained modules stripped (2,124 -> 1,099 lines combined). The remaining Krea2 runtime was renamed `nodes/krea2_convrot_nvfp4/` -> `nodes/krea2_convrot_int8/` and now serves **Krea2 ConvRot INT8 only** (stock INT8 forward + low-rank LoRA residual). In addition, the leftover SageAttention2 module (`hswq/hswq_sa2_accel.py`, 423 lines), whose `attention_accel` widget had already been removed in v3.5.8, was deleted.

**SDXL ConvRot INT8, SDXL ConvRot NVFP4, Z Image INT8 and Z Image ConvRot NVFP4 are byte-for-byte unchanged** - verified by git blob hash comparison of 27 files under `nodes/nvfp4/`, `nodes/zimage_nvfp4/`, `nodes/sdxl_int8/` plus the INT8 loader nodes between tag `v3.5.9` and `v3.6.0`: zero differences.

> **Loading a Krea2 ConvRot NVFP4 checkpoint is no longer supported.** Saved workflows that still carry the `Krea2 ConvRot NVFP4` value execute without crashing (the loaders absorb unknown keyword arguments via `**kwargs` and fall back to the INT8 / default paths), but the NVFP4-specific load stack that served them is gone. Krea2 support continues via **ConvRot INT8** - the only `weight_dtype` verified with the DisTorch2 loader.

---

## What Was Removed

### 1. UNet Loader option (both loaders, one source)

- `Krea2 ConvRot NVFP4` was deleted from the `weight_dtype` list of **HSWQ ConvRot INT8/ConvRot NVFP4 UNet Loader** (`HSWQFP8E4M3UNetLoader`, `hswq/zimage_fp8_e4m3_unet.py`).
- The **DisTorch2 variant** (`HSWQUNetLoaderDisTorch2`) inherits the same option list through `copy.deepcopy(cls.INPUT_TYPES())` in `wrappers.py`, so one source change removed the option from both loaders.

### 2. Krea2 NVFP4 loader + dead dispatch (783 lines, 3 files)

| File | Lines | Reason |
| :--- | ---: | :--- |
| `nodes/krea2_convrot_nvfp4/load_unet.py` | 182 | Krea2 NVFP4 loader body + `install_krea2_nvfp4_unet_dispatch` wrap - dead once the option was removed |
| `nodes/krea2_convrot_nvfp4/nvfp4_comfy_parity.py` | 178 | parity module whose apply function had zero callers in the repo (full-tree search) |
| `hswq/hswq_sa2_accel.py` | 423 | SageAttention2 module; its `attention_accel` widget and all call sites were already removed in v3.5.8 (`71dd4f4`), zero imports remained |

### 3. NVFP4-only runtime modules (1,984 lines, 8 files)

Deleted from the renamed `nodes/krea2_convrot_int8/` package (line counts measured from the removed blobs):

| File | Lines | NVFP4-only role |
| :--- | ---: | :--- |
| `int8_runtime.py` | 540 | pooled NVFP4 act quantize + `scaled_mm_nvfp4` TC GEMM |
| `kitchen_quant_ops_repair.py` | 386 | kitchen NVFP4 layout-import stub repair |
| `int8_gemm.py` | 270 | FP4 E2M1 unpack + one-shot weight bake |
| `int8_conf.py` | 224 | NVFP4 conf parse / packed-K / txtlayers detection fixes |
| `int8_load.py` | 202 | packed-NVFP4 Linear load (`is_nvfp4_conf` routing) |
| `int8_hadamard.py` | 162 | Hadamard matrices for the NVFP4 rotate wraps |
| `int8_tc_gate.py` | 107 | NVFP4 TensorCore availability gate (SM >= 100) |
| `int8_addmm_patch.py` | 93 | `aten::addmm` -> `hswq_scaled_mm_nvfp4` handler |

### 4. NVFP4 code paths stripped from the 3 retained modules

| File (old -> new) | Before | After |
| :--- | ---: | ---: |
| `nvfp4_forward.py` -> `int8_forward.py` | 671 | 137 |
| `comfy_quant_nvfp4.py` -> `comfy_quant_int8_krea2.py` | 515 | 183 |
| `nvfp4_lora_bake.py` -> `int8_lora_bake.py` | 938 | 779 |

`apply_comfy_quant_nvfp4_patches` was renamed `apply_comfy_quant_int8_patches`; the NVFP4 load routing, packed-K/txtlayers detection fixes, addmm/kitchen hooks, and the NVFP4 first bake pass are gone.

---

## What Remains (Krea2 INT8-only runtime, 4 files)

| File | Role |
| :--- | :--- |
| `comfy_quant_int8_krea2.py` | `apply_comfy_quant_int8_patches()` - installs only the Krea2 `mixed_precision_ops` wrap (stock INT8 forward + LoRA residual) |
| `int8_forward.py` | `make_int8_linear_forward()` - stock forward (comfy_kitchen `int8_linear` + online ConvRot) plus `_add_krea2_lora_residual`, kept verbatim from v3.5.9 with the same `_hswq_krea2_lora_res` storage format |
| `int8_lora_bake.py` | `ModelPatcherDynamic.load` / `load_models_gpu` bake hooks - the remaining-QT (INT8) pass unchanged from v3.5.9 |
| `__init__.py` | package exports (loader-only symbols removed) |

---

## Technical Details

### 1. Why the removed code was a no-op for Krea2 ConvRot INT8 (evidence-based)

Every removed path was gated on NVFP4-only state that the INT8 load path never sets (verified against the `v3.5.9` git source, not from memory):

- NVFP4 load routing: `if is_nvfp4_conf(conf) and (has_nvfp4_scale or dtype == uint8)` - INT8 packs carry `int8_tensorwise` conf and int8 storage, so they always fell through to `_orig_load`.
- `convert_weight` / `set_weight` ConvRot wraps: gated on `getattr(self, "_hswq_nvfp4_convrot", False)` - this flag is set only by the retired Krea2 NVFP4 arm path. INT8 online activation rotation runs through `patches/comfy_quant_int8.py` (a separate mechanism) and was never touched by these wraps.
- packed-K / txtlayers detection fixes: operate on NVFP4-packed projector/Linear storage (`weight_scale_2`); INT8 packs are unaffected.
- addmm handler / TC gate / kitchen repair / FP4 gemm / runtime pool: all keyed on `TensorCoreNVFP4Layout`; INT8 tensors use `TensorWiseINT8Layout`.
- NVFP4 first bake pass: 0 candidates on INT8 packs (no NVFP4 layers), so removing it does not change INT8 bake results; the INT8 residual pass (`bake_remaining_quant_patches_on_dynamic_patcher`) is retained as-is.

### 2. Shared peel protocol preserved (SDXL / Z Image coexistence)

The Krea2 mp wrap keeps the shared stamps read by the SDXL (`nodes/nvfp4/`) and Z Image (`nodes/zimage_nvfp4/`) peel walkers: `_hswq_nvfp4_stack_ver = 2`, `_hswq_nvfp4_orig_mp`, `_hswq_nvfp4_full_forward`, and the closure name `stock_forward` for forward unwrapping. Krea2-only stamps were renamed NVFP4 -> INT8 (`_hswq_krea2_stack` kept; `_hswq_krea2_nvfp4_pack` -> `_hswq_krea2_int8_pack`; `_hswq_krea2_nvfp4_lora_bake` -> `_hswq_krea2_int8_lora_bake`), with setter / getter / purge gate updated consistently in `patches/comfy_quant_int8.py` and `patches/hswq_purge_rearm.py` (measured: 0 stale-name references).

### 3. Verification performed (all measured)

- **0-byte proof for SDXL / Z Image**: `git rev-parse v3.5.9:<path>` vs `HEAD:<path>` over 27 files (`nodes/nvfp4/`, `nodes/zimage_nvfp4/`, `nodes/sdxl_int8/`, INT8 loader nodes) - all blob hashes identical.
- **AST import-symbol check**: 141 relative-import symbols across the changed and dependent modules resolved against module-level definitions - 0 problems.
- `py_compile` on every changed Python file - all OK.
- Loader source contains 0 occurrences of the removed option string; live (`custom_nodes`) tree verified file-by-file: 93 Python files hash-identical to the repo.

---

## File Changes Summary

| File | Change |
| :--- | :--- |
| `hswq/zimage_fp8_e4m3_unet.py` | Modified - `Krea2 ConvRot NVFP4` removed from `weight_dtype` |
| `hswq/hswq_sa2_accel.py` | Deleted (423 lines) |
| `nodes/krea2_convrot_nvfp4/` -> `nodes/krea2_convrot_int8/` | Package renamed (`git mv`, 100% history) |
| `nodes/krea2_convrot_nvfp4/load_unet.py` | Deleted (182 lines) |
| `nodes/krea2_convrot_nvfp4/nvfp4_comfy_parity.py` | Deleted (178 lines) |
| `nodes/krea2_convrot_int8/int8_forward.py` | Renamed + rewritten INT8-only (671 -> 137 lines) |
| `nodes/krea2_convrot_int8/comfy_quant_int8_krea2.py` | Renamed + stripped NVFP4 paths (515 -> 183 lines) |
| `nodes/krea2_convrot_int8/int8_lora_bake.py` | Renamed, NVFP4 pass removed (938 -> 779 lines) |
| `nodes/krea2_convrot_int8/{int8_load,int8_conf,int8_gemm,int8_runtime,int8_tc_gate,int8_addmm_patch,int8_hadamard,kitchen_quant_ops_repair}.py` | Deleted (8 NVFP4-only modules, 1,984 lines) |
| `patches/comfy_quant_int8.py` | Modified - Krea2 INT8 branch import/call/stamp names only |
| `patches/hswq_purge_rearm.py` | Modified - krea2 gate reduced to the live mp-stack stamp |
| `prestartup_script.py` | Modified - dead `_krea2_load_module()` import hook removed |
| `README.md` / `zhmd/README.md` | Modified - SA2 section removed; Krea2 ConvRot NVFP4 **ended** table row added; stale option mentions removed |
| `changelog.md` / `zhmd/CHANGELOG.md` | Modified - v3.6.0 entries added |
| `zhmd/v3.6.0.md` | Added | Chinese release notes (this page's 中文 version) |

---

## How to Update

Update via ComfyUI Manager or pull directly in your `custom_nodes/ComfyUI-HSWQ-Loader-and-Tools` directory:
```bash
git pull
```
Restart ComfyUI to apply the changes. After restart, the UNet loader dropdown no longer offers `Krea2 ConvRot NVFP4`, and **Krea2 ConvRot INT8** loading (including LoRA with the low-rank residual bake) behaves exactly as in v3.5.9. As always, place **General Purge VRAM V2** at the end of HSWQ workflows with the `HSWQ` toggle ON.
