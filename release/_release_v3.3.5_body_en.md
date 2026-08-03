<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><font color="#4b5563"><b>EN</b></font></td>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/zhmd/v3.3.5.md"><font color="#ffffff"><b>中文</b></font></a></td>
  </tr>
</table>

## Overview

**v3.3.5** is a large post-**v3.3.4** product release for **Z Image / ZIT ConvRot NVFP4** coexistence with **SDXL ConvRot NVFP4 / INT8**. It does **not** change the Distorch Hadamard storage gate from v3.3.4. It separates identities, peels shared package contamination, and clears Z Image hooks before every SDXL product load so LoRA bake and TC Linear.forward stay on the SDXL path after Z Image.

Full walkthrough (problem → path → full source → meaning) from baseline `a9d372…`: [`md/HSWQ_FROM_a9d372_PROBLEM_COUNTERMEASURES_GUIDE.md`](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/md/HSWQ_FROM_a9d372_PROBLEM_COUNTERMEASURES_GUIDE.md)

Included commits (tag tip `fba715c` and docs on `main` after the tag): `6b52de2`, `916bb89`, `f030d71`, `a533656`, `fba715c`, plus Chinese guide / switcher follow-ups.

---

## ① What was wrong

### A. Shared `nodes/nvfp4` for two different beings

Z Image ConvRot NVFP4 needs **stock MixedPrecision GEMM + online act rotate** (`_hswq_nvfp4_convrot_parity`). SDXL ConvRot NVFP4 needs the **TC Linear.forward** product stack (`_hswq_nvfp4_full_forward` / `_hswq_nvfp4_product_tc`). Putting both under one package caused parity overlays, VER=8 INT8-protect Linear bake, and Dynamic.load bake hooks to survive into the next SDXL session.

### B. One dropdown string for two loaders

Both UNet (Z Image) and Checkpoint (SDXL) offered **`ConvRot NVFP4`**. Selecting the same string armed the wrong stack or left Z Image bake on SDXL.

### C. SDXL after Z Image → salt-pepper / LoRA fall-off

Owner sequence **SDXL → Z Image → SDXL**:

1. After Z Image, `comfy_parity` stayed on `ops._load_quantized_module` / `mixed_precision_ops`.
2. ZI `ModelPatcherDynamic.load` bake hijacked SDXL → INT8 protect on SDXL NVFP4, `nvfp4_baked=0`, salt-pepper.
3. Later: ZI VER=8 `[HSWQ ConvRot LoRA] int8_protect` on SDXL INT8 → LoRA strength collapses on the **3rd prompt**.
4. Peeling `ops` wrappers alone was not enough: Z Image mutates `mp0.Linear` **in place**; live `convert_weight` / `set_weight` stayed VER=8.

---

## ② Files added / modified

| Path | Change |
|------|--------|
| `nodes/zimage_nvfp4/**` | New package: Z Image parity load, `zi_comfy_quant_nvfp4`, ZI LoRA bake, `ZI_NVFP4_WEIGHT_DTYPE` |
| `nodes/nvfp4/**` | SDXL TC product only; clear ZI contamination on SDXL entry |
| `nodes/nvfp4/comfy_quant_nvfp4.py` | `_clear_zimage_parity_contamination_for_sdxl`; `NVFP4_WEIGHT_DTYPE = "ConvRot NVFP4"` |
| `nodes/nvfp4/nvfp4_forward.py` | `peel_all_nvfp4_linear_lora_bake` / `attach_nvfp4_linear_lora_bake` |
| `nodes/zimage_nvfp4/load_unet.py` | `ZI_NVFP4_WEIGHT_DTYPE = "Z Image ConvRot NVFP4"` |
| `nodes/zimage_nvfp4/nvfp4_comfy_parity.py` | `peel_non_product_nvfp4_ops`, `restore_nvfp4_tc_product_stack` |
| `nodes/zimage_nvfp4/nvfp4_lora_bake.py` | `uninstall_zimage_nvfp4_lora_bake` (Dynamic.load / load_models_gpu + Linear peel) |
| `md/HSWQ_FROM_a9d372_PROBLEM_COUNTERMEASURES_GUIDE.md` | Full EN countermeasures guide (P1–P7 + Appendix A) |
| `zhmd/HSWQ_FROM_a9d372_PROBLEM_COUNTERMEASURES_GUIDE.md` | Chinese guide |
| `changelog.md` / `zhmd/CHANGELOG.md` | Version **3.3.5** overview + release URL |
| `zhmd/v3.3.5.md` | Chinese release notes (this structure) |
| `release/_release_v3.3.5_body_en.md` | English GitHub Release body (this file) |

`pyproject.toml` / `__init__.__version__` remain **`3.3.4`** on this tag tip (Manager / Registry string unchanged). Product history and GitHub tag for this work are **v3.3.5**.

---

## ③ Full text of key added / modified code

### 3.1 Separate dropdown identities

**Z Image** (`nodes/zimage_nvfp4/load_unet.py`):

```python
# ZI/Krea UNet dropdown ONLY — never share the SDXL Checkpoint Loader string.
# SDXL uses nodes/nvfp4 NVFP4_WEIGHT_DTYPE == "ConvRot NVFP4" (separate being).
ZI_NVFP4_WEIGHT_DTYPE = "Z Image ConvRot NVFP4"
```

**SDXL** (`nodes/nvfp4/comfy_quant_nvfp4.py`):

```python
# Z Image / Krea UNet uses ZI_NVFP4_WEIGHT_DTYPE == "Z Image ConvRot NVFP4"
# (separate being; never share this string).
NVFP4_WEIGHT_DTYPE = "ConvRot NVFP4"
```

### 3.2 SDXL entry always clears Z Image contamination

```python
def _clear_zimage_parity_contamination_for_sdxl() -> None:
    """Peel Z Image comfy_parity + ZI bake hooks before SDXL TC / INT8 load.

    Owner log (SDXL → Z Image → SDXL): after Z Image, ``comfy_parity`` stays on
    ``ops._load_quantized_module`` / ``mixed_precision_ops`` and ZI Dynamic.load
    bake hijacks SDXL → ``arm INT8 protect`` on SDXL NVFP4, ``nvfp4_baked=0``,
    salt-pepper. Later: ZI VER=8 ``[HSWQ ConvRot LoRA] int8_protect`` on SDXL
    INT8 → LoRA falls off on the 3rd prompt. Restore product TC (or peel to
    stock) and uninstall ZI bake hooks.
    """
    try:
        from ..zimage_nvfp4.nvfp4_comfy_parity import (
            peel_non_product_nvfp4_ops,
            restore_nvfp4_tc_product_stack,
        )

        restore_nvfp4_tc_product_stack()
        try:
            import comfy.ops as ops

            peel_non_product_nvfp4_ops(ops)
        except Exception:
            pass
    except Exception as e:
        logger.warning("[HSWQ NVFP4] restore TC stack for SDXL failed: %s", e)
    try:
        from ..zimage_nvfp4.nvfp4_lora_bake import uninstall_zimage_nvfp4_lora_bake

        uninstall_zimage_nvfp4_lora_bake()
    except Exception as e:
        logger.warning("[HSWQ NVFP4] uninstall ZI bake hooks for SDXL failed: %s", e)
    # Z Image mutates mp0.Linear in place; peel ops wrappers alone leaves VER=8.
    try:
        import comfy.ops as ops

        Lin = ops.mixed_precision_ops().Linear
        peeled = peel_all_nvfp4_linear_lora_bake(Lin)
        mp_fn = ops.mixed_precision_ops
        if getattr(mp_fn, "_hswq_nvfp4_product_tc", False):
            if attach_nvfp4_linear_lora_bake(Lin) or peeled:
                _console(
                    "[HSWQ NVFP4] SDXL product Linear LoRA bake VER=1 on live Linear"
                )
        elif peeled:
            _console(
                "[HSWQ NVFP4] peeled Z Image Linear LoRA bake (int8_protect) "
                "off live Linear — SDXL INT8/stock safe"
            )
    except Exception as e:
        logger.warning(
            "[HSWQ NVFP4] peel live Linear LoRA bake for SDXL failed: %s", e
        )
```

Called at the start of `apply_comfy_quant_nvfp4_patches()` (SDXL path) **before** any early-return / stack install.

### 3.3 Peel foreign `ops` wrappers down to stock or SDXL product_tc

```python
def peel_non_product_nvfp4_ops(ops) -> bool:
    """Peel Z Image / comfy_parity mp+load wrappers down to stock or SDXL product_tc.

    Used when PRODUCT was never saved (INT8-only → Z Image → SDXL) so restore
    cannot reinstate TC, but ZI mp must not keep attaching VER=8 Linear bake.
    """
    changed = False
    cur = getattr(ops, "mixed_precision_ops", None)
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if getattr(cur, "_hswq_nvfp4_product_tc", False):
            if ops.mixed_precision_ops is not cur:
                ops.mixed_precision_ops = cur
                changed = True
            break
        is_foreign = bool(
            getattr(cur, "_hswq_nvfp4_comfy_only", False)
            or (
                getattr(cur, "_hswq_nvfp4_stack_ver", 0)
                and not getattr(cur, "_hswq_nvfp4_product_tc", False)
            )
        )
        if not is_foreign:
            if ops.mixed_precision_ops is not cur:
                ops.mixed_precision_ops = cur
                changed = True
            break
        nxt = getattr(cur, "_hswq_nvfp4_orig_mp", None) or getattr(
            cur, "_hswq_orig_mixed_precision_ops", None
        )
        if nxt is None:
            break
        ops.mixed_precision_ops = nxt
        changed = True
        cur = nxt

    cur_l = getattr(ops, "_load_quantized_module", None)
    seen_l: set[int] = set()
    while cur_l is not None and id(cur_l) not in seen_l:
        seen_l.add(id(cur_l))
        if getattr(cur_l, "_hswq_nvfp4_product_tc", False):
            if ops._load_quantized_module is not cur_l:
                ops._load_quantized_module = cur_l
                changed = True
            break
        is_foreign_l = bool(
            getattr(cur_l, "_hswq_nvfp4_comfy_only", False)
            or getattr(cur_l, "_hswq_int8_protect_in_load", False)
            or getattr(cur_l, "_hswq_int8_protect_arm_v2", False)
            or getattr(cur_l, "_hswq_int8_decode_patched", False)
            or (
                getattr(cur_l, "_hswq_nvfp4_full_load", False)
                and not getattr(cur_l, "_hswq_nvfp4_product_tc", False)
            )
        )
        if not is_foreign_l:
            if ops._load_quantized_module is not cur_l:
                ops._load_quantized_module = cur_l
                changed = True
            break
        nxt_l = _closure_named(cur_l, "orig_load") or _closure_named(
            cur_l, "original_load"
        )
        if nxt_l is None:
            nxt_l = getattr(cur_l, "_hswq_nvfp4_orig_load", None)
        if nxt_l is None:
            break
        ops._load_quantized_module = nxt_l
        changed = True
        cur_l = nxt_l
    return changed
```

### 3.4 Peel in-place Linear bake (VER=8) that ops peel cannot undo

```python
def peel_all_nvfp4_linear_lora_bake(Lin) -> bool:
    """Strip every HSWQ Linear bake wrap down to stock convert/set.

    Z Image ``install_nvfp4_comfy_parity`` mutates ``mp0.Linear`` in place
    (VER=8 ``[HSWQ ConvRot LoRA] int8_protect``). Peeling
    ``ops.mixed_precision_ops`` alone does not undo that class mutation, so
    SDXL INT8 after Z Image still bakes through ZI wraps and LoRA falls off
    on the 3rd prompt. Call this from SDXL clear / ZI uninstall.
    """
    changed = False
    for attr in ("convert_weight", "set_weight"):
        fn = getattr(Lin, attr, None)
        if not callable(fn):
            continue
        if int(getattr(fn, "_hswq_nvfp4_lora_bake_ver", 0) or 0) <= 0:
            continue
        stock = _peel_lora_bake_wrap(fn)
        if stock is not fn:
            setattr(Lin, attr, stock)
            changed = True
    return changed
```

### 3.5 Uninstall Z Image Dynamic / GPU bake hooks

`uninstall_zimage_nvfp4_lora_bake()` (full function in `nodes/zimage_nvfp4/nvfp4_lora_bake.py`):

- Deep-cleans `ModelPatcherDynamic.load` (including INT8 wraps that captured ZI as `true_orig`)
- Clears `load_models_gpu` ZI bake hook
- Reinstalls a **clean** INT8 Dynamic.load bake when a contaminated INT8 wrap was discarded
- Calls `peel_all_nvfp4_linear_lora_bake` on live `Linear`

### 3.6 Z Image arm path (package peel)

```python
def apply_nvfp4_patches() -> None:
    """Arm Z Image ConvRot NVFP4 (parity) + INT8 load (core ConvRot)."""
    from .zi_comfy_quant_nvfp4 import apply_comfy_quant_nvfp4_patches
    from ...patches.comfy_quant_int8 import apply_comfy_quant_int8_patches
    from .nvfp4_comfy_parity import (
        apply_nvfp4_comfy_parity,
        require_convrot_parity_forward,
    )
    from .nvfp4_lora_bake import install_zimage_nvfp4_lora_bake

    if not apply_comfy_quant_nvfp4_patches():
        raise RuntimeError(...)
    if not apply_nvfp4_comfy_parity():
        raise RuntimeError(...)
    require_convrot_parity_forward()
    apply_comfy_quant_int8_patches()
    if not install_zimage_nvfp4_lora_bake(force=True):
        raise RuntimeError(...)
```

All of the above lives under `nodes/zimage_nvfp4`. SDXL continues under `nodes/nvfp4` only.

---

## ④ Meaning

| Piece | Meaning |
|-------|---------|
| Package peel | Z Image parity / ZI bake cannot edit SDXL TC files by accident; two trees, two identities. |
| `Z Image ConvRot NVFP4` vs `ConvRot NVFP4` | Dropdown / branch never share a string; bake and forward stacks cannot be confused by UI. |
| `_clear_zimage_parity_contamination_for_sdxl` | Every SDXL product entry restores TC (or peels to stock), uninstalls ZI bake, peels VER=8 Linear, re-attaches SDXL VER=1 when product_tc is live. |
| `peel_non_product_nvfp4_ops` | Walks `_hswq_nvfp4_orig_*` chains; stops at `product_tc` or non-foreign stock. Covers INT8-only → ZI → SDXL when PRODUCT was never saved. |
| `peel_all_nvfp4_linear_lora_bake` | Fixes the **in-place class mutation** that ops peel misses — required for 3rd-prompt INT8 LoRA after ZI. |
| `uninstall_zimage_nvfp4_lora_bake` | Stops ZI Dynamic.load / GPU bake from wrapping SDXL; discards INT8 wraps that nested ZI as `true_orig`. |
| Countermeasures guide | Audit document for P1–P7 (parity vs TC, DistOrch rotate, Hadamard poison, hybrid INT8 bake, package peel, dropdown, SDXL clear). |

### Operator notes

1. Update to tag **v3.3.5** (or current `main` that includes the Chinese guide).
2. Restart ComfyUI completely.
3. Z Image / ZIT UNet: select **`Z Image ConvRot NVFP4`** (not the SDXL string).
4. SDXL Checkpoint: select **`ConvRot NVFP4`** as before.
5. After Z Image, load SDXL again — console should show contamination clear / Linear peel / product VER=1 reattach as applicable; no ZI ENTER on SDXL bake.
6. Keep **General Purge VRAM V2** (`HSWQ` on) when using HSWQ NVFP4 / INT8 (v3.3.4 Hadamard gate still applies).

### Compatibility

| Item | Policy |
|------|--------|
| Quantizer | **Only** [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization) |
| Z Image / ZIT ConvRot NVFP4 | `nodes/zimage_nvfp4` — stock GEMM + online act rotate |
| SDXL ConvRot NVFP4 | `nodes/nvfp4` — TC Linear.forward product path |
| SDXL / ZI INT8 | Product INT8 path; ZI VER=8 protect must not remain on live Linear after clear |
| ComfyUI-master | Not modified by this extension |
| v3.3.4 Distorch Hadamard `_tensor_storage_ok` | Unchanged; still required for 2nd+ gens after purge |
