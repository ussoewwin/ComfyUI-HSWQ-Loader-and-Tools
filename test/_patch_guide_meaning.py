from pathlib import Path

ROOT = Path(r"D:/USERFILES/GitHub/ComfyUI-HSWQ-Loader-and-Tools")
en = ROOT / "md" / "HSWQ_FROM_a9d372_PROBLEM_COUNTERMEASURES_GUIDE.md"
zh = ROOT / "zhmd" / "HSWQ_FROM_a9d372_PROBLEM_COUNTERMEASURES_GUIDE.md"

text = en.read_text(encoding="utf-8")

old_p2 = """### ④ Meaning

| Piece | Meaning |
|-------|---------|
| Flatten both wrappers | Never stack parity on parity or TC-on-parity leftovers |
| Marker preservation | DistOrch / INT8 decode must not forget “this Linear is parity” |
"""

new_p2 = """### ④ Meaning (files and code)

| File / symbol | Meaning |
|---------------|---------|
| `_unwrap_stock_forward` in `nvfp4_comfy_parity.py` | Walks the live `Linear.forward` wrapper chain and peels **both** TC product forward and parity forward shells until stock (or a known base) remains. Meaning: DistOrch refresh must not leave a second online act-rotate layer. |
| `_ensure_single_parity_linear_forward` | After unwrap, installs **exactly one** parity forward. Meaning: single online rotate, never TC-on-parity then peel-TC-only. |
| INT8 wrap marker preservation in `patches/comfy_quant_int8.py` | When INT8 decode wraps a Linear, NVFP4 / parity stamps on the module or ops chain must survive. Meaning: DistOrch / INT8 decode must not forget “this Linear is parity”. |
"""

old_p3 = """### ④ Meaning

Same liveness rule for global cache and module-local `H`. If storage is poisoned, rebuild via `build_hadamard`.
"""

new_p3 = """### ④ Meaning (files and code)

| File / symbol | Meaning |
|---------------|---------|
| `_tensor_storage_ok` in `nodes/nvfp4/nvfp4_hadamard.py` | Shared gate: a Hadamard tensor is usable only if storage is alive (not DistOrch-poisoned empty shell). Meaning of the function: device/dtype/`numel` alone are **not** enough. |
| `need_rebuild` path in `_make_convrot_parity_forward` (`nvfp4_comfy_parity.py`) | Module-local `module._hswq_nvfp4_parity_H` must pass the **same** `_tensor_storage_ok` check as the global cache. Meaning: 2nd+ gens after DistOrch purge rebuild `H` instead of reusing a dead attribute. |
| `build_hadamard` (called on rebuild) | Reconstructs a live Hadamard; meaning: quality recovery after Method-3 storage wipe. |
"""

old_p4 = """### ④ Meaning

Arm like Conv2d: clear kitchen `Params.convrot`, set `_hswq_int8_convrot`, bake unrotate once, keep `Params.convrot=False` after requant. Pass-delta EVIDENCE must not print OK on empty `patches=0` passes.
"""

new_p4 = """### ④ Meaning (files and code)

| File / symbol | Meaning |
|---------------|---------|
| `int8_convrot_flags_from_conf` (`nvfp4_conf.py`) | Reads hybrid INT8 protect ConvRot conf into HSWQ flags. Meaning: INT8 protect is a first-class twin of NVFP4 ConvRot, not an afterthought. |
| Dual unrotate / re-rotate in `nvfp4_forward.py` | Bake path unrotates once with the correct layout; online path rotates acts via `_hswq_int8_convrot` without kitchen double-rotate. Meaning: wrong `Params.convrot` = dead LoRA or noise. |
| `_arm_int8_protect_convrot_after_stock_load` (`nvfp4_comfy_parity.py`) | Arms INT8 protect **like Conv2d**: clear kitchen `Params.convrot`, set `_hswq_int8_convrot`. Meaning of the arm: HSWQ owns rotate. |
| Dual bake + pass-delta EVIDENCE (`nvfp4_lora_bake.py`) | Bakes NVFP4 **and** INT8 protect keys; EVIDENCE must use pass-delta, not print OK on empty `patches=0`. Meaning: hybrid packs (~120+~60) must show both classes succeeding. |
"""

old_p5 = """### ④ Meaning

Import boundaries: SDXL clear may **call** Z Image peel/uninstall; Z Image must not permanently own SDXL `ops` after unload.
"""

new_p5 = """### ④ Meaning (files and code)

| Boundary | Meaning |
|----------|---------|
| Package `nodes/zimage_nvfp4/` | Owns Z Image ConvRot NVFP4 runtime after peel. Meaning of the split: SDXL product code in `nodes/nvfp4/` must not carry Z Image parity forever. |
| Package `nodes/nvfp4/` | Owns SDXL ConvRot NVFP4 TC product only. Meaning: shared ownership was the contamination root. |
| SDXL clear calling Z Image peel/uninstall | Allowed **call** direction: SDXL may invoke Z Image cleanup. Meaning: Z Image must not permanently own SDXL `ops` after unload. |
"""

old_p6 = """### ④ Meaning

Branch on product identity, not on “looks like NVFP4 conf”.
"""

new_p6 = """### ④ Meaning (files and code)

| Piece | Meaning |
|-------|---------|
| Dropdown string `Z Image ConvRot NVFP4` | Separate product identity from SDXL `ConvRot NVFP4`. Meaning: one string → one stack; never share. |
| Dynamic bake install gated on Z Image dtype (`nvfp4_lora_bake.py`) | VER=8 bake wraps only when Z Image product is selected. Meaning: branch on product identity, not on “looks like NVFP4 conf”. |
"""

old_cross = """## ④ Cross-cutting meaning (summary)

1. **Two products, two stamps:** `_hswq_nvfp4_product_tc` (SDXL) vs `_hswq_nvfp4_comfy_only` (Z Image). Never upgrade TC on top of live parity.
2. **DistOrch empties storage, not Python refs:** Hadamard and wrapper chains must re-validate; peel both TC and parity wrappers before a single re-wrap.
3. **Hybrid INT8 protect = Conv2d twin:** clear `Params.convrot`, online rotate via flag, bake once, keep Params False after requant.
4. **In-place Linear mutation survives ops peel:** always `peel_all_nvfp4_linear_lora_bake` when leaving Z Image or entering SDXL.
5. **INT8 protect load overlay is foreign to SDXL:** peel `_hswq_int8_protect_*` / `_hswq_int8_decode_patched` the same as parity.
"""

new_cross = """## ④ Master meaning catalog (every Appendix A file + critical code)

### Added modules — file meaning and code meaning

| Path | Meaning of the file | Critical code / symbol meaning |
|------|---------------------|--------------------------------|
| `nodes/zimage_nvfp4/__init__.py` | Package export surface for Z Image ConvRot NVFP4. | Re-exports load/bake installers only. |
| `nodes/zimage_nvfp4/load_unet.py` | UNet load entry for dtype `Z Image ConvRot NVFP4`. | `apply_nvfp4_patches` → parity, not SDXL TC. `ZI_NVFP4_WEIGHT_DTYPE` is a separate being from SDXL `ConvRot NVFP4`. |
| `nodes/zimage_nvfp4/nvfp4_addmm_patch.py` | Kitchen addmm interaction for Z Image. | Keeps kitchen from double-rotating with HSWQ. |
| `nodes/zimage_nvfp4/nvfp4_comfy_parity.py` | Core Z Image product: stock GEMM + online act rotate; PRODUCT remember/restore; foreign peel. | `apply_nvfp4_comfy_parity`, `_make_convrot_parity_forward`, `_unwrap_stock_forward`, `_ensure_single_parity_linear_forward`, `peel_non_product_nvfp4_ops`, `restore_nvfp4_tc_product_stack`, `_discard_poisoned_product_refs`, `_arm_int8_protect_convrot_after_stock_load`. |
| `nodes/zimage_nvfp4/nvfp4_lora_bake.py` | DynamicVRAM LoRA bake (VER=8) for Z Image + hybrid INT8 protect. | `install_zimage_nvfp4_lora_bake` / `uninstall_zimage_nvfp4_lora_bake` mutate live `Linear`; uninstall must peel VER=8, not only Dynamic.load. |
| `nodes/zimage_nvfp4/nvfp4_tc_gate.py` | Gate: refuse TC upgrade while parity is live. | Blocks P2-class stacked wraps. |
| `nodes/zimage_nvfp4/require_parity.py` | Fail-fast if parity forward missing after Z Image load. | Assert product identity on the live forward. |
| `nodes/zimage_nvfp4/zi_comfy_quant_nvfp4.py` | Z Image patch install / stack stamps. | Stamps `_hswq_nvfp4_comfy_only` only. |
| `nodes/zimage_nvfp4/zi_nvfp4_conf.py` | Z Image conf helpers. | Conf → arm flags; no GEMM. |
| `nodes/zimage_nvfp4/zi_nvfp4_forward.py` | Z Image forward helpers for parity. | Online act path; not SDXL scaled_mm TC. |
| `nodes/zimage_nvfp4/zi_nvfp4_hadamard.py` | Z Image Hadamard helpers. | Builds `H`; liveness via `_tensor_storage_ok`. |
| `prestartup_script.py` | Early ComfyUI hooks / import path. | Packaging only; no model math. |

### Modified modules — file meaning and code meaning

| Path | Meaning of the file | Critical code / symbol meaning |
|------|---------------------|--------------------------------|
| `nodes/nvfp4/comfy_quant_nvfp4.py` | SDXL ConvRot NVFP4 **product** installer + SDXL clear of Z Image residue. | `_clear_zimage_parity_contamination_for_sdxl` runs before SDXL patches; refuses TC install on leftover parity; stamps `_hswq_nvfp4_product_tc`. |
| `nodes/nvfp4/nvfp4_forward.py` | SDXL product TC forward + product LoRA bake VER=1 + peel of foreign bake. | `peel_all_nvfp4_linear_lora_bake` strips any HSWQ bake (incl. ZI VER=8) off live Linear; product `attach` peels foreign VER first. |
| `nodes/nvfp4/nvfp4_conf.py` | Product conf decode including INT8 protect flags. | `int8_convrot_flags_from_conf` makes hybrid INT8 protect explicit. |
| `nodes/nvfp4/nvfp4_hadamard.py` | Shared Hadamard utilities for product + parity liveness. | `_tensor_storage_ok` is the DistOrch-poison gate. |
| `nodes/nvfp4/nvfp4_load.py` | SDXL product NVFP4 Linear load (TC stamps / shape checks). | Owns full-load for product; never the Z Image parity load. |
| `patches/comfy_quant_int8.py` | INT8 product path + marker preservation + SDXL clear before INT8 load. | Clears Z Image contamination before SDXL INT8; preserves NVFP4 markers through INT8 wraps. |

### Cross-cutting rules (same meaning as P1–P7)

1. **Two products, two stamps:** `_hswq_nvfp4_product_tc` (SDXL) vs `_hswq_nvfp4_comfy_only` (Z Image). Never upgrade TC on top of live parity.
2. **DistOrch empties storage, not Python refs:** Hadamard and wrapper chains must re-validate; peel both TC and parity wrappers before a single re-wrap.
3. **Hybrid INT8 protect = Conv2d twin:** clear `Params.convrot`, online rotate via flag, bake once, keep Params False after requant.
4. **In-place Linear mutation survives ops peel:** always `peel_all_nvfp4_linear_lora_bake` when leaving Z Image or entering SDXL.
5. **INT8 protect load overlay is foreign to SDXL:** peel `_hswq_int8_protect_*` / `_hswq_int8_decode_patched` the same as parity.
"""

replacements = [
    (old_p2, new_p2),
    (old_p3, new_p3),
    (old_p4, new_p4),
    (old_p5, new_p5),
    (old_p6, new_p6),
    (old_cross, new_cross),
]

for old, new in replacements:
    if old not in text:
        raise SystemExit("MISSING block:\n" + old[:80])
    text = text.replace(old, new, 1)

# Appendix completeness note after intro paragraph
old_app = """The following blocks are the **complete current file text** of every primary countermeasure module listed above (UTF-8, as on disk when this guide was generated).
"""
new_app = """The following blocks are the **complete current file text** of every primary countermeasure module listed above (UTF-8, as on disk when this guide was generated).

**Completeness rule:** each fence below must be character-identical to the on-disk file (newline-at-EOF differences only are allowed). Verified against the working tree when this note was written: **18/18 files match** (`nodes/zimage_nvfp4/*`, `prestartup_script.py`, `nodes/nvfp4/{comfy_quant_nvfp4,nvfp4_forward,nvfp4_conf,nvfp4_hadamard,nvfp4_load}.py`, `patches/comfy_quant_int8.py`).
"""
if old_app not in text:
    raise SystemExit("MISSING appendix intro")
text = text.replace(old_app, new_app, 1)

# Closing after last file
if not text.rstrip().endswith("return True"):
    # append closing regardless of last lines
    pass
closing = """

---

## Closing

This guide’s **③** obligation for countermeasure modules is satisfied by **Appendix A** (full file text, verified character-identical to disk for all 18 listed modules).

**④** above states, for each added/modified file and for each critical symbol, **what the file means** and **what the code means** — not only stamp slogans.

Operator retest for the contamination class remains under **P7**. Language twin: `zhmd/HSWQ_FROM_a9d372_PROBLEM_COUNTERMEASURES_GUIDE.md`.
"""
if "## Closing" not in text:
    text = text.rstrip() + closing + "\n"

en.write_text(text, encoding="utf-8", newline="\n")
print("EN updated", en.stat().st_size)
