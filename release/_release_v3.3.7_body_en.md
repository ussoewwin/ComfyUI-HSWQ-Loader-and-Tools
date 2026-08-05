<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><font color="#4b5563"><b>EN</b></font></td>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/zhmd/v3.3.7.md"><font color="#ffffff"><b>中文</b></font></a></td>
  </tr>
</table>

## Overview

**v3.3.7** is a **license / provenance / packaging** release on top of **v3.3.6** (HSWQ Torch Compile + ZI INT8 peel). It makes this loader repo’s license surface consistent (**GPL-3.0** only), separates **upstream HSWQ AGPL-3.0** from this pack, rewrites third-party provenance without “copy” wording, vendors **Batched Detailer** helpers so **Impact Pack is not required at runtime**, hardens **HSWQ Torch Compile** (always per-block compile; UI toggle removed), and reorders README so the UNet Loader section sits under SDXL Checkpoint Loader.

Tag tip: `93f046a` (`docs: add changelog v3.3.7 license overview and bump version`).

Included commits since **v3.3.6** (`8fe5df5`…`93f046a`):

| Commit | Summary |
|--------|---------|
| `96ff8d6` / `87a9195` | v3.3.6 language switcher + Chinese release notes |
| `d6cd6ec` | bump package version to 3.3.6 |
| `d0f1301` | Torch Compile: always compile transformer blocks; remove UI toggle |
| `43a7f17` | drop Apache-2.0 remnant; project license is GPL-3.0 |
| `01cebd3` | document HSWQ AGPL upstream + KJ Torch Compile provenance |
| `22e834b` | reword USDU / Torch Compile provenance (no “copy” phrasing) |
| `6611cd1` | document Impact Pack provenance for Batched Detailer |
| `86fba26` | move UNet Loader section under SDXL Checkpoint Loader |
| `f3c4b6c` | vendor Batched Detailer helpers; no Impact Pack runtime import |
| `93f046a` | changelog v3.3.7 + bump `pyproject.toml` / `__init__.__version__` to **3.3.7** |

---

## ① License cleanup — this repo is GPL-3.0 only

### What was wrong

Leftover **Apache-2.0** wording and a duplicate license file (`LICENCE.txt`) conflicted with the real project license (**GPL-3.0** in `LICENSE` / `pyproject.toml`). Readers could think the loader pack was Apache-licensed or dual-licensed incorrectly.

### What changed (`43a7f17`)

| Path | Change |
|------|--------|
| `LICENCE.txt` | **Deleted** (Apache-2.0 remnant, 201 lines) |
| `pyproject.toml` | `license` / classifiers aligned to **GPL-3.0** |
| `README.md` / `zhmd/README.md` | License section rewritten |

### Correct license split (important)

| Work | License | Where |
|------|---------|--------|
| **This repository** (ComfyUI loaders / tools) | **GPL-3.0** | `LICENSE`, `pyproject.toml` |
| **Upstream HSWQ** (quantize scripts / method docs) | **AGPL-3.0** | [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization) — **separate repo**; AGPL there does **not** replace this pack’s GPL |

README now states explicitly that HSWQ (the quantization method) is an original work by ussoewwin under **AGPL-3.0**, and that this ComfyUI loader pack is a **related but distinct** repository under **GPL-3.0**.

---

## ② Provenance wording — USDU / Torch Compile / Batched Detailer

### Goal

Document **based-on / developed-from** third-party GPL works **without** “copied from” phrasing, while keeping:

- copyright notices
- GPL share-alike obligations
- clear statement that **runtime install of the original package is not required** when helpers ship in-tree

### USDU (`22e834b`, License section)

**HSWQ Ultimate SD Upscale** was developed based on [ComfyUI_UltimateSDUpscale](https://github.com/ssitu/ComfyUI_UltimateSDUpscale) (ssitu, GPL-3.0), with original HSWQ / FP8 / torch.compile / Auto scale improvements. Shipping `usdu_bundle` in-tree does **not** remove GPL obligations to the ssitu original. Installing the separate UltimateSDUpscale package is **not** required at runtime.

### Torch Compile (`01cebd3`, `22e834b`)

**HSWQ Torch Compile** was developed based on [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) `TorchCompileModelAdvanced` (kijai, GPL-3.0), with HSWQ / USDU / Distorch / Windows inductor hardening. At runtime it calls ComfyUI core `comfy_api.torch_helpers.set_torch_compile_wrapper` and does **not** import the KJNodes package. That does **not** remove copyright / GPL obligation to the KJNodes original. KJ provenance does **not** apply to the HSWQ quantization method itself.

### Batched Detailer (`6611cd1`, then `f3c4b6c`)

**HSWQ Batched Detailer (SEGS)** was developed based on [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) Detailer (SEGS) / DetailerForEach (ltdrdata, GPL-3.0), with original improvements **especially to keep HSWQ quantized UNets usable** (INT8 / NVFP4 ConvRot, Dynamic VRAM, QuantizedTensor paths) while keeping the Impact Pack SEGS interface.

---

## ③ Batched Detailer — no Impact Pack runtime import (`f3c4b6c`)

### What was wrong

The node previously imported `impact.*` at runtime, so ComfyUI-Impact-Pack had to be installed as a **hard dependency** even though this pack already aimed to be standalone for USDU / Torch Compile.

### What changed

| Path | Role |
|------|------|
| `nodes/batched_detailer_lib/__init__.py` | Re-exports `SEG`, `utils`, `core`, `impact_sampling`, `wildcards` |
| `nodes/batched_detailer_lib/utils.py` | SEGS helpers (crop / paste / etc.) |
| `nodes/batched_detailer_lib/core.py` | Detailer core pieces used by the batched loop |
| `nodes/batched_detailer_lib/sampling.py` | KSampler wrapper path used by the detailer |
| `nodes/batched_detailer_lib/wildcards.py` | Wildcard processing (identity `process()` where needed) |
| `nodes/batched_detailer_lib/NOTICE` | Attribution notice |
| `nodes/batched_detailer_lib/IMPACT_PACK_LICENSE.txt` | Full GPL-3.0 text for Impact Pack provenance |
| `nodes/hswq_batched_detailer.py` | Imports **only** `.batched_detailer_lib` — **no** `impact.*` |
| `README.md` / `zhmd/README.md` | Standalone + provenance wording |

### Import shape (after)

```python
from .batched_detailer_lib import (
    SEG,
    impact_sampling,
    utils,
    wildcards,
)
from .batched_detailer_lib import core
```

### Runtime vs workflow wiring

- **Runtime**: Impact Pack package install is **not** required for this node to load.
- **Workflow**: You may still wire SEGS from Impact detectors (or any SEGS producer). That is optional graph wiring, not an import dependency.
- Shipping `nodes/batched_detailer_lib/` does **not** remove GPL obligations to the Impact Pack original.

---

## ④ HSWQ Torch Compile — always per-block compile (`d0f1301`)

### What changed

| Item | Before (v3.3.6 UI) | After (v3.3.7) |
|------|--------------------|----------------|
| `compile_transformer_blocks_only` | BOOLEAN input (default `True`) | **Removed** from `INPUT_TYPES` |
| Block key collection | Only when toggle ON | **Always** `_collect_block_keys(diffusion_model)` |
| Fallback | Entire `diffusion_model` when toggle OFF / empty | Same fallback when no known blocks: `["diffusion_model"]` |
| Old workflows | — | Extra kwargs ignored via `**_kwargs` so old graphs with the removed input do not crash |

### Behavior note

Former default was already **ON**. v3.3.7 makes that the only path: compile per known transformer block container; if none are found, warn and compile `diffusion_model` as a whole.

---

## ⑤ README layout — UNet Loader under SDXL Checkpoint (`86fba26`)

EN / ZH README: **HSWQ ConvRot INT8/ConvRot NVFP4 UNet Loader** section moved to sit under **SDXL Checkpoint Loader**, so loader docs follow the product hierarchy users actually use (Checkpoint first, then UNet).

---

## ⑥ Version bump

| File | Value |
|------|-------|
| `pyproject.toml` `[project].version` | **3.3.7** |
| `__init__.__version__` | **3.3.7** |
| `changelog.md` / `zhmd/CHANGELOG.md` | Version **3.3.7** overview + release URL |

---

## ⑦ Files touched (summary)

| Area | Paths |
|------|-------|
| License | deleted `LICENCE.txt`; `LICENSE` remains GPL-3.0; `pyproject.toml`; README / zhmd README License |
| Provenance docs | README / zhmd README (USDU, Torch Compile, Batched Detailer, HSWQ AGPL note) |
| Batched Detailer | `nodes/hswq_batched_detailer.py`, `nodes/batched_detailer_lib/**` |
| Torch Compile | `nodes/hswq_torch_compile.py` |
| Changelog | `changelog.md`, `zhmd/CHANGELOG.md` |
| Version | `pyproject.toml`, `__init__.py` |

---

## Upgrade notes

1. Update / reinstall this custom node pack to **3.3.7** (Manager or `git pull` on the tagged commit).
2. **Batched Detailer**: Impact Pack is optional. If you only used it for this node’s imports, you can remove that hard requirement; keep Impact Pack only if you still need its detector / SEGS producer nodes.
3. **Torch Compile**: reopen or ignore the removed `compile_transformer_blocks_only` widget; per-block compile is always on.
4. License redistributors: ship **GPL-3.0** for this pack; keep third-party notices (ssitu USDU, KJNodes, Impact Pack); do **not** treat upstream HSWQ AGPL as this loader’s license file.

---

## Related links

- Changelog: [`changelog.md`](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/changelog.md)
- Chinese notes: [`zhmd/v3.3.7.md`](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/zhmd/v3.3.7.md)
- Upstream HSWQ (AGPL-3.0): https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization
- Previous release: [v3.3.6](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.3.6)
