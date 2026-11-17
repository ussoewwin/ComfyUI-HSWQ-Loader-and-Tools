<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/zhmd/v3.7.4.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

# ComfyUI-HSWQ-Loader-and-Tools v3.7.4

## Overview

**Fix for the HSWQ SAM3 Loader (ConvRot INT8): a restored workflow no longer reports the model as missing.** In v3.7.3 the loader gained a folder-path input (`sam3_name` + required `sam3_dir`) and a CLIP output. When a saved workflow was reopened, the frontend compared the restored file name against the dropdown's option list; because that option list still belonged to the folder that was set before the restore, the node was marked as a missing model even though the file existed. v3.7.4 re-runs the frontend's missing-model scan once the dropdown's option list for the restored folder is in place, so the node is no longer reported as missing. The file dropdown still follows the folder typed into `sam3_dir`, the model is still loaded from that folder only, and a non-INT8 (FP16) SAM3 checkpoint still loads normally.

---

## Features

- **No false "missing model" on workflow open**: the dropdown's option list is rebuilt for the folder the saved workflow restored, then the missing-model scan runs again, clearing the stale warning.
- **Drop-in loader for SAM3 ConvRot INT8 packs (unchanged)**: inputs `sam3_name` + `sam3_dir`, outputs `model` (MODEL) + `clip` (CLIP).
- **Arbitrary folder (unchanged)**: `sam3_dir` is required. The model is loaded from that folder only; an empty folder is rejected, with no fallback to another folder.
- **CLIP output (unchanged)**: the checkpoint's text encoder is exposed as a standard ComfyUI `CLIP` output, ready for `CLIPTextEncode` (text-conditioned detection).
- **Automatic hardware safety fallback (unchanged)**: unaligned layers (`K % 4 != 0`) compute in float precision without crashing cuBLAS INT8 GEMM.
- **Native INT8 VRAM retention (unchanged)**: weights stay in 8-bit (`TensorWiseINT8Layout`), cutting memory requirements significantly.
- **Seamless FP16 compatibility (unchanged)**: FP16 SAM3 checkpoints load with stock ComfyUI settings (no `MixedPrecisionOps`, no behavior change).

---

## Compatibility

- **HSWQ SAM3 Detect**: the loader's `model` output feeds `HSWQ SAM3 Detect` (category `HSWQ/Detection`) for mask / bbox / image outputs.
- **Stock ComfyUI SAM3 detection / tracking nodes**: the `MODEL` output is a standard ComfyUI model.
- Only checkpoints produced by [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization) are in scope; other third-party SAM3 quantization packs are not supported.

---

## Links

- **Changelog**: see [`changelog.md`](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/changelog.md) (v3.7.4 entry).
- **Chinese release notes**: [`zhmd/v3.7.4.md`](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/zhmd/v3.7.4.md).
- **Previous release**: [v3.7.3](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.7.3).
