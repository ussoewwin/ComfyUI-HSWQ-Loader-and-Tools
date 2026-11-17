<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/zhmd/v3.7.3.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

# ComfyUI-HSWQ-Loader-and-Tools v3.7.3

## Overview

**A new loader node for SAM3 (Segment Anything 3) checkpoints, with ConvRot INT8 support and a CLIP output.** The previous releases shipped the SAM3 model as a node inside the HSWQ quantization pack, but it was never documented as a loader. v3.7.3 documents and ships it as a first-class loader node, usable from any folder, with the checkpoint's text encoder exposed as a `CLIP` output.

`HSWQ SAM3 Loader (ConvRot INT8)` loads ConvRot / TensorWise INT8-quantized SAM3 checkpoints directly into VRAM in 8-bit precision (`QuantizedTensor` / `TensorWiseINT8Layout`) and runs them through `comfy_kitchen`'s `int8_linear` kernel with online activation rotation (`convrot`). The model file and the folder it is loaded from are selected separately (`sam3_name` + required `sam3_dir`); the file dropdown follows the folder typed into `sam3_dir`. The node returns the SAM3 `model` (MODEL) plus the checkpoint's text encoder `clip` (CLIP), which can be fed into `CLIPTextEncode` for text-conditioned detection.

Layers whose dimensions are not a multiple of 4 (such as `boxRPB_embed_x` with `K=2`) are handled safely: those layers compute in float precision while all heavy backbone and transformer blocks stay in accelerated INT8 Tensor Core precision. Non-INT8 (FP16) SAM3 checkpoints load normally with no `MixedPrecisionOps` attached, so the same node works for both INT8 and FP16 packs.

---

## Features

- **Drop-in loader for SAM3 ConvRot INT8 packs**: inputs `sam3_name` + `sam3_dir`, outputs `model` (MODEL) + `clip` (CLIP).
- **Arbitrary folder**: `sam3_dir` is required. The model is loaded from that folder only; an empty folder is rejected, with no fallback to another folder.
- **CLIP output**: the checkpoint's text encoder is exposed as a standard ComfyUI `CLIP` output, ready for `CLIPTextEncode` (text-conditioned detection).
- **Automatic hardware safety fallback**: unaligned layers (`K % 4 != 0`) compute in float precision without crashing cuBLAS INT8 GEMM.
- **Native INT8 VRAM retention**: weights stay in 8-bit (`TensorWiseINT8Layout`), cutting memory requirements significantly.
- **Seamless FP16 compatibility**: FP16 SAM3 checkpoints load with stock ComfyUI settings (no `MixedPrecisionOps`, no behavior change).

---

## Compatibility

- **HSWQ SAM3 Detect**: the loader's `model` output feeds `HSWQ SAM3 Detect` (category `HSWQ/Detection`) for mask / bbox / image outputs.
- **Stock ComfyUI SAM3 detection / tracking nodes**: the `MODEL` output is a standard ComfyUI model.
- Only checkpoints produced by [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization) are in scope; other third-party SAM3 quantization packs are not supported.

---

## Links

- **Changelog**: see [`changelog.md`](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/changelog.md) (v3.7.3 entry).
- **Chinese release notes**: [`zhmd/v3.7.3.md`](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/zhmd/v3.7.3.md).
- **Previous release**: [v3.7.2](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.7.2).
