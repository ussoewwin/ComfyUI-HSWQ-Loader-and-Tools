<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/zhmd/v3.7.6.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

## Overview

**Flux1 hybrid packs now load through the HSWQ UNet Loader with a dedicated `Flux1 ConvRot W4A8` weight dtype.** A Flux1 hybrid pack carries 170 `asym_w4a8_int8` layers plus 144 `int8_tensorwise` protect layers in a single file (the measured pack `F1D_unlockV3_hswq_r170_hybid_convrot_w4a8.safetensors` holds 8,989,783,965 bytes across 1,748 keys). The packed 4-bit weights are never rewritten: the INT8 protect layers get a requantizing LoRA bake, and every W4A8 layer adds the LoRA delta as a factorised residual after the stock grouped-int8 GEMM.

Flux1 hybrid ConvRot W4A8 is the Hybrid-Sensitivity-Weighted-Quantization format for Flux1: one safetensors carries two quantization tiers side by side. The heavy transformer linears are quantized to 4-bit weights with 8-bit activations (`asym_w4a8_int8`, weight group size 16 under a group-256 Hadamard activation rotation), while 144 lower-impact layers stay at 8-bit weights as `int8_tensorwise` protect layers, and the boundary embedders (`img_in`, `txt_in`, `vector_in.in_layer`) hold their own INT8 tiers. Sensitivity ranking — not a keep-ratio percentage — decides which layers go to W4A8. The full method and measurement records are in the [Flux1 Hybrid ConvRot W4A8 Technical Guide](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/Flux1_Hybrid_ConvRot_W4A8_Technical_Guide.md).

---

## Features

- **Dedicated dtype plus auto-detect**: `weight_dtype` = `Flux1 ConvRot W4A8` selects the path explicitly, and `default` reaches it through detection with the dispatch order INT8 → NVFP4 → W4A8, so a protected hybrid pack is not stolen by the INT8-only detect.
- **INT8 protect layers keep a real requant bake**: the measured run reports `total=144 requant=144 int8_round=0 shape_skip=0` and `path OK (all requant)` — no layer falls back to the delta-destroying stochastic int8 rounding.
- **W4A8 layers keep the packed weights untouched**: the bake moves each LoRA delta into factorised terms applied after the stock grouped-int8 GEMM. The measured run reports `baked 170 W4A8 layers as residuals (skipped non-W4A8 keys=144)` and `terms_set=170`.
- **Arming order and re-arming**: the loader arms the INT8 stack first and the W4A8 stack second, and re-arming always yields one INT8+W4A8 pair instead of nesting wrappers.
- **Residual execution is logged**: first hit, every 500 hits, and the finished-run count print as `EVIDENCE` lines with `terms_set` / `residual_layers` / `residual_hits`; the load line shows `LoRA bake=pending at patcher load` instead of pre-bake zeros.
- **Measured on the real pack**: a 16-step run with EasyCache skipping 6 steps reports `residual_layers=170` and `residual_hits=1700` (170 layers × 10 effective steps), with no ERROR, Traceback, or BROKEN line.
- **Numerical check against float32 references**: across all 170 W4A8 layers the executed residual matches `x @ (sum of deltas)^T` with cosine min 0.999998450 (mean 0.999998625, max|diff| 1.206112e-02 — the bf16 output rounding), with both LoRAs stacked on all 170 layers; across the 144 INT8 layers the baked weight moves by 1.0007–1.0338 × the delta norm (mean 1.0084) with `cos(W1-W0, delta)` min 0.9041 / mean 0.9886.

---

## Compatibility

- **Pack scope**: the W4A8 dtype is for Flux1 hybrid packs quantized with [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization); it is not a generic 4-bit loader.
- **Unrelated paths unchanged**: Z Image ConvRot NVFP4, Krea2 / SDXL / SD1.5 ConvRot INT8, FP8, and plain FP16 loads take the same code paths as before. The loader node is displayed as **HSWQ UNet Loader** (DisTorch2 variant: **HSWQ UNet Loader (DisTorch2)**).
- **DisTorch2**: the W4A8 dtype is not verified under the DisTorch2 loader yet; Krea2 ConvRot INT8 remains the only `weight_dtype` verified there.

---

## Links

- **Changelog**: see [`changelog.md`](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/changelog.md) (v3.7.6 entry).
- **Technical guide**: [Flux1 Hybrid ConvRot W4A8 Technical Guide](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/Flux1_Hybrid_ConvRot_W4A8_Technical_Guide.md).
- **Chinese release notes**: [`zhmd/v3.7.6.md`](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/zhmd/v3.7.6.md).
- **Previous release**: [v3.7.5](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.7.5).
