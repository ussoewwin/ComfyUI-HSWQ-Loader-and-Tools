<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/zhmd/v3.6.1.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

# ComfyUI-HSWQ-Loader-and-Tools v3.6.1

## Overview

**SDXL ConvRot NVFP4 development has been officially cancelled upstream.** The `ConvRot NVFP4` `weight_dtype` option has been removed from the `HSWQ Checkpoint Loader (SDXL)` node (`HSWQCheckpointLoaderSDXL`), and the SDXL NVFP4 benchmark script `nodes/nvfp4bench_sdxl.py` (594 lines) has been deleted.

Deterministic latent trajectory measurements upstream showed that 4-bit precision (NVFP4) fails to maintain latent trajectory fidelity on SDXL. Support for SDXL is now fully unified and focused on **ConvRot INT8 (reverse hybrid)**.

The shared `nodes/nvfp4/` package has been renamed to the dedicated `nodes/sdxl_convrot_nvfp4/`, and Z Image NVFP4 runtime code has been consolidated into the self-contained package `nodes/zimage_convrot_nvfp4/`.

Backward compatibility for previously published checkpoints is preserved: when `weight_dtype` is left on **`default`**, safetensors carrying comfy_quant NVFP4 markers are automatically routed through the NVFP4 loader stack.

---

## Technical Reasons for Upstream Cancellation

### 1. Legacy Decoded SSIM Was Masked by VAE Decoding
Previous evaluation relied on decoded image structural similarity against FP16 baselines (SSIM ~0.95). In-depth investigation revealed:
- **Non-linear smoothing in SDXL VAE**: The SDXL VAE decoder possesses aggressive non-linear error suppression and spatial reconstruction capabilities. Tensors suffering severe latent trajectory divergence inside UNet latent space still decoded into superficially clean pixel images.
- **Visual similarity vs latent fidelity**: Decoded SSIM masked latent manifold degradation, which manifested as prompt drift, detail collapse, and composition instability under complex workflows (multi-step denoising, ControlNet integration, or LoRA stacking).

### 2. Failure Under the Deterministic Latent-Trajectory Gate
Upstream HSWQ instituted a deterministic latent-trajectory gate (step-by-step per-tensor cosine similarity comparison and bifurcation detection across 25 random seeds):
- **Precision limit of 4-bit (NVFP4)**: Across SDXL UNet's extensive attention layers, truncation error and quantization noise compounded exponentially throughout the reverse diffusion trajectory.
- **Cosine mean threshold missed**: Across the 25 evaluation seeds, SDXL NVFP4 failed to reach the required ≥ 0.90 final trajectory cosine threshold (gating standard is ≥ 0.95), exhibiting multiple irreversible trajectory bifurcations.
- **Breakdown of deterministic reproducibility**: Under identical seeds and prompts, NVFP4 latent states deviated significantly from FP16 ground truth, failing high-fidelity production requirements.

### 3. SDXL Unification on ConvRot INT8 (Reverse Hybrid)
In contrast to NVFP4, SDXL **ConvRot INT8 (reverse hybrid)** demonstrated exceptional stability:
- **Trajectory-impact ranking**: Operating directly on the FP16 checkpoint as sole input, layers are ranked by trajectory impact; only the K lowest-impact layers are packed as FULL ConvRot INT8, while all sensitive layers remain native FP16.
- **Passes 25-seed gate**: Achieved **cosine mean ≥ 0.95 with 0/25 bifurcations** across the deterministic 25-seed gate.
- **Decision**: SDXL cannot reliably compress to 4-bit NVFP4 under existing diffusion formulations. Full focus on ConvRot INT8 is the optimal engineering path for visual quality and mathematical stability.

---

## What Was Removed and Changed

### 1. SDXL Checkpoint Loader Option (`__init__.py`)
- Removed `"ConvRot NVFP4"` from the `weight_dtype` dropdown widget of `HSWQ Checkpoint Loader (SDXL)` (`HSWQCheckpointLoaderSDXL`).
- Supported options are now:
  - `"default"` (stock / auto-detection)
  - `"int8_tensorwise"` (SDXL ConvRot INT8 reverse hybrid)

### 2. Benchmark Harness Deleted (`nodes/nvfp4bench_sdxl.py`)
- Completely deleted `nodes/nvfp4bench_sdxl.py` (594 lines).
- Following upstream cancellation, the harness is no longer needed.

### 3. Module Decoupling and Renaming
- Renamed `nodes/nvfp4/` to `nodes/sdxl_convrot_nvfp4/`, establishing a clean boundary for SDXL NVFP4 legacy runtime code.
- Consolidated Z Image NVFP4 logic into `nodes/zimage_convrot_nvfp4/` as a self-contained package.
- Fixed import paths across `__init__.py`, `nodes/hswq_sampler.py`, `nodes/nunchaku_usdu.py`, and `nodes/krea2_convrot_int8/int8_lora_bake.py`.

---

## Backward Compatibility for Existing Workflows

- **Legacy NVFP4 Checkpoint Loading**:
  - Previously published SDXL NVFP4 checkpoints remain loadable.
  - Setting `weight_dtype` to **`default`** triggers `checkpoint_looks_like_comfy_quant_nvfp4`, automatically routing execution to `nodes/sdxl_convrot_nvfp4/` and Tensor Core execution.
- **Saved Workflows**:
  - Existing workflows with serialized `ConvRot NVFP4` inputs execute without crash via parameter absorption.
- **Tensor Boost Toggle**:
  - The `tensor_boost` toggle on HSWQ Sampler and HSWQ Ultimate SD Upscale continues to provide CUDA Graph acceleration for legacy SDXL NVFP4 checkpoints on NVIDIA Blackwell (SM >= 100).

---

## Model Support Matrix

| Family | Precision / Architecture | Implementation | Status |
| :--- | :--- | :--- | :--- |
| **SDXL** | **ConvRot INT8 (reverse hybrid)** | `nodes/sdxl_convrot_int8` | **Supported** (cosine mean ≥ 0.95) |
| **SDXL** | ConvRot NVFP4 | `nodes/sdxl_convrot_nvfp4` | **Cancelled** (UI option removed, `default` auto-detect retained) |
| **Z Image / ZIT** | **ConvRot NVFP4 (Comfy parity)** | `nodes/zimage_convrot_nvfp4` | **Supported** (self-contained package) |
| **Krea2** | **ConvRot INT8** | `nodes/krea2_convrot_int8` | **Supported** (DisTorch2 verified) |

---

## File Changes Summary

| File | Change |
| :--- | :--- |
| `__init__.py` | Modified - removed `ConvRot NVFP4` from `weight_dtype`; updated imports; bumped `__version__` to `3.6.1` |
| `nodes/nvfp4bench_sdxl.py` | Deleted (594 lines) |
| `nodes/sdxl_convrot_nvfp4/` | Renamed from `nodes/nvfp4/`, updated internal references |
| `nodes/zimage_convrot_nvfp4/` | Consolidated Z Image NVFP4 into self-contained package |
| `nodes/hswq_sampler.py` | Modified - updated runtime import to `nodes.sdxl_convrot_nvfp4` |
| `nodes/nunchaku_usdu.py` | Modified - updated runtime import to `nodes.sdxl_convrot_nvfp4` |
| `nodes/krea2_convrot_int8/` | Modified - updated cross-module peel imports |
| `README.md` / `zhmd/README.md` | Modified - documented cancellation; removed Hugging Face NVFP4 links; updated Tensor Boost descriptions |
| `changelog.md` / `zhmd/CHANGELOG.md` | Modified - recorded v3.6.1 release entries |
| `pyproject.toml` | Modified - bumped version to `3.6.1` |
| `zhmd/v3.6.1.md` | Added - Chinese release specification document |
