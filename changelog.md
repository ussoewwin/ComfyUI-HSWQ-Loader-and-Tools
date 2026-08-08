# Changelog

<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="zhmd/CHANGELOG.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

## Version 3.4.0

- **Added**: **SDXL ConvRot NVFP4 Blackwell Tensor Boost** — Per-Weight CUDA Graph auto-dispatch on SM >= 100 (B200 / GB200, RTX 5090 / SM120) inside `nodes/nvfp4/` only (Z Image / INT8 / FP8 / stock paths untouched). Eliminates shape-shared weight `.copy_()` on replay; adaptive `M` cap 16384; capture / hit console logs and `nvfp4_forward_stats()` (`blackwell_graph_hits`, `blackwell_tensor_boost_active`). Independent **`tensor_boost` BOOLEAN** on **HSWQ Sampler** and **HSWQ Ultimate SD Upscale** (default OFF; Loader has no toggle) via `HSWQ_NVFP4_TENSORBOOST` / `HSWQ_NVFP4_CUDAGRAPH`, with `clear_nvfp4_cudagraphs()` on OFF so USDU tiles avoid VRAM blow-up. **ON raises VRAM by several GB** (CUDA Graph arenas) — **RTX 5090 32 GB+** recommended for upscale / Tensor Boost headroom; sampler path **16 GB+**. Docs: `md/HSWQ_SDXL_NVFP4_BLACKWELL_ACCELERATION_GUIDE.md`.
- See [Release Notes v3.4.0](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.4.0) for details.

## Version 3.3.9

- **ComfyUI 0.30.2 compatibility & Krea2 parity contamination** (commits `21792a8`..`ecd6bc0`):
  - **Perf**: Krea2 ConvRot INT8 GPU-cached Hadamard matrices (`native_convert_int8.get_hadamard_on_device`), 200-module early exit for full-model INT8/SVDQ scans, `mixed_precision_ops` re-entry guards, and `disabled` set normalization.
  - **Perf / VRAM**: ZI NVFP4 `load_models_gpu` bake hook fast-skip (no patches + no baked keys -> skip; non-dynamic model -> skip), reducing VRAM pressure from redundant full-diagnostics on every GPU load.
  - **Perf**: Krea2 ConvRot INT8 progressive slowdown across runs (1st ~4s/step, 2nd 4s->16s->22s->26s/step) **fixed**. Root cause: Z Image `comfy_parity` wrappers left on `mixed_precision_ops` / `_load_quantized_module` armed `_hswq_int8_convrot` on Krea2 INT8 ConvRot layers and installed `forward_parity` (online Hadamard act rotate) on every Linear -> unnecessary rotation every step -> CUDA fragmentation worsening each run. Fix: `_clear_zimage_parity_contamination_for_sdxl()` called before Krea2 stock load (same as SDXL path).
  - **Compat**: `Parameter.data` unwrap for ComfyUI 0.30.2 lazy weight repr, `comfy.weight_adapter.lora` import fallback, `calculate_weight` `intermediate_dtype` default = `torch.float32`, `LowVramPatch.__call__` `original_weights` kwarg, `state_dict` `extra_quant_params`.
  - **Docs**: Technical guide `md/HSWQ_COMFYUI_0_30_2_COMPATIBILITY_FIX_GUIDE.md` covering all root causes, fixes, and verification.
- See [Release Notes v3.3.9](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.3.9) for details.

## Version 3.3.8

- **Added**: **HSWQ Sampler** `clip_perfect_offload (Krea2 only)` toggle — frees the Krea2 text encoder before sampling (drops its patcher from `current_loaded_models`) to reach bench-parity VRAM on tight cards. Krea2-scoped both ways via the loader tag `_hswq_is_krea2` and exact `comfy.text_encoders.krea2` module identity (no class-name guessing); off by default, strict boolean read, no global allocator ops, and any failure is caught so a run never breaks. UI widget now shows the `(Krea2 only)` scope tag. Docs: EN/ZH README node sections and new `md/HSWQ_KREA2_TE_OFFLOAD_GUIDE.md`.
- See [Release Notes v3.3.8](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.3.8) for details.

## Version 3.3.7

- **Fixed / Changed (license & provenance)**: Dropped leftover Apache-2.0 wording so this loader repo is consistently **GPL-3.0**; clarified that upstream **HSWQ** ([Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)) remains **AGPL-3.0** and is separate from this package’s license. Reworked README / zhmd provenance for **USDU**, **Torch Compile (KJNodes)**, and **Batched Detailer (Impact Pack)** without “copy” phrasing. Batched Detailer now ships helpers in `nodes/batched_detailer_lib/` so Impact Pack is **not** required at runtime while GPL attribution remains.
- See [Release Notes v3.3.7](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.3.7) for details.

## Version 3.3.6

- **Added / Fixed**: **HSWQ Torch Compile** node (`HSWQTorchCompileModel`) — ComfyUI `set_torch_compile_wrapper` path without KJNodes; forces `compile_threads=1` and `worker_start_method=subprocess` so SeedVR2 / `utils.install_util` spawn crashes stay off; defaults inductor + `max-autotune-no-cudagraphs`. **ZI INT8 peel**: `peel_non_product_nvfp4_ops` dives under PRODUCT NVFP4 load when the under layer is foreign INT8 / ZI protect, so SDXL INT8 survives after Z Image. Docs: EN/ZH README node sections, technical guide, BETA badge removed.
- See [Release Notes v3.3.6](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.3.6) for details.

## Version 3.3.5

- **Fixed / Changed**: Large Z Image ConvRot NVFP4 hardening after v3.3.4 — peel Z Image into dedicated `nodes/zimage_nvfp4` (no shared ownership with SDXL `nodes/nvfp4` Tensor Core product); separate dropdown **`Z Image ConvRot NVFP4`** vs SDXL **`ConvRot NVFP4`** and branch Dynamic VRAM LoRA bake accordingly; clear Z Image **comfy_parity** load overlay + in-place Linear bake (**VER=8**) + INT8-protect arm residue when returning to SDXL INT8 / SDXL ConvRot NVFP4 so salt-pepper, LoRA fall-off, and full noise after SDXL → Z Image → SDXL no longer stick.
- See [Release Notes v3.3.5](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.3.5) for details.

## Version 3.3.4

- **Fixed**: Z Image / ZIT **ConvRot NVFP4** / INT8 protect — after **Distorch** purge, module-local `_hswq_nvfp4_parity_H` reused under a weaker gate than global `_tensor_storage_ok` → **2nd+ gen** quality decay. Parity now shares `_tensor_storage_ok`.
- See [Release Notes v3.3.4](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.3.4) for details.

## Version 3.3.3

- **Fixed**: Z Image hybrid packs (**ConvRot NVFP4** + **ConvRot INT8 protect**) — Dynamic VRAM LoRA bake now covers **both** Linear families. INT8 protect is armed like Conv2d (clear kitchen `Params.convrot`, keep False after requant); dual bake + pass-delta EVIDENCE (`NVFP4_LORA_BAKE_*` / `INT8_PROTECT_LORA_BAKE_*`) so leftover LowVramPatch on protect layers no longer leaves dead LoRA or noise.
- See [Release Notes v3.3.3](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.3.3) for details.

## Version 3.3.2

- **Fixed**: Z Image / ZIT **ConvRot NVFP4** salt-and-pepper noise on the **2nd generation after a DistOrch VRAM purge**. INT8 decode wrap was dropping NVFP4 stack markers, so a later “upgrade” re-wrapped the Tensor Core product path over the Comfy parity stack; DistOrch refresh then peeled only the TC layer and left **double online act rotate** on reload. Markers are preserved through the INT8 wrap so parity refresh no longer re-arms a second rotate.
- See [Release Notes v3.3.2](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.3.2) for details.

## Version 3.3.1

- **Added**: Z Image / ZIT **ConvRot NVFP4** support on **HSWQ ConvRot INT8/ConvRot NVFP4 UNet Loader** (`weight_dtype`: `ConvRot NVFP4`, or `default` with NVFP4 auto-detect). Uses the bench-matched Comfy parity path (stock MixedPrecision GEMM + online act rotate) under `nodes/zimage_nvfp4`, including mixed NVFP4 + INT8 protect packs and Dynamic VRAM LoRA bake — not the SDXL Checkpoint Loader Tensor Core product path. **Supported only for models quantized with [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization).**
- See [Release Notes v3.3.1](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.3.1) for details.

## Version 3.3.0

- **Changed**: Remaining ComfyUI node class IDs renamed from Nunchaku-prefixed names to HSWQ-prefixed IDs (`HSWQSaveImage`, `HSWQCheckpointLoaderSDXL`, `HSWQSDXLLoraStackV3`, `HSWQZImageDiTLoader`, and related JS hooks).
- See [Release Notes v3.3.0](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.3.0) for details.

## Version 3.2.9

- **Changed**: `pyproject.toml` `[project].name` updated to match the new repository identity so the ComfyUI registry category is **comfyui-hswq-loader-and-tools**.
- **Changed**: Re-registered the node pack with ComfyUI under the corrected project name.
- See [Release Notes v3.2.9](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.2.9) for details.

## Version 3.2.8

- **Changed**: Repository renamed to **ComfyUI-HSWQ-Loader-and-Tools**.
- **Changed**: Node renamed from **HSWQ&Nunchaku Ultimate SD Upscale** to **HSWQ Ultimate SD Upscale** (class, ID, and title).
- See [Release Notes v3.2.8](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.2.8) for details.

## Version 3.2.7

- **Removed**: In-node INT8 W8A8 Triton Linear acceleration (Plan B) — fused kernels, `install.py` Triton stage, and the **Triton accelerate** UI toggle. INT8 Linear speed now relies on ComfyUI + `comfy_kitchen` (`int8_linear` cuda → triton → eager). This extension keeps INT8 load compatibility patches only (Conv2d / LoRA / ControlLora / handoff).
- See [Release Notes v3.2.7](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.2.7) for details.

## Version 3.2.6

- **Added**: Public INT8 W8A8 Triton Linear acceleration (Plan B) for HSWQ INT8 loaders — fused row-wise activation quant → INT8 GEMM → dequant without relying on Comfy `--enable-triton-backend`; Windows/Linux Triton install in `install.py`; UI toggle **Triton accelerate**; tiled rowwise quant so wide layers (e.g. K=10240) stay on the fused path.
- See [Release Notes v3.2.6](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.2.6) for details.

## Version 3.2.5

- **Fixed**: `requirements.txt` install failure on outdated portable/embedded Python environments — a transitive, wheel-less source dependency (`filterpy`, pulled by `facexlib`) forced a source build that crashed with `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'` on Python 3.12+ because the environment shipped an old `setuptools`. An `install.py` now upgrades `pip` / `setuptools` / `wheel` before installing `requirements.txt`, so ComfyUI-Manager install/update repairs the build tooling first and the legacy source build succeeds.
- See [Release Notes v3.2.5](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.2.5) for details.

## Version 3.2.4

- **Fixed**: SDXL LoRA-type ControlNet (e.g. `anytest`) producing all-black output under INT8 quantization — `ControlLora.pre_run` borrowed the INT8 base UNet weights via `diffusion_model.state_dict()`, which returned flattened raw `int8`/`uint8` tensors instead of `QuantizedTensor`, so the borrowed weights were never dequantized. The patch intercepts that `state_dict()` and dequantizes the INT8 base weights on the fly (full-weight ControlNets such as `canny` were unaffected, and the issue did not occur in FP8).
- See [Release Notes v3.2.4](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.2.4) for details.

## Version 3.2.3

- **Added**: **HSWQ Sampler** — a KSampler-equivalent node that behaves exactly like the standard ComfyUI KSampler, but automatically adds all of RES4LYF's samplers and schedulers when [RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF) is installed. It reproduces Forge's dynamic sampler generation so the full Runge-Kutta (`rk_beta`) sampler family stays selectable and runnable in vanilla ComfyUI.
- See [Release Notes v3.2.3](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.2.3) for details.

## Version 3.2.2

- **Fixed**: INT8→Nunchaku VRAM handoff false-positive on non-SVDQ loads (including SDXL INT8 normal generation) — SVDQ detection no longer uses bare `"nunchaku" in __module__` (this extension’s INT8 Conv2d path contains that substring); handoff `_VER = 10` arms only for real Nunchaku SVDQ on the BaseModel, and native comfy_quant INT8 (any architecture) never arms handoff.
- See [Release Notes v3.2.2](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.2.2) for details.

## Version 3.2.1

- **Fixed**: INT8 HSWQ (Dynamic VRAM) → Nunchaku SVDQ coexistence Abort — LowVramPatch and Dynamic LoRA bake restricted to `comfy.quant_ops.QuantizedTensor` only (never bare `torch.int8`); unidirectional VRAM handoff uses `detach(unpatch_all=True)` before SVDQ load.
- **Removed**: Reintroduced **HSWQ Pin Buffer Cache** again (not required for the Abort fix; Detailer-scoped pin pooling remains obsolete after AIMDO HostBuffer).
- **Docs**: Rewrote `md/HSWQ_INT8_NUNCHAKU_COEXISTENCE_GUIDE.md` for verified Abort causes vs PinCache correlation.
- See [Release Notes v3.2.1](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.2.1) for details.

## Version 3.2.0

- **Removed**: **HSWQ Pin Buffer Cache** (`nodes/hswq_pin_cache.py` and Detailer `hswq_pin_cache_scope`) — redundant after ComfyUI Dynamic VRAM / AIMDO `HostBuffer` updates (no thrashing `unpin` path). Batched Detailer three-phase flow kept; use native ComfyUI pin behavior.
- **Changed**: Display title forced to **HSWQ Checkpoint Loader (SDXL)** for the SDXL checkpoint loader node.
- See [Release Notes v3.2.0](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.2.0) for details.

## Version 3.1.9

- **Added**: Native **comfy_quant INT8** (`int8_tensorwise`) load path for SDXL checkpoints — **HSWQ FP8/INT8 Loader (VRAM Opt)** auto-detects INT8 vs Scaled FP8; **HSWQ FP8 E4M3 UNet Loader** gains `int8_tensorwise` / auto-detect. Extension-side Conv2d quant support and INT8-safe LoRA bake under Dynamic VRAM.
- See [Release Notes v3.1.9](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.1.9) for details.

## Version 3.1.8

- **Added**: **HSWQ Save Image** (`NunchakuSaveImage`) — save `IMAGE` output as PNG or JPG (JPEG quality when JPG is selected).
- **Added**: **Nunchaku Ultimate SD Upscale** — `upscale_by` dropdown with **Auto** mode and `target_height` (default 4320) to derive scale from input height; fixed magnifications 0.05–4.00 remain available.
- See [Release Notes v3.1.8](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.1.8) for details.

## Version 3.1.7

- **Fixed**: Critical fix for severe output noise and `RuntimeError` in `NunchakuUltimateSDUpscale` when used with Lumina/HunYuan-DiT architectures. Corrected the conditioning tensor slicing logic to accurately extract T5/LLM features from concatenated tensors.
- See [Release Notes v3.1.7](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.1.7) for details.

## Version 3.1.3

- **Fixed**: Workaround for `RuntimeError` in `NunchakuUltimateSDUpscale` caused by a recent ComfyUI core change that concatenates multi-encoder conditioning along the feature dimension (e.g., 7680 instead of 2560) for Lumina/HunYuan-based models. Added automatic detection and truncation of these embeddings before sampling.
- See [Release Notes v3.1.3](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.1.3) for details.

## Version 3.1.2

- **Fixed**: Pin Buffer Cache (monkey-patch for `comfy.pinned_memory.pin_memory` / `unpin_memory`) is now enabled only while running `HSWQ Batched Detailer (SEGS)`. Outside of Detailer SEGS, the extension delegates back to ComfyUI's original pin/unpin behavior to avoid side effects in other nodes/workflows.

## Version 3.1.1

- **Fixed**: Bug fixes and corrections (loader registration, zimage model handling, USDU crop model patch).
- See [Release Notes v3.1.1](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.1.1) for details.

## Version 3.1.0

- **Added** two new nodes:
  - **HSWQ FP8 E4M3 UNet Loader** (`HSWQFP8E4M3UNetLoader`) — Standard UNet loader for HSWQ FP8 E4M3 models; extension also installs a Pin Buffer Cache to reduce `cudaHostRegister`/`cudaHostUnregister` overhead under Dynamic VRAM Loading.
  - **HSWQ Batched Detailer (SEGS)** — Detailer (SEGS)–style node that runs VAE encode → UNet sample → VAE decode in three phases (all encodes, then all samples, then all decodes) to minimize model switching and improve performance with Dynamic VRAM Loading.
- See [Release Notes v3.1.0](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.1.0) for details.

## Version 3.0.2

- **README**: FP8 (fp8e4m3) and torch.compile subsection updated — purpose (use this node with FP8 and torch.compile together) and patches description.
- See [Release Notes v3.0.2](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/3.0.2) for details.

## Version 3.0.0

- **Breaking**: Aligned with SDXL SVDQ deprecation (see IMPORTANT NOTICE at top). Node registration reduced to the following three only:
  - **Nunchaku-ussoewwin SDXL Integrated Loader** (Checkpoint Loader style: single checkpoint)
  - **Nunchaku-ussoewwin SDXL DiT Loader (DualCLIP)** (UNet + CLIP from separate files)
  - **Nunchaku Ultimate SD Upscale**
- **Removed** from registration (no longer appear in ComfyUI):
  - Nunchaku-ussoewwin Z-Image-Turbo DiT Loader
  - Nunchaku-ussoewwin SDXL LoRA Stack V3
  - Nunchaku Apply First Block Cache Patch Advanced
- Future SDXL workflows are intended to use fp8e4m3 with standard ComfyUI loaders where applicable.

## Version 2.6.6

- **Fixed**: Fixed `AttributeError: 'Logger' object has no attribute 'mgpu_mm_log'` error that was causing prompt execution to crash. Replaced all instances of `logger.mgpu_mm_log()` with `logger.info()` in `model_management_mgpu.py`, `device_utils.py`, and `wrappers.py`.

## Version 2.6.3

- Added **Checkpoint Loader (SDXL)** node
  - Loads MODEL and CLIP from standard SDXL checkpoints with optional device selection and FP8 precision support
- Nunchaku SDXL SVDQ (4-bit) development discontinued; repository status updated (see IMPORTANT NOTICE at top)
- See [Release Notes v2.6.3](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/2.6.3) for details

## Version 2.6.2

- Fixed NunchakuUltimateSDUpscale node registration issue with Nunchaku 1.2.0
  - Improved error handling in INPUT_TYPES to prevent node registration failures
  - Node is standalone: uses bundled `usdu_bundle` and does not require ComfyUI_UltimateSDUpscale to be installed
  - See [Issue #2](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/issues/2) for details
- See [Release Notes v2.6.2](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/2.6.2) for details

## Version 2.6.1

- Optimized LoRA processing performance for SDXL models
- See [Release Notes v2.6.1](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/2.6.1) for details

## Version 2.6

- Fixed ControlNet support for SDXL models (OpenPose, Depth, Canny, etc.)
- See [Release Notes v2.6](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/2.6) for details

## Version 2.5

- Added SDXL Integrated Loader node for unified checkpoint loading
  - Supports loading both UNet and CLIP from a single checkpoint file
  - Includes Flash Attention 2 support (enabled by default)
  - Automatically detects model configuration from checkpoint keys
- Reorganized node documentation order
- Updated SDXL DiT Loader with advanced user warning
- See [Release Notes v2.5](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/2.5) for details

## Version 2.4

- Added Flash Attention 2 support for SDXL DiT Loader
  - Optional acceleration feature enabled by default
  - Automatically applies FA2 to all attention layers (typically 140 layers in SDXL models)
  - Requires Flash Attention 2 to be installed in your environment
  - Can be disabled via the `enable_fa2` parameter if needed
- Updated SDXL DiT Loader node image
- See [Release Notes v2.4](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/2.4) for details

## Version 2.3

- Added Nunchaku Ultimate SD Upscale nodes with improved color normalization
- Improved First Block Cache with residual injection for better quality
- Fixed USDU color normalization for Nunchaku SDXL VAE output
- Fixed module reference separation to prevent data loss
- Optimized cache similarity calculation using fused kernels
- Added Flash Attention 2 support for SDXL DiT Loader (optional, enabled by default)
- See [Release Notes v2.3](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/2.3) for details

## Version 2.2

- Added First Block Cache feature for Nunchaku SDXL models
- See [Release Notes v2.2](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/2.2) for details

## Version 2.1

- Published LoRA Loader technical documentation
- See [Release Notes v2.1](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-z-image-turbo-loader/releases/tag/2.1) for details

## Version 2.0

- Added SDXL DIT Loader support
- Added SDXL LoRA support
- Added ControlNet support for SDXL models
- See [Release Notes v2.0](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-z-image-turbo-loader/releases/tag/2.0) for details

## Version 1.1

- Added Diffsynth ControlNet support for Z-Image-Turbo models
  - Note: Does not work with standard model patch loader. Requires a custom node developed by the author.
- See [Release Notes v1.1](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-z-image-turbo-loader/releases/tag/1.1) for details

## 2025-12-25

- Fixed import error for `NunchakuZImageDiTLoader` node by improving alternative import method with better path resolution (see [Issue #1](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/issues/1))
