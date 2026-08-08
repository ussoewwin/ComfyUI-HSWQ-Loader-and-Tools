# ComfyUI-HSWQ-ConvRot-INT8/ConvRot-NVFP4-Loader-and-Tools

<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="zhmd/README.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

<p align="center">
<img src="https://raw.githubusercontent.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/main/icon.png?v=2" width="128">
</p>

## Overview

This custom node pack loads and runs **[Hybrid-Sensitivity-Weighted-Quantization (HSWQ)](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)** packs and related ComfyUI-compatible quantized SDXL / Z Image weights.

**HSWQ** (the quantization method, scripts, and upstream docs) is an **original work by ussoewwin**, published separately under the **GNU Affero General Public License v3 (AGPL-3.0)** at [ussoewwin/Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization). This ComfyUI loader pack is a related but distinct repository (see License below).

HSWQ is a high-fidelity quantization line for diffusion UNets. Current public HSWQ work focuses on **ConvRot INT8** and **ConvRot NVFP4** for **SDXL**, plus **ConvRot NVFP4** for **Z Image / ZIT** UNets (sensitivity / importance analysis, DualMonitor + weighted-histogram FP16 protection, then FULL ConvRot on the remainder). It is **not** a keep-ratio percentage scheme: keep ratio is fixed at **0 (r0)**; FP16 layers are chosen by automatic analysis under a fixed MiB budget.

| Path | Role in this repo |
| :--- | :--- |
| **HSWQ ConvRot INT8 (SDXL V3.1)** | ComfyUI `int8_tensorwise` packs; load via **HSWQ Checkpoint Loader (SDXL)** (`weight_dtype`: `int8_tensorwise` / INT8 auto-detect). **Supported only for models quantized with [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization).** |
| **HSWQ ConvRot NVFP4 (SDXL)** | ComfyUI `nvfp4` packs (Linear→NVFP4, Conv2d→INT8 + ConvRot); load via the **same** **HSWQ Checkpoint Loader (SDXL)** (`weight_dtype`: `ConvRot NVFP4`, or `default` with NVFP4 auto-detect). **Supported only for models quantized with [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization).** |
| **HSWQ ConvRot NVFP4 (Z Image / ZIT)** | ComfyUI `nvfp4` UNet packs (often Linear NVFP4 + INT8 protect); load via **HSWQ ConvRot INT8/ConvRot NVFP4 UNet Loader** (`weight_dtype`: `ConvRot NVFP4`, or `default` with NVFP4 auto-detect). Uses the bench-matched **Comfy parity** path (stock GEMM + online act rotate), not the SDXL Tensor Core product path. **Supported only for models quantized with [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization).** |
| **FP8 (E4M3)** | HSWQ **FP8 development has ended** (technical docs remain upstream). Loaders here may still accept existing FP8 weights where ComfyUI supports them |
| **Z Image 8-bit** | HSWQ-specific Z Image INT8 development / publication **ended**. Prefer **native ConvRot INT8** for Z Image (typically SSIM > 0.99). HSWQ INT8 continues for **SDXL**. **Z Image ConvRot NVFP4** is supported via the UNet loader above |

Upstream HSWQ targets (reference): ConvRot INT8 SSIM about **0.94–0.98**, ConvRot NVFP4 about **0.95**, with roughly **30–40%** smaller files than FP16 while keeping standard ComfyUI loader compatibility.

**Quantization scripts, How-to docs, and benchmarks:** [ussoewwin/Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)

**Published HSWQ SDXL models (ConvRot INT8):** [Hugging Face — Hybrid-Sensitivity-Weighted-Quantization-SDXL-ConvRot-INT8](https://huggingface.co/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization-SDXL-ConvRot-INT8)

**Published HSWQ SDXL models (ConvRot NVFP4):** [Hugging Face — Hybrid-Sensitivity-Weighted-Quantization-SDXL-ConvRot-NVFP4](https://huggingface.co/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization-SDXL-ConvRot-NVFP4)

<p align="center">
<img src="https://raw.githubusercontent.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/main/logo.png" width="400">
</p>

## Installation

### Quick Install

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
# Windows
git clone https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools "%USERPROFILE%\ComfyUI\custom_nodes\ComfyUI-HSWQ-Loader-and-Tools"

# Linux/Mac
git clone https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools ~/ComfyUI/custom_nodes/ComfyUI-HSWQ-Loader-and-Tools
```

Restart ComfyUI to load the nodes.

## Nodes

### HSWQ Checkpoint Loader (SDXL)

<img src="png/fp8e4m3.png?v=3" alt="HSWQ Checkpoint Loader (SDXL) Node" width="400">

ComfyUI node that loads **MODEL** and **CLIP** from standard SDXL checkpoints, with optional device selection and **FP8 / INT8 / ConvRot NVFP4** precision support. Use it like the standard Load Checkpoint node; it outputs MODEL and CLIP only (no VAE).

**SDXL ConvRot INT8** and **SDXL ConvRot NVFP4** are **supported only for models quantized with [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)**. Other third-party ConvRot INT8 / ConvRot NVFP4 packs are out of scope.

**ConvRot NVFP4:** Select `weight_dtype` = **`ConvRot NVFP4`**, or leave **`default`** when the checkpoint has comfy_quant `nvfp4` markers — the loader routes to this extension’s NVFP4 stack (Linear → NVFP4 Tensor Core / `scaled_mm_nvfp4` + optional act ConvRot; Conv2d → INT8 + ConvRot via the INT8 patches). NVFP4 dispatch is installed **after** INT8 dispatch so mixed packs (NVFP4 Linear + INT8 Conv) are not stolen by INT8-only auto-detect.

This loader does **not** ship an in-node Triton accelerate toggle. INT8 Linear speed is left to **ComfyUI + `comfy_kitchen`** (`int8_linear`: cuda → triton → eager). This extension keeps INT8 **load compatibility** patches (Conv2d / LoRA / ControlLora / handoff) and the **NVFP4** load + forward patches under `nodes/nvfp4/`.

#### Features

- **Checkpoint Loading**: Loads both UNet (MODEL) and CLIP from a single SDXL checkpoint file (same as standard Load Checkpoint)
- **Device Selection**: Optional device parameter to choose GPU (e.g. `cuda:0`, `cuda:1`) or CPU for model loading
- **FP8 weight dtype**: `fp8_e4m3fn`, `fp8_e4m3fn_fast`, `fp8_e5m2` (plus `default` for non-forced dtype)
- **INT8 weight dtype**: `int8_tensorwise` — **HSWQ SDXL ConvRot INT8** via ComfyUI `MixedPrecisionOps` (this extension also patches **Conv2d** quant load so SD UNet INT8 works, not Linear-only). **Supported only for models quantized with [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization).**
- **ConvRot NVFP4 weight dtype**: `ConvRot NVFP4` — **HSWQ SDXL ConvRot NVFP4** (`comfy_quant` `nvfp4`: Linear NVFP4 + Conv2d INT8 / ConvRot); applies `nodes/nvfp4` patches (packed-K detect, full NVFP4 Linear load, Tensor Core forward). **Supported only for models quantized with [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization).**
- **INT8 auto-detect**: If the safetensors looks like comfy_quant INT8 (and is not NVFP4), the loader uses the MixedPrecisionOps / INT8 path even when `weight_dtype` is not set to `int8_tensorwise`
- **NVFP4 auto-detect**: If `weight_dtype` is `default` and the checkpoint looks like comfy_quant NVFP4, the loader uses the ConvRot NVFP4 path automatically
- **Standard ComfyUI Integration**: Uses `load_checkpoint_guess_config`; compatible with standard ComfyUI workflows
- **No Triton accelerate widget**: UI is checkpoint / weight dtype / device only; fused INT8 Linear acceleration is not controlled from this node

#### Usage Notes

- **Inputs**: `ckpt_name` (checkpoint file), `weight_dtype` (`default` / FP8 options / `int8_tensorwise` / `ConvRot NVFP4`), and optionally `device`
- **Outputs**: MODEL and CLIP only; use a separate VAE loader if needed
- **Category**: Loaders (`loaders`)
- **SDXL ConvRot INT8 / ConvRot NVFP4 compatibility**: **Only** checkpoints quantized with [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)
- **ConvRot NVFP4 models**: Published packs — [Hybrid-Sensitivity-Weighted-Quantization-SDXL-ConvRot-NVFP4](https://huggingface.co/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization-SDXL-ConvRot-NVFP4)
- **INT8 speed**: Rely on ComfyUI / `comfy_kitchen` for Linear acceleration; this node does not install or toggle Triton
- **INT8 + LoRA**: For INT8 LoRA bake / Status logging details, see `md/HSWQ_INT8_AND_LORA_TECHNICAL_GUIDE.md`
- **VRAM purge (required for HSWQ ConvRot INT8 / ConvRot NVFP4)**: When you load with **HSWQ ConvRot INT8** or **HSWQ ConvRot NVFP4**, always place **General Purge VRAM V2** from [ComfyUI-DistorchMemoryManager](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager) at the **end** of the workflow and turn its **`HSWQ`** toggle **on**. HSWQ residual GPU/host memory (and NVFP4 runtime pools / CUDA graphs) is not fully released by ComfyUI's generic unload, so a second generation after the first can fail (e.g. `quantize_nvfp4` / `PyCapsule` / `pooled TC path failed`) without this purge.

### HSWQ ConvRot INT8/ConvRot NVFP4 UNet Loader

<img src="png/hswqunet.png?v=3" alt="HSWQ ConvRot INT8/ConvRot NVFP4 UNet Loader" width="400">

Standard ComfyUI UNet loader wrapper for diffusion models under `diffusion_models` (**Z Image / ZIT** and other UNet packs). Loads **MODEL** with FP8, INT8, and **ConvRot NVFP4** weight dtypes.

**Z Image / ZIT ConvRot NVFP4** is **supported only for models quantized with [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)**. Other third-party ConvRot NVFP4 UNet packs are out of scope.

- **General FP8 / INT8**: Same idea as the stock UNet loader (HSWQ FP8 E4M3, Scaled FP8, and native comfy_quant / `int8_tensorwise` when selected or auto-detected). Not limited to HSWQ-only weights for those modes.
- **ConvRot NVFP4 (Z Image / ZIT)**: Select `weight_dtype` = **`ConvRot NVFP4`**, or leave **`default`** when the UNet safetensors has comfy_quant / HSWQ `nvfp4` markers. Routes to this extension’s UNet NVFP4 stack under `nodes/nvfp4/` with the **Comfy parity** path used by `hswq/benchmark` (stock MixedPrecision GEMM + online act rotate; ConvRot Linear LoRA bake kept). **Do not** expect the SDXL Checkpoint Loader’s Tensor Core product path here — SDXL NVFP4 stays on the Checkpoint Loader; Z Image NVFP4 stays on this UNet loader. **Supported only for models quantized with [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization).**
- **INT8 / NVFP4 auto-detect**: INT8-looking packs use the INT8 path; NVFP4-looking packs use the ConvRot NVFP4 path when `weight_dtype` is `default` (NVFP4 dispatch is installed after INT8 so mixed packs are not stolen by INT8-only detect).

**Inputs**: `unet_name`, `weight_dtype` (`default` / FP8 options / `int8_tensorwise` / `ConvRot NVFP4`).

This loader does **not** ship an in-node Triton accelerate toggle. INT8 Linear speed is left to **ComfyUI + `comfy_kitchen`** (`int8_linear`: cuda → triton → eager). This extension keeps INT8 **load compatibility** patches (Conv2d / LoRA / ControlLora / handoff) and the **NVFP4** UNet patches under `nodes/nvfp4/`.

- **Z Image / ZIT ConvRot NVFP4 compatibility**: **Only** UNet packs quantized with [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)

**VRAM purge**: For **ConvRot NVFP4** (and HSWQ INT8) UNet loads, place **General Purge VRAM V2** from [ComfyUI-DistorchMemoryManager](https://github.com/ussoewwin/ComfyUI-DistorchMemoryManager) at the end of the workflow with **`HSWQ`** on — same reason as the SDXL Checkpoint Loader section.

### HSWQ Ultimate SD Upscale

<img src="png/usdu_auto_workflow.png" alt="HSWQ Ultimate SD Upscale" width="400">

**Provenance:** This node was developed based on **[ComfyUI_UltimateSDUpscale](https://github.com/ssitu/ComfyUI_UltimateSDUpscale)** (ssitu, **GPL-3.0**), itself derived from Coyote-A’s Ultimate SD Upscale for Automatic1111, with original improvements and features added for HSWQ / FP8 / torch.compile / Auto scale (`usdu_bundle`, `nodes/nunchaku_usdu.py`, `usdu_compat_patches.py`). Shipping a standalone `usdu_bundle` does **not** remove the copyright / GPL obligation to the ssitu original.

#### Features

- **Tile-based Upscaling**: Processes images in tiles to handle high-resolution upscaling efficiently
- **Color Normalization**: Always normalizes Nunchaku SDXL VAE output to full dynamic range (0.0-1.0) before upscaling, fixing pale/washed-out colors
- **Multiple Modes**: Supports Linear, Chess, and None tile modes
- **Seam Fixing**: Includes multiple seam fixing modes (None, Band Pass, Half Tile, Half Tile + Intersections)
- **Module Isolation**: Prevents module reference conflicts with other custom nodes

#### Upscale magnification (`upscale_by` / `target_height`)

- **`upscale_by`**: Dropdown with **Auto** or fixed magnification values from **0.05** to **4.00** (step 0.05).
- **`target_height`**: Target output height in pixels (default **4320**). Used **only when `upscale_by` is Auto**.
- **Auto mode**: Reads the input image height from the connected `image`, then sets  
  `scale = target_height / input_height` (clamped to 0.05–4.0).
- **Fixed magnification**: When you pick a numeric value (e.g. **2.00**), that scale is used directly and **`target_height` is ignored**.

Example: input height 1080, `upscale_by = Auto`, `target_height = 4320` → scale 4.0 → output height 4320.

#### Usage Notes

- **Standalone**: Installing the separate `ComfyUI_UltimateSDUpscale` package is **not** required at runtime. This node ships `usdu_bundle` in-tree. That is a packaging choice; provenance and GPL-3.0 obligations still apply (see Provenance above).
- **Color Range**: Automatically normalizes Nunchaku SDXL VAE's compressed color range (e.g., 0.15-0.85) to full range (0.0-1.0) to restore proper contrast and color saturation
- **Module Safety**: Uses isolated module loading to prevent conflicts with other custom nodes
- **ssitu / UltimateSDUpscale**: Base of the tiled upscale design ([ComfyUI_UltimateSDUpscale](https://github.com/ssitu/ComfyUI_UltimateSDUpscale), GPL-3.0)

#### Tensor Boost (`tensor_boost`)

Optional **`tensor_boost`** BOOLEAN (default **OFF**) for **SDXL ConvRot NVFP4** on NVIDIA Blackwell (SM >= 100: B200 / GB200, RTX 5090 / SM120). When **ON**, enables Per-Weight CUDA Graph acceleration inside `nodes/nvfp4/` and **VRAM use rises by several GB** (CUDA Graph arenas). That headroom is why **RTX 5090 with 32 GB+** is recommended on this upscale path. When **OFF** (recommended for tiled upscale), clears the CUDA Graph cache via `clear_nvfp4_cudagraphs()` and runs Eager Pooled so per-tile shape changes do not stack Graph arenas / blow VRAM / spill to system RAM.

Intended pairing with **HSWQ Sampler**: turn Tensor Boost **ON** for the fixed-resolution base pass, then keep it **OFF** on this upscaler. Loader has no toggle. Details: `md/HSWQ_SDXL_NVFP4_BLACKWELL_ACCELERATION_GUIDE.md`.

**Recommended:** **RTX 5090 with 32 GB VRAM or more** (Tensor Boost ON adds several GB of VRAM; tiled high-res upscale needs that headroom).

#### FP8 (fp8e4m3) and torch.compile
- **Purpose:** Use this node with FP8 quantized models (e.g. HSWQ SDXL) and torch.compile together.
- **Patches:** On load, this extension applies compatibility patches (`usdu_compat_patches.py`) that fix copy_ shape mismatch, FP8 linear/addmm bias–out_features mismatch, control embedder weight layout, and Lumina modulate/apply_gate dimension issues so the node works with FP8 and torch.compile.

### HSWQ Save Image

<img src="png/saveimage.png" alt="HSWQ Save Image" width="400">

ComfyUI output node that saves images to your ComfyUI **output** folder as **PNG** or **JPG**.

#### Features

- **Format selection**: **PNG** (default) or **JPG**
- **Filename prefix**: Same behavior as the built-in Save Image node (default `ComfyUI`)
- **JPEG quality**: **quality (JPG only)** (1–100, default 95); ignored when format is PNG
- **PNG metadata**: Embeds workflow `prompt` and `extra_pnginfo` in PNG text chunks when available

#### Usage Notes

- **Inputs**: `images` (IMAGE), `format`, `filename_prefix`, `quality (JPG only)`
- **Category**: `image` (output node; no return socket)
- **Output path**: Uses ComfyUI's standard output directory via `folder_paths.get_output_directory()`

### HSWQ Batched Detailer (SEGS)

<img src="png/detailersegs.png" alt="HSWQ Batched Detailer (SEGS)" width="400">

**Provenance:** This node was developed based on **[ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)** Detailer (SEGS) / DetailerForEach (ltdrdata, **GPL-3.0**), with original improvements and features added **especially to maintain compatibility with HSWQ** quantized UNets (INT8 / NVFP4 ConvRot, Dynamic VRAM, QuantizedTensor paths) while keeping the Impact Pack SEGS interface. Shipping in-tree helpers under `nodes/batched_detailer_lib/` does **not** remove the copyright / GPL obligation to the Impact Pack original.

**Detailer (SEGS)**-style node that processes face (or other) segments in **three phases** instead of per-segment encode → sample → decode. This greatly reduces how often VAE and UNet are loaded and unloaded when using Dynamic VRAM Loading.

#### Problem with per-segment processing

Typical DetailerForEach runs, for each segment:

1. VAE encode  
2. KSampler (UNet)  
3. VAE decode  

So the pipeline does: VAE load → UNet load → VAE load → UNet load → … With many segments this causes repeated model switches and Dynamic VRAM reloads, leading to long stalls (especially with CUDAGraphs).

#### What HSWQ Batched Detailer does

- **HSWQ compatibility**: Keeps Impact Pack Detailer (SEGS) behavior while remaining usable with HSWQ quantized models (ConvRot INT8 / NVFP4, Dynamic VRAM, QuantizedTensor-related paths)
- **Phase 1 (VAE)**: Encode all segments → VAE is loaded once.  
- **Phase 2 (UNet)**: Run KSampler for all encoded latents → UNet is loaded once.  
- **Phase 3 (VAE)**: Decode all refined latents and paste back → VAE is loaded once.
- **Standalone**: Installing the separate [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) package is **not** required at runtime for this node. Helpers live in `nodes/batched_detailer_lib/`. Upstream detector / SEGS producer nodes from Impact Pack (or another SEGS source) may still be used in the workflow if you want face/object detection; that is optional workflow wiring, not an import dependency of this node. Provenance and GPL-3.0 obligations still apply (see Provenance above).

Model switches drop from **O(3n)** to **O(2)** (one VAE load, one UNet load per run). Input/output (INPUT_TYPES, RETURN_TYPES, etc.) is compatible with the original Detailer (SEGS) interface; behavior for a single segment is unchanged.

See Provenance above for GPL-3.0 base and HSWQ-compatibility improvements (`nodes/hswq_batched_detailer.py`, `nodes/batched_detailer_lib/`).

### HSWQ Sampler

<img src="png/sampler.png?v=2" alt="HSWQ Sampler" width="400">

A KSampler-equivalent node that behaves exactly like the standard ComfyUI KSampler, but **automatically adds all of RES4LYF's samplers and schedulers** when [RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF) is installed. It reproduces the dynamic sampler generation logic found in Forge so that the full Runge-Kutta (`rk_beta`) sampler family stays selectable and runnable in vanilla ComfyUI.

**Recommended:** **16 GB VRAM or more**.

#### Why this node exists

In Forge, RES4LYF's `beta/__init__.py` dynamically generates wrapper functions calling `sample_rk_beta` for every entry in `RK_SAMPLER_NAMES_BETA_NO_FOLDERS` (100+ RK samplers) and registers them into `extra_samplers`. The ComfyUI version of RES4LYF does not contain this logic, so many of those samplers become unselectable from the standard KSampler. This node supplements that missing difference.

#### Features

- **Standard KSampler behavior**: Same inputs (`model`, `seed`, `steps`, `cfg`, `sampler_name`, `scheduler`, `positive`, `negative`, `latent_image`, `denoise`) and output (`LATENT`); backed by `nodes.common_ksampler`
- **Automatic RES4LYF sampler discovery**: Scans `sys.modules` at `INPUT_TYPES` time, handling both `RES4LYF` and `custom_nodes.RES4LYF` module names (with a partial-match fallback), so load order does not matter
- **Forge-identical RK wrapper generation**: Builds `sample_fn` / `sample_ode_fn` closures for all RK sampler names, auto-generating ODE variants while excluding implicit samplers (gauss-legendre, radau, lobatto, etc.)
- **Reliable re-injection**: Registers every sampler into both `KSampler.SAMPLERS` (UI selectable) and `comfy.k_diffusion.sampling` via `setattr` (actual inference), guarding against RES4LYF's `importlib.reload()` wiping out function references
- **Scheduler merge**: Includes ComfyUI's `SCHEDULER_HANDLERS` in addition to the standard scheduler list
- **`clip_perfect_offload (Krea2 only)`**: Optional toggle that frees the Krea2 text encoder before sampling (see below)
- **`tensor_boost`**: Optional BOOLEAN (default **OFF**) for Blackwell Tensor Boost on SDXL ConvRot NVFP4 (see below)

#### Tensor Boost (`tensor_boost`)

Optional **`tensor_boost`** BOOLEAN (default **OFF**) for **SDXL ConvRot NVFP4** on NVIDIA Blackwell (SM >= 100: B200 / GB200, RTX 5090 / SM120). When **ON**, enables Per-Weight CUDA Graph auto-dispatch inside `nodes/nvfp4/` for faster fixed-resolution sampling (e.g. 1024×1024) and **VRAM use rises by several GB**. When **OFF**, clears graphs and uses Eager Pooled (no Graph-arena stack). Controlled via `HSWQ_NVFP4_TENSORBOOST` / `HSWQ_NVFP4_CUDAGRAPH`. Does **not** affect Z Image / INT8 / FP8 / stock paths. The Checkpoint Loader has no toggle — only this sampler and **HSWQ Ultimate SD Upscale**. Details: `md/HSWQ_SDXL_NVFP4_BLACKWELL_ACCELERATION_GUIDE.md`.

#### Krea2 text-encoder offload (`clip_perfect_offload`)

The optional **`clip_perfect_offload (Krea2 only)`** toggle reproduces the Krea2 benchmark's `clip.cond_stage_model.cpu()` behaviour: once the prompt is encoded, the Krea2 text encoder (TE) is no longer needed for the sampling loop, so unloading it before sampling frees the VRAM the DiT needs. On tight-VRAM cards the resident TE and the DiT would otherwise co-reside during sampling and push the run into OOM or loader thrashing.

The feature is deliberately narrow and safe:

- **Off by default**, exposed as an explicit opt-in widget.
- **Krea2-scoped both ways**: it runs only when the **MODEL is a Krea2 diffusion model** *and* only ever unloads a **Krea2 text encoder**. Both are identified by the loader's load-time tag (`_hswq_is_krea2`) and by exact module identity (`comfy.text_encoders.krea2`) — never by class-name guessing — so Z Image / Lumina2, Flux, SDXL, Qwen and WAN are never touched.
- **Strict toggle read**: only a real boolean `True` enables it. A misaligned value from an old saved workflow is refused and logged as OFF, so it cannot fire while the UI shows the toggle off.
- **Globally isolated**: the TE is freed by dropping its patcher from `current_loaded_models` (Python refcount); every other model (DiT / VAE / ControlNet) stays resident, and **no** global allocator op (`soft_empty_cache` / `empty_cache` / `unload_all_models`) is ever called, so a concurrent non-Krea2 workflow sharing the CUDA allocator is never disturbed.
- **Never breaks a run**: if the model is not Krea2 the toggle is ignored and logged; any unload failure is caught and sampling proceeds normally.

#### Usage Notes

- **Optional dependency**: Without RES4LYF installed, it works as a plain KSampler
- **Category**: `sampling`
- **Extensibility**: Designed as a thin UI wrapper so future HSWQ / Z-Image quantized-inference arguments can be intercepted in `sample()` without patching the ComfyUI core
- **Krea2 TE offload**: Leave `clip_perfect_offload (Krea2 only)` off for every non-Krea2 model; turn it on for Krea2 to reach bench-parity VRAM. For old workflows the pre-rename key `clip_perfect_offload` still maps to the same toggle.
- **Tensor Boost**: Prefer **ON** for the base SDXL ConvRot NVFP4 pass on Blackwell (**several GB more VRAM**); keep **OFF** on **HSWQ Ultimate SD Upscale** during tiles. This sampler: **16 GB VRAM or more**. Upscale / Tensor Boost headroom: **RTX 5090 32 GB+**.
- **Details**: See `md/hswq_sampler_technical_reference.md`, `md/HSWQ_KREA2_TE_OFFLOAD_GUIDE.md`, and `md/HSWQ_SDXL_NVFP4_BLACKWELL_ACCELERATION_GUIDE.md`

### HSWQ Torch Compile

<img src="png/torchcompile.png?v=2" alt="HSWQ Torch Compile" width="400">

ComfyUI node that wraps a loaded **MODEL** with PyTorch `torch.compile` for HSWQ diffusion paths (SDXL ConvRot INT8 / ConvRot NVFP4, Z Image / ZIT ConvRot NVFP4, and related USDU / Distorch workflows).

**Provenance:** This node was developed based on **[ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)** `TorchCompileModelAdvanced` (kijai), with original improvements and features added for HSWQ / USDU / Distorch / Windows inductor hardening. At runtime it calls ComfyUI core `comfy_api.torch_helpers.set_torch_compile_wrapper` and does **not** import the KJNodes package; that does **not** remove the copyright / GPL obligation to the KJNodes original.

Defaults are chosen for HSWQ + Ultimate SD Upscale: **inductor** + **`max-autotune-no-cudagraphs`**, `fullgraph` off, and Distorch weight-cast helpers marked eager so multi-tile USDU does not explode recompiles or hit CUDA-graph / `cudaMallocAsync` pool errors.

On Windows, other extensions (for example SeedVR2) may raise inductor `compile_threads` and force ProcessPool **spawn**. Spawn children can re-import ComfyUI `main.py` after `nodes.py` has inserted `comfy/` on `sys.path`, which shadows the `utils` package and crashes with `ModuleNotFoundError: No module named 'utils.install_util'`. This node forces **serial** inductor compile (`compile_threads=1`, `worker_start_method=subprocess`) and shuts down any already-warmed spawn compile pools before applying the wrapper.

#### Features

- **MODEL → MODEL**: Clones the input model and applies `torch.compile` via ComfyUI’s compile wrapper
- **Safe HSWQ defaults**: `backend=inductor`, `mode=max-autotune-no-cudagraphs`, `fullgraph=False`, `dynamic=false`
- **Block-only compile**: When enabled, compiles known transformer / block containers (`layers`, `double_blocks`, `single_blocks`, `transformer_blocks`, `blocks`, …) as `diffusion_model.<name>.<i>`; falls back to the whole `diffusion_model` if none are found
- **Distorch / dynamic VRAM**: Optional patch marks `comfy.ops` cast helpers (and `comfy_aimdo` raw-ptr helper when present) as eager graph breaks
- **Static parameter shapes**: Optional `force_parameter_static_shapes` to reduce symbolic `torch.Size` errors under nested inductor tracing of cast paths
- **Inductor hardening for ComfyUI**: Forces `compile_threads=1` and `worker_start_method=subprocess`; clears spawn ProcessPools left by other nodes
- **Optional `disable_dynamic_vram`**: Clones with `disable_dynamic=True` when the ComfyUI build supports it

#### Usage Notes

- **Inputs**: `model` (MODEL), `backend`, `fullgraph`, `mode`, `dynamic`, `compile_transformer_blocks_only`, `dynamo_cache_size_limit`, `force_parameter_static_shapes`, `patch_distorch_weight_cast`, `debug_compile_keys`; optional `disable_dynamic_vram`
- **Outputs**: MODEL (compiled clone)
- **Category**: `HSWQ/torchcompile`
- **Placement**: After **HSWQ Checkpoint Loader (SDXL)** or **HSWQ ConvRot INT8/ConvRot NVFP4 UNet Loader** (and any LoRA), before the sampler / **HSWQ Ultimate SD Upscale**
- **Avoid `cudagraphs` with multi-tile USDU**: Prefer inductor; CUDA graphs often fail on tiled / pool allocation paths
- **KJNodes**: Base of the torch.compile UI / wrapper design (GPL-3.0). Installing KJNodes is **not** required to run this node; attribution and GPL obligations still apply
- **Details**: See `md/HSWQ_TORCH_COMPILE_AND_ZI_INT8_PEEL_GUIDE.md`

## Changelog

See [changelog.md](changelog.md).

## Safety & License Notice

### Model Distribution & Usage

* **This repository does NOT distribute any model checkpoints, weights, or training data.**
* All model files (including SDXL checkpoints, quantized UNet files, CLIP, VAE, LoRA, and ControlNet models) **must be obtained separately by the user**.
* Users are solely responsible for ensuring that **all downloaded or generated model files comply with their respective licenses** (e.g., CreativeML Open RAIL, custom research licenses, etc.).
* The author does **not grant any rights** to redistribute, modify, or use third-party models beyond what is permitted by their original licenses.

### Quantized & Derived Models

* Quantized models (e.g., SVDQ / FP4 / INT4) are considered **derivative works** of the original checkpoints.
* Before sharing or redistributing quantized models, verify that the **original model license explicitly allows redistribution and derivative works**.

## License (GNU GPL v3)

This project is licensed under the **GNU General Public License, Version 3**.

### Key Points

* Copyright © 2024–2026 ussoewwin
* **Upstream HSWQ** ([Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)) is **fully developed by ussoewwin** and is licensed under **AGPL-3.0**. That repo’s AGPL terms apply to the quantization code and docs there; they are **not** the same as this loader’s license file
* **This repository** (ComfyUI loaders / tools) is licensed under **GPL-3.0** as stated above
* **HSWQ Ultimate SD Upscale** (`usdu_bundle/`, `nodes/nunchaku_usdu.py`, related USDU patches) was developed based on [ComfyUI_UltimateSDUpscale](https://github.com/ssitu/ComfyUI_UltimateSDUpscale) (ssitu, GPL-3.0), with original improvements and features added. Copyright in the ssitu original is retained; original work in this repository is © ussoewwin
* **HSWQ Torch Compile** (`nodes/hswq_torch_compile.py`) was developed based on ComfyUI-KJNodes torch.compile nodes ([ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes), GPL-3.0), with original improvements and features added. Copyright in the KJNodes original is retained; original work in this repository is © ussoewwin. That KJ provenance does **not** apply to the HSWQ quantization method itself
* **HSWQ Batched Detailer (SEGS)** (`nodes/hswq_batched_detailer.py`, `nodes/batched_detailer_lib/`) was developed based on [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) Detailer (SEGS) (ltdrdata, GPL-3.0), with original improvements and features added **especially to maintain HSWQ compatibility**. Copyright in the Impact Pack original is retained; original work in this repository is © ussoewwin. Installing Impact Pack is **not** required at runtime for this node
* You are free to **use, modify, and distribute** this software under GPL-3.0 terms.
* When you distribute this software or a modified version, you **must**:
  * Keep the copyright and license notices (including third-party / derived-work notices such as ssitu UltimateSDUpscale, KJNodes, and Impact Pack)
  * Provide the corresponding source code
  * License the distributed work under **GPL-3.0** (reciprocal / share-alike terms)
* This software is provided **"AS IS"**, without warranties or conditions of any kind.

See the full license text in [`LICENSE`](./LICENSE). Upstream HSWQ AGPL text lives in that project’s own `LICENSE`.
