<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/zhmd/v3.6.4.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

# ComfyUI-HSWQ-Loader-and-Tools v3.6.4

## Overview

**Whole-denoiser CUDA Graph Replay has been integrated into the SDXL ConvRot INT8 runtime**, delivering a **+22.1% steady inference speedup (3.27-3.28 → 3.35-3.36 it/s)** and a **+17.5% reduction in wall-clock time (9.23s → 7.62s avg wall/seed at 1024x1024)** with **bit-identical latent trajectories (maxdiff 0.0, 0/25 bifurcated)**.

Following the success of whole-denoiser launch cost elimination on the SD1.5 INT8 stack, this release ports the architecture to SDXL. While pooled activation rotation previously solved the FP32 promotion bottleneck (slashing per-step rotate time from 257.3 ms to 52.5 ms), forward execution remained CPU-launch bound. Whole-denoiser CUDA Graph Replay eliminates the remaining CPU launch overhead across hundreds of kernel dispatches, amortizing capture during the first step and executing subsequent steps entirely on GPU.

The implementation is housed in a dedicated SDXL-only module (`nodes/sdxl_convrot_int8/sdxl_convrot_fast.py`) and wired graph-first into both the Checkpoint and UNet loader branches in `patches/comfy_quant_int8.py` with automatic eager fallback.

---

## Technical Architecture & Implementation Details

### 1. Dedicated SDXL Graph Executor (`_Sd15CudaGraphExecutor`)
Implemented in `nodes/sdxl_convrot_int8/sdxl_convrot_fast.py` (lines 359-500):
- **Signature Keying**: Replay graphs are cached per `(x, t, context)` input signature, with tensor shape, dtype, and device sorted. Any runtime dimension or context change safely triggers a fresh capture.
- **Buffer Isolation**: Static copy-in buffers are loaded before every replay invocation; output is returned via `sy.clone()` as an eager-equivalent tensor, ensuring output memory is never aliased into the internal CUDA Graph memory pool.
- **Robust Capture Setup**: Operates under `capture_error_mode="relaxed"`, executing 3 dedicated warmup passes on a side CUDA stream with full synchronization before and after capture.
- **Resolution Boundary**: `_MAX_CAPTURE_LATENT_SIDE = 128` covers up to 1024x1024 generation. Latents exceeding this dimension bypass capture and execute via the eager path.
- **Soft Failure Handling**: Any capture error logs a clear warning and falls back to eager execution without disarming the graph infrastructure for other signatures.

### 2. Capture Guard in Pooled Activation Rotation (`_capture_active()`)
During development, a critical silent failure mode was uncovered:
- **The Pitfall**: The pooled activation rotate mechanism utilizes shared per-shape buffers across layers. During graph capture and warmup, reusing static shared buffers corrupted intermediate activation states across multiple graph nodes, resulting in catastrophic replay divergence (maxdiff 2.31 / 3.41 / 4.02 across 3 test configurations).
- **The Solution**: The executor asserts `_STATE["in_graph_capture"] = True` during warmup, capture, and copy-in phases.
- **Guarded Execution**: Both `_pooled_rotate` and `_pooled_helper_rotate` inspect `_capture_active()`. Inside the capture window, they automatically switch to standard, unshared activation rotation. The captured graph records pristine, unshared kernel sequences, restoring replay numerics to **bit-equal precision (maxdiff 0.0)**.
- Outside the capture window, standard non-capturable eager steps continue to utilize pooled buffers for optimal speed.

### 3. Graph-First Arm Ladder in Loader Branches
Both loader branches in `patches/comfy_quant_int8.py` implement an automated graph-first fallback ladder:
```python
if is_convrot and needs_conv2d and _model_is_sdxl_unet(model):
    _arm_sdxl_cudagraph(model)  # Attempts CUDA Graph arming
    # If arming succeeds: graph-first execution enabled
    # If arming fails: logs warning and arms pooled-rotate eager path
```
The two execution paths are strictly mutually exclusive per model instance, preventing stale-buffer corruption.

### 4. Framework Compatibility & Fail-Closed Audit Updates
- **ComfyUI v2.14 / inner.adm_channels Compatibility**: In ComfyUI v2.14, `adm_in_channels` was relocated from `UNetModel`. The loader arm guard was updated to inspect `dm.adm_in_channels` or `inner.adm_channels` (2816 on SDXL), resolving an issue where the loader previously refused to arm under CUDA 13.2.
- **Fail-Closed AST Rotate Audit**: ComfyUI v2.14 introduced `backends/ascend/__init__.py` rotate call sites. Because HSWQ maintains a strict fail-closed security audit against uninspected rotation callers, `backends/ascend/__init__.py` was verified and registered in `_AUDITED_ROTATE_FILES`, satisfying the AST contract.

---

## Runtime Capture Blockers Resolved

Three runtime edge cases that trigger CUDA Graph stream capture aborts were systematically diagnosed and resolved:

| Failure / Error | Root Cause | Engineering Resolution |
|---|---|---|
| `Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned` | `model_sampling.timestep()` reads `log_sigmas` residing on CPU after ModelPatcher load under CUDA 13.2. | Executor performs a one-time migration of `model_sampling` buffers to the active GPU device prior to side-stream warmup. |
| `dependency created on uncaptured work in another stream` (`cudaErrorStreamCaptureIsolation`) | Asynchronous mid-forward stream events triggered by ComfyUI weight-offloading routines. | Weights must be fully resident on device (`full_load=True`). Any uncaptured cross-stream synchronization drops gracefully to eager mode. |
| Replay divergence under pooled rotate (maxdiff 2.31–4.02) | Shared intermediate output buffers reused across sequential layers inside the captured graph stream. | `_capture_active()` guard forces unshared rotation during capture window (§2). |

Because `model_sampling.timestep()` buffers are migrated once on-device, eager execution numerics remain strictly bit-identical.

---

## Production Benchmark Measurements

Evaluated on the reference benchmark stack (`benchmark/sdxl_int8_traj_compare.py`, commit `e84e60b`) utilizing the reference reverse-hybrid pack (K=597, CLIP INT8 133, 1024x1024, 25 fixed seeds, 25 steps, `dpmpp_2m` / `karras`):

| Metric | FP16 Eager Baseline | SDXL INT8 Graph Replay | Difference / Gain |
|---|---|---|---|
| **Steady Speed (it/s)** | 2.60 – 2.76 it/s | **3.35 – 3.36 it/s** | **+22.1% faster** |
| **Avg Wall Time / Seed** | 9.23 s | **7.62 s** | **+17.5% faster** (includes capture step) |
| **Capture Passes Required** | — | 1 (`x=(2,4,128,128), ctx=(2,77,2048)`) | Amortized on step 1 |
| **Final-Cosine Mean** | 0.96223 | **0.96223** | **Bit-identical** |
| **Trajectory Bifurcations** | 0 / 25 | **0 / 25** | None |
| **Max Absolute Difference** | Baseline | **0.00000** | Perfect numerical identity |

---

## Architectural Scope & Isolation

- **Zero Cross-Contamination**: All SDXL CUDA Graph replay logic resides exclusively within `nodes/sdxl_convrot_int8/sdxl_convrot_fast.py` and guarded SDXL loader branches in `patches/comfy_quant_int8.py`.
- **Untouched Architectures**: SD1.5, Z Image, Krea2, FLUX, Qwen, and SAM3 pipelines do not import or execute any part of this code (0-byte footprint).
- **Two-Stage Verification Gate**: Code was first deployed and tested on the isolated benchmark stack (Stage 1 gate: bit-equal replay +22.1% at 1024x1024), followed by production loader wiring (Stage 2).

---

## Operator Runbook & Diagnostics

When loading an SDXL ConvRot INT8 pack (`int8_tensorwise` or reverse hybrid):
1. **Normal Armed Operation**: The console outputs `[HSWQ INT8][SDXL graph] armed`, followed by `captured #1` on the first denoising step. Subsequent steps replay at full 3.35+ it/s.
2. **Arm Fallback Warning**: If `arm failed ... fallback to pooled-rotate eager path` appears, execution continues reliably at the pre-graph speed baseline (3.27–3.28 it/s).
3. **Capture Skip**: If `capture failed` appears for a non-standard tensor shape, that specific signature executes via eager mode without affecting other resolutions or image quality.
