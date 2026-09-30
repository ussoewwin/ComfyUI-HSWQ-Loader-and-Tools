# HSWQ Speed & VRAM Optimization Plan — ConvRot INT8 / Hybrid ConvRot NVFP4

- Created: 2026-09-30
- Status: Plan (not implemented; all items are proposals pending Owner approval)
- Scope: `ComfyUI-HSWQ-Loader-and-Tools` runtime paths (ConvRot INT8 + Hybrid ConvRot NVFP4), informed by a full read of the NVIDIA Model Optimizer repository
- Method: every claim below was verified against the actual source on this machine (file + line references included). No unverified assertion.

---

## 0. Measured current state (code-verified baseline)

### 0.1 ConvRot INT8 path

| Component | File | Verified behavior |
|---|---|---|
| Linear GEMM | comfy_kitchen `backends/cuda/__init__.py::int8_linear` (4,393 L) | cuBLAS INT8 GEMM; has a fused ConvRot kernel family (`_fused_convrot_ok` gate, `_CONVROT_FUSED_MAX_K`) and an M=1 fast kernel (`int8_linear_m1`, `k <= 2560 or (6144 and n<=128)`) |
| Conv2d | `patches/comfy_quant_int8.py:952` `_make_quantized_conv2d` | Online act rotate (`rotate_activation_nchw`, fp32 accumulate) -> `cast_bias_weight` -> `F.conv2d`; weight dequant happens per call via cast path (`forward_comfy_cast_weights` L1062-1083) |
| SDXL fast path | `nodes/sdxl_int8/sdxl_convrot_fast.py` | Forward-window arm/disarm: swaps `_rotate_activation` -> pooled-buffer `_pooled_rotate`, disables kitchen fused-kernel gates (`_should_use_convrot_fused_kernel` -> False), **pool fully released outside the window** (`_disarm_kernels` L280-293). Fail-closed call-site audit before wrapping (L319-325) |
| LoRA bake | `patches/comfy_quant_int8.py:1381-1525` | baked-key set + `patches_uuid` invalidation (`_maybe_invalidate_baked_keys`), LowVramPatch strip for baked keys, LowVramPatch `intermediate_dtype` fix (L1323) |
| LoRA Conv2d roundtrip | `patches/comfy_quant_int8.py:1089-1174` | convert_weight unrotates on CPU (`build_hadamard(gs, device="cpu")` L1124), set_weight re-rotates on CPU (L1147-1151) |

### 0.2 Hybrid ConvRot NVFP4 path

| Component | File | Verified behavior |
|---|---|---|
| Product TC path (SDXL / Z Image) | `nodes/nvfp4/nvfp4_forward.py` (584 L) | pooled act quantize -> `scaled_mm_nvfp4_pooled` -> raw `_C.cublas_gemm_blockwise_fp4`; **CUDA Graph wired**: per-weight graph on Blackwell (`_bw and orig_m <= _PER_WEIGHT_GRAPH_MAX_M=16384`), env gates `HSWQ_NVFP4_CUDAGRAPH` / `HSWQ_NVFP4_TENSORBOOST` (`nvfp4_conf.py::is_nvfp4_cudagraph_enabled`) |
| Graph caches | `nodes/nvfp4/nvfp4_runtime.py` | shape-shared LRU 32 + per-weight LRU 500 (`_PER_WEIGHT_GRAPH_CACHE_MAX`, "~140 unique Linears in SDXL UNet"); output intentionally NOT pooled (residual lifetime); per-call amax in rotated domain |
| Krea2 parity path | `nodes/krea2_convrot_nvfp4/nvfp4_forward.py` (671 L) | same TC primitives, **CUDA Graph NOT wired** (`cudagraph` references = 0 in this module; `nvfp4_quant_mm_cudagraph` defined in its runtime but never called); act scale per-call amax with optional freeze `HSWQ_NVFP4_ACT_AMAX_FREEZE` (`nvfp4_runtime.py`); alpha caching keyed to scale object identity (`_hswq_nvfp4_alpha_bound_scale` L240-251) |
| Weight residency | `nvfp4_forward.py:134-161` `_plain_weight_cached` | weight stays PACKED (explicit "VRAM win -- never bake on the TC path"); bake is fallback-only (`bake_nvfp4_weight_inplace` L496) |
| LoRA residual | `nvfp4_forward.py:278-348` | low-rank float residual added post-forward (4-bit requant rounds away deltas); GPU-cached terms (`_hswq_krea2_lora_res_gpu`) |
| Hadamard rotate | `nvfp4_runtime.py:64-95` `rotate_last_dim_pooled` | dense 256x256 GEMM in fp32 accumulate; measured "~15x faster than butterfly" (comment L427-429); pooled out buffers `_ROT_OUT_POOL` |

### 0.3 Shared infra

- DisTorch2 (`distorch/distorch_2.py`, MultiGPU port), SA2 (`hswq/hswq_sa2_accel.py`, pattern resolved by checkpoint probe with cross-check L118-152), VAE tiled, torch compile (inductor-configured `nodes/hswq_torch_compile.py`), mgpu memory logging.
- GPU assumption: Blackwell consumer (RTX 5060 Ti measured in prior plans; SM>=100 gate `is_blackwell_gpu`).

### 0.4 Prior-plan context (from `md/2026-09-10_sa2_attention_acceleration_plan.md`)

- SA3 was fully rejected (FP4 attention error accumulates; NVFP4+SA3 cos 0.0556). SA2 is the accepted attention accelerator.
- Phase 0 profile (1024x1024, 6 steps, RTX 5060 Ti): **attention 26.4% / rotate 16.9% / cuBLAS FP4 GEMM 19.6% / quantize 1.8%**. Attention and rotate are the top-2 bottlenecks after attention is handled.

---

## 1. Plan A — Hybrid ConvRot NVFP4 (priority 1: largest headroom)

### A1. Wire CUDA Graph into the Krea2 parity path

- **Evidence**: `nodes/krea2_convrot_nvfp4/nvfp4_runtime.py` defines `_GRAPH_CACHE` (LRU 32) and `nvfp4_quant_mm_cudagraph` but no forward module calls it (grep: 0 references in `nvfp4_forward.py`).
- **Change**: port the per-weight graph dispatch from `nodes/nvfp4/nvfp4_forward.py` (per-weight cache, M <= 16384, stable weight address) into `nodes/krea2_convrot_nvfp4/nvfp4_forward.py::_tc_forward_pooled`.
- **Gate**: enable only when the act scale is frozen (`HSWQ_NVFP4_ACT_AMAX_FREEZE=1`) or the module carries a static `input_scale` -- a graph replays a captured `scale_a`; per-call amax is incompatible with graph replay. Keep a per-module opt-out `_hswq_nvfp4_no_cudagraph` mirroring the product path.
- **Separation rule**: the Krea2 parity path gets its own copy of the dispatch logic in its own folder; do NOT merge with `nodes/nvfp4/` (separation philosophy; the two paths have different scale semantics).
- **Expected effect**: removes per-call Python/CUDA-launch overhead on the parity path; product path already proved the mechanism (graph hits counter `_BLACKWELL_GRAPH_HITS` exists).
- **Risk**: low. Fallback to eager pooled execution on any capture failure (product path pattern).

### A2. Make act-scale freeze the default on the parity path

- **Evidence**: `ensure_act_scale_cached` computes amax on first call; `ensure_act_scale_amax` re-derivation remains live for unfrozen layers; the alpha rebind branch exists specifically because scale objects can swap (L240-251).
- **Change**: default `HSWQ_NVFP4_ACT_AMAX_FREEZE` semantics to ON (env can still force OFF), and finalize all layer amaxes during the first forward pass (step-0 sweep), turning `scale_a` into constants. This deletes the alpha-rebind branch, makes A1 graphs replay-safe everywhere, and removes first-step jitter.
- **Accuracy guard**: the rotated-domain amax convention must stay exactly as-is (`scale_a` computed AFTER rotation; converter rotates first then amax -- L229-231 comment). Freeze stores that rotated-domain value; it does not change the domain.
- **Validation**: SSIM trajectory comparison frozen-vs-dynamic on the standard 1024x1024/12-step protocol; acceptance = SSIM delta within noise of the dynamic path.

### A3. Fuse rotate + NVFP4 quantize (Triton)

- **Evidence**: rotate is the #2 kernel cost (16.9% of step time, prior-plan Phase 0 profile) and is a separate fp32 GEMM + copy (`rotate_last_dim_pooled` does `matmul(out=)` then `out.copy_(out32)` L93-94 -- one extra full-tensor pass).
- **Change**: one Triton kernel that reads bf16/fp16 x, applies the 256x256 Hadamard via the existing K=64 H4 Kron structure (`nvfp4_hadamard.py::_apply_kron_h4_unnorm`), accumulates in fp32 registers, then does amax + E2M1 pack + block-scale emit directly (reusing the `quantize_nvfp4` CUDA primitive for the pack step, or reimplementing pack in Triton following ModelOpt's `fp4_kernel.py`).
- **Reference implementations (ModelOpt)**: `modelopt/torch/kernels/quantization/gemm/fp4_kernel.py` (400 L Triton: `static_blockwise_fp4_fake_quant_kernel`, `compute_fp4_scales`), `attention/bmm2_qdq.py` (`_p_qdq_nvfp4`, `_v_qdq_nvfp4`, on-write fake-quant fused into attention operands -- same fusion shape as rotate+quant).
- **Expected effect**: removes the `out32->out` copy and one global-memory round trip; combined with A2 the whole rotate+quant stage becomes one kernel.
- **Risk**: medium. Numerical parity gate: per-layer output cos-sim vs current path >= 0.999 on the bench harness (`nvfp4bench_sdxl.py` exists for exactly this).

### A4. Per-layer weight-scale sweep (offline, quantizer-side)

- **Evidence**: ModelOpt `kernels/quantization/gemm/nvfp4_fp8_scale_sweep.py` (290 L) evaluates N FP8-scale candidates in one Triton kernel (`_fp8_scale_sweep_kernel`) including a Hessian-weighted variant (`nvfp4_fp8_scale_sweep_hessian`).
- **Change**: port the sweep into the HSWQ quantization script. HSWQ already computes per-layer sensitivity (DualMonitor + weighted histogram); use that importance map as the Hessian proxy weights. For each layer, generate scale candidates from the `_fp8_scale_candidates.py` logic, evaluate end-to-end layer-output MSE in one pass, pick the min-error scale, and bake it into the pack as `input_scale`.
- **Payoff**: better SSIM at identical size/speed, or the same SSIM with more layers pushed to NVFP4 (fewer FP16 protect layers -> smaller packs, faster). This directly competes with the FP16-protect budget: measure layers-moved-vs-SSIM and re-derive the protection set.
- **Also applicable**: ModelOpt's Local-Hessian weight-scale search (`model_calib.py::local_hessian_calibrate`, `_LocalHessianAccumulator`) is the same idea with a per-layer Hessian accumulator; compare both scorers offline and keep the winner.

---

## 2. Plan B — ConvRot INT8 (priority 2)

### B1. Layer-wise gate for the kitchen M=1 fused kernel on Z Image / Krea2 INT8

- **Evidence**: kitchen `int8_linear` has an M=1 fused ConvRot path limited to `k <= 2560 or (k==6144 and n<=128)` (`convrot_m1_supported` / `nonconvrot_m1_supported`). The SDXL fast path deliberately disables the fused-kernel gates (`sdxl_convrot_fast.py` `_arm_kernels` L255-262) because the pooled rotate wins for SDXL shapes.
- **Change**: for Z Image / Krea2 INT8 (NOT SDXL -- its path is separate and proven), add a per-layer gate that uses `int8_linear_m1` where the shape fits and falls through to the existing route where it does not. No change to the SDXL fast path (separation).
- **Expected effect**: verify per-layer M first with a profile pass before implementing; do not implement on the unverified assumption that M=1 dominates.

### B2. Packed-weight INT8 Conv2d (implicit GEMM)

- **Evidence**: `QuantizedConv2d.forward_comfy_cast_weights` (L1062-1083) casts the (still-INT8) weight to float per call through `cast_bias_weight`; the conv itself runs in float on dequantized weights.
- **Change**: dedicated SDXL-first kernel: im2col + INT8 GEMM with the weight staying INT8 in its rotated basis, activation rotated per an A3-style fused kernel (INT8 variant: rotate->quantize->GEMM fused; kitchen's `quantize_int8_convrot_staged` and `_convrot_int8_fused_shared_memory_bytes` show the smem budget approach).
- **Scope limit**: SDXL 3x3 stride-1 pad-1 convs first (the dominant pattern); other shapes keep the current path. Verify `k%4` (K = in_channels x 9 after im2col -- most SDXL convs have in_channels in {320,640,1280,2560}, K in {2880, 5760, 11520, 23040}, all %4==0 by construction, but the gate must still check).
- **Reference**: ModelOpt `kernels/quantization/conv/implicit_gemm_cuda.py` (229 L, conv3d FP4 implicit GEMM with fp4_fake_quant) for the launch/tiling structure.
- **Expected effect**: removes per-call weight dequant traffic on conv layers (the largest remaining INT8 overhead after attention+rotate), and removes the float weight residency during the conv call.

### B3. Differential / async LoRA bake

- **Evidence**: `_bake_int8_patches_on_dynamic_patcher` re-bakes on `patches_uuid` change; `convert_weight`/`set_weight` unrotate/re-rotate on CPU with `build_hadamard(gs, device="cpu")` per module (L1124, L1150).
- **Change**:
  1. keep a per-uuid bake manifest (key -> baked tensor source hash); on uuid change, bake only keys whose patch dict actually changed (differential bake);
  2. move the Hadamard build to a cached GPU tensor (the SDXL fast path already proves GPU-cached Hadamard via `get_hadamard_on_device`);
  3. run the bake stream-async off the sampling stream and fence before the first sampling step that uses the patched model.
- **Expected effect**: stacked-LoRA switches stop re-processing unchanged layers entirely; first-switch latency drops by the async overlap.

---

## 3. Plan C — VRAM reduction (both paths)

### C1. Text-encoder quantization: HSWQ-flavored Q8_0 (largest VRAM win)

- **Evidence**: ModelOpt ships GGML-format CUDA encoders: `kernels/quantization/ggml/q8_0.cu`, `iq1_s.cu`, `iq2_{xxs,xs,s}.cu`, with a clean per-format registry (`quantization/ggml/common.py::IQFormat`: `name/block_size/block_bytes/effective_bits` + `fake_quant` dispatch; `registry.py::IQ_FORMAT_REGISTRY`).
- **Change**: add a CLIP/T5 loader path in the HSWQ pack that quantizes the text encoder to Q8_0 with per-layer FP16 protection taken from the same sensitivity analysis used for the UNet (a "TE pack" with `comfy_quant`-style conf per layer). Q8_0 is ~8.5 bits -> T5-XXL-class encoders drop roughly ~2x; protected layers keep FP8/FP16.
- **Why Q8_0 first**: text encoders are the most sensitive to IQ2-class grid noise in practice; Q8_0 is near-lossless while still halving. IQ2_XXS (2.0625 bpw) is the follow-up experiment for CLIP-L-style encoders only, gated by SSIM eval.
- **Integration**: must not fight ComfyUI-GGUF (which already covers GGUF CLIP loaders); the differentiator is HSWQ sensitivity-aware mixed protection inside one pack. If Owner prefers, make the loader accept ComfyUI-GGUF files for TE and keep HSWQ TE packs as the optional quality path -- decide before implementing.

### C2. Re-pack fallback-baked layers automatically

- **Evidence**: the TC-path fallback bakes weight to dense float (`bake_nvfp4_weight_inplace`, `nvfp4_forward.py` L489-503) and never returns it to packed residency; counters (`_DEQUANT_FALLBACKS`) exist but nothing re-packs.
- **Change**: record baked-away keys in `_hswq_baked_float_keys`; on the next sampling start (or an explicit "restore packed residency" hook), requantize from the float weight back to `TensorCoreNVFP4Layout` and free the dense copy. Gate on the original QT being recoverable (retain packed bytes on CPU: ~1/3 of weight size, freed on unload).
- **Expected effect**: long sessions with intermittent non-Blackwell fallbacks (or TC-gate refusals) stop leaking dense float residency.

### C3. Residency/efficiency status dump

- **Evidence**: pieces exist but are scattered (`_TC_HITS`, `_DEQUANT_FALLBACKS`, `_TC_FLOPS`, baked-key sets, `_GRAPH_CACHE` sizes, `_lora_patcher_stats`).
- **Change**: one aggregated dump (the existing `summarize_int8_lora_capability` pattern) reporting: packed-resident layers, baked-float layers, graph hit rate, pooled-buffer count, per-path FLOPs. Zero behavior change; measurement-only; feeds every decision above.

### C4. Skip-softmax sparse attention (video/USDU only)

- **Evidence**: ModelOpt `sparsity/attention_sparsity` (calibration -> `DynamicThresholdCalibrator` -> Flash/Triton skip-softmax with decode/prefill phase separation) is proven for long-context decode; SA2 already covers the dense case well.
- **Decision**: static SDXL 1024x1024 (64x64 tokens) has too little sequence length for threshold skipping to pay; do NOT pursue for SDXL. Candidate only for SeedVR2/USDU tile-decode work where sequence length is large. Keep out of this cycle unless Owner requests it.

---

## 4. Implementation order (effect / risk, with hard separation)

| # | Item | Type | Size | Risk | Depends on |
|---|---|---|---|---|---|
| 1 | C3 residency dump | infra | S | none | -- |
| 2 | A2 scale-freeze default | speed | S | low (SSIM gate) | C3 to verify residency |
| 3 | A1 Krea2 CUDA Graph | speed | M | low | A2 (graph-safe scales) |
| 4 | B3 differential LoRA bake | speed | M | low | -- |
| 5 | C1 TE Q8_0 pack | VRAM | M | medium (new format) | Owner decision on GGUF coexistence |
| 6 | B2 INT8 packed conv2d | speed+VRAM | L | medium | bench harness |
| 7 | A3 rotate+quant fusion | speed | L | medium | parity gate >= 0.999 |
| 8 | A4 scale sweep | quality | M | low (offline) | sensitivity data format |
| 9 | B1 M=1 gate (ZI/Krea2) | speed | M | medium | profile-first proof |

## 5. Validation protocol (every item)

- SSIM: the established 1024x1024 / 12-step trajectory comparison vs the current main branch, same seeds.
- Speed: per-step ms via existing bench nodes (`nvfp4bench_sdxl.py`, SA2 `_log_stats` pattern); report kernel-level counters (TC hits / fallbacks / graph hits / rotate counts) from the status dump.
- VRAM: `multigpu_memory_log` snapshots at load / mid-run / peak.
- Separation: every change lands as a dedicated module or a clearly-gated branch; shared patches (`comfy_quant_int8.py`) take only minimal, cause-scoped edits; non-target architectures verified behavior-identical (no-op) before merge.

---

## 6. References (ModelOpt files verified on this machine)

- `modelopt/torch/kernels/quantization/gemm/fp4_kernel.py` / `fp4_kernel_hopper.py` -- Triton FP4 blockwise fake-quant kernels
- `modelopt/torch/kernels/quantization/gemm/nvfp4_fp8_scale_sweep.py` + `_fp8_scale_candidates.py` -- one-kernel scale-candidate sweep (incl. Hessian-weighted)
- `modelopt/torch/kernels/quantization/conv/implicit_gemm_cuda.py` + `implicit_gemm_kernel.cu` -- implicit-GEMM quantized conv structure
- `modelopt/torch/kernels/quantization/ggml/` -- Q8_0/IQ1_S/IQ2_XXS/IQ2_XS/IQ2_S CUDA encoders + `IQFormat` registry
- `modelopt/torch/kernels/quantization/attention/bmm2_qdq.py` -- QDQ fusion into attention operands
- `modelopt/torch/sparsity/attention_sparsity/` -- threshold-calibrated skip-softmax (video-scale context only)
- `modelopt/torch/quantization/calib/nvfp4_act_headroom.py` -- act headroom scale calibration concept
- `modelopt/torch/quantization/model_calib.py::local_hessian_calibrate` + `_LocalHessianAccumulator` -- Local-Hessian weight-scale search
- `modelopt/torch/quantization/utils/shared_input.py::find_shared_input_groups` -- tied/shared-weight scale consistency
