# SA3 NVFP4 → HSWQ NVFP4 Acceleration Transfer Plan

- Created: 2026-09-09
- Restored: 2026-09-10 (English version; restored after the earlier rejection verdict was invalidated — see section 9)
- Status: Plan (partially implemented; improve A/C measured, improve D verdict **invalidated** and pending re-verification)
- Target: accuracy and speed improvements of HSWQ NVFP4 (Z-Image / ConvRot TC W4A4)
- Sources: full read-through of `D:\USERFILES\fp8e4m3\SageAttention` (every statement in this plan was verified against the source)

---

## 0. Goals and success criteria

| Item | Current | Goal |
|---|---|---|
| Accuracy (moodyRealMix_xhsEdition TC W4A4) | **nv100 confirmed: mean 0.97366** (20 x 10-digit seeds, cloud) | Keep (no degradation) |
| Accuracy (moodyProMix_collectorsEdition) | nv100 confirmed (0.96033) | Keep (no degradation) |
| Speed (NVFP4 Linear pre-processing) | 3 passes: rotate GEMM -> quantize -> GEMM | Fuse rotate + quantize into a single pre-processing kernel |
| calib `input_scale` dependency | convrot paths require calibration | Determine whether a block-scale-only mode can remove the calibration requirement |

**Overall success criteria**: achieve at least one of the above and do not break the existing parity check (`nvfp4_comfy_parity`).

### Measured-results summary (2026-09-10, first execution of this plan)

| Item | Result |
|---|---|
| Improve A (mean shift) | **Rejected**: broke the real model (seed42 0.615 vs OFF 0.984). Helps on a single layer, degrades across layers |
| Improve C (block-only) | **Implementation failed**: broke the default path (no env) and dropped the existing score to ~0.974 -> 0.924; all changes reverted |
| Improve B / D | Not started at that time (D later attempted; see its section) |
| Baseline re-measurement | After restoring 2015feb, existing nv100 = mean 0.96849 (10-digit seeds, local). The -0.005 gap vs cloud 0.97366 is an environment difference |

**Most important lesson**: a change must **never alter the default path (no env, no flag)**.
Immediately after implementing, re-measure the existing score (regression test) and confirm non-degradation before evaluating the new feature.

---

## 1. SA3 (SageAttention3) NVFP4 implementation analysis — measured from source

### 1.1 Overall pipeline (`sageattention3_blackwell/sageattn3/api.py`)

```
q, k, v (bf16/fp16, [B, H, L, D])
  -> preprocess_qkv (per_block_mean: bool = True)
     1. k -= k.mean(dim=-2)          # shift by the tensor-wide mean of K
     2. pad q,k,v from L to a multiple of 128 (zero fill; K,V padded the same way)
     3. [per_block_mean=True] triton_group_mean: split Q into groups of 128 tokens,
        qm[group] = mean, q -= qm  (per-block mean removal)
        [per_block_mean=False] qm = q.mean(dim=-2), q -= qm  (global mean)
     4. delta_s = qm @ k^T  (fp32, [B,H,num_groups,L_k])
  -> scale_and_quant_fp4 (Q) / _permute (K) / _transpose (V)
     - output: uint8-packed FP4 (D/2) + e4m3 scale (D/16)
  -> blockscaled_fp4_attn -> fp4attn_cuda.fwd (Blackwell kernel)
  -> output slice [:, :, :QL, :]
```

### 1.2 Quantization kernel (`csrc/quantization/fp4_quantization_4d.cu`)

- `CVT_FP4_ELTS_PER_THREAD = 16`, `BLOCK_SIZE = 128` (tokens)
- **Scale granularity: 16 elements per block**, `SF = amax / 6.0` -> rounded to `e4m3`, reciprocal applied, then `e2m1` pack
- FP4 conversion uses the PTX instruction `cvt.rn.satfinite.e2m1x2.f32`
- **SF memory layout**: 128x4 tiles
  `offset = (col/4)*256 + (col%4) + (row/16)*4 + (row%16)*16`
  (same family as the cuBLAS blockwise FP4 / tcgen05 SF layout; compatible with HSWQ's
  `scaled_mm_nvfp4_pooled` `(roundup_m 128, roundup_sk 4)`)
- **permute variant (for K)**: swaps in units of 32 tokens using `[0,1,8,9,16,17,24,25,2,3,...]`,
  pre-matching the MMA operand layout
- **trans variant (for V)**: transposes in shared memory, then applies the same quantization
- One kernel covers padding, amax, scale, FP4 packing and layout alignment end-to-end

### 1.3 Attention main loop (`csrc/blackwell/mainloop_tma_ws.h`)

- MMA instruction: `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3`
  (warp-level block-scaled FP4 MMA, used for both QK^T and PV)
- **`add_delta_s`**: adds the precomputed `delta_s` (float4) into the Q@K^T accumulator,
  restoring in software what per-block mean removal subtracted
- **`quantize` lambda (online FP4 quantization of P)**:
  softmax output P -> absmax -> `ue4m3` scale -> `packed_float_to_e2m1` -> straight into the PV MMA;
  `__shfl_xor_sync` combines SFs across the quad
- `softmax_fused.online_softmax_with_quant`: fuses softmax with P quantization
- Constraints: sm120/sm121 only; head_dim 64/128 only (256 falls back to SDPA in `api.py`;
  `DISPATCH_HEAD_DIM` / `launch.h static_assert` have no 256 branch = unimplemented);
  no GQA (`TORCH_CHECK(num_heads == num_heads_k)`); all of Q/K/V must have seqlen padded to a multiple of 128

### 1.4 SA3 design philosophy (implications for HSWQ)

1. **Smoothing is done by shifting**, not rotating (mean shift + delta restoration).
   Rotation (ConvRot/Hadamard) and shifting are **mathematically orthogonal** improvements
2. **No per-tensor scale**: a 16-element block scale (e4m3) alone is sufficient
3. **All pre-processing fused into one kernel**: no separate pad / amax / scale / pack / permute passes
4. **P is kept in FP4**: an attention-specific optimization (keeping the intermediate between GEMMs in FP4)

---

## 2. Measured behaviour of the current HSWQ NVFP4 path (bottleneck identification)

### 2.1 Current Linear forward (`nodes/zimage_nvfp4/zi_nvfp4_forward.py`)

```
input (bf16, >=3D)
  -> reshape 2D
  -> [ConvRot only] rotate_last_dim_pooled(input, H, gs=256)
       <- dense Hadamard GEMM (256x256), torch.matmul, output reuses _ROT_OUT_POOL
  -> _tc_forward_pooled
      - ensure_act_scale (calib input_scale / amax freeze)
      - alpha = scale_a * scale_b cache
      - [opt-in CUDA Graph] nvfp4_quant_mm_cudagraph(_perweight)
      - eager: quantize_nvfp4_act_pooled  (comfy_kitchen _C.quantize_nvfp4)
               -> scaled_mm_nvfp4_pooled   (_C.cublas_gemm_blockwise_fp4)
```

### 2.2 Known issues (code comments and measured facts from memory)

| # | Issue | Evidence |
|---|---|---|
| P1 | The dense Hadamard GEMM (gs=256) used for rotation is expensive. A butterfly version `rotate_last_dim_fast` exists but is **not wired into the production path** | `zi_nvfp4_forward.py` comment ("butterfly is ~15x slower, so dense is used"), `zi_nvfp4_hadamard.rotate_last_dim_fast` unused |
| P2 | rotate -> write back to memory -> quantize kernel reads again -> writes back -> GEMM: **two extra global-memory round trips** | two-stage structure in `nvfp4_runtime.py` |
| P3 | Layers without a calib `input_scale` freeze the amax (first amax pinned) and never follow distribution changes. **Note: behaviour differs per subsystem**: (1) `nvfp4_runtime.py` freezes the step-0 amax into `module._hswq_nvfp4_act_scale`; (2) the Krea2 side found a 0.05 SSIM degradation caused by step-0 noise amax and switched to per-call online amax; (3) the Z-Image side calls `ensure_act_scale` directly but caches `alpha = scale_a * scale_b` in `module._hswq_nvfp4_alpha` on first use (implicit freeze) | `ensure_act_scale_cached`, comments in `krea2_convrot_nvfp4/nvfp4_runtime.py` |
| P4 | For small M, NVFP4 GEMM loses to FP16 (quantization overhead dominates) | comment on `_GRAPH_MAX_M = 512` |
| P5 | Accuracy: nv90-nv84 never reached 0.95. Scale tuning (nv values) has plateaued | memory/2026-08-26.md, K-search results |

### 2.3 Structure of the bottleneck

- **Speed**: P1 + P2 share one root cause (pre-processing is split into separate passes); P4 is partly caused by the same pre-processing cost
- **Accuracy**: P3 plus the limits of the per-tensor scale scheme. A 16-element e4m3 block scale is already fine-grained enough, so what remains is **smoothing the activation distribution itself** (rotation is already applied; shifting is not)

---

## 3. Detailed design of the improvements

> Design principle (Owner's philosophy): **each improvement lives in a dedicated code path**.
> "Shared code plus conditionals" is forbidden. Switch explicitly via an env flag or a separate
> function/file; never change the behaviour of the default path.

### Improve A: per-block mean shift (accuracy; the Linear-side delta_s)

**Origin**: SA3 `preprocess_qkv` group mean + delta_s.

**Math**: for a Linear `y = W·x`, with groups g of 128 tokens:

```
mu_g = mean(x in g)                  # (num_groups, in_features)
x'   = x - mu_g                      # shift within the group (broadcast)
delta_y = mu_g @ W^T                 # one small GEMM, (num_groups, out_features)
y    = W·x' + delta_y                # add delta_y broadcast per group
```

- `delta_y` is a single small GEMM of shape `(num_groups, out_features)` (groups = M/128).
  Same `mu @ W^T` form as SA3's `delta_s = qm @ k^T`
- **Ordering vs ConvRot**: Hadamard is a linear operation inside a group (gs=256), so
  `(x - mu)H = xH - muH`. The implementation therefore **takes the 128-token mean after
  rotation** (the same position as SA3: smoothing the distribution immediately before quantization)
- **Interference with `input_scale`**: shifting lowers the amax, so the calibrated scale and the
  measured distribution mismatch -> evaluate together with improve C (block-only), or recalibrate
  on the shifted distribution
- **alpha-cache interference**: the Z-Image side caches `alpha = scale_a * scale_b` in
  `module._hswq_nvfp4_alpha` on first use. Mean shift changes activations and therefore `scale_a`,
  so **the dedicated path must not use the alpha cache and must recompute every call**
- **Dedicated path**: a separate function `_tc_forward_pooled_meanshift` enabled by
  `HSA_MEANSHIFT=1` (never rewrite the existing `_tc_forward_pooled`)

**Risks**
- "128 tokens" after flattening the image latent is spatially contiguous, but depending on head
  ordering and channel layout a group may become an unnatural set -> decide by measurement
- Forgetting the `delta_y` addition breaks everything completely (immediately detectable in parity)

**Expected effect**: the smoothing that worked for SA3 video/image generation. Smaller activation
dynamic range -> lower W4A4 quantization error -> **the main candidate for reaching the 0.95 line**.

**Measured results (2026-09-10) — REJECTED**

| Configuration (seed 42, 1024x1024, 12 steps, TC) | final-cos |
|---|---|
| mean shift OFF (new-type artifact) | 0.98432 |
| mean shift ON | **0.60676** (broken) |
| mean shift ON + `scale_a=1.0` | 0.93669 |
| mean shift ON + excluding noise_refiner layers | 0.85668 |
| Single-layer unit test (ideal conditions) | 0.993 -> 0.9995 (improved) |

- **Cause**: layers such as noise_refiner have an extreme DC component in the input
  (mu/x ratio 0.77-0.998, output DC around 800). `delta` (`mu @ W^T`) dominates the output, and its
  small inaccuracy shifts the output DC by 42-64%
- It works correctly on a single layer (reduces quantization error) but consistently degrades
  on the real model (100 layers of accumulation)
- The implementation also had a side effect that broke the default path (see the default-path risk below)
- **Verdict**: rejected for this model (Qwen-Image / NextDiT). May be revisited for models with small DC
- **Measurement validity**: this test did not use the attention override, so it is **unaffected**
  by the harness bug described in section 9

### Improve B: fused rotate + quantize kernel (speed; the main candidate)

**Origin**: SA3 `fp4_quantization_4d.cu` (quantization completed in a single kernel).

**Current 3 passes**:
```
[bf16 act] -> matmul (rotate, gs=256 dense H) -> global write-back
           -> _C.quantize_nvfp4 (read again -> FP4) -> global write-back
           -> cublas_gemm_blockwise_fp4
```

**Fused single pass**:
```
[bf16 act] -> fused_rotquant_kernel:
   1. read the tile from global once (bf16)
   2. apply the gs=256 Hadamard butterfly in shared memory / registers
      (log4(256) = 4 stages of h4; same math as zi_nvfp4_hadamard.rotate_last_dim_fast)
   3. per 16 elements: amax -> SF = amax/6 -> e4m3
   4. e2m1 pack (cvt.rn.satfinite.e2m1x2)
   5. store the SF in the 128x4 layout (cuBLAS compatible; reuse SA3's offset formula)
   -> cublas_gemm_blockwise_fp4
```

- Removes two global-memory round trips (one bf16 read remains + the FP4 write, but the re-read
  disappears) and one kernel launch
- **Avoids materializing the Hadamard matrix** (256x256 bf16 = 128KB -> only butterfly constants)
- A **second kernel for non-ConvRot layers** (quantize only, no rotation) lives in the same
  extension as a **separate kernel** (never mixed; SA3's kernel can be reused almost as-is)
- Implementation form: a new CUDA extension `hswq_fused_rotquant`
  (`nodes/zimage_nvfp4/csrc/` or a standalone directory). Windows builds follow SA3's `setup.py`
  precedent. The MSVC SAL workaround (`#undef __in/__out/__inout`) goes **inside the `.cu`
  sources** (`api.cu` L20-25, `fp4_quantization_4d.cu` L19-24), not in `setup.py`
- Butterfly in CUDA: `zi_nvfp4_hadamard._apply_kron_h4_unnorm` is recursive Python
  `torch.matmul` with log4(256) = 4 stages. The CUDA version must become **4 in-place butterfly
  stages in shared memory** (using only the 4x4 Hadamard constants)
- Verification (two separated stages):
  1. **SF layout**: fused kernel vs current path must be **bit-exact**. Never accept a cosine
     threshold here: a 1-bit SF difference causes silent corruption
  2. **FP4 qdata**: first measure the cosine between butterfly and dense-GEMM rotate outputs,
     quantifying how far floating-point ordering differences push values across FP4 rounding
     boundaries, then set thresholds based on that
     (shapes: M in {1, 16, 128, 1024, 4096} x the model's main K/N shapes)

**Dedicated path**: enabled by `HSWQ_FUSED_ROTQUANT=1`. On failure it must **raise and log**
rather than auto-falling back (a silent fall back to the old path would contaminate measurements).

**Expected effect**: clears P1 + P2 + P4 (the pre-processing family). Aims for a line that is
consistently faster than FP16 while staying on the eager path without CUDA Graphs.

### Improve C: block-scale-only mode (determining whether calibration can be dropped)

**Origin**: SA3 works with only an e4m3 block scale, no per-tensor scale.

**Design**:
- With `HSWQ_NVFP4_BLOCKONLY=1`: pin `ensure_act_scale` to 1.0, use `alpha = scale_b` (weight side
  only). **Removes the per-tensor scale computation and the calibration dependency**
- **Invalidate the alpha cache**: the Z-Image side caches `alpha = scale_a * scale_b` in
  `module._hswq_nvfp4_alpha`. Even with `scale_a = 1.0` the existing cache would win, so the
  **block-only path must not use the alpha cache and must use `alpha = scale_b` directly**
- Applies explicitly to `_hswq_nvfp4_convrot` layers only (already-rotated distributions are
  smooth, so block-only may suffice). Non-convrot layers stay on the current path (separated)
- Evaluation: moodyRealMix 20 seeds cosine. **If the nv search (nv100-nv84) becomes unnecessary,
  the conversion workflow itself gets simpler**

**Expected effect**: "zero calibration cost" while keeping current accuracy, or accuracy gains when
combined with improve A. It is the cheapest change (a few lines of Python), so **do it first**.

**Measured results (2026-09-10) — IMPLEMENTATION FAILED, fully reverted**

- Implemented, but on the default path (no env) it degraded the existing calibrated NVFP4 score from
  **0.974 to ~0.924**
- All changes reverted (`8c04a1b` restored the exact 2015feb state)
- **Lesson**: "no env means the old path" is not sufficient by design. Immediately after implementing,
  **re-measure the existing score** and confirm non-degradation (a regression test is mandatory).
  Skipping it means silently breaking existing working behaviour
- Conditions for retrying: full separation into dedicated functions (never rewrite existing ones)
  plus a passing regression test
- **Measurement validity**: this test did not use the attention override, so it is **unaffected**
  by the harness bug described in section 9

### Improve D: introducing SA3 attention (additive acceleration)

- HSWQ NVFP4 covers Linear layers and does not touch attention -> **SA3 attention multiplies with it**
- Wheel at the time: `dist/sageattn3-1.0.0+cu132torch2.14.0-cp313/cp314-win_amd64.whl`
- **Pre-flight checklist**
  - [ ] ComfyUI Python version (3.13/3.14?) matches the wheel and torch version (2.14?)
  - [ ] CUDA runtime **>= 12.8** (setup.py requirement; not limited to 13.2)
  - [ ] GPU is sm120/sm121 (RTX 50 series)
  - [ ] Z-Image attention head_dim is **64 or 128** (SA3 implements only 64/128; 256+ falls back
        to SDPA in `api.py`; no branch in `DISPATCH_HEAD_DIM` / `launch.h static_assert`)
- Integration form: expose SA3 through a **dedicated node/flag** in the existing patch layer
  (`patches/` or the attention replacement nodes). SDPA fallback conditions (head_dim >= 256,
  unsupported shapes, GQA) must match SA3's constraints
- Accuracy: the README claims near-lossless for image generation, but parity must be measured on Z-Image

**Measured results (2026-09-10) — verdict INVALIDATED; re-verification required (see section 9)**

The pre-flight checklist passed in full (sageattn3 wheel installed, sm121, head_dim 128, CUDA 13.2).
Implementation was confined to the benchmark side (default path unchanged):
`zi_convrot_nvfp4_traj_compare.py --attention sage3`, `zi_int8_bench.py --attention sage3`
(attention overridden on the quantized model only; the FP16 baseline kept stock attention).

**SA3 standalone test** (random q/k/v, fp16, HND, head_dim 128, compared against SDPA) — **VALID measurement**

| Condition | cos vs SDPA |
|---|---|
| seq=4096, per_block_mean=True | **0.98192** |
| seq=4096, per_block_mean=False | 0.98185 |
| seq=4128 / 1024 | comparable (0.9817-0.9821) |

-> **SA3 (FP4 attention) carries a constant ~1.8% inherent error vs SDPA** (independent of per_block_mean).
This test does not use the model harness, so it is unaffected by the harness bug.

**Real-model verification** (FP16 baseline reference, local 5060 Ti) — **INVALID: harness bug**

| Combination | Metric | Result |
|---|---|---|
| NVFP4 Linear (existing nv100) + SA3 | final-cos (12 steps, seed42) | **0.0556 (collapse)** — same for per_block_mean True/False |
| INT8 Linear (sci_1off) + SA3 | latent-cos (25 steps, seed42) | **0.0909 (collapse)** — 43.3 s inference (30% faster than INT8 alone at 62.5 s) |
| INT8 Linear alone (no SA3) | latent-cos | 0.9862 (healthy) |
| NVFP4 Linear alone (no SA3) | final-cos | 0.987 / 0.984 (healthy) |

**Why these rows are invalid**: the SA3 attention override used the same layout bug
(`[B,H,N,D]` reshaped without `transpose(1,2)`) as the first SA2 harness. The buggy harness
produced 0.06006 for SA2 and 0.086 for FP16-only — the same order as the 0.0556 above.
Consequently **the "SA3 collapses the trajectory" conclusion is not supported by this data**
(see section 9 for the full correction record).

What remains valid for SA3:
- The standalone ~1.8% per-call error above (measured without the model harness)
- The INT8+SA3 speed observation (30% faster) is a wall-clock measurement and is likewise
  unaffected by the layout bug (although the quality of those runs is invalid)

What must be re-measured with the fixed harness: NVFP4+SA3 and INT8+SA3 quality.

- **Current verdict: NOT REJECTED — verdict invalidated, re-verification pending**
- The `--attention sage3` implementation was later removed from both benchmarks per the Owner's
  instruction (SA3 code and the `register_attention_function("sage3", ...)` core registration were
  cleaned out). Re-testing requires re-adding it as a **dedicated path, fully separate from SA2**

### Improve E (reference, out of scope): keeping P in FP4

Keeping the post-softmax P in FP4 is attention-specific. Z-Image's MLP/Linear layers have
non-linearities (SiLU etc.) between them, so it is not applicable. **Not implemented in this plan**
(raw material for a future self-written attention).

### Improve F (deferred): cuBLAS -> custom MMA

- SA3's instruction itself (`mxf4nvf4` MMA) is the same family cuBLAS blockwise FP4 uses internally.
  SA3 is fast because of its fusion design
- The small-M problem should improve once improve B removes pre-processing overhead ->
  **measure improve B first; revisit only if the GEMM itself remains the bottleneck**

---

## 4. Implementation phases (ordered by dependency and cost)

### Phase 0: baseline measurement (half a day)

- [ ] Obtain a kernel breakdown of the current eager path (torch profiler / nsys)
  - Buckets: rotate matmul / quantize_nvfp4 / cublas_gemm / other, plus wall-time ratios
  - **Measurement workflow**: moodyRealMix TC W4A4, 1024x1024, 12 steps, 1 seed
  - **nsys example**: `nsys profile --stats=true -t cuda,nvtx -o baseline -- python ...`
  - **FP16 baseline**: run the same workflow in FP16 and capture an nsys trace plus wall time under
    identical conditions; tabulate per-kernel ratios
- [ ] Accuracy baseline: re-confirm the current best moodyRealMix result (nv90-nv84 scores)
- [ ] Record moodyProMix (nv100) as the regression baseline
- **Exit criteria**: "pre-improvement numbers" fit in a single table
  (accuracy: 20-seed cosine summary; speed: per-kernel wall-time ratios for FP16 / NVFP4)

### Phase 1: improve C block-only experiment (half a day, minimal code)

- [ ] `HSWQ_NVFP4_BLOCKONLY` env plus a convrot-only branch (dedicated function)
- [ ] **Verify the alpha-cache invalidation**: confirm the dedicated path does not read
  `module._hswq_nvfp4_alpha` and uses `alpha = scale_b` directly
- [ ] Measure moodyRealMix 20-seed cosine with and without the alpha cache (print everything)
  -> simultaneously investigate whether the step-0 amax problem (found on Krea2) also exists on Z-Image
- [ ] Verdict: if equal to the calibrated result (+/-0.005), adopt as the calibration-free option
- **Exit criteria**: full 20-seed rows plus summary, with a clear adopt/reject verdict

### Phase 2: improve A mean-shift prototype (1 day, Python level)

- [ ] Implement the post-rotation 128-token mean shift plus `delta_y = W·mu^T` in Python
  (correctness first; no fusion yet)
- [ ] Parity: FP16 identical with mean shift ON/OFF, NVFP4 shows an accuracy improvement
- [ ] Measure moodyRealMix 20 seeds -> **decide here whether 0.95 is reachable**
- [ ] Check `input_scale` interference (two configurations, with and without improve C)
- **Exit criteria**: full 20-seed rows plus summary. If it misses, analyse the cause (distribution
  histograms)
- Note: if the effect is weak, only test the two shift positions (before/after rotation) and then stop.
  Do not multiply configurations endlessly

### Phase 3: improve B fused kernel (2-3 days, CUDA)

- [ ] Scaffold the `hswq_fused_rotquant` extension (reuse SA3's `setup.py`, including the MSVC workaround)
- [ ] Kernel 1: butterfly + quantize for convrot (gs=256)
- [ ] Kernel 2: quantize only (non-convrot; reuse SA3)
- [ ] Unit tests (two-stage verification):
  1. SF layout: fused kernel vs the current two-stage path -> **bit-exact required**
  2. FP4 qdata: measure the butterfly-vs-dense rotate cosine first, then set thresholds
     (shapes: M in {1, 16, 128, 1024, 4096} x the model's main K/N shapes)
- [ ] Wiring: `HSWQ_FUSED_ROTQUANT=1` swaps the pre-processing inside `_tc_forward_pooled`
- [ ] Benchmark: same conditions as Phase 0, wall-time comparison (average of 3)
- **Exit criteria**: all unit tests green plus a benchmark table (FP16 / current NVFP4 / fused NVFP4)
- Note: never modify a kernel before the cause is established by measurement (debugger/dump comparison).
  Debug dumps belong in scripts under `tools/`, never in production code
- Note: **recalibration caveat**. A calibrated `input_scale` assumes the accuracy characteristics of
  the dense GEMM path. Switching to butterfly changes the floating-point ordering and therefore the
  accuracy characteristics, so decide by measurement whether recalibration is needed after the fused
  kernel lands

### Phase 4: improve D SA3 attention integration (1 day)

- [ ] Pass the full pre-flight checklist (environment version matching)
- [ ] Replace Z-Image attention with `sageattn3_blackwell` via a dedicated node/flag
- [ ] Parity image generation plus wall-time measurement (ON/OFF independently of improves A/B)
- [ ] **Re-verify with the fixed harness** (see section 9): the original collapse numbers are invalid
- **Exit criteria**: visual image parity plus recorded numbers (PSNR/cosine) and an extra benchmark row

### Phase 5: integrated verification and finalisation (1 day)

- [ ] Run the chosen combination over the final 20 seeds x 12 steps (all rows + summary + GEMM MODE)
- [ ] moodyProMix nv100 regression (confirm no degradation)
- [ ] Final wall time in the real image workflow
- [ ] Record results in MEMORY.md / memory/YYYY-MM-DD.md and update repo docs if needed
  (docs keep general guidance only; model-specific numbers go to the reference sections)

**Estimated total: 6-8 days**

---

## 5. Verification protocol (following the existing rules)

| Item | Procedure |
|---|---|
| Accuracy | 20 seeds x 12 steps TC(W4A4) cosine. **Print every row plus the summary and GEMM MODE** |
| Speed | Median of 3 runs with identical seed/settings. Report the ratio against the FP16 baseline |
| Parity | Existing `nvfp4_comfy_parity` flow, run with each improvement ON/OFF |
| **Non-regression** | **Re-measure the existing calibrated NVFP4 baseline under identical conditions after a change, and confirm non-degradation before evaluating the new feature (the direct countermeasure to the 2026-09-10 failure)** |
| Seeds | Use the **canonical 10-digit seeds (42, 137, 5517, 92048, ...)**; ad-hoc seeds move the mean by up to 0.05 |
| Environment spread | Local (5060 Ti) vs cloud (5090 etc.) varies by about +/-0.005 in the mean. Exact equality is not required; values in the 0.97 range count as equivalent |
| Target collation | Before measuring, verify the artifact (genuine nv100) and the seeds (canonical 10-digit) |
| On failure | Delete artifacts from failed runs immediately (existing rule). Record at least a one-line failure cause first |

---

## 6. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **Breaking the default path (no env)** (highest priority) | Existing working behaviour breaks and the score drops 0.974 -> 0.924 (2026-09-10 example) | Additive changes only (never rewrite existing code). Re-measure the existing score immediately after implementing |
| mean shift breaking on the real model (extreme-DC layers) | Improve A rejected (2026-09-10 example) | Measure the per-layer mu/x ratio before implementing. Do not apply to models with extreme-DC layers |
| Evaluating a wrong artifact / seed | Invalid comparison leads to a wrong conclusion (2026-09-10 example) | Verify the artifact and seeds against the canonical ones before measuring (§5) |
| SF layout implementation error (fused kernel) | Silent corruption (quality loss that still passes parity) | Bit-exact unit test; on failure use a dump-comparison tool under `tools/` |
| mean-shift group structure not matching image statistics | Improve A has no effect | Early stop decision in Phase 2; save distribution histograms |
| torch/CUDA version mismatch (SA3 wheel) | Improve D does not run | The Phase 4 checklist blocks everything up front; rebuild from source if mismatched (build scripts exist) |
| comfy_kitchen `_C` private API change | The fused-kernel connection point breaks | Confine the connection to one function (`_tc_forward_pooled`); pin versions in setup |
| Fused kernel does not run in some environments | Phase 3 fails entirely | With the flag OFF the default path stays exactly as-is (raise an exception instead of hiding behind a fallback) |
| Windows MSVC + CUDA build problems | Build fails | Follow SA3's `setup.py` / `build_*.cmd` precedent (the SAL workaround lives inside the `.cu` sources) |
| The step-0 amax problem generalising to Z-Image | The alpha cache causes the same mis-scale as on Krea2, degrading accuracy | Phase 1 compares with/without the alpha cache; investigate whether Z-Image degrades the same way |
| Recalibration needed after switching to butterfly | The calibrated `input_scale` was computed assuming dense GEMM, mismatching the butterfly path | Decide by measurement after improve B; if needed add a butterfly-specific recalibration in Phase 3 |
| Distributing the `hswq_fused_rotquant` CUDA extension | Users without an MSVC + CUDA toolchain cannot use it | Add pre-built wheel build/publish steps to Phase 3 (use the build-windows-whl / build-linux-whl skills) |
| **Attention-swap harness layout bug** (new, 2026-09-10) | Invalid performance/quality verdicts (SA2 and SA3 were both misjudged) | Always run a **control experiment that returns the same math** (SDPA inside the override) before judging; copy the stock output shaping exactly (`transpose(1,2)` before reshape when `skip_output_reshape=False`) |

---

## 7. Target file map

### Read (reference implementations)
- `D:\USERFILES\fp8e4m3\SageAttention\sageattention3_blackwell\sageattn3\api.py` — pre-processing and quantization calls
- `D:\USERFILES\fp8e4m3\SageAttention\sageattention3_blackwell\csrc\quantization\fp4_quantization_4d.cu` — fused quantization kernel (base for improve B)
- `D:\USERFILES\fp8e4m3\SageAttention\sageattention3_blackwell\csrc\blackwell\mainloop_tma_ws.h` — delta_s / P quantization
- `D:\USERFILES\fp8e4m3\SageAttention\sageattention3_blackwell\setup.py` — Windows build precedent

### Change (HSWQ side)
- `nodes/zimage_nvfp4/zi_nvfp4_forward.py` — dedicated paths for improves A/C, wiring point for improve B
- `nodes/nvfp4/nvfp4_runtime.py` — scale pinning for improve C (convrot-only branch)
- `nodes/zimage_nvfp4/zi_nvfp4_hadamard.py` — butterfly implementation (reference when porting to CUDA)
- New: `hswq_fused_rotquant/` (CUDA extension) or `nodes/zimage_nvfp4/csrc/`
- Improve D: dedicated patch/nodes for the attention swap (following the existing patch structure)

### Do not change
- `nodes/nvfp4/nvfp4_forward.py` (SDXL side) — kept separate from Z-Image (never mixed)
- Existing parity / bench scripts (they are the test target)

---

## 8. Decision history (rationale for this plan)

- Improve B is positioned as the main speed candidate and improve A as the main accuracy candidate because
  SA3's speed advantage comes from fusion and smoothing design, while the instruction itself is the same
  family as cuBLAS (the same reason improve F is deferred)
- per-block mean is mathematically orthogonal to rotation, so it can be added straightforwardly to a
  ConvRot implementation such as HSWQ
- Improve C is done first because it has the lowest cost and provides the baseline data needed to
  evaluate improve A's `input_scale` interference
- Because Z-Image has non-linear layers, SA3's P-in-FP4 cannot be transferred (explicitly out of scope
  as improve E)

---

## 9. Correction record — attention-swap harness bug (2026-09-10)

### What happened

While implementing the SA2 attention swap, the first harness reshaped `[B,H,N,D]` into `[B,N,H*D]`
**without the `transpose(1,2)`** that the stock implementation (`attention_pytorch`) performs. The
resulting head/token scrambling collapsed the 12-step trajectory, which was initially reported as
"SA2 breaks" and, in the same family of harness templates, as "SA3 breaks".

### Evidence

| Run | Harness | Result |
|---|---|---|
| SA2, first harness (buggy) | reshape without transpose | final-cos **0.06006** |
| FP16 only, same buggy harness | reshape without transpose | final-cos **0.086** |
| SA2, fixed harness | `transpose(1,2)` then reshape | final-cos **0.98576** (seed42) |
| SA2, fixed harness, control (SDPA inside the override) | same math as stock | per-step cos ~ 1.000 (final 0.999968) |
| SA2, fixed harness, 20 canonical seeds | correct | mean **0.97468**, sd 0.01771, no bifurcation, 16320/16320 calls covered |
| SA3, model runs (2026-09-10) | same buggy template | final-cos 0.0556 (NVFP4) / 0.0909 (INT8) -> **invalid** |

### What is invalidated and what stands

| Item | Status |
|---|---|
| "SA2 breaks the NVFP4 trajectory" | **Invalid** (harness bug). SA2 measured at mean 0.97468 and -15.4% speed vs the existing NVFP4 |
| "SA3 breaks the NVFP4/INT8 trajectory" (0.0556 / 0.0909) | **Invalid** (same bug). Re-measurement required |
| SA3 standalone per-call error (~1.8% vs SDPA, 0.98192) | **Valid** (does not use the model harness) |
| INT8+SA3 wall-clock speed-up (30%) | **Valid as a timing observation**; the quality of those runs is invalid |
| Improve A (mean shift) rejection | **Valid** (measured without the attention override) |
| Improve C (block-only) default-path regression | **Valid** (measured without the attention override) |

### Mandatory rule (added)

**Any attention-swap implementation must be validated with a control experiment that returns the
same math (SDPA inside the override) before speed or quality is judged.** Copy the output shaping of
the stock implementation exactly. Never publish a collapse verdict without that control.

---

## 10. Re-verification with a validated harness (2026-09-11) - SA3 degrades; work PAUSED

Section 9 invalidated the earlier collapse numbers (0.0556 / 0.0909) because that harness lacked the
`transpose(1,2)` step. The re-measurement below therefore starts with a **control experiment**, using the
same layout handling that was validated for the SA2 path.

### 10.1 Control experiment (harness validation - mandatory first step)

| Run | Attention inside the override | Result |
|---|---|---|
| A | none (stock attention, reference) | wall 33.46 s |
| C | SDPA (mathematically identical to stock) | per-step cos min **0.999809**, final **0.999809** => HARNESS CLEAN |
| B | SA3 (`sageattn3_blackwell`, `per_block_mean=True`) | per-step cos 1.000002 -> **0.765044** |

Because the control reaches ~1.000, the SA3 degradation below is a genuine numeric effect of SA3 and not a
harness artifact. (Contrast with the invalidated run, where the SDPA control also collapsed to 0.086.)

### 10.2 SA3 measurement (FP16 model only - no quantization, 12 steps, seed 42, 1024x1024, cfg 2.5)

| step | control (cos vs stock) | SA3 (cos vs stock) |
|---|---|---|
| 1 | 1.000002 | 1.000002 |
| 2 | 1.000003 | 0.999694 |
| 3 | 1.000002 | 0.998093 |
| 4 | 1.000000 | 0.994854 |
| 5 | 0.999997 | 0.989061 |
| 6 | 0.999992 | 0.979289 |
| 7 | 0.999984 | 0.963279 |
| 8 | 0.999971 | 0.938652 |
| 9 | 0.999954 | 0.903040 |
| 10 | 0.999925 | 0.856243 |
| 11 | 0.999886 | 0.806606 |
| 12 | 0.999809 | **0.765044** |

- SA3 coverage: **816/816 attention calls**, 0 fallbacks, 0 errors
- wall: stock 33.46 s | control 38.62 s | SA3 31.87 s (= **-4.7 %** vs stock; SA2 reaches -15.4 %)

### 10.3 Interpretation

- The degradation is a **monotonic accumulation across steps**, consistent with SA3's standalone per-call
  error (~1.8 %, cos 0.98192 vs SDPA) accumulating over 30 layers x 12 steps.
- SA2, measured with the same harness, does not degrade (FP16-only 0.9984; NVFP4 seed42 0.98576; 20-seed
  mean 0.97468). The difference is the per-call error: SA2 0.999258 vs SA3 0.98192.
- Section 9 stands as the record of the invalid measurement; this section supersedes it with a validated
  measurement. The collapse is real, but the magnitude differs (0.765 for FP16-only, not 0.0556).

### 10.4 Status: PAUSED (Owner order, 2026-09-11 01:27)

Improve D is **paused**. Options recorded - none selected:

1. Tune SA3 settings (`per_block_mean=False`, head-dim handling) and re-measure the per-call error
2. Per-step / per-layer attribution to judge a hybrid (SA3 on layers that tolerate it)
3. Finalise rejection on the evidence above

No repository file was changed for this test: it ran from a temporary script outside the repo
(`workspace/.openclaw/tmp/sa3_control.py`), and the SA3 core registration in `attention.py` remains removed.
