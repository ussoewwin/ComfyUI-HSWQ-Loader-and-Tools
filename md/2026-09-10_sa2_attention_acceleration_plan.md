# SA2 (SageAttention 2) Integration Plan — Attention Acceleration for HSWQ Quantized Linear

- Created: 2026-09-10 (v3: English rewrite per repository-document language policy; v2 added the Phase 0/1 measurements)
- Status: **Phase 0 / Phase 1 complete — quality and speed both pass the acceptance criteria (Phase 2 20-seed statistics in progress)**
- Target: coexisting acceleration of HSWQ NVFP4 / INT8 Linear with SA2 attention
- Sources: full read-through of `D:\USERFILES\fp8e4m3\SageAttention\sageattention\`, probing of the installed package, and measurements on the real Z-Image model
- Predecessor: `2026-09-09_sa3_nvfp4_hswq_acceleration_plan.md` (SA3: improvements A/C/D all rejected, B not started)

## Revision history

| Version | Content |
|---|---|
| v1 | Initial plan (SA2 source read-through; only part of Phase 0 measured) |
| v2 | Phase 0 complete (per-call error on the real model), Phase 1 complete (12-step trajectory + speed), added the mandatory layout requirement for the harness |
| v3 | Rewritten in English (repository-document language policy) and updated with the Phase 2 20-seed results measured so far |

---

## 0. Background — why SA3 died and where SA2 stands

### 0.1 Measured results of the previous plan (SA3)

| Improvement | Result | Cause |
|---|---|---|
| A (mean shift on Linear) | **Rejected** | On layers with an extreme DC component the injected delta_y shifted the output DC by 42–64% (seed42: 0.615 vs 0.984 with the feature off) |
| C (block-only) | **Implementation failed** | Broke the default path (existing 0.974 -> ~0.924) -> fully reverted |
| D (SA3 attention) | **Rejected** | SA3 FP4 attention error accumulated across layers (NVFP4+SA3 = 0.0556, INT8+SA3 = 0.0909) |
| B (rotate+quantize fusion / butterfly) | **Not started** | Independent and still valid; can be combined with this plan |

**Phase 0 profile of the current NVFP4 path** (1024x1024, 6 steps, RTX 5060 Ti): softmax/attention 26.4% / rotate 16.9% / cuBLAS FP4 GEMM 19.6% / quantize 1.8%. **Attention is the largest single bottleneck**, so SA2 has the largest potential payoff.

### 0.2 Structural differences between SA3 and SA2

| Item | SA3 (FP4 attention) | SA2 (INT8/FP8 attention) |
|---|---|---|
| Q/K quantization | FP4 e2m1 (1-bit mantissa = 8 values) | **INT8** (256 values; per-warp / per-block / per-thread) |
| V quantization | FP4 e2m1 | **FP8 E4M3** (3-bit mantissa) |
| P (softmax output) | FP4 e2m1 | FP16 -> FP8 E4M3 (kept accurate via `S_FP8_OFFSET` = 8.807) |
| GQA / MQA | Not supported | **Supported** (expands via `nqheads // nkheads`) |
| head_dim | 64/128 only | **64/128/256** (padding; > 256 raises ValueError) |
| Blackwell | sm120/121 only | **sm100/120/121** (runs the SM89 kernel family) |

### 0.3 Measured results (all values below are measurements, not estimates)

**Z-Image (moodyRealMix_xhsEdition) / 1024x1024 / euler + simple / cfg 2.5 / RTX 5060 Ti 16GB**

| # | Measurement | Result |
|---|---|---|
| 1 | SA2 standalone (random q/k/v, seq 4096, head_dim 128, fp16) | cos vs SDPA **0.999258** (SA3: 0.98192) |
| 2 | **Per-call error on the real model** (NVFP4 model, 34 calls, bf16) | cos **0.99970–0.99999** / max-abs relative error **1.1–5.5%** |
| 3 | **FP16 model, 12 steps** (stock vs SA2, no quantization) | final x0 cos **0.998427** (min per-step 0.999046, no bifurcation) |
| 4 | **NVFP4 model, 12 steps** (stock vs SA2, same process) | final x0 cos **0.985622** (per-step 1.000 -> 0.988, smooth) |
| 5 | **NVFP4 + SA2 vs FP16 reference** (12 steps, seed 42, TC W4A4) | final-cos **0.98576** (NVFP4 alone = 0.98700 -> delta **-0.0012**, same-image) |
| 6 | **Speed, NVFP4 model, 12 steps** (same process) | stock **13.68 s** -> SA2 **11.57 s** = **-15.4%** (2.11 s saved, 0.176 s/step) |
| 7 | Speed, FP16 model, 12 steps | 34.13 s -> 29.34 s = **-14.0%** |
| 8 | Attention-only speed (1 forward = 34 calls) | SDPA 189.7 ms -> SA2 82.7 ms = **2.29x** |
| 9 | SA2 coverage (12 steps total) | **816 / 816 calls (100%)**, 0 fallbacks, 0 errors |
| 10 | Effect on the quantized GEMM | **GEMM MODE stays TC (W4A4)**, dequant_fallbacks = 0 |
| 11 | **Phase 2, 20 canonical seeds, NVFP4 + SA2** | mean **0.97468**, sd 0.01771, 95%CI +/-0.00776, min 0.93704, max 0.99241; >=0.98: 10/20, >=0.95: 17/20, <0.90: 0/20; same-image 10/20; SA2 coverage 16320/16320, 0 errors |
| 12 | Phase 2, 20 canonical seeds, default path (sdpa) | **in progress** (paired comparison pending) |

**Conclusion**: SA2 accelerates attention by 2.29x and the whole 12-step run by **-15.4%**, while the quality delta is only **cos -0.0012** (within the +/-0.005 local-vs-cloud environment spread) and the trajectory does not bifurcate. **Adoptable.**

### 0.4 Mandatory harness requirement and the control-experiment lesson (important)

The v1 harness (reused from the SA3 template) had a **layout conversion bug** that collapsed the 12-step trajectory (final-cos 0.086, bifurcation at step 10). It nearly got misread as an SA2 quality failure.

| Fact | Detail |
|---|---|
| Stock (`attention_pytorch`) I/O | input `[B,H,N,D]` (skip_reshape=True) -> output **`transpose(1,2)` then `[B,N,H*D]`** |
| Buggy harness | reshaped `[B,H,N,D]` **without transposing** -> head/token order scrambled |
| How it was found | **Control experiment**: call `F.sdpa` inside the override (mathematically equivalent to stock) -> **also collapsed to 0.085**, proving the harness, not SA2, was at fault |
| After the fix | the same control gives **per-step cos ~ 1.000 (final 0.999968) = HARNESS CLEAN** |

**Lessons (mandatory procedure from now on)**
1. After implementing any attention swap, **first validate the harness itself with a control experiment that returns the same math** before judging speed or quality.
2. For layout conversions involving `transpose`, **copy the output shaping of the stock implementation (`attention_pytorch`) exactly**.

---

## 1. SA2 source read-through (verified facts)

### 1.1 Per-architecture kernel map (`core.py`)

| Arch | Q/K | V | PV accumulator | Kernel |
|---|---|---|---|---|
| SM75 | INT8 | FP16 | — | `sageattn_qk_int8_pv_fp16_triton` |
| SM80/86/87 | INT8 | FP16 | fp32 | `sageattn_qk_int8_pv_fp16_cuda` |
| SM89 (Ada) | INT8 | FP8 E4M3 | fp32 / fp32+fp32 / **fp32+fp16** | `sageattn_qk_int8_pv_fp8_cuda` |
| SM90 (Hopper) | INT8 | FP8 E4M3 | fp32+fp32 | `sageattn_qk_int8_pv_fp8_cuda_sm90` |
| **SM100/120/121 (Blackwell)** | INT8 | FP8 E4M3 | fp32 / **fp32+fp16** | `sageattn_qk_int8_pv_fp8_cuda` + `qk_quant_gran="per_warp"` |

### 1.2 Blackwell dispatch (`core.py` L171-178, verified by execution)

```python
elif arch in {"sm100", "sm120", "sm121"}:
    if get_cuda_version() < (12, 8):
        pv_accum_dtype = "fp32"        # safe mode
    else:
        pv_accum_dtype = "fp32+fp16"   # SA2++ (CUDA 13.2 here -> this branch)
    return sageattn_qk_int8_pv_fp8_cuda(..., qk_quant_gran="per_warp", pv_accum_dtype=pv_accum_dtype)
```

Measured: `sageattention.core._cuda_archs = ['sm120']`, `torch.version.cuda = 13.2` -> the **SA2++ path (fp32+fp16, per_warp) is selected**.

### 1.3 Installed environment (measured)

| Item | Value |
|---|---|
| Package | `sageattention-2.2.0.post6+cu132torch2.14.0` |
| Compiled modules | `_fused`, `_qattn_sm80`, `_qattn_sm89` (sm89 includes sm100/120/121) |
| SM89/SM90 flags | `SM89_ENABLED=True`, `SM90_ENABLED=True` |
| CUDA / arch | 13.2 / sm120 (RTX 5060 Ti 16GB) |
| fp16 per_warp on sm120 | **kernel not provided** (`no kernel image is available`) -> only the fp8 path is usable |

### 1.4 smooth_k (K mean subtraction, contained inside attention)

- `sageattn_qk_int8_pv_fp8_cuda(..., smooth_k=True)` is the **default** (verified).
- Subtracts the sequence-wise mean of K before INT8 quantization, shrinking the dynamic range.
- For GQA it expands with `repeat_interleave` to the Q head count; `quant_per_block_int8_fuse_sub_mean_cuda` fuses the subtract and the quantize.
- **Same concept as SA3 improvement A, but confined to attention (never applied to the Linear pre-processing)**, so it structurally avoids the "extreme DC layer" failure mode.
- Measured: `smooth_k=True` vs `False` gave essentially identical per-call cos (0.999993 vs 0.999993) on this model.

### 1.5 FP8 V quantization and the softmax offset trick

- V: `per_channel_fp8` (per-channel amax along head_dim; `scale_max=448`, or `2.25` on SA2++).
- P: `S_FP8_OFFSET = 8.807` (in log2 space maps P's maximum of 1.0 onto E4M3's 448; cancelled automatically by the final normalization).

### 1.6 head_dim padding rules (`core.py` L75-89, verified)

| Native head_dim | Padded to |
|---|---|
| < 64 | 64 |
| 65–127 | 128 |
| 129–255 | 256 |
| > 256 | **ValueError** (SDPA fallback required) |

### 1.7 Real-model call conditions (measured)

| Item | Value |
|---|---|
| Call site | `comfy.ldm.lumina.model.JointAttention` -> `optimized_attention_masked(..., skip_reshape=True, transformer_options=...)` |
| Active backend | `attention_pytorch` (sage/flash/xformers disabled, pytorch enabled) |
| mask | **None for all 34 calls** (16320 calls over 20 seeds: all None) |
| Shapes / dtype | 30 calls `(1,30,4128,128)`, 2 calls `(1,30,4096,128)`, 2 calls `(1,30,32,128)` / fp16 (FP16 model), bf16 (NVFP4 model) |
| q/k/v amax (typical) | 5.4–10.9 / 6.6–10.9 / 68–478 |

### 1.8 Build assets

- Prebuilt wheel: `sageattention-2.2.0+cu132torch2.12.0` (cp312/cp313)
- MSVC SAL workaround: `#undef __in/__out/__inout` at the top of `fused.cu` (same approach as SA3)
- CUDA requirements: SM89 >= 12.4 / SM90 >= 12.3 / **SM120 >= 12.8**

---

## 2. Verification plan

### Phase 0: SA2 standalone accuracy and dispatch — DONE

- [x] Package check: `sageattention 2.2.0.post6+cu132torch2.14.0`, `_qattn_sm89` present
- [x] Dispatch check: `sm120` -> `sageattn_qk_int8_pv_fp8_cuda(per_warp, fp32+fp16)`
- [x] Standalone accuracy: random q/k/v -> cos 0.999258 (seq 4096) / 0.999254 (4128)
- [x] Per-call error on the real model: cos 0.99970–0.99999, max-abs relative error 1.1–5.5%
- [ ] Extra conditions (bf16 standalone / head_dim 64 and 256 / `pv_accum_dtype="fp32+fp32"` comparison) — not run; not required for the adoption decision

**Acceptance**: cos >= 0.999 -> **pass**

### Phase 1: real-model verification — quantized Linear + SA2 attention — DONE

- [x] **NVFP4 Linear (nv100) + SA2**: 12 steps x seed42, TC(W4A4) -> final-cos **0.98576** (baseline 0.98700, delta -0.0012, same-image)
- [x] **FP16 + SA2** (no quantization): 12 steps -> final x0 cos **0.998427** (SA2's own trajectory impact is 0.16%)
- [x] **Speed**: NVFP4 12 steps 13.68 s -> 11.57 s (**-15.4%**); FP16 12 steps (**-14.0%**)
- [x] Coverage: 816/816 calls (100%), 0 fallbacks, GEMM MODE stays TC (W4A4)
- [ ] INT8 Linear + SA2 (`zi_int8_bench.py --attention sage2`) — not run (INT8 follow-up)

**Acceptance**: final-cos >= 0.96 (degradation <= 0.015 from NVFP4 alone) -> **pass (actual degradation 0.0012)**

### Phase 2: 20-seed statistics — IN PROGRESS

- [x] **NVFP4 + SA2**, 20 canonical 10-digit seeds x 12 steps: mean **0.97468**, sd 0.01771, 95%CI +/-0.00776, min 0.93704, max 0.99241, >=0.98: 10/20, same-image 10/20, 0 bifurcations
- [ ] Default path (sdpa) with the same 20 seeds — running (paired comparison pending)
- [ ] INT8 + SA2 follow-up
- [ ] Consolidate SDPA fallback conditions (head_dim > 256 / mask present / import failure)

**Exit criteria**: 20-seed summary + final benchmark table

---

## 3. Implementation design

**Design rules (established from the SA3 failures and the Phase 1 harness bug)**
1. **Never change the behavior of the default path (no env var, no flag).**
2. After any change, re-measure the existing default-path score and confirm non-regression before evaluating the new feature.
3. Additive changes only (never rewrite existing code); enable explicitly through a dedicated flag.
4. **Validate the harness itself first with a control experiment that returns the same math** (never confuse an SA2 quality issue with a harness bug).

### 3.1 Integration (implemented)

Added `--attention {sdpa,sage2}` to `benchmark/zi_convrot_nvfp4_traj_compare.py`. **Default remains sdpa (existing behavior unchanged).** Implemented as `apply_sage2_attention()` / `print_sage2_attn_stats()`, kept **fully separate from any SA3 code path**. The FP16 baseline always keeps stock attention; only the quantized model is swapped.

Phase 2 support added as well: `--canonical-seeds` (the 20 canonical 10-digit seeds) plus median / sd / 95%CI and threshold counts in the summary.

### 3.2 Layout conversion (mandatory specification, confirmed by measurement)

```python
# input normalization (same semantics as attention_pytorch)
if skip_reshape:                      # q,k,v = [B,H,N,D]
    b, _, _, dim_head = q.shape
else:                                 # q,k,v = [B,N,H,D] -> transpose
    b, n, _ = q.shape; dim_head = q.shape[-1] // heads
    qh = q.view(b, n, heads, dim_head).transpose(1, 2)   # same for k, v

out = sageattn(qh, kh, vh, tensor_layout="HND", is_causal=False)   # [B,H,N,D]

# output shaping (getting this wrong collapses the trajectory)
if skip_output_reshape:
    return out                                   # [B,H,N,D]
return out.transpose(1, 2).reshape(b, -1, heads * dim_head)   # [B,N,H*D]
```

### 3.3 SDPA fallback conditions (implemented)

| Condition | Behavior |
|---|---|
| `mask is not None` | SDPA (never triggered on this model: all 16320 calls had mask None) |
| `head_dim > 256` | SDPA (SA2 raises ValueError) |
| `import sageattention` fails | SDPA |
| Any runtime exception | SDPA (reason logged) |

---

## 4. Verification protocol (inherited from the SA3 plan plus new lessons)

| Item | Procedure |
|---|---|
| Accuracy | 20 seeds x 12 steps, TC(W4A4) cosine. **Print every row, the summary and GEMM MODE in full** |
| Speed | Median of 3 runs with identical seed/settings; report the ratio against the FP16 baseline |
| Parity | Existing `nvfp4_comfy_parity` flow, with SA2 on and off |
| **Non-regression** | **Re-measure the default path (sdpa) after the change and confirm non-regression** (done: 0.98700, identical to the previous value) |
| **Harness validation** | **Run the SDPA-inside-the-override control first** (confirm per-step cos ~ 1) |
| Seeds | Use the canonical 10-digit seeds |
| Environment spread | Local (5060 Ti) vs cloud varies by mean +/-0.005; values in the 0.97 range count as equivalent |
| Target collation | Verify the artifact (genuine) and the seeds (canonical) before measuring |
| On failure | Delete artifacts from failed runs immediately, after recording a one-line failure cause |

---

## 5. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **Layout conversion bug** (actually happened in v1) | A collapsed trajectory gets misread as an SA2 quality failure | Validate with the SDPA-inside-the-override control (per-step cos ~ 1) before judging |
| Accuracy decay from layer accumulation | The same failure mode as SA3 | Measured: FP16+SA2 0.9984, NVFP4+SA2 0.98576 (no bifurcation) |
| Kernel malfunction on Blackwell | Unknown bugs | Measured: 16320/16320 calls fine, 0 errors |
| No fp16 per_warp kernel on sm120 | The fp16 path is unusable | Use the fp8 (SA2++) path; measured to be fine |
| 20-seed mean falls short of the bar | Cannot adopt | Phase 2 resolves it (single-run delta is -0.0012, so unlikely) |
| INT8 Linear + SA2 not yet measured | INT8 path status unknown | Follow-up in Phase 2 |
| Breaking the default path | Score regression (happened with SA3) | `--attention` flag, default sdpa, re-measure existing scores after the change (done) |
| Speedup too small | Adoption not worthwhile | Measured -15.4% on NVFP4 12 steps; bar was "reject if <= 10%" |
| SA3 remnants lingering | Confusion about which attention path is live | Removed all SA3 code and the `register_attention_function("sage3", ...)` registration from every ComfyUI `attention.py` copy |

---

## 6. File map

### Read (reference implementations)
- `D:\USERFILES\fp8e4m3\SageAttention\sageattention\core.py` — dispatch and attention implementation (L171-178 is the Blackwell branch)
- `D:\USERFILES\fp8e4m3\SageAttention\sageattention\quant.py` — INT8/FP8 quantization and fused ops
- `D:\USERFILES\fp8e4m3\SageAttention\sageattention\sm89_compile.py` — Blackwell bindings
- `D:\USERFILES\fp8e4m3\SageAttention\csrc\qattn\` — SM89/Blackwell kernels
- `D:\USERFILES\GitHub\hswq\ComfyUI-master\comfy\ldm\modules\attention.py` — `attention_pytorch` (reference for output shaping)
- `D:\USERFILES\GitHub\hswq\ComfyUI-master\comfy\ldm\lumina\model.py` — Z-Image attention call site

### Changed (additive only; implemented)
- `benchmark/zi_convrot_nvfp4_traj_compare.py` — `--attention {sdpa,sage2}`, `apply_sage2_attention()`, `print_sage2_attn_stats()`, `--canonical-seeds`, 20-seed statistics
- `ComfyUI-master/comfy/ldm/modules/attention.py` — removed SA3 remnants (SA2 registration kept)

### Not started (add later if needed)
- `benchmark/zi_int8_bench.py` — add `--attention sage2` for the INT8 path (Phase 2 follow-up)

### Untouched
- HSWQ Linear forward (NVFP4 / INT8 paths) — never modified
- Existing parity / bench default behavior
- Other `ComfyUI-master` files (monkey-patching stays inside the benchmark script)

---

## 7. Relationship to the SA3 plan

This plan is the **successor to improvement D (SA3 attention)** of the SA3 plan. The SA3 measurements are recorded in section 0.1 above (the SA3 plan document itself has been retired, since SA3 was rejected; git history retains it).

| SA3 plan improvement | Status | Relation to this plan |
|---|---|---|
| A (mean shift) | Rejected | SA2 provides the same function as `smooth_k` inside attention (never applied to Linear) |
| B (fusion kernel / butterfly) | Not started | **Independent and still valid**; combinable with SA2 (Layer-side speedup) |
| C (block-only) | Implementation failed | Unrelated to SA2 |
| D (SA3 attention) | Rejected | **Replaced by SA2 attention in this plan. Phase 1/2 pass the acceptance criteria** |
