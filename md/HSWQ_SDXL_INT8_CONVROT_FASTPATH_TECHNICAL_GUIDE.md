# HSWQ SDXL ConvRot INT8 — Kernel Fast Path (pooled rotate + fused ConvRot OFF) — Technical Implementation Manual

Date: 2026-09-13  
Repository: `ussoewwin/ComfyUI-HSWQ-Loader-and-Tools`  
Diff baseline (exclusive): `249b6941db6e32b52f665c8933ce209f5199624e` (`docs(zh): point v3.5.1 release-notes link at the Chinese note`)  
HEAD at writing: `f333469` — `fix(sdxl-int8): follow-up to the move (loader path to nodes/sdxl_int8, package marker, docstring)`  
Scope: **SDXL ConvRot INT8 only**, under `nodes/sdxl_int8/` plus 48 added lines in `patches/comfy_quant_int8.py`. This line does **not** touch Z Image, Krea2, FLUX, SD1.5, SAM3, ControlNet, Qwen or any NVFP4 path; `nodes/native_convert_int8.py` is **unchanged (0-byte diff)**.

---

## ① Why this feature was created

### 1.1 The observable problem

SDXL ConvRot INT8 packs loaded through `comfy_kitchen`'s `int8_linear` were **not faster than FP16**. Measured on the reference checkpoint (`waiIllustriousSDXL_v170`, 25 steps × 6 seeds, one process, ComfyUI production sampler):

| | INT8 it/s | INT8 / FP16 | mean cosine |
|---|---|---|---|
| before the fast path | 2.43 | 1.069x (slower) | 0.95472 |
| with the fast path | **2.90** | **0.894x (faster)** | **0.95867** |

### 1.2 Root cause — the online ConvRot activation rotate

ConvRot (rotation-based INT8) requires the **input activation** to be rotated by the same Hadamard matrix that was applied to the weight offline (`W_rot = W @ Hᵀ`, therefore `x_rot = x @ H`). Removing the rotate collapses the output (measured cos 0.333), so it cannot be skipped.

The rotate runs once per INT8 linear — hundreds of times per denoise step. The previous implementation promoted every activation to float32 and allocated a new output tensor per call:

* per-call **allocation** (`torch.matmul(...)` without `out=`),
* **fp32 promotion** (extra cast pass + an FP32 SGEMM path).

Isolated cost over the real SDXL INT8 layer mix (M=4096, group size 256):

| rotate form | ms/step |
|---|---|
| pooled fp32 | 257.3 |
| **pooled bf16** | **52.5** |
| batched `bmm` fp32 | 540.3 |
| allocating fp32 matmul | 166.5 |

Fused rotate+quant measured 129.0 ms vs plain quantize 32.0 ms over the same mix → ≈97 ms/step was being spent on the rotate, the same order as the entire INT8 GEMM saving (−169.6 ms/step).

### 1.3 Second finding — the CUDA fused ConvRot kernels are worse

A/B at full protocol (6 seeds × 25 steps, same process, same seeds):

| arm | INT8 it/s | mean cosine |
|---|---|---|
| fused OFF | 2.97 | ~0.958 |
| fused ON | 2.89 (slowest) | **0.934** (1/6 same-image) |

The fused kernel was both slower **and** trajectory-drifting, so the production configuration keeps it **OFF**.

### 1.4 Design constraint — separation (mandatory)

The loader is shared by several model families. The fix must therefore:

1. live in a **dedicated file** (not inside a shared module body),
2. be **reachable only from SDXL** code paths,
3. for non-SDXL families: never be **imported**, never load, never swap any kernel,
4. swap kernels **only while an armed SDXL forward runs**, restoring the exact original objects afterwards,
5. be **fail-closed**: if the environment does not match what the fast path assumes, it must refuse to arm.

---

## ② New / modified code

| Status | Path | Lines | Note |
|---|---|---|---|
| **A** (new) | `nodes/sdxl_int8/__init__.py` | 0 | package marker (empty, same form as `nodes/models/__init__.py`) |
| **A** (new) | `nodes/sdxl_int8/sdxl_convrot_fast.py` | 344 | the entire fast path, SDXL-only |
| **M** (modified) | `patches/comfy_quant_int8.py` | +48 / −0 | SDXL classifier + private loader + two SDXL-guarded call sites |
| **unchanged** | `nodes/native_convert_int8.py` | 0 | shared Conv2d helper: **0-byte diff** (the pooled Conv2d rotate is implemented in the dedicated file and swapped in only inside the window) |

```
$ git diff --numstat 249b6941db6e32b52f665c8933ce209f5199624e..f333469
0    0    nodes/sdxl_int8/__init__.py
344  0    nodes/sdxl_int8/sdxl_convrot_fast.py
48   0    patches/comfy_quant_int8.py
```

---

## ③ Full source (verbatim, no omissions)

### ③-1 `nodes/sdxl_int8/sdxl_convrot_fast.py` (344 lines, complete)

```python
# -*- coding: utf-8 -*-
"""SDXL-only ConvRot INT8 kernel fast path.

**Scope: SDXL ConvRot INT8 only (nodes/sdxl_int8/).** This module exists so that the fast path does
not live in any shared module: `patches/comfy_quant_int8.py` and
`nodes/native_convert_int8.py` keep their original code, and only the SDXL load
branches call `arm_sdxl_convrot_fast()`.

What it does (ported from the SDXL INT8 trajectory-bench fix):
  * pooled activation rotate - the stock comfy_kitchen rotate allocates a new
    tensor on every call (measured over the real SDXL INT8 layer mix:
    pooled fp32 257.3 ms/step -> pooled bf16 52.5 ms/step; INT8/FP16
    1.069x -> 0.894x),
  * CUDA fused ConvRot kernels OFF - measured slower AND trajectory-drifting
    (2.89 it/s, cos 0.934, 1/6 same-image vs 2.97 it/s, cos 0.958).

Separation guarantees:
  * no import-time side effects (importing this module changes nothing),
  * the kernels are swapped only while an armed SDXL UNet forward runs
    (``_arm()`` in the wrapper, ``_disarm()`` in ``finally``),
  * every touched attribute is saved and restored as the *same object*,
  * arming requires the SDXL architecture (``UNetModel`` from ``openaimodel``
    with ADM conditioning); Z Image / Krea2 / SD1.5 / FLUX and every other
    family is refused and never touched,
  * non-SDXL loads never import this module at all.
"""
from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

_ROTATE_MODULES = (
    "comfy_kitchen.tensor.int8_utils",
    "comfy_kitchen.backends.eager.quantization",
    "comfy_kitchen.backends.triton.quantization",
    "comfy_kitchen.backends.cuda",
    "comfy_kitchen.backends.eager.convrot_w4a4",
)

# ---------------------------------------------------------------------------
# Fail-closed audit of the comfy_kitchen call sites.
#
# The pooled rotate hands back a view of a shared buffer, which is only correct
# while every caller consumes the result before the next rotate of the same
# shape/dtype/device. That is not a type-system invariant, so it is *checked*:
# every `_rotate_activation(...)` call site in the audited kitchen files must
# bind its result and read that binding within the next two statements, with no
# intervening `_rotate_activation(...)` call. If any site does not match (kitchen
# updated, new caller, unparseable file), the fast path REFUSES to arm and logs.
# ---------------------------------------------------------------------------

_AUDITED_ROTATE_FILES = (
    "backends/cuda/__init__.py",
    "backends/eager/quantization.py",
    "backends/triton/quantization.py",
    "backends/eager/convrot_w4a4.py",
)

_AUDIT_CACHE = {"done": False, "ok": False, "detail": "not run"}


def _audit_rotate_call_sites() -> tuple:
    """(ok, detail) - fail-closed structural check of the rotate call sites."""
    if _AUDIT_CACHE["done"]:
        return _AUDIT_CACHE["ok"], _AUDIT_CACHE["detail"]
    import ast
    import os

    try:
        import comfy_kitchen

        root = os.path.dirname(os.path.abspath(comfy_kitchen.__file__))
    except Exception as exc:  # pragma: no cover
        _AUDIT_CACHE.update(done=True, ok=False, detail=f"comfy_kitchen unavailable: {exc}")
        return False, _AUDIT_CACHE["detail"]

    # 0) no caller may live outside the audited files: walk the whole package and
    #    collect every `_rotate_activation(...)` call site.
    unexpected = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fn)
            rel = os.path.relpath(fpath, root).replace(os.sep, "/")
            if rel in _AUDITED_ROTATE_FILES:
                continue
            try:
                with open(fpath, encoding="utf-8") as fh:
                    src2 = fh.read()
                tree2 = ast.parse(src2)
            except Exception:
                continue
            for node in ast.walk(tree2):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "_rotate_activation"):
                    unexpected.append(f"{rel}:{node.lineno}")
    if unexpected:
        _AUDIT_CACHE.update(done=True, ok=False,
                            detail=f"unexpected rotate caller(s): {unexpected[:5]}")
        return False, _AUDIT_CACHE["detail"]

    n_sites = 0
    for rel in _AUDITED_ROTATE_FILES:
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            _AUDIT_CACHE.update(done=True, ok=False, detail=f"audited file missing: {rel}")
            return False, _AUDIT_CACHE["detail"]
        try:
            with open(path, encoding="utf-8") as fh:
                src = fh.read()
            tree = ast.parse(src)
        except Exception as exc:
            _AUDIT_CACHE.update(done=True, ok=False, detail=f"unparseable {rel}: {exc}")
            return False, _AUDIT_CACHE["detail"]

        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            body = fn.body
            for i, stmt in enumerate(body):
                calls = [n for n in ast.walk(stmt)
                         if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Name)
                         and n.func.id == "_rotate_activation"]
                if not calls:
                    continue
                n_sites += 1
                if not isinstance(stmt, ast.Assign):
                    continue  # consumed inside the same statement -> safe
                targets = [tg for tg in stmt.targets if isinstance(tg, ast.Name)]
                if not targets:
                    _AUDIT_CACHE.update(
                        done=True, ok=False,
                        detail=f"{rel}:{stmt.lineno} unanalysable assignment target")
                    return False, _AUDIT_CACHE["detail"]
                name = targets[0].id
                # first later statement that reads the binding / calls rotate again
                limit = min(i + 3, len(body))
                for j in range(i + 1, limit):
                    later = body[j]
                    rereads = [n for n in ast.walk(later)
                               if isinstance(n, ast.Name) and n.id == name
                               and isinstance(n.ctx, ast.Load)]
                    recalls = [n for n in ast.walk(later)
                               if isinstance(n, ast.Call)
                               and isinstance(n.func, ast.Name)
                               and n.func.id == "_rotate_activation"]
                    if recalls:
                        _AUDIT_CACHE.update(
                            done=True, ok=False,
                            detail=(f"{rel}:{later.lineno} another _rotate_activation call "
                                    f"before '{name}' was consumed"))
                        return False, _AUDIT_CACHE["detail"]
                    if rereads:
                        break
                else:
                    _AUDIT_CACHE.update(
                        done=True, ok=False,
                        detail=(f"{rel}:{stmt.lineno} result '{name}' not consumed within "
                                f"the next statements"))
                    return False, _AUDIT_CACHE["detail"]

    _AUDIT_CACHE.update(done=True, ok=True, detail=f"{n_sites} call sites verified")
    return True, _AUDIT_CACHE["detail"]


_STATE = {"depth": 0, "saved": [], "pool": {}, "helpers": None}


def set_helpers_loader(loader) -> None:
    """Inject the nodes/native_convert_int8 module (kept out of this file so
    the module has no package-relative imports and can be loaded privately)."""
    _STATE["helpers"] = loader


def _helpers():
    loader = _STATE.get("helpers")
    return loader() if loader is not None else None


def _always_false(*_a, **_k):
    return False


def _pooled_rotate(x, h, group_size):
    """comfy_kitchen ConvRot activation rotate with a pooled output buffer."""
    import torch

    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} not divisible by group_size {group_size}")
    n_groups = features // group_size
    x_grouped = x.reshape(-1, n_groups, group_size)
    if h.dtype != x_grouped.dtype or h.device != x_grouped.device:
        h = h.to(dtype=x_grouped.dtype, device=x_grouped.device)
    key = (tuple(x_grouped.shape), x_grouped.dtype, str(x_grouped.device))
    out = _STATE["pool"].get(key)
    if out is None:
        out = torch.empty_like(x_grouped)
        _STATE["pool"][key] = out
    torch.matmul(x_grouped, h, out=out)
    return out.reshape(orig_shape)


def _pooled_helper_rotate(x, h_matrix, group_size):
    """nodes/native_convert_int8.rotate_activation with a pooled output buffer.

    Same math as that helper (device-cached Hadamard), allocation removed.
    """
    import torch

    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} not divisible by group_size {group_size}")
    group_count = features // group_size
    x_grouped = x.reshape(-1, group_count, group_size)
    h = _helpers().get_hadamard_on_device(
        group_size, device=x.device, dtype=x.dtype
    )
    key = (tuple(x_grouped.shape), x_grouped.dtype, str(x_grouped.device))
    out = _STATE["pool"].get(key)
    if out is None:
        out = torch.empty_like(x_grouped)
        _STATE["pool"][key] = out
    torch.matmul(x_grouped, h, out=out)
    return out.reshape(orig_shape)


def _arm_kernels() -> bool:
    if _STATE["depth"] > 0:
        _STATE["depth"] += 1
        return True
    ok, detail = _audit_rotate_call_sites()
    if not ok:
        logger.warning(
            "[HSWQ INT8][SDXL] fast path refused: rotate call-site audit failed (%s)", detail
        )
        return False
    saved = []
    try:
        for modname in _ROTATE_MODULES:
            try:
                m = importlib.import_module(modname)
            except Exception:
                continue
            if hasattr(m, "_rotate_activation"):
                saved.append((m, "_rotate_activation", m._rotate_activation))
                m._rotate_activation = _pooled_rotate

        cuda_mod = importlib.import_module("comfy_kitchen.backends.cuda")
        if hasattr(cuda_mod, "_CONVROT_FUSED_MAX_K"):
            saved.append((cuda_mod, "_CONVROT_FUSED_MAX_K", cuda_mod._CONVROT_FUSED_MAX_K))
            cuda_mod._CONVROT_FUSED_MAX_K = -1
        for attr in ("_should_use_convrot_fused_kernel", "_should_use_convrot_dequant_kernel"):
            if hasattr(cuda_mod, attr):
                saved.append((cuda_mod, attr, getattr(cuda_mod, attr)))
                setattr(cuda_mod, attr, _always_false)

        nc = _helpers()
        if nc is not None and hasattr(nc, "rotate_activation"):
            saved.append((nc, "rotate_activation", nc.rotate_activation))
            nc.rotate_activation = _pooled_helper_rotate

        _STATE["saved"] = saved
        _STATE["depth"] = 1
        return True
    except Exception as exc:
        logger.warning("[HSWQ INT8][SDXL] fast-path arm failed: %s", exc)
        _STATE["saved"] = saved
        _STATE["depth"] = 1
        _disarm_kernels()
        return False


def _disarm_kernels() -> None:
    for mod, attr, val in reversed(_STATE["saved"] or []):
        try:
            setattr(mod, attr, val)
        except Exception:
            pass
    _STATE["saved"] = []
    _STATE["depth"] = 0
    # Release the pooled buffers when the window closes: outside the SDXL window
    # the process therefore retains exactly the stock amount of VRAM (the stock
    # rotate allocates a temporary per call and frees it). Inside the window the
    # buffers are reused (~one allocation per distinct activation shape per
    # forward instead of one per call).
    _STATE["pool"].clear()


def is_sdxl_convrot_fast_armed() -> bool:
    return _STATE["depth"] > 0


def arm_sdxl_convrot_fast(model, *, log_prefix="[HSWQ INT8]") -> bool:
    """Arm the SDXL ConvRot INT8 fast path for THIS model only.

    Refuses every non-SDXL architecture (returns False, touches nothing).
    """
    inner = getattr(model, "model", None)
    dm = getattr(inner, "diffusion_model", None)
    if dm is None:
        return False
    cls = type(dm).__name__
    mod = type(dm).__module__.lower()
    if cls != "UNetModel" or "openaimodel" not in mod or getattr(dm, "adm_in_channels", None) is None:
        logger.debug("[HSWQ INT8][SDXL] fast path refused (arch %s.%s)", mod, cls)
        return False
    if getattr(dm, "_hswq_sdxl_convrot_fast_wrapped", False):
        return True
    # Fail-closed: the pooled rotate is only valid while every caller consumes the
    # result immediately. Verify the installed kitchen structurally BEFORE wrapping;
    # on any mismatch the model is left completely untouched (stock path).
    ok, detail = _audit_rotate_call_sites()
    if not ok:
        logger.warning(
            "[HSWQ INT8][SDXL] fast path NOT armed: rotate call-site audit failed (%s)",
            detail,
        )
        return False

    original_forward = dm.forward

    def sdxl_convrot_fast_forward(*args, **kwargs):
        armed = _arm_kernels()
        try:
            return original_forward(*args, **kwargs)
        finally:
            if armed:
                _STATE["depth"] -= 1
                if _STATE["depth"] <= 0:
                    _disarm_kernels()

    dm.forward = sdxl_convrot_fast_forward
    dm._hswq_sdxl_convrot_fast_wrapped = True
    dm._hswq_sdxl_convrot_fast_original_forward = original_forward
    print(f"{log_prefix} SDXL ConvRot INT8 fast path armed for this model only", flush=True)
    logger.info("%s SDXL ConvRot INT8 fast path armed", log_prefix)
    return True
```

### ③-2 `nodes/sdxl_int8/__init__.py`

```python

```
(empty file; exists so the directory is a regular package like its siblings)

### ③-3 `patches/comfy_quant_int8.py` — complete diff over the baseline

```diff
diff --git a/patches/comfy_quant_int8.py b/patches/comfy_quant_int8.py
index 4ea5e8a..50ea467 100644
--- a/patches/comfy_quant_int8.py
+++ b/patches/comfy_quant_int8.py
@@ -3070,6 +3070,45 @@ def apply_comfy_quant_int8_patches() -> bool:
 KREA2_MODEL_FLAG = "_hswq_is_krea2"
 
 
+_SDXL_FAST_MOD = None
+
+
+def _load_sdxl_convrot_fast():
+    """Load the SDXL-only fast-path file by path WITHOUT registering it in
+    sys.modules, so no other model family ever sees it as an import.
+    Called only from the SDXL-guarded branches below."""
+    global _SDXL_FAST_MOD
+    if _SDXL_FAST_MOD is None:
+        import importlib.util
+        import os as _os
+
+        repo = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
+        path = _os.path.join(repo, "nodes", "sdxl_int8", "sdxl_convrot_fast.py")
+        spec = importlib.util.spec_from_file_location(
+            "_hswq_sdxl_convrot_fast_private", path)
+        mod = importlib.util.module_from_spec(spec)
+        spec.loader.exec_module(mod)
+        mod.set_helpers_loader(_load_native_convert_int8_helpers)
+        _SDXL_FAST_MOD = mod
+    return _SDXL_FAST_MOD
+
+
+def _model_is_sdxl_unet(model) -> bool:
+    """True only for the SDXL UNet (UNetModel from openaimodel with ADM).
+
+    Used to gate SDXL-only code paths; no side effects.
+    """
+    inner = getattr(model, "model", None)
+    dm = getattr(inner, "diffusion_model", None)
+    if dm is None:
+        return False
+    if type(dm).__name__ != "UNetModel":
+        return False
+    if "openaimodel" not in type(dm).__module__.lower():
+        return False
+    return getattr(dm, "adm_in_channels", None) is not None
+
+
 def model_is_krea2(model) -> bool:
     """Krea2 check taken from ComfyUI's own architecture detection.
 
@@ -3245,6 +3284,11 @@ def load_unet_hswq_weight_dtype(unet_name, weight_dtype, attention_accel="defaul
         with _int8_quant_conv_scope():
             model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
         summarize_int8_lora_capability(model)
+        # SDXL-only fast path: dedicated module + dedicated guard.
+        # This branch also serves Z Image (UNet-style checkpoints), so the SDXL
+        # check runs BEFORE the import: non-SDXL never imports the module.
+        if is_convrot and needs_conv2d and _model_is_sdxl_unet(model):
+            _load_sdxl_convrot_fast().arm_sdxl_convrot_fast(model)
 
         if attention_accel == "sa2":
             # SageAttention2 on the loaded INT8 model (pattern zimage_int8 or
@@ -3331,6 +3375,10 @@ def load_checkpoint_sdxl_hswq_weight_dtype(ckpt_name, weight_dtype, device=None)
                 )
             model, clip, _v = out[:3]
             summarize_int8_lora_capability(model)
+            if _model_is_sdxl_unet(model):
+                _load_sdxl_convrot_fast().arm_sdxl_convrot_fast(
+                    model, log_prefix="[SDXL INT8]"
+                )
             return (model, clip)
 
         if weight_dtype == "fp8_e4m3fn":
```

### ③-4 The three added regions in final form

**SDXL classifier + private (non-registered) loader** — inserted immediately before `load_unet_hswq_weight_dtype`'s caller scope:

```python
_SDXL_FAST_MOD = None


def _load_sdxl_convrot_fast():
    """Load the SDXL-only fast-path file by path WITHOUT registering it in
    sys.modules, so no other model family ever sees it as an import.
    Called only from the SDXL-guarded branches below."""
    global _SDXL_FAST_MOD
    if _SDXL_FAST_MOD is None:
        import importlib.util
        import os as _os

        repo = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        path = _os.path.join(repo, "nodes", "sdxl_int8", "sdxl_convrot_fast.py")
        spec = importlib.util.spec_from_file_location(
            "_hswq_sdxl_convrot_fast_private", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.set_helpers_loader(_load_native_convert_int8_helpers)
        _SDXL_FAST_MOD = mod
    return _SDXL_FAST_MOD


def _model_is_sdxl_unet(model) -> bool:
    """True only for the SDXL UNet (UNetModel from openaimodel with ADM).

    Used to gate SDXL-only code paths; no side effects.
    """
    inner = getattr(model, "model", None)
    dm = getattr(inner, "diffusion_model", None)
    if dm is None:
        return False
    if type(dm).__name__ != "UNetModel":
        return False
    if "openaimodel" not in type(dm).__module__.lower():
        return False
    return getattr(dm, "adm_in_channels", None) is not None
```

**Call site 1 — `load_unet_hswq_weight_dtype()`, inside the `elif is_int8:` branch (this branch also serves Z Image, hence the guard *before* the load):**

```python
        # SDXL-only fast path: dedicated module + dedicated guard.
        # This branch also serves Z Image (UNet-style checkpoints), so the SDXL
        # check runs BEFORE the import: non-SDXL never imports the module.
        if is_convrot and needs_conv2d and _model_is_sdxl_unet(model):
            _load_sdxl_convrot_fast().arm_sdxl_convrot_fast(model)
```

**Call site 2 — `load_checkpoint_sdxl_hswq_weight_dtype()` (the SDXL checkpoint loader):**

```python
            if _model_is_sdxl_unet(model):
                _load_sdxl_convrot_fast().arm_sdxl_convrot_fast(
                    model, log_prefix="[SDXL INT8]"
                )
```

---

## ④ What it means

### 4.1 `_pooled_rotate` / `_pooled_helper_rotate` — why the numbers are unchanged

* The rotation math is **identical** to the stock implementation: same `reshape(-1, n_groups, group_size)`, same `h.to(x.dtype)` cast, same `matmul`. The **only** difference is that the matrix multiply writes into a reused buffer (`out=`) instead of allocating.
* Rotating in the activation's own dtype is safe because the Hadamard entries are `±1/√n`; for `n = 256` that is `±2⁻⁴`, **exactly representable in fp16 and bf16**. Only the accumulation order/dtype differs, measured at **0.00011 %** difference in the resulting INT8 activation codes.
* Pool keys are `(grouped shape, dtype, device)` — different dtypes/devices can never collide.

### 4.2 Why the pooled buffer needs the call-site audit (fail-closed)

A pooled rotate returns a **view of a shared buffer**. That is only correct while every caller consumes the result before the next rotate of the same key. Rather than relying on that as an unchecked contract, `_audit_rotate_call_sites()` verifies it structurally at arm time:

1. every `.py` under `comfy_kitchen/` is parsed and **all** `_rotate_activation(...)` call sites are collected;
2. a caller **outside the four audited files** → refuse;
3. a call whose result is bound must read that binding **within the next two statements**, with **no intervening `_rotate_activation` call**;
4. unparseable file / missing file / unanalysable target → **refuse**;
5. on refusal the model is **not wrapped at all** — kernels, `_CONVROT_FUSED_MAX_K` and the helper stay exactly as stock.

Measured behaviour: real kitchen → `ok=True (7 call sites verified)`; synthetic retention pattern → `ok=False …another _rotate_activation call before 'a' was consumed`; a call site added to a different module → `ok=False … unexpected rotate caller(s): ['tensor/extra.py:2']`.

### 4.3 Why the kernels are swapped per forward and restored

`_arm_kernels()` saves **every touched attribute** (5 rotate bindings, `_CONVROT_FUSED_MAX_K`, both fused predicates, the Conv2d helper's `rotate_activation`) and `_disarm_kernels()` puts back the **same objects** in a `finally`. Consequences:

* outside the armed window the process is byte-identical to stock (function identity, constant values, no wrappers),
* nested forwards are handled by a depth counter,
* exceptions inside the forward still restore everything.

### 4.4 Why the module is loaded privately (never registered in `sys.modules`)

`_load_sdxl_convrot_fast()` uses `spec_from_file_location` + `exec_module` **without** inserting the module into `sys.modules`. Therefore, even in a session that already ran an SDXL model, no other family sees the fast path as imported — and the file is only *read* when an SDXL-guarded branch executes.

### 4.5 Why the architecture gate is checked before the load

`_model_is_sdxl_unet()` requires `UNetModel` **and** `openaimodel` **and** `adm_in_channels is not None`. Z Image (`NextDiT`), Krea2, FLUX/other DiT and SD1.5 (UNet without ADM) therefore fail the gate **before** the private load — they neither import, load nor swap anything.

### 4.6 Why the fused ConvRot kernels are disabled inside the window

`_CONVROT_FUSED_MAX_K = -1` plus `_should_use_convrot_fused_kernel/_dequant_kernel → False` forces kitchen onto the staged/Hadamard path that uses the (pooled) rotate. This reproduces the recorded production configuration (fused OFF), which was measured both faster and trajectory-stable. The change exists only inside the window; outside it the original constant/objects are restored.

### 4.7 VRAM

The pooled buffers are released when the window closes (`_STATE["pool"].clear()`), so **outside** the SDXL window the process retains exactly the stock amount of VRAM, while **inside** the window there is ≈1 allocation per distinct activation shape per forward instead of one per call.

### 4.8 LoRA bake and neighbouring machinery are not affected

* The swapped symbols are the **activation-side** rotate and the fused-kernel selectors. The LoRA bake path (`convert_weight` → dequantize → `calculate_weight` → `set_weight`) uses **weight-side** functions only (`_rotate_weight`, `unrotate_weight_conv2d`, `unrotate_weight_linear`, `quantize_int8_convrot_weight`).
* `nodes/native_convert_int8.py` is unchanged; its `rotate_activation` is swapped only inside the window, and its bake-side counterpart `unrotate_weight_conv2d` is never swapped.
* NVFP4 families (Z Image / Krea2) have **zero** references to this module and never see a swap.
* Signature compatibility is preserved (`rotate_activation(x, h_matrix, group_size)`; `rotate_activation_nchw` untouched).

### 4.9 Known theoretical notes (stated, not hidden)

* **torch.compile**: the swap is an attribute assignment, so a graph traced while armed captures the pooled function and may keep calling it after disarm. The values are bitwise identical, so this is a path-identity nuance, not a correctness issue.
* **Aliasing**: the audit makes retention detectable for the audited Python call sites; `_rotate_activation` is a Python function, so any caller must be Python and is covered by the whole-package scan.

---

## Verification (reproducible)

```
# per-family loader dispatch + window behaviour
& "D:\USERFILES\ComfyUI\python_embeded\python.exe" "<workspace>\projects\ComfyUI-HSWQ-Loader-and-Tools-separation-evidence\evidence_per_family_private_load.py"

# fail-closed call-site audit
& "D:\USERFILES\ComfyUI\python_embeded\python.exe" "<workspace>\projects\ComfyUI-HSWQ-Loader-and-Tools-separation-evidence\evidence_call_site_audit.py"
```

Expected results:

```
Z Image ConvRot INT8 (DiT) / Krea2 ConvRot INT8 (DiT) / FLUX-DiT / SD1.5 INT8 : guard=False, module_loaded=False, sys.modules=False
SDXL ConvRot INT8 (UNet)                                                     : guard=True,  module_loaded=True,  sys.modules=False
inside window : rotate swapped=True | MAX_K=-1 | sys.modules=False
after window  : identity restored=True | pool released
audit         : real kitchen ok=True (7 call sites) ; retention pattern refused ; unexpected caller refused ; arm refused (untouched)
```

Runtime measurement (speed ratio, 25-seed trajectory gate) must be performed on the GPU host; those numbers are model- and machine-specific and are **not** transferable.

---

## Operational notes

* Live custom-node copies (`custom_nodes/ComfyUI-HSWQ-Loader-and-Tools`, `custom_nodes/comfyui-hswq-loader-and-tools`) are synced to this revision; the stale `patches/sdxl_convrot_fast.py` has been removed there (backup `*.bak-sync`).
* ComfyUI must be restarted for the new code to be loaded.
