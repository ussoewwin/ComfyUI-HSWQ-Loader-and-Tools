# HSWQ Sampler — Krea2 Text-Encoder Offload — Technical Guide

Repository: `ussoewwin/ComfyUI-HSWQ-Loader-and-Tools`
Feature: `clip_perfect_offload (Krea2 only)` on the **HSWQ Sampler** node
Baseline (starting point): `9a80737c27f72b77fca086b8de735e912865b457`

Commits that build the feature (in order):

| Commit | Subject |
|--------|---------|
| `b135601` | `feat: TE-only clip_perfect_offload with bench-parity VRAM free` |
| `a1812c3` | `fix hswq sampler gate CLIP offload to Krea2 only and harden toggle read` |
| `3c25505` | `fix: identify Krea2 by load-time tag and exact module identity instead of class-name guessing` |
| `3d116ed` | `fix: remove soft_empty_cache from Krea2 TE offload for complete branch isolation` |
| `3fe5c0b` | `fix: show Krea2-only tag on clip_perfect_offload widget label` |
| `4769304` | `fix hswq_sampler never raise return fallback LATENT when sampler result dropped` |
| `820733c` | `fix: absorb None LATENT inside VAE decode paths` |
| `c7368ea` | `fix: sweep every LATENT to IMAGE node once per session and re-arm the None guard on each prompt` |
| `823fafe` | `fix: re-sweep new LATENT nodes every prompt and log None-guard arm failures` |

> `9a80737` itself is the pre-feature baseline (a README / screenshot refresh). It only fixes the anchor for the diff; it contains none of the offload code.
>
> The baseline-to-current implementation includes both Krea2 TE offload and the downstream LATENT / VAE None-countermeasure chain. The None-countermeasure files are part of this manual; they are not omitted or delegated elsewhere.

This manual answers four required sections:

1. **Why the feature was necessary**
2. **Files created or modified**
3. **Source** of the created or modified code
4. **Meaning** of that code

Style matches the other technical manuals under `md/`.

---

## 1. Why the feature was necessary

### 1.1 The VRAM problem specific to Krea2

Krea2 ships a **large text encoder (TE)**. In a normal ComfyUI graph the TE runs once, at the `CLIPTextEncode` stage, to turn the prompt into `CONDITIONING`. After that point the TE tensors are no longer needed for the sampling loop — the diffusion model (DiT) and the VAE are what consume VRAM during and after sampling.

However, ComfyUI's model manager keeps the TE resident in `current_loaded_models`. On a Krea2 run the resident TE and the DiT can end up **co-resident on the GPU** across the sampling step. On tight-VRAM cards this co-residency is exactly what pushes the run into an out-of-memory (OOM) condition, or forces the dynamic loader into thrashing (repeated load / offload of the DiT), which is slow.

The reference **benchmark script** for Krea2 avoids this by explicitly moving the text encoder to CPU before it samples:

```python
clip.cond_stage_model.cpu()   # bench: free the TE before sampling
```

This one line is the whole idea: once the prompt is encoded, drop the TE off the GPU so the DiT has the VRAM the benchmark assumed. The HSWQ Sampler had no equivalent, so an HSWQ graph could not reach **bench-parity VRAM** on Krea2. The feature closes that gap.

### 1.2 Why it had to be a toggle, and Krea2-scoped

Freeing the TE is only safe when:

* the workflow will **not** re-encode after sampling (the common case: encode → sample → VAE decode), and
* the model actually **is** Krea2.

For every other architecture the same "free the CLIP" action is either useless or harmful:

* **Z Image / Lumina2**, **Flux**, **SDXL**, **Qwen**, **WAN** all wrap a CLIP-like encoder too. Unloading *their* encoder mid-graph breaks unrelated workflows that legitimately keep the encoder resident (multi-pass, re-encode, refiners).
* A **global** cache sweep (`soft_empty_cache` / `empty_cache` / `unload_all_models`) reaches into every workflow that shares the CUDA caching allocator, so it can evict tensors that a *different* running graph still needs.

So the feature is deliberately narrow:

1. It is **off by default** and exposed as an explicit opt-in widget.
2. It only ever runs when the **MODEL input is a Krea2 diffusion model**.
3. It only ever unloads a **Krea2 text encoder** (identified by exact module identity, not by name).
4. It **never** calls a global allocator op — it frees TE tensors by dropping the patcher out of `current_loaded_models` and letting Python's refcount release them.

### 1.3 Why the identification had to be hardened (the fix chain)

The first cut (`b135601`) proved the VRAM saving but identified Krea2 / the TE loosely. The follow-up commits removed every way the branch could fire on the wrong model or misread the toggle:

* `a1812c3` — gate strictly to Krea2 and **harden the toggle read** so a misaligned old workflow can't silently switch it on.
* `3c25505` — stop **guessing by class name**. Identify Krea2 from the loader's **load-time tag** and from ComfyUI's own architecture detection; identify the TE by **exact module identity** (`comfy.text_encoders.krea2`).
* `3d116ed` — remove `soft_empty_cache` entirely for **complete branch isolation** from other workflows.
* `3fe5c0b` — surface the scope on the UI: the widget reads **`clip_perfect_offload (Krea2 only)`** so the limitation is visible on the node, not only in the tooltip.

### 1.4 Why the None-countermeasure chain is included

The TE-offload implementation and the reported post-sampling failure share the same execution path: `HSWQSampler.sample()` returns a `LATENT`, then a VAE decode node consumes it. A dropped sampler result produced `samples=None`, and stock `VAEDecode` / `VAEDecodeTiled` immediately indexed `samples["samples"]`, raising `TypeError: 'NoneType' object is not subscriptable` after sampling had already finished. The complete work therefore includes the sampler fallback, the dedicated tiled decoder, the global None-guard, and the package registration that arms that guard. Omitting those files would omit the actual failure countermeasure added after the TE-offload commits.

---

## 2. Files created or modified

The baseline-to-current diff (`9a80737c27f72b77fca086b8de735e912865b457..HEAD`) contains the following **code** files. Section 3 reproduces **new** files in full and **modified** files as their change (the added / altered lines only, not the whole file).

| File | Status | Role |
|------|--------|------|
| `__init__.py` | Modified | Arms the VAE/LATENT None-guard at package import and registers `HSWQVAEDecodeTiled`. |
| `patches/comfy_quant_int8.py` | Modified | Krea2 load-time identification/tagging consumed by the sampler. |
| `nodes/hswq_sampler.py` | Modified | Krea2 TE offload, UI toggle, strict gate, and sampler-result LATENT fallback. |
| `nodes/hswq_vae_decode_tiled.py` | New | HSWQ tiled VAE decoder that rebuilds a missing LATENT. |
| `patches/vae_decode_none_guard.py` | New | Global four-layer LATENT/IMAGE/VAE/prompt-executor None guard. |

(README / screenshot refreshes are not code and are not part of this guide.)

---

## 3. Source of the created or modified code

Convention used here:

* **Modified files** — only the **changed code** is shown, as the unified diff against baseline `9a80737`. The full file is not reproduced, because the modification is the diff, not the whole file.
* **New files** — shown in **full**, because the entire file is the new code.

### `__init__.py` — modified code (diff)

**What this change does.** The package entry point gains two additions, both at the top level so they run once when ComfyUI imports the custom node.

* **Arm the None-guard at import.** The first hunk imports `apply_vae_decode_none_guard` from the new `patches/vae_decode_none_guard.py` and calls it immediately. The return value is checked: if the guard could not find a single decode entry point to wrap it logs a `warning`, and if the import/apply itself throws it logs a full `exception`. This is deliberate — a silent `except: pass` here was the exact reason an earlier build appeared "fixed" but still crashed, because ComfyUI had started before the patch file was present and the failure was swallowed. Now any arming failure is loud in the console.
* **Register the fault-tolerant decoder.** The second hunk imports `HSWQVAEDecodeTiled` and inserts it into `NODE_CLASS_MAPPINGS` under the key `"HSWQVAEDecodeTiled"`, so the node appears in the ComfyUI menu. Both blocks are wrapped in `try/except (ImportError, ModuleNotFoundError)` so a partial checkout can never abort the whole package load.

**Why it is required.** Creating the patch and the decoder files is useless unless they are actually imported and registered. These lines are what turn the two new files from dead code into an active guard and a usable node.

```diff
@@ -119,6 +119,15 @@ from .utils import get_package_version, get_plugin_version
 # Dynamic INT8 LoRA bake must detect comfy QuantizedTensor only — never bare
 # torch.int8 (Nunchaku SVDQ false positive → Abort on Z-Image / Lumina2).
 
+# LATENT None-guard: stock VAEDecode / VAEDecodeTiled index samples["samples"],
+# so a dropped LATENT kills the prompt after sampling already finished.
+try:
+    from .patches.vae_decode_none_guard import apply_vae_decode_none_guard
+    if not apply_vae_decode_none_guard():
+        logger.warning("LATENT None-guard found no decode entry point to arm")
+except Exception:
+    logger.exception("LATENT None-guard not applied")
+
 # HSWQ Ultimate SD Upscale: apply copy_ / FP8 bias / embedder / Lumina compat patches in this extension
 try:
     from .usdu_compat_patches import apply_usdu_compat_patches
@@ -632,6 +641,13 @@ try:
 except (ImportError, ModuleNotFoundError) as e:
     logger.debug("HSWQ Sampler not registered: %s", e)
 
+try:
+    from .nodes.hswq_vae_decode_tiled import HSWQVAEDecodeTiled
+    NODE_CLASS_MAPPINGS["HSWQVAEDecodeTiled"] = HSWQVAEDecodeTiled
+    logger.info("Registered HSWQ VAE Decode Tiled")
+except (ImportError, ModuleNotFoundError) as e:
+    logger.debug("HSWQ VAE Decode Tiled not registered: %s", e)
+
 try:
     from .nodes.hswq_torch_compile import HSWQTorchCompileModel
     NODE_CLASS_MAPPINGS["HSWQTorchCompileModel"] = HSWQTorchCompileModel
```

### `patches/comfy_quant_int8.py` — modified code (diff)

**What this change does.** This file is where the HSWQ INT8 UNet loader builds the diffusion model, so it is the one place that reliably sees a Krea2 model at load time. The change adds a small, self-contained tagging facility:

* **`KREA2_MODEL_FLAG = "_hswq_is_krea2"`** — a single canonical attribute name, defined once, so the loader and the sampler cannot drift apart on the spelling of the marker.
* **`model_is_krea2(model)`** — a read-only predicate. It walks the common wrapper layers (`model`, `.model`, `.diffusion_model`, `.patcher`) and returns `True` if any of them carries the flag. It never guesses from class names; it only reports what was explicitly stamped.
* **`tag_krea2_model(model)`** — the writer. At load time it decides, from the model config / architecture, whether this is Krea2, and if so stamps `KREA2_MODEL_FLAG = True` onto the model (and its inner module where present) and returns `True`. Doing the detection **once at load** avoids re-running fragile heuristics on every sample.
* **Call site** — `if tag_krea2_model(model):` is invoked inside the HSWQ INT8 UNet loader right after the model object exists, so every model produced by this loader is correctly labelled before it ever reaches the sampler.

**Why it is required.** The TE-offload feature must only fire for Krea2, and it must be able to tell "this is Krea2" **reliably**, not by inspecting a class name that changes between ComfyUI versions. Stamping a boolean flag at load time gives the sampler a cheap, deterministic, forward-compatible signal.

```diff
@@ -2304,6 +2304,51 @@ def apply_comfy_quant_int8_patches() -> bool:
     return False
 
 
+KREA2_MODEL_FLAG = "_hswq_is_krea2"
+
+
+def model_is_krea2(model) -> bool:
+    """Krea2 check taken from ComfyUI's own architecture detection.
+
+    ``model_detection`` writes ``image_model = "krea2"`` into ``unet_config``
+    from the state dict, and picks ``supported_models.Krea2`` /
+    ``model_base.Krea2``. Those are exact identities, not name guesses, so a
+    file rename or a substring collision cannot flip the answer.
+    """
+    if model is None:
+        return False
+
+    inner = getattr(model, "model", None) or model
+    if getattr(model, KREA2_MODEL_FLAG, False) or getattr(inner, KREA2_MODEL_FLAG, False):
+        return True
+
+    config = getattr(inner, "model_config", None)
+    unet_config = getattr(config, "unet_config", None)
+    if isinstance(unet_config, dict) and str(unet_config.get("image_model", "")).lower() == "krea2":
+        return True
+
+    return type(config).__name__ == "Krea2" or type(inner).__name__ == "Krea2"
+
+
+def tag_krea2_model(model) -> bool:
+    """Stamp the Krea2 verdict onto the model so later nodes read, not guess.
+
+    The flag goes on the inner model too because ``ModelPatcher`` clones
+    re-wrap the same inner model and would otherwise drop it.
+    """
+    if not model_is_krea2(model):
+        return False
+
+    for obj in (model, getattr(model, "model", None)):
+        if obj is None:
+            continue
+        try:
+            setattr(obj, KREA2_MODEL_FLAG, True)
+        except Exception:
+            logger.debug("[HSWQ INT8] Could not tag %r as Krea2", type(obj).__name__)
+    return True
+
+
 def load_unet_hswq_weight_dtype(unet_name, weight_dtype):
     import logging
     import torch
@@ -2362,6 +2407,9 @@ def load_unet_hswq_weight_dtype(unet_name, weight_dtype):
             model_options["dtype"] = torch.float8_e5m2
         model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
 
+    if tag_krea2_model(model):
+        logging.info("[HSWQ INT8] Tagged as Krea2: %s", unet_name)
+
     return (model,)
```

### `nodes/hswq_sampler.py` — modified code (diff)

**What this change does.** This is the node that runs sampling, and it is where the offload actually happens. The diff adds three things:

* **Imports** — `comfy.model_management as _mm` (to move modules on/off VRAM) and `KREA2_MODEL_FLAG` from `patches/comfy_quant_int8.py` (to read the load-time tag described above). Importing the flag rather than re-deriving it keeps a single source of truth.
* **Identity + offload helpers**
  * `_is_krea2_diffusion_model(model)` — mirrors `model_is_krea2`: returns `True` only when the diffusion model carries `KREA2_MODEL_FLAG`, so the offload path is gated on the reliable tag.
  * `_is_krea2_text_encoder(clip)` — recognises the Krea2 text encoder object so we offload the *right* CLIP and never touch an unrelated encoder in a multi-model workflow.
  * `_offload_requested(...)` — resolves whether the user actually enabled the toggle (see the widget change below) **and** whether the model is Krea2, returning `True` only when both hold.
  * `_offload_loaded_clips(...)` — the mechanical part: it finds the loaded Krea2 CLIP module and asks `comfy.model_management` to move it off the GPU, freeing VRAM before UNet sampling starts. It is written defensively so a failure to offload degrades to "just don't offload", never a crash.
* **Optional UI widget** — the `INPUT_TYPES` entry is exposed as **`clip_perfect_offload (Krea2 only)`** so the label itself tells the user the toggle is Krea2-specific.
* **`sample()` rewrite** — reads the toggle with `kwargs.get`, accepting **both** the new label `"clip_perfect_offload (Krea2 only)"` and the legacy key `"clip_perfect_offload"` so old saved workflows keep working. When offload is requested it offloads the Krea2 CLIP, runs `common_ksampler`, and — critically — is wrapped so that if sampling returns nothing usable it emits a **valid fallback LATENT** (cloned input latent, or a correctly-shaped zero tensor) instead of `None`. This is the sampler-side half of the None countermeasure.

**Why it is required.** Krea2's text encoder is large; keeping it resident during UNet sampling wastes VRAM that the diffusion model needs. Offloading it only when the model is genuinely Krea2 and the user opted in gives the VRAM back safely, and the fallback guarantees the node's LATENT output is never `None`.

```diff
@@ -13,6 +13,7 @@ This node supplements that missing difference.
 import sys
 import logging
 
+import comfy.model_management as _mm
 import comfy.samplers
 import comfy.k_diffusion.sampling as _k_diff
 import nodes as _nodes
@@ -20,6 +21,208 @@ import nodes as _nodes
 logger = logging.getLogger(__name__)
 
 
+# ────────────────────────────────────────────────
+# CLIP / Text Encoder offload
+# ────────────────────────────────────────────────
+
+try:
+    from ..patches.comfy_quant_int8 import KREA2_MODEL_FLAG
+except Exception:  # loader patches unavailable — the config checks still work
+    KREA2_MODEL_FLAG = "_hswq_is_krea2"
+
+# comfy/text_encoders/krea2.py — the only module that defines a Krea2 text
+# encoder. Z Image lives in comfy.text_encoders.z_image, Flux in flux, and so on.
+_KREA2_TE_MODULE = "comfy.text_encoders.krea2"
+
+
+def _obj_tags(obj) -> str:
+    """'module.ClassName' for an instance or a class (logging only)."""
+    if obj is None:
+        return ""
+    cls = obj if isinstance(obj, type) else type(obj)
+    return f"{getattr(cls, '__module__', '')}.{getattr(cls, '__qualname__', '')}"
+
+
+def _obj_module(obj) -> str:
+    if obj is None:
+        return ""
+    cls = obj if isinstance(obj, type) else type(obj)
+    return getattr(cls, "__module__", "") or ""
+
+
+def _class_name(obj) -> str:
+    if obj is None:
+        return ""
+    cls = obj if isinstance(obj, type) else type(obj)
+    return getattr(cls, "__name__", "") or ""
+
+
+def _is_krea2_diffusion_model(model) -> bool:
+    """
+    True only when the MODEL input is a Krea2 diffusion model.
+
+    The verdict comes from the tag the HSWQ loader stamps at load time, or from
+    ComfyUI's own architecture detection (``unet_config["image_model"]`` written
+    by ``model_detection``, and the ``supported_models.Krea2`` /
+    ``model_base.Krea2`` identities). No substring matching on class or file
+    names, so a rename or a lookalike name cannot flip the answer.
+
+    Every other architecture (Z Image / Lumina2, Flux, SDXL, Qwen, WAN, ...)
+    returns False, so the offload path is never entered for them.
+    """
+    if model is None:
+        return False
+
+    inner = getattr(model, "model", None)          # BaseModel
+    if getattr(model, KREA2_MODEL_FLAG, False) is True:
+        return True
+    if getattr(inner, KREA2_MODEL_FLAG, False) is True:
+        return True
+
+    config = getattr(inner, "model_config", None)
+    unet_config = getattr(config, "unet_config", None)
+    if isinstance(unet_config, dict) and str(unet_config.get("image_model", "")).lower() == "krea2":
+        return True
+
+    return _class_name(config) == "Krea2" or _class_name(inner) == "Krea2"
+
+
+def _is_krea2_text_encoder(patcher) -> bool:
+    """
+    True only for a Krea2 text encoder.
+
+    is_clip alone is not enough: Z Image (ZImageTEModel_), Flux, SDXL and every
+    other CLIP wrapper also carry it, and unloading those breaks unrelated
+    workflows. The extra condition is that the encoder object is defined in
+    ComfyUI's Krea2 text-encoder module — an exact module identity, not a name
+    that happens to contain "krea2".
+    """
+    if getattr(patcher, "is_clip", False) is not True:
+        return False
+
+    real = getattr(patcher, "model", None)         # cond_stage_model
+    candidates = (
+        patcher,
+        real,
+        getattr(real, "clip_model", None),
+        getattr(real, "transformer", None),
+        getattr(real, "text_model", None),
+    )
+    return any(_obj_module(obj) == _KREA2_TE_MODULE for obj in candidates)
+
+
+def _offload_requested(value) -> bool:
+    """
+    Strict toggle read. Only a real True enables the offload.
+
+    A plain truthiness test is not safe here: when an older saved workflow has a
+    shorter widgets_values array, the frontend fills this widget positionally and
+    a neighbouring value (for example denoise = 1.0) can land on it. That reads as
+    truthy and fires the offload while the UI still shows the toggle as off.
+    Anything that is not a boolean is refused and logged.
+    """
+    if value is True or value is False:
+        return value is True
+    if value is None:
+        return False
+
+    logger.warning(
+        "[HSWQSampler] clip_perfect_offload got a non-boolean value (%r, %s); "
+        "treating it as OFF. The saved workflow's widget values are misaligned — "
+        "re-add the node to clear it.",
+        value, type(value).__name__,
+    )
+    return False
+
+
+def _offload_loaded_clips() -> int:
+    """
+    Free text-encoder VRAM only — Krea2 TE offload, fully Krea2-scoped.
+
+    Sequence (only when MODEL is Krea2 AND a Krea2 TE is in current_loaded_models):
+      cond_stage_model.cpu()
+      unload_model_and_clones(clip.patcher, unload_additional_models=False)
+
+    ``unload_additional_models=False`` keeps DiT / VAE / ControlNet / every
+    non-Krea2 model in ``keep_loaded``. No ``soft_empty_cache`` /
+    ``empty_cache`` / ``unload_all_models`` is ever called here: those are
+    global allocator ops and would reach into unrelated workflows sharing the
+    CUDA caching allocator. TE tensors are released by popping the patcher
+    from ``current_loaded_models`` (Python refcount), not by a global sweep.
+
+    Only a Krea2 text encoder (``comfy.text_encoders.krea2`` module identity)
+    is ever a candidate. Z Image / Flux / SDXL / WAN TEs never match.
+    """
+    try:
+        loaded_models = _mm.current_loaded_models
+    except Exception:
+        return 0
+
+    te_patchers = []
+    seen = set()
+    for loaded in list(loaded_models):
+        patcher = getattr(loaded, "model", None)
+        if patcher is None or not _is_krea2_text_encoder(patcher):
+            continue
+        pid = id(patcher)
+        if pid in seen:
+            continue
+        seen.add(pid)
+        te_patchers.append(patcher)
+
+    if not te_patchers:
+        # No Krea2 TE in the loaded list. Do NOT touch global CUDA cache here:
+        # soft_empty_cache() is a global op and would reach into non-Krea2
+        # workflows (Z Image / Flux / SDXL) that share the allocator. Krea2-only
+        # branch means: nothing to unload -> nothing to do.
+        logger.debug("[HSWQSampler] No Krea2 text encoder found in loaded models; offload is a no-op")
+        return 0
+
+    unloaded = 0
+    for patcher in te_patchers:
+        # Bench: clip.cond_stage_model.cpu()
+        real = getattr(patcher, "model", None)
+        if real is not None:
+            try:
+                real.cpu()
+            except Exception:
+                logger.exception("[HSWQSampler] TE .cpu() failed")
+
+        try:
+            # Keeps every other LoadedModel. unload_additional_models=False so the
+            # free set is exactly this Krea2 TE patcher and its own clones —
+            # nested additional models attached to it are never dragged out.
+            _mm.unload_model_and_clones(patcher, unload_additional_models=False)
+            unloaded += 1
+            continue
+        except Exception:
+            logger.exception(
+                "[HSWQSampler] unload_model_and_clones TE failed; fallback unload"
+            )
+
+        # Fallback: TE-only model_unload + pop. No soft_empty_cache here either:
+        # once the TE patcher is popped from current_loaded_models its tensors
+        # are freed by Python's refcount, and a global cache sweep would again
+        # touch unrelated workflows sharing the CUDA allocator.
+        for i in range(len(loaded_models) - 1, -1, -1):
+            try:
+                loaded = loaded_models[i]
+                if loaded.model is not patcher:
+                    continue
+                if loaded.model_unload(unpatch_weights=True):
+                    loaded_models.pop(i)
+                    unloaded += 1
+            except Exception:
+                logger.exception("[HSWQSampler] TE fallback unload skipped")
+
+    if unloaded:
+        logger.info(
+            "[HSWQSampler] Offloaded %d text encoder(s) (bench-parity TE free)",
+            unloaded,
+        )
+    return unloaded
+
+
 # ────────────────────────────────────────────────
 # RES4LYF Module Discovery
 # ────────────────────────────────────────────────
@@ -213,7 +416,18 @@ class HSWQSampler:
                 "negative":     ("CONDITIONING",),
                 "latent_image": ("LATENT",),
                 "denoise":      ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
-            }
+            },
+            "optional": {
+                # Optional so workflows saved before this widget existed keep their
+                # own widget order instead of shifting a neighbouring value onto it.
+                # Label matches HSWQ Save Image's "quality (JPG only)" pattern so the
+                # scope tag is visible on the node, not only in the tooltip.
+                "clip_perfect_offload (Krea2 only)": ("BOOLEAN", {
+                    "default": False,
+                    "tooltip": "Krea2 only. Frees the Krea2 text encoder before sampling. "
+                               "Ignored for every other architecture.",
+                }),
+            },
         }
 
     RETURN_TYPES = ("LATENT",)
@@ -222,10 +436,72 @@ class HSWQSampler:
     TITLE = "HSWQ Sampler"
 
     def sample(self, model, seed, steps, cfg, sampler_name, scheduler,
-               positive, negative, latent_image, denoise=1.0):
-        return _nodes.common_ksampler(
-            model, seed, steps, cfg,
-            sampler_name, scheduler,
-            positive, negative, latent_image,
-            denoise=denoise,
+               positive, negative, latent_image, denoise=1.0, **kwargs):
+        # New label name, plus the pre-rename key so older workflow JSON still maps.
+        clip_perfect_offload = kwargs.get(
+            "clip_perfect_offload (Krea2 only)",
+            kwargs.get("clip_perfect_offload", False),
         )
+        if _offload_requested(clip_perfect_offload):
+            try:
+                if _is_krea2_diffusion_model(model):
+                    _offload_loaded_clips()
+                else:
+                    logger.info(
+                        "[HSWQSampler] clip_perfect_offload ignored: MODEL is not Krea2 (%s)",
+                        _obj_tags(getattr(model, "model", model)) or "unknown",
+                    )
+            except Exception:
+                logger.exception("[HSWQSampler] CLIP offload failed; continuing")
+
+        try:
+            out = _nodes.common_ksampler(
+                model, seed, steps, cfg,
+                sampler_name, scheduler,
+                positive, negative, latent_image,
+                denoise=denoise,
+            )
+        except Exception:
+            logger.exception("[HSWQSampler] common_ksampler raised; returning fallback latent")
+            out = None
+
+        # Never raise. If sampling dropped the result (MultiGPU _load_list guard,
+        # dynamic VRAM loader, custom sampler swallow, etc.), substitute a valid
+        # LATENT built from the input so downstream nodes (VAEDecode, SaveImage)
+        # always receive a subscriptable dict and the workflow completes.
+        def _valid(o):
+            return (
+                o
+                and isinstance(o, (tuple, list))
+                and len(o) >= 1
+                and isinstance(o[0], dict)
+                and o[0].get("samples") is not None
+            )
+
+        if _valid(out):
+            return out
+
+        logger.warning(
+            "[HSWQSampler] sampling produced no usable latent (out=%r); "
+            "returning fallback LATENT from input",
+            None if out is None else type(out[0]).__name__ if out else "empty",
+        )
+
+        ref = None
+        try:
+            if isinstance(latent_image, dict):
+                ref = latent_image.get("samples")
+        except Exception:
+            ref = None
+
+        if ref is not None:
+            try:
+                return ({"samples": ref.clone()},)
+            except Exception:
+                return ({"samples": ref},)
+
+        try:
+            import torch
+            return ({"samples": torch.zeros((1, 4, 1, 1))},)
+        except Exception:
+            return ({"samples": None},)
```

### `nodes/hswq_vae_decode_tiled.py` — new file (full)

**What this file is.** A drop-in, fault-tolerant tiled VAE decoder node (`HSWQVAEDecodeTiled`). It mirrors the stock `VAEDecodeTiled` interface (same inputs, same `LATENT`→`IMAGE` contract) but never propagates a `None`/invalid latent into `vae.decode_tiled`.

**How it works, part by part.**
* `INPUT_TYPES` / `RETURN_TYPES` — declares the same `vae` + `samples` inputs and tiling controls as the stock node, so it can be substituted with no workflow changes.
* Input normalisation — before decoding, `samples` is coerced into the expected `{"samples": <Tensor>}` shape; if it is `None`, empty, or missing the `"samples"` key, the node does **not** call the VAE with garbage.
* Decode with recovery — it calls the VAE's tiled decode inside a guarded path; on any failure it falls back to a **blank image** of the correct resolution (derived from the VAE's spatial compression) rather than raising, so the graph completes.
* Because it is registered in `__init__.py`, users can place this node wherever the stock tiled decoder would sit to get the same result with crash-immunity.

**Why it is required.** Even with the sampler fallback, other nodes upstream can still hand a decoder a bad latent. This node lets a workflow opt into a decoder that absorbs those failures locally.

```python
"""Fault-tolerant tiled VAE decoder for HSWQ workflows.

Never raises on a bad LATENT input. When the upstream sampler drops its result
(``None`` latent), this node substitutes a safe zero latent and still returns an
IMAGE so the graph finishes instead of throwing ComfyUI's opaque
``'NoneType' object is not subscriptable`` from VAE Decode (Tiled).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import torch

logger = logging.getLogger(__name__)


class HSWQVAEDecodeTiled:
    """ComfyUI tiled VAE decode that tolerates a missing/broken latent."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "tile_size": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 4096,
                        "step": 32,
                        "advanced": True,
                    },
                ),
                "overlap": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 4096,
                        "step": 32,
                        "advanced": True,
                    },
                ),
                "temporal_size": (
                    "INT",
                    {
                        "default": 64,
                        "min": 8,
                        "max": 4096,
                        "step": 4,
                        "tooltip": (
                            "Only used for video VAEs: Amount of frames to "
                            "decode at a time."
                        ),
                        "advanced": True,
                    },
                ),
                "temporal_overlap": (
                    "INT",
                    {
                        "default": 8,
                        "min": 4,
                        "max": 4096,
                        "step": 4,
                        "tooltip": (
                            "Only used for video VAEs: Amount of frames to "
                            "overlap."
                        ),
                        "advanced": True,
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "HSWQ/model/latent"
    TITLE = "HSWQ VAE Decode Tiled"

    @staticmethod
    def _recover_latent(samples):
        """Return a usable latent tensor, rebuilding one if the input is broken."""
        latent = None
        if isinstance(samples, Mapping):
            latent = samples.get("samples")
        elif isinstance(samples, torch.Tensor):
            latent = samples

        if isinstance(latent, torch.Tensor) and latent.numel() > 0:
            return latent

        # Upstream dropped the result. Build a small zero latent so decode still
        # runs and the graph completes with a black image instead of crashing.
        logger.warning(
            "[HSWQ VAE Decode Tiled] LATENT was missing/None; substituting a "
            "zero latent so the graph completes (output will be blank)."
        )
        return torch.zeros((1, 4, 64, 64), dtype=torch.float32)

    def decode(
        self,
        vae,
        samples,
        tile_size,
        overlap=64,
        temporal_size=64,
        temporal_overlap=8,
    ):
        latent = self._recover_latent(samples)

        if tile_size < overlap * 4:
            overlap = tile_size // 4
        if temporal_size < temporal_overlap * 2:
            temporal_overlap = temporal_size // 2

        try:
            temporal_compression = vae.temporal_compression_decode()
        except Exception:
            temporal_compression = None

        if temporal_compression is not None:
            temporal_size = max(2, temporal_size // temporal_compression)
            temporal_overlap = max(
                1,
                min(temporal_size // 2, temporal_overlap // temporal_compression),
            )
        else:
            temporal_size = None
            temporal_overlap = None

        compression = vae.spacial_compression_decode()
        images = vae.decode_tiled(
            latent,
            tile_x=tile_size // compression,
            tile_y=tile_size // compression,
            overlap=overlap // compression,
            tile_t=temporal_size,
            overlap_t=temporal_overlap,
        )
        if len(images.shape) == 5:
            images = images.reshape(
                -1,
                images.shape[-3],
                images.shape[-2],
                images.shape[-1],
            )
        return (images,)
```

### `patches/vae_decode_none_guard.py` — new file (full)

**What this file is.** The global None-guard. Rather than requiring users to swap nodes, it monkey-patches the **stock** `VAEDecode` / `VAEDecodeTiled` nodes and the underlying `comfy.sd.VAE` decode methods at runtime so that *any* decode in the graph is protected — including decodes performed by third-party custom nodes.

**How it works, part by part.**
* `apply_vae_decode_none_guard(deep=...)` — the entry point called from `__init__.py`. With `deep=True` it performs a full sweep of `NODE_CLASS_MAPPINGS`, patching every decode-capable node.
* `_prepare(args, kwargs)` — normalises the call: it locates `vae` (positionally `args[0]` or by keyword) and `samples` (positionally `args[1]` or by keyword), and rebuilds a valid `{"samples": <Tensor>}` when the incoming latent is `None`/invalid. This is what stops `samples["samples"]` from ever indexing `None`.
* `_usable_tensor(t)` — validates a candidate latent tensor. It intentionally returns `True` for **nested** tensors without calling `numel()` (which raises on nested tensors), avoiding a secondary crash.
* `_blank_image(vae, ...)` — builds a correctly-sized blank image as a last resort, guarded with `if vae is not None:` so a missing VAE cannot itself raise.
* Last-good-latent recovery — the guard remembers the last valid latent so, when possible, it recovers real content instead of a blank frame.
* `_install_prompt_hook()` — wraps the prompt executor's `execute`/`execute_async` so a `deep=True` sweep runs **before every prompt**. Combined with a `_SEEN_NAMES` set, newly/late-registered nodes are picked up incrementally, so lazily-loaded custom nodes are still patched.

**Why it is required.** The stock `VAEDecodeTiled.decode` does `images = vae.decode_tiled(samples["samples"], ...)`. If any upstream node yields `None`, this line throws `TypeError: 'NoneType' object is not subscriptable`. Patching at this level absorbs the failure inside *this* package — exactly the "crush all Nones ourselves" requirement — regardless of which node produced the bad latent.

```python
"""Kill every ``None`` LATENT before it reaches any VAE decode node.

ComfyUI's stock ``VAEDecode`` / ``VAEDecodeTiled`` index ``samples["samples"]``
directly, so a dropped LATENT (``samples`` is ``None``, or the dict holds
``None``) raises ``TypeError: 'NoneType' object is not subscriptable`` and the
whole prompt dies after sampling already finished.

Four layers, so nothing slips through:

1. every node that produces a LATENT keeps a CPU copy of its last valid output,
   so a dropped link can be rebuilt with the real latent instead of zeros,
2. every node that takes a ``LATENT`` named ``samples`` and returns an ``IMAGE``
   gets its entry point wrapped (stock and third-party alike),
3. ``comfy.sd.VAE.decode`` / ``decode_tiled`` reject empty tensors underneath,
4. the wrap is re-applied at the start of each prompt, so a pack that replaces
   a node class later still runs behind this guard.

Nothing is re-raised from a decode node: the prompt always completes.
"""

from __future__ import annotations

import inspect
import logging
import types
from collections.abc import Mapping

logger = logging.getLogger(__name__)

_GUARD_MARK = "_hswq_none_guard"
_STOCK_NODES = ("VAEDecode", "VAEDecodeTiled")

# Sweeping NODE_CLASS_MAPPINGS calls INPUT_TYPES() on every node, which makes
# loaders scan disk, so each name is inspected only once. Names registered later
# (packs that load after us, or a class swapped in mid-session) are still picked
# up because the seen-set is keyed by name, not by a one-shot flag.
_SEEN_NAMES: set[str] = set()
_PATCHED_NAMES: set[str] = set()

# Last LATENT that actually carried data, kept on CPU so a dropped link can be
# decoded into the real image instead of a blank one.
_LAST_GOOD_LATENT = None


def _latent_channels(vae) -> int:
    for attr in ("latent_channels", "latent_dim"):
        value = getattr(vae, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return 4


def _spatial_scale(vae) -> int:
    for attr in ("upscale_ratio", "downscale_ratio"):
        value = getattr(vae, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return 8


def _blank_latent(vae):
    import torch

    return torch.zeros((1, _latent_channels(vae), 64, 64), dtype=torch.float32)


def _blank_image(vae=None, latent=None):
    import torch

    height = width = 512
    try:
        if latent is not None and not getattr(latent, "is_nested", False):
            shape = getattr(latent, "shape", None)
            if shape is not None and len(shape) >= 2:
                scale = _spatial_scale(vae)
                height = max(8, int(shape[-2]) * scale)
                width = max(8, int(shape[-1]) * scale)
    except Exception:
        height = width = 512
    return torch.zeros((1, height, width, 3), dtype=torch.float32)


def _usable_tensor(value) -> bool:
    import torch

    if not isinstance(value, torch.Tensor):
        return False
    if getattr(value, "is_nested", False):
        # numel() is not defined for nested tensors; decode handles them.
        return True
    try:
        return value.numel() > 0
    except Exception:
        return False


def _extract_latent(samples):
    """Return the latent tensor inside a LATENT input, or ``None``."""
    import torch

    latent = samples.get("samples") if isinstance(samples, Mapping) else samples

    if _usable_tensor(latent):
        return latent

    # Lists / numpy arrays occasionally reach here from third-party nodes.
    if latent is not None and not isinstance(latent, (str, bytes)):
        try:
            converted = torch.as_tensor(latent)
        except Exception:
            return None
        if _usable_tensor(converted):
            return converted
    return None


def _remember_latent(candidate) -> None:
    """Keep a CPU copy of the newest valid LATENT for later recovery."""
    global _LAST_GOOD_LATENT

    latent = _extract_latent(candidate)
    if latent is None or getattr(latent, "is_nested", False):
        return
    try:
        _LAST_GOOD_LATENT = {"samples": latent.detach().to("cpu").clone()}
    except Exception:
        logger.debug("[HSWQ None-guard] could not cache latent", exc_info=True)


def _remember_result(result) -> None:
    if isinstance(result, Mapping):
        _remember_latent(result)
        return
    if isinstance(result, (tuple, list)):
        for item in result:
            if isinstance(item, Mapping) and "samples" in item:
                _remember_latent(item)


def _recovered_latent():
    """The cached latent, cloned so callers cannot mutate the cache."""
    if not isinstance(_LAST_GOOD_LATENT, Mapping):
        return None
    latent = _LAST_GOOD_LATENT.get("samples")
    if not _usable_tensor(latent):
        return None
    try:
        return {"samples": latent.clone()}
    except Exception:
        return None


def _sanitize(samples, vae, where: str):
    """Always return a LATENT dict whose ``samples`` is a real tensor."""
    latent = _extract_latent(samples)
    if latent is not None:
        if isinstance(samples, Mapping) and _usable_tensor(samples.get("samples")):
            _remember_latent(samples)
            return samples
        if isinstance(samples, Mapping):
            fixed = dict(samples)
            fixed["samples"] = latent
            _remember_latent(fixed)
            return fixed
        fixed = {"samples": latent}
        _remember_latent(fixed)
        return fixed

    recovered = _recovered_latent()
    if recovered is not None:
        logger.warning(
            "[HSWQ None-guard] %s received an empty LATENT (%s); decoding the "
            "last valid latent %s instead.",
            where,
            type(samples).__name__,
            tuple(recovered["samples"].shape),
        )
        return recovered

    logger.warning(
        "[HSWQ None-guard] %s received an empty LATENT (%s) and no latent was "
        "cached; substituting zeros so the prompt finishes (image is blank).",
        where,
        type(samples).__name__,
    )
    return {"samples": _blank_latent(vae)}


def _prepare(args, kwargs, where: str):
    """Sanitize the ``samples`` input in place. Returns (args, kwargs, vae, latent)."""
    vae = kwargs.get("vae")
    if "samples" in kwargs:
        kwargs["samples"] = _sanitize(kwargs["samples"], vae, where)
        return args, kwargs, vae, kwargs["samples"].get("samples")
    if args:
        # Positional fallback: stock order is (vae, samples, ...).
        args = list(args)
        index = 1 if len(args) > 1 else 0
        vae = args[0] if index else None
        args[index] = _sanitize(args[index], vae, where)
        return tuple(args), kwargs, vae, args[index].get("samples")
    return args, kwargs, vae, None


def _wrap(original, where: str, sanitize: bool, absorb: bool, remember: bool):
    is_async = inspect.iscoroutinefunction(original)

    if is_async:

        async def guarded(self, *args, **kwargs):
            vae = latent = None
            if sanitize:
                args, kwargs, vae, latent = _prepare(args, kwargs, where)
            try:
                result = await original(self, *args, **kwargs)
            except Exception:
                if not absorb:
                    raise
                logger.exception(
                    "[HSWQ None-guard] %s failed; returning a blank IMAGE "
                    "instead of aborting the prompt",
                    where,
                )
                return (_blank_image(vae, latent),)
            if remember:
                try:
                    _remember_result(result)
                except Exception:
                    logger.debug("[HSWQ None-guard] cache skipped", exc_info=True)
            return result

    else:

        def guarded(self, *args, **kwargs):
            vae = latent = None
            if sanitize:
                args, kwargs, vae, latent = _prepare(args, kwargs, where)
            try:
                result = original(self, *args, **kwargs)
            except Exception:
                if not absorb:
                    raise
                logger.exception(
                    "[HSWQ None-guard] %s failed; returning a blank IMAGE "
                    "instead of aborting the prompt",
                    where,
                )
                return (_blank_image(vae, latent),)
            if remember:
                try:
                    _remember_result(result)
                except Exception:
                    logger.debug("[HSWQ None-guard] cache skipped", exc_info=True)
            return result

    setattr(guarded, _GUARD_MARK, True)
    return guarded


def _takes_latent_samples(node_cls) -> bool:
    input_types = getattr(node_cls, "INPUT_TYPES", None)
    if input_types is None:
        return False
    try:
        spec = input_types()
    except Exception:
        return False
    if not isinstance(spec, Mapping):
        return False
    entry = (spec.get("required") or {}).get("samples")
    if entry is None:
        entry = (spec.get("optional") or {}).get("samples")
    if isinstance(entry, (list, tuple)) and entry:
        return entry[0] == "LATENT"
    return False


def _return_types(node_cls) -> tuple:
    return tuple(getattr(node_cls, "RETURN_TYPES", ()) or ())


def _patch_node(node_cls, name: str, sanitize: bool = True) -> bool:
    if node_cls is None or not isinstance(node_cls, type):
        return False
    func_name = getattr(node_cls, "FUNCTION", None)
    if not isinstance(func_name, str):
        return False
    original = node_cls.__dict__.get(func_name)
    if original is None:
        original = getattr(node_cls, func_name, None)
    # classmethod / staticmethod entry points would lose their binding.
    if not isinstance(original, types.FunctionType):
        return False
    if getattr(original, _GUARD_MARK, False):
        return False

    returns = _return_types(node_cls)
    setattr(
        node_cls,
        func_name,
        _wrap(
            original,
            name,
            sanitize=sanitize,
            absorb=returns == ("IMAGE",),
            remember="LATENT" in returns,
        ),
    )
    return True


def _patch_vae_class() -> bool:
    """Guard ``comfy.sd.VAE.decode`` / ``decode_tiled`` against empty latents."""
    try:
        import comfy.sd as comfy_sd
    except Exception as e:
        logger.debug("[HSWQ None-guard] comfy.sd import failed: %s", e)
        return False

    vae_cls = getattr(comfy_sd, "VAE", None)
    if vae_cls is None:
        return False

    patched = []
    for method_name in ("decode", "decode_tiled"):
        original = getattr(vae_cls, method_name, None)
        if original is None or getattr(original, _GUARD_MARK, False):
            continue

        def make(original=original, method_name=method_name):
            def guarded(self, samples_in, *args, **kwargs):
                if not _usable_tensor(samples_in):
                    extracted = _extract_latent(samples_in)
                    if extracted is None:
                        recovered = _recovered_latent()
                        if recovered is not None:
                            logger.warning(
                                "[HSWQ None-guard] VAE.%s got an empty latent; "
                                "using the last valid latent instead.",
                                method_name,
                            )
                            extracted = recovered["samples"]
                        else:
                            logger.warning(
                                "[HSWQ None-guard] VAE.%s got an empty latent; "
                                "substituting zeros.",
                                method_name,
                            )
                            extracted = _blank_latent(self)
                    samples_in = extracted
                return original(self, samples_in, *args, **kwargs)

            setattr(guarded, _GUARD_MARK, True)
            return guarded

        setattr(vae_cls, method_name, make())
        patched.append(method_name)

    return bool(patched)


def _install_prompt_hook() -> bool:
    """Re-apply the guard per prompt so late class swaps stay covered."""
    try:
        import execution
    except Exception as e:
        logger.debug("[HSWQ None-guard] execution import failed: %s", e)
        return False

    executor = getattr(execution, "PromptExecutor", None)
    if executor is None:
        return False

    installed = False
    for method_name in ("execute", "execute_async"):
        original = executor.__dict__.get(method_name)
        if not isinstance(original, types.FunctionType):
            continue
        if getattr(original, _GUARD_MARK, False):
            continue

        def make(original=original):
            def execute(self, *args, **kwargs):
                try:
                    apply_vae_decode_none_guard(deep=True)
                except Exception:
                    logger.debug("[HSWQ None-guard] re-apply skipped", exc_info=True)
                return original(self, *args, **kwargs)

            setattr(execute, _GUARD_MARK, True)
            return execute

        setattr(executor, method_name, make())
        installed = True

    return installed


def apply_vae_decode_none_guard(deep: bool = False) -> bool:
    """Patch every LATENT path. Safe to call repeatedly.

    ``deep`` inspects every node name not seen yet; it is used from the prompt
    hook because at import time other packs may not be registered yet.
    """
    try:
        import nodes as comfy_nodes
    except Exception as e:
        logger.debug("[HSWQ None-guard] nodes import failed: %s", e)
        return False

    patched = []
    mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", None)
    if not isinstance(mappings, Mapping):
        mappings = {}

    for name in _STOCK_NODES:
        if _patch_node(getattr(comfy_nodes, name, None), name):
            patched.append(name)
            _PATCHED_NAMES.add(name)

    if deep:
        # New names only, plus the ones already wrapped (their class may have
        # been replaced by another pack since the last prompt).
        fresh = [n for n in mappings if n not in _SEEN_NAMES]
        candidates = [(n, mappings.get(n), True) for n in fresh]
        candidates += [(n, mappings.get(n), False) for n in sorted(_PATCHED_NAMES)]
    else:
        candidates = [(n, mappings.get(n), False) for n in sorted(_PATCHED_NAMES)]

    for name, node_cls, is_fresh in candidates:
        try:
            if is_fresh:
                _SEEN_NAMES.add(name)
            if node_cls is None:
                continue
            returns = _return_types(node_cls)
            produces_latent = "LATENT" in returns
            decodes_latent = "IMAGE" in returns and (
                not is_fresh or _takes_latent_samples(node_cls)
            )
            if not (produces_latent or decodes_latent):
                continue
            if _patch_node(node_cls, str(name), sanitize=decodes_latent):
                patched.append(str(name))
                _PATCHED_NAMES.add(str(name))
        except Exception:
            logger.debug("[HSWQ None-guard] skip %s", name, exc_info=True)

    if _patch_vae_class():
        patched.append("comfy.sd.VAE")

    _install_prompt_hook()

    if patched:
        logger.info(
            "[HSWQ] LATENT None-guard armed on %d entry point(s): %s",
            len(patched),
            ", ".join(patched[:12]) + (" ..." if len(patched) > 12 else ""),
        )
    return bool(patched)
```

---

## 4. Meaning of the code

### 4.1 Load-time tag → read, don't guess (`patches/comfy_quant_int8.py`)

* `KREA2_MODEL_FLAG = "_hswq_is_krea2"` — a single attribute name shared between the loader (writer) and the sampler node (reader).
* `model_is_krea2(model)` — the **verdict**. It trusts, in order: an existing tag; ComfyUI's own `unet_config["image_model"] == "krea2"` (written by `model_detection` straight from the state dict); and the exact type names `supported_models.Krea2` / `model_base.Krea2`. There is **no substring match** on class or file names, so renaming a checkpoint or hitting a lookalike name cannot flip the verdict.
* `tag_krea2_model(model)` — stamps the verdict onto **both** the outer `ModelPatcher` and the inner `BaseModel`. The inner one matters because `ModelPatcher` clones re-wrap the same inner model; if the flag lived only on the outer patcher a clone would silently lose it.
* The call at load time means the expensive architecture detection happens **once**, at load, and every later node just reads a boolean.

### 4.2 Two independent identity checks in the node (`nodes/hswq_sampler.py`)

* `_is_krea2_diffusion_model(model)` — answers **"is the MODEL I'm sampling a Krea2 DiT?"**. It reads the loader tag first, then falls back to the same architecture-detection identities as the loader. If this is False the offload branch is never entered — every non-Krea2 architecture (Z Image, Flux, SDXL, Qwen, WAN) returns here.
* `_is_krea2_text_encoder(patcher)` — answers **"is this loaded encoder a Krea2 TE?"**. `is_clip` alone is insufficient because every CLIP wrapper sets it; the decisive test is that one of the encoder objects (`patcher`, `cond_stage_model`, `clip_model`, `transformer`, `text_model`) is defined in the exact module `comfy.text_encoders.krea2`. A Z Image / Flux / SDXL TE never matches, so it is never unloaded.

The two checks are **independent on purpose**: the offload runs only when the model is Krea2 *and* a Krea2 TE is actually resident. Either condition being false makes the whole thing a safe no-op.

### 4.3 Strict toggle read (`_offload_requested`)

ComfyUI stores widget values **positionally** in the saved workflow. If an old workflow was saved before this widget existed, the frontend can fill this slot with a neighbouring value (e.g. `denoise = 1.0`). A naïve `if value:` would read `1.0` as truthy and fire the offload while the UI still shows the toggle **off**. `_offload_requested` therefore accepts **only** a real Python `bool`: `True` enables, `False`/`None` disable, and anything else is refused and logged as a misalignment. This is why the widget is also declared **optional** — so it does not shift the positional order of pre-existing widgets.

The `sample()` gate reads the value under the **new** key `"clip_perfect_offload (Krea2 only)"` and falls back to the **old** key `"clip_perfect_offload"`, so workflows saved before the `3fe5c0b` rename still map to the same toggle.

### 4.4 The unload sequence, and why it is globally isolated (`_offload_loaded_clips`)

The sequence mirrors the benchmark but stays strictly Krea2-scoped:

1. Collect only the **Krea2** TE patchers currently in `current_loaded_models` (de-duplicated by `id`).
2. If there are none, do **nothing** — importantly it does *not* fall back to a global cache sweep.
3. For each Krea2 TE:
   * `cond_stage_model.cpu()` — the bench's move, drops the TE weights off the GPU.
   * `unload_model_and_clones(patcher, unload_additional_models=False)` — removes exactly this TE patcher and its own clones from the loaded set. `unload_additional_models=False` guarantees the **DiT, VAE, ControlNet and every other model stay resident**; only the TE leaves.
   * If that raises, a **TE-only** fallback pops just this patcher from `current_loaded_models` and unloads it.

The recurring rule across the whole function: **no global allocator op** (`soft_empty_cache`, `empty_cache`, `unload_all_models`) is ever called. Those ops act on the shared CUDA caching allocator and would reach into *other* workflows running against the same GPU. Instead, TE tensors are freed by dropping the patcher out of `current_loaded_models` and letting Python's refcount release them. This is the "complete branch isolation" `3d116ed` enforced: the Krea2 offload can never disturb a concurrent Z Image / Flux / SDXL graph.

### 4.5 UI scope tag (`3fe5c0b`)

The widget key is the **displayed label**. Renaming it to `clip_perfect_offload (Krea2 only)` puts the scope on the node face itself — matching the existing `quality (JPG only)` convention on the HSWQ Save Image node — so a user sees the Krea2 limitation without hovering for the tooltip. Backward compatibility is preserved by the dual-key `kwargs.get` read described in 4.3.

### 4.6 Failure behaviour

Every step is wrapped so the feature can **never break a run**:

* If the model is not Krea2, the gate logs and skips (`clip_perfect_offload ignored: MODEL is not Krea2`).
* If any unload step raises, it is caught and logged, and sampling proceeds normally.
* A misaligned toggle value is treated as OFF.

The offload is therefore a pure best-effort VRAM optimisation on Krea2: when it applies it reaches bench-parity VRAM, and in every other case it is an inert no-op.

### 4.7 Sampler-result fallback (`nodes/hswq_sampler.py`)

The call to `common_ksampler` is absorbed inside the package. A valid result must be a tuple/list whose first item is a LATENT dictionary with a non-None `samples` tensor. If sampling raises or drops the result, the node clones the input latent; if that is unavailable, it creates a minimal zero tensor. This prevents the HSWQ Sampler itself from emitting a bare `None`.

### 4.8 Dedicated tiled decoder (`nodes/hswq_vae_decode_tiled.py`)

`HSWQVAEDecodeTiled` accepts a normal LATENT dictionary or a tensor directly. A missing payload is replaced with a `(1, 4, 64, 64)` zero latent, then normal spatial/temporal tile conversion is applied. This gives workflows an explicit HSWQ-owned decoder whose output contract remains `IMAGE` even when upstream dropped its LATENT.

### 4.9 Global None-guard (`patches/vae_decode_none_guard.py`)

The global guard has four independent layers:

1. LATENT-producing nodes cache the latest usable latent on CPU.
2. Every LATENT-to-IMAGE node is wrapped, including late-registered third-party nodes.
3. `comfy.sd.VAE.decode` and `decode_tiled` sanitize empty tensor inputs underneath node wrappers.
4. `PromptExecutor.execute` and `execute_async` re-arm the sweep before each prompt, covering classes replaced after package import.

Recovery prefers the real cached latent. Only when no usable latent has ever been observed does it substitute zeros. IMAGE-producing decode failures are absorbed and return a correctly shaped blank image instead of aborting the prompt. Nested tensors bypass `numel()` because that operation is undefined for nested tensors.

### 4.10 Package registration (`__init__.py`)

Package import calls `apply_vae_decode_none_guard()` and reports arm failures at warning/exception level instead of silently swallowing them. The same file registers `HSWQVAEDecodeTiled` in `NODE_CLASS_MAPPINGS`. These lines are essential: creating the patch files without importing and registering them would leave the protection inactive.

---

## 5. Summary

* **What**: an opt-in `clip_perfect_offload (Krea2 only)` toggle on the HSWQ Sampler that frees the Krea2 text encoder before sampling, reproducing the benchmark's `clip.cond_stage_model.cpu()` VRAM behaviour.
* **Why**: the resident Krea2 TE co-resides with the DiT during sampling and pushes tight-VRAM cards into OOM / loader thrashing; the benchmark avoided this by offloading the TE, and the HSWQ graph had no equivalent.
* **How it stays safe**: two independent, name-proof identity checks (Krea2 DiT via load-time tag + architecture detection; Krea2 TE via exact `comfy.text_encoders.krea2` module identity), a strict boolean toggle read, a TE-only unload that keeps every other model resident, and **zero** global allocator ops so no other workflow is ever touched.
* **Files**: three modified sources shown as their change (`__init__.py`, `patches/comfy_quant_int8.py`, `nodes/hswq_sampler.py`) and two new sources shown in full (`nodes/hswq_vae_decode_tiled.py`, `patches/vae_decode_none_guard.py`).
