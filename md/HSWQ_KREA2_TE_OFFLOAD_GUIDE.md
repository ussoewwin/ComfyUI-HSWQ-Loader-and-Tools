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

> `9a80737` itself is the pre-feature baseline (a README / screenshot refresh). It only fixes the anchor for the diff; it contains none of the offload code.
>
> A sixth commit on the same file, `4769304` (`fix hswq_sampler never raise return fallback LATENT`), is the downstream **None-guard** and is **not** part of the TE-offload feature. It is mentioned here only because it lives in the same `sample()` method; its behaviour is documented separately.

This manual answers four required sections:

1. **Why the feature was necessary**
2. **Files created or modified**
3. **Full source** of the created or modified code
4. **Meaning** of that code

Style matches the other public manuals under `md/`.

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

---

## 2. Files created or modified

No new files were created for this feature. Two existing files were modified.

| File | Role in this feature | Introduced by |
|------|----------------------|---------------|
| `nodes/hswq_sampler.py` | The offload logic and the UI toggle live here: Krea2 model check, Krea2 TE check, strict toggle read, the TE-only unload sequence, and the `sample()` gate that calls it. | `b135601`, `a1812c3`, `3c25505`, `3d116ed`, `3fe5c0b` |
| `patches/comfy_quant_int8.py` | The HSWQ loader stamps a **load-time Krea2 tag** (`_hswq_is_krea2`) onto the model so the sampler node reads a verdict instead of guessing. Adds `KREA2_MODEL_FLAG`, `model_is_krea2()`, `tag_krea2_model()`, and the call at load time. | `3c25505` |

---

## 3. Full source of the created or modified code

### 3.1 `nodes/hswq_sampler.py` — offload section (current)

This is the complete text-encoder offload section as it stands after `3fe5c0b`.

```python
# ────────────────────────────────────────────────
# CLIP / Text Encoder offload
# ────────────────────────────────────────────────

try:
    from ..patches.comfy_quant_int8 import KREA2_MODEL_FLAG
except Exception:  # loader patches unavailable — the config checks still work
    KREA2_MODEL_FLAG = "_hswq_is_krea2"

# comfy/text_encoders/krea2.py — the only module that defines a Krea2 text
# encoder. Z Image lives in comfy.text_encoders.z_image, Flux in flux, and so on.
_KREA2_TE_MODULE = "comfy.text_encoders.krea2"


def _obj_tags(obj) -> str:
    """'module.ClassName' for an instance or a class (logging only)."""
    if obj is None:
        return ""
    cls = obj if isinstance(obj, type) else type(obj)
    return f"{getattr(cls, '__module__', '')}.{getattr(cls, '__qualname__', '')}"


def _obj_module(obj) -> str:
    if obj is None:
        return ""
    cls = obj if isinstance(obj, type) else type(obj)
    return getattr(cls, "__module__", "") or ""


def _class_name(obj) -> str:
    if obj is None:
        return ""
    cls = obj if isinstance(obj, type) else type(obj)
    return getattr(cls, "__name__", "") or ""


def _is_krea2_diffusion_model(model) -> bool:
    """
    True only when the MODEL input is a Krea2 diffusion model.

    The verdict comes from the tag the HSWQ loader stamps at load time, or from
    ComfyUI's own architecture detection (``unet_config["image_model"]`` written
    by ``model_detection``, and the ``supported_models.Krea2`` /
    ``model_base.Krea2`` identities). No substring matching on class or file
    names, so a rename or a lookalike name cannot flip the answer.

    Every other architecture (Z Image / Lumina2, Flux, SDXL, Qwen, WAN, ...)
    returns False, so the offload path is never entered for them.
    """
    if model is None:
        return False

    inner = getattr(model, "model", None)          # BaseModel
    if getattr(model, KREA2_MODEL_FLAG, False) is True:
        return True
    if getattr(inner, KREA2_MODEL_FLAG, False) is True:
        return True

    config = getattr(inner, "model_config", None)
    unet_config = getattr(config, "unet_config", None)
    if isinstance(unet_config, dict) and str(unet_config.get("image_model", "")).lower() == "krea2":
        return True

    return _class_name(config) == "Krea2" or _class_name(inner) == "Krea2"


def _is_krea2_text_encoder(patcher) -> bool:
    """
    True only for a Krea2 text encoder.

    is_clip alone is not enough: Z Image (ZImageTEModel_), Flux, SDXL and every
    other CLIP wrapper also carry it, and unloading those breaks unrelated
    workflows. The extra condition is that the encoder object is defined in
    ComfyUI's Krea2 text-encoder module — an exact module identity, not a name
    that happens to contain "krea2".
    """
    if getattr(patcher, "is_clip", False) is not True:
        return False

    real = getattr(patcher, "model", None)         # cond_stage_model
    candidates = (
        patcher,
        real,
        getattr(real, "clip_model", None),
        getattr(real, "transformer", None),
        getattr(real, "text_model", None),
    )
    return any(_obj_module(obj) == _KREA2_TE_MODULE for obj in candidates)


def _offload_requested(value) -> bool:
    """
    Strict toggle read. Only a real True enables the offload.

    A plain truthiness test is not safe here: when an older saved workflow has a
    shorter widgets_values array, the frontend fills this widget positionally and
    a neighbouring value (for example denoise = 1.0) can land on it. That reads as
    truthy and fires the offload while the UI still shows the toggle as off.
    Anything that is not a boolean is refused and logged.
    """
    if value is True or value is False:
        return value is True
    if value is None:
        return False

    logger.warning(
        "[HSWQSampler] clip_perfect_offload got a non-boolean value (%r, %s); "
        "treating it as OFF. The saved workflow's widget values are misaligned — "
        "re-add the node to clear it.",
        value, type(value).__name__,
    )
    return False


def _offload_loaded_clips() -> int:
    """
    Free text-encoder VRAM only — Krea2 TE offload, fully Krea2-scoped.

    Sequence (only when MODEL is Krea2 AND a Krea2 TE is in current_loaded_models):
      cond_stage_model.cpu()
      unload_model_and_clones(clip.patcher, unload_additional_models=False)

    ``unload_additional_models=False`` keeps DiT / VAE / ControlNet / every
    non-Krea2 model in ``keep_loaded``. No ``soft_empty_cache`` /
    ``empty_cache`` / ``unload_all_models`` is ever called here: those are
    global allocator ops and would reach into unrelated workflows sharing the
    CUDA caching allocator. TE tensors are released by popping the patcher
    from ``current_loaded_models`` (Python refcount), not by a global sweep.

    Only a Krea2 text encoder (``comfy.text_encoders.krea2`` module identity)
    is ever a candidate. Z Image / Flux / SDXL / WAN TEs never match.
    """
    try:
        loaded_models = _mm.current_loaded_models
    except Exception:
        return 0

    te_patchers = []
    seen = set()
    for loaded in list(loaded_models):
        patcher = getattr(loaded, "model", None)
        if patcher is None or not _is_krea2_text_encoder(patcher):
            continue
        pid = id(patcher)
        if pid in seen:
            continue
        seen.add(pid)
        te_patchers.append(patcher)

    if not te_patchers:
        # No Krea2 TE in the loaded list. Do NOT touch global CUDA cache here:
        # soft_empty_cache() is a global op and would reach into non-Krea2
        # workflows (Z Image / Flux / SDXL) that share the allocator. Krea2-only
        # branch means: nothing to unload -> nothing to do.
        logger.debug("[HSWQSampler] No Krea2 text encoder found in loaded models; offload is a no-op")
        return 0

    unloaded = 0
    for patcher in te_patchers:
        # Bench: clip.cond_stage_model.cpu()
        real = getattr(patcher, "model", None)
        if real is not None:
            try:
                real.cpu()
            except Exception:
                logger.exception("[HSWQSampler] TE .cpu() failed")

        try:
            # Keeps every other LoadedModel. unload_additional_models=False so the
            # free set is exactly this Krea2 TE patcher and its own clones —
            # nested additional models attached to it are never dragged out.
            _mm.unload_model_and_clones(patcher, unload_additional_models=False)
            unloaded += 1
            continue
        except Exception:
            logger.exception(
                "[HSWQSampler] unload_model_and_clones TE failed; fallback unload"
            )

        # Fallback: TE-only model_unload + pop. No soft_empty_cache here either:
        # once the TE patcher is popped from current_loaded_models its tensors
        # are freed by Python's refcount, and a global cache sweep would again
        # touch unrelated workflows sharing the CUDA allocator.
        for i in range(len(loaded_models) - 1, -1, -1):
            try:
                loaded = loaded_models[i]
                if loaded.model is not patcher:
                    continue
                if loaded.model_unload(unpatch_weights=True):
                    loaded_models.pop(i)
                    unloaded += 1
            except Exception:
                logger.exception("[HSWQSampler] TE fallback unload skipped")

    if unloaded:
        logger.info(
            "[HSWQSampler] Offloaded %d text encoder(s) (bench-parity TE free)",
            unloaded,
        )
    return unloaded
```

### 3.2 `nodes/hswq_sampler.py` — the widget and the `sample()` gate (current)

The optional input widget in `INPUT_TYPES`:

```python
            "optional": {
                # Optional so workflows saved before this widget existed keep their
                # own widget order instead of shifting a neighbouring value onto it.
                # Label matches HSWQ Save Image's "quality (JPG only)" pattern so the
                # scope tag is visible on the node, not only in the tooltip.
                "clip_perfect_offload (Krea2 only)": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Krea2 only. Frees the Krea2 text encoder before sampling. "
                               "Ignored for every other architecture.",
                }),
            },
```

The offload gate at the top of `sample()` (the fallback-LATENT block below it belongs to the separate None-guard, `4769304`):

```python
    def sample(self, model, seed, steps, cfg, sampler_name, scheduler,
               positive, negative, latent_image, denoise=1.0, **kwargs):
        # New label name, plus the pre-rename key so older workflow JSON still maps.
        clip_perfect_offload = kwargs.get(
            "clip_perfect_offload (Krea2 only)",
            kwargs.get("clip_perfect_offload", False),
        )
        if _offload_requested(clip_perfect_offload):
            try:
                if _is_krea2_diffusion_model(model):
                    _offload_loaded_clips()
                else:
                    logger.info(
                        "[HSWQSampler] clip_perfect_offload ignored: MODEL is not Krea2 (%s)",
                        _obj_tags(getattr(model, "model", model)) or "unknown",
                    )
            except Exception:
                logger.exception("[HSWQSampler] CLIP offload failed; continuing")
```

### 3.3 `patches/comfy_quant_int8.py` — the load-time Krea2 tag (added in `3c25505`)

The flag name, the identity check, the stamping helper, and the call site:

```python
KREA2_MODEL_FLAG = "_hswq_is_krea2"


def model_is_krea2(model) -> bool:
    """Krea2 check taken from ComfyUI's own architecture detection.

    ``model_detection`` writes ``image_model = "krea2"`` into ``unet_config``
    from the state dict, and picks ``supported_models.Krea2`` /
    ``model_base.Krea2``. Those are exact identities, not name guesses, so a
    file rename or a substring collision cannot flip the answer.
    """
    if model is None:
        return False

    inner = getattr(model, "model", None) or model
    if getattr(model, KREA2_MODEL_FLAG, False) or getattr(inner, KREA2_MODEL_FLAG, False):
        return True

    config = getattr(inner, "model_config", None)
    unet_config = getattr(config, "unet_config", None)
    if isinstance(unet_config, dict) and str(unet_config.get("image_model", "")).lower() == "krea2":
        return True

    return type(config).__name__ == "Krea2" or type(inner).__name__ == "Krea2"


def tag_krea2_model(model) -> bool:
    """Stamp the Krea2 verdict onto the model so later nodes read, not guess.

    The flag goes on the inner model too because ``ModelPatcher`` clones
    re-wrap the same inner model and would otherwise drop it.
    """
    if not model_is_krea2(model):
        return False

    for obj in (model, getattr(model, "model", None)):
        if obj is None:
            continue
        try:
            setattr(obj, KREA2_MODEL_FLAG, True)
        except Exception:
            logger.debug("[HSWQ INT8] Could not tag %r as Krea2", type(obj).__name__)
    return True
```

The call site, at the end of the HSWQ INT8 UNet loader (after the model is built):

```python
    if tag_krea2_model(model):
        logging.info("[HSWQ INT8] Tagged as Krea2: %s", unet_name)

    return (model,)
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

---

## 5. Summary

* **What**: an opt-in `clip_perfect_offload (Krea2 only)` toggle on the HSWQ Sampler that frees the Krea2 text encoder before sampling, reproducing the benchmark's `clip.cond_stage_model.cpu()` VRAM behaviour.
* **Why**: the resident Krea2 TE co-resides with the DiT during sampling and pushes tight-VRAM cards into OOM / loader thrashing; the benchmark avoided this by offloading the TE, and the HSWQ graph had no equivalent.
* **How it stays safe**: two independent, name-proof identity checks (Krea2 DiT via load-time tag + architecture detection; Krea2 TE via exact `comfy.text_encoders.krea2` module identity), a strict boolean toggle read, a TE-only unload that keeps every other model resident, and **zero** global allocator ops so no other workflow is ever touched.
* **Files**: `nodes/hswq_sampler.py` (the toggle + logic) and `patches/comfy_quant_int8.py` (the load-time Krea2 tag). No new files.
