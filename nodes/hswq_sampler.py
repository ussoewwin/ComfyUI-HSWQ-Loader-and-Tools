"""
HSWQ Sampler — A node fully equivalent to the standard ComfyUI KSampler.
If RES4LYF (custom_nodes/RES4LYF) is loaded,
it automatically adds all of its samplers / schedulers.

## Reason for bridging the gap with Forge
Forge's modules/RES4LYF/beta/__init__.py dynamically generates wrappers
for rk_sampler_beta.sample_rk_beta for all entries in RK_SAMPLER_NAMES_BETA_NO_FOLDERS
and adds them to extra_samplers.
The ComfyUI version of beta/__init__.py does not have this logic.
This node supplements that missing difference.
"""
import sys
import logging

import comfy.model_management as _mm
import comfy.samplers
import comfy.k_diffusion.sampling as _k_diff
import nodes as _nodes

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────
# CLIP / Text Encoder offload
# ────────────────────────────────────────────────

def _obj_tags(obj) -> str:
    """Lower-cased 'module.ClassName' for an instance or a class."""
    if obj is None:
        return ""
    cls = obj if isinstance(obj, type) else type(obj)
    return f"{getattr(cls, '__module__', '')}.{getattr(cls, '__qualname__', '')}".lower()


def _looks_krea2(obj) -> bool:
    return "krea2" in _obj_tags(obj)


def _is_krea2_diffusion_model(model) -> bool:
    """
    True only when the MODEL input is a Krea2 diffusion model.

    Every other architecture (Z Image / Lumina2, Flux, SDXL, Qwen, WAN, ...)
    must return False so the offload path is never entered for them.
    """
    if model is None:
        return False

    inner = getattr(model, "model", None)          # BaseModel
    if _looks_krea2(model) or _looks_krea2(inner):
        return True

    model_config = getattr(inner, "model_config", None)
    if _looks_krea2(model_config):
        return True

    unet_config = getattr(model_config, "unet_config", None)
    if isinstance(unet_config, dict):
        for key in ("image_model", "model_type", "arch"):
            value = unet_config.get(key)
            if isinstance(value, str) and "krea2" in value.lower():
                return True

    if _looks_krea2(getattr(inner, "diffusion_model", None)):
        return True

    return False


def _is_krea2_text_encoder(patcher) -> bool:
    """
    True only for a Krea2 text encoder.

    is_clip alone is not enough: Z Image (ZImageTEModel_), Flux, SDXL and every
    other CLIP wrapper also carry it. Unloading those breaks unrelated
    workflows, so the Krea2 class/module tag is required on top of is_clip.
    """
    if getattr(patcher, "is_clip", False) is not True:
        return False

    if _looks_krea2(patcher):
        return True

    real = getattr(patcher, "model", None)         # cond_stage_model
    if _looks_krea2(real):
        return True

    for attr in ("clip_model", "transformer", "text_model"):
        if _looks_krea2(getattr(real, attr, None)):
            return True

    return False


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
    Free text-encoder VRAM only — same shape as krea2_int8_bench TE offload.

    Bench sequence (before DiT sample):
      cond_stage_model.cpu()
      unload_model_and_clones(clip.patcher)
      soft path via free_memory(..., keep_loaded=non-TE)

    Leaving TE weights only via model_unload+pop keeps ~3 GiB in the CUDA
    caching allocator (reserved). unload_model_and_clones keeps DiT / VAE /
    ControlNet in keep_loaded and soft_empty_caches after the TE drop — that
    is the ~3 GiB gap vs the bench peak. Hard empty_cache / unload_all_models
    are NOT used (they rebuild DiT cudaMalloc and were measured slower).

    Only a Krea2 text encoder is a candidate. The caller must already have
    confirmed the MODEL input is a Krea2 diffusion model.
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
        # Encode may have left reserved blocks with TE already absent from the
        # loaded list (MultiGPU CPU TE). Reclaim for DiT without touching models.
        try:
            _mm.soft_empty_cache()
        except Exception:
            pass
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

        # Fallback: TE-only model_unload + pop, then soft_empty_cache (no hard empty_cache).
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
        try:
            _mm.soft_empty_cache()
        except Exception:
            pass

    if unloaded:
        logger.info(
            "[HSWQSampler] Offloaded %d text encoder(s) (bench-parity TE free)",
            unloaded,
        )
    return unloaded


# ────────────────────────────────────────────────
# RES4LYF Module Discovery
# ────────────────────────────────────────────────

def _find_res4lyf_mod():
    """Find the RES4LYF module containing extra_samplers from sys.modules."""
    for cand in ("RES4LYF", "custom_nodes.RES4LYF"):
        m = sys.modules.get(cand)
        if m is not None and hasattr(m, "extra_samplers"):
            return m
    for name, m in list(sys.modules.items()):
        if m is not None and "RES4LYF" in name and hasattr(m, "extra_samplers"):
            return m
    return None


def _find_rk_sampler_beta_mod():
    """Find the module where sample_rk_beta can be retrieved for comfy.k_diffusion.sampling."""
    # RES4LYF.beta.rk_sampler_beta can be registered under multiple names
    for cand in (
        "RES4LYF.beta.rk_sampler_beta",
        "custom_nodes.RES4LYF.beta.rk_sampler_beta",
        "beta.rk_sampler_beta",
    ):
        m = sys.modules.get(cand)
        if m is not None and hasattr(m, "sample_rk_beta"):
            return m
    # Fallback: scan submodules of the RES4LYF module
    for name, m in list(sys.modules.items()):
        if m is not None and "rk_sampler_beta" in name and hasattr(m, "sample_rk_beta"):
            return m
    return None


def _find_rk_coefficients_mod():
    """Find the module containing RK_SAMPLER_NAMES_BETA_NO_FOLDERS."""
    for cand in (
        "RES4LYF.beta.rk_coefficients_beta",
        "custom_nodes.RES4LYF.beta.rk_coefficients_beta",
        "beta.rk_coefficients_beta",
    ):
        m = sys.modules.get(cand)
        if m is not None and hasattr(m, "RK_SAMPLER_NAMES_BETA_NO_FOLDERS"):
            return m
    for name, m in list(sys.modules.items()):
        if m is not None and "rk_coefficients_beta" in name and hasattr(m, "RK_SAMPLER_NAMES_BETA_NO_FOLDERS"):
            return m
    return None


# ────────────────────────────────────────────────
# Forge Compatibility: Generate and register wrappers for all rk_types
# ────────────────────────────────────────────────

# Do not create ODE versions for implicit samplers (same condition as Forge)
_IMPLICIT_KEYWORDS = (
    "gauss-legendre", "radau", "lobatto",
    "irk_exp_diag", "kraaijevanger", "qin_zhang",
    "pareschi", "crouzeix",
)


def _build_rk_extra_samplers(rk_mod, names) -> dict:
    """
    Identical logic to Forge's beta/__init__.py L92-L119.
    Generates sample_fn / sample_ode_fn closures for all entries in
    RK_SAMPLER_NAMES_BETA_NO_FOLDERS.
    """
    result = {}

    for sampler_name in names:
        if sampler_name == "none":
            continue

        def make_fn(rk_type):
            def sample_fn(model, x, sigmas, extra_args=None, callback=None, disable=None):
                return rk_mod.sample_rk_beta(
                    model, x, sigmas, None, extra_args, callback, disable,
                    rk_type=rk_type,
                )
            sample_fn.__name__ = f"sample_{rk_type}"
            return sample_fn

        result[sampler_name] = make_fn(sampler_name)

        # ODE versions (excluding implicit types)
        if not any(kw in sampler_name for kw in _IMPLICIT_KEYWORDS):
            ode_name = f"{sampler_name}_ode"

            def make_ode_fn(rk_type):
                def sample_ode_fn(model, x, sigmas, extra_args=None, callback=None, disable=None):
                    return rk_mod.sample_rk_beta(
                        model, x, sigmas, None, extra_args, callback, disable,
                        rk_type=rk_type, eta=0.0, eta_substep=0.0,
                    )
                sample_ode_fn.__name__ = f"sample_{rk_type}_ode"
                return sample_ode_fn

            result[ode_name] = make_ode_fn(sampler_name)

    # generic rk_beta
    result["rk_beta"] = rk_mod.sample_rk_beta

    return result


def _ensure_all_registered(extra: dict) -> None:
    """
    Registers all entries in extra_samplers to KSampler.SAMPLERS and
    comfy.k_diffusion.sampling.
    """
    samplers_list = comfy.samplers.KSampler.SAMPLERS
    insert_after = "uni_pc_bh2"
    try:
        insert_idx = samplers_list.index(insert_after)
    except ValueError:
        insert_idx = len(samplers_list) - 1

    added = 0
    for name, fn in extra.items():
        # Add to KSampler.SAMPLERS
        if name not in samplers_list:
            samplers_list.insert(insert_idx + 1, name)
            insert_idx += 1
            added += 1

        # Inject function into comfy.k_diffusion.sampling (supplements missing functions from reload)
        attr = f"sample_{name}"
        if not hasattr(_k_diff, attr):
            setattr(_k_diff, attr, fn)

    if added:
        logger.info("[HSWQSampler] Registered %d RES4LYF samplers into KSampler.SAMPLERS", added)


# ────────────────────────────────────────────────
# INPUT_TYPES Helpers
# ────────────────────────────────────────────────

def _get_samplers() -> list:
    res4lyf  = _find_res4lyf_mod()
    rk_mod   = _find_rk_sampler_beta_mod()
    coef_mod = _find_rk_coefficients_mod()

    if res4lyf is not None and rk_mod is not None and coef_mod is not None:
        names = getattr(coef_mod, "RK_SAMPLER_NAMES_BETA_NO_FOLDERS", [])
        # Generate wrappers for all rk_types identically to Forge
        rk_extra = _build_rk_extra_samplers(rk_mod, names)
        # Merge with existing extra_samplers
        extra = dict(getattr(res4lyf, "extra_samplers", {}))
        extra.update(rk_extra)
        _ensure_all_registered(extra)
    elif res4lyf is not None:
        extra = getattr(res4lyf, "extra_samplers", {})
        _ensure_all_registered(extra)

    return list(comfy.samplers.KSampler.SAMPLERS)


def _get_schedulers() -> list:
    handlers: dict = getattr(comfy.samplers, "SCHEDULER_HANDLERS", {})
    names: list = list(comfy.samplers.KSampler.SCHEDULERS)
    for name in handlers:
        if name not in names:
            names.append(name)
    return names


# ────────────────────────────────────────────────
# Node Main Class
# ────────────────────────────────────────────────

class HSWQSampler:
    @classmethod
    def INPUT_TYPES(cls):
        samplers   = _get_samplers()
        schedulers = _get_schedulers()
        logger.debug(
            "[HSWQSampler] INPUT_TYPES: %d samplers, %d schedulers",
            len(samplers), len(schedulers),
        )
        return {
            "required": {
                "model":        ("MODEL",),
                "seed":         ("INT",   {"default": 0,   "min": 0,   "max": 0xffffffffffffffff}),
                "steps":        ("INT",   {"default": 20,  "min": 1,   "max": 10000}),
                "cfg":          ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (samplers,),
                "scheduler":    (schedulers,),
                "positive":     ("CONDITIONING",),
                "negative":     ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise":      ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                # Optional so workflows saved before this widget existed keep their
                # own widget order instead of shifting a neighbouring value onto it.
                "clip_perfect_offload": ("BOOLEAN", {
                    "default": False,
                    "label_on": "enabled (Krea2 only)",
                    "label_off": "disabled (Krea2 only)",
                    "tooltip": "Krea2 only. Frees the Krea2 text encoder before sampling. "
                               "Ignored for every other architecture.",
                }),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    TITLE = "HSWQ Sampler"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler,
               positive, negative, latent_image, denoise=1.0,
               clip_perfect_offload=False):
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

        out = _nodes.common_ksampler(
            model, seed, steps, cfg,
            sampler_name, scheduler,
            positive, negative, latent_image,
            denoise=denoise,
        )

        # common_ksampler must always hand back (latent_dict,). Anything else means
        # an upstream patch swallowed the result, and letting it through only
        # surfaces as an opaque NoneType error inside VAEDecode.
        if (
            not out
            or out[0] is None
            or "samples" not in out[0]
            or out[0]["samples"] is None
        ):
            raise RuntimeError(
                "[HSWQSampler] sampling returned no latent "
                f"(got {type(out[0]).__name__ if out else 'None'}). "
                "A model patch or custom sampler dropped the result."
            )
        return out
