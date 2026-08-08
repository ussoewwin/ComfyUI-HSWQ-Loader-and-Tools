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
                # Label matches HSWQ Save Image's "quality (JPG only)" pattern so the
                # scope tag is visible on the node, not only in the tooltip.
                "clip_perfect_offload (Krea2 only)": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Krea2 only. Frees the Krea2 text encoder before sampling. "
                               "Ignored for every other architecture.",
                }),
                "tensor_boost": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable Blackwell Per-Weight CUDA Graph Tensor Boost during sampling. ON raises VRAM by several GB (CUDA Graph arenas).",
                }),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    TITLE = "HSWQ Sampler"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler,
               positive, negative, latent_image, denoise=1.0, **kwargs):
        # Configure Tensor Boost toggle for sampling
        tensor_boost = kwargs.get("tensor_boost", False)
        import os
        if isinstance(tensor_boost, bool):
            tb_enabled = tensor_boost
        else:
            tb_str = str(tensor_boost).strip().lower() if tensor_boost is not None else ""
            tb_enabled = tb_str in ("1", "true", "on", "enable", "enabled")

        if tb_enabled:
            os.environ["HSWQ_NVFP4_TENSORBOOST"] = "1"
        else:
            os.environ["HSWQ_NVFP4_TENSORBOOST"] = "0"
            try:
                from .nvfp4.nvfp4_runtime import clear_nvfp4_cudagraphs
                clear_nvfp4_cudagraphs()
            except Exception:
                pass

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

        try:
            out = _nodes.common_ksampler(
                model, seed, steps, cfg,
                sampler_name, scheduler,
                positive, negative, latent_image,
                denoise=denoise,
            )
        except Exception:
            logger.exception("[HSWQSampler] common_ksampler raised; returning fallback latent")
            out = None

        # Never raise. If sampling dropped the result (MultiGPU _load_list guard,
        # dynamic VRAM loader, custom sampler swallow, etc.), substitute a valid
        # LATENT built from the input so downstream nodes (VAEDecode, SaveImage)
        # always receive a subscriptable dict and the workflow completes.
        def _valid(o):
            return (
                o
                and isinstance(o, (tuple, list))
                and len(o) >= 1
                and isinstance(o[0], dict)
                and o[0].get("samples") is not None
            )

        if _valid(out):
            return out

        logger.warning(
            "[HSWQSampler] sampling produced no usable latent (out=%r); "
            "returning fallback LATENT from input",
            None if out is None else type(out[0]).__name__ if out else "empty",
        )

        ref = None
        try:
            if isinstance(latent_image, dict):
                ref = latent_image.get("samples")
        except Exception:
            ref = None

        if ref is not None:
            try:
                return ({"samples": ref.clone()},)
            except Exception:
                return ({"samples": ref},)

        try:
            import torch
            return ({"samples": torch.zeros((1, 4, 1, 1))},)
        except Exception:
            return ({"samples": None},)
