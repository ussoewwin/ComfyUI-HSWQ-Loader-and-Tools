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

import comfy.samplers
import comfy.k_diffusion.sampling as _k_diff
import nodes as _nodes

logger = logging.getLogger(__name__)


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
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    TITLE = "HSWQ Sampler"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler,
               positive, negative, latent_image, denoise=1.0):
        return _nodes.common_ksampler(
            model, seed, steps, cfg,
            sampler_name, scheduler,
            positive, negative, latent_image,
            denoise=denoise,
        )
