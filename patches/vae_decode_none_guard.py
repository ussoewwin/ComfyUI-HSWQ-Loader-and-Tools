"""Kill every ``None`` LATENT before it reaches ComfyUI's VAE decode nodes.

ComfyUI's stock ``VAEDecode`` / ``VAEDecodeTiled`` index ``samples["samples"]``
directly, so a dropped LATENT (``samples`` is ``None``, or the dict holds
``None``) raises ``TypeError: 'NoneType' object is not subscriptable`` and the
whole prompt dies after sampling already finished.

This module patches those nodes, and ``comfy.sd.VAE.decode`` /
``decode_tiled`` underneath them, so the failure is absorbed here: a blank
IMAGE is returned and the graph completes. Nothing is re-raised.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

logger = logging.getLogger(__name__)

_APPLIED = False


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
    if latent is not None and hasattr(latent, "shape") and len(latent.shape) >= 2:
        scale = _spatial_scale(vae)
        height = max(8, int(latent.shape[-2]) * scale)
        width = max(8, int(latent.shape[-1]) * scale)
    return torch.zeros((1, height, width, 3), dtype=torch.float32)


def _extract_latent(samples):
    """Return the latent tensor inside a LATENT input, or ``None``."""
    import torch

    if isinstance(samples, Mapping):
        latent = samples.get("samples")
    else:
        latent = samples

    if isinstance(latent, torch.Tensor) and latent.numel() > 0:
        return latent
    return None


def _sanitize(samples, vae, where: str):
    """Always return a LATENT dict whose ``samples`` is a real tensor."""
    latent = _extract_latent(samples)
    if latent is not None:
        if isinstance(samples, Mapping):
            return samples
        return {"samples": latent}

    logger.warning(
        "[HSWQ None-guard] %s received an empty LATENT (%s); substituting a "
        "zero latent so the prompt finishes (image will be blank).",
        where,
        type(samples).__name__,
    )
    return {"samples": _blank_latent(vae)}


def _patch_node(node_cls, name: str, wrap):
    if node_cls is None:
        return False
    if getattr(node_cls, "_hswq_none_guard", False) is True:
        return False
    original = getattr(node_cls, "decode", None)
    if original is None:
        return False
    node_cls.decode = wrap(original)
    node_cls._hswq_none_guard = True
    logger.debug("[HSWQ None-guard] patched %s.decode", name)
    return True


def _wrap_vae_decode(original):
    def decode(self, vae, samples, *args, **kwargs):
        samples = _sanitize(samples, vae, "VAEDecode")
        try:
            return original(self, vae, samples, *args, **kwargs)
        except Exception:
            logger.exception(
                "[HSWQ None-guard] VAEDecode failed; returning blank IMAGE "
                "instead of aborting the prompt"
            )
            return (_blank_image(vae, samples.get("samples")),)

    return decode


def _wrap_vae_decode_tiled(original):
    def decode(self, vae, samples, *args, **kwargs):
        samples = _sanitize(samples, vae, "VAEDecodeTiled")
        try:
            return original(self, vae, samples, *args, **kwargs)
        except Exception:
            logger.exception(
                "[HSWQ None-guard] VAEDecodeTiled failed; returning blank "
                "IMAGE instead of aborting the prompt"
            )
            return (_blank_image(vae, samples.get("samples")),)

    return decode


def _patch_vae_class() -> bool:
    """Guard ``comfy.sd.VAE.decode`` / ``decode_tiled`` against ``None``."""
    try:
        import comfy.sd as comfy_sd
    except Exception as e:
        logger.debug("[HSWQ None-guard] comfy.sd import failed: %s", e)
        return False

    vae_cls = getattr(comfy_sd, "VAE", None)
    if vae_cls is None or getattr(vae_cls, "_hswq_none_guard", False) is True:
        return False

    patched = []

    for method_name in ("decode", "decode_tiled"):
        original = getattr(vae_cls, method_name, None)
        if original is None:
            continue

        def make(original=original, method_name=method_name):
            def guarded(self, samples_in, *args, **kwargs):
                import torch

                if not isinstance(samples_in, torch.Tensor) or samples_in.numel() == 0:
                    extracted = _extract_latent(samples_in)
                    if extracted is None:
                        logger.warning(
                            "[HSWQ None-guard] VAE.%s got an empty latent; "
                            "substituting zeros.",
                            method_name,
                        )
                        extracted = _blank_latent(self)
                    samples_in = extracted
                return original(self, samples_in, *args, **kwargs)

            return guarded

        setattr(vae_cls, method_name, make())
        patched.append(method_name)

    if not patched:
        return False

    vae_cls._hswq_none_guard = True
    logger.debug("[HSWQ None-guard] patched comfy.sd.VAE: %s", ", ".join(patched))
    return True


def apply_vae_decode_none_guard() -> bool:
    """Patch stock VAE decode paths. Safe to call repeatedly."""
    global _APPLIED
    if _APPLIED:
        return True

    try:
        import nodes as comfy_nodes
    except Exception as e:
        logger.debug("[HSWQ None-guard] nodes import failed: %s", e)
        return False

    patched = []
    if _patch_node(getattr(comfy_nodes, "VAEDecode", None), "VAEDecode", _wrap_vae_decode):
        patched.append("VAEDecode")
    if _patch_node(
        getattr(comfy_nodes, "VAEDecodeTiled", None),
        "VAEDecodeTiled",
        _wrap_vae_decode_tiled,
    ):
        patched.append("VAEDecodeTiled")
    if _patch_vae_class():
        patched.append("comfy.sd.VAE")

    if not patched:
        logger.debug("[HSWQ None-guard] nothing to patch")
        return False

    _APPLIED = True
    logger.info("[HSWQ] LATENT None-guard applied: %s", ", ".join(patched))
    return True
