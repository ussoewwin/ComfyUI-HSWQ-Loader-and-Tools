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
