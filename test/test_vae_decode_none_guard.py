"""Check that the LATENT None-guard recovers the real latent, not just zeros."""

import logging
import os
import sys
import types

import torch

logging.basicConfig(level=logging.INFO)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class FakeVAE:
    latent_channels = 16
    upscale_ratio = 8

    def spacial_compression_decode(self):
        return 8

    def temporal_compression_decode(self):
        return None

    def decode_tiled(self, latent, **kwargs):
        return torch.zeros(
            (latent.shape[0], latent.shape[-2] * 8, latent.shape[-1] * 8, 3)
        )


class VAEDecodeTiled:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "vae": ("VAE",)}}

    def decode(self, vae, samples, tile_size=512):
        compression = vae.spacial_compression_decode()
        return (vae.decode_tiled(samples["samples"], tile_x=tile_size // compression),)


class FakeSampler:
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent_image": ("LATENT",)}}

    def sample(self, latent_image):
        return ({"samples": torch.ones((1, 16, 96, 64))},)


class LateDecoder(VAEDecodeTiled):
    """A pack that registers itself after the first sweep."""


fake_nodes = types.ModuleType("nodes")
fake_nodes.VAEDecodeTiled = VAEDecodeTiled
fake_nodes.NODE_CLASS_MAPPINGS = {
    "VAEDecodeTiled": VAEDecodeTiled,
    "FakeSampler": FakeSampler,
}
sys.modules["nodes"] = fake_nodes

from patches.vae_decode_none_guard import apply_vae_decode_none_guard  # noqa: E402

assert apply_vae_decode_none_guard(deep=True), "guard did not patch anything"

vae = FakeVAE()
decoder = VAEDecodeTiled()

# 1. sampler output is cached
FakeSampler().sample(latent_image={"samples": torch.zeros((1, 16, 96, 64))})

# 2. samples=None must decode the cached latent (96x64 -> 768x512)
img = decoder.decode(vae=vae, samples=None)[0]
print("None ->", tuple(img.shape))
assert tuple(img.shape) == (1, 768, 512, 3), img.shape

# 3. dict holding None must also recover
img = decoder.decode(vae=vae, samples={"samples": None})[0]
print("{'samples': None} ->", tuple(img.shape))
assert tuple(img.shape) == (1, 768, 512, 3), img.shape

# 4. a healthy latent passes through untouched
img = decoder.decode(vae=vae, samples={"samples": torch.zeros((1, 16, 32, 32))})[0]
print("healthy ->", tuple(img.shape))
assert tuple(img.shape) == (1, 256, 256, 3), img.shape

# 5. vae=None must not raise; blank IMAGE instead
img = decoder.decode(vae=None, samples=None)[0]
print("vae=None ->", tuple(img.shape))
assert img.ndim == 4 and img.shape[-1] == 3, img.shape

# 6. positional call path (cache now holds the 32x32 latent from step 4)
img = decoder.decode(vae, None)[0]
print("positional ->", tuple(img.shape))
assert tuple(img.shape) == (1, 256, 256, 3), img.shape

# 7. a node registered after the first sweep is still wrapped
fake_nodes.NODE_CLASS_MAPPINGS["LateDecoder"] = LateDecoder
apply_vae_decode_none_guard(deep=True)
img = LateDecoder().decode(vae=vae, samples=None)[0]
print("late-registered ->", tuple(img.shape))
assert tuple(img.shape) == (1, 256, 256, 3), img.shape

print("ALL OK")
