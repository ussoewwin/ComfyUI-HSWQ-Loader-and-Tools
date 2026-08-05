"""Kill every ``None`` LATENT before it reaches any VAE decode node.

ComfyUI's stock ``VAEDecode`` / ``VAEDecodeTiled`` index ``samples["samples"]``
directly, so a dropped LATENT (``samples`` is ``None``, or the dict holds
``None``) raises ``TypeError: 'NoneType' object is not subscriptable`` and the
whole prompt dies after sampling already finished.

The guard works on three layers so nothing slips through:

1. every registered node that takes a ``LATENT`` named ``samples`` and returns
   an ``IMAGE`` gets its entry point wrapped (stock and third-party alike),
2. ``comfy.sd.VAE.decode`` / ``decode_tiled`` reject empty tensors underneath,
3. the wrap is re-applied at the start of each prompt, so a pack that replaces
   the node class later still runs behind this guard.

Nothing is re-raised: a blank IMAGE is returned and the graph completes.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

logger = logging.getLogger(__name__)

_GUARD_MARK = "_hswq_none_guard"
_STOCK_NODES = ("VAEDecode", "VAEDecodeTiled")

# Full NODE_CLASS_MAPPINGS sweeps call INPUT_TYPES() on every node, which makes
# loaders scan disk. Sweep once, then only re-check the nodes we already wrapped.
_SWEPT = False
_PATCHED_NAMES: set[str] = set()


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


def _sanitize(samples, vae, where: str):
    """Always return a LATENT dict whose ``samples`` is a real tensor."""
    latent = _extract_latent(samples)
    if latent is not None:
        if isinstance(samples, Mapping) and _usable_tensor(samples.get("samples")):
            return samples
        if isinstance(samples, Mapping):
            fixed = dict(samples)
            fixed["samples"] = latent
            return fixed
        return {"samples": latent}

    logger.warning(
        "[HSWQ None-guard] %s received an empty LATENT (%s); substituting a "
        "zero latent so the prompt finishes (image will be blank).",
        where,
        type(samples).__name__,
    )
    return {"samples": _blank_latent(vae)}


def _wrap_decode(original, where: str, absorb_errors: bool):
    def guarded(self, *args, **kwargs):
        vae = kwargs.get("vae")
        if "samples" in kwargs:
            kwargs["samples"] = _sanitize(kwargs["samples"], vae, where)
            fixed_latent = kwargs["samples"].get("samples")
        elif args:
            # Positional fallback: stock order is (vae, samples, ...).
            args = list(args)
            index = 1 if len(args) > 1 else 0
            if index == 0:
                vae = None
            else:
                vae = args[0]
            args[index] = _sanitize(args[index], vae, where)
            fixed_latent = args[index].get("samples")
            args = tuple(args)
        else:
            fixed_latent = None

        try:
            return original(self, *args, **kwargs)
        except Exception:
            if not absorb_errors:
                raise
            logger.exception(
                "[HSWQ None-guard] %s failed; returning blank IMAGE instead of "
                "aborting the prompt",
                where,
            )
            return (_blank_image(vae, fixed_latent),)

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


def _returns_single_image(node_cls) -> bool:
    return tuple(getattr(node_cls, "RETURN_TYPES", ()) or ()) == ("IMAGE",)


def _patch_node(node_cls, name: str) -> bool:
    if node_cls is None or not isinstance(node_cls, type):
        return False
    func_name = getattr(node_cls, "FUNCTION", None)
    if not isinstance(func_name, str):
        return False
    original = node_cls.__dict__.get(func_name) or getattr(node_cls, func_name, None)
    if original is None or not callable(original):
        return False
    if getattr(original, _GUARD_MARK, False):
        return False
    setattr(
        node_cls,
        func_name,
        _wrap_decode(original, name, absorb_errors=_returns_single_image(node_cls)),
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
    original = getattr(executor, "execute", None)
    if original is None or getattr(original, _GUARD_MARK, False):
        return False

    def execute(self, *args, **kwargs):
        try:
            apply_vae_decode_none_guard(deep=True)
        except Exception:
            logger.debug("[HSWQ None-guard] re-apply skipped", exc_info=True)
        return original(self, *args, **kwargs)

    setattr(execute, _GUARD_MARK, True)
    executor.execute = execute
    return True


def apply_vae_decode_none_guard(deep: bool = False) -> bool:
    """Patch every LATENT->IMAGE decode path. Safe to call repeatedly.

    ``deep`` walks the whole node registry once; it is used from the prompt hook
    because at import time other packs may not be registered yet.
    """
    global _SWEPT

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

    full_sweep = deep and not _SWEPT
    if full_sweep:
        candidates = list(mappings.items())
    else:
        candidates = [(n, mappings.get(n)) for n in sorted(_PATCHED_NAMES)]

    for name, node_cls in candidates:
        try:
            if node_cls is None:
                continue
            if "IMAGE" not in tuple(getattr(node_cls, "RETURN_TYPES", ()) or ()):
                continue
            if full_sweep and not _takes_latent_samples(node_cls):
                continue
            if _patch_node(node_cls, str(name)):
                patched.append(str(name))
                _PATCHED_NAMES.add(str(name))
        except Exception:
            logger.debug("[HSWQ None-guard] skip %s", name, exc_info=True)

    if full_sweep:
        _SWEPT = True

    if _patch_vae_class():
        patched.append("comfy.sd.VAE")

    _install_prompt_hook()

    if patched:
        logger.info("[HSWQ] LATENT None-guard applied: %s", ", ".join(patched))
    return bool(patched)
