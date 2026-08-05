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
