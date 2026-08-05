"""
HSWQ Torch Compile — torch.compile wrapper for HSWQ diffusion models.

Why this exists (vs KJ TorchCompileModelAdvanced alone):
- HSWQ NVFP4 / INT8 ConvRot parity runs MixedPrecision + online Hadamard rotate.
  Under inductor nested tracing, ``forward_comfy_cast_weights`` can see weights
  that already have symbolic sizes and raise
  ``Expect size to be a plain tuple of ints but got torch.Size([s81, s16])``.
  Parity forward is already marked ``torch.compiler.disable`` in
  ``nodes/zimage_nvfp4/nvfp4_comfy_parity.py``; this node keeps compile defaults
  that match that path (inductor + max-autotune-no-cudagraphs).
- ``backend=cudagraphs`` with USDU / multi-tile shapes hits
  ``cudaMallocAsync`` / ``checkPoolLiveAllocations`` pool errors. Defaults
  forbid that combination.
- Distorch dynamic VRAM weight cast must graph-break (same idea as KJ aimdo
  patch) or USDU recompiles explode.
- On Windows, other nodes (e.g. SeedVR2) may raise ``compile_threads`` and set
  ``worker_start_method=spawn``. Spawn children re-import ``main.py`` with the
  parent ``sys.path`` where ``nodes.py`` already inserted ``comfy/`` first.
  That shadows ComfyUI's ``utils`` package with ``comfy/utils.py`` and crashes::

      ModuleNotFoundError: No module named 'utils.install_util'

  This node forces serial inductor compile (``compile_threads=1``) so that
  ProcessPool spawn is not used for HSWQ compile.

Uses ComfyUI core ``comfy_api.torch_helpers.set_torch_compile_wrapper`` —
no KJNodes dependency.
"""
from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)
# Match usdu_bundle/usdu_patch.py: Comfy console often misses bare module loggers.
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
logger.propagate = False

_DISTORCH_COMPILE_PATCHED = False
_INDUCTOR_COMFY_HARDENED = False


def _configure_inductor_for_comfy() -> None:
    """Force serial inductor compile so spawn workers never re-import main.py.

    ComfyUI ``nodes.py`` does ``sys.path.insert(0, .../comfy)``. Spawn children
    inherit that path; ``import utils`` then binds ``comfy/utils.py`` (module)
    instead of ``ComfyUI/utils/`` (package) → ``utils.install_util`` fails.
    """
    global _INDUCTOR_COMFY_HARDENED
    if _INDUCTOR_COMFY_HARDENED:
        return
    _INDUCTOR_COMFY_HARDENED = True

    # Env wins for inductor decide_*; set before touching config when possible.
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    # Avoid ProcessPoolExecutor(spawn) path used when worker_start_method=spawn.
    os.environ.setdefault("TORCHINDUCTOR_WORKER_START", "subprocess")
    os.environ.pop("TORCHINDUCTOR_PIPELINE_GEMM_AUTOTUNING", None)
    os.environ.pop("TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC", None)

    try:
        import torch._inductor.config as inductor_config
    except Exception as e:
        logger.warning("[HSWQ TorchCompile] could not import inductor config: %s", e)
        return

    prev_threads = getattr(inductor_config, "compile_threads", None)
    prev_start = getattr(inductor_config, "worker_start_method", None)
    inductor_config.compile_threads = 1
    if hasattr(inductor_config, "worker_start_method"):
        # subprocess uses compile_worker.__main__, not ComfyUI main.py.
        inductor_config.worker_start_method = "subprocess"
    if hasattr(inductor_config, "pipeline_max_autotune_gemm"):
        inductor_config.pipeline_max_autotune_gemm = False
    if hasattr(inductor_config, "autotune_in_subproc"):
        inductor_config.autotune_in_subproc = False

    # Drop any already-warmed spawn ProcessPool (e.g. SeedVR2 raised threads).
    try:
        from torch._inductor.async_compile import AsyncCompile, shutdown_compile_workers

        shutdown_compile_workers()
        if hasattr(AsyncCompile.process_pool, "cache_clear"):
            AsyncCompile.process_pool.cache_clear()
        if hasattr(AsyncCompile.pool, "cache_clear"):
            AsyncCompile.pool.cache_clear()
    except Exception as e:
        logger.debug("[HSWQ TorchCompile] compile worker shutdown skipped: %s", e)

    try:
        from torch._inductor.autotune_process import AutotuneProcessPool

        if hasattr(AutotuneProcessPool, "shutdown_instance"):
            AutotuneProcessPool.shutdown_instance()
        elif getattr(AutotuneProcessPool, "_instance", None) is not None:
            AutotuneProcessPool._instance = None
    except Exception:
        pass

    logger.info(
        "[HSWQ TorchCompile] inductor hardened for ComfyUI "
        "(compile_threads %s→1, worker_start_method %s→subprocess, "
        "pipeline/autotune_subproc off)",
        prev_threads,
        prev_start,
    )

# Layer container names used by Z-Image / Lumina2 / Flux-family / Wan, etc.
_LAYER_ATTR_NAMES = (
    "layers",
    "double_blocks",
    "single_blocks",
    "transformer_blocks",
    "blocks",
    "visual_transformer_blocks",
    "text_transformer_blocks",
    "patch_blocks",
    "pixel_blocks",
)


def _patch_distorch_ops_for_compile() -> None:
    """Mark Distorch / weight-cast helpers as eager graph breaks under dynamo."""
    global _DISTORCH_COMPILE_PATCHED
    if _DISTORCH_COMPILE_PATCHED:
        return
    _DISTORCH_COMPILE_PATCHED = True
    try:
        import comfy.ops as ops
    except ImportError:
        return
    names = (
        "cast_bias_weight",
        "uncast_bias_weight",
        "cast_modules_with_vbar",
        "resolve_cast_module_with_vbar",
    )
    for name in names:
        fn = getattr(ops, name, None)
        if fn is not None and callable(fn):
            setattr(ops, name, torch._dynamo.disable(fn))
    try:
        import comfy_aimdo.torch as _at

        if hasattr(_at, "get_tensor_from_raw_ptr"):
            _at.get_tensor_from_raw_ptr = torch._dynamo.disable(_at.get_tensor_from_raw_ptr)
    except Exception:
        pass
    logger.info(
        "[HSWQ TorchCompile] marked comfy.ops weight-cast helpers as eager "
        "(Distorch / dynamic VRAM recompile reduction)"
    )


def _collect_block_keys(diffusion_model) -> list[str]:
    keys: list[str] = []
    for layer_name in _LAYER_ATTR_NAMES:
        blocks = getattr(diffusion_model, layer_name, None)
        if blocks is None:
            continue
        try:
            n = len(blocks)
        except TypeError:
            continue
        for i in range(n):
            keys.append(f"diffusion_model.{layer_name}.{i}")
    return keys


def _build_compile_kwargs(backend: str, mode: str, fullgraph: bool, dynamic):
    """torch.compile forbids mode and options together; mode wins when set."""
    kw = {"backend": backend, "fullgraph": fullgraph, "dynamic": dynamic}
    if mode and mode != "default":
        kw["mode"] = mode
    return kw


class HSWQTorchCompileModel:
    """HSWQ-dedicated torch.compile for Z-Image / SDXL HSWQ + USDU."""

    TITLE = "HSWQ Torch Compile"
    CATEGORY = "HSWQ/torchcompile"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    DESCRIPTION = (
        "torch.compile for HSWQ models. Defaults: inductor + "
        "max-autotune-no-cudagraphs (safe with USDU / Distorch). "
        "Avoid cudagraphs with multi-tile USDU."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "backend": (
                    ["inductor", "cudagraphs"],
                    {
                        "default": "inductor",
                        "tooltip": "Use inductor with USDU. cudagraphs often fails on multi-tile / cudaMallocAsync pools.",
                    },
                ),
                "fullgraph": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Full graph mode. Keep False for HSWQ ConvRot / Distorch paths.",
                    },
                ),
                "mode": (
                    [
                        "default",
                        "max-autotune",
                        "max-autotune-no-cudagraphs",
                        "reduce-overhead",
                    ],
                    {
                        "default": "max-autotune-no-cudagraphs",
                        "tooltip": "HSWQ default avoids CUDA graphs (USDU / dynamic shapes).",
                    },
                ),
                "dynamic": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Dynamic shape tracing. Prefer off unless tile shapes vary every step.",
                    },
                ),
                "dynamo_cache_size_limit": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 1024,
                        "step": 1,
                        "tooltip": "torch._dynamo.config.cache_size_limit",
                    },
                ),
                "force_parameter_static_shapes": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "torch._dynamo.config.force_parameter_static_shapes — helps avoid symbolic weight Size errors.",
                    },
                ),
                "patch_distorch_weight_cast": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Mark comfy.ops cast_bias_weight etc. as eager (Distorch / dynamic VRAM).",
                    },
                ),
                "debug_compile_keys": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Log compile keys"},
                ),
            },
            "optional": {
                "disable_dynamic_vram": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Clone model with disable_dynamic=True when ComfyUI supports it.",
                    },
                ),
            },
        }

    def patch(
        self,
        model,
        backend,
        fullgraph,
        mode,
        dynamic,
        dynamo_cache_size_limit,
        force_parameter_static_shapes,
        patch_distorch_weight_cast,
        debug_compile_keys,
        disable_dynamic_vram=False,
        **_kwargs,
    ):
        from comfy_api.torch_helpers import set_torch_compile_wrapper

        # Before first inductor compile: kill spawn ProcessPool (utils shadow).
        _configure_inductor_for_comfy()

        if backend == "cudagraphs":
            logger.warning(
                "[HSWQ TorchCompile] backend=cudagraphs with USDU / multi-tile "
                "often hits cudaMallocAsync pool errors. Prefer inductor."
            )
        if backend == "cudagraphs" and mode == "max-autotune-no-cudagraphs":
            logger.warning(
                "[HSWQ TorchCompile] backend=cudagraphs ignores "
                "max-autotune-no-cudagraphs; CUDA graphs still apply."
            )

        if disable_dynamic_vram:
            try:
                m = model.clone(disable_dynamic=True)
            except TypeError:
                logger.warning(
                    "[HSWQ TorchCompile] This ComfyUI build cannot disable "
                    "dynamic VRAM via clone(disable_dynamic=True)."
                )
                m = model.clone()
        else:
            m = model.clone()

        diffusion_model = m.get_model_object("diffusion_model")
        torch._dynamo.config.cache_size_limit = dynamo_cache_size_limit
        if hasattr(torch._dynamo.config, "force_parameter_static_shapes"):
            torch._dynamo.config.force_parameter_static_shapes = force_parameter_static_shapes

        if patch_distorch_weight_cast:
            _patch_distorch_ops_for_compile()

        # Always compile per transformer block (former toggle default ON).
        compile_key_list = _collect_block_keys(diffusion_model)
        if not compile_key_list:
            logger.warning(
                "[HSWQ TorchCompile] No known transformer blocks found; "
                "compiling entire diffusion_model"
            )
            compile_key_list = ["diffusion_model"]
        if debug_compile_keys:
            logger.info("[HSWQ TorchCompile] compile keys (%d):", len(compile_key_list))
            for key in compile_key_list:
                logger.info(" - %s", key)

        compile_kwargs = _build_compile_kwargs(backend, mode, fullgraph, bool(dynamic))
        try:
            set_torch_compile_wrapper(model=m, keys=compile_key_list, **compile_kwargs)
        except Exception as e:
            raise RuntimeError("HSWQ TorchCompile failed") from e

        logger.info(
            "[HSWQ TorchCompile] applied keys=%d backend=%s mode=%s fullgraph=%s dynamic=%s",
            len(compile_key_list),
            backend,
            mode,
            fullgraph,
            bool(dynamic),
        )
        return (m,)


NODE_CLASS_MAPPINGS = {
    "HSWQTorchCompileModel": HSWQTorchCompileModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HSWQTorchCompileModel": "HSWQ Torch Compile",
}
