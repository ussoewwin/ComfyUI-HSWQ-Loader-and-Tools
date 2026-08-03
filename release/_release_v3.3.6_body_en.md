<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><font color="#4b5563"><b>EN</b></font></td>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/zhmd/v3.3.6.md"><font color="#ffffff"><b>中文</b></font></a></td>
  </tr>
</table>

## 1. Nodes created or modified

### 1.1 Created — `HSWQTorchCompileModel` (display: **HSWQ Torch Compile**)

| Item | Value |
|------|-------|
| Class | `HSWQTorchCompileModel` |
| Comfy type / mapping key | `HSWQTorchCompileModel` |
| Display name | `HSWQ Torch Compile` |
| Category | `HSWQ/torchcompile` |
| File | `nodes/hswq_torch_compile.py` (**new** in `06d3ac2`) |
| I/O | `MODEL` → `MODEL` |
| Experimental | `True` |
| KJNodes dependency | **None** — uses ComfyUI core `comfy_api.torch_helpers.set_torch_compile_wrapper` |

**INPUT_TYPES (defaults):**

| Input | Type | Default | Role |
|-------|------|---------|------|
| `model` | MODEL | — | Model to wrap |
| `backend` | inductor / cudagraphs | `inductor` | Prefer inductor with USDU |
| `fullgraph` | BOOLEAN | `False` | Keep False for ConvRot / Distorch |
| `mode` | enum | `max-autotune-no-cudagraphs` | Avoid CUDA graphs with USDU |
| `dynamic` | auto / true / false | `false` | Shape tracing |
| `compile_transformer_blocks_only` | BOOLEAN | `True` | Per-block keys when possible |
| `dynamo_cache_size_limit` | INT | `64` | `torch._dynamo.config.cache_size_limit` |
| `force_parameter_static_shapes` | BOOLEAN | `True` | Reduce symbolic `torch.Size` on weights |
| `patch_distorch_weight_cast` | BOOLEAN | `True` | Mark `comfy.ops` cast helpers eager |
| `debug_compile_keys` | BOOLEAN | `False` | Log compile keys |
| `disable_dynamic_vram` | BOOLEAN (optional) | `False` | `clone(disable_dynamic=True)` when supported |

**Helpers in the same file (not separate nodes):**

| Symbol | Role |
|--------|------|
| `_configure_inductor_for_comfy` | Force `compile_threads=1`, `worker_start_method=subprocess`; shut down spawn ProcessPools |
| `_patch_distorch_ops_for_compile` | `torch._dynamo.disable` on Distorch weight-cast helpers |
| `_collect_block_keys` | Build `diffusion_model.<blocks>.<i>` keys from known container names |
| `_build_compile_kwargs` | Build kwargs for `set_torch_compile_wrapper` |

### 1.2 Modified — package registration in `__init__.py`

| Item | Value |
|------|-------|
| File | `__init__.py` |
| Change | `try/except` import of `HSWQTorchCompileModel` into `NODE_CLASS_MAPPINGS` |
| Commit | `06d3ac2` |

Display name comes from class `TITLE` via the existing mapping builder:

```python
NODE_DISPLAY_NAME_MAPPINGS = {k: getattr(v, "TITLE", k) for k, v in NODE_CLASS_MAPPINGS.items()}
```

### 1.3 Modified — Z Image / INT8 peel (not a new Comfy node)

| Item | Value |
|------|-------|
| File | `nodes/zimage_nvfp4/nvfp4_comfy_parity.py` |
| Function | `peel_non_product_nvfp4_ops` |
| Primary caller | `restore_nvfp4_tc_product_stack` |
| Commit | `06993f7` |
| User-facing node | **None new** — ops-stack restore after **Z Image → SDXL INT8** |

---

## 2. Full source of created or modified code

### 2.1 Full file — `nodes/hswq_torch_compile.py`

```python
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
    EXPERIMENTAL = True
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
                    ["auto", "true", "false"],
                    {
                        "default": "false",
                        "tooltip": "Dynamic shape tracing. Prefer false unless tile shapes vary every step.",
                    },
                ),
                "compile_transformer_blocks_only": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Compile per-block (layers/blocks/…). Faster and more stable for Z-Image.",
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
        compile_transformer_blocks_only,
        dynamo_cache_size_limit,
        force_parameter_static_shapes,
        patch_distorch_weight_cast,
        debug_compile_keys,
        disable_dynamic_vram=False,
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

        dynamic_kv = {"true": True, "false": False, "auto": None}
        try:
            dynamic_val = dynamic_kv[dynamic]
        except KeyError as e:
            raise ValueError(f"Invalid dynamic arg {dynamic!r}") from e

        compile_key_list: list[str] = []
        if compile_transformer_blocks_only:
            compile_key_list = _collect_block_keys(diffusion_model)
            if not compile_key_list:
                logger.warning(
                    "[HSWQ TorchCompile] No known transformer blocks found; "
                    "compiling entire diffusion_model"
                )
            elif debug_compile_keys:
                logger.info("[HSWQ TorchCompile] compile keys:")
                for key in compile_key_list:
                    logger.info(" - %s", key)
        if not compile_key_list:
            compile_key_list = ["diffusion_model"]

        compile_kwargs = _build_compile_kwargs(backend, mode, fullgraph, dynamic_val)
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
            dynamic_val,
        )
        return (m,)


NODE_CLASS_MAPPINGS = {
    "HSWQTorchCompileModel": HSWQTorchCompileModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HSWQTorchCompileModel": "HSWQ Torch Compile",
}
```

### 2.2 Modified registration — `__init__.py` (exact block)

```python
try:
    from .nodes.hswq_torch_compile import HSWQTorchCompileModel
    NODE_CLASS_MAPPINGS["HSWQTorchCompileModel"] = HSWQTorchCompileModel
    logger.info("Registered HSWQ Torch Compile")
except (ImportError, ModuleNotFoundError) as e:
    logger.debug("HSWQ Torch Compile not registered: %s", e)
```

### 2.3 Full function after fix — `peel_non_product_nvfp4_ops` (`06993f7`)

```python
def peel_non_product_nvfp4_ops(ops) -> bool:
    """Peel Z Image / comfy_parity mp+load wrappers down to stock or SDXL product_tc.

    Used when PRODUCT was never saved (INT8-only → Z Image → SDXL) so restore
    cannot reinstate TC, but ZI mp must not keep attaching VER=8 Linear bake.
    """
    changed = False
    cur = getattr(ops, "mixed_precision_ops", None)
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if getattr(cur, "_hswq_nvfp4_product_tc", False):
            if ops.mixed_precision_ops is not cur:
                ops.mixed_precision_ops = cur
                changed = True
            break
        is_foreign = bool(
            getattr(cur, "_hswq_nvfp4_comfy_only", False)
            or (
                getattr(cur, "_hswq_nvfp4_stack_ver", 0)
                and not getattr(cur, "_hswq_nvfp4_product_tc", False)
            )
        )
        if not is_foreign:
            if ops.mixed_precision_ops is not cur:
                ops.mixed_precision_ops = cur
                changed = True
            break
        nxt = getattr(cur, "_hswq_nvfp4_orig_mp", None) or getattr(
            cur, "_hswq_orig_mixed_precision_ops", None
        )
        if nxt is None:
            break
        ops.mixed_precision_ops = nxt
        changed = True
        cur = nxt

    def _is_foreign_int8_protect_load(fn) -> bool:
        """ZI / INT8-protect / parity load wraps that must not survive onto SDXL."""
        return bool(
            getattr(fn, "_hswq_nvfp4_comfy_only", False)
            or getattr(fn, "_hswq_int8_protect_in_load", False)
            or getattr(fn, "_hswq_int8_protect_arm_v2", False)
            or getattr(fn, "_hswq_int8_decode_patched", False)
            or (
                getattr(fn, "_hswq_nvfp4_full_load", False)
                and not getattr(fn, "_hswq_nvfp4_product_tc", False)
            )
        )

    def _next_load_under(fn):
        # arm overlay closes over ``cur``; comfy_only / INT8 use orig*/_orig_load.
        for name in ("cur", "orig_load", "original_load", "_orig_load"):
            nxt = _closure_named(fn, name)
            if nxt is not None:
                return nxt
        return getattr(fn, "_hswq_nvfp4_orig_load", None)

    cur_l = getattr(ops, "_load_quantized_module", None)
    seen_l: set[int] = set()
    while cur_l is not None and id(cur_l) not in seen_l:
        seen_l.add(id(cur_l))
        if getattr(cur_l, "_hswq_nvfp4_product_tc", False):
            under = _next_load_under(cur_l)
            # PRODUCT restored on top of ZI protect arm still calls that arm for
            # int8_tensorwise+convrot (same conf shape as SDXL INT8 ConvRot).
            if under is not None and _is_foreign_int8_protect_load(under):
                ops._load_quantized_module = under
                changed = True
                cur_l = under
                continue
            if ops._load_quantized_module is not cur_l:
                ops._load_quantized_module = cur_l
                changed = True
            break
        if not _is_foreign_int8_protect_load(cur_l):
            if ops._load_quantized_module is not cur_l:
                ops._load_quantized_module = cur_l
                changed = True
            break
        nxt_l = _next_load_under(cur_l)
        if nxt_l is None:
            break
        ops._load_quantized_module = nxt_l
        changed = True
        cur_l = nxt_l
    return changed
```

### 2.4 Caller context — `restore_nvfp4_tc_product_stack` (unchanged call site; peels when PRODUCT refs missing)

```python
    # INT8-only → Z Image → SDXL: PRODUCT was never saved. Peel ZI / parity so
    # SDXL INT8 does not keep attaching VER=8 ``[HSWQ ConvRot LoRA] int8_protect``.
    peeled = peel_non_product_nvfp4_ops(ops)
    _PARITY_APPLIED = False
    if peeled:
        if (
            getattr(ops.mixed_precision_ops, "_hswq_nvfp4_product_tc", False)
            and getattr(ops._load_quantized_module, "_hswq_nvfp4_product_tc", False)
        ):
            remember_nvfp4_tc_product_stack(
                ops._load_quantized_module, ops.mixed_precision_ops
            )
            _console(
                "[HSWQ NVFP4] restored product TC stack via peel "
                "(SDXL path; parity off)"
            )
        else:
            _console(
                "[HSWQ NVFP4] peeled non-product NVFP4 ops "
                "(stock/INT8 base; no product_tc PRODUCT — SDXL INT8 LoRA safe)"
            )
        return True
```

---

## 3. Meaning of the Torch Compile code

### 3.1 Why a dedicated HSWQ node exists

Generic KJ `TorchCompileModelAdvanced` is not HSWQ-aware. HSWQ paths combine:

- ConvRot NVFP4 / INT8 protect **online act rotate** (parity forward)
- Distorch / dynamic VRAM weight cast
- Ultimate SD Upscale (USDU) multi-tile shapes
- Other extensions (e.g. SeedVR2) that raise inductor `compile_threads` and force **spawn**

This node hardens defaults and inductor process policy for that stack, without depending on KJNodes.

### 3.2 `_configure_inductor_for_comfy` — SeedVR2 / Windows spawn crash

**Failure mode:**

1. ComfyUI `nodes.py` inserts `.../comfy` at `sys.path[0]`.
2. Another node (SeedVR2) sets `compile_threads > 1` and `worker_start_method=spawn`.
3. Inductor ProcessPool children re-import ComfyUI `main.py` with that `sys.path`.
4. `import utils` resolves to **`comfy/utils.py` (module)** instead of **`ComfyUI/utils/` (package)**.
5. Crash: `ModuleNotFoundError: No module named 'utils.install_util'`.

**Fix in this node:**

| Setting | Value | Why |
|---------|-------|-----|
| `TORCHINDUCTOR_COMPILE_THREADS` / `compile_threads` | `1` | Serial compile — no ProcessPool spawn for HSWQ |
| `worker_start_method` | `subprocess` | Uses inductor compile worker entry, not ComfyUI `main.py` |
| `pipeline_max_autotune_gemm` / `autotune_in_subproc` | off | Avoid extra subprocess / spawn paths |
| `shutdown_compile_workers` + pool `cache_clear` | yes | Drop already-warmed spawn pools from SeedVR2 |

Called **before** `set_torch_compile_wrapper` on every `patch()`.

### 3.3 Defaults vs USDU / Distorch / symbolic sizes

| Default | Meaning |
|---------|---------|
| `backend=inductor` | `cudagraphs` + multi-tile USDU often hits `cudaMallocAsync` / pool live allocation errors |
| `mode=max-autotune-no-cudagraphs` | Autotune without CUDA graphs |
| `fullgraph=False` | ConvRot / Distorch need graph breaks |
| `force_parameter_static_shapes=True` | Reduces `Expect size to be a plain tuple of ints but got torch.Size([s81, s16])` under nested inductor tracing of cast paths |
| `patch_distorch_weight_cast=True` | Marks `cast_bias_weight` / related ops + optional `comfy_aimdo` raw-ptr helper as eager so USDU does not explode recompiles |

Parity Linear forward is already `torch.compiler.disable` in `nvfp4_comfy_parity.py`; this node’s defaults align with that path.

### 3.4 Compile key selection

When `compile_transformer_blocks_only=True`, keys are built from known containers:

`layers`, `double_blocks`, `single_blocks`, `transformer_blocks`, `blocks`, `visual_transformer_blocks`, `text_transformer_blocks`, `patch_blocks`, `pixel_blocks`

as `diffusion_model.<name>.<i>`.

If none are found, falls back to compiling the whole `diffusion_model`.

### 3.5 Runtime flow of `patch()`

```text
_configure_inductor_for_comfy()
  → clone model (optional disable_dynamic)
  → set dynamo cache / force_parameter_static_shapes
  → optional _patch_distorch_ops_for_compile()
  → collect keys
  → set_torch_compile_wrapper(model, keys, backend/mode/fullgraph/dynamic)
  → return cloned MODEL
```

### 3.6 What this node does **not** do

- Does not disable DistOrch purge
- Does not change Z Image NVFP4 parity Hadamard / ConvRot arming
- Does not replace KJNodes globally — it is an HSWQ menu node only
- Does not fix USDU grain after SeedVR2 by itself; it only makes `torch.compile` safe on the HSWQ path under ComfyUI + spawn pollution

---

## 4. Z Image / INT8 peel fix (`06993f7`)

### 4.1 Problem this fix targets

Sequence:

```text
SDXL INT8 (or INT8-only)  →  Z Image NVFP4 / INT8 protect (comfy_parity arms)  →  SDXL INT8 again
```

`restore_nvfp4_tc_product_stack` must put SDXL-safe `_load_quantized_module` / `mixed_precision_ops` back.

When PRODUCT TC refs were never saved (`_PRODUCT_LOAD` / `_PRODUCT_MP` are `None`), restore falls through to **`peel_non_product_nvfp4_ops`**.

**Bug before `06993f7`:**

- Load peel stopped at the first function stamped `_hswq_nvfp4_product_tc`.
- That PRODUCT tip could still **close over** a Z Image / INT8 protect arm underneath.
- For conf shapes matching **`int8_tensorwise + convrot`** (same shape as SDXL INT8 ConvRot), the foreign arm still ran.
- Result: Z Image VER=8 / INT8 protect Linear bake / protect arm behavior leaked onto SDXL INT8 (LoRA / load contamination).

`mixed_precision_ops` peel (upper half of the function) was already able to stop on clean `product_tc`; the **load** half needed the same “look under PRODUCT” discipline plus broader next-layer discovery.

### 4.2 What changed (diff summary)

| Before | After |
|--------|-------|
| Foreign load flags inlined in the loop | `_is_foreign_int8_protect_load(fn)` helper |
| Next layer: `orig_load` / `original_load` / `_hswq_nvfp4_orig_load` only | `_next_load_under(fn)` also walks closure name **`cur`** (INT8 protect arm overlay) |
| On `_hswq_nvfp4_product_tc`: **stop** | On PRODUCT: if under is foreign INT8/ZI protect → **set tip to under and continue peeling** |

Foreign markers recognized:

- `_hswq_nvfp4_comfy_only`
- `_hswq_int8_protect_in_load`
- `_hswq_int8_protect_arm_v2`
- `_hswq_int8_decode_patched`
- `_hswq_nvfp4_full_load` **without** `_hswq_nvfp4_product_tc`

### 4.3 Exact behavior after the fix

```text
while load tip:
  if tip has product_tc:
      under = _next_load_under(tip)
      if under is foreign INT8/ZI protect:
          tip = under          # dive under poisoned PRODUCT
          continue peel
      keep tip as PRODUCT; break
  if tip is not foreign:
      keep tip; break
  tip = _next_load_under(tip)  # peel foreign
```

Poisoned PRODUCT tip is not treated as “done.” The loop peels the foreign arm(s) under it until stock / clean INT8 / a non-foreign load remains. If a clean `product_tc` tip remains after peel, `restore_nvfp4_tc_product_stack` may `remember_nvfp4_tc_product_stack` again. Otherwise console reports stock/INT8 base with SDXL INT8 LoRA-safe peel.

### 4.4 Pre-fix load loop (for comparison)

```python
        if getattr(cur_l, "_hswq_nvfp4_product_tc", False):
            if ops._load_quantized_module is not cur_l:
                ops._load_quantized_module = cur_l
                changed = True
            break
        is_foreign_l = bool(
            getattr(cur_l, "_hswq_nvfp4_comfy_only", False)
            or getattr(cur_l, "_hswq_int8_protect_in_load", False)
            or getattr(cur_l, "_hswq_int8_protect_arm_v2", False)
            or getattr(cur_l, "_hswq_int8_decode_patched", False)
            or (
                getattr(cur_l, "_hswq_nvfp4_full_load", False)
                and not getattr(cur_l, "_hswq_nvfp4_product_tc", False)
            )
        )
        if not is_foreign_l:
            ...
            break
        nxt_l = _closure_named(cur_l, "orig_load") or _closure_named(
            cur_l, "original_load"
        )
        if nxt_l is None:
            nxt_l = getattr(cur_l, "_hswq_nvfp4_orig_load", None)
```

Missing: dive-under PRODUCT; missing: closure cell name `cur`.

### 4.5 Scope boundaries

| In scope | Out of scope |
|----------|----------------|
| Ops `_load_quantized_module` chain after Z Image → SDXL INT8 restore/peel | DistOrch VRAM purge policy |
| INT8 protect / comfy_parity / non-product full_load wrappers | SeedVR2 → USDU grain diagnosis |
| Companion of earlier peel work (`f030d71` family) | Changing Z Image parity forward itself |

### 4.6 Relation to Torch Compile

These are **independent** commits in one work stream:

| Commit | Surface |
|--------|---------|
| `06d3ac2` | New Comfy node **HSWQ Torch Compile** |
| `06993f7` | Silent ops peel inside `nvfp4_comfy_parity.py` |

Using the Torch Compile node does not apply the peel; loading Z Image then SDXL INT8 (via restore/peel path) does not require the Torch Compile node.

---

## 5. Audit checklist

```
□ md/HSWQ_TORCH_COMPILE_AND_ZI_INT8_PEEL_GUIDE.md exists
□ Section 1 lists HSWQTorchCompileModel + __init__ registration + peel (no new peel node)
□ Section 2 includes full hswq_torch_compile.py, __init__ block, full peel_non_product_nvfp4_ops
□ Section 3 explains inductor spawn / USDU / Distorch / keys
□ Section 4 explains PRODUCT-under-foreign peel and markers
□ Commits 06d3ac2 and 06993f7 named
```

---

## 6. Related files (read-only pointers)

| Path | Role |
|------|------|
| `nodes/hswq_torch_compile.py` | Torch Compile node |
| `__init__.py` | Registration |
| `nodes/zimage_nvfp4/nvfp4_comfy_parity.py` | Parity + peel + restore |
| `md/HSWQ_ZIMAGE_CONVROT_NVFP4_TECHNICAL_GUIDE.md` | Broader Z Image NVFP4 manual |

---

End of guide.

