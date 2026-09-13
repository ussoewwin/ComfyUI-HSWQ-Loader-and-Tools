# -*- coding: utf-8 -*-
"""SDXL-only ConvRot INT8 kernel fast path.

**Scope: SDXL ConvRot INT8 only.** This module exists so that the fast path does
not live in any shared module: `patches/comfy_quant_int8.py` and
`nodes/native_convert_int8.py` keep their original code, and only the SDXL load
branches call `arm_sdxl_convrot_fast()`.

What it does (ported from the SDXL INT8 trajectory-bench fix):
  * pooled activation rotate - the stock comfy_kitchen rotate allocates a new
    tensor on every call (measured over the real SDXL INT8 layer mix:
    pooled fp32 257.3 ms/step -> pooled bf16 52.5 ms/step; INT8/FP16
    1.069x -> 0.894x),
  * CUDA fused ConvRot kernels OFF - measured slower AND trajectory-drifting
    (2.89 it/s, cos 0.934, 1/6 same-image vs 2.97 it/s, cos 0.958).

Separation guarantees:
  * no import-time side effects (importing this module changes nothing),
  * the kernels are swapped only while an armed SDXL UNet forward runs
    (``_arm()`` in the wrapper, ``_disarm()`` in ``finally``),
  * every touched attribute is saved and restored as the *same object*,
  * arming requires the SDXL architecture (``UNetModel`` from ``openaimodel``
    with ADM conditioning); Z Image / Krea2 / SD1.5 / FLUX and every other
    family is refused and never touched,
  * non-SDXL loads never import this module at all.
"""
from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

_ROTATE_MODULES = (
    "comfy_kitchen.tensor.int8_utils",
    "comfy_kitchen.backends.eager.quantization",
    "comfy_kitchen.backends.triton.quantization",
    "comfy_kitchen.backends.cuda",
    "comfy_kitchen.backends.eager.convrot_w4a4",
)

# ---------------------------------------------------------------------------
# Fail-closed audit of the comfy_kitchen call sites.
#
# The pooled rotate hands back a view of a shared buffer, which is only correct
# while every caller consumes the result before the next rotate of the same
# shape/dtype/device. That is not a type-system invariant, so it is *checked*:
# every `_rotate_activation(...)` call site in the audited kitchen files must
# bind its result and read that binding within the next two statements, with no
# intervening `_rotate_activation(...)` call. If any site does not match (kitchen
# updated, new caller, unparseable file), the fast path REFUSES to arm and logs.
# ---------------------------------------------------------------------------

_AUDITED_ROTATE_FILES = (
    "backends/cuda/__init__.py",
    "backends/eager/quantization.py",
    "backends/triton/quantization.py",
    "backends/eager/convrot_w4a4.py",
)

_AUDIT_CACHE = {"done": False, "ok": False, "detail": "not run"}


def _audit_rotate_call_sites() -> tuple:
    """(ok, detail) - fail-closed structural check of the rotate call sites."""
    if _AUDIT_CACHE["done"]:
        return _AUDIT_CACHE["ok"], _AUDIT_CACHE["detail"]
    import ast
    import os

    try:
        import comfy_kitchen

        root = os.path.dirname(os.path.abspath(comfy_kitchen.__file__))
    except Exception as exc:  # pragma: no cover
        _AUDIT_CACHE.update(done=True, ok=False, detail=f"comfy_kitchen unavailable: {exc}")
        return False, _AUDIT_CACHE["detail"]

    # 0) no caller may live outside the audited files: walk the whole package and
    #    collect every `_rotate_activation(...)` call site.
    unexpected = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fn)
            rel = os.path.relpath(fpath, root).replace(os.sep, "/")
            if rel in _AUDITED_ROTATE_FILES:
                continue
            try:
                with open(fpath, encoding="utf-8") as fh:
                    src2 = fh.read()
                tree2 = ast.parse(src2)
            except Exception:
                continue
            for node in ast.walk(tree2):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "_rotate_activation"):
                    unexpected.append(f"{rel}:{node.lineno}")
    if unexpected:
        _AUDIT_CACHE.update(done=True, ok=False,
                            detail=f"unexpected rotate caller(s): {unexpected[:5]}")
        return False, _AUDIT_CACHE["detail"]

    n_sites = 0
    for rel in _AUDITED_ROTATE_FILES:
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            _AUDIT_CACHE.update(done=True, ok=False, detail=f"audited file missing: {rel}")
            return False, _AUDIT_CACHE["detail"]
        try:
            with open(path, encoding="utf-8") as fh:
                src = fh.read()
            tree = ast.parse(src)
        except Exception as exc:
            _AUDIT_CACHE.update(done=True, ok=False, detail=f"unparseable {rel}: {exc}")
            return False, _AUDIT_CACHE["detail"]

        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            body = fn.body
            for i, stmt in enumerate(body):
                calls = [n for n in ast.walk(stmt)
                         if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Name)
                         and n.func.id == "_rotate_activation"]
                if not calls:
                    continue
                n_sites += 1
                if not isinstance(stmt, ast.Assign):
                    continue  # consumed inside the same statement -> safe
                targets = [tg for tg in stmt.targets if isinstance(tg, ast.Name)]
                if not targets:
                    _AUDIT_CACHE.update(
                        done=True, ok=False,
                        detail=f"{rel}:{stmt.lineno} unanalysable assignment target")
                    return False, _AUDIT_CACHE["detail"]
                name = targets[0].id
                # first later statement that reads the binding / calls rotate again
                limit = min(i + 3, len(body))
                for j in range(i + 1, limit):
                    later = body[j]
                    rereads = [n for n in ast.walk(later)
                               if isinstance(n, ast.Name) and n.id == name
                               and isinstance(n.ctx, ast.Load)]
                    recalls = [n for n in ast.walk(later)
                               if isinstance(n, ast.Call)
                               and isinstance(n.func, ast.Name)
                               and n.func.id == "_rotate_activation"]
                    if recalls:
                        _AUDIT_CACHE.update(
                            done=True, ok=False,
                            detail=(f"{rel}:{later.lineno} another _rotate_activation call "
                                    f"before '{name}' was consumed"))
                        return False, _AUDIT_CACHE["detail"]
                    if rereads:
                        break
                else:
                    _AUDIT_CACHE.update(
                        done=True, ok=False,
                        detail=(f"{rel}:{stmt.lineno} result '{name}' not consumed within "
                                f"the next statements"))
                    return False, _AUDIT_CACHE["detail"]

    _AUDIT_CACHE.update(done=True, ok=True, detail=f"{n_sites} call sites verified")
    return True, _AUDIT_CACHE["detail"]


_STATE = {"depth": 0, "saved": [], "pool": {}, "helpers": None}


def set_helpers_loader(loader) -> None:
    """Inject the nodes/native_convert_int8 module (kept out of this file so
    the module has no package-relative imports and can be loaded privately)."""
    _STATE["helpers"] = loader


def _helpers():
    loader = _STATE.get("helpers")
    return loader() if loader is not None else None


def _always_false(*_a, **_k):
    return False


def _pooled_rotate(x, h, group_size):
    """comfy_kitchen ConvRot activation rotate with a pooled output buffer."""
    import torch

    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} not divisible by group_size {group_size}")
    n_groups = features // group_size
    x_grouped = x.reshape(-1, n_groups, group_size)
    if h.dtype != x_grouped.dtype or h.device != x_grouped.device:
        h = h.to(dtype=x_grouped.dtype, device=x_grouped.device)
    key = (tuple(x_grouped.shape), x_grouped.dtype, str(x_grouped.device))
    out = _STATE["pool"].get(key)
    if out is None:
        out = torch.empty_like(x_grouped)
        _STATE["pool"][key] = out
    torch.matmul(x_grouped, h, out=out)
    return out.reshape(orig_shape)


def _pooled_helper_rotate(x, h_matrix, group_size):
    """nodes/native_convert_int8.rotate_activation with a pooled output buffer.

    Same math as that helper (device-cached Hadamard), allocation removed.
    """
    import torch

    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} not divisible by group_size {group_size}")
    group_count = features // group_size
    x_grouped = x.reshape(-1, group_count, group_size)
    h = _helpers().get_hadamard_on_device(
        group_size, device=x.device, dtype=x.dtype
    )
    key = (tuple(x_grouped.shape), x_grouped.dtype, str(x_grouped.device))
    out = _STATE["pool"].get(key)
    if out is None:
        out = torch.empty_like(x_grouped)
        _STATE["pool"][key] = out
    torch.matmul(x_grouped, h, out=out)
    return out.reshape(orig_shape)


def _arm_kernels() -> bool:
    if _STATE["depth"] > 0:
        _STATE["depth"] += 1
        return True
    ok, detail = _audit_rotate_call_sites()
    if not ok:
        logger.warning(
            "[HSWQ INT8][SDXL] fast path refused: rotate call-site audit failed (%s)", detail
        )
        return False
    saved = []
    try:
        for modname in _ROTATE_MODULES:
            try:
                m = importlib.import_module(modname)
            except Exception:
                continue
            if hasattr(m, "_rotate_activation"):
                saved.append((m, "_rotate_activation", m._rotate_activation))
                m._rotate_activation = _pooled_rotate

        cuda_mod = importlib.import_module("comfy_kitchen.backends.cuda")
        if hasattr(cuda_mod, "_CONVROT_FUSED_MAX_K"):
            saved.append((cuda_mod, "_CONVROT_FUSED_MAX_K", cuda_mod._CONVROT_FUSED_MAX_K))
            cuda_mod._CONVROT_FUSED_MAX_K = -1
        for attr in ("_should_use_convrot_fused_kernel", "_should_use_convrot_dequant_kernel"):
            if hasattr(cuda_mod, attr):
                saved.append((cuda_mod, attr, getattr(cuda_mod, attr)))
                setattr(cuda_mod, attr, _always_false)

        nc = _helpers()
        if nc is not None and hasattr(nc, "rotate_activation"):
            saved.append((nc, "rotate_activation", nc.rotate_activation))
            nc.rotate_activation = _pooled_helper_rotate

        _STATE["saved"] = saved
        _STATE["depth"] = 1
        return True
    except Exception as exc:
        logger.warning("[HSWQ INT8][SDXL] fast-path arm failed: %s", exc)
        _STATE["saved"] = saved
        _STATE["depth"] = 1
        _disarm_kernels()
        return False


def _disarm_kernels() -> None:
    for mod, attr, val in reversed(_STATE["saved"] or []):
        try:
            setattr(mod, attr, val)
        except Exception:
            pass
    _STATE["saved"] = []
    _STATE["depth"] = 0
    # Release the pooled buffers when the window closes: outside the SDXL window
    # the process therefore retains exactly the stock amount of VRAM (the stock
    # rotate allocates a temporary per call and frees it). Inside the window the
    # buffers are reused (~one allocation per distinct activation shape per
    # forward instead of one per call).
    _STATE["pool"].clear()


def is_sdxl_convrot_fast_armed() -> bool:
    return _STATE["depth"] > 0


def arm_sdxl_convrot_fast(model, *, log_prefix="[HSWQ INT8]") -> bool:
    """Arm the SDXL ConvRot INT8 fast path for THIS model only.

    Refuses every non-SDXL architecture (returns False, touches nothing).
    """
    inner = getattr(model, "model", None)
    dm = getattr(inner, "diffusion_model", None)
    if dm is None:
        return False
    cls = type(dm).__name__
    mod = type(dm).__module__.lower()
    if cls != "UNetModel" or "openaimodel" not in mod or getattr(dm, "adm_in_channels", None) is None:
        logger.debug("[HSWQ INT8][SDXL] fast path refused (arch %s.%s)", mod, cls)
        return False
    if getattr(dm, "_hswq_sdxl_convrot_fast_wrapped", False):
        return True
    # Fail-closed: the pooled rotate is only valid while every caller consumes the
    # result immediately. Verify the installed kitchen structurally BEFORE wrapping;
    # on any mismatch the model is left completely untouched (stock path).
    ok, detail = _audit_rotate_call_sites()
    if not ok:
        logger.warning(
            "[HSWQ INT8][SDXL] fast path NOT armed: rotate call-site audit failed (%s)",
            detail,
        )
        return False

    original_forward = dm.forward

    def sdxl_convrot_fast_forward(*args, **kwargs):
        armed = _arm_kernels()
        try:
            return original_forward(*args, **kwargs)
        finally:
            if armed:
                _STATE["depth"] -= 1
                if _STATE["depth"] <= 0:
                    _disarm_kernels()

    dm.forward = sdxl_convrot_fast_forward
    dm._hswq_sdxl_convrot_fast_wrapped = True
    dm._hswq_sdxl_convrot_fast_original_forward = original_forward
    print(f"{log_prefix} SDXL ConvRot INT8 fast path armed for this model only", flush=True)
    logger.info("%s SDXL ConvRot INT8 fast path armed", log_prefix)
    return True
