# Technical Guide: ComfyUI-HSWQ-Loader-and-Tools NVFP4 torch.compile Fix

> Target environment: `D:\USERFILES\ComfyUI\python_embeded\python.exe` (Python 3.13.13) / torch 2.13.0+cu132 / GPU NVIDIA GeForce RTX 5060 Ti (CC 12.0)
> Target node: `D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\ComfyUI-HSWQ-Loader-and-Tools`
> Fixed feature: two crashes that occurred when running "USDU + Lumina2 NVFP4 + HSWQ Torch Compile"

---

## ① Error Contents

This fix resolves two independent errors.

### Error A: `AssertionError: Mixing fake modes NYI` (`BackendCompilerFailed` with backend='inductor')

```
torch._dynamo.exc.BackendCompilerFailed:
backend='inductor' failed:
  ...
  torch/_functorch/_aot_autograd/...  (run_functionalized_fw_and_collect_metadata)
  comfy_kitchen/tensor/base.py:362 in __torch_dispatch__
  nodes/krea2_convrot_nvfp4/nvfp4_gemm.py:105 in dequantize_nvfp4
      out = torch.nn.functional.embedding(out.int(), lut).squeeze(-1)
  torch/nn/functional.py:2615 in embedding
  torch/_compile.py:54 in inner
      return disable_fn(*args, **kwargs)
  ...
AssertionError: Mixing fake modes NYI
```

(It reports that two distinct FakeTensorModes are mixed, in the form `x.fake_mode=0x... vs self=0x...`. The address values vary between runs.)

### Error B: `UnicodeDecodeError: 'cp932' codec can't decode byte 0x94`

```
File "C:\...\repro_crash.py", line 38, in <module>
    f = torch.compile(linear_qt, backend="inductor", dynamic=False)
  File "torch\__init__.py", line 2836, in compile
  ...
  File "torch\_inductor\kernel\mm_grouped.py", line 142, in <module>
    source=load_kernel_template("cutedsl_mm_grouped"),
  File "torch\_inductor\utils.py", line 4770, in load_template
    return f.read()
UnicodeDecodeError: 'cp932' codec can't decode byte 0x94 in position 618: illegal multibyte sequence
```

Both are "fatal". Error A is the actual compilation failure. Error B only occurs on Japanese Windows (cp932 locale) and kills `torch.compile` the moment it tries to read a kernel template (in a real workflow, without the env vars this one fires first and masks Error A).

---

## ② Root Causes

### Root cause of Error A

When `torch.compile(linear_qt, backend="inductor")` compiles `F.linear(x_qt, w_qt, bias)` (where x_qt / w_qt are `QuantizedTensor` subclasses):

1. **Dynamo does not trace into a tensor subclass's `__torch_dispatch__`.** It records `torch._C._nn.linear` as an opaque FX node only.
2. The inductor backend (= aot_autograd) runs the "metadata collection pass" (`run_functionalized_fw_and_collect_metadata`), re-executing the FX graph under a **new FakeTensorMode (mode B)**.
3. During this re-execution, `torch._C._nn.linear` is invoked, dispatching to `QuantizedTensor.__torch_dispatch__` → the HSWQ addmm handler → `hswq_scaled_mm_nvfp4` → `dequantize_nvfp4`, which runs **eagerly (in Python)**.
4. `dequantize_nvfp4` performs its LUT decode using `F.embedding`. PyTorch wraps `torch.embedding` in **`torch._compile.disable`**; when executed under mode B, the raw `torch.embedding` re-enters dispatch.
5. At that point the arguments are:
   - `out.int()` — a FakeTensor derived from the QT's inner tensor `_qdata` (**mode A = the mode dynamo used when it first faked the QT**)
   - `lut` — a tensor just created (**mode B**)
   so **two FakeTensorModes are mixed**. FakeTensorMode's argument validation (`validate_and_convert_non_fake_tensors`) detects this and raises `AssertionError: Mixing fake modes NYI`.

**Fundamental cause**: the QT's inner tensors (`_qdata`, `_params.scale`, `_params.block_scale`) were faked under dynamo's mode A, but the AOT metadata pass re-fakes only the *outer* args under mode B, without re-faking the subclass's inner tensors. As a result, any aten op inside `__torch_dispatch__` that mixes an inner tensor (mode A) with a freshly-created tensor (mode B) crashes.

The crash surfaces at `F.embedding` because that is the op wrapped in `torch._compile.disable`; it re-enters the fake mode and triggers validation.

### Root cause of Error B

On first compilation, `torch._inductor` reads the Triton/CUTLASS kernel templates (`*.py.jinja`). This read is done by `load_template` in `torch/_inductor/utils.py`, which calls:

```python
with open(template_dir / f"{name}.py.jinja") as f:
    return f.read()
```

i.e. it uses the builtin `open()` **without an explicit encoding**. The default encoding of `open()` is **locale-dependent**, and on Japanese Windows it is **cp932**. The template files are written in UTF-8, so decoding their UTF-8 byte sequences (e.g. `0x94`) as cp932 raises `UnicodeDecodeError`.

`load_kernel_template` (`kernel/mm_common.py`) and `load_flex_template` (`kernel/flex/common.py`) are both `functools.partial(load_template, template_dir=...)`, so **all template loading funnels through this single `load_template`**.

---

## ③ Fix Overview

### Fix A (Mixing fake modes)

The FP4 decode was registered as a **`torch.library.custom_op` (`hswq::dequantize_nvfp4`) with a `register_fake` meta kernel**.

- During fake tracing, the implementation body is never executed; only a **shape-only fake kernel** runs. This prevents the mixed-mode dispatch re-entry.
- The real implementation (LUT decode) runs **only on real tensors**.
- Numerics are **bit-identical** to the previous LUT path (verified: `torch.equal` matches, `max|diff| = 0.0`).

### Fix B (cp932)

Python's UTF-8 mode (`PYTHONUTF8=1`) **cannot be enabled after interpreter startup**, so a new module `win_utf8_patch.py` replicates it at runtime, applied from the earliest possible point in the node.

1. Set the `PYTHONUTF8` / `PYTHONIOENCODING` env vars (helps subprocesses).
2. Reconfigure stdin/stdout/stderr to UTF-8 (avoids `UnicodeEncodeError` on non-ASCII output to a cp932 console).
3. Monkey-patch `io.open` / `builtins.open` so a text-mode `open(path)` with no explicit `encoding` **defaults to UTF-8** instead of the locale (binary mode / explicit encoding / file objects / descriptors are left untouched).

It is applied in two places: `prestartup_script.py` (which ComfyUI runs *before* `__init__.py`) and `__init__.py` (just before `import torch`). It is idempotent, so being loaded multiple times is safe.

---

## ④ Files Added / Modified

All under `D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\ComfyUI-HSWQ-Loader-and-Tools`.

| Kind | File | Change |
|------|------|--------|
| Modified | `nodes\krea2_convrot_nvfp4\nvfp4_gemm.py` | Converted FP4 decode into the `hswq::dequantize_nvfp4` custom op |
| New | `win_utf8_patch.py` | Forces the process-wide default text encoding to UTF-8 |
| Modified | `prestartup_script.py` | Added loading of `win_utf8_patch.py` at the top |
| Modified | `__init__.py` | Added loading of `win_utf8_patch.py` at the top (before `import torch`) |
| Restored | `nodes\krea2_convrot_nvfp4\nvfp4_addmm_patch.py` | Reverted a mistaken edit made during investigation back to the correct original state (**no net change**) |

---

## ⑤ Full Text of the Added / Modified Code

### 5-1. `nodes\krea2_convrot_nvfp4\nvfp4_gemm.py` (full text, after fix)

```python
"""HSWQ-owned NVFP4 GEMM helpers (ConvRot NVFP4).

ComfyUI / comfy_kitchen do **not** ship ConvRot×NVFP4 load+forward.
The Linear hot path lives in ``nvfp4_forward._tc_forward_pooled`` →
``nvfp4_runtime.scaled_mm_nvfp4_pooled`` (raw ``_C.cublas_gemm_blockwise_fp4``
with pre-validated shapes; weight stays packed). Never torch native
``F.scaled_mm`` FP4 / registry dispatch — path A sticky-poisons SM120.

This module owns:
  - NVFP4 unpack / dequant (FP4 E2M1 + block scales)
  - one-shot weight bake (replace QT Parameter → dense float; free packed) —
    TC-failure fallback only, never the hot path
  - float GEMM ``a @ b.T`` (+ optional bias) for residual QT×QT addmm edges

Runtime patches under ``benchmark/krea2_nvfp4`` must use these entry points.
"""
from __future__ import annotations

from typing import Optional

import torch

# FP4 E2M1 decode LUT (same values as kitchen float_utils / eager dequant).
_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
_E2M1_LUT_CACHE: dict = {}

# dtype ↔ int code for the ``hswq::dequantize_nvfp4`` custom-op schema.
_DTYPE_TO_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.float64: 3,
}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def from_blocked(blocked_matrix, num_rows: int, num_cols: int):
    """Reverse cuBLAS 32×4×4 block-scale swizzle → (num_rows, num_cols)."""
    n_row_blocks = _ceil_div(num_rows, 128)
    n_col_blocks = _ceil_div(num_cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    step1 = blocked_matrix.reshape(-1, 32, 16)
    step2 = step1.reshape(-1, 32, 4, 4).transpose(1, 2)
    step3 = step2.reshape(n_row_blocks, n_col_blocks, 4, 32, 4)
    step4 = step3.reshape(n_row_blocks, n_col_blocks, 128, 4)
    step5 = step4.permute(0, 2, 1, 3)
    unblocked = step5.reshape(padded_rows, padded_cols)
    return unblocked[:num_rows, :num_cols]


def clear_cuda_sticky_error() -> None:
    try:
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
    except Exception:
        pass


def _dequantize_nvfp4_op_impl(
    qx: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    block_scales: torch.Tensor,
    output_dtype_code: int,
    hi_first: bool,
) -> torch.Tensor:
    """Real (eager) NVFP4 dequant body — same math as the pre-compile LUT path."""
    output_type = _CODE_TO_DTYPE[output_dtype_code]

    key = (str(qx.device), output_type)
    lut = _E2M1_LUT_CACHE.get(key)
    if lut is None:
        lut = torch.tensor(
            _E2M1_VALUES, device=qx.device, dtype=output_type
        ).unsqueeze(1)
        _E2M1_LUT_CACHE[key] = lut

    lo = qx & 0x0F
    hi = qx >> 4
    if hi_first:
        out = torch.stack([hi, lo], dim=-1).view(*qx.shape[:-1], -1)
    else:
        out = torch.stack([lo, hi], dim=-1).view(*qx.shape[:-1], -1)
    out = torch.nn.functional.embedding(out.int(), lut).squeeze(-1)

    orig_shape = out.shape
    block_size = 16
    out = out.reshape(orig_shape[0], -1, block_size)
    num_blocks_per_row = orig_shape[1] // block_size
    block_scales_unswizzled = from_blocked(
        block_scales, num_rows=orig_shape[0], num_cols=num_blocks_per_row
    )
    if per_tensor_scale.device != qx.device or per_tensor_scale.dtype != output_type:
        per_tensor_scale = per_tensor_scale.to(device=qx.device, dtype=output_type)
    total_scale = per_tensor_scale * block_scales_unswizzled.to(output_type)
    data_dequantized = out * total_scale.unsqueeze(-1)
    return data_dequantized.view(orig_shape).to(output_type)


@torch.library.custom_op("hswq::dequantize_nvfp4", mutates_args=())
def _dequantize_nvfp4_op(  # noqa: F811  (op impl is decorated below)
    qx: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    block_scales: torch.Tensor,
    output_dtype_code: int,
    hi_first: bool,
) -> torch.Tensor:
    """Registered custom op wrapping the HSWQ LUT dequant.

    Why an op at all: the raw LUT body uses ``F.embedding``, which PyTorch
    wraps in ``torch._compile.disable``. When the addmm/linear dispatch handler
    runs under inductor's AOT metadata pass (FX interpreter with FakeTensor
    args from two FakeTensorModes), that disabled-but-fake-executed embedding
    re-enters dispatch and raises ``AssertionError: Mixing fake modes NYI`` →
    ``BackendCompilerFailed``. Wrapping the decode as a custom op with a
    ``register_fake`` meta kernel makes it opaque to the tracer: fake tracing
    only runs the shape-only fake impl below, and the eager body runs only on
    real tensors. Identical numerics to the pre-compile path.
    """
    return _dequantize_nvfp4_op_impl(
        qx, per_tensor_scale, block_scales, output_dtype_code, hi_first
    )


@_dequantize_nvfp4_op.register_fake
def _dequantize_nvfp4_op_fake(
    qx, per_tensor_scale, block_scales, output_dtype_code, hi_first
):
    output_type = _CODE_TO_DTYPE[output_dtype_code]
    return torch.empty(
        (*qx.shape[:-1], qx.shape[-1] * 2),
        dtype=output_type,
        device=qx.device,
    )


def dequantize_nvfp4(
    qx,
    per_tensor_scale,
    block_scales,
    output_type=None,
    *,
    hi_first: bool = True,
):
    """Unpack NVFP4 uint8×2 + block scales → dense float tensor (HSWQ-owned).

    Thin wrapper over the ``hswq::dequantize_nvfp4`` custom op (compile-safe).
    """
    if output_type is None:
        output_type = torch.bfloat16

    if not isinstance(per_tensor_scale, torch.Tensor):
        per_tensor_scale = torch.tensor(
            per_tensor_scale, device=qx.device, dtype=torch.float32
        )
    if per_tensor_scale.device != qx.device:
        per_tensor_scale = per_tensor_scale.to(device=qx.device)

    return _dequantize_nvfp4_op(
        qx,
        per_tensor_scale,
        block_scales,
        _DTYPE_TO_CODE[output_type],
        bool(hi_first),
    )


def hswq_scaled_mm_nvfp4(
    a_qdata,
    w_qdata,
    *,
    tensor_scale_a,
    tensor_scale_b,
    block_scale_a,
    block_scale_b,
    bias=None,
    out_dtype=None,
    alpha: Optional[object] = None,
    orig_m: Optional[int] = None,
    orig_n: Optional[int] = None,
    out=None,
):
    """HSWQ ConvRot-NVFP4 GEMM: dequant both sides → ``a @ w.T`` (+ bias).

    Never calls ``comfy_kitchen`` ``scaled_mm_nvfp4`` / CUBLAS blockwise FP4.
    ``alpha`` is ignored (scales live inside dequant).
    """
    import torch

    _ = alpha
    if out_dtype is None:
        out_dtype = torch.bfloat16

    if isinstance(tensor_scale_a, torch.nn.Parameter):
        tensor_scale_a = tensor_scale_a.data
    if isinstance(tensor_scale_b, torch.nn.Parameter):
        tensor_scale_b = tensor_scale_b.data
    if isinstance(a_qdata, torch.nn.Parameter):
        a_qdata = a_qdata.data
    if isinstance(w_qdata, torch.nn.Parameter):
        w_qdata = w_qdata.data
    if isinstance(block_scale_a, torch.nn.Parameter):
        block_scale_a = block_scale_a.data
    if isinstance(block_scale_b, torch.nn.Parameter):
        block_scale_b = block_scale_b.data

    a_dq = dequantize_nvfp4(
        a_qdata, tensor_scale_a, block_scale_a, output_type=out_dtype
    )
    w_dq = dequantize_nvfp4(
        w_qdata, tensor_scale_b, block_scale_b, output_type=out_dtype
    )
    result = torch.mm(a_dq, w_dq.t())

    bias_arg = bias
    if bias is None or (isinstance(bias, torch.Tensor) and bias.numel() == 0):
        bias_arg = None
    elif isinstance(bias, torch.nn.Parameter):
        bias_arg = bias.data
    if bias_arg is not None:
        result = result + bias_arg.to(dtype=result.dtype, device=result.device)

    if orig_m is None:
        orig_m = int(a_qdata.shape[0])
    if orig_n is None:
        orig_n = int(w_qdata.shape[0])
    if result.shape[0] != orig_m or result.shape[1] != orig_n:
        result = result[:orig_m, :orig_n]

    if out is not None:
        if out.shape != result.shape or out.dtype != result.dtype or out.device != result.device:
            raise ValueError("out buffer shape/dtype/device mismatch")
        out.copy_(result)
        return out
    return result


def bake_nvfp4_weight_inplace(module, weight_qt, out_dtype):
    """One-shot NVFP4 → dense float bake; drop packed QT (no dual VRAM).

    Prior ``_hswq_nvfp4_w_dequant`` cache kept packed ``_qdata`` **and** a full
    BF16/FP16 matrix → peak VRAM ≈ BF16 (or worse) while still paying ConvRot +
    dequant cost. Bake replaces ``module.weight`` with a plain Parameter and
    releases the QuantizedTensor so only one residency remains.
    """
    import torch
    from comfy_kitchen.tensor.base import QuantizedTensor
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout

    cur = module.weight
    if isinstance(cur, torch.nn.Parameter):
        cur_data = cur.data
    else:
        cur_data = cur
    if not isinstance(cur_data, QuantizedTensor):
        # Already baked; align dtype for this forward if needed.
        if cur_data.dtype != out_dtype:
            return cur_data.to(dtype=out_dtype)
        return cur_data

    # Prefer layout-owned dequant (CUDA path when kitchen provides it); HSWQ
    # LUT unpack is the fallback if dequantize is unavailable.
    try:
        w_f = weight_qt.dequantize()
        if not isinstance(w_f, torch.Tensor):
            raise TypeError("dequantize did not return a Tensor")
        w_f = w_f.to(dtype=out_dtype)
    except Exception:
        w_qdata, scale_b, block_scale_b = TensorCoreNVFP4Layout.get_plain_tensors(
            weight_qt
        )
        orig_n = int(weight_qt._params.orig_shape[0])
        orig_k = int(weight_qt._params.orig_shape[1])
        w_f = dequantize_nvfp4(w_qdata, scale_b, block_scale_b, output_type=out_dtype)
        if w_f.shape[0] != orig_n or w_f.shape[1] != orig_k:
            w_f = w_f[:orig_n, :orig_k].contiguous()

    w_f = w_f.detach().contiguous()
    module.weight = torch.nn.Parameter(w_f, requires_grad=False)
    if hasattr(module, "_hswq_nvfp4_w_dequant"):
        try:
            delattr(module, "_hswq_nvfp4_w_dequant")
        except Exception:
            module._hswq_nvfp4_w_dequant = None
    return module.weight.data


def dequantize_weight_cached(module, weight_qt, out_dtype):
    """Backward-compatible alias → ``bake_nvfp4_weight_inplace`` (no dual cache)."""
    return bake_nvfp4_weight_inplace(module, weight_qt, out_dtype)
```

### 5-2. `win_utf8_patch.py` (new file, full text)

```python
"""Force UTF-8 as the process-wide default text encoding (Japanese Windows cp932 fix).

``torch.compile`` → ``torch._inductor`` reads kernel templates (``.py.jinja``)
and source files through the builtin ``open()`` with the *locale* default
encoding. On Japanese Windows the locale is cp932, so reading any UTF-8 file
raises::

    UnicodeDecodeError: 'cp932' codec can't decode byte 0x94 in position ...:
    illegal multibyte sequence

which is fatal inside ``torch.compile`` (e.g. ``torch/_inductor/utils.py``
``load_template``, reached via ``load_kernel_template`` / ``load_flex_template``).

This module replicates Python's ``PYTHONUTF8=1`` (UTF-8 mode) behaviour at
runtime, which is otherwise impossible to enable after interpreter startup:

1. set ``PYTHONUTF8`` / ``PYTHONIOENCODING`` env vars (helps subprocesses);
2. reconfigure stdin/stdout/stderr to UTF-8 (avoids ``UnicodeEncodeError``
   when torch/logging print non-ASCII to a cp932 console);
3. patch ``io.open`` / ``builtins.open`` so a text-mode ``open(path)`` with no
   explicit ``encoding`` defaults to UTF-8 instead of the locale.

Idempotent — safe to import from multiple entry points (``prestartup_script.py``
and the node ``__init__.py``).
"""
from __future__ import annotations

import builtins
import io
import os
import sys

_MARKER = "_hswq_utf8_patched"


def _apply() -> None:
    # Idempotency: if we (or another copy of this module) already patched,
    # builtins.open carries our marker.
    if getattr(builtins.open, _MARKER, False):
        return

    # 1) Environment — picked up by subprocesses and late readers.
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    # 2) Stdio streams → UTF-8.
    for _name in ("stdin", "stdout", "stderr"):
        _stream = getattr(sys, _name, None)
        if _stream is None:
            continue
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # 3) Default text encoding for open()/io.open()/pathlib → UTF-8.
    _orig_open = io.open

    def _utf8_open(
        file,
        mode="r",
        buffering=-1,
        encoding=None,
        errors=None,
        newline=None,
        closefd=True,
        opener=None,
    ):
        # Only change the *default* (encoding=None) for text mode on a path.
        # File objects / descriptors / binary mode / explicit encoding pass
        # through untouched — strictly additive, no behaviour regression.
        if (
            encoding is None
            and isinstance(mode, str)
            and "b" not in mode
            and isinstance(file, (str, bytes, os.PathLike))
        ):
            encoding = "utf-8"
        return _orig_open(file, mode, buffering, encoding, errors, newline, closefd, opener)

    _utf8_open.__name__ = "open"
    _utf8_open.__qualname__ = "open"
    _utf8_open.__doc__ = _orig_open.__doc__
    _utf8_open.__module__ = "io"
    _utf8_open.__wrapped__ = _orig_open
    setattr(_utf8_open, _MARKER, True)

    io.open = _utf8_open
    builtins.open = _utf8_open


_apply()
```

### 5-3. `prestartup_script.py` (added section, full text)

The following was inserted immediately after the existing `import sys` (before `_ROOT = ...`). All other existing code is unchanged.

```python
# --- cp932 → UTF-8: apply before any file is read with the locale encoding ---
import importlib.util as _ilu
import os as _os

_utf8_patch_spec = _ilu.spec_from_file_location(
    "win_utf8_patch",
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "win_utf8_patch.py"),
)
_utf8_patch_mod = _ilu.module_from_spec(_utf8_patch_spec)
_utf8_patch_spec.loader.exec_module(_utf8_patch_mod)
del _ilu, _os, _utf8_patch_spec, _utf8_patch_mod
```

### 5-4. `__init__.py` (added section, full text)

The following was inserted at the very top of the file (before `import logging`). All other existing code is unchanged.

```python
# --- cp932 → UTF-8: apply before torch (and any file read) uses the locale encoding ---
import importlib.util as _ilu
import os as _os

_utf8_patch_spec = _ilu.spec_from_file_location(
    "win_utf8_patch",
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "win_utf8_patch.py"),
)
_utf8_patch_mod = _ilu.module_from_spec(_utf8_patch_spec)
_utf8_patch_spec.loader.exec_module(_utf8_patch_mod)
del _ilu, _os, _utf8_patch_spec, _utf8_patch_mod
```

### 5-5. `nodes\krea2_convrot_nvfp4\nvfp4_addmm_patch.py` (restored, no net change)

Edits mistakenly made during investigation (changing the `orig_n` index, rewriting the fallback with `d_mat2.t()`, etc.) were reverted to the correct original state. **The final state is identical to the original implementation** (no net change). The two correct points are:

```python
        try:
            result = hswq_scaled_mm_nvfp4(
                input_qdata,
                weight_qdata,
                tensor_scale_a=scale_a,
                tensor_scale_b=scale_b,
                block_scale_a=block_scale_a,
                block_scale_b=block_scale_b,
                bias=bias,
                out_dtype=out_dtype,
            )
            orig_m = mat1._params.orig_shape[0]
            orig_n = mat2._params.orig_shape[1]   # ← [1] is correct (weight has orig_shape=(K, N))
            return _slice_to_original_shape(result, orig_m, orig_n)
        except (RuntimeError, TypeError) as e:
            note_scaled_mm_failure(e)
            return torch.addmm(*dequantize_args((bias, mat1, mat2)))
```

---

## ⑥ What the Code Means

### 6-1. The custom-op conversion in `nvfp4_gemm.py`

- **`_DTYPE_TO_CODE` / `_CODE_TO_DTYPE`**: A custom-op schema can only accept `int` / `bool` (plus Tensor) as non-tensor arguments, so `torch.dtype` is mapped to an integer code both ways. 0=float32, 1=float16, 2=bfloat16, 3=float64.
- **`_dequantize_nvfp4_op_impl`**: The body of the original `dequantize_nvfp4` (the LUT decode that only runs on real tensors). `lo = qx & 0x0F` / `hi = qx >> 4` extract the two FP4 nibbles packed in a uint8; `F.embedding` looks up the E2M1 LUT to decode to real values; `from_blocked` unswizzles the cuBLAS block scales and multiplies them in. **Numerics are identical to before the change**.
- **`@torch.library.custom_op("hswq::dequantize_nvfp4", mutates_args=())`**: Registers the decode as a custom operator so dynamo / AOT treat it as a single opaque node instead of tracing its body.
- **`@_dequantize_nvfp4_op.register_fake`**: The meta implementation called during fake tracing (metadata collection at compile time). It only computes the output shape `(*qx.shape[:-1], qx.shape[-1] * 2)` from the input `qx` and returns an empty tensor. Because it **never executes the real body (including `F.embedding`)**, the dispatch re-entry that caused `Mixing fake modes` does not happen.
- **`dequantize_nvfp4` (public wrapper)**: Keeps the original signature. Converts `output_type` to a code and normalizes `per_tensor_scale` (non-tensor float → tensor, device alignment) before calling the custom op. Callers (`hswq_scaled_mm_nvfp4`, `bake_nvfp4_weight_inplace`, etc.) are unaffected.

### 6-2. What `win_utf8_patch.py` means

- **`_MARKER` idempotency guard**: Checks the marker attribute on `builtins.open` and does nothing if already patched. This prevents double-wrapping when loaded from both `prestartup_script.py` and `__init__.py`.
- **`os.environ.setdefault("PYTHONUTF8", "1")`**: Makes Python subprocesses launched after startup start in UTF-8 mode (it cannot change the current process's `open()` default, but it does affect children). `PYTHONIOENCODING=utf-8` covers child stdio.
- **`sys.std{in,out,err}.reconfigure(encoding="utf-8", errors="replace")`**: Prevents non-ASCII output (Japanese logs, etc.) from raising `UnicodeEncodeError` even when the console is cp932.
- **`_utf8_open` (monkey-patch)**: Replaces `io.open` / `builtins.open` with a wrapper that injects `encoding="utf-8"` when encoding is unspecified, in text mode, on a path. This turns `load_template`'s `open(template_dir / f"{name}.py.jinja")` into a UTF-8 read, eliminating the cp932 decode error. The condition is tightly scoped (path only, text only, `encoding=None` only), so the only behavioural change versus the original is "default to UTF-8" — binary read/write and explicit encodings are unaffected.
- **`_utf8_open.__wrapped__ = _orig_open` etc.**: Aligns `__name__` / `__qualname__` / `__doc__` / `__module__` with the original function so the wrapper behaves like `open` in tracebacks and `inspect`.

### 6-3. The loading snippet in `prestartup_script.py` / `__init__.py`

- `importlib.util.spec_from_file_location("win_utf8_patch", <absolute path>)` loads the sibling `win_utf8_patch.py` as a module **independent of the package name**. The node folder is named `ComfyUI-HSWQ-Loader-and-Tools` (contains hyphens), which is not a valid Python import name, so it avoids relying on `from . import ...` and loads reliably by absolute path.
- `del _ilu, _os, ...` cleans up the temporary names so the module namespace stays clean.
- **`prestartup_script.py`**: ComfyUI's `main.py` (`execute_prestartup_script`) runs this before `__init__.py`, so the UTF-8 patch is applied at the earliest possible point in the process.
- **`__init__.py`**: Placing it **before `import torch`** guarantees UTF-8 is already active the first time torch reads a file (safety net).

---

## Verification (on the actual RTX 5060 Ti)

- `dequantize_nvfp4` (custom op) vs an independent reference LUT: **`max|diff| = 0.0` (`torch.equal` matches)** → numerics unchanged
- `torch.compile(linear_qt, backend="inductor", dynamic=False)`: **runs without crashing** `(256, 1024) bf16`
- **Without** the `PYTHONUTF8` / `PYTHONIOENCODING` env vars, bare `open()` reads as UTF-8 and `torch.compile` passes through `load_template`

## How to Apply

`prestartup_script.py` only runs at startup, so **a ComfyUI restart is required**.
