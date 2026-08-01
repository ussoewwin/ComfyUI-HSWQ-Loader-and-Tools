"""Pooled NVFP4 act quantize + scaled_mm (HSWQ).

Stock ``ck.quantize_nvfp4`` / ``ck.scaled_mm_nvfp4`` allocate fresh tensors on
every Linear call. With ~18k TC hits per SDXL sample that allocation overhead
dominates wall time vs FP16. This module reuses CUDA buffers keyed by shape.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional

# (padded_rows, padded_cols, device_str) -> (qx uint8, sx_uint8)
# Safe to reuse: only live during one Linear forward (quantize → mm reads sync).
_ACT_Q_POOL: dict = {}
# (m, group_count, group_size, dtype, device_str) -> rotated
# Safe: consumed inside the same Linear before return.
_ROT_OUT_POOL: dict = {}
# CUDA Graph: shape-shared (weight copied each replay). LRU-capped to avoid OOM.
_GRAPH_CACHE: OrderedDict = OrderedDict()
_GRAPH_CACHE_MAX = 32
# Only graph small-M calls (microbench: NVFP4 mm loses to FP16 when M is small).
_GRAPH_MAX_M = 512
# NOTE: MM *output* must NOT be pooled — UNet residuals keep layer outputs alive
# across later layers; a reused buffer would corrupt activations.


def clear_nvfp4_cudagraphs() -> None:
    _GRAPH_CACHE.clear()


def clear_nvfp4_runtime_pools() -> None:
    _ACT_Q_POOL.clear()
    _ROT_OUT_POOL.clear()
    clear_nvfp4_cudagraphs()


def _dev_key(t) -> str:
    return str(t.device)


def rotate_last_dim_pooled(x, h_matrix, group_size: int):
    """Same as ``rotate_last_dim`` but reuses the matmul output buffer."""
    import torch

    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} not divisible by group_size {group_size}")
    group_count = features // group_size
    x_grouped = x.reshape(-1, group_count, group_size)
    if h_matrix.device != x.device or h_matrix.dtype != x.dtype:
        h_matrix = h_matrix.to(dtype=x.dtype, device=x.device)
    m = x_grouped.shape[0]
    key = (m, group_count, group_size, x.dtype, _dev_key(x))
    out = _ROT_OUT_POOL.get(key)
    if out is None or out.shape != x_grouped.shape:
        out = torch.empty_like(x_grouped)
        _ROT_OUT_POOL[key] = out
    torch.matmul(x_grouped, h_matrix, out=out)
    return out.reshape(orig_shape)


def quantize_nvfp4_act_pooled(
    x: "torch.Tensor",
    per_tensor_scale: "torch.Tensor",
    *,
    pad_16x: bool,
):
    """CUDA ``quantize_nvfp4`` with reused qx / block-scale buffers.

    Returns ``(qx, block_scale_f8e4m3, padded_rows, padded_cols)``.
    """
    import torch
    from comfy_kitchen.backends.cuda import _C, _wrap_for_dlpack, roundup

    if not x.is_contiguous():
        x = x.contiguous()

    orig_rows, orig_cols = x.shape
    if pad_16x:
        num_rows = roundup(orig_rows, 16)
        num_cols = roundup(orig_cols, 16)
    else:
        num_rows, num_cols = orig_rows, orig_cols
        if num_rows % 16 != 0 or num_cols % 16 != 0:
            raise ValueError(
                f"NVFP4 act dims must be divisible by 16 without pad_16x, "
                f"got {(orig_rows, orig_cols)}"
            )

    scale_rows = roundup(num_rows, 128)
    scale_cols = roundup(num_cols // 16, 4)
    key = (num_rows, num_cols, _dev_key(x))
    buf = _ACT_Q_POOL.get(key)
    if buf is None:
        qx = torch.empty(
            (num_rows, num_cols // 2),
            device=x.device,
            dtype=torch.uint8,
            memory_format=torch.contiguous_format,
        )
        sx_uint8 = torch.empty((scale_rows, scale_cols), device=x.device, dtype=torch.uint8)
        _ACT_Q_POOL[key] = (qx, sx_uint8)
    else:
        qx, sx_uint8 = buf

    # Zero only when padded tiles exist (kitchen always zeros — skip exact shapes).
    if (
        scale_rows > orig_rows
        or scale_cols > (orig_cols // 16)
        or num_rows > orig_rows
        or num_cols > orig_cols
    ):
        sx_uint8.zero_()

    if per_tensor_scale.dim() == 0:
        per_tensor_scale = per_tensor_scale.reshape(1)
    if per_tensor_scale.dtype != torch.float32:
        per_tensor_scale = per_tensor_scale.to(dtype=torch.float32)
    if per_tensor_scale.device != x.device:
        per_tensor_scale = per_tensor_scale.to(device=x.device)

    stream_ptr = torch.cuda.current_stream(x.device).cuda_stream
    _C.quantize_nvfp4(
        _wrap_for_dlpack(x),
        _wrap_for_dlpack(per_tensor_scale),
        _wrap_for_dlpack(qx),
        _wrap_for_dlpack(sx_uint8),
        0.0,  # epsilon
        pad_16x,
        True,  # hi_first
        stream_ptr,
    )
    sx = sx_uint8.view(torch.float8_e4m3fn)
    return qx, sx, num_rows, num_cols


def scaled_mm_nvfp4_pooled(
    a_qdata,
    w_qdata,
    *,
    tensor_scale_a,
    tensor_scale_b,
    block_scale_a,
    block_scale_b,
    bias,
    out_dtype,
    alpha: Optional["torch.Tensor"] = None,
    orig_m: Optional[int] = None,
    orig_n: Optional[int] = None,
    out: Optional["torch.Tensor"] = None,
):
    """cuBLAS NVFP4 GEMM (kitchen CUDA path). ``out`` optional for CUDA Graph."""
    import torch
    from comfy_kitchen.backends.cuda import (
        _C,
        _wrap_for_dlpack,
        get_cublas_workspace,
        roundup,
        DTYPE_TO_CODE,
    )

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

    if alpha is None:
        alpha = tensor_scale_a * tensor_scale_b
    elif isinstance(alpha, torch.nn.Parameter):
        alpha = alpha.data
    if alpha.dtype != torch.float32:
        alpha = alpha.to(torch.float32)
    if alpha.dim() == 0:
        alpha = alpha.reshape(1)

    m, k_a = a_qdata.shape
    n, k_b = w_qdata.shape
    if k_a != k_b:
        raise ValueError("Matrix dimensions do not match")
    if n % 8 != 0:
        raise ValueError("B tensor must have 8 alignment in N dimension")

    k = 2 * k_a
    block_length = 16
    if block_scale_a.dtype != torch.float8_e4m3fn:
        raise ValueError(f"Unsupported scale dtype: {block_scale_a.dtype}")

    roundup_m = roundup(m, 128)
    roundup_n = roundup(n, 128)
    roundup_sk = roundup(k // block_length, 4)
    if block_scale_a.size() != (roundup_m, roundup_sk):
        raise ValueError(f"Invalid A scale shape {tuple(block_scale_a.shape)}")
    if block_scale_b.size() != (roundup_n, roundup_sk):
        raise ValueError(f"Invalid B scale shape {tuple(block_scale_b.shape)}")

    if out is None:
        out = torch.empty(m, n, dtype=out_dtype, device=a_qdata.device)
    elif out.shape != (m, n) or out.dtype != out_dtype or out.device != a_qdata.device:
        raise ValueError("out buffer shape/dtype/device mismatch")

    if bias is None or (isinstance(bias, torch.Tensor) and bias.numel() == 0):
        bias_arg = torch.empty(0, device=a_qdata.device, dtype=torch.float16)
    else:
        if isinstance(bias, torch.nn.Parameter):
            bias = bias.data
        bias_arg = bias

    out_dtype_code = DTYPE_TO_CODE[out_dtype]
    stream_ptr = torch.cuda.current_stream(a_qdata.device).cuda_stream
    _C.cublas_gemm_blockwise_fp4(
        _wrap_for_dlpack(w_qdata),
        _wrap_for_dlpack(block_scale_b.view(torch.uint8)),
        _wrap_for_dlpack(a_qdata),
        _wrap_for_dlpack(block_scale_a.view(torch.uint8)),
        _wrap_for_dlpack(out),
        out_dtype_code,
        _wrap_for_dlpack(bias_arg),
        _wrap_for_dlpack(get_cublas_workspace()),
        False,  # accumulate
        _wrap_for_dlpack(alpha),
        stream_ptr,
    )

    if orig_m is None:
        orig_m = m
    if orig_n is None:
        orig_n = n
    if out.shape[0] != orig_m or out.shape[1] != orig_n:
        return out[:orig_m, :orig_n]
    return out


# CUDA Graph helpers (cache lives at module top as _GRAPH_CACHE).


def _run_quant_mm(
    x_2d,
    *,
    scale_a,
    w_qdata,
    weight_scale,
    block_scale_w,
    bias,
    out_dtype,
    alpha,
    pad_16x: bool,
    orig_m: int,
    orig_n: int,
    out,
):
    a_qdata, block_scale_a, _pr, _pc = quantize_nvfp4_act_pooled(
        x_2d, scale_a, pad_16x=pad_16x
    )
    return scaled_mm_nvfp4_pooled(
        a_qdata,
        w_qdata,
        tensor_scale_a=scale_a,
        tensor_scale_b=weight_scale,
        block_scale_a=block_scale_a,
        block_scale_b=block_scale_w,
        bias=bias,
        out_dtype=out_dtype,
        alpha=alpha,
        orig_m=orig_m,
        orig_n=orig_n,
        out=out,
    )


def nvfp4_quant_mm_cudagraph(
    x,
    *,
    w_qdata,
    weight_scale,
    block_scale_w,
    scale_a,
    bias,
    out_dtype,
    alpha,
    pad_16x: bool,
    orig_n: int,
    weight_key: int = 0,
    freeze_scales: bool = False,
):
    """
    Capture once / replay: CUDA quantize_nvfp4 + cuBLAS NVFP4 GEMM.

    Graphs are **shape-shared** (not per-weight): each replay copies weight /
    scales into static buffers. Cache is LRU-capped (``_GRAPH_CACHE_MAX``).
    Skips when ``M > _GRAPH_MAX_M`` (caller should use eager; large-M TC already
    beats FP16 without graphs).

    ``weight_key`` / ``freeze_scales`` kept for API compat; ignored for sharing.
    """
    import torch
    from comfy_kitchen.backends.cuda import roundup

    if x.dim() != 2:
        raise ValueError("nvfp4_quant_mm_cudagraph expects 2D x")
    m, k = int(x.shape[0]), int(x.shape[1])
    if m > _GRAPH_MAX_M:
        raise ValueError("nvfp4_quant_mm_cudagraph: M too large for shape-shared graph")
    n = int(w_qdata.shape[0])
    has_bias = bias is not None and not (
        isinstance(bias, torch.Tensor) and bias.numel() == 0
    )
    out_m = roundup(m, 16) if pad_16x else m
    # No weight_key: one graph per shape (copy weight each replay).
    key = (
        m,
        k,
        n,
        out_m,
        tuple(w_qdata.shape),
        tuple(block_scale_w.shape),
        str(x.dtype),
        str(out_dtype),
        str(x.device),
        bool(pad_16x),
        has_bias,
        int(orig_n),
    )

    entry = _GRAPH_CACHE.get(key)
    if entry is not None:
        _GRAPH_CACHE.move_to_end(key)
    else:
        while len(_GRAPH_CACHE) >= _GRAPH_CACHE_MAX:
            _GRAPH_CACHE.popitem(last=False)

        static_x = torch.empty(m, k, dtype=x.dtype, device=x.device)
        static_out = torch.empty(out_m, n, dtype=out_dtype, device=x.device)
        static_w = torch.empty_like(w_qdata)
        static_bs_w = torch.empty_like(block_scale_w)
        static_scale_a = torch.empty(1, dtype=torch.float32, device=x.device)
        static_scale_b = torch.empty(1, dtype=torch.float32, device=x.device)
        static_alpha = torch.empty(1, dtype=torch.float32, device=x.device)
        static_bias = (
            torch.empty_like(bias) if has_bias else None
        )

        def _as_f32_1(t):
            if isinstance(t, torch.nn.Parameter):
                t = t.data
            t = t.to(device=x.device, dtype=torch.float32).reshape(1)
            return t

        static_w.copy_(w_qdata)
        static_bs_w.copy_(block_scale_w)
        static_scale_a.copy_(_as_f32_1(scale_a))
        static_scale_b.copy_(_as_f32_1(weight_scale))
        static_alpha.copy_(_as_f32_1(alpha))
        bias_arg = static_bias if has_bias else None
        if has_bias:
            static_bias.copy_(bias)

        s = torch.cuda.Stream(device=x.device)
        s.wait_stream(torch.cuda.current_stream(x.device))
        with torch.cuda.stream(s):
            static_x.copy_(x)
            for _ in range(2):
                _run_quant_mm(
                    static_x,
                    scale_a=static_scale_a,
                    w_qdata=static_w,
                    weight_scale=static_scale_b,
                    block_scale_w=static_bs_w,
                    bias=bias_arg,
                    out_dtype=out_dtype,
                    alpha=static_alpha,
                    pad_16x=pad_16x,
                    orig_m=m,
                    orig_n=orig_n,
                    out=static_out,
                )
        torch.cuda.current_stream(x.device).wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _run_quant_mm(
                static_x,
                scale_a=static_scale_a,
                w_qdata=static_w,
                weight_scale=static_scale_b,
                block_scale_w=static_bs_w,
                bias=bias_arg,
                out_dtype=out_dtype,
                alpha=static_alpha,
                pad_16x=pad_16x,
                orig_m=m,
                orig_n=orig_n,
                out=static_out,
            )
        entry = (
            g,
            static_x,
            static_out,
            static_w,
            static_bs_w,
            static_scale_a,
            static_scale_b,
            static_alpha,
            static_bias,
            has_bias,
        )
        _GRAPH_CACHE[key] = entry

    (
        g,
        static_x,
        static_out,
        static_w,
        static_bs_w,
        static_scale_a,
        static_scale_b,
        static_alpha,
        static_bias,
        entry_has_bias,
    ) = entry

    def _copy_f32_1(dst, src):
        if isinstance(src, torch.nn.Parameter):
            src = src.data
        if src.dtype != torch.float32 or src.device != dst.device:
            src = src.to(device=dst.device, dtype=torch.float32)
        dst.copy_(src.reshape(1))

    static_x.copy_(x)
    static_w.copy_(w_qdata)
    static_bs_w.copy_(block_scale_w)
    _copy_f32_1(static_scale_a, scale_a)
    _copy_f32_1(static_scale_b, weight_scale)
    _copy_f32_1(static_alpha, alpha)
    if entry_has_bias:
        static_bias.copy_(bias)
    g.replay()
    return static_out[:m, :orig_n].clone()


# device -> float32 scalar 1/(F8_E4M3_MAX * F4_E2M1_MAX); avoids py-float / on CUDA amax
_INV_NVFP4_AMAX_DENOM: dict = {}
# device -> float32 ones(1) for missing checkpoint input_scale
_ONES_SCALE: dict = {}


def _device_ones_scale(device):
    import torch

    key = str(device)
    t = _ONES_SCALE.get(key)
    if t is None or t.device != device:
        t = torch.ones(1, device=device, dtype=torch.float32)
        _ONES_SCALE[key] = t
    return t


def _inv_amax_denom(device):
    import torch
    from comfy_kitchen.float_utils import F4_E2M1_MAX, F8_E4M3_MAX

    key = str(device)
    t = _INV_NVFP4_AMAX_DENOM.get(key)
    if t is None or t.device != device:
        t = torch.tensor(
            1.0 / (float(F8_E4M3_MAX) * float(F4_E2M1_MAX)),
            device=device,
            dtype=torch.float32,
        )
        _INV_NVFP4_AMAX_DENOM[key] = t
    return t


def ensure_act_scale(x, scale):
    """Return float32 scale tensor on ``x.device``.

    Checkpoint tensor / Parameter: cast once and return.
    ``None``: caller should use ``ensure_act_scale_cached`` (module cache).
    """
    import torch

    if scale is None:
        return _device_ones_scale(x.device)
    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, device=x.device, dtype=torch.float32)
    if scale.device != x.device or scale.dtype != torch.float32:
        scale = scale.to(device=x.device, dtype=torch.float32)
    if scale.dim() == 0:
        scale = scale.reshape(1)
    return scale


def ensure_act_scale_amax(x):
    """Online amax scale (GPU-only multiply)."""
    import torch

    s = torch.amax(x.abs()).to(dtype=torch.float32) * _inv_amax_denom(x.device)
    return s.reshape(1)


def ensure_act_scale_cached(module, x, scale):
    """Act scale with **one amax per module** when checkpoint omits input_scale.

    ``test.safetensors`` has zero ``input_scale`` keys. Doing amax on every
    Linear (~18k/sample) made NVFP4 slower than FP16. Freezing after the first
    forward keeps quality near online-amax while removing the steady-state cost.
    """
    import torch

    if getattr(module, "_hswq_nvfp4_scale_placeholder", False) or scale is None:
        cached = getattr(module, "_hswq_nvfp4_act_scale", None)
        if cached is not None and cached.device == x.device:
            return cached
        s = ensure_act_scale_amax(x)
        module._hswq_nvfp4_act_scale = s
        return s

    return ensure_act_scale(x, scale)
