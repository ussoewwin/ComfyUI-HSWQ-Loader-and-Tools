"""Hadamard helpers for FULL offline ConvRot + online act rotation (HSWQ).

ZI-port: ``_tensor_storage_ok`` poisoned-storage rejection,
``clear_hadamard_global_caches`` (Distorch Method 2c / parity clear), and
storage-ok gating inside ``build_hadamard`` / ``_h4``.
"""
from __future__ import annotations

import math

_HADAMARD_CACHE: dict = {}
_H4_CACHE: dict = {}


def _tensor_storage_ok(t) -> bool:
    """False after Distorch nuclear kill / empty-storage reuse (UAF risk)."""
    if t is None:
        return False
    try:
        if int(getattr(t, "numel", lambda: 0)()) <= 0:
            return False
        st = t.untyped_storage() if hasattr(t, "untyped_storage") else t.storage()
        if int(st.nbytes()) <= 0:
            return False
        # Shape must match a square Hadamard (or 4x4 h4); reject emptied shells.
        if getattr(t, "ndim", 0) == 2:
            if int(t.shape[0]) != int(t.shape[1]) or int(t.shape[0]) < 4:
                return False
        return True
    except Exception:
        return False


def clear_hadamard_global_caches() -> int:
    """Drop module-level Hadamard caches (Distorch Method 2c / parity clear).

    Method 3 may ``t.data = empty`` on tensors still referenced by these dicts.
    Returning them on the next gen rotates with dead/garbage ``H`` and quality
    decays as CUDA reallocates the freed region (2nd→3rd→4th gen worse).
    """
    n = len(_HADAMARD_CACHE) + len(_H4_CACHE)
    _HADAMARD_CACHE.clear()
    _H4_CACHE.clear()
    return n

def build_hadamard(size: int, device="cpu", dtype=None):
    """Build (and cache) a normalized Hadamard matrix.

    Always construct in float32 (CPU or GPU), then cast to ``dtype``.
    Building the Kronecker product directly in float16 destroys ConvRot
    orthonormality and collapses NVFP4 quality.
    """
    import torch

    if dtype is None:
        dtype = torch.float32
    device = torch.device(device) if not isinstance(device, torch.device) else device
    cache_key = (size, str(device), dtype)
    cached = _HADAMARD_CACHE.get(cache_key)
    if cached is not None and _tensor_storage_ok(cached):
        return cached
    if cached is not None:
        _HADAMARD_CACHE.pop(cache_key, None)
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")

    master_key = (size, str(device), torch.float32)
    h_matrix = _HADAMARD_CACHE.get(master_key)
    if h_matrix is None or not _tensor_storage_ok(h_matrix):
        if h_matrix is not None:
            _HADAMARD_CACHE.pop(master_key, None)
        h4 = torch.tensor(
            [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
            dtype=torch.float32,
            device=device,
        )
        h_matrix = h4
        current_size = 4
        while current_size < size:
            h_matrix = torch.kron(h_matrix, h4)
            current_size *= 4
        h_matrix = h_matrix / (size**0.5)
        _HADAMARD_CACHE[master_key] = h_matrix
    if dtype != torch.float32:
        h_matrix = h_matrix.to(dtype=dtype)
    _HADAMARD_CACHE[cache_key] = h_matrix
    return h_matrix


def _h4(device, dtype):
    import torch

    key = (str(device), dtype)
    h = _H4_CACHE.get(key)
    if h is None or not _tensor_storage_ok(h):
        if h is not None:
            _H4_CACHE.pop(key, None)
        h = torch.tensor(
            [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
            dtype=dtype,
            device=device,
        )
        _H4_CACHE[key] = h
    return h


def _apply_kron_h4_unnorm(x2d, size: int):
    """Right-multiply by unnormalized Kronecker power of h4 (same as build_hadamard).

    ``x2d`` shape ``(M, size)`` with ``size == 4**k``. Equivalent to
    ``x2d @ kron_power(h4)`` before the ``/sqrt(size)`` normalization.
    """
    import torch

    if size == 4:
        return torch.matmul(x2d, _h4(x2d.device, x2d.dtype))
    p = size // 4
    # H_size = H_p ⊗ h4  (left-associated kron growth in build_hadamard)
    x = x2d.reshape(-1, p, 4)
    y = torch.matmul(x, _h4(x2d.device, x2d.dtype))  # apply h4 on last dim
    # apply H_p on the middle dim: for each of 4 cols, (M,p) @ H_p
    yt = y.transpose(1, 2).reshape(-1, p)
    yt = _apply_kron_h4_unnorm(yt, p)
    z = yt.reshape(-1, 4, p).transpose(1, 2)
    return z.reshape(-1, size)


def rotate_last_dim(x, h_matrix, group_size: int):
    import torch

    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} not divisible by group_size {group_size}")
    group_count = features // group_size
    x_grouped = x.reshape(-1, group_count, group_size)
    if h_matrix.device == x.device and h_matrix.dtype == x.dtype:
        h = h_matrix
    else:
        h = h_matrix.to(dtype=x.dtype, device=x.device)
    return torch.matmul(x_grouped, h).reshape(orig_shape)


def rotate_weight_linear(weight, h_matrix, group_size: int):
    """Offline Linear: W_rot = W @ H^T (group-wise along in_features)."""
    import torch

    if getattr(weight, "ndim", 0) != 2:
        raise ValueError(f"Linear weight must be 2D, got ndim={getattr(weight, 'ndim', None)}")
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features {in_features} not divisible by group_size {group_size}"
        )
    group_count = in_features // group_size
    weight_grouped = weight.view(out_features, group_count, group_size)
    h_t = h_matrix.T.to(dtype=weight.dtype, device=weight.device)
    return torch.matmul(weight_grouped, h_t).reshape(weight.shape)


def unrotate_weight_linear(weight, h_matrix, group_size: int):
    """Inverse of rotate_weight_linear: W = W_rot @ H (for LoRA float space)."""
    import torch

    if getattr(weight, "ndim", 0) != 2:
        raise ValueError(f"Linear weight must be 2D, got ndim={getattr(weight, 'ndim', None)}")
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features {in_features} not divisible by group_size {group_size}"
        )
    group_count = in_features // group_size
    weight_grouped = weight.view(out_features, group_count, group_size)
    h = h_matrix.to(dtype=weight.dtype, device=weight.device)
    return torch.matmul(weight_grouped, h).reshape(weight.shape)


def rotate_last_dim_fast(x, group_size: int):
    """Same math as ``rotate_last_dim`` + ``build_hadamard``, O(n log n) butterflies.

    Computes butterfly additions in float32 to eliminate bfloat16 accumulation
    rounding errors during online activation rotation.
    """
    import torch

    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} not divisible by group_size {group_size}")
    orig_dtype = x.dtype
    flat = x.reshape(-1, group_size).float()
    y = _apply_kron_h4_unnorm(flat, group_size)
    y = y * (group_size**-0.5)
    return y.to(dtype=orig_dtype).reshape(orig_shape)
