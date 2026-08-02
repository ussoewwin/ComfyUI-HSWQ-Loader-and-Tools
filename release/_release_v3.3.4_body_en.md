<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><font color="#4b5563"><b>EN</b></font></td>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/zhmd/v3.3.4.md"><font color="#ffffff"><b>中文</b></font></a></td>
  </tr>
</table>

## ① What was wrong

Z Image / ZIT **ConvRot NVFP4** (and INT8 protect Linears on the same Comfy parity online-act-rotate path) keep a per-module Hadamard matrix on `module._hswq_nvfp4_parity_H` and reuse it every forward.

After a **Distorch** VRAM purge, Method 3 can empty / poison tensor storage (`t.data = empty` style) while Python still holds the same attribute. The **global** Hadamard cache in `nodes/nvfp4/nvfp4_hadamard.py` already rejected poisoned tensors with `_tensor_storage_ok` before reuse. The **module-local** path in `_make_convrot_parity_forward` did **not**. It only checked:

- `h.device` / `h.dtype` vs `input`
- `h.numel() == 0`
- CUDA vs non-CUDA mismatch
- and, as a late add-on, `storage.nbytes() == 0`

So a shell that still reported a matching device/dtype (and sometimes non-zero nbytes / numel shape that `_tensor_storage_ok` would still reject) could be reused. Online act rotate then ran with dead/garbage `H`. Generation 1 looked fine; **generation 2 and later** after purge decayed (noise / quality drop), matching the comment already in the clear path: 2nd→3rd→4th gen gradually worse.

**Broken reuse gate (before v3.3.4)** inside `forward_parity`:

```python
            h = getattr(self, "_hswq_nvfp4_parity_H", None)
            need_rebuild = True
            if h is not None:
                try:
                    need_rebuild = (
                        h.device != input.device
                        or h.dtype != input.dtype
                        or int(h.numel()) == 0
                        or (bool(input.is_cuda) and not bool(h.is_cuda))
                    )
                    if not need_rebuild:
                        # DistOrch Method 3 may empty storage while device/dtype
                        # still match — rebuild or 2nd gen becomes noise.
                        st = (
                            h.untyped_storage()
                            if hasattr(h, "untyped_storage")
                            else h.storage()
                        )
                        need_rebuild = int(st.nbytes()) == 0
                except Exception:
                    need_rebuild = True
```

---

## ② Files added / modified

| Path | Change |
|------|--------|
| `nodes/zimage_nvfp4/nvfp4_comfy_parity.py` | Module-local `H` reuse now calls `_tensor_storage_ok`; related comments use product spelling **Distorch** |
| `nodes/nvfp4/nvfp4_hadamard.py` | Docstrings: `DistOrch` → **Distorch** (helper body unchanged; already used by global cache) |
| `pyproject.toml` | Version `3.3.3` → `3.3.4` |
| `__init__.py` | `__version__` `3.3.3` → `3.3.4` |
| `changelog.md` | Version **3.3.4** entry |
| `zhmd/CHANGELOG.md` | Version **3.3.4** entry |
| `zhmd/v3.3.4.md` | Chinese release notes (this structure) |
| `release/_release_v3.3.4_body_en.md` | English GitHub Release body (this file) |

No new Python module files were added for the fix. The behavioral fix is only in `nvfp4_comfy_parity.py` (plus version / docs surfaces).

---

## ③ Full text of added / modified code

### 3.1 `nodes/nvfp4/nvfp4_hadamard.py` — `_tensor_storage_ok` (gate the parity path now shares)

```python
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
```

### 3.2 `nodes/nvfp4/nvfp4_hadamard.py` — `clear_hadamard_global_caches` docstring (spelling only)

```python
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
```

### 3.3 `nodes/zimage_nvfp4/nvfp4_comfy_parity.py` — full `_make_convrot_parity_forward` after fix

```python
def _make_convrot_parity_forward(stock_forward):
    """Stock MixedPrecision forward + online act rotate for ConvRot Linears.

    NVFP4: ``_hswq_nvfp4_convrot`` (Params.convrot cleared at load).
    INT8 protect: ``_hswq_int8_convrot`` (Params.convrot cleared at load —
    same as Conv2d). Kitchen must **not** see Params.convrot=True or
    int8_linear double-rotates with this path.
    """
    from ..nvfp4.nvfp4_hadamard import build_hadamard, rotate_last_dim

    def forward_parity(self, input, *args, **kwargs):
        nv = bool(getattr(self, "_hswq_nvfp4_convrot", False))
        i8 = bool(getattr(self, "_hswq_int8_convrot", False))
        if nv or i8:
            if nv:
                gs = int(getattr(self, "_hswq_nvfp4_convrot_groupsize", 256) or 256)
            else:
                gs = int(getattr(self, "_hswq_int8_convrot_groupsize", 256) or 256)
            h = getattr(self, "_hswq_nvfp4_parity_H", None)
            need_rebuild = True
            if h is not None:
                try:
                    from ..nvfp4.nvfp4_hadamard import _tensor_storage_ok

                    # Global cache already rejects poisoned H via
                    # _tensor_storage_ok; module-local H must use the same
                    # check. nbytes==0 alone misses emptied shells that still
                    # report device/dtype (Distorch Method 3), so 2nd+ gen
                    # rotates with garbage and quality decays.
                    need_rebuild = (
                        h.device != input.device
                        or h.dtype != input.dtype
                        or (bool(input.is_cuda) and not bool(h.is_cuda))
                        or not _tensor_storage_ok(h)
                    )
                except Exception:
                    need_rebuild = True
            if need_rebuild:
                h = build_hadamard(gs, device=input.device, dtype=input.dtype)
                self._hswq_nvfp4_parity_H = h
            input = rotate_last_dim(input, h, gs)
        return stock_forward(self, input, *args, **kwargs)

    forward_parity._hswq_nvfp4_convrot_parity = True  # type: ignore[attr-defined]
    return forward_parity
```

### 3.4 Related comment / docstring spelling in the same parity file (no behavior change)

`clear_nvfp4_parity_hadamard_caches` docstring and comments now say **Distorch** (not `DistOrch`). `_unwrap_stock_forward` docstring likewise. Bodies unchanged.

---

## ④ Meaning

| Piece | Meaning |
|-------|---------|
| `_tensor_storage_ok(t)` | Returns False if `t` is missing, empty (`numel` / `nbytes`), not a valid square Hadamard shape, or raises when probed — i.e. after Distorch Method 3 emptied or corrupted storage while a Python reference remains. |
| Global `build_hadamard` (already) | Would not return a poisoned cache entry; rebuilds when `_tensor_storage_ok` fails. |
| Old module gate | Could keep `_hswq_nvfp4_parity_H` when device/dtype looked fine and `nbytes != 0`, even when the tensor was still unsafe by `_tensor_storage_ok` rules → `rotate_last_dim` with garbage `H` on **2nd+ gens** after purge. |
| New `need_rebuild = … or not _tensor_storage_ok(h)` | Module-local reuse uses the **same** liveness rule as the global cache. If storage is poisoned, rebuild via `build_hadamard` and write a fresh `_hswq_nvfp4_parity_H`. |
| Dropped standalone `numel==0` / late `nbytes==0` branch | Covered inside `_tensor_storage_ok` (`numel` and `nbytes` checks plus shape). Avoids a thinner second policy that diverged from the global path. |
| `rotate_last_dim(input, h, gs)` | Unchanged contract: online act rotate for NVFP4 / INT8 protect ConvRot. Only the decision of **which `h` is trusted** changed. |

### Operator notes

1. Update to tag **v3.3.4**.
2. Restart ComfyUI completely.
3. Keep **General Purge VRAM V2** (`HSWQ` on) at workflow end when using HSWQ NVFP4 / INT8.
4. Confirm 2nd (and later) gens after purge no longer decay from poisoned module-local Hadamard reuse.

### Compatibility

| Item | Policy |
|------|--------|
| Scope | Z Image / ZIT **ConvRot NVFP4** / INT8 protect online rotate on **HSWQ ConvRot INT8/ConvRot NVFP4 UNet Loader** |
| Quantizer | **Only** [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization) |
| SDXL ConvRot NVFP4 | Unchanged (Checkpoint Loader + `nodes/nvfp4` TC product path) |
| ComfyUI-master | Not modified by this extension |
