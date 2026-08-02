<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><font color="#4b5563"><b>EN</b></font></td>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/zhmd/v3.3.3.md"><font color="#ffffff"><b>中文</b></font></a></td>
  </tr>
</table>


## 1. What was wrong

### Pack shape

A hybrid Z Image checkpoint (example: ConvRot NVFP4 × 120 + INT8 protect ConvRot × 60) has two families:

| Kind | `comfy_quant.format` | Offline weights | Online acts |
|------|----------------------|-----------------|-------------|
| NVFP4 ConvRot | `nvfp4` | Rotated, packed | Parity rotates acts (`_hswq_nvfp4_convrot`) |
| INT8 protect ConvRot | `int8_tensorwise` + `convrot` | Rotated INT8 QT | Must match Conv2d contract |

At **v3.3.2**, LoRA bake for **NVFP4** ConvRot Linears already existed (`convert_weight` unrotate → apply LoRA → `set_weight` re-rotate). **INT8 protect ConvRot Linears did not bake correctly.**

### Failure symptoms (measured logs)

1. Dynamic VRAM attaches a `LowVramPatch` per LoRA key (example: `patches=180`).
2. After the NVFP4 pass bakes ~120 keys, **~60 INT8 protect keys remain** (`patches_left=60`).
3. The INT8 Dynamic bake in `patches/comfy_quant_int8.py` **often does not fire** on this hybrid path.
4. Leftover LowVramPatch on ConvRot INT8 protect breaks generation (dead LoRA / noise).

### Root cause (two layers)

**A. Bake rotation basis**  
Kitchen `dequantize` **already unrotates** when `Params.convrot=True`. Bake therefore requires:

1. Clear `Params.convrot` at load (same as Conv2d).
2. Set `_hswq_int8_convrot` so parity rotates acts.
3. In `convert_weight`, unrotate the rotated-basis float **once**.
4. In `set_weight`, after re-rotate → requant, keep **`Params.convrot=False` again**.

Baking with `Params.convrot=True` → kitchen + bake double-unrotate → **dead LoRA**.  
Clearing Params then letting requant restore `True` while parity still rotates → **double act rotate → noise**.

**B. Success visibility**  
A shared sample-log quota and session-total-only evidence lines buried INT8 success under NVFP4. Pass-delta EVIDENCE, peer verdicts on both sides, and sample keys make success unambiguous.

### Wrong direction (recorded, then corrected)

`f338b44` tried bake-only with `Params.convrot=True`. That **doubles** kitchen unrotate. Correct path is `80d5f2a`: **arm like Conv2d** (clear Params; keep False after requant).

### Success evidence line (validated)

```
EVIDENCE (...): NVFP4_LORA_BAKE_OK INT8_PROTECT_LORA_BAKE_OK this_pass |
  nvfp4 convert_unrotate=120 set_rerotate=120 nvfp4_baked=120 sample_nvfp4_keys=[…] |
  int8_protect convert_unrotate=60 set_rerotate=60 int8_baked=60 sample_int8_keys=[…] |
  session_total nv_c/s=120/120 int8_c/s=60/60
```

Empty re-bake with `patches=0` (e.g. VAE) must **not** reprint OK (pass-delta gate).

---

## 2. Files added or changed

Diff vs `v3.3.2` (LoRA-related only):

| File | Role |
|------|------|
| `nodes/nvfp4/nvfp4_conf.py` | Detect INT8 protect ConvRot from `comfy_quant` |
| `nodes/nvfp4/nvfp4_forward.py` | Unrotate / re-rotate both kinds, Params clear, EVIDENCE |
| `nodes/nvfp4/comfy_quant_nvfp4.py` | Re-export snapshot / evidence |
| `nodes/zimage_nvfp4/nvfp4_comfy_parity.py` | Arm INT8 protect like Conv2d at load; parity rotates both |
| `nodes/zimage_nvfp4/nvfp4_lora_bake.py` | Dual bake, sample keys, pass-delta EVIDENCE |

### Commits (`v3.3.2` → HEAD, paths above)

| Commit | Summary |
|--------|---------|
| `ea178ed` | Unrotate INT8 protect during hybrid LoRA bake |
| `3167ab9` | Arm INT8 protect ConvRot on hybrid NVFP4 packs |
| `f338b44` | Bake-only attempt (**wrong**; discarded later) |
| `80d5f2a` | Conv2d-twin arm, clear Params, keep False after requant |
| `bc89c43` | Per-kind counters + `INT8_PROTECT_LORA_BAKE_*` EVIDENCE |
| `d9dad05` | Pass-delta EVIDENCE only (no fake OK on empty pass) |
| `d77ff3f` | Peer `NVFP4_LORA_BAKE_*` + sample keys |

---

## 3. Full text of added / rewritten code

Below is the **full text of every function / block added or rewritten** for this fix (tree at `d77ff3f`). TC forward paths outside LoRA bake are omitted. Canonical full files live at the repository paths above.

### 3.1 `nodes/nvfp4/nvfp4_conf.py` — new helpers

```python
def is_int8_tensorwise_conf(conf: Optional[dict]) -> bool:
    return isinstance(conf, dict) and str(conf.get("format") or "").lower() == "int8_tensorwise"


def int8_convrot_flags_from_conf(conf: Optional[dict]) -> tuple[bool, int]:
    """Return (enabled, groupsize) for INT8 protect ConvRot comfy_quant.

    Do **not** reuse ``convrot_flags_from_conf`` — that helper is NVFP4-only and
    always returns False for ``int8_tensorwise``. Used by load arm to set
    ``_hswq_int8_convrot`` and clear kitchen ``Params.convrot`` (Conv2d twin).
    """
    if not is_int8_tensorwise_conf(conf):
        return False, 256
    params_conf = conf.get("params", {})
    if not isinstance(params_conf, dict):
        params_conf = {}
    enabled = bool(conf.get("convrot", False)) or bool(params_conf.get("convrot", False))
    if not enabled:
        return False, 256
    gs = int(conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256)) or 256)
    return True, gs
```

### 3.2 `nodes/nvfp4/comfy_quant_nvfp4.py` — re-exports

```python
from .nvfp4_forward import (
    attach_nvfp4_linear_lora_bake,
    log_nvfp4_lora_bake_evidence,
    make_nvfp4_linear_forward,
    nvfp4_forward_stats,
    nvfp4_lora_bake_counters,
    reset_nvfp4_forward_stats,
    reset_nvfp4_lora_log_counters,
    snapshot_nvfp4_lora_bake_counters,
)

# in __all__:
    "log_nvfp4_lora_bake_evidence",
    "snapshot_nvfp4_lora_bake_counters",
```

### 3.3 `nodes/nvfp4/nvfp4_forward.py` — LoRA bake stack (full)

```python
# Counters for bench / diagnostics (reset per run if needed)
_TC_HITS = 0
_DEQUANT_FALLBACKS = 0
# Per-kind totals (always incremented) + per-kind sample log caps.
# Shared max of 8 hid all int8_protect samples (nvfp4 filled the quota first).
_LORA_CONVERT_TOTAL = {"nvfp4": 0, "int8_protect": 0}
_LORA_SET_TOTAL = {"nvfp4": 0, "int8_protect": 0}
_LORA_CONVERT_LOGGED = {"nvfp4": 0, "int8_protect": 0}
_LORA_SET_LOGGED = {"nvfp4": 0, "int8_protect": 0}
_LORA_KIND_LOG_MAX = 4
# Bump when convert_weight / set_weight ConvRot LoRA bake changes.
# v2: also unrotate/re-rotate INT8 protect ConvRot (``_hswq_int8_convrot``).
# Hybrid ZI packs = ConvRot NVFP4 + ConvRot INT8 protect — both need bake basis.
# v3: bake-time fallback if load arm missed and Params.convrot still True on INT8 QT.
# v4: (reverted) bake-only with Params.convrot — WRONG: kitchen dequant already
#     unrotates when Params.convrot=True → double unrotate → LoRA dead.
# v5: Conv2d twin — arm flag + clear Params; after set_weight keep Params=False
#     (noise was requant restoring Params while parity still rotated).
# v6: per-kind LoRA bake counters + EVIDENCE log (int8_protect must be visible).
# v7: pass-delta EVIDENCE only (no stale OK spam on empty re-bake / VAE load).
# v8: peer NVFP4_LORA_BAKE_* verdict + sample_nvfp4_keys (same weight as INT8).
_NVFP4_LORA_BAKE_VER = 8


def reset_nvfp4_lora_log_counters() -> None:
    for d in (
        _LORA_CONVERT_TOTAL,
        _LORA_SET_TOTAL,
        _LORA_CONVERT_LOGGED,
        _LORA_SET_LOGGED,
    ):
        for k in d:
            d[k] = 0


def _lora_bake_kind(module) -> str:
    if getattr(module, "_hswq_nvfp4_convrot", False):
        return "nvfp4"
    return "int8_protect"


def nvfp4_lora_bake_counters() -> dict:
    """Totals for convert unrotate / set re-rotate by ConvRot kind."""
    return {
        "convert_unrotate_nvfp4": int(_LORA_CONVERT_TOTAL.get("nvfp4", 0)),
        "convert_unrotate_int8_protect": int(_LORA_CONVERT_TOTAL.get("int8_protect", 0)),
        "set_rerotate_nvfp4": int(_LORA_SET_TOTAL.get("nvfp4", 0)),
        "set_rerotate_int8_protect": int(_LORA_SET_TOTAL.get("int8_protect", 0)),
    }


def snapshot_nvfp4_lora_bake_counters() -> dict:
    """Copy of totals for pass-delta EVIDENCE (before bake → after bake)."""
    return dict(nvfp4_lora_bake_counters())


def _counter_delta(before: dict | None, after: dict | None) -> dict:
    b = before or {}
    a = after or nvfp4_lora_bake_counters()
    keys = (
        "convert_unrotate_nvfp4",
        "convert_unrotate_int8_protect",
        "set_rerotate_nvfp4",
        "set_rerotate_int8_protect",
    )
    return {k: int(a.get(k, 0)) - int(b.get(k, 0)) for k in keys}


def _lora_bake_side_verdict(prefix: str, baked: int, convert_n: int, set_n: int) -> str:
    """Peer verdict for one ConvRot kind (NVFP4 or INT8 protect)."""
    match = convert_n == set_n == int(baked)
    if int(baked) > 0 and match and convert_n > 0:
        return f"{prefix}_OK"
    if int(baked) > 0 and not match:
        return f"{prefix}_MISMATCH"
    if int(baked) == 0 and convert_n == 0:
        return f"{prefix}_N/A"
    return f"{prefix}_MISSING"


def _fmt_sample_keys(label: str, keys: list | None) -> str:
    if not keys:
        return ""
    shown = ", ".join(str(k) for k in keys[:3])
    return f" {label}=[{shown}]"


def log_nvfp4_lora_bake_evidence(
    tag: str = "",
    *,
    before: dict | None = None,
    nvfp4_baked: int = 0,
    int8_baked: int = 0,
    sample_nvfp4_keys: list | None = None,
    sample_int8_keys: list | None = None,
    force: bool = False,
) -> str | None:
    """Emit pass-scoped EVIDENCE only when this bake pass actually ran hooks.

    NVFP4 and INT8 protect are peer sides (same verdict shape + key samples).
    Returns the message if emitted, else None (silent skip for empty re-bake).
    """
    after = nvfp4_lora_bake_counters()
    d = _counter_delta(before, after)
    i8c = d["convert_unrotate_int8_protect"]
    i8s = d["set_rerotate_int8_protect"]
    nvc = d["convert_unrotate_nvfp4"]
    nvs = d["set_rerotate_nvfp4"]
    this_pass_hooks = (i8c + i8s + nvc + nvs) > 0
    this_pass_layer = (int(nvfp4_baked) + int(int8_baked)) > 0
    if not force and not this_pass_hooks and not this_pass_layer:
        return None

    nv_verdict = _lora_bake_side_verdict(
        "NVFP4_LORA_BAKE", int(nvfp4_baked), nvc, nvs
    )
    i8_verdict = _lora_bake_side_verdict(
        "INT8_PROTECT_LORA_BAKE", int(int8_baked), i8c, i8s
    )

    suffix = f" ({tag})" if tag else ""
    nv_samples = _fmt_sample_keys("sample_nvfp4_keys", sample_nvfp4_keys)
    i8_samples = _fmt_sample_keys("sample_int8_keys", sample_int8_keys)
    msg = (
        f"[HSWQ ConvRot LoRA] EVIDENCE{suffix}: {nv_verdict} {i8_verdict} "
        f"this_pass | "
        f"nvfp4 convert_unrotate={nvc} set_rerotate={nvs} "
        f"nvfp4_baked={int(nvfp4_baked)}{nv_samples} | "
        f"int8_protect convert_unrotate={i8c} set_rerotate={i8s} "
        f"int8_baked={int(int8_baked)}{i8_samples} | "
        f"session_total nv_c/s="
        f"{after['convert_unrotate_nvfp4']}/"
        f"{after['set_rerotate_nvfp4']} "
        f"int8_c/s="
        f"{after['convert_unrotate_int8_protect']}/"
        f"{after['set_rerotate_int8_protect']}"
    )
    logger.info(msg)
    print(msg, flush=True)
    return msg


def _clear_int8_qt_params_convrot(module) -> bool:
    """Force Params.convrot=False on INT8 QT (must stay False after requant)."""
    import dataclasses

    try:
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return False
    w = getattr(module, "weight", None)
    qt = w if isinstance(w, QuantizedTensor) else getattr(w, "data", None)
    if qt is None or not isinstance(qt, QuantizedTensor):
        return False
    params = getattr(qt, "_params", None)
    if params is None or not bool(getattr(params, "convrot", False)):
        return False
    new_params = dataclasses.replace(params, convrot=False)
    try:
        object.__setattr__(qt, "_params", new_params)
        return True
    except Exception:
        pass
    try:
        qt._params = new_params
        return True
    except Exception:
        return False


def _linear_convrot_lora_groupsize(module) -> int | None:
    """Groupsize for offline ConvRot Linear LoRA bake, or None if not ConvRot.

    Hybrid Z Image packs:
      - NVFP4: ``_hswq_nvfp4_convrot`` (Params cleared; parity rotates).
      - INT8 protect: ``_hswq_int8_convrot`` (Params cleared; parity rotates).
        Kitchen dequant with Params.convrot=True already unrotates — bake must
        see Params=False so convert unrotates rotated-basis float once.
    """
    if getattr(module, "_hswq_nvfp4_convrot", False):
        return int(getattr(module, "_hswq_nvfp4_convrot_groupsize", 256) or 256)
    if getattr(module, "_hswq_int8_convrot", False):
        return int(getattr(module, "_hswq_int8_convrot_groupsize", 256) or 256)
    # Late arm if load missed: Params.convrot still True on INT8 QT.
    try:
        from comfy.quant_ops import QuantizedTensor

        w = getattr(module, "weight", None)
        qt = w if isinstance(w, QuantizedTensor) else getattr(w, "data", None)
        if qt is None or not isinstance(qt, QuantizedTensor):
            return None
        layout_cls = getattr(qt, "_layout_cls", None) or ""
        if isinstance(layout_cls, type):
            layout_cls = getattr(layout_cls, "__name__", "") or ""
        if str(layout_cls) != "TensorWiseINT8Layout":
            return None
        params = getattr(qt, "_params", None)
        if params is None or not bool(getattr(params, "convrot", False)):
            return None
        gs = int(getattr(params, "convrot_groupsize", 256) or 256)
        module._hswq_int8_convrot = True
        module._hswq_int8_convrot_groupsize = gs
        _clear_int8_qt_params_convrot(module)
        return gs
    except Exception:
        return None


def make_nvfp4_linear_convert_weight(stock_convert_weight):
    """Wrap Linear.convert_weight: dequant then unrotate ConvRot weights for LoRA bake.

    Handles ConvRot NVFP4 **and** ConvRot INT8 protect (hybrid Z Image packs).
    Clear INT8 Params.convrot **before** stock dequant — kitchen already
    unrotates when Params.convrot=True (would double-unrotate with bake).
    """
    import torch
    from comfy.quant_ops import QuantizedTensor

    def convert_weight(self, weight, inplace=False, **kwargs):
        # Arm / clear Params before dequant (Conv2d twin).
        gs = _linear_convrot_lora_groupsize(self)
        if callable(stock_convert_weight):
            out = stock_convert_weight(self, weight, inplace=inplace, **kwargs)
        elif isinstance(weight, QuantizedTensor):
            out = weight.dequantize()
        else:
            out = weight
        if gs is None:
            gs = _linear_convrot_lora_groupsize(self)
        if gs is not None and out is not None and getattr(out, "ndim", 0) == 2:
            h = build_hadamard(gs, device="cpu", dtype=torch.float32)
            out = unrotate_weight_linear(out, h, gs)
            kind = _lora_bake_kind(self)
            _LORA_CONVERT_TOTAL[kind] = int(_LORA_CONVERT_TOTAL.get(kind, 0)) + 1
            if int(_LORA_CONVERT_LOGGED.get(kind, 0)) < _LORA_KIND_LOG_MAX:
                _LORA_CONVERT_LOGGED[kind] = int(_LORA_CONVERT_LOGGED.get(kind, 0)) + 1
                logger.info(
                    "[HSWQ ConvRot LoRA] Linear.convert_weight #%s (%s): unrotate "
                    "gs=%s in=%s/%s -> out=%s/%s",
                    _LORA_CONVERT_TOTAL[kind],
                    kind,
                    gs,
                    type(weight).__name__,
                    getattr(weight, "dtype", None),
                    type(out).__name__,
                    getattr(out, "dtype", None),
                )
        return out

    convert_weight._hswq_nvfp4_lora_bake_ver = _NVFP4_LORA_BAKE_VER  # type: ignore[attr-defined]
    return convert_weight


def make_nvfp4_linear_set_weight(stock_set_weight):
    """Wrap Linear.set_weight: re-rotate ConvRot float weights before requant.

    Handles ConvRot NVFP4 **and** ConvRot INT8 protect (hybrid Z Image packs).
    After INT8 requant, force Params.convrot=False (parity rotates acts;
    kitchen must not also rotate).
    """
    import torch

    def set_weight(
        self,
        weight,
        inplace_update=False,
        seed=None,
        return_weight=False,
        **kwargs,
    ):
        gs = _linear_convrot_lora_groupsize(self)
        if gs is not None and getattr(weight, "ndim", 0) == 2:
            h = build_hadamard(gs, device="cpu", dtype=torch.float32)
            weight = rotate_weight_linear(weight, h, gs)
            kind = _lora_bake_kind(self)
            _LORA_SET_TOTAL[kind] = int(_LORA_SET_TOTAL.get(kind, 0)) + 1
            if int(_LORA_SET_LOGGED.get(kind, 0)) < _LORA_KIND_LOG_MAX:
                _LORA_SET_LOGGED[kind] = int(_LORA_SET_LOGGED.get(kind, 0)) + 1
                logger.info(
                    "[HSWQ ConvRot LoRA] Linear.set_weight #%s (%s): re-rotate "
                    "gs=%s shape=%s layout=%s",
                    _LORA_SET_TOTAL[kind],
                    kind,
                    gs,
                    tuple(weight.shape) if hasattr(weight, "shape") else "?",
                    getattr(self, "layout_type", None),
                )
        out = stock_set_weight(
            self,
            weight,
            inplace_update=inplace_update,
            seed=seed,
            return_weight=return_weight,
            **kwargs,
        )
        # Requant may restore Params.convrot — keep cleared for INT8 protect.
        if getattr(self, "_hswq_int8_convrot", False):
            _clear_int8_qt_params_convrot(self)
        return out

    set_weight._hswq_nvfp4_lora_bake_ver = _NVFP4_LORA_BAKE_VER  # type: ignore[attr-defined]
    return set_weight


def attach_nvfp4_linear_lora_bake(Lin) -> bool:
    """Ensure MixedPrecision Linear has ConvRot LoRA convert/set wraps. Returns True if applied/upgraded."""
    applied = False
    cvt = getattr(Lin, "convert_weight", None)
    if callable(cvt) and getattr(cvt, "_hswq_nvfp4_lora_bake_ver", 0) < _NVFP4_LORA_BAKE_VER:
        Lin.convert_weight = make_nvfp4_linear_convert_weight(cvt)
        applied = True
    sw = getattr(Lin, "set_weight", None)
    if callable(sw) and getattr(sw, "_hswq_nvfp4_lora_bake_ver", 0) < _NVFP4_LORA_BAKE_VER:
        Lin.set_weight = make_nvfp4_linear_set_weight(sw)
        applied = True
    return applied
```

### 3.4 `nodes/zimage_nvfp4/nvfp4_comfy_parity.py` — arm / clear / parity / load (full)

```python
def _chain_has_int8_protect_in_load(fn) -> bool:
    """True if load chain already arms INT8 protect ConvRot after stock load."""
    cur = fn
    seen = set()
    for _ in range(8):
        if cur is None or id(cur) in seen:
            return False
        seen.add(id(cur))
        if getattr(cur, "_hswq_int8_protect_in_load", False):
            return True
        if getattr(cur, "_hswq_int8_protect_arm_v2", False):
            return True
        if getattr(cur, "_hswq_int8_decode_patched", False):
            cur = _closure_named(cur, "original_load")
            continue
        if _is_tc_full_load(cur):
            cur = _closure_named(cur, "_orig_load")
            continue
        if getattr(cur, "_hswq_nvfp4_comfy_only", False):
            return False
        return False
    return False


def _ensure_int8_protect_arm_overlay() -> None:
    """Hot-refresh: wrap current load so INT8 protect Linears get act-rotate arm.

    No-op when ``_load_quantized_module_comfy_only`` already has
    ``_hswq_int8_protect_in_load`` (fresh install path).
    """
    try:
        import comfy.ops as ops
    except Exception:
        return
    cur = ops._load_quantized_module
    if _chain_has_int8_protect_in_load(cur):
        return
    from ..nvfp4.nvfp4_conf import decode_comfy_quant_conf

    def _load_int8_protect_arm_overlay(
        module,
        super_load,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
        load_extra_params=False,
    ):
        conf = decode_comfy_quant_conf(state_dict.get(f"{prefix}comfy_quant"))
        out = cur(
            module,
            super_load,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
            load_extra_params=load_extra_params,
        )
        _arm_int8_protect_convrot_after_stock_load(module, conf)
        return out

    _load_int8_protect_arm_overlay._hswq_int8_protect_arm_v2 = True  # type: ignore[attr-defined]
    ops._load_quantized_module = _load_int8_protect_arm_overlay
    _console(
        "[HSWQ NVFP4] comfy_parity: INT8 protect ConvRot arm overlay installed "
        "(hot refresh; online act rotate for protect Linears)"
    )


def _is_int8_tensorwise_convrot_conf(conf) -> bool:
    """True for INT8 protect Linear layers stamped with ConvRot offline rotate."""
    from ..nvfp4.nvfp4_conf import int8_convrot_flags_from_conf

    enabled, _gs = int8_convrot_flags_from_conf(conf)
    return bool(enabled)


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
                    need_rebuild = (
                        h.device != input.device
                        or h.dtype != input.dtype
                        or int(h.numel()) == 0
                        or (bool(input.is_cuda) and not bool(h.is_cuda))
                    )
                    if not need_rebuild:
                        st = (
                            h.untyped_storage()
                            if hasattr(h, "untyped_storage")
                            else h.storage()
                        )
                        need_rebuild = int(st.nbytes()) == 0
                except Exception:
                    need_rebuild = True
            if need_rebuild:
                h = build_hadamard(gs, device=input.device, dtype=input.dtype)
                self._hswq_nvfp4_parity_H = h
            input = rotate_last_dim(input, h, gs)
        return stock_forward(self, input, *args, **kwargs)

    forward_parity._hswq_nvfp4_convrot_parity = True  # type: ignore[attr-defined]
    return forward_parity


def _clear_int8_qt_params_convrot(module) -> bool:
    """Force ``Params.convrot=False`` on INT8 QT weight (Conv2d-style).

    ``layout_params`` on Parameter is unreliable for QuantizedTensor; clear
    ``qt._params`` via dataclasses.replace. Leaving Params.convrot=True while
    ``_hswq_int8_convrot`` is set double-rotates acts (parity + kitchen).
    """
    import dataclasses

    try:
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return False
    w = getattr(module, "weight", None)
    qt = w if isinstance(w, QuantizedTensor) else getattr(w, "data", None)
    if qt is None or not isinstance(qt, QuantizedTensor):
        return False
    params = getattr(qt, "_params", None)
    if params is None or not bool(getattr(params, "convrot", False)):
        return False
    new_params = dataclasses.replace(params, convrot=False)
    try:
        object.__setattr__(qt, "_params", new_params)
        return True
    except Exception:
        pass
    try:
        qt._params = new_params
        return True
    except Exception:
        return False


def _arm_int8_protect_convrot_after_stock_load(module, conf) -> None:
    """Arm INT8 protect ConvRot Linear like Conv2d: flag + clear Params.convrot.

    Kitchen ``dequantize_int8_convrot_weight`` already unrotates when
    Params.convrot=True — LoRA bake must see Params=False so convert gets
    rotated-basis float and unrotates once. Online act rotate is parity
    (``_hswq_int8_convrot``). Requant must keep Params.convrot=False
    (see ``nvfp4_forward`` set_weight).
    """
    global _LOAD_INT8_CONVROT_ARMED
    from ..nvfp4.nvfp4_conf import int8_convrot_flags_from_conf

    enabled, gs = int8_convrot_flags_from_conf(conf)
    if not enabled:
        return
    module._hswq_int8_convrot = True
    module._hswq_int8_convrot_groupsize = int(gs)
    cleared = _clear_int8_qt_params_convrot(module)
    _LOAD_INT8_CONVROT_ARMED += 1
    if _LOAD_INT8_CONVROT_ARMED <= 4 or _LOAD_INT8_CONVROT_ARMED % 20 == 0:
        _console(
            f"[HSWQ NVFP4][diag] arm INT8 protect ConvRot "
            f"#{_LOAD_INT8_CONVROT_ARMED} gs={gs} params_cleared={cleared}"
        )


# Inside _load_quantized_module_comfy_only (after stock load):
        if is_nvfp4_conf(conf):
            _arm_convrot_after_stock_load(module, conf)
        else:
            _arm_int8_protect_convrot_after_stock_load(module, conf)

# Flag on that load wrapper:
    _load_quantized_module_comfy_only._hswq_int8_protect_in_load = True
```

### 3.5 `nodes/zimage_nvfp4/nvfp4_lora_bake.py` — dual bake + EVIDENCE (changed surface, full)

Hook version:

```python
_BAKE_HOOK_VER = 7
```

NVFP4 pass (sample keys; defer INT8 rem) full:

```python
def bake_nvfp4_convrot_patches_on_dynamic_patcher(patcher, device_to) -> dict:
    """Bake LoRA into ConvRot NVFP4 Linears after ModelPatcherDynamic.load."""
    stats = {
        "baked_nvfp4": 0,
        "candidates": 0,
        "skipped_no_set": 0,
        "skipped_not_nvfp4": 0,
        "skipped_not_convrot": 0,
        "cleared_already": 0,
        "unresolved": 0,
        "sample_nvfp4_keys": [],
    }
    if not getattr(patcher, "patches", None):
        return stats
    try:
        import comfy.model_patcher as mp
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return stats

    global _SKIP_SAMPLE_LOGS
    already = _get_baked_key_set(patcher.model)
    uuid = getattr(patcher, "patches_uuid", None)
    prev_uuid = getattr(patcher.model, "_hswq_zi_nvfp4_baked_uuid", None)
    if prev_uuid is not None and prev_uuid != uuid:
        already.clear()

    # Group keys by module so LowVramPatch clear happens once per module.
    by_module: dict[str, list] = {}
    modules: dict[str, object] = {}
    for key, module_path, param_key, module in _iter_patch_weight_keys(patcher):
        stats["candidates"] += 1
        if key in already:
            attr = param_key + "_lowvram_function"
            if getattr(module, attr, None) is not None:
                setattr(module, attr, None)
            try:
                del patcher.patches[key]
            except KeyError:
                pass
            stats["cleared_already"] += 1
            continue
        if not _module_is_nvfp4_convrot(module):
            stats["skipped_not_convrot"] += 1
            if _SKIP_SAMPLE_LOGS < _SKIP_SAMPLE_MAX:
                w, _, _ = mp.get_key_weight(patcher.model, key)
                qt = _qt_payload(w, QuantizedTensor)
                _SKIP_SAMPLE_LOGS += 1
                params = getattr(qt, "_params", None) if qt is not None else None
                params_convrot = bool(getattr(params, "convrot", False)) if params else False
                _console(
                    f"[HSWQ ZI NVFP4 LoRA] nv_pass_defer_int8_rem sample "
                    f"#{_SKIP_SAMPLE_LOGS}: {key} layout={_qt_layout_name(qt)!r} "
                    f"nvfp4_convrot={getattr(module, '_hswq_nvfp4_convrot', False)} "
                    f"int8_convrot={getattr(module, '_hswq_int8_convrot', False)} "
                    f"params_convrot={params_convrot} "
                    f"(not a failure — baked in INT8 rem pass)"
                )
            continue
        weight, set_func, _convert_func = mp.get_key_weight(patcher.model, key)
        if weight is None:
            continue
        if not _qt_is_nvfp4(weight, QuantizedTensor):
            stats["skipped_not_nvfp4"] += 1
            continue
        if set_func is None:
            stats["skipped_no_set"] += 1
            _console(
                f"[HSWQ ZI NVFP4 LoRA] WARN cannot bake {key}: "
                "NVFP4 QT but no set_weight"
            )
            continue
        by_module.setdefault(module_path, []).append((param_key, key))
        modules[module_path] = module

    for module_path, keys_to_bake in by_module.items():
        n = _bake_keys_on_module(
            patcher, modules[module_path], keys_to_bake, device_to, already
        )
        stats["baked_nvfp4"] += n
        if n > 0 and len(stats["sample_nvfp4_keys"]) < 3:
            for _pk, full_key in keys_to_bake:
                if full_key not in stats["sample_nvfp4_keys"]:
                    stats["sample_nvfp4_keys"].append(full_key)
                if len(stats["sample_nvfp4_keys"]) >= 3:
                    break

    if stats["baked_nvfp4"] > 0:
        patcher.model._hswq_zi_nvfp4_baked_uuid = uuid

    return stats
```

INT8 rem pass full:

```python
def bake_remaining_quant_patches_on_dynamic_patcher(patcher, device_to) -> dict:
    """Bake leftover QT LoRA (ConvRot INT8 protect etc.) that NVFP4 pass skipped.

    Hybrid packs: NVFP4 ConvRot is baked first; INT8 protect ConvRot Linears
    use ``_hswq_int8_convrot`` + cleared Params (Conv2d twin) via
    ``Linear.convert_weight`` / ``set_weight`` (``_NVFP4_LORA_BAKE_VER`` >= 5).
    """
    stats = {
        "baked_int8": 0,
        "baked_other_qt": 0,
        "candidates": 0,
        "skipped_no_set": 0,
        "skipped_not_qt": 0,
        "cleared_already": 0,
        "sample_int8_keys": [],
    }
    if not getattr(patcher, "patches", None):
        return stats
    try:
        import comfy.model_patcher as mp
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return stats

    already = _get_baked_key_set(patcher.model)
    uuid = getattr(patcher, "patches_uuid", None)

    by_module: dict[str, list] = {}
    modules: dict[str, object] = {}
    kinds: dict[str, str] = {}
    for key, module_path, param_key, module in _iter_patch_weight_keys(patcher):
        stats["candidates"] += 1
        if key in already:
            attr = param_key + "_lowvram_function"
            if getattr(module, attr, None) is not None:
                setattr(module, attr, None)
            try:
                del patcher.patches[key]
            except KeyError:
                pass
            stats["cleared_already"] += 1
            continue
        weight, set_func, _convert_func = mp.get_key_weight(patcher.model, key)
        if weight is None:
            continue
        qt = _qt_payload(weight, QuantizedTensor)
        if qt is None:
            stats["skipped_not_qt"] += 1
            continue
        if set_func is None:
            stats["skipped_no_set"] += 1
            _console(
                f"[HSWQ ZI NVFP4 LoRA] WARN cannot bake leftover {key}: "
                f"QT layout={_qt_layout_name(qt)!r} but no set_weight"
            )
            continue
        if module_path not in kinds:
            if _qt_is_int8_tensorwise(weight, QuantizedTensor):
                kinds[module_path] = "int8"
            elif _qt_is_nvfp4(weight, QuantizedTensor):
                kinds[module_path] = "nvfp4"
            else:
                kinds[module_path] = "other"
        by_module.setdefault(module_path, []).append((param_key, key))
        modules[module_path] = module

    for module_path, keys_to_bake in by_module.items():
        n = _bake_keys_on_module(
            patcher, modules[module_path], keys_to_bake, device_to, already
        )
        if kinds.get(module_path) == "int8":
            stats["baked_int8"] += n
            if len(stats["sample_int8_keys"]) < 3:
                for _pk, full_key in keys_to_bake:
                    if full_key not in stats["sample_int8_keys"]:
                        stats["sample_int8_keys"].append(full_key)
                    if len(stats["sample_int8_keys"]) >= 3:
                        break
        else:
            stats["baked_other_qt"] += n

    if stats["baked_int8"] > 0 or stats["baked_other_qt"] > 0:
        patcher.model._hswq_zi_nvfp4_baked_uuid = uuid
        if stats["baked_int8"] > 0:
            patcher.model._hswq_int8_baked_uuid = uuid

    return stats
```

Status + EVIDENCE full:

```python
def _dump_bake_status(
    nv_stats: dict,
    rem_stats: dict,
    patcher,
    reason: str,
    counters_before: dict | None = None,
) -> None:
    global _STATUS_LOGS
    nv_n = int(nv_stats.get("baked_nvfp4", 0) or 0)
    i8 = int(rem_stats.get("baked_int8", 0) or 0)
    # Empty re-bake / VAE: do not spam status or stale EVIDENCE.
    if nv_n == 0 and i8 == 0 and int(rem_stats.get("baked_other_qt", 0) or 0) == 0:
        return
    if _STATUS_LOGS >= _STATUS_LOG_MAX:
        return
    _STATUS_LOGS += 1
    left = len(getattr(patcher, "patches", None) or {})
    skip_i8_in_nv_pass = int(nv_stats.get("skipped_not_convrot", 0) or 0)
    uuid = getattr(patcher, "patches_uuid", None)
    uuid_s = f"{uuid}"[:8] if uuid is not None else "-"
    _console(
        "[HSWQ ZI NVFP4 LoRA] Dynamic.load bake "
        f"#{_STATUS_LOGS} ({reason}): "
        f"nvfp4_baked={nv_n} "
        f"int8_baked={i8} "
        f"other_qt_baked={rem_stats.get('baked_other_qt', 0)} "
        f"nv_candidates={nv_stats.get('candidates', 0)} "
        f"rem_candidates={rem_stats.get('candidates', 0)} "
        f"nv_pass_skip_int8_rem={skip_i8_in_nv_pass} "
        f"(INT8 rem baked separately as int8_baked) "
        f"patches_left={left} patches_uuid={uuid_s}"
    )
    try:
        from ..nvfp4.nvfp4_forward import log_nvfp4_lora_bake_evidence

        log_nvfp4_lora_bake_evidence(
            tag=f"bake#{_STATUS_LOGS}/{reason}",
            before=counters_before,
            nvfp4_baked=nv_n,
            int8_baked=i8,
            sample_nvfp4_keys=list(nv_stats.get("sample_nvfp4_keys") or []),
            sample_int8_keys=list(rem_stats.get("sample_int8_keys") or []),
        )
    except Exception as e:
        _console(f"[HSWQ ConvRot LoRA] EVIDENCE log failed: {e}")
    if left > 0:
        sample = list((getattr(patcher, "patches", None) or {}).keys())[:4]
        _console(
            f"[HSWQ ZI NVFP4 LoRA] WARN patches_left={left} after bake "
            f"sample_keys={sample}"
        )
```

Orchestration (snapshot before bake) full:

```python
def run_zimage_nvfp4_lora_bake_on_patcher(patcher, device_to=None, reason: str = "wrap") -> bool:
    """Bake NVFP4 ConvRot + leftover QT if this patcher is a ZI NVFP4 pack with LoRA."""
    model = getattr(patcher, "model", None)
    if model is None:
        return False
    diag = _nvfp4_convrot_diag(model)
    has_flag = bool(diag["has"])
    has_baked = bool(getattr(model, "_hswq_zi_nvfp4_baked_keys", None))
    n_patches = len(getattr(patcher, "patches", None) or {})
    if not has_flag and not has_baked:
        # Fallback: patches present and QT visible via get_key_weight
        if n_patches == 0 or not _patcher_has_quant_via_keys(patcher):
            return False
    if n_patches == 0 and not has_baked:
        return False
    if device_to is None:
        device_to = getattr(patcher, "load_device", None)
    try:
        from ..nvfp4.nvfp4_forward import snapshot_nvfp4_lora_bake_counters

        counters_before = snapshot_nvfp4_lora_bake_counters()
    except Exception:
        counters_before = None
    nv_stats = bake_nvfp4_convrot_patches_on_dynamic_patcher(patcher, device_to=device_to)
    rem_stats = bake_remaining_quant_patches_on_dynamic_patcher(
        patcher, device_to=device_to
    )
    _dump_bake_status(
        nv_stats, rem_stats, patcher, reason=reason, counters_before=counters_before
    )
    return True
```

`Dynamic.load` / `load_models_gpu` call `run_zimage_nvfp4_lora_bake_on_patcher` after stock load (`_BAKE_HOOK_VER = 7`). Canonical full module: `nodes/zimage_nvfp4/nvfp4_lora_bake.py`.

---

## 4. What it means

### Load time (parity)

| Action | Meaning |
|--------|---------|
| `int8_convrot_flags_from_conf` | Read ConvRot stamp on protect Linears (do not reuse NVFP4-only helper). |
| `_arm_int8_protect_convrot_after_stock_load` | Same contract as Conv2d: set `_hswq_int8_convrot`, **clear** `Params.convrot`. |
| Parity forward rotates if `nv` **or** `i8` | Match offline rotated weights to online acts; avoid kitchen double-rotate. |
| Hot-refresh overlay | Protect layers stay armed after DistOrch / reload. |

### Bake time (`convert_weight` / `set_weight`)

| Step | Meaning |
|------|---------|
| Clear Params before dequant | Kitchen must not unrotate; float stays rotated-basis until bake unrotates once. |
| Unrotate → LoRA → re-rotate → requant | Apply LoRA in **unrotated** space (same as NVFP4). |
| Clear Params again after requant | Requant must not revive kitchen rotate; parity owns act rotate. |
| `_NVFP4_LORA_BAKE_VER = 8` | Force re-wrap of MixedPrecision Linear when the contract changes. |

### Dual Dynamic bake

| Pass | Meaning |
|------|---------|
| NVFP4 first | Bake ConvRot NVFP4 keys; INT8 deferred (`skipped_not_convrot` ≠ failure). |
| INT8 rem second | Bake leftover QT (protect ConvRot); clear LowVramPatch / patches. |
| `patches_left=0` | Success condition for a full hybrid LoRA load. |

### EVIDENCE

| Field | Meaning |
|-------|---------|
| `this_pass` convert/set deltas | Hooks ran in **this** bake (not stale session totals). |
| `NVFP4_LORA_BAKE_OK` / `INT8_PROTECT_LORA_BAKE_OK` | Peer sides: baked == convert == set. |
| `sample_*_keys` | Concrete keys folded (auditability). |
| Silent on empty pass | No fake OK when `patches=0` (VAE / reload). |

### Contracts not to break

1. Hybrid = bake **both** ConvRot paths.  
2. INT8 protect arm = **Conv2d twin** (Params cleared; never bake-only with Params=True).  
3. `f338b44` direction is **wrong**; correct Params policy is `80d5f2a`.  
4. Do not edit `ComfyUI-master`; all of this lives under this custom-node tree.

---

## Quick verify

After loading a hybrid ZI pack + LoRA under Dynamic VRAM, check the console for:

1. Arm: `arm INT8 protect ConvRot #… params_cleared=True`
2. Bake: `nvfp4_baked=120 int8_baked=60 patches_left=0` (counts match the pack)
3. EVIDENCE: both `*_LORA_BAKE_OK`, matching convert/set, sample keys present
4. No EVIDENCE spam on a later load with `patches=0`

---

*This guide covers LoRA bake diffs from tag `v3.3.2` (`23ca013`) through `d77ff3f`.*

