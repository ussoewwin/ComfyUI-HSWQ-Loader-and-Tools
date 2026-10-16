"""
ComfyUI core-safe patches for native comfy_quant INT8 (int8_tensorwise).

Upstream MixedPrecisionOps only quant-loads Linear / Embedding / MoE.
SD UNet INT8 checkpoints also store Conv2d weights as int8 + comfy_quant, which
fails with: Only Tensors of floating point and complex dtype can require gradients.

Also normalizes bare-string / double-encoded comfy_quant JSON some exporters write.

LoRA: native Linear already has convert_weight + set_weight (dequant → bake →
requant, same idea as BobJohnson24/ComfyUI-INT8-Fast). Injected Conv2d must
mirror that set_weight; without it ModelPatcher falls back to rounding into
int8 and LoRA deltas on Conv layers vanish.

Vendored under nodes/sd1.5 int8 for SD1.5 ConvRot INT8 loading.
Runtime monkey-patch only — do not permanently edit ComfyUI-master, and do not
import this module from ComfyUI-nunchaku-unofficial-loader (copy lives here).
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import threading

logger = logging.getLogger(__name__)
_PATCHES_APPLIED = False

# LoRA bake path logs (rate-limited so console stays readable)
_LORA_CONVERT_LOG_MAX = 0  # quiet; Status dump is enough
_LORA_SET_LOG_MAX = 0
_LORA_PATCHER_LOG_MAX = 0  # per-key bake lines off; Status dump is enough
_lora_convert_logs = 0
_lora_set_logs = 0
_lora_patcher_logs = 0
_lora_patcher_stats = {
    "calls": 0,
    "with_set_func": 0,
    "without_set_func": 0,
    "with_convert_func": 0,
}

# LoRA key attach / skip accounting (last load_lora_for_models call)
_lora_attach_last = {
    "lora_name": "",
    "strength_model": None,
    "strength_clip": None,
    "lora_file_keys": 0,
    "mapped_keys": 0,
    "applied_unet": 0,
    "applied_clip": 0,
    "applied_unet_keys": [],
    "applied_clip_keys": [],
    "not_mapped": [],
    "mapped_but_not_attached": [],
    "add_patches_skipped_unet": [],
}
# One entry per load_lora_for_models call (stacked loaders → multiple entries)
_lora_attach_history = []
# key -> "requant" | "int8_round" recorded during bake
_lora_bake_by_key = {}
# Set by LoraLoader.load_lora wrap (and cleared after attach)
_current_lora_name = None
_current_lora_strength_model = None
_current_lora_strength_clip = None
_lora_shape_skips = []  # list of (lora_name, key, reason)
_LORA_SKIP_PRINT_MAX = 40


def _console(msg: str) -> None:
    """Always visible in ComfyUI console (print + INFO)."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        try:
            print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)
        except Exception:
            pass
    try:
        logger.info(msg)
    except Exception:
        pass


def record_lora_shape_skip(lora_name: str, key: str, reason: str) -> None:
    """Called from LoraDiff reshape/numel skip path."""
    _lora_shape_skips.append((str(lora_name), str(key), str(reason)))


def _basename_lora(name: str) -> str:
    if not name:
        return name
    return os.path.basename(str(name).replace("\\", "/"))


# WeightAdapterBase class attrs — NOT filenames (was the lora=lora bug)
_ADAPTER_TYPE_NAMES = frozenset({"lora", "loha", "lokr", "oft", "boft", "glora"})


def _looks_like_lora_filename(s) -> bool:
    if not isinstance(s, str):
        return False
    s = s.strip()
    if not s or s.lower() in _ADAPTER_TYPE_NAMES:
        return False
    low = s.lower()
    if low.endswith((".safetensors", ".pt", ".ckpt", ".bin", ".sft")):
        return True
    if "/" in s or "\\" in s:
        return True
    # Short folder-relative names without extension still count as filenames
    if len(s) >= 2 and not s.startswith("diffusion_model"):
        return True
    return False


def _lora_line(msg: str) -> None:
    """One visible console line (print only — no print+logger twin)."""
    print(msg, flush=True)


def _slot_skip_count(entry: dict) -> int:
    return len(entry.get("not_mapped") or []) + len(
        entry.get("mapped_but_not_attached") or []
    )


def _slot_applied_count(entry: dict) -> int:
    return int(entry.get("applied_unet") or 0) + int(entry.get("applied_clip") or 0)


def _format_lora_slot_line(slot_i: int, entry: dict, include_bake: bool = False) -> str:
    """lora_name / applied_keys / skipped_keys — always present."""
    name = entry.get("lora_name") or "(unknown)"
    sm = entry.get("strength_model")
    sc = entry.get("strength_clip")
    u = int(entry.get("applied_unet") or 0)
    c = int(entry.get("applied_clip") or 0)
    applied = u + c
    skip = _slot_skip_count(entry)
    parts = [
        f"Slot {slot_i}:",
        f"lora_name='{name}'",
        f"applied_keys={applied} (unet={u} clip={c})",
        f"skipped_keys={skip}",
    ]
    if sm is not None:
        parts.append(f"strength_model={sm}")
    if sc is not None:
        parts.append(f"strength_clip={sc}")
    if include_bake:
        verdict, rq, ir, nb = _per_lora_bake_verdict(entry)
        parts.append(f"bake rq={rq} ir={ir} nb={nb}")
        if verdict == "OK_requant":
            parts.append("→ APPLIED ✓")
        elif verdict == "BROKEN_int8_round":
            parts.append("→ BROKEN ✗")
        elif verdict == "N/A_CLIP_only":
            parts.append("→ CLIP_only ✓")
        else:
            parts.append(f"→ {verdict}")
    else:
        if applied > 0:
            parts.append("→ APPLIED ✓")
        else:
            parts.append("→ SKIPPED ✗")
    return f"[HSWQ LoRA Status] {' | '.join(parts)}"


def _log_lora_slot_attach(entry: dict) -> None:
    """Emit one Status line immediately when a LoRA is attached (any loader)."""
    n = len(_lora_attach_history)
    if n == 1:
        _lora_line("[HSWQ LoRA Status] Processing LoRA slot(s):")
    _lora_line(_format_lora_slot_line(n, entry, include_bake=False))
    _lora_line(
        f"[HSWQ LoRA Status]   file_keys={entry.get('lora_file_keys', 0)} "
        f"mapped={entry.get('mapped_keys', 0)} "
        f"not_mapped={len(entry.get('not_mapped') or [])} "
        f"mapped_not_attached={len(entry.get('mapped_but_not_attached') or [])}"
    )


def _set_current_lora_name(name, strength_model=None, strength_clip=None) -> None:
    """Store real filename/UI name; never store adapter type 'lora'."""
    global _current_lora_name, _current_lora_strength_model, _current_lora_strength_clip
    if _looks_like_lora_filename(name):
        _current_lora_name = _basename_lora(name)
    if strength_model is not None:
        _current_lora_strength_model = strength_model
    if strength_clip is not None:
        _current_lora_strength_clip = strength_clip


def _path_is_under_loras_dir(path: str) -> bool:
    """True if path is inside any registered loras/ folder (any loader)."""
    if not path:
        return False
    try:
        import folder_paths

        bases = folder_paths.get_folder_paths("loras") or []
    except Exception:
        bases = []
    norm = os.path.normcase(os.path.abspath(str(path)))
    for base in bases:
        try:
            b = os.path.normcase(os.path.abspath(str(base)))
            if norm == b or norm.startswith(b + os.sep):
                return True
        except Exception:
            continue
    # Fallback when folder list not ready yet
    low = str(path).replace("\\", "/").lower()
    return "/loras/" in low or low.endswith("/loras")


def _resolve_lora_name(loaded_patches=None) -> str:
    """Filename for the LoRA currently being attached (any loader → common hooks)."""
    global _current_lora_name
    if _looks_like_lora_filename(_current_lora_name):
        return _basename_lora(_current_lora_name)

    try:
        import inspect

        # Common local names used by many LoRA loader nodes / helpers
        keys = (
            "lora_name",
            "lora_path",
            "lora",
            "path",
            "filename",
            "file_path",
            "lora_file",
            "name",
        )
        for frame in inspect.stack()[1:24]:
            loc = frame.frame.f_locals
            for key in keys:
                cand = loc.get(key)
                if _looks_like_lora_filename(cand):
                    return _basename_lora(cand)
            # Widget-style dicts: {'lora': '<file>', 'on': True, 'strength': ...}
            for cand in loc.values():
                if not isinstance(cand, dict):
                    continue
                ui = cand.get("lora")
                if _looks_like_lora_filename(ui) and (
                    "strength" in cand or "on" in cand or "strengthTwo" in cand
                ):
                    return _basename_lora(ui)
    except Exception:
        pass

    return f"unknown_lora#{len(_lora_attach_history) + 1}"


def reset_int8_lora_log_counters() -> None:
    global _lora_convert_logs, _lora_set_logs, _lora_patcher_logs
    global _current_lora_name, _current_lora_strength_model, _current_lora_strength_clip
    _lora_convert_logs = 0
    _lora_set_logs = 0
    _lora_patcher_logs = 0
    _lora_patcher_stats.update(
        calls=0, with_set_func=0, without_set_func=0, with_convert_func=0
    )
    _lora_shape_skips.clear()
    _lora_attach_history.clear()
    _lora_bake_by_key.clear()
    _current_lora_name = None
    _current_lora_strength_model = None
    _current_lora_strength_clip = None
    _lora_attach_last.update(
        lora_name="",
        strength_model=None,
        strength_clip=None,
        lora_file_keys=0,
        mapped_keys=0,
        applied_unet=0,
        applied_clip=0,
        applied_unet_keys=[],
        applied_clip_keys=[],
        not_mapped=[],
        mapped_but_not_attached=[],
        add_patches_skipped_unet=[],
    )
    dump_int8_lora_bake_stats._dumped_this_load = False


def summarize_int8_lora_capability(model) -> dict:
    """Scan loaded MODEL / diffusion_model and print LoRA hook readiness."""
    try:
        from comfy.ops import QuantizedTensor
    except ImportError:
        QuantizedTensor = type(None)

    diffusion = model
    # ModelPatcher -> BaseModel -> diffusion_model
    if hasattr(model, "model") and hasattr(model.model, "diffusion_model"):
        diffusion = model.model.diffusion_model
    elif hasattr(model, "diffusion_model"):
        diffusion = model.diffusion_model

    n_lin = n_conv = 0
    lin_set = conv_set = 0
    lin_cvt = conv_cvt = 0
    lin_q = conv_q = 0
    sample_missing = []

    for name, mod in diffusion.named_modules():
        cls = type(mod).__name__
        is_lin = "Linear" in cls
        is_conv = "Conv2d" in cls
        if not is_lin and not is_conv:
            continue
        has_set = callable(getattr(mod, "set_weight", None))
        has_cvt = callable(getattr(mod, "convert_weight", None))
        w = getattr(mod, "weight", None)
        is_q = False
        if QuantizedTensor is not type(None):
            is_q = isinstance(w, QuantizedTensor) or isinstance(
                getattr(w, "data", None), QuantizedTensor
            )
        layout = getattr(mod, "layout_type", None)
        if is_lin:
            n_lin += 1
            lin_set += int(has_set)
            lin_cvt += int(has_cvt)
            lin_q += int(is_q or layout is not None)
        else:
            n_conv += 1
            conv_set += int(has_set)
            conv_cvt += int(has_cvt)
            conv_q += int(is_q or layout is not None)
            if (not has_set or not has_cvt) and len(sample_missing) < 5:
                sample_missing.append(
                    f"{name} set={has_set} convert={has_cvt} layout={layout}"
                )

    _lora_line("[HSWQ INT8 LoRA] ===== load summary =====")
    _lora_line(
        f"[HSWQ INT8 LoRA] Linear: {n_lin}  set_weight={lin_set}  convert_weight={lin_cvt}  quantized/layout={lin_q}"
    )
    _lora_line(
        f"[HSWQ INT8 LoRA] Conv2d: {n_conv}  set_weight={conv_set}  convert_weight={conv_cvt}  quantized/layout={conv_q}"
    )
    if conv_set < n_conv or conv_cvt < n_conv:
        _lora_line(
            "[HSWQ INT8 LoRA] WARN: some Conv2d lack set/convert — LoRA on those layers will round into int8 and die"
        )
        for s in sample_missing:
            _lora_line(f"[HSWQ INT8 LoRA]   missing: {s}")
    else:
        _lora_line(
            "[HSWQ INT8 LoRA] OK: Conv2d has set_weight+convert_weight (dequant -> bake -> requant)"
        )
    _lora_line("[HSWQ INT8 LoRA] =========================")
    return {
        "linear": n_lin,
        "conv2d": n_conv,
        "linear_set_weight": lin_set,
        "conv_set_weight": conv_set,
    }


def decode_comfy_quant_conf(raw):
    """Decode a comfy_quant marker into a dict layer config."""
    import torch

    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if torch.is_tensor(raw):
        conf = json.loads(raw.numpy().tobytes())
    elif isinstance(raw, (bytes, bytearray, memoryview)):
        conf = json.loads(bytes(raw))
    elif isinstance(raw, str):
        conf = raw
    else:
        conf = raw

    while isinstance(conf, str):
        try:
            parsed = json.loads(conf)
        except (TypeError, json.JSONDecodeError):
            return {"format": conf}
        if parsed is conf:
            return {"format": conf}
        conf = parsed

    if isinstance(conf, dict):
        return conf
    raise TypeError(f"comfy_quant config must be a dict or format string, got {type(conf).__name__}")


def checkpoint_looks_like_comfy_quant_int8(state_dict_or_path) -> bool:
    """True if checkpoint has comfy_quant INT8 markers (native MixedPrecisionOps path).

    Accepts a loaded state_dict, or a filesystem path (probes via safetensors without full load).
    """
    import torch

    if isinstance(state_dict_or_path, (str, os.PathLike)):
        return _probe_path_comfy_quant_int8(str(state_dict_or_path))

    state_dict = state_dict_or_path
    has_marker = False
    has_int8 = False
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        if key.endswith(".comfy_quant"):
            has_marker = True
            conf = decode_comfy_quant_conf(value)
            if isinstance(conf, dict) and conf.get("format") == "int8_tensorwise":
                return True
        if key.endswith(".weight") and value.dtype == torch.int8:
            has_int8 = True
    return has_marker and has_int8


def _probe_path_comfy_quant_int8(path: str) -> bool:
    """Lightweight safetensors probe for int8_tensorwise."""
    import torch

    try:
        from safetensors import safe_open
    except ImportError:
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            comfy_keys = [k for k in keys if k.endswith(".comfy_quant")]
            for ck in comfy_keys[:16]:
                conf = decode_comfy_quant_conf(f.get_tensor(ck))
                if isinstance(conf, dict) and conf.get("format") == "int8_tensorwise":
                    return True
            if comfy_keys:
                for k in keys:
                    if not k.endswith(".weight"):
                        continue
                    if f.get_tensor(k).dtype == torch.int8:
                        return True
                    break
            meta = f.metadata() or {}
            if "_quantization_metadata" in meta:
                try:
                    qm = json.loads(meta["_quantization_metadata"])
                    layers = qm.get("layers", {}) if isinstance(qm, dict) else {}
                    for v in layers.values():
                        if isinstance(v, str) and v == "int8_tensorwise":
                            return True
                        if isinstance(v, dict) and v.get("format") == "int8_tensorwise":
                            return True
                except (TypeError, json.JSONDecodeError):
                    pass
    except Exception as e:
        logger.debug("[HSWQ INT8] probe failed for %s: %s", path, e)
        return False
    return False


def _normalize_comfy_quant_tensor(value):
    import torch

    conf = decode_comfy_quant_conf(value)
    if conf is None:
        return None
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def _patch_convert_old_quants() -> bool:
    try:
        import torch
        import comfy.utils as utils_module
    except ImportError:
        return False

    original = getattr(utils_module, "convert_old_quants", None)
    if original is None or getattr(original, "_hswq_int8_patched", False):
        return False

    def convert_old_quants_pre(state_dict, model_prefix="", metadata=None):
        if metadata is None:
            metadata = {}
        # Normalize string layer configs in metadata before upstream json.dumps(v).
        if isinstance(metadata, dict) and "_quantization_metadata" in metadata:
            try:
                quant_meta = json.loads(metadata["_quantization_metadata"])
            except (TypeError, json.JSONDecodeError):
                quant_meta = None
            if isinstance(quant_meta, dict) and isinstance(quant_meta.get("layers"), dict):
                layers = quant_meta["layers"]
                changed = False
                for k, v in list(layers.items()):
                    if isinstance(v, str):
                        layers[k] = {"format": v}
                        changed = True
                    elif not isinstance(v, dict):
                        raise TypeError(
                            f"quantization layer config for {k} must be dict or format string, got {type(v).__name__}"
                        )
                if changed:
                    metadata = dict(metadata)
                    metadata["_quantization_metadata"] = json.dumps(quant_meta)

        state_dict, metadata = original(state_dict, model_prefix=model_prefix, metadata=metadata)

        # Re-normalize any .comfy_quant tensors (file-embedded or metadata-written).
        for key in list(state_dict.keys()):
            if not key.endswith(".comfy_quant"):
                continue
            normalized = _normalize_comfy_quant_tensor(state_dict[key])
            if normalized is None:
                state_dict.pop(key, None)
            else:
                state_dict[key] = normalized
        return state_dict, metadata

    convert_old_quants_pre._hswq_int8_patched = True
    utils_module.convert_old_quants = convert_old_quants_pre
    return True


def _quant_config_has_int8_tensorwise(quant_config) -> bool:
    """True if MixedPrecisionOps quant_config targets int8_tensorwise layers."""
    if not isinstance(quant_config, dict) or not quant_config:
        return False
    for v in quant_config.values():
        if isinstance(v, dict) and v.get("format") == "int8_tensorwise":
            return True
        if v == "int8_tensorwise":
            return True
    return False


# INT8 Conv2d inject must NOT run for FP MixedPrecisionOps.
# detect_layer_quantization() only returns {"mixed_ops": True} for both INT8 and FP8,
# so we gate Conv2d injection on this load-scoped flag (set only in INT8 load helpers).
_int8_quant_conv_tls = threading.local()


@contextlib.contextmanager
def _int8_quant_conv_scope():
    prev = getattr(_int8_quant_conv_tls, "active", False)
    _int8_quant_conv_tls.active = True
    try:
        yield
    finally:
        _int8_quant_conv_tls.active = prev


def _should_inject_int8_conv(quant_config) -> bool:
    if getattr(_int8_quant_conv_tls, "active", False):
        return True
    return _quant_config_has_int8_tensorwise(quant_config)


def _model_has_int8_quantized_weights(model) -> bool:
    """True if model already carries INT8 / QuantizedTensor weights (post-load)."""
    try:
        import torch
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return False
    for _, module in model.named_modules():
        w = getattr(module, "weight", None)
        if w is None:
            continue
        if isinstance(w, QuantizedTensor):
            return True
        dtype = getattr(w, "dtype", None)
        if dtype is not None and dtype in (torch.int8, getattr(torch, "qint8", torch.int8)):
            return True
    return False


def _rotate_activation_nhwc(x_nhwc, h_matrix, group_size: int):
    """Pooled bf16/fp16 rotate of an NHWC tensor's channel axis (no permute-back).

    Shared core for _rotate_activation_nchw (fp16-conv fallback) and the
    int8-IMMA conv path. Output buffers pooled per (shape,dtype,device).
    """
    import torch

    B, Hh, Ww, C = x_nhwc.shape
    in_dtype = x_nhwc.dtype
    group_count = C // group_size
    h = h_matrix
    if h.dtype != in_dtype or h.device != x_nhwc.device:
        h = h.to(dtype=in_dtype, device=x_nhwc.device)
    key = (B, Hh, Ww, C, in_dtype, str(x_nhwc.device))
    out = _ROT_POOL_NCHW.get(key)
    if out is None:
        out = torch.empty(B, Hh, Ww, C, dtype=in_dtype, device=x_nhwc.device)
        _ROT_POOL_NCHW[key] = out
    xg = x_nhwc.reshape(B, Hh * Ww, group_count, group_size)
    torch.matmul(xg, h, out=out.view(B, Hh * Ww, group_count, group_size))
    return out


_ROT_POOL_NCHW: dict = {}


_INT8_CONV_STATS = {"imma_triton": 0, "imma_torch": 0, "fallback": 0}

_TRITON_TRIED = False
_TRITON_KERNELS = None
_AQ_POOL: dict = {}


def _get_triton_convrot_kernels():
    """Lazy-build the fused ConvRot Triton kernels (gather-rowwise-quant +
    dequant-bias-NCHW-store). Returns (quant_k, dequant_k, triton) or None.

    Triton is NOT used for the matmul: int8 tl.dot measured 1.3-4x SLOWER than
    cublasLt torch._int_mm on sm_120 (0.59-0.92ms vs 0.23ms). Triton replaces
    only the torch passes around the GEMM that dominate wall time:
      quant side: F.pad + as_strided + im2col contiguous + kitchen rowwise
        (two full fp16 [T,K] round trips, ~280MB/call at 1024^2 stage0)
      store side: float32 dequant chain (3 elementwise passes) + to(fp16) +
        permute(0,3,1,2).contiguous()
    Validation vs the exact torch path (unit test): scales bit-exact, int8
    codes within +-1 at 0.88% of positions (kitchen reciprocal-multiply
    rounding), epilogue within 1 ulp at 1/163840.
    """
    global _TRITON_TRIED, _TRITON_KERNELS
    if _TRITON_TRIED:
        return _TRITON_KERNELS
    _TRITON_TRIED = True
    try:
        import triton
        import triton.language as tl

        try:
            from triton.language.extra import libdevice as tld
        except Exception:
            from triton.language.extra.cuda import libdevice as tld

        @triton.jit
        def _convrot_quant_kernel(
            xr_ptr, aq_ptr, as_ptr,
            H, W, Cs, hs, ws,
            kh, kw, sh, sw, ph, pw, Kdim,
            stride_xb, stride_xh, stride_xw,
            stride_ab,
            BLOCK_K: tl.constexpr,
        ):
            t = tl.program_id(0)
            b = t // (hs * ws)
            rem = t % (hs * ws)
            oy = rem // ws
            ox = rem % ws
            hp0 = oy * sh - ph
            wp0 = ox * sw - pw
            base = xr_ptr + b * stride_xb
            amax = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for k0 in range(0, Kdim, BLOCK_K):
                kk = k0 + tl.arange(0, BLOCK_K)
                kmask = kk < Kdim
                r = kk // (kw * Cs)
                rem2 = kk % (kw * Cs)
                s = rem2 // Cs
                c = rem2 % Cs
                hp = hp0 + r
                wp = wp0 + s
                ok = kmask & (hp >= 0) & (hp < H) & (wp >= 0) & (wp < W)
                v = tl.load(base + hp * stride_xh + wp * stride_xw + c, mask=ok, other=0.0).to(tl.float32)
                amax = tl.maximum(amax, tl.abs(v))
            row_amax = tl.max(amax, axis=0)
            scale = tl.maximum(row_amax, 1e-6) / 127.0
            tl.store(as_ptr + t, scale)
            for k0 in range(0, Kdim, BLOCK_K):
                kk = k0 + tl.arange(0, BLOCK_K)
                kmask = kk < Kdim
                r = kk // (kw * Cs)
                rem2 = kk % (kw * Cs)
                s = rem2 // Cs
                c = rem2 % Cs
                hp = hp0 + r
                wp = wp0 + s
                ok = kmask & (hp >= 0) & (hp < H) & (wp >= 0) & (wp < W)
                v = tl.load(base + hp * stride_xh + wp * stride_xw + c, mask=ok, other=0.0).to(tl.float32)
                q = tld.rint(v / scale)
                q = tl.minimum(tl.maximum(q, -127.0), 127.0)
                tl.store(aq_ptr + t * stride_ab + kk, q.to(tl.int8), mask=kmask)

        @triton.jit
        def _convrot_dequant_kernel(
            acc_ptr, as_ptr, ws_ptr, bias_ptr, out_ptr,
            T, Oout, hs, ws_,
            stride_ab,
            stride_ob, stride_oo, stride_oh,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        ):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            mmask = offs_m < T
            nmask = offs_n < Oout
            a = tl.load(acc_ptr + offs_m[:, None] * stride_ab + offs_n[None, :],
                        mask=mmask[:, None] & nmask[None, :], other=0).to(tl.float32)
            asc = tl.load(as_ptr + offs_m, mask=mmask, other=0.0)
            wsc = tl.load(ws_ptr + offs_n, mask=nmask, other=0.0)
            bias = tl.load(bias_ptr + offs_n, mask=nmask, other=0.0)
            o = a * asc[:, None] * wsc[None, :] + bias[None, :]
            o = tl.minimum(tl.maximum(o, -65504.0), 65504.0)
            bidx = offs_m // (hs * ws_)
            rem = offs_m % (hs * ws_)
            oy = rem // ws_
            ox = rem % ws_
            dst = out_ptr + bidx[:, None] * stride_ob + oy[:, None] * stride_oh \
                + ox[:, None] + offs_n[None, :] * stride_oo
            tl.store(dst, o.to(tl.float16), mask=mmask[:, None] & nmask[None, :])

        _TRITON_KERNELS = (_convrot_quant_kernel, _convrot_dequant_kernel, triton)
    except Exception as e:
        _console(f"[HSWQ INT8] triton fused convrot unavailable ({type(e).__name__}); "
                 "torch-IMMA path only")
        _TRITON_KERNELS = None
    return _TRITON_KERNELS


def _int8_conv_rot_imma(module, x):
    """ConvRot INT8 Conv2d via cublasLt IMMA (torch._int_mm) + im2col.

    Replaces the per-call 'dequant weight -> fp16 cudnn conv' tax with a true
    W8A8 GEMM: rotate the activation (pooled bf16/fp16), im2col by as_strided
    view, rowwise-quantize the [T,K] matrix with kitchen's CUDA quantizer,
    one _int_mm against the offline-rotated INT8 weight, then dequantize the
    int32 accumulator with (act row scale x weight out-channel scale) and add
    the fp16 bias.

    Semantics mirror comfy_kitchen.int8_linear for Linear (rowwise dynamic
    activation quant + pre-rotated INT8 weight), closing the Linear/Conv split.
    Returns None when any gate fails so the caller falls back to the fp16 path
    (groups/dilation/padding_mode, dtype, alignment, T/K/N floors, LoRA
    weight_function residency, non-4D QT payloads).
    """
    import torch
    import torch.nn.functional as F

    if x.dtype not in (torch.float16, torch.bfloat16) or not x.is_cuda or x.ndim != 4:
        return None
    if getattr(module, "groups", 1) != 1 or any(d != 1 for d in module.dilation):
        return None
    if module.padding_mode != "zeros" or getattr(module, "weight_function", None):
        return None
    gs = int(getattr(module, "_hswq_convrot_groupsize", 256) or 256)
    B, C, H, W = x.shape
    kh, kw = module.kernel_size
    sh, sw = module.stride
    ph, pw = module.padding
    if C % gs != 0:
        return None
    k_dim = kh * kw * C
    weight = module.weight
    from comfy_kitchen.tensor import QuantizedTensor
    from comfy_kitchen.backends.cuda import quantize_int8_rowwise

    qdata = weight._qdata if isinstance(weight, QuantizedTensor) else (
        getattr(getattr(weight, "data", None), "_qdata", None)
    )
    if qdata is None or qdata.dtype != torch.int8 or qdata.ndim != 4:
        return None
    o, ic, khw, kww = qdata.shape
    if (ic, khw, kww) != (C, kh, kw) or k_dim % 8 != 0 or o % 8 != 0:
        return None
    hs = (H + 2 * ph - kh) // sh + 1
    ws_ = (W + 2 * pw - kw) // sw + 1
    t = B * hs * ws_
    if t < 16:
        return None

    h = getattr(module, "_hswq_convrot_H", None)
    if h is None or h.device != x.device or h.dtype != x.dtype:
        h = _build_hadamard(gs, device=x.device, dtype=x.dtype)
        module._hswq_convrot_H = h

    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    x_rot = _rotate_activation_nhwc(x_nhwc, h, gs)

    cache = getattr(module, "_hswq_imma_wcache", None)
    key = (qdata.data_ptr(), o, ic, kh, kw, str(qdata.device), qdata._version)
    if cache is not None and cache[0] == key:
        wmat, wscale_1d, bias_f = cache[1], cache[2], cache[3]
        if wscale_1d.device != x.device:
            # Staged-CPU scale (same mechanism as the build-time pin): pin once
            # and refresh the cache so every later call sees a CUDA scale.
            wscale_1d = wscale_1d.to(x.device).contiguous()
            bias_f = bias_f.to(x.device)
            module._hswq_imma_wcache = (key, wmat, wscale_1d, bias_f)
    else:
        params = weight._params if isinstance(weight, QuantizedTensor) else weight.data._params
        wscale = params.scale
        if wscale is None or int(wscale.numel()) != o:
            return None
        wmat = qdata.permute(2, 3, 1, 0).reshape(k_dim, o).contiguous()
        # Pin the small scale to the activation device. ComfyUI may stage the
        # QuantizedTensor params (scale) on CPU while the int8 payload (qdata)
        # is on GPU; a CPU scale makes BOTH the triton dequant kernel
        # ("Pointer argument cannot be accessed from Triton (cpu tensor?)" —
        # measured) and the torch-IMMA fallback (out.mul_) fail with a device
        # error. bias_f below is already pinned to wmat.device.
        wscale_1d = wscale.reshape(o).float().to(x.device).contiguous()
        bias_f = (
            module.bias.detach().float().to(wmat.device).contiguous()
            if module.bias is not None
            else torch.zeros(o, dtype=torch.float32, device=wmat.device)
        )
        module._hswq_imma_wcache = (key, wmat, wscale_1d, bias_f)

    tk = _get_triton_convrot_kernels()
    # Measured: triton launches INSIDE a CUDA-graph capture work under the
    # default caching allocator (17.4x it/s replay) but are rejected under the
    # cudaMallocAsync graph pool ('Pointer argument cannot be accessed from
    # Triton (cpu tensor?)', args all on cuda:0). The try/except below splits
    # the two failure regimes: a capture-phase failure only skips triton for
    # that capture (no process-wide disarm), an eager failure keeps the old
    # once-latched warn. The capture gate is decided at launch, not globally.
    if tk is not None and x.dtype == torch.float16:
        try:
            quant_k, dequant_k, triton = tk
            akey = (t, k_dim, str(x.device))
            aq = _AQ_POOL.get(akey)
            if aq is None:
                aq = torch.empty((t, k_dim), dtype=torch.int8, device=x.device)
                _AQ_POOL[akey] = aq
            ascale = torch.empty(t, dtype=torch.float32, device=x.device)
            quant_k[(t,)](
                x_rot, aq, ascale,
                H, W, C, hs, ws_, kh, kw, sh, sw, ph, pw, k_dim,
                x_rot.stride(0), x_rot.stride(1), x_rot.stride(2),
                aq.stride(0),
                BLOCK_K=1024, num_warps=4,
            )
            acc = torch._int_mm(aq, wmat)
            out = torch.empty((B, o, hs, ws_), dtype=torch.float16, device=x.device)
            grid = (triton.cdiv(t, 64), triton.cdiv(o, 64))
            dequant_k[grid](
                acc, ascale, wscale_1d, bias_f, out,
                t, o, hs, ws_,
                acc.stride(0),
                out.stride(0), out.stride(1), out.stride(2),
                BLOCK_M=64, BLOCK_N=64, num_warps=4,
            )
            _INT8_CONV_STATS["imma_triton"] += 1
            if _INT8_CONV_STATS["imma_triton"] == 1:
                _console("[HSWQ INT8] conv W8A8 IMMA: triton fused gather-quant + "
                         "dequant-NCHW store active")
            return out
        except Exception as e:
            # Capture-phase triton failures must NOT poison the process: eager
            # launches keep the fused path. Latch the once-warn only when the
            # failure happened outside an active capture.
            try:
                _capturing_now = torch.cuda.is_current_stream_capturing()
            except Exception:
                _capturing_now = False
            if _capturing_now:
                if _INT8_CONV_STATS.get("triton_capture_warn") is None:
                    _INT8_CONV_STATS["triton_capture_warn"] = 1
                    _console("[HSWQ INT8] triton fused conv unsupported during CUDA-graph "
                             "capture (cudaMallocAsync pool); torch-IMMA used for capture/"
                             "replay, fused path stays armed for eager launches")
            elif _INT8_CONV_STATS.get("triton_warn") is None:
                _INT8_CONV_STATS["triton_warn"] = 1
                try:
                    _wd = lambda t: (f"{type(t).__name__}:{t.device}:ptr{t.data_ptr()}"
                                     if torch.is_tensor(t) else str(t))
                    _console(
                        "[HSWQ INT8][diag] triton-fail args: "
                        f"x={_wd(x)} x_rot={_wd(x_rot)} aq={_wd(aq)} ascale={_wd(ascale)} "
                        f"qdata={_wd(qdata)} wmat={_wd(wmat)} wscale_1d={_wd(wscale_1d)} "
                        f"bias_f={_wd(bias_f)} acc={_wd(acc)} out={_wd(out)} "
                        f"capture={_capturing_now} "
                        f"err={type(e).__name__}: {str(e)[:160]}"
                    )
                except Exception:
                    pass
                _console(f"[HSWQ INT8] triton fused conv failed once ({type(e).__name__}: "
                         f"{str(e)[:120]}); staying on torch-IMMA")

    if ph or pw:
        x_rot = F.pad(x_rot, (0, 0, pw, pw, ph, ph))
    Hp, Wp = H + 2 * ph, W + 2 * pw
    row_stride = Wp * C
    img_stride = Hp * Wp * C
    im2col = x_rot.as_strided(
        (B, hs, ws_, kh, kw, C),
        (img_stride, sh * row_stride, sw * C, row_stride, C, 1),
    ).reshape(t, k_dim).contiguous()
    aq, ascale = quantize_int8_rowwise(im2col)

    acc = torch._int_mm(aq, wmat)  # int32 [T, O]
    out = acc.float()
    out = out.mul_(ascale.float())
    out = out.mul_(wscale_1d.view(1, -1))
    if module.bias is not None:
        out = out.add_(module.bias.detach().float().to(out.device).view(1, -1))
    if x.dtype == torch.float16:
        out = out.clamp(-65504.0, 65504.0)
    out = out.to(x.dtype)
    out = out.view(B, hs, ws_, o).permute(0, 3, 1, 2)
    return out.contiguous()


# --- ConvRot-only helpers (plain INT8 path never calls these) ---
_HADAMARD_CACHE: dict = {}
_OPS_PATCH_VER = 4  # Conv2d ConvRot: cache H on module GPU (no per-fwd CPU→CUDA)


def _build_hadamard(size: int, device="cpu", dtype=None):
    """Build Hadamard in float32 then cast (bf16/fp16 kron destroys online fidelity)."""
    import math

    import torch

    if dtype is None:
        dtype = torch.float32
    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
    # Always kron in float32 (matches offline W@H^T / nvfp4_hadamard.build_hadamard).
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
    if dtype != torch.float32:
        h_matrix = h_matrix.to(dtype=dtype)
    _HADAMARD_CACHE[cache_key] = h_matrix
    return h_matrix


def _rotate_last_dim(x, h_matrix, group_size: int):
    """Online act rotation → float32 (no bf16 mid-cast before Conv/Linear)."""
    import torch

    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} not divisible by group_size {group_size}")
    group_count = features // group_size
    if x.dtype == torch.float32 and h_matrix.device == x.device and h_matrix.dtype == torch.float32:
        x_grouped = x.reshape(-1, group_count, group_size)
        return torch.matmul(x_grouped, h_matrix).reshape(orig_shape)
    x_f = x.reshape(-1, group_count, group_size).float()
    h = h_matrix.to(dtype=torch.float32, device=x.device)
    return torch.matmul(x_f, h).reshape(orig_shape)


def _rotate_activation_nchw(x, h_matrix, group_size: int, out_dtype=None):
    """Rotate an NCHW activation for ConvRot Conv2d (pooled bf16, SD1.5 hot path).

    SD1.5 measured (25 seeds x 25 steps, RTX 5060 Ti): the previous fp32-accumulate
    rotate made the INT8 branch -24.8% wall vs FP16; the fp32 round-trip
    (to-fp32 + fp32 matmul + to-fp16 per conv call, 98 convs) was the dominant
    cost. Kitchen's own SDXL measurement shows pooled bf16 rotate is 5x faster
    (pooled fp32 257.3ms vs pooled bf16 52.5ms for the SDXL layer mix) and at
    least as accurate (cos 0.95867 vs 0.95472) because Hadamard entries are
    +-2^-4 — exact in bf16/fp16 — so no fp32 forcing is needed online.

    Output returns in the graph dtype before permute-back so the conv and the
    following attention never run fp32 (conv fp16 0.745ms vs fp32 1.568ms).
    Output buffers are pooled per (shape, dtype, device) to avoid a fresh
    allocation per conv call.
    """
    if x.ndim != 4:
        raise ValueError(f"NCHW activation must be 4D, got ndim={x.ndim}")
    in_dtype = x.dtype if out_dtype is None else out_dtype
    B, C, Hh, Ww = x.shape
    group_count = C // group_size
    h = h_matrix
    if h.dtype != in_dtype or h.device != x.device:
        h = h.to(dtype=in_dtype, device=x.device)
    import torch

    key = (B, Hh, Ww, C, in_dtype, str(x.device))
    out = _ROT_POOL_NCHW.get(key)
    if out is None:
        out = torch.empty(B, Hh, Ww, C, dtype=in_dtype, device=x.device)
        _ROT_POOL_NCHW[key] = out
    xg = x.permute(0, 2, 3, 1).contiguous().view(B, Hh * Ww, group_count, group_size)
    torch.matmul(xg, h, out=out.view(B, Hh * Ww, group_count, group_size))
    return out.permute(0, 3, 1, 2).contiguous()


_ROT_POOL_NCHW: dict = {}


def _unrotate_last_dim(x, h_matrix, group_size: int):
    """Inverse of ``_rotate_last_dim`` (H is orthogonal → same matmul with H^T).

    Matmul in float32 for the same reason as online rotate.
    """
    import torch

    orig_shape = x.shape
    features = orig_shape[-1]
    group_count = features // group_size
    h_t = h_matrix.T
    if x.dtype == torch.float32 and h_t.device == x.device and h_t.dtype == torch.float32:
        x_grouped = x.reshape(-1, group_count, group_size)
        return torch.matmul(x_grouped, h_t).reshape(orig_shape)
    x_f = x.reshape(-1, group_count, group_size).float()
    h = h_t.to(dtype=torch.float32, device=x.device)
    out = torch.matmul(x_f, h).reshape(orig_shape)
    return out.to(dtype=x.dtype)


def _unrotate_weight_conv2d(weight, h_matrix, group_size: int):
    if weight.ndim != 4:
        raise ValueError(f"Conv2d weight must be 4D, got ndim={weight.ndim}")
    out_c, in_c, k_h, k_w = weight.shape
    flat = weight.permute(0, 2, 3, 1).contiguous().view(-1, in_c)
    flat_un = _unrotate_last_dim(flat, h_matrix, group_size)
    return flat_un.view(out_c, k_h, k_w, in_c).permute(0, 3, 1, 2).contiguous()


def _rotate_weight_conv2d(weight, h_matrix, group_size: int):
    import torch

    if weight.ndim != 4:
        raise ValueError(f"Conv2d weight must be 4D, got ndim={weight.ndim}")
    out_c, in_c, k_h, k_w = weight.shape
    flat = weight.permute(0, 2, 3, 1).contiguous().view(-1, in_c)
    orig_shape = flat.shape
    group_count = in_c // group_size
    weight_grouped = flat.view(orig_shape[0], group_count, group_size)
    flat_rot = torch.matmul(
        weight_grouped, h_matrix.T.to(dtype=weight.dtype, device=weight.device)
    ).reshape(orig_shape)
    return flat_rot.view(out_c, k_h, k_w, in_c).permute(0, 3, 1, 2).contiguous()


def _qt_payload(weight, QuantizedTensor):
    if weight is None:
        return None
    if isinstance(weight, QuantizedTensor):
        return weight
    data = getattr(weight, "data", None)
    if isinstance(data, QuantizedTensor):
        return data
    return None


def _arm_hswq_conv2d_convrot(module, QuantizedTensor):
    """ConvRot branch only: keep online rotate on module; clear kitchen Params.convrot.

    Kitchen dequantize_int8_convrot_* is 2D-only. Stamping Params.convrot=True on
    4D then .dequantize() raises NoCapableBackendError (exactly 2D).
    Plain INT8 Conv2d (no convrot stamp) never enters this path.
    """
    import dataclasses

    import torch

    qt = _qt_payload(getattr(module, "weight", None), QuantizedTensor)
    if qt is None:
        return
    params = getattr(qt, "_params", None)
    qdata = getattr(qt, "_qdata", None)
    if params is None or qdata is None:
        return
    if getattr(qdata, "ndim", None) != 4:
        return
    if not bool(getattr(params, "convrot", False)):
        return

    gs = int(getattr(params, "convrot_groupsize", 256) or 256)
    module._hswq_convrot = True
    module._hswq_convrot_groupsize = gs
    new_params = dataclasses.replace(params, convrot=False)
    # Prefer in-place params swap. Reconstructing QT needs layout *string*
    # (_layout_cls), not a layout object — wrong arg → empty AssertionError.
    try:
        object.__setattr__(qt, "_params", new_params)
        return
    except Exception:
        pass
    try:
        qt._params = new_params
        return
    except Exception:
        pass
    layout_cls = getattr(qt, "_layout_cls", None)
    if not isinstance(layout_cls, str):
        layout_cls = getattr(module, "layout_type", None)
    if not isinstance(layout_cls, str):
        return
    new_qt = type(qt)(qdata, layout_cls, new_params)
    module.weight = torch.nn.Parameter(new_qt, requires_grad=False)


def _make_quantized_conv2d(ops_module, MixedPrecisionOps, disabled):
    """Build MixedPrecisionOps.Conv2d class using current comfy.ops helpers."""
    import torch

    CastWeightBiasOp = ops_module.CastWeightBiasOp
    QuantizedTensor = ops_module.QuantizedTensor
    cast_bias_weight = ops_module.cast_bias_weight
    uncast_bias_weight = ops_module.uncast_bias_weight
    run_every_op = ops_module.run_every_op
    _load_quantized_module = ops_module._load_quantized_module
    _quantized_weight_state_dict = ops_module._quantized_weight_state_dict
    _quantized_apply = ops_module._quantized_apply

    class Conv2d(torch.nn.Module, CastWeightBiasOp):
        _disabled_formats = disabled
        _hswq_quant_conv2d = True

        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
            device=None,
            dtype=None,
        ):
            super().__init__()
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            if isinstance(stride, int):
                stride = (stride, stride)
            if isinstance(padding, int):
                padding = (padding, padding)
            if isinstance(dilation, int):
                dilation = (dilation, dilation)

            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.groups = groups
            self.padding_mode = padding_mode
            self.factory_kwargs = {"device": device, "dtype": MixedPrecisionOps._compute_dtype}
            self._orig_shape = (out_channels, in_channels // groups, kernel_size[0], kernel_size[1])

            if bias:
                self.bias = torch.nn.Parameter(
                    torch.empty(out_channels, **self.factory_kwargs), requires_grad=False
                )
            else:
                self.register_parameter("bias", None)

            self.weight = None
            self.quant_format = None
            self.layout_type = None
            self._full_precision_mm = MixedPrecisionOps._full_precision_mm
            self._full_precision_mm_config = False
            # ConvRot branch flags (plain INT8 stays False)
            self._hswq_convrot = False
            self._hswq_convrot_groupsize = 256

        def reset_parameters(self):
            return None

        def _load_from_state_dict(self, *args):
            _load_quantized_module(self, super()._load_from_state_dict, *args, load_extra_params=False)
            # ConvRot-only: arm online rotate + clear 4D Params.convrot
            _arm_hswq_conv2d_convrot(self, QuantizedTensor)

        def state_dict(self, *args, destination=None, prefix="", **kwargs):
            sd = destination if destination is not None else {}
            sd = _quantized_weight_state_dict(self, sd, prefix)
            # ConvRot-only: re-stamp on export (Params.convrot cleared for safe dequant)
            if getattr(self, "_hswq_convrot", False):
                cq_key = f"{prefix}comfy_quant"
                conf = {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": int(
                        getattr(self, "_hswq_convrot_groupsize", 256) or 256
                    ),
                }
                sd[cq_key] = torch.tensor(
                    list(json.dumps(conf, separators=(",", ":")).encode("utf-8")),
                    dtype=torch.uint8,
                )
            return sd

        def _conv_forward(self, input, weight, bias):
            if self.padding_mode != "zeros":
                return torch.nn.functional.conv2d(
                    torch.nn.functional.pad(
                        input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                    ),
                    weight,
                    bias,
                    self.stride,
                    (0, 0),
                    self.dilation,
                    self.groups,
                )
            return torch.nn.functional.conv2d(
                input, weight, bias, self.stride, self.padding, self.dilation, self.groups
            )

        def forward_comfy_cast_weights(self, input):
            # Mirror MixedPrecision Linear: when weight is QuantizedTensor and
            # Dynamic VRAM uses weight_lowvram_function, want_requant=True so
            # post_cast dequant → LoRA → requant (want_requant=False left QT
            # in the resident path after the first step and killed LoRA).
            # ConvRot branch: W8A8 IMMA path first (rotate + im2col view +
            # rowwise act quant + torch._int_mm + fused dequant epilogue).
            # Any gate miss (groups/dilation/padding_mode, LoRA weight_function,
            # dtype, alignment, T/K floors) falls back to the pooled rotate +
            # dequantized fp16 cudnn conv below.
            if getattr(self, "_hswq_convrot", False):
                out = _int8_conv_rot_imma(self, input)
                if out is not None:
                    return out
                _INT8_CONV_STATS["fallback"] += 1
                gs = int(getattr(self, "_hswq_convrot_groupsize", 256) or 256)
                h = getattr(self, "_hswq_convrot_H", None)
                if (
                    h is None
                    or h.device != input.device
                    or h.dtype != input.dtype
                ):
                    h = _build_hadamard(
                        gs, device=input.device, dtype=input.dtype
                    )
                    self._hswq_convrot_H = h
                # The ConvRot weight is stored as W @ H^T, so the activation MUST
                # be rotated too (x @ H) for x_rot @ W_rot^T == x @ W^T.
                # Pooled rotate in the graph dtype; the conv must not run in
                # fp32 (measured 0.745 -> 1.568 ms) or leak fp32 into attention.
                input = _rotate_activation_nchw(input, h, gs, out_dtype=input.dtype)
            want_requant = isinstance(getattr(self, "weight", None), QuantizedTensor)
            weight, bias, offload_stream = cast_bias_weight(
                self,
                input,
                offloadable=True,
                compute_dtype=getattr(input, "dtype", None),
                want_requant=want_requant,
            )
            x = self._conv_forward(input, weight, bias)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return x

        def forward(self, input, *args, **kwargs):
            run_every_op()
            return self.forward_comfy_cast_weights(input)

        def convert_weight(self, weight, inplace=False, **kwargs):
            # Same contract as MixedPrecisionOps.Linear: LoRA / ModelPatcher
            # dequant → calculate_weight → set_weight (see ComfyUI-INT8-Fast bake path).
            global _lora_convert_logs
            out = weight.dequantize() if isinstance(weight, QuantizedTensor) else weight
            # ConvRot branch only: unrotate to original float basis for LoRA
            if getattr(self, "_hswq_convrot", False) and out is not None and getattr(out, "ndim", 0) == 4:
                gs = int(getattr(self, "_hswq_convrot_groupsize", 256) or 256)
                h = _build_hadamard(gs, device="cpu", dtype=torch.float32)
                out = _unrotate_weight_conv2d(out, h, gs)
            if _lora_convert_logs < _LORA_CONVERT_LOG_MAX:
                _lora_convert_logs += 1
                wdtype = getattr(weight, "dtype", None)
                odtype = getattr(out, "dtype", None)
                _console(
                    f"[HSWQ INT8 LoRA] Conv2d.convert_weight #{_lora_convert_logs}: "
                    f"in={type(weight).__name__}/{wdtype} -> out={type(out).__name__}/{odtype} "
                    f"layout={getattr(self, 'layout_type', None)} "
                    f"convrot={getattr(self, '_hswq_convrot', False)}"
                )
            return out

        def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
            # Mirror MixedPrecisionOps.Linear.set_weight so Conv2d LoRA bake
            # does not fall through to stochastic_rounding(..., int8), which
            # destroys float LoRA deltas (INT8-Fast: normal LoRA loader works).
            global _lora_set_logs
            layout = getattr(self, "layout_type", None)
            path = "requant" if layout is not None else "cast_only"
            # ConvRot branch only: convert_weight returned unrotated; re-rotate before requant
            if getattr(self, "_hswq_convrot", False) and getattr(weight, "ndim", 0) == 4:
                gs = int(getattr(self, "_hswq_convrot_groupsize", 256) or 256)
                h = _build_hadamard(gs, device="cpu", dtype=torch.float32)
                weight = _rotate_weight_conv2d(weight, h, gs)
            if _lora_set_logs < _LORA_SET_LOG_MAX:
                _lora_set_logs += 1
                _console(
                    f"[HSWQ INT8 LoRA] Conv2d.set_weight #{_lora_set_logs}: "
                    f"path={path} float_in={getattr(weight, 'dtype', None)} "
                    f"shape={tuple(weight.shape) if hasattr(weight, 'shape') else '?'} "
                    f"seed={seed} layout={layout} "
                    f"convrot={getattr(self, '_hswq_convrot', False)}"
                )
            if layout is not None:
                weight = self.weight.requantize_from_float(
                    weight,
                    scale="recalculate",
                    stochastic_rounding=seed,
                    inplace_ops=True,
                ).to(self.weight.dtype)
            else:
                weight = weight.to(self.weight.dtype)
            if return_weight:
                return weight

            assert inplace_update is False
            self.weight = torch.nn.Parameter(weight, requires_grad=False)

        def _apply(self, fn, recurse=True):
            return _quantized_apply(self, fn, recurse)

        @property
        def _reversed_padding_repeated_twice(self):
            return tuple(x for x in reversed(self.padding) for _ in range(2))

    return Conv2d


def _patch_ops_decode_and_conv() -> bool:
    try:
        import comfy.ops as ops_module
    except ImportError:
        return False

    ops_module._decode_comfy_quant_conf = decode_comfy_quant_conf

    original_load = getattr(ops_module, "_load_quantized_module", None)
    if original_load is None:
        return False

    if not getattr(original_load, "_hswq_int8_decode_patched", False):

        def _load_quantized_module_patched(
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
            key = f"{prefix}comfy_quant"
            if key in state_dict:
                normalized = _normalize_comfy_quant_tensor(state_dict[key])
                if normalized is None:
                    state_dict.pop(key, None)
                else:
                    state_dict[key] = normalized
            return original_load(
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

        _load_quantized_module_patched._hswq_int8_decode_patched = True
        ops_module._load_quantized_module = _load_quantized_module_patched

    # Also normalize Embedding's direct json.loads path by wrapping Embedding._load_from_state_dict
    # is covered if convert_old_quants + file markers are normalized; keep load wrapper as safety.

    original_mp = getattr(ops_module, "mixed_precision_ops", None)
    if original_mp is None or not callable(original_mp):
        return False
    # Versioned so ConvRot branch upgrade re-applies; plain path unchanged.
    if getattr(original_mp, "_hswq_int8_ops_ver", 0) >= _OPS_PATCH_VER:
        return True

    def mixed_precision_ops_force_conv(
        quant_config=None, compute_dtype=None, full_precision_mm=False, disabled=None
    ):
        if quant_config is None:
            quant_config = {}
        if compute_dtype is None:
            import torch

            compute_dtype = torch.bfloat16
        if disabled is None:
            disabled = []
        result = original_mp(
            quant_config=quant_config,
            compute_dtype=compute_dtype,
            full_precision_mm=full_precision_mm,
            disabled=disabled,
        )
        # Inject Quantized Conv2d only during INT8 load scope (or explicit
        # int8_tensorwise quant_config). FP MixedPrecisionOps must keep upstream
        # Conv2d — detect_layer_quantization() is {"mixed_ops": True} for both.
        if _should_inject_int8_conv(quant_config):
            result.Conv2d = _make_quantized_conv2d(ops_module, result, disabled)
        return result

    mixed_precision_ops_force_conv._hswq_int8_conv_patched = True
    mixed_precision_ops_force_conv._hswq_int8_ops_ver = _OPS_PATCH_VER
    ops_module.mixed_precision_ops = mixed_precision_ops_force_conv
    return True


def _patch_lowvram_patch_float_intermediate() -> bool:
    """Fix LowVramPatch intermediate_dtype for INT8 / QuantizedTensor weights.

    Upstream LowVramPatch passes intermediate_dtype=weight.dtype. When the
    weight is still int8/Char (or a QuantizedTensor), LoRA matmul casts to
    int8 and either errors or silently produces a no-op delta — same bug as
    BobJohnson24/ComfyUI-INT8-Fast#76.
    """
    try:
        import torch
        import comfy.lora
        import comfy.model_patcher as mp
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return False

    LowVramPatch = getattr(mp, "LowVramPatch", None)
    if LowVramPatch is None:
        return False
    original = getattr(LowVramPatch, "__call__", None)
    if original is None or getattr(original, "_hswq_int8_lora_dtype", False):
        return getattr(original, "_hswq_int8_lora_dtype", False)

    def __call__(self, weight):
        patches = (
            self.prepared_patches
            if self.prepared_patches is not None
            else self.patches[self.key]
        )
        w = weight
        if isinstance(w, QuantizedTensor):
            w = w.dequantize()
        dtype = getattr(w, "dtype", None)
        if dtype is not None and hasattr(dtype, "is_floating_point") and not dtype.is_floating_point:
            idtype = torch.float32
        elif dtype is not None and hasattr(dtype, "is_floating_point") and dtype.is_floating_point:
            idtype = dtype
        else:
            idtype = torch.float32
        return comfy.lora.calculate_weight(patches, w, self.key, intermediate_dtype=idtype)

    __call__._hswq_int8_lora_dtype = True
    LowVramPatch.__call__ = __call__
    return True


def _get_baked_key_set(model) -> set:
    s = getattr(model, "_hswq_int8_baked_keys", None)
    if s is None:
        s = set()
        model._hswq_int8_baked_keys = s
    return s


def _maybe_invalidate_baked_keys(patcher) -> None:
    """If patches_uuid changed (new LoRA), allow those keys to be baked again."""
    model = patcher.model
    baked_uuid = getattr(model, "_hswq_int8_baked_uuid", None)
    cur = getattr(patcher, "patches_uuid", None)
    if baked_uuid is None or cur is None:
        return
    if baked_uuid != cur and patcher.patches:
        _get_baked_key_set(model).clear()
        model._hswq_int8_baked_uuid = None


def _strip_lowvram_for_baked_keys(patcher) -> int:
    """Dynamic.load re-attaches LowVramPatch; clear it for already-baked keys.

    Shared modules keep their VBAR ``_v`` across loads. Re-attaching LoRA on
    top of baked INT8 weights would double-apply; clearing lowvram avoids that.
    """
    _maybe_invalidate_baked_keys(patcher)
    baked = getattr(patcher.model, "_hswq_int8_baked_keys", None)
    if not baked:
        return 0
    cleared = 0
    for name, module in patcher.model.named_modules():
        for param_key in ("weight", "bias"):
            key = f"{name}.{param_key}"
            if key not in baked:
                continue
            attr = param_key + "_lowvram_function"
            if getattr(module, attr, None) is not None:
                setattr(module, attr, None)
                cleared += 1
            # Drop from this patcher's dict so later loads do not re-attach
            if key in patcher.patches:
                try:
                    del patcher.patches[key]
                except KeyError:
                    pass
    return cleared


def _bake_int8_patches_on_dynamic_patcher(patcher, device_to) -> int:
    """Bake LoRA into INT8 modules after ModelPatcherDynamic.load.

    Dynamic VRAM attaches LowVramPatch on weight_lowvram_function and asserts
    force_patch_weights=False. For comfy_quant INT8 that path often leaves
    LoRA attached in the patcher dict but visually inert (keys count OK,
    bake logs absent). We bake via convert_weight/set_weight (requant).

    Critical VBAR rule (2nd-gen FaceDetailer OOM):
      ModelVBAR.alloc is a bump allocator (offset only grows). Deleting
      module._v after bake makes the next load call alloc() again → VBAR OOM.
      Keep ``_v``. Clear LowVramPatch, bake, then pop patches + drop the
      pre-bake backup entry so restore_loaded_backups does not undo bake.
    """
    if not getattr(patcher, "patches", None):
        return 0
    try:
        import comfy.model_patcher as mp
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return 0

    _maybe_invalidate_baked_keys(patcher)
    already = _get_baked_key_set(patcher.model)
    baked = 0
    for name, module in patcher.model.named_modules():
        keys_to_bake = []
        for param_key in ("weight", "bias"):
            key = f"{name}.{param_key}"
            if key not in patcher.patches:
                continue
            if key in already:
                # Already baked under this patches_uuid; clear re-attached LowVramPatch
                attr = param_key + "_lowvram_function"
                if getattr(module, attr, None) is not None:
                    setattr(module, attr, None)
                try:
                    del patcher.patches[key]
                except KeyError:
                    pass
                continue
            weight, set_func, convert_func = mp.get_key_weight(patcher.model, key)
            if weight is None:
                continue
            is_qt = isinstance(weight, QuantizedTensor)
            # Bake when requant hooks exist (Linear/Conv INT8), or QT without
            # set_func would still be broken — skip those and log below.
            if set_func is None:
                if is_qt:
                    _console(
                        f"[HSWQ INT8 LoRA] WARN cannot bake {key}: "
                        "QuantizedTensor but no set_weight (int8_round risk)"
                    )
                continue
            keys_to_bake.append((param_key, key))

        if not keys_to_bake:
            continue

        # Clear LowVramPatch so bake uses Parameter + set_weight, not lazy patch.
        # Do NOT unpin/delete module._v — that causes 2nd-load VBAR OOM.
        for param_key, _key in keys_to_bake:
            if hasattr(module, param_key + "_lowvram_function"):
                setattr(module, param_key + "_lowvram_function", None)

        for _param_key, key in keys_to_bake:
            patcher.patch_weight_to_device(key, device_to=device_to)
            # Drop pre-bake backup so the next Dynamic.load restore keeps baked weights
            if key in patcher.backup:
                try:
                    del patcher.backup[key]
                except KeyError:
                    pass
            try:
                del patcher.patches[key]
            except KeyError:
                pass
            already.add(key)
            baked += 1

    if baked > 0:
        patcher.model._hswq_int8_baked_uuid = getattr(patcher, "patches_uuid", None)

    return baked


def _patch_model_patcher_dynamic_int8_lora_bake() -> bool:
    """After ModelPatcherDynamic.load, bake INT8 LoRA via set_weight."""
    try:
        import comfy.model_patcher as mp
    except ImportError:
        return False

    Dynamic = getattr(mp, "ModelPatcherDynamic", None)
    if Dynamic is None:
        return False
    original = getattr(Dynamic, "load", None)
    if original is None:
        return False
    _DYN_VER = 4
    if getattr(original, "_hswq_int8_lora_bake_ver", 0) >= _DYN_VER:
        return True
    true_orig = getattr(original, "_hswq_orig_dynamic_load", original)

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False, dirty=False):
        result = true_orig(
            self,
            device_to=device_to,
            lowvram_model_memory=lowvram_model_memory,
            force_patch_weights=force_patch_weights,
            full_load=full_load,
            dirty=dirty,
        )
        # INT8 LoRA bake only — skip for Nunchaku / FP8 / plain bf16 Dynamic loads.
        if not _model_has_int8_quantized_weights(self.model) and not getattr(
            self.model, "_hswq_int8_baked_keys", None
        ):
            return result
        # Load re-attaches LowVramPatch for any keys still in patches / clones
        _strip_lowvram_for_baked_keys(self)
        if self.patches:
            n = _bake_int8_patches_on_dynamic_patcher(self, device_to=device_to)
            if n > 0 or _lora_attach_history or (_lora_attach_last.get("mapped_keys") or 0) > 0:
                dump_int8_lora_bake_stats(force=True)
        elif _lora_attach_history or (_lora_attach_last.get("mapped_keys") or 0) > 0:
            # Patches already consumed by a prior bake; still emit Status once if needed
            dump_int8_lora_bake_stats(force=False)
        return result

    load._hswq_int8_lora_bake = True
    load._hswq_int8_lora_bake_ver = _DYN_VER
    load._hswq_orig_dynamic_load = true_orig
    Dynamic.load = load
    return True


def _patch_model_patcher_lora_logs() -> bool:
    """Log whether LoRA bake uses set_weight (requant) or int8_round fallback."""
    try:
        import comfy.model_patcher as mp
    except ImportError:
        return False

    original = getattr(mp.ModelPatcher, "patch_weight_to_device", None)
    if original is None or getattr(original, "_hswq_int8_lora_log", False):
        return getattr(original, "_hswq_int8_lora_log", False)

    def patch_weight_to_device_logged(self, key, device_to=None, inplace_update=False, return_weight=False, force_cast=False):
        global _lora_patcher_logs
        weight, set_func, convert_func = mp.get_key_weight(self.model, key)
        if key in self.patches:
            _lora_patcher_stats["calls"] += 1
            if set_func is not None:
                _lora_patcher_stats["with_set_func"] += 1
            else:
                _lora_patcher_stats["without_set_func"] += 1
            if convert_func is not None:
                _lora_patcher_stats["with_convert_func"] += 1

            path = "requant" if set_func is not None else "int8_round"
            _lora_bake_by_key[key] = path
            if _lora_patcher_logs < _LORA_PATCHER_LOG_MAX:
                _lora_patcher_logs += 1
                wdtype = getattr(weight, "dtype", None)
                warn = ""
                if set_func is None and wdtype is not None and str(wdtype) in ("torch.int8", "int8"):
                    warn = "  << BROKEN for INT8 (LoRA delta will be destroyed)"
                owners = [
                    e["lora_name"]
                    for e in _lora_attach_history
                    if key in (e.get("applied_unet_keys") or [])
                ]
                owner_s = ",".join(owners[:3]) if owners else "-"
                if len(owners) > 3:
                    owner_s += f"+{len(owners) - 3}"
                _console(
                    f"[HSWQ INT8 LoRA] bake #{_lora_patcher_logs}: key={key} "
                    f"path={path} lora={owner_s} weight_dtype={wdtype} "
                    f"convert={'yes' if convert_func else 'no'} "
                    f"set={'yes' if set_func else 'no'}{warn}"
                )
            # After stacked UNet keys are baked, dump per-LoRA summary once
            target = sum(int(e.get("applied_unet") or 0) for e in _lora_attach_history)
            if target <= 0:
                target = int(_lora_attach_last.get("applied_unet") or 0)
            # Unique baked keys may be less than sum (shared keys across LoRAs)
            unique_target = len(
                {
                    k
                    for e in _lora_attach_history
                    for k in (e.get("applied_unet_keys") or [])
                }
            ) or target
            if (
                unique_target > 0
                and _lora_patcher_stats["calls"] >= unique_target
                and not getattr(dump_int8_lora_bake_stats, "_dumped_this_load", False)
            ):
                # Do NOT set the flag before dump (that made dump a no-op).
                dump_int8_lora_bake_stats(force=False)


        return original(
            self,
            key,
            device_to=device_to,
            inplace_update=inplace_update,
            return_weight=return_weight,
            force_cast=force_cast,
        )

    patch_weight_to_device_logged._hswq_int8_lora_log = True
    mp.ModelPatcher.patch_weight_to_device = patch_weight_to_device_logged
    return True


def _per_lora_bake_verdict(entry: dict) -> tuple[str, int, int, int]:
    """Return (verdict, requant, int8_round, not_baked) for one LoRA attach entry."""
    unet_keys = entry.get("applied_unet_keys") or []
    clip_n = int(entry.get("applied_clip") or 0)
    unet_n = int(entry.get("applied_unet") or 0)
    if unet_n == 0 and clip_n > 0:
        return ("N/A_CLIP_only", 0, 0, 0)
    if unet_n == 0:
        return ("SKIP_no_keys", 0, 0, 0)
    requant = 0
    int8_round = 0
    not_baked = 0
    for k in unet_keys:
        path = _lora_bake_by_key.get(k)
        if path == "requant":
            requant += 1
        elif path == "int8_round":
            int8_round += 1
        else:
            not_baked += 1
    if int8_round > 0:
        return ("BROKEN_int8_round", requant, int8_round, not_baked)
    if requant == 0 and not_baked == unet_n:
        return ("WARN_not_baked_yet", requant, int8_round, not_baked)
    if requant > 0 and int8_round == 0:
        return ("OK_requant", requant, int8_round, not_baked)
    return ("PARTIAL", requant, int8_round, not_baked)


def dump_int8_lora_bake_stats(force: bool = False) -> None:
    """Full Status dump: lora_name / applied_keys / skipped_keys (+ bake if any)."""
    if not force and getattr(dump_int8_lora_bake_stats, "_dumped_this_load", False):
        return
    dump_int8_lora_bake_stats._dumped_this_load = True

    history = list(_lora_attach_history) if _lora_attach_history else []
    if not history and (_lora_attach_last.get("mapped_keys") or 0) > 0:
        history = [dict(_lora_attach_last)]

    n = len(history)
    _lora_line(f"[HSWQ LoRA Status] ===== bake summary ({n} slot(s)) =====")
    if not history:
        _lora_line(
            "[HSWQ LoRA Status] Slot -: | lora_name='(none)' | applied_keys=0 | skipped_keys=0 | → SKIPPED ✗"
        )
    ok_n = 0
    for i, a in enumerate(history, 1):
        line = _format_lora_slot_line(i, a, include_bake=True)
        _lora_line(line)
        verdict, _rq, _ir, _nb = _per_lora_bake_verdict(a)
        if verdict in ("OK_requant", "N/A_CLIP_only") or _slot_applied_count(a) > 0:
            if verdict != "BROKEN_int8_round":
                ok_n += 1
    _lora_line(
        f"[HSWQ LoRA Status] Summary: {ok_n}/{n} LoRA(s) with applied keys"
    )

    s = _lora_patcher_stats
    if s["calls"] == 0:
        _lora_line("[HSWQ LoRA Bake] not yet (model not on GPU)")
        return
    _lora_line(
        f"[HSWQ LoRA Bake] total={s['calls']} requant={s['with_set_func']} "
        f"int8_round={s['without_set_func']} shape_skip={len(_lora_shape_skips)}"
    )
    if s["without_set_func"] > 0:
        _lora_line(
            "[HSWQ LoRA Bake] WARNING: int8_round used — those layers are broken"
        )
    else:
        _lora_line("[HSWQ LoRA Bake] path OK (all requant)")
    if _lora_shape_skips:
        for name, key, reason in _lora_shape_skips[:_LORA_SKIP_PRINT_MAX]:
            _lora_line(
                f"[HSWQ LoRA Bake] shape_skip | '{name}' | {key} | {reason}"
            )


def _patch_lora_loader_name_context() -> bool:
    """Capture name from nodes.LoraLoader when any node calls it."""
    try:
        import nodes as nodes_mod
    except ImportError:
        return False

    LoraLoader = getattr(nodes_mod, "LoraLoader", None)
    if LoraLoader is None:
        return False
    original = getattr(LoraLoader, "load_lora", None)
    if original is None:
        return False
    _NAME_VER = 6
    if getattr(original, "_hswq_lora_name_ctx_ver", 0) >= _NAME_VER:
        return True
    true_orig = getattr(original, "_hswq_orig_load_lora", original)

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        global _current_lora_name, _current_lora_strength_model, _current_lora_strength_clip
        prev = (
            _current_lora_name,
            _current_lora_strength_model,
            _current_lora_strength_clip,
        )
        _set_current_lora_name(lora_name, strength_model, strength_clip)
        try:
            return true_orig(self, model, clip, lora_name, strength_model, strength_clip)
        finally:
            (
                _current_lora_name,
                _current_lora_strength_model,
                _current_lora_strength_clip,
            ) = prev

    load_lora._hswq_lora_name_ctx = True
    load_lora._hswq_lora_name_ctx_ver = _NAME_VER
    load_lora._hswq_orig_load_lora = true_orig
    LoraLoader.load_lora = load_lora
    return True


def _patch_loras_folder_path_name() -> bool:
    """Any loader that resolves folder_paths 'loras' → capture filename."""
    try:
        import folder_paths
    except ImportError:
        return False

    _PATH_VER = 3
    ok = False

    for fname in ("get_full_path", "get_full_path_or_raise"):
        original = getattr(folder_paths, fname, None)
        if original is None:
            continue
        if getattr(original, "_hswq_lora_path_name_ver", 0) >= _PATH_VER:
            ok = True
            continue
        true_orig = getattr(original, "_hswq_orig_get_full_path", original)

        def _make(orig):
            def wrapped(folder_name, filename):
                if folder_name == "loras":
                    _set_current_lora_name(filename)
                return orig(folder_name, filename)

            wrapped._hswq_lora_path_name_ver = _PATH_VER
            wrapped._hswq_orig_get_full_path = orig
            return wrapped

        setattr(folder_paths, fname, _make(true_orig))
        ok = True
    return ok


def _patch_load_torch_file_lora_name() -> bool:
    """Any loader that load_torch_file(lora_path) → capture basename."""
    try:
        import comfy.utils as utils_mod
    except ImportError:
        return False
    original = getattr(utils_mod, "load_torch_file", None)
    if original is None:
        return False
    _TORCH_VER = 1
    if getattr(original, "_hswq_lora_torch_name_ver", 0) >= _TORCH_VER:
        return True
    true_orig = getattr(original, "_hswq_orig_load_torch_file", original)

    def load_torch_file(ckpt, *args, **kwargs):
        if isinstance(ckpt, (str, os.PathLike)):
            p = str(ckpt)
            if _path_is_under_loras_dir(p):
                _set_current_lora_name(p)
        return true_orig(ckpt, *args, **kwargs)

    load_torch_file._hswq_lora_torch_name_ver = _TORCH_VER
    load_torch_file._hswq_orig_load_torch_file = true_orig
    utils_mod.load_torch_file = load_torch_file
    return True


def _patch_load_lora_key_counts() -> bool:
    """Wrap load_lora + load_lora_for_models for applied/skipped key counts."""
    try:
        import comfy.lora as lora_mod
        import comfy.sd as sd_mod
        import comfy.weight_adapter as weight_adapter
    except ImportError:
        return False

    orig_load_lora = getattr(lora_mod, "load_lora", None)
    orig_for_models = getattr(sd_mod, "load_lora_for_models", None)
    if orig_load_lora is None or orig_for_models is None:
        return False

    _KEY_VER = 6
    if getattr(orig_for_models, "_hswq_lora_key_count_ver", 0) >= _KEY_VER:
        _patch_lora_loader_name_context()
        _patch_loras_folder_path_name()
        _patch_load_torch_file_lora_name()
        return True

    if getattr(orig_for_models, "_hswq_lora_key_count", False):
        orig_for_models = getattr(
            orig_for_models, "_hswq_orig_for_models", orig_for_models
        )
    if getattr(orig_load_lora, "_hswq_lora_key_count", False):
        orig_load_lora = getattr(orig_load_lora, "_hswq_orig_load_lora", orig_load_lora)

    _ctx = {"patch_dict": {}, "not_mapped": [], "file_keys": 0}

    def load_lora_counted(lora, to_load, log_missing=True):
        patch_dict = {}
        loaded_keys = set()
        for x in to_load:
            alpha_name = "{}.alpha".format(x)
            alpha = None
            if alpha_name in lora.keys():
                alpha = lora[alpha_name].item()
                loaded_keys.add(alpha_name)

            dora_scale_name = "{}.dora_scale".format(x)
            dora_scale = None
            if dora_scale_name in lora.keys():
                dora_scale = lora[dora_scale_name]
                loaded_keys.add(dora_scale_name)

            for adapter_cls in weight_adapter.adapters:
                adapter = adapter_cls.load(x, lora, alpha, dora_scale, loaded_keys)
                if adapter is not None:
                    patch_dict[to_load[x]] = adapter
                    loaded_keys.update(adapter.loaded_keys)
                    continue

            w_norm_name = "{}.w_norm".format(x)
            b_norm_name = "{}.b_norm".format(x)
            w_norm = lora.get(w_norm_name, None)
            b_norm = lora.get(b_norm_name, None)

            if w_norm is not None:
                loaded_keys.add(w_norm_name)
                patch_dict[to_load[x]] = ("diff", (w_norm,))
                if b_norm is not None:
                    loaded_keys.add(b_norm_name)
                    patch_dict["{}.bias".format(to_load[x][: -len(".weight")])] = (
                        "diff",
                        (b_norm,),
                    )

            diff_name = "{}.diff".format(x)
            diff_weight = lora.get(diff_name, None)
            if diff_weight is not None:
                patch_dict[to_load[x]] = ("diff", (diff_weight,))
                loaded_keys.add(diff_name)

            diff_bias_name = "{}.diff_b".format(x)
            diff_bias = lora.get(diff_bias_name, None)
            if diff_bias is not None:
                patch_dict["{}.bias".format(to_load[x][: -len(".weight")])] = (
                    "diff",
                    (diff_bias,),
                )
                loaded_keys.add(diff_bias_name)

            set_weight_name = "{}.set_weight".format(x)
            set_weight = lora.get(set_weight_name, None)
            if set_weight is not None:
                patch_dict[to_load[x]] = ("set", (set_weight,))
                loaded_keys.add(set_weight_name)

        not_mapped = [x for x in lora.keys() if x not in loaded_keys]
        _ctx["patch_dict"] = patch_dict
        _ctx["not_mapped"] = not_mapped
        _ctx["file_keys"] = len(lora) if hasattr(lora, "keys") else 0

        if log_missing:
            for x in not_mapped:
                logging.warning("lora key not loaded: {}".format(x))

        return patch_dict

    def load_lora_for_models_counted(
        model, clip, lora, strength_model, strength_clip, lora_metadata=None
    ):
        new_model, new_clip = orig_for_models(
            model, clip, lora, strength_model, strength_clip, lora_metadata
        )
        loaded = _ctx.get("patch_dict") or {}
        not_mapped = list(_ctx.get("not_mapped") or [])
        file_key_count = int(_ctx.get("file_keys") or 0)
        lora_name = _resolve_lora_name(loaded)

        unet_keys = set(new_model.patches.keys()) if new_model is not None else set()
        if new_clip is not None and hasattr(new_clip, "patcher"):
            clip_keys = set(new_clip.patcher.patches.keys())
        else:
            clip_keys = set()

        applied_unet_keys = []
        applied_clip_keys = []
        mapped_but_not = []
        add_patches_miss = []
        for x in loaded:
            key = x if isinstance(x, str) else x[0]
            in_u = key in unet_keys
            in_c = key in clip_keys
            if in_u:
                applied_unet_keys.append(key)
            if in_c:
                applied_clip_keys.append(key)
            if not in_u and not in_c:
                mapped_but_not.append(x)
                add_patches_miss.append(x)

        applied_unet = len(applied_unet_keys)
        applied_clip = len(applied_clip_keys)

        entry = {
            "lora_name": lora_name,
            "strength_model": strength_model,
            "strength_clip": strength_clip,
            "lora_file_keys": file_key_count,
            "mapped_keys": len(loaded),
            "applied_unet": applied_unet,
            "applied_clip": applied_clip,
            "applied_unet_keys": list(applied_unet_keys),
            "applied_clip_keys": list(applied_clip_keys),
            "not_mapped": sorted(str(x) for x in not_mapped),
            "mapped_but_not_attached": list(mapped_but_not),
            "add_patches_skipped_unet": list(add_patches_miss),
        }
        _lora_attach_last.update(entry)
        _lora_attach_history.append(dict(entry))
        _log_lora_slot_attach(entry)
        return (new_model, new_clip)

    load_lora_counted._hswq_lora_key_count = True
    load_lora_counted._hswq_orig_load_lora = orig_load_lora
    load_lora_for_models_counted._hswq_lora_key_count = True
    load_lora_for_models_counted._hswq_lora_key_count_ver = _KEY_VER
    load_lora_for_models_counted._hswq_orig_for_models = orig_for_models
    lora_mod.load_lora = load_lora_counted
    sd_mod.load_lora_for_models = load_lora_for_models_counted
    _patch_lora_loader_name_context()
    _patch_loras_folder_path_name()
    _patch_load_torch_file_lora_name()
    return True


def _patch_kitchen_convrot_float32_act_rotate() -> bool:
    """Force kitchen INT8 ConvRot online act rotation to float32 matmul.

    Offline W@H^T uses float32 H. bf16 ``x @ H`` alone adds ~1e-2 MSE/layer and
    compounds across UNet Linears into latent SSIM collapse. Also disable CUDA
    fused ConvRot kernels so rotation goes through the patched Python path.
    """
    import importlib

    import torch

    try:
        iu = importlib.import_module("comfy_kitchen.tensor.int8_utils")
    except Exception as e:
        logger.warning("[HSWQ INT8] kitchen int8_utils not found: %s", e)
        return False

    # Measured (25 steps x 6 seeds, one process, fused kernel OFF in both arms):
    #   fp32 rotate (previous) : INT8 2.43 it/s, ratio vs FP16 1.069x, cos 0.95472
    #   pooled bf16 rotate     : INT8 2.90 it/s, ratio vs FP16 0.894x, cos 0.95867
    # Isolated rotate cost over the real SDXL INT8 layer mix (M=4096):
    #   pooled fp32 257.3 ms | pooled bf16 52.5 ms | bmm 540.3 | allocating 166.5
    # Hadamard entries are +-2^-4 (exact in fp16/bf16); the INT8 activation codes
    # differ from the fp32-rotate result by 0.00011% on average (measured).
    _ROT_POOL_HSWQ: dict = {}

    def _rotate_activation_hswq(x, h, group_size: int):
        orig_shape = x.shape
        features = orig_shape[-1]
        if features % group_size != 0:
            raise ValueError(f"features {features} not divisible by group_size {group_size}")
        n_groups = features // group_size
        x_grouped = x.reshape(-1, n_groups, group_size)
        if h.dtype != x_grouped.dtype or h.device != x_grouped.device:
            h = h.to(dtype=x_grouped.dtype, device=x_grouped.device)
        key = (tuple(x_grouped.shape), x_grouped.dtype, str(x_grouped.device))
        out = _ROT_POOL_HSWQ.get(key)
        if out is None:
            out = torch.empty_like(x_grouped)
            _ROT_POOL_HSWQ[key] = out
        torch.matmul(x_grouped, h, out=out)
        return out.reshape(orig_shape)

    iu._rotate_activation = _rotate_activation_hswq

    # Hadamard stays at its natural construction: entries are +-1/sqrt(n) and for
    # n=256 that is +-2^-4, which is exactly representable in fp16 and bf16, so no
    # float32 forcing is required for the online rotate.

    rebound = ["comfy_kitchen.tensor.int8_utils"]
    for modname in (
        "comfy_kitchen.backends.eager.quantization",
        "comfy_kitchen.backends.triton.quantization",
        "comfy_kitchen.backends.cuda",
        "comfy_kitchen.backends.eager.convrot_w4a4",
    ):
        try:
            m = importlib.import_module(modname)
        except Exception:
            continue
        if hasattr(m, "_rotate_activation"):
            m._rotate_activation = _rotate_activation_hswq
            rebound.append(modname)

    # CUDA fused ConvRot kernels: KEPT ON for gs256 Linears (SD1.5 branch).
    # Controlled A/B (FP16 and INT8 resident, interleaved 3-step reps at
    # 1024x1024, medians): fused OFF ratio INT8/FP16 = 1.211, fused ON = 1.126.
    # The kitchen fused quantizer (rotate+rowwise quant in one kernel, fp32
    # accumulation) also removes the fp16 activation round-trip the pooled
    # Python rotate pays per gs256 Linear. gs64 Linears and the injected
    # Conv2d stay on the pooled bf16 rotate (kitchen fused kernel is
    # group_size-256-only; verified: 'convrot64 fused kernel only supports
    # group_size 256', hard RuntimeError).
    # The SDXL/Krea2 vendored copies keep fused OFF (their own measured
    # settings); this divergence is deliberate and documented.
    _console(
        "[HSWQ INT8] kitchen ConvRot act-rotate → pooled bf16 fallback "
        f"(rebound={','.join(rebound)}; CUDA fused gs256 kept ON for SD1.5)"
    )
    return True


def _patch_sd15_cudagraph_executor() -> bool:
    try:
        import comfy.sd as _sd
    except ImportError:
        return False

    def _maybe_arm(model):
        if getattr(_int8_quant_conv_tls, "active", False) and model is not None:
            try:
                _arm_sd15_cudagraph(model)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[HSWQ INT8][SD1.5 graph] arm failed: %s", exc)
        return model

    # Path A: load_diffusion_model (UNet-file loaders) returns a ModelPatcher.
    orig_ldm = getattr(_sd, "load_diffusion_model", None)
    if orig_ldm is not None and not getattr(orig_ldm, "_hswq_sd15_graph_wrapped", False):
        def load_diffusion_model_sd15(unet_path, model_options={}, disable_dynamic=False):
            return _maybe_arm(
                orig_ldm(unet_path, model_options=model_options, disable_dynamic=disable_dynamic)
            )

        load_diffusion_model_sd15._hswq_sd15_graph_wrapped = True
        _sd.load_diffusion_model = load_diffusion_model_sd15

    # Path B: load_checkpoint_guess_config (full-checkpoint loaders) returns
    # (model, clip, vae, ...); the bench and Load Checkpoint both use it.
    orig_lcgc = getattr(_sd, "load_checkpoint_guess_config", None)
    if orig_lcgc is not None and not getattr(orig_lcgc, "_hswq_sd15_graph_wrapped", False):
        def load_checkpoint_guess_config_sd15(*args, **kwargs):
            out = orig_lcgc(*args, **kwargs)
            try:
                out = list(out)
                out[0] = _maybe_arm(out[0])
                out = tuple(out)
            except Exception:
                pass
            return out

        load_checkpoint_guess_config_sd15._hswq_sd15_graph_wrapped = True
        _sd.load_checkpoint_guess_config = load_checkpoint_guess_config_sd15

    return True


def _sd15_is_sd15_unet(model) -> bool:
    """SD1.5-only gate: model_config class SD15 + UNetModel diffusion_model.

    Blocks SDXL / Z-Image / Krea2 / Qwen (their model_config classes differ),
    so the CUDA-graph executor never arms outside SD1.5.
    """
    try:
        cfg = getattr(getattr(model, "model", None), "model_config", None)
        if type(cfg).__name__ != "SD15":
            return False
        dm = getattr(getattr(model, "model", None), "diffusion_model", None)
        if type(dm).__name__ != "UNetModel":
            return False
        return int(getattr(dm, "in_channels", 0)) == 4
    except Exception:
        return False


class _Sd15CudaGraphExecutor:
    """Per-(x,t,context) CUDA Graph replay for the SD1.5 INT8 UNet.

    The 512x512 INT8 path is CPU-launch-bound (comfy_kitchen::int8_linear +
    QuantizedTensor dispatch cost dominates; GPU work is halved vs FP16 but
    cannot hide the Python/op launch overhead at small token counts). Capturing
    the whole denoiser forward into a CUDA Graph removes the per-op CPU cost and
    makes INT8 faster than FP16 at 512 (measured 0.807x vs 1.231x eager).

    Correctness contract: capture buffers are copied-in before each replay, so
    the graph output is identical to eager (measured max|graph-eager| = 0.0).
    Capture is keyed on the input signature; a signature change re-captures.

    Capture uses capture_error_mode="relaxed". The default "global" mode aborts
    on harmless ops that are illegal only during capture (e.g. diagnostic/
    synchronising calls); relaxed tolerates them and the graph captures cleanly
    at every tested resolution (latent 64 / 80 / 96 / 128). Verified 512-1024.

    Scope: capture is attempted only for latent sides up to _MAX_CAPTURE_LATENT_SIDE
    (covers up to 1024x1024, the practical SD1.5 max). Anything larger runs eager.
    """

    _MAX_CAPTURE_LATENT_SIDE = 128  # covers up to 1024x1024; larger runs eager
    def __init__(self, base_model):
        self._base = base_model
        self._orig = base_model.apply_model
        self._graphs = {}
        self._captures = 0
        self._skip = set()
        base_model.apply_model = self._call
        try:
            base_model._hswq_sd15_graph_executor = self
        except Exception:
            pass

    def disarm(self):
        if getattr(self._base, "apply_model", None) == self._call:
            self._base.apply_model = self._orig
        self._graphs.clear()

    def _key(self, x, t, c):
        import torch

        tens = tuple(
            sorted(
                (k, tuple(v.shape), str(v.dtype))
                for k, v in c.items()
                if torch.is_tensor(v)
            )
        )
        return (tuple(x.shape), str(x.dtype), str(x.device), tens)

    def _call(self, x, t, **c):
        import torch

        # Large-latent-only guard: skip capture beyond the practical SD1.5 range.
        try:
            if max(int(x.shape[-1]), int(x.shape[-2])) > self._MAX_CAPTURE_LATENT_SIDE:
                return self._orig(x, t, **c)
        except Exception:
            return self._orig(x, t, **c)
        key = self._key(x, t, c)
        if key in self._skip:
            return self._orig(x, t, **c)
        ent = self._graphs.get(key)
        if ent is None:
            ent = self._capture(x, t, c)
            if ent is None:
                self._skip.add(key)
                return self._orig(x, t, **c)
            self._graphs[key] = ent
        sx, st, sc, graph, sy = ent
        sx.copy_(x)
        st.copy_(t)
        for k, buf in sc.items():
            src = c[k]
            if torch.is_tensor(src):
                buf.copy_(src)
        graph.replay()
        return sy.clone()

    def _capture(self, x, t, c):
        import torch

        dev = x.device
        sx = x.clone()
        st = t.clone()
        # Tensor inputs are moved to the graph device BEFORE capture: an
        # unpinned CPU->CUDA copy is illegal while capturing.
        sc = {
            k: (v.to(dev) if torch.is_tensor(v) else v).clone()
            for k, v in c.items()
            if torch.is_tensor(v)
        }
        c_in = {k: (sc[k] if torch.is_tensor(v) else v) for k, v in c.items()}
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._orig(sx, st, **c_in)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode="relaxed"):
                sy = self._orig(sx, st, **c_in)
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[HSWQ INT8][SD1.5 graph] capture failed, eager fallback: %s", exc
            )
            # A failed capture can leave capture state pending; make sure the
            # context is quiescent so eager fallback stays usable.
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            return None
        self._captures += 1
        ctx_shape = (
            tuple(c["c_crossattn"].shape)
            if torch.is_tensor(c.get("c_crossattn"))
            else "-"
        )
        _console(
            f"[HSWQ INT8][SD1.5 graph] captured #{self._captures} "
            f"x={tuple(x.shape)} ctx={ctx_shape}"
        )
        return (sx, st, sc, graph, sy)


def _arm_sd15_cudagraph(model) -> bool:
    """Arm the CUDA-graph executor on an SD1.5 INT8 model (no-op otherwise)."""
    import torch  # noqa: F401

    if not _sd15_is_sd15_unet(model):
        return False
    base = getattr(model, "model", None)
    if base is None or getattr(base, "apply_model", None) is None:
        return False
    if isinstance(getattr(base, "apply_model", None), _Sd15CudaGraphExecutor):
        return True
    _Sd15CudaGraphExecutor(base)
    _console("[HSWQ INT8][SD1.5 graph] CUDA-graph executor armed (512 CPU-bound path)")
    return True


def _patch_sd15_clip_quant_metadata() -> bool:
    """SD1.5-only: make ComfyUI consume ConvRot INT8 quantization metadata for the
    checkpoint-embedded CLIP (cond_stage_model.*).

    Why: SDClipModel builds its Linear layers with comfy.ops.manual_cast unless
    model_options["quantization_metadata"] is set (comfy/sd1_clip.py). The SD1.5
    text-encoder load path in comfy.sd never sets it, so the CLIP's int8 weights
    and .comfy_quant / .weight_scale companions load as raw fp16 (absmax 127) and
    the text conditioning collapses (cos ~ 0.10) or NaNs.

    Fix: wrap comfy.sd.load_state_dict_guess_config and, when the state dict
    carries SD1.5 CLIP comfy_quant layers, pass the CLIP-restricted quantization
    metadata through te_model_options so the text encoder is built with
    mixed_precision_ops (INT8 + ConvRot). Key space is remapped to the internal
    CLIP module names (cond_stage_model.* -> clip_l.*), matching
    SD15.process_clip_state_dict.
    """
    try:
        import comfy.sd as _sd
    except ImportError:
        return False
    original = getattr(_sd, "load_state_dict_guess_config", None)
    if original is None or getattr(original, "_hswq_sd15_clip_meta", False):
        return getattr(original, "_hswq_sd15_clip_meta", False)

    def _clip_quant_layers_from_sd(state_dict, metadata):
        layers = {}
        # 1. File-embedded .comfy_quant tensors (authoritative for our packs).
        for k in list(state_dict.keys()):
            if not (k.startswith("cond_stage_model.") and k.endswith(".comfy_quant")):
                continue
            base = k[: -len(".comfy_quant")]
            conf = decode_comfy_quant_conf(state_dict[k])
            if conf is None:
                continue
            # cond_stage_model.transformer... -> clip_l.transformer...
            rest = base[len("cond_stage_model."):]
            layers["clip_l." + rest] = conf
        # 2. Metadata fallback (some packs omit the per-layer tensor).
        if not layers and isinstance(metadata, dict) and "_quantization_metadata" in metadata:
            try:
                qm = json.loads(metadata["_quantization_metadata"])
                for k, v in qm.get("layers", {}).items():
                    if not k.startswith("cond_stage_model."):
                        continue
                    rest = k[len("cond_stage_model."):]
                    layers["clip_l." + rest] = v
            except (TypeError, json.JSONDecodeError):
                pass
        return layers

    def _complete_sd15_clip_optional_params(out):
        """text_projection / logit_scale are not in the SD1.5 checkpoint; the FP16
        CLIP loads text_projection as an unused all-zeros Linear. Under INT8 the
        same Linear is a quantized ops.Linear whose weight stays None (the pack key
        cannot be placed at clip_l.transformer.text_projection because
        SD15.process_clip_state_dict force-nests cond_stage_model.transformer.*
        into .text_model.), so the INT8 forward crashes on the None weight. Complete
        it with a plain fp16 zeros weight exactly like the FP16 path.
        """
        try:
            clip = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
            if clip is None:
                return
            csm = getattr(clip, "cond_stage_model", None)
            inner = getattr(csm, "clip_l", None)
            if inner is None:
                return
            import torch as _t
            tr = getattr(inner, "transformer", None)
            tp = getattr(tr, "text_projection", None)
            if tp is not None and getattr(tp, "weight", None) is None:
                hidden = 768
                try:
                    emb = tr.text_model.embeddings.token_embedding.weight
                    hidden = int(emb.shape[1])
                    dtype = emb.dtype
                except Exception:
                    dtype = _t.float16
                tp.weight = _t.nn.Parameter(
                    _t.zeros((hidden, hidden), dtype=dtype), requires_grad=False
                )
            if not hasattr(inner, "logit_scale") or inner.logit_scale is None:
                inner.logit_scale = _t.nn.Parameter(_t.tensor(4.6055))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[HSWQ INT8][SD1.5 CLIP] optional-param completion skipped: %s", exc)

    def load_state_dict_guess_config_sd15_clip(sd, *args, **kwargs):
        # Only SD1.5 packs: presence of cond_stage_model.* comfy_quant Linear layers
        # AND a CLIP-L-only text encoder (SDXL/SD3/Flux carry clip_g/t5 and never
        # reach this shape). Gated by model detection to keep other archs untouched.
        try:
            has_clip_quant = any(
                k.startswith("cond_stage_model.") and k.endswith(".comfy_quant")
                for k in sd.keys()
            )
        except Exception:
            has_clip_quant = False
        if not has_clip_quant:
            return original(sd, *args, **kwargs)

        clip_layers = _clip_quant_layers_from_sd(sd, kwargs.get("metadata"))
        if not clip_layers:
            return original(sd, *args, **kwargs)

        te_model_options = dict(kwargs.get("te_model_options") or {})
        te_model_options["quantization_metadata"] = {
            "format_version": "1.0",
            "layers": clip_layers,
        }
        kwargs["te_model_options"] = te_model_options
        _console(
            f"[HSWQ INT8][SD1.5 CLIP] injecting ConvRot INT8 metadata for "
            f"{len(clip_layers)} text-encoder layers"
        )
        out = original(sd, *args, **kwargs)
        _complete_sd15_clip_optional_params(out)
        return out

    load_state_dict_guess_config_sd15_clip._hswq_sd15_clip_meta = True
    _sd.load_state_dict_guess_config = load_state_dict_guess_config_sd15_clip
    return True


def apply_comfy_quant_int8_patches() -> bool:
    """Install INT8 comfy_quant patches once. Returns True if applied (or already applied)."""
    global _PATCHES_APPLIED
    ok_keys = _patch_load_lora_key_counts()
    ok_name = _patch_lora_loader_name_context()
    ok_path = _patch_loras_folder_path_name()
    ok_torch = _patch_load_torch_file_lora_name()
    ok_lowvram = _patch_lowvram_patch_float_intermediate()
    ok_dyn_bake = _patch_model_patcher_dynamic_int8_lora_bake()
    ok_sd15_graph = _patch_sd15_cudagraph_executor()
    ok_sd15_clip = _patch_sd15_clip_quant_metadata()
    # Kitchen float32 rotate must run even if ops patches were applied earlier
    # in-process (bench / loader re-entry).
    ok_kitchen_f32 = False
    if not getattr(apply_comfy_quant_int8_patches, "_kitchen_f32_done", False):
        ok_kitchen_f32 = _patch_kitchen_convrot_float32_act_rotate()
        apply_comfy_quant_int8_patches._kitchen_f32_done = bool(ok_kitchen_f32)
    if _PATCHES_APPLIED:
        return True
    ok_utils = _patch_convert_old_quants()
    ok_ops = _patch_ops_decode_and_conv()
    ok_lora_log = _patch_model_patcher_lora_logs()
    if ok_ops:
        _PATCHES_APPLIED = True
        _console(
            "[HSWQ INT8] comfy_quant patches applied "
            f"(Conv2d quant load + decode + ConvRot-safe 4D"
            f"{' + convert_old_quants' if ok_utils else ''}"
            f"{' + LoRA bake logs' if ok_lora_log else ''}"
            f"{' + LoRA key counts' if ok_keys else ''}"
            f"{' + LoRA name' if ok_name or ok_path or ok_torch else ''}"
            f"{' + LowVramPatch float dtype' if ok_lowvram else ''}"
            f"{' + Dynamic INT8 LoRA bake' if ok_dyn_bake else ''}"
            f"{' + SD1.5 CUDA-graph executor' if ok_sd15_graph else ''}"
            f"{' + SD1.5 CLIP INT8 metadata' if ok_sd15_clip else ''}"
            f"{' + kitchen float32 ConvRot rotate' if ok_kitchen_f32 else ''})"
        )
        return True
    logger.warning(
        "[HSWQ INT8] Failed to apply comfy_quant patches (ops=%s utils=%s)",
        ok_ops,
        ok_utils,
    )
    return False


def load_unet_hswq_weight_dtype(unet_name, weight_dtype):
    import logging
    import torch
    import folder_paths
    import comfy.sd

    # INT8 Conv2d + comfy_quant decode patches only when loading INT8.
    # Do not apply for FP8 / default weight_dtype (those stay on original path).
    unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
    is_int8 = weight_dtype == "int8_tensorwise" or checkpoint_looks_like_comfy_quant_int8(unet_path)

    if is_int8:
        apply_comfy_quant_int8_patches()
        model_options = {}
        reset_int8_lora_log_counters()
        logging.info("[HSWQ INT8] Loading UNet via MixedPrecisionOps (int8_tensorwise / comfy_quant)")
        print(f"[HSWQ INT8] Loading UNet: {unet_name}", flush=True)
        with _int8_quant_conv_scope():
            model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        summarize_int8_lora_capability(model)
    else:
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)

    return (model,)


def load_checkpoint_sdxl_hswq_weight_dtype(ckpt_name, weight_dtype, device=None):
    import sys
    import torch
    import folder_paths
    import comfy.sd

    pkg = sys.modules[__name__.rsplit(".", 2)[0]]
    get_current_device = pkg.get_current_device
    set_current_device = pkg.set_current_device
    sdxl_logger = pkg.sdxl_logger

    original_device = get_current_device()
    if device is not None:
        set_current_device(device)
    try:
        # INT8 Conv2d + comfy_quant decode only when checkpoint is INT8.
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        # Auto-detect native comfy_quant INT8; do not force float8 dtype over int8 weights.
        is_int8 = weight_dtype == "int8_tensorwise" or checkpoint_looks_like_comfy_quant_int8(ckpt_path)

        model_options = {}
        if is_int8:
            apply_comfy_quant_int8_patches()
            reset_int8_lora_log_counters()
            sdxl_logger.info(
                "[SDXL INT8] Loading checkpoint via MixedPrecisionOps "
                "(int8_tensorwise / comfy_quant): %s",
                ckpt_name,
            )
            with _int8_quant_conv_scope():
                out = comfy.sd.load_checkpoint_guess_config(
                    ckpt_path,
                    output_vae=False,
                    output_clip=True,
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    model_options=model_options,
                )
            model, clip, _v = out[:3]
            summarize_int8_lora_capability(model)
            return (model, clip)

        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=False,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            model_options=model_options,
        )
        model, clip, _v = out[:3]
        return (model, clip)
    finally:
        set_current_device(original_device)


def install_int8_option_dispatch(node_class_mappings) -> bool:
    if not isinstance(node_class_mappings, dict):
        return False

    # Do NOT apply INT8 patches at node registration / import.
    # Patches install only inside load_unet_hswq_weight_dtype /
    # load_checkpoint_sdxl_hswq_weight_dtype when INT8 is actually loaded.

    _FP8_WEIGHT_DTYPES = frozenset({"fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"})

    unet_cls = node_class_mappings.get("HSWQFP8E4M3UNetLoader")
    if unet_cls is not None:
        _orig_load_unet = unet_cls.load_unet

        def load_unet(self, unet_name, weight_dtype):
            # Explicit FP8 choices stay on the original FP loader body — never INT8 helper.
            if weight_dtype in _FP8_WEIGHT_DTYPES:
                return _orig_load_unet(self, unet_name, weight_dtype)
            if weight_dtype == "int8_tensorwise":
                return load_unet_hswq_weight_dtype(unet_name, weight_dtype)
            # default: auto-detect INT8 checkpoints only; otherwise original FP path.
            import folder_paths

            unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
            if checkpoint_looks_like_comfy_quant_int8(unet_path):
                return load_unet_hswq_weight_dtype(unet_name, weight_dtype)
            return _orig_load_unet(self, unet_name, weight_dtype)

        unet_cls.load_unet = load_unet

    sdxl_cls = node_class_mappings.get("NunchakuUssoewwinCheckpointLoaderSDXL")
    if sdxl_cls is not None:
        _orig_load_checkpoint = sdxl_cls.load_checkpoint

        def load_checkpoint(self, ckpt_name, weight_dtype, device=None):
            if weight_dtype in _FP8_WEIGHT_DTYPES:
                return _orig_load_checkpoint(self, ckpt_name, weight_dtype, device)
            if weight_dtype == "int8_tensorwise":
                return load_checkpoint_sdxl_hswq_weight_dtype(ckpt_name, weight_dtype, device)
            import folder_paths

            ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
            if checkpoint_looks_like_comfy_quant_int8(ckpt_path):
                return load_checkpoint_sdxl_hswq_weight_dtype(ckpt_name, weight_dtype, device)
            return _orig_load_checkpoint(self, ckpt_name, weight_dtype, device)

        sdxl_cls.load_checkpoint = load_checkpoint

    return True
