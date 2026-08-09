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

Applied from ComfyUI-HSWQ-Loader-and-Tools so ComfyUI core updates do not wipe it.
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
    print(msg, flush=True)
    logger.info(msg)


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


def _comfy_quant_conf_has_convrot(conf) -> bool:
    if not isinstance(conf, dict):
        return False
    if conf.get("convrot") is True:
        return True
    params = conf.get("params")
    if isinstance(params, dict) and params.get("convrot") is True:
        return True
    return False


def checkpoint_looks_like_comfy_quant_convrot(state_dict_or_path) -> bool:
    """True if checkpoint marks int8_tensorwise layers with ConvRot (Hadamard)."""
    if isinstance(state_dict_or_path, (str, os.PathLike)):
        return _probe_path_comfy_quant_convrot(str(state_dict_or_path))

    state_dict = state_dict_or_path
    import torch

    for key, value in state_dict.items():
        if not key.endswith(".comfy_quant"):
            continue
        if not torch.is_tensor(value) and not isinstance(value, (dict, bytes, bytearray, str)):
            continue
        conf = decode_comfy_quant_conf(value)
        if _comfy_quant_conf_has_convrot(conf):
            return True
    return False


def checkpoint_needs_hswq_int8_conv2d(state_dict_or_path) -> bool:
    """True for SDXL/ZI-style UNets that need HSWQ INT8 Conv2d patches.

    Keyed off architecture (``input_blocks`` / ``middle_block`` / ``output_blocks``),
    not off ConvRot. DiT/Krea2 (``double_blocks`` / ``single_blocks``) returns False
    so ConvRot stock load stays free of our Conv2d inject (VRAM).
    """
    if isinstance(state_dict_or_path, (str, os.PathLike)):
        return _probe_path_needs_hswq_int8_conv2d(str(state_dict_or_path))

    keys = list(state_dict_or_path.keys())
    return _keys_need_hswq_int8_conv2d(keys)


def _keys_need_hswq_int8_conv2d(keys) -> bool:
    sdxl = False
    dit = False
    for k in keys:
        if (
            ".input_blocks." in k
            or ".middle_block." in k
            or ".output_blocks." in k
            or k.startswith("input_blocks.")
            or k.startswith("middle_block.")
            or k.startswith("output_blocks.")
        ):
            sdxl = True
        if (
            ".double_blocks." in k
            or ".single_blocks." in k
            or ".joint_blocks." in k
            or k.startswith("double_blocks.")
            or k.startswith("single_blocks.")
            or k.startswith("joint_blocks.")
        ):
            dit = True
        if sdxl and dit:
            break
    # Prefer SDXL Conv2d path when UNet blocks exist; DiT-only → no inject.
    if sdxl:
        return True
    return False


def _probe_path_needs_hswq_int8_conv2d(path: str) -> bool:
    try:
        from safetensors import safe_open
    except ImportError:
        # Filename heuristics only as last resort.
        base = os.path.basename(path).lower()
        if "krea" in base or "dit" in base:
            return False
        return True
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            return _keys_need_hswq_int8_conv2d(list(f.keys()))
    except Exception as e:
        logger.debug("[HSWQ INT8] SDXL/ZI Conv2d need probe failed for %s: %s", path, e)
        base = os.path.basename(path).lower()
        if "krea" in base or "convrot" in base or "int8convrot" in base:
            return False
        return True


def _probe_path_comfy_quant_convrot(path: str) -> bool:
    """Lightweight safetensors probe for comfy_quant.convrot=true."""
    try:
        from safetensors import safe_open
    except ImportError:
        return "convrot" in os.path.basename(path).lower()
    base = os.path.basename(path).lower()
    name_hint = "convrot" in base or "int8convrot" in base
    comfy_keys = []
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            comfy_keys = [k for k in keys if k.endswith(".comfy_quant")]
            for ck in comfy_keys[:32]:
                conf = decode_comfy_quant_conf(f.get_tensor(ck))
                if _comfy_quant_conf_has_convrot(conf):
                    return True
            meta = f.metadata() or {}
            if "_quantization_metadata" in meta:
                try:
                    qm = json.loads(meta["_quantization_metadata"])
                    layers = qm.get("layers", {}) if isinstance(qm, dict) else {}
                    for v in layers.values():
                        if isinstance(v, dict) and _comfy_quant_conf_has_convrot(v):
                            return True
                except (TypeError, json.JSONDecodeError):
                    pass
    except Exception as e:
        logger.debug("[HSWQ INT8] ConvRot probe failed for %s: %s", path, e)
        return name_hint
    # Filename alone is enough for *Int8Convrot* when markers were stripped/odd.
    return name_hint


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
    # Only while an HSWQ INT8 UNet/Checkpoint load explicitly opens the scope.
    # Do NOT key off quant_config alone: once mixed_precision_ops is monkeypatched,
    # stock UNETLoader / Krea2 ConvRot loads also build MixedPrecisionOps with
    # int8_tensorwise config — injecting our Conv2d there is wrong for DiT/ConvRot
    # and can inflate VRAM vs stock.
    _ = quant_config
    return bool(getattr(_int8_quant_conv_tls, "active", False))


def _module_path_is_real_nunchaku_package(mod: str) -> bool:
    """True only for real Nunchaku package modules — never this unofficial-loader.

    INT8 Conv2d from this extension lives under a path containing ``nunchaku``;
    a bare ``\"nunchaku\" in path`` false-positive armed VRAM handoff on
    non-SVDQ loads (SDXL INT8 and any other architecture using those Conv2d)
    and destroyed normal generation. Substring match is forbidden.
    """
    mod_l = (mod or "").lower().replace("\\", "/")
    if not mod_l:
        return False
    # This extension / INT8 patch path must never count as SVDQ.
    if (
        "unofficial" in mod_l
        or "comfy_quant_int8" in mod_l
        or "nunchaku-unofficial" in mod_l
        or "nunchaku_unofficial" in mod_l
    ):
        return False
    if mod_l == "nunchaku" or mod_l.startswith("nunchaku."):
        return True
    if ".nunchaku." in mod_l:
        return True
    return False


def _model_is_nunchaku_svdq(model) -> bool:
    """True only when the graph carries real Nunchaku SVDQ modules.

    ComfyUI registers Z-Image as ``Lumina2`` — classname checks for
    ``Nunchaku`` / ``ZImage`` miss that. Any SVDQ / ComfyNunchaku module means
    never run comfy_quant INT8 Dynamic LoRA bake.

    Branch: everything that is not real SVDQ (SDXL, Flux, ZIT, native INT8,
    FP, …) returns False. Module-path checks must not match this
    unofficial-loader package (see ``_module_path_is_real_nunchaku_package``).
    """
    if model is None:
        return False
    roots = [model]
    dm = getattr(model, "diffusion_model", None)
    if dm is not None:
        roots.append(dm)
    inner = getattr(model, "model", None)
    if inner is not None and inner is not model:
        roots.append(inner)
        dm2 = getattr(inner, "diffusion_model", None)
        if dm2 is not None:
            roots.append(dm2)
    seen = set()
    _checked = 0
    _MAX_CHECK_SVDQ = 100  # Early exit: SVDQ modules are typically at the top
    for root in roots:
        rid = id(root)
        if rid in seen:
            continue
        seen.add(rid)
        try:
            named = root.named_modules()
        except Exception:
            continue
        for _, module in named:
            cls_name = type(module).__name__
            if (
                "SVDQ" in cls_name
                or "Nunchaku" in cls_name
                or cls_name.startswith("ComfyNunchaku")
            ):
                return True
            mod = getattr(type(module), "__module__", "") or ""
            if _module_path_is_real_nunchaku_package(mod):
                return True
            _checked += 1
            if _checked >= _MAX_CHECK_SVDQ:
                break
    return False


def _model_is_zimage_nvfp4_parity(model) -> bool:
    """True for Z Image / Krea ConvRot NVFP4 parity packs (not SDXL TC).

    Absolute branch: shared INT8 Dynamic bake must not own these models.
    Z Image bake lives under ``nodes/zimage_nvfp4/nvfp4_lora_bake.py``.
    """
    if model is None:
        return False
    for _, module in model.named_modules():
        if getattr(module, "_hswq_nvfp4_convrot_parity", False):
            return True
        fwd = getattr(module, "forward", None)
        if fwd is not None and getattr(fwd, "_hswq_nvfp4_convrot_parity", False):
            return True
    return False


def _model_has_int8_quantized_weights(model) -> bool:
    """True for native comfy_quant QuantizedTensor (INT8 / NVFP4 / FP8 QT).

    Must NOT treat bare ``torch.int8`` weights as comfy_quant INT8.
    Nunchaku SVDQ / Z-Image / Lumina2 modules often use int8 storage; a false
    positive here arms Dynamic.load INT8 LoRA bake and can Abort those paths.

    SDXL ConvRot NVFP4 (Checkpoint Loader dropdown ``ConvRot NVFP4``):
      bake must see NVFP4 Linear QT so ``patch_weight_to_device`` runs
      nodes/nvfp4 ConvRot convert/set_weight (3.3.0 SDXL path).

    Z Image ConvRot NVFP4 (UNet dropdown ``Z Image ConvRot NVFP4``):
      do not treat as the SDXL bake path — see ``_model_is_zimage_nvfp4_parity``.
    """
    if _model_is_nunchaku_svdq(model):
        return False
    try:
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return False
    _checked = 0
    _MAX_CHECK = 200  # Early exit: only scan first 200 modules
    for _, module in model.named_modules():
        cls_name = type(module).__name__
        if "SVDQ" in cls_name or "Nunchaku" in cls_name:
            continue
        w = getattr(module, "weight", None)
        if w is None:
            continue
        if isinstance(w, QuantizedTensor):
            return True
        # 0.30.2: Parameter wrapping QuantizedTensor
        if hasattr(w, "data") and isinstance(w.data, QuantizedTensor):
            return True
        # Fast path: layout_type set means quantized
        if getattr(module, "layout_type", None) is not None:
            return True
        _checked += 1
        if _checked >= _MAX_CHECK:
            break
    return False


def _load_native_convert_int8_helpers():
    """Lazy-load Hadamard / rotate helpers from sibling native_convert_int8.py."""
    import importlib.util
    import sys

    global _NATIVE_CONVERT_INT8_MOD
    if _NATIVE_CONVERT_INT8_MOD is not None:
        return _NATIVE_CONVERT_INT8_MOD
    name = "native_convert_int8_for_hswq_conv2d"
    # torch.compile / Dynamo re-imports helpers by this module name (LOAD_GLOBAL).
    # Without sys.modules registration, Dynamo raises ModuleNotFoundError.
    existing = sys.modules.get(name)
    if existing is not None:
        _NATIVE_CONVERT_INT8_MOD = existing
        return existing
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "native_convert_int8.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"native_convert_int8.py not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    _NATIVE_CONVERT_INT8_MOD = mod
    return mod


_NATIVE_CONVERT_INT8_MOD = None


def _qt_payload(weight, QuantizedTensor):
    """Unwrap Parameter → QuantizedTensor if needed."""
    if weight is None:
        return None
    if isinstance(weight, QuantizedTensor):
        return weight
    data = getattr(weight, "data", None)
    if isinstance(data, QuantizedTensor):
        return data
    return None


def _arm_hswq_conv2d_convrot(module, QuantizedTensor):
    """Full ConvRot on Conv2d: keep online rotate on module; clear kitchen Params.convrot.

    Kitchen dequantize_int8_convrot_* is 2D-only. Stamping Params.convrot=True on
    4D weights and calling .dequantize() crashes. Weights stay in rotated basis;
    forward rotates NCHW activations; LoRA convert_weight unrotates to float space.
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
            self._hswq_convrot = False
            self._hswq_convrot_groupsize = 256

        def reset_parameters(self):
            return None

        def _load_from_state_dict(self, *args):
            _load_quantized_module(self, super()._load_from_state_dict, *args, load_extra_params=False)
            _arm_hswq_conv2d_convrot(self, QuantizedTensor)

        def state_dict(self, *args, destination=None, prefix="", **kwargs):
            sd = destination if destination is not None else {}
            sd = _quantized_weight_state_dict(self, sd, prefix, extra_quant_params=("input_scale", "pre_quant_scale"))
            # Re-stamp ConvRot on export (Params.convrot cleared for safe 4D dequant).
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
            if getattr(self, "_hswq_convrot", False):
                nc = _load_native_convert_int8_helpers()
                gs = int(getattr(self, "_hswq_convrot_groupsize", 256) or 256)
                # Use GPU-cached Hadamard via nc.get_hadamard_on_device
                h = nc.get_hadamard_on_device(gs, device=input.device, dtype=input.dtype)
                input = nc.rotate_activation_nchw(input, h, gs)
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
            # ConvRot weights are stored rotated; unrotate to original float basis for LoRA.
            # LowVRAM may re-materialize QT with Params.convrot still True — clear
            # before dequantize (kitchen ConvRot dequant is 2D-only).
            global _lora_convert_logs
            if isinstance(weight, QuantizedTensor):
                _arm_hswq_conv2d_convrot(self, QuantizedTensor)
                qt = _qt_payload(weight, QuantizedTensor)
                if qt is not None:
                    params = getattr(qt, "_params", None)
                    qdata = getattr(qt, "_qdata", None)
                    if (
                        params is not None
                        and qdata is not None
                        and getattr(qdata, "ndim", None) == 4
                        and bool(getattr(params, "convrot", False))
                    ):
                        import dataclasses

                        gs = int(getattr(params, "convrot_groupsize", 256) or 256)
                        self._hswq_convrot = True
                        self._hswq_convrot_groupsize = gs
                        new_params = dataclasses.replace(params, convrot=False)
                        try:
                            object.__setattr__(qt, "_params", new_params)
                        except Exception:
                            qt._params = new_params
                out = weight.dequantize()
            else:
                out = weight
            if getattr(self, "_hswq_convrot", False) and out is not None and getattr(out, "ndim", 0) == 4:
                nc = _load_native_convert_int8_helpers()
                gs = int(getattr(self, "_hswq_convrot_groupsize", 256) or 256)
                h = nc.build_hadamard(gs, device="cpu", dtype=torch.float32)
                out = nc.unrotate_weight_conv2d(out, h, gs)
                # Note: unrotate stays on CPU since LoRA bake is CPU-side
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
            # ConvRot: convert_weight returned unrotated float; re-rotate before requant.
            global _lora_set_logs
            layout = getattr(self, "layout_type", None)
            path = "requant" if layout is not None else "cast_only"
            if getattr(self, "_hswq_convrot", False) and getattr(weight, "ndim", 0) == 4:
                nc = _load_native_convert_int8_helpers()
                gs = int(getattr(self, "_hswq_convrot_groupsize", 256) or 256)
                h = nc.build_hadamard(gs, device="cpu", dtype=torch.float32)
                weight = nc.rotate_weight_conv2d(weight, h, gs)
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
    _OPS_PATCH_VER = 4  # Conv2d full ConvRot: online act rotate + safe 4D dequant
    true_orig = getattr(original_mp, "_hswq_orig_mixed_precision_ops", original_mp)
    if (
        getattr(original_mp, "_hswq_int8_ops_ver", 0) >= _OPS_PATCH_VER
        and getattr(original_mp, "_hswq_int8_conv_patched", False)
    ):
        # Heal older INT8 wraps that dropped NVFP4 stack markers (false upgrade).
        if int(getattr(original_mp, "_hswq_nvfp4_stack_ver", 0) or 0) <= 0:
            inner = true_orig
            for _ in range(6):
                if inner is None:
                    break
                v = int(getattr(inner, "_hswq_nvfp4_stack_ver", 0) or 0)
                if v > 0:
                    try:
                        original_mp._hswq_nvfp4_stack_ver = v
                    except Exception:
                        pass
                    break
                if getattr(inner, "_hswq_nvfp4_comfy_only", False):
                    try:
                        original_mp._hswq_nvfp4_comfy_only = True
                        if hasattr(inner, "_hswq_nvfp4_orig_mp"):
                            original_mp._hswq_nvfp4_orig_mp = getattr(
                                inner, "_hswq_nvfp4_orig_mp"
                            )
                    except Exception:
                        pass
                inner = getattr(inner, "_hswq_nvfp4_orig_mp", None) or getattr(
                    inner, "_hswq_orig_mixed_precision_ops", None
                )
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
            disabled = set()
        elif isinstance(disabled, list):
            disabled = set(disabled)
        result = true_orig(
            quant_config=quant_config,
            compute_dtype=compute_dtype,
            full_precision_mm=full_precision_mm,
            disabled=disabled,
        )
        # Inject Quantized Conv2d only during HSWQ INT8 load scope
        # (_int8_quant_conv_scope). Never from quant_config alone — that would
        # also hit stock UNETLoader / Krea2 ConvRot MixedPrecision builds.
        if _should_inject_int8_conv(quant_config):
            result.Conv2d = _make_quantized_conv2d(ops_module, result, disabled)
        return result

    mixed_precision_ops_force_conv._hswq_orig_mixed_precision_ops = true_orig
    mixed_precision_ops_force_conv._hswq_int8_conv_patched = True
    mixed_precision_ops_force_conv._hswq_int8_ops_ver = _OPS_PATCH_VER
    # Preserve NVFP4 markers through INT8 wrap so Z Image early-return works
    # (lost stack_ver → false "upgraded stack" → TC over parity → noise).
    for _attr in (
        "_hswq_nvfp4_stack_ver",
        "_hswq_nvfp4_comfy_only",
        "_hswq_nvfp4_orig_mp",
    ):
        if hasattr(original_mp, _attr):
            try:
                setattr(
                    mixed_precision_ops_force_conv,
                    _attr,
                    getattr(original_mp, _attr),
                )
            except Exception:
                pass
    ops_module.mixed_precision_ops = mixed_precision_ops_force_conv
    return True


def _patch_lowvram_patch_float_intermediate() -> bool:
    """Fix LowVramPatch intermediate_dtype for comfy_quant QuantizedTensor only.

    Upstream LowVramPatch passes intermediate_dtype=weight.dtype. When the
    weight is still a QuantizedTensor (int8 storage), LoRA matmul casts to
    int8 and either errors or silently produces a no-op delta — same bug as
    BobJohnson24/ComfyUI-INT8-Fast#76.

    Must NOT divert bare ``torch.int8`` tensors. Nunchaku SVDQ / Lumina2 use
    int8 storage; grabbing them here corrupts fused CUDA (Abort in
    ``_forward_silu_gating``) even when VRAM handoff already freed GPU memory.
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
    _LV_VER = 3
    if original is None or getattr(original, "_hswq_int8_lora_dtype_ver", 0) >= _LV_VER:
        return getattr(original, "_hswq_int8_lora_dtype", False)
    true_orig = getattr(original, "_hswq_orig_lowvram_call", original)

    def __call__(self, weight):
        # QuantizedTensor only. Bare int8 / float / None -> upstream unchanged.
        if weight is None or not isinstance(weight, QuantizedTensor):
            return true_orig(self, weight)
        patches = (
            self.prepared_patches
            if self.prepared_patches is not None
            else self.patches[self.key]
        )
        w = weight.dequantize()
        dtype = getattr(w, "dtype", None)
        if dtype is not None and hasattr(dtype, "is_floating_point") and dtype.is_floating_point:
            idtype = dtype
        else:
            idtype = torch.float32
        # 0.30.2: calculate_weight accepts original_weight=None
        try:
            return comfy.lora.calculate_weight(patches, w, self.key, intermediate_dtype=idtype, original_weights=None)
        except TypeError:
            # Fallback for older ComfyUI without original_weights param
            return comfy.lora.calculate_weight(patches, w, self.key, intermediate_dtype=idtype)

    __call__._hswq_int8_lora_dtype = True
    __call__._hswq_int8_lora_dtype_ver = _LV_VER
    __call__._hswq_orig_lowvram_call = true_orig
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
    if _model_is_nunchaku_svdq(getattr(patcher, "model", None)):
        return 0
    # Absolute branch: Z Image / Krea parity packs use zimage_nvfp4 bake only.
    # Never apply the SDXL (3.3.0 nodes/nvfp4) Dynamic bake to them.
    if _model_is_zimage_nvfp4_parity(getattr(patcher, "model", None)):
        return 0
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
            # 0.30.2: weight may be Parameter(QuantizedTensor) - unwrap for isinstance check
            weight_inner = weight.data if hasattr(weight, "data") else weight
            # SDXL path (3.3.0): bake all comfy_quant QuantizedTensor — never bare int8.
            # NVFP4 Linear uses nodes/nvfp4 ConvRot convert/set_weight.
            # Z Image path never reaches here (parity early-return above).
            if not isinstance(weight_inner, QuantizedTensor):
                continue
            if set_func is None:
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
    _DYN_VER = 8
    # Never wrap over live Z Image bake. Capturing ZI as true_orig leaves
    # ``Dynamic.load ENTER … model=SDXL`` after ZI uninstall (closure), and SDXL
    # LoRA bake strength breaks even when Status shows APPLIED.
    if getattr(original, "_hswq_zi_nvfp4_lora_bake", False):
        return True
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
        # INT8 LoRA bake only — never touch Nunchaku SVDQ (class is often Lumina2).
        if _model_is_nunchaku_svdq(self.model):
            return result
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


def _force_detach_int8_dynamic_models(device=None, keep_patchers=None) -> int:
    """Offload INT8 Dynamic VRAM (VBAR + hostbufs) without destroying QT weights.

    free_memory often stops after partially_unload and leaves HostBuffers, so
    Nunchaku sees ``0.00 MB usable`` and Aborts. We must fully offload INT8
    Dynamic models before SVDQ load.

    Critical: use ``unpatch_weights=False`` / ``detach(unpatch_all=False)``.
    ``unpatch_all=True`` unpatches INT8 QuantizedTensor + baked LoRA and causes
    black / noise on the next normal SDXL INT8 KSampler.
    """
    try:
        import comfy.model_management as mm
    except ImportError:
        return 0

    keep_ids = {id(p) for p in (keep_patchers or []) if p is not None}
    unloaded = 0
    i = 0
    while i < len(mm.current_loaded_models):
        lm = mm.current_loaded_models[i]
        patcher = lm.model
        if patcher is None:
            i += 1
            continue
        if id(patcher) in keep_ids:
            i += 1
            continue
        if device is not None and getattr(lm, "device", None) is not None:
            try:
                if str(lm.device) != str(device):
                    i += 1
                    continue
            except Exception:
                pass
        is_dyn = False
        try:
            is_dyn = bool(patcher.is_dynamic())
        except Exception:
            is_dyn = False
        if not is_dyn:
            i += 1
            continue
        base = getattr(patcher, "model", None)
        if base is None or not _model_has_int8_quantized_weights(base):
            i += 1
            continue
        # Preserve QT + baked LoRA; only free GPU / VBAR occupancy.
        try:
            lm.model_unload(unpatch_weights=False)
        except TypeError:
            try:
                patcher.detach(unpatch_all=False)
            except TypeError:
                try:
                    patcher.detach(False)
                except Exception as exc:
                    _console(f"[HSWQ INT8→Nunchaku] detach(False) failed: {exc!r}")
            except Exception as exc:
                _console(f"[HSWQ INT8→Nunchaku] detach(unpatch_all=False) failed: {exc!r}")
            try:
                fin = getattr(lm, "model_finalizer", None)
                if fin is not None:
                    fin.detach()
            except Exception:
                pass
            try:
                lm.model_finalizer = None
                lm.real_model = None
            except Exception:
                pass
        except Exception as exc:
            _console(f"[HSWQ INT8→Nunchaku] model_unload(False) failed: {exc!r}")
            i += 1
            continue
        mm.current_loaded_models.pop(i)
        unloaded += 1
    if unloaded > 0:
        try:
            mm.soft_empty_cache()
        except Exception:
            pass
    return unloaded


def _patch_load_models_gpu_int8_nunchaku_handoff() -> bool:
    """Before Nunchaku SVDQ load, offload INT8 Dynamic VRAM without unpatch."""
    try:
        import comfy.model_management as mm
    except ImportError:
        return False

    original = getattr(mm, "load_models_gpu", None)
    if original is None:
        return False
    # v10 = handoff arms ONLY for real Nunchaku SVDQ. All other loads
    # (SDXL / Flux / ZIT / native INT8 / FP / …) pass through untouched.
    # free_memory → model_unload(unpatch_weights=True) kills non-SVDQ INT8.
    _VER = 10
    if getattr(original, "_hswq_int8_nunchaku_handoff_ver", 0) >= _VER:
        return True
    true_orig = getattr(original, "_hswq_orig_load_models_gpu", original)

    def load_models_gpu(
        models,
        memory_required=0,
        force_patch_weights=False,
        minimum_memory_required=None,
        force_full_load=False,
    ):
        keep = []
        need_handoff = False
        device = None
        for m in models or []:
            keep.append(m)
            for mm_extra in getattr(m, "model_patches_models", lambda: [])() or []:
                keep.append(mm_extra)
            base = getattr(m, "model", None)
            # Branch A: native comfy_quant INT8 (any architecture) — never handoff.
            if base is not None and _model_has_int8_quantized_weights(base):
                continue
            # Branch B: only real Nunchaku SVDQ on the BaseModel arms handoff.
            # Do not probe the ModelPatcher itself (false positives).
            if base is not None and _model_is_nunchaku_svdq(base):
                need_handoff = True
                if device is None:
                    device = getattr(m, "load_device", None)
        if need_handoff:
            n = _force_detach_int8_dynamic_models(device=device, keep_patchers=keep)
            # Second pass: any INT8 Dynamic still listed (missed first pass) —
            # never leave them for free_memory(unpatch=True).
            n2 = _force_detach_int8_dynamic_models(device=None, keep_patchers=keep)
            try:
                mm.soft_empty_cache()
            except Exception as exc:
                _console(f"[HSWQ INT8→Nunchaku] soft_empty_cache failed: {exc!r}")
            _console(
                f"[HSWQ INT8→Nunchaku] VRAM handoff before SVDQ load "
                f"(INT8 Dynamic offload keep-weights={n + n2}, no free_memory unpatch)"
            )
        return true_orig(
            models,
            memory_required=memory_required,
            force_patch_weights=force_patch_weights,
            minimum_memory_required=minimum_memory_required,
            # Full load after handoff avoids Nunchaku 0.00 MB usable Abort.
            # Non-SVDQ loads never set need_handoff (branches A/B above).
            force_full_load=True if need_handoff else force_full_load,
        )

    load_models_gpu._hswq_int8_nunchaku_handoff = True
    load_models_gpu._hswq_int8_nunchaku_handoff_ver = _VER
    load_models_gpu._hswq_orig_load_models_gpu = true_orig
    mm.load_models_gpu = load_models_gpu
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


def _patch_controllora_int8_dequant() -> bool:
    """Dequantize borrowed base-UNet quantized weights in ControlLora.pre_run.

    LoRA-type ControlNets (``lora_controlnet`` marker, e.g. anytest) build a
    control_model that BORROWS the base UNet's own weights via
    ``diffusion_model.state_dict()`` and injects them with ``set_attr_param``.
    The control_model uses ``ControlLoraOps`` (plain float ops); its forward
    calls ``comfy.ops.cast_bias_weight``, which cannot reconstruct a quantized
    weight without its scale.

    Root cause (confirmed from logs + comfy/ops.py):
    ``MixedPrecisionOps.state_dict`` (``_quantized_weight_state_dict``) does NOT
    emit ``QuantizedTensor`` objects. It FLATTENS each quantized ``weight`` into
    separate tensors:
      * ``X.weight``        -> raw int8 qdata      (torch.int8)
      * ``X.weight_scale``  -> per-tensor scale    (torch.float32)
      * ``X.comfy_quant``   -> JSON metadata       (torch.uint8)
      * ``X.input_scale`` / ``X.weight_scale_2`` -> extra params (fp8/nvfp4)
    So ``ControlLora.pre_run`` injects the RAW int8 ``X.weight`` (no scale) into
    the float control_model, and forward feeds int8 straight into
    ``F.linear`` / ``conv2d`` -> NaN / black output. FP8 avoids this only
    because its dtype differs from the compute dtype.

    Fix: wrap ``diffusion_model.state_dict`` during ``ControlLora.pre_run`` and
    return a DEQUANTIZED state dict: for every module whose ``.weight`` is a
    ``QuantizedTensor``, replace ``X.weight`` with ``weight.dequantize()`` (a
    real float tensor) and drop the now-meaningless sidecar keys
    (``X.weight_scale``, ``X.weight_scale_2``, ``X.comfy_quant``,
    ``X.input_scale``). All non-quant weights, biases and buffers pass through
    unchanged. Full-weight ControlNets (Canny) never enter
    ``ControlLora.pre_run`` and are unaffected; the real anytest LoRA weights
    (``.up`` / ``.down``) are plain fp16 and are not touched.
    """
    try:
        import comfy.controlnet as cn
        import comfy.utils
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return False

    ControlLora = getattr(cn, "ControlLora", None)
    if ControlLora is None:
        return False
    original = getattr(ControlLora, "pre_run", None)
    _CL_VER = 2
    if original is None or getattr(original, "_hswq_int8_controllora_ver", 0) >= _CL_VER:
        return getattr(original, "_hswq_int8_controllora", False)
    true_orig = getattr(original, "_hswq_orig_controllora_pre_run", original)

    def _dequantized_state_dict(diffusion_model, orig_sd):
        """Return diffusion_model.state_dict() with quantized weights turned
        back into float tensors and their scale/metadata sidecars removed.

        ``orig_sd`` is the ORIGINAL bound ``state_dict`` method captured before
        we replaced ``diffusion_model.state_dict``. It MUST be used here instead
        of ``diffusion_model.state_dict()`` to avoid re-entering our wrapper
        (which caused ``RecursionError: maximum recursion depth exceeded``)."""
        full = orig_sd()

        # Collect the state-dict prefix of every quantized weight.
        quant_weight_keys = {}
        for name, module in diffusion_model.named_modules():
            w = getattr(module, "weight", None)
            if isinstance(w, QuantizedTensor):
                key = (name + "." if name else "") + "weight"
                quant_weight_keys[key] = w

        out = {}
        n_dequant = 0
        n_drop = 0
        for k, v in full.items():
            replaced = False
            dropped = False
            for wk, qt in quant_weight_keys.items():
                if k == wk:
                    # raw int8 qdata -> real float weight
                    try:
                        out[k] = qt.dequantize()
                        n_dequant += 1
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "[HSWQ INT8] ControlLora: dequantize failed for %s: %s",
                            k, e,
                        )
                        out[k] = v
                    replaced = True
                    break
                base = wk[: -len("weight")]  # "X."
                if (
                    (k.startswith(wk) and k != wk)      # X.weight_scale / weight_scale_2
                    or k == base + "comfy_quant"        # uint8 JSON metadata
                    or k == base + "input_scale"        # fp8 extra param
                ):
                    dropped = True
                    break
            if replaced:
                continue
            if dropped:
                n_drop += 1
                continue
            out[k] = v

        print(
            f"[HSWQ INT8][ControlLora] dequantized state_dict: "
            f"weights dequantized(int8->float)={n_dequant}, "
            f"sidecar keys dropped(scale/comfy_quant/input_scale)={n_drop}, "
            f"total keys out={len(out)}",
            flush=True,
        )
        return out

    def pre_run(self, model, percent_to_timestep_function):
        diffusion_model = getattr(model, "diffusion_model", None)
        patched = False
        orig_sd = None
        if diffusion_model is not None:
            orig_sd = diffusion_model.state_dict

            def dequant_state_dict(*a, **kw):
                # Only intercept the argument-less borrow call ControlLora makes;
                # fall back to the original for any keyword/destination usage.
                if a or kw:
                    return orig_sd(*a, **kw)
                return _dequantized_state_dict(diffusion_model, orig_sd)

            print(
                "[HSWQ INT8][ControlLora] pre_run ENTER "
                "(LoRA-type ControlNet / lora_controlnet path) "
                "-> wrapping diffusion_model.state_dict for INT8 base-weight dequant",
                flush=True,
            )
            diffusion_model.state_dict = dequant_state_dict
            patched = True
        else:
            print(
                "[HSWQ INT8][ControlLora] pre_run ENTER but model has no "
                "diffusion_model; running unpatched",
                flush=True,
            )

        try:
            result = true_orig(self, model, percent_to_timestep_function)
        finally:
            if patched:
                # Remove the instance-level override so the class method is used again.
                try:
                    del diffusion_model.state_dict
                except AttributeError:
                    diffusion_model.state_dict = orig_sd

        print(
            "[HSWQ INT8][ControlLora] pre_run EXIT (base weights injected as float)",
            flush=True,
        )
        logger.info(
            "[HSWQ INT8] ControlLora: injected dequantized base UNet weights "
            "(anytest / lora_controlnet black-output fix)"
        )
        return result

    pre_run._hswq_int8_controllora = True
    pre_run._hswq_int8_controllora_ver = _CL_VER
    pre_run._hswq_orig_controllora_pre_run = true_orig
    ControlLora.pre_run = pre_run
    print(
        "[HSWQ INT8][ControlLora] pre_run patch INSTALLED "
        "(v%d): borrowed INT8 base weights dequantized via state_dict wrap "
        "for LoRA-type ControlNet (anytest fix)" % _CL_VER,
        flush=True,
    )
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
    ok_handoff = _patch_load_models_gpu_int8_nunchaku_handoff()
    ok_controllora = _patch_controllora_int8_dequant()
    # Re-apply ops when patch version bumps (e.g. Conv2d inject gate change).
    ok_ops = _patch_ops_decode_and_conv()
    if _PATCHES_APPLIED:
        return True
    ok_utils = _patch_convert_old_quants()
    ok_lora_log = _patch_model_patcher_lora_logs()
    if ok_ops:
        _PATCHES_APPLIED = True
        _console(
            "[HSWQ INT8] comfy_quant patches applied "
            f"(Conv2d quant load + decode"
            f"{' + convert_old_quants' if ok_utils else ''}"
            f"{' + LoRA bake logs' if ok_lora_log else ''}"
            f"{' + LoRA key counts' if ok_keys else ''}"
            f"{' + LoRA name' if ok_name or ok_path or ok_torch else ''}"
            f"{' + LowVramPatch float dtype' if ok_lowvram else ''}"
            f"{' + Dynamic INT8 LoRA bake' if ok_dyn_bake else ''}"
            f"{' + INT8→Nunchaku VRAM handoff' if ok_handoff else ''}"
            f"{' + ControlLora INT8 dequant' if ok_controllora else ''})"
        )
        return True
    logger.warning(
        "[HSWQ INT8] Failed to apply comfy_quant patches (ops=%s utils=%s)",
        ok_ops,
        ok_utils,
    )
    return False


KREA2_MODEL_FLAG = "_hswq_is_krea2"


def model_is_krea2(model) -> bool:
    """Krea2 check taken from ComfyUI's own architecture detection.

    ``model_detection`` writes ``image_model = "krea2"`` into ``unet_config``
    from the state dict, and picks ``supported_models.Krea2`` /
    ``model_base.Krea2``. Those are exact identities, not name guesses, so a
    file rename or a substring collision cannot flip the answer.
    """
    if model is None:
        return False

    inner = getattr(model, "model", None) or model
    if getattr(model, KREA2_MODEL_FLAG, False) or getattr(inner, KREA2_MODEL_FLAG, False):
        return True

    config = getattr(inner, "model_config", None)
    unet_config = getattr(config, "unet_config", None)
    if isinstance(unet_config, dict) and str(unet_config.get("image_model", "")).lower() == "krea2":
        return True

    return type(config).__name__ == "Krea2" or type(inner).__name__ == "Krea2"


def tag_krea2_model(model) -> bool:
    """Stamp the Krea2 verdict onto the model so later nodes read, not guess.

    The flag goes on the inner model too because ``ModelPatcher`` clones
    re-wrap the same inner model and would otherwise drop it.
    """
    if not model_is_krea2(model):
        return False

    for obj in (model, getattr(model, "model", None)):
        if obj is None:
            continue
        try:
            setattr(obj, KREA2_MODEL_FLAG, True)
        except Exception:
            logger.debug("[HSWQ INT8] Could not tag %r as Krea2", type(obj).__name__)
    return True


def load_unet_hswq_weight_dtype(unet_name, weight_dtype):
    import logging
    import torch
    import folder_paths
    import comfy.sd

    # INT8 Conv2d patches: SDXL/ZI UNet (architecture), even if Linear has ConvRot.
    # Krea2/DiT ConvRot: stock-equivalent load — Conv2d inject inflates VRAM vs stock.
    unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
    is_convrot = checkpoint_looks_like_comfy_quant_convrot(unet_path)
    is_int8 = weight_dtype == "int8_tensorwise" or checkpoint_looks_like_comfy_quant_int8(unet_path)
    needs_conv2d = checkpoint_needs_hswq_int8_conv2d(unet_path)

    if is_int8 and is_convrot and not needs_conv2d:
        # Peel Z Image comfy_parity + ZI bake hooks before Krea2/DiT stock load.
        # Without this, comfy_parity stays on mixed_precision_ops / _load_quantized_module
        # from the previous Z Image run, arming _hswq_int8_convrot on Krea2 INT8 ConvRot
        # layers and installing forward_parity (online act rotate) on every Linear.
        # Krea2 does not need or want Z Image parity - it uses stock MixedPrecision -
        # but the leftover wrapper makes forward_parity fire Hadamard rotations every
        # step, causing progressive slowdown (4s/step -> 16s -> 22s -> 26s across runs).
        try:
            from ..nodes.nvfp4.comfy_quant_nvfp4 import (
                _clear_zimage_parity_contamination_for_sdxl,
            )

            _clear_zimage_parity_contamination_for_sdxl()
        except Exception as e:
            logging.warning(
                "[HSWQ INT8] clear Z Image NVFP4 contamination for Krea2 failed: %s", e
            )
        model_options = {}
        logging.info(
            "[HSWQ INT8] DiT/Krea2 ConvRot — stock-equivalent load "
            "(no INT8 Conv2d patches): %s",
            unet_name,
        )
        print(
            f"[HSWQ INT8] ConvRot DiT/Krea2 stock-equivalent load: {unet_name}",
            flush=True,
        )
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
    elif is_int8:
        apply_comfy_quant_int8_patches()
        model_options = {}
        reset_int8_lora_log_counters()
        if is_convrot and needs_conv2d:
            logging.info(
                "[HSWQ INT8] SDXL/ZI + ConvRot FULL — MixedPrecision + INT8 Conv2d "
                "(Linear: kitchen online; Conv2d: HSWQ online act rotate): %s",
                unet_name,
            )
            print(
                f"[HSWQ INT8] SDXL/ZI ConvRot FULL (Linear+Conv2d) load: {unet_name}",
                flush=True,
            )
        else:
            logging.info(
                "[HSWQ INT8] Loading UNet via MixedPrecisionOps (int8_tensorwise / comfy_quant)"
            )
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

    if tag_krea2_model(model):
        logging.info("[HSWQ INT8] Tagged as Krea2: %s", unet_name)

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
            # Z Image comfy_parity + ZI Dynamic bake must not wrap SDXL INT8.
            try:
                from ..nodes.nvfp4.comfy_quant_nvfp4 import (
                    _clear_zimage_parity_contamination_for_sdxl,
                )

                _clear_zimage_parity_contamination_for_sdxl()
            except Exception as e:
                sdxl_logger.warning(
                    "[SDXL INT8] clear Z Image NVFP4 contamination failed: %s", e
                )
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

    sdxl_cls = node_class_mappings.get("HSWQCheckpointLoaderSDXL")
    if sdxl_cls is not None:
        _orig_load_checkpoint = sdxl_cls.load_checkpoint

        def load_checkpoint(self, ckpt_name, weight_dtype, device=None, **kwargs):
            if weight_dtype in _FP8_WEIGHT_DTYPES:
                return _orig_load_checkpoint(self, ckpt_name, weight_dtype, device=device, **kwargs)
            if weight_dtype == "int8_tensorwise":
                return load_checkpoint_sdxl_hswq_weight_dtype(
                    ckpt_name, weight_dtype, device=device
                )
            import folder_paths

            ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
            if checkpoint_looks_like_comfy_quant_int8(ckpt_path):
                return load_checkpoint_sdxl_hswq_weight_dtype(
                    ckpt_name, weight_dtype, device=device
                )
            return _orig_load_checkpoint(self, ckpt_name, weight_dtype, device=device, **kwargs)

        sdxl_cls.load_checkpoint = load_checkpoint

    return True
