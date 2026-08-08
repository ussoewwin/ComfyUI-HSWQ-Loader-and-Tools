import logging
import os
from pathlib import Path

__version__ = "3.4.0"

import torch
from packaging.version import InvalidVersion, Version

# SDXL MultiGPU support (from comfyui-multigpu)
import comfy.model_management as mm

# Nunchaku SVDQ model download snippet removed (Nunchaku SDXL / Z-Image Turbo support dropped).

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info("=" * 40 + " ComfyUI-nunchaku Initialization " + "=" * 40)

# SDXL MultiGPU initialization
sdxl_logger = logging.getLogger("SDXL")
sdxl_logger.propagate = False
if not sdxl_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    sdxl_logger.addHandler(handler)
    sdxl_logger.setLevel(logging.INFO)

# SDXL device management
current_device = mm.get_torch_device()
current_text_encoder_device = mm.text_encoder_device()
current_unet_offload_device = mm.unet_offload_device()

def set_current_device(device):
    """Set the current device context for SDXL operations."""
    global current_device
    current_device = device
    sdxl_logger.debug(f"[SDXL Initialization] current_device set to: {device}")

def set_current_text_encoder_device(device):
    """Set the current text encoder device context for CLIP models."""
    global current_text_encoder_device
    current_text_encoder_device = device
    sdxl_logger.debug(f"[SDXL Initialization] current_text_encoder_device set to: {device}")

def set_current_unet_offload_device(device):
    """Set the current UNet offload device context."""
    global current_unet_offload_device
    current_unet_offload_device = device
    sdxl_logger.debug(f"[SDXL Initialization] current_unet_offload_device set to: {device}")

def get_current_device():
    """Get the current device context for SDXL operations at runtime."""
    return current_device

def get_current_text_encoder_device():
    """Get the current text encoder device context for CLIP models at runtime."""
    return current_text_encoder_device

def get_current_unet_offload_device():
    """Get the current UNet offload device context at runtime."""
    return current_unet_offload_device

def get_torch_device_patched():
    """Return SDXL-aware device selection for patched mm.get_torch_device."""
    from .device_utils import get_device_list, is_accelerator_available
    device = None
    if (not is_accelerator_available() or mm.cpu_state == mm.CPUState.CPU or "cpu" in str(current_device).lower()):
        device = torch.device("cpu")
    else:
        devs = set(get_device_list())
        device = torch.device(current_device) if str(current_device) in devs else torch.device("cpu")
    sdxl_logger.debug(f"[SDXL Core Patching] get_torch_device_patched returning device: {device} (current_device={current_device})")
    return device

def text_encoder_device_patched():
    """Return SDXL-aware text encoder device for patched mm.text_encoder_device."""
    from .device_utils import get_device_list, is_accelerator_available
    device = None
    if (not is_accelerator_available() or mm.cpu_state == mm.CPUState.CPU or "cpu" in str(current_text_encoder_device).lower()):
        device = torch.device("cpu")
    else:
        devs = set(get_device_list())
        device = torch.device(current_text_encoder_device) if str(current_text_encoder_device) in devs else torch.device("cpu")
    sdxl_logger.info(f"[SDXL Core Patching] text_encoder_device_patched returning device: {device} (current_text_encoder_device={current_text_encoder_device})")
    return device

def unet_offload_device_patched():
    """Return SDXL-aware UNet offload device for patched mm.unet_offload_device."""
    from .device_utils import get_device_list, is_accelerator_available
    device = None
    if (not is_accelerator_available() or mm.cpu_state == mm.CPUState.CPU or "cpu" in str(current_unet_offload_device).lower()):
        device = torch.device("cpu")
    else:
        devs = set(get_device_list())
        device = torch.device(current_unet_offload_device) if str(current_unet_offload_device) in devs else torch.device("cpu")
    sdxl_logger.debug(f"[SDXL Core Patching] unet_offload_device_patched returning device: {device} (current_unet_offload_device={current_unet_offload_device})")
    return device

sdxl_logger.info(f"[SDXL Core Patching] Patching mm.get_torch_device, mm.text_encoder_device, mm.unet_offload_device")
sdxl_logger.info(f"[SDXL DEBUG] Initial current_device: {current_device}")
sdxl_logger.info(f"[SDXL DEBUG] Initial current_text_encoder_device: {current_text_encoder_device}")
sdxl_logger.info(f"[SDXL DEBUG] Initial current_unet_offload_device: {current_unet_offload_device}")

mm.get_torch_device = get_torch_device_patched
mm.text_encoder_device = text_encoder_device_patched
mm.unet_offload_device = unet_offload_device_patched

from .utils import get_package_version, get_plugin_version

# PinCache / PinDebug must NOT install (removed; ComfyUI hostbuf handles pin).
# INT8 comfy_quant patches are NOT applied at import — only when an
# INT8 HSWQ / SDXL load path calls apply_comfy_quant_int8_patches().
# Dynamic INT8 LoRA bake must detect comfy QuantizedTensor only — never bare
# torch.int8 (Nunchaku SVDQ false positive → Abort on Z-Image / Lumina2).

# LATENT None-guard: stock VAEDecode / VAEDecodeTiled index samples["samples"],
# so a dropped LATENT kills the prompt after sampling already finished.
try:
    from .patches.vae_decode_none_guard import apply_vae_decode_none_guard
    if not apply_vae_decode_none_guard():
        logger.warning("LATENT None-guard found no decode entry point to arm")
except Exception:
    logger.exception("LATENT None-guard not applied")

# HSWQ Ultimate SD Upscale: apply copy_ / FP8 bias / embedder / Lumina compat patches in this extension
try:
    from .usdu_compat_patches import apply_usdu_compat_patches
    apply_usdu_compat_patches()
except Exception as e:
    logger.debug("USDU compat patches not applied: %s", e)

# torch.compile + FP8 + Lumina/Flux + comfy_kitchen compatibility patches.
# Apply only safe patches from NunchakuFluxLoraStacker patch module:
#   - LoraDiff.calculate_weight: skip LoRA when reshape != weight.shape (root fix)
#   - lumina.modulate / apply_gate: safe fallback on hidden_dim mismatch
#
# Note: do NOT apply global F.linear / matmul / mul / add patches.
# Those catch RuntimeError and slice tensors, but ComfyUI core uses the same
# RuntimeError to detect and skip bad LoRA; a global patch would steal that
# handling and can crash later (e.g. RMSNorm) with broken ranks.
try:
    import importlib
    _lora_stacker_patches = None
    _lora_stacker_root = Path(__file__).parent.parent / "ComfyUI-NunchakuFluxLoraStacker" / "patches"
    _lora_stacker_patch_file = _lora_stacker_root / "zimage_fp8_torchcompile.py"
    if _lora_stacker_patch_file.exists():
        spec = importlib.util.spec_from_file_location(
            "nunchaku_lora_stacker_patches.zimage_fp8_torchcompile",
            str(_lora_stacker_patch_file),
        )
        if spec and spec.loader:
            _lora_stacker_patches = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_lora_stacker_patches)

    _applied = []
    if _lora_stacker_patches:
        # Skip LoRA when reshape != weight.shape (root fix: block bad LoRA)
        _fn = getattr(_lora_stacker_patches, "_apply_lumina_modulate_patch", None)
        if _fn and _fn():
            _applied.append("lumina.modulate")
        _fn = getattr(_lora_stacker_patches, "_apply_lumina_apply_gate_patch", None)
        if _fn and _fn():
            _applied.append("lumina.apply_gate")

        # LoraDiff.calculate_weight patch lives at end of apply_patches(); apply directly
        try:
            try:
                import comfy.weight_adapter.lora as _lora_mod
            except ImportError:
                import comfy.lora as _lora_mod
            _LoraDiff = getattr(_lora_mod, "LoraDiff", None)
            if _LoraDiff is not None:
                _orig_cw = getattr(_LoraDiff, "calculate_weight", None)
                if _orig_cw and not getattr(_orig_cw, "_hswq_lora_skip_patched", False):
                    import torch as _torch_lora

                    def _lora_skip_calculate_weight(
                        self, weight, key, strength, strength_model, offset,
                        function, intermediate_dtype=_torch_lora.float32, original_weight=None,
                    ):
                        v = self.weights
                        reshape = v[5]
                        if reshape is not None and tuple(reshape) != weight.shape:
                            logger.warning(
                                "LoRA %s: skip %s (reshape %s != weight %s) [HSWQ compat]",
                                self.name, key, list(reshape), list(weight.shape),
                            )
                            return weight
                        try:
                            lora_diff = _torch_lora.mm(
                                v[0].flatten(start_dim=1), v[1].flatten(start_dim=1)
                            )
                            if lora_diff.numel() != weight.numel():
                                logger.warning(
                                    "LoRA %s: skip %s (numel %d != %d) [HSWQ compat]",
                                    self.name, key, lora_diff.numel(), weight.numel(),
                                )
                                return weight
                        except Exception:
                            return weight
                        return _orig_cw(
                            self, weight=weight, key=key, strength=strength,
                            strength_model=strength_model, offset=offset,
                            function=function, intermediate_dtype=intermediate_dtype,
                            original_weight=original_weight,
                        )

                    _lora_skip_calculate_weight._hswq_lora_skip_patched = True
                    _LoraDiff.calculate_weight = _lora_skip_calculate_weight
                    _applied.append("LoraDiff.calculate_weight")
        except Exception as e:
            logger.debug("LoraDiff patch failed: %s", e)

    if _applied:
        logger.info("z_image/FP8/torch.compile compat patches applied: %s", ", ".join(_applied))
    else:
        logger.debug("No torch.compile compat patches applied")
except Exception as e:
    logger.debug("Torch compile compat patches not applied: %s", e)

# Check if _patch_model method exists in NunchakuZImageTransformer2DModel
try:
    from nunchaku.models.transformers.transformer_zimage import NunchakuZImageTransformer2DModel
    if hasattr(NunchakuZImageTransformer2DModel, '_patch_model'):
        logger.info("NunchakuZImageTransformer2DModel._patch_model method found - patch may not be required")
    else:
        logger.warning("NunchakuZImageTransformer2DModel._patch_model method NOT found - patch is required")

    # ---------------------------------------------------------------------
    # ComfyUI ModelPatcher integration:
    # - ComfyUI passes ControlNet patches via transformer_options["patches"]["double_block"]
    # - Nunchaku Z-Image forward does not natively call those patches.
    # This monkey-patch wraps each transformer block and calls double_block patches
    # after each block, matching ComfyUI's NextDiT/QwenImage patching behavior.
    # ---------------------------------------------------------------------
    if not getattr(NunchakuZImageTransformer2DModel, "_comfyui_mp_patched", False):
        import inspect
        from comfy.ldm.flux.layers import EmbedND

        _orig_forward = NunchakuZImageTransformer2DModel.forward

        def _apply_double_block_patches(
            parent: "NunchakuZImageTransformer2DModel",
            block_index: int,
            unified_in,
            unified_out,
            adaln_input,
        ):
            """
            Apply ComfyUI ModelPatcher's double_block patches after a transformer block.
            IMPORTANT: do NOT change module hierarchy (LoRA matching depends on `layers.N.*` paths).
            """
            transformer_options = getattr(parent, "_comfyui_transformer_options", None)
            if not isinstance(transformer_options, dict):
                return unified_out

            patches = transformer_options.get("patches", {})
            if not isinstance(patches, dict):
                return unified_out

            double_block_patches = patches.get("double_block", [])
            if not double_block_patches:
                return unified_out

            # ComfyUI conventions
            transformer_options["block_index"] = block_index
            transformer_options.setdefault("block_type", "double")

            original_x_list = getattr(parent, "_comfyui_original_x_list", None)
            cap_len = getattr(parent, "_comfyui_cap_len", None)
            if not isinstance(cap_len, int) or cap_len < 0:
                cap_feats_list = getattr(parent, "_comfyui_cap_feats", None)
                if isinstance(cap_feats_list, list) and len(cap_feats_list) > 0 and hasattr(cap_feats_list[0], "shape"):
                    cap_len = int(cap_feats_list[0].shape[0])
                else:
                    cap_len = 0

            img_len = getattr(parent, "_comfyui_img_len", None)
            if not isinstance(img_len, int) or img_len < 0:
                img_len = 0

            unified = unified_out
            for p in double_block_patches:
                # Z-Image-Turbo uses List[torch.Tensor] for x, but ZImageControlPatch expects (B,C,H,W) tensor
                patch_x = original_x_list
                if isinstance(original_x_list, list) and len(original_x_list) > 0:
                    patch_x = torch.stack(original_x_list, dim=0)  # (B, C, F, H, W)
                    if patch_x.shape[2] == 1:
                        patch_x = patch_x.squeeze(2)  # (B, C, H, W)

                patch_in = {"x": patch_x, "block_index": block_index, "transformer_options": transformer_options}

                if unified is not None and hasattr(unified, "shape"):
                    # diffusers Z-Image order: [img_tokens, txt_tokens]
                    patch_in["img"] = unified[:, :img_len]
                    patch_in["txt"] = unified[:, img_len:img_len + cap_len]
                else:
                    patch_in["img"] = unified
                    patch_in["txt"] = None

                if unified_in is not None and hasattr(unified_in, "shape"):
                    patch_in["img_input"] = unified_in[:, :img_len]
                else:
                    patch_in["img_input"] = None

                # Build ComfyUI-compatible RoPE freqs for image tokens (pe) using EmbedND (Lumina style)
                pe_cached = getattr(parent, "_comfyui_pe_img", None)
                pe_key = getattr(parent, "_comfyui_pe_key", None)
                rope_options = transformer_options.get("rope_options", None) if isinstance(transformer_options, dict) else None
                h_scale = 1.0
                w_scale = 1.0
                h_start = 0.0
                w_start = 0.0
                if isinstance(rope_options, dict):
                    try:
                        h_scale = float(rope_options.get("scale_y", 1.0))
                        w_scale = float(rope_options.get("scale_x", 1.0))
                        h_start = float(rope_options.get("shift_y", 0.0))
                        w_start = float(rope_options.get("shift_x", 0.0))
                    except Exception:
                        h_scale = 1.0
                        w_scale = 1.0
                        h_start = 0.0
                        w_start = 0.0

                want_key = (img_len, cap_len, getattr(patch_x, "shape", None), h_scale, w_scale, h_start, w_start, unified.device, unified.dtype)
                if pe_cached is None or pe_key != want_key:
                    # Default Z-Image settings from Nunchaku docs: rope_theta=256, axes_dims=[32,48,48] (sum=128)
                    rope_theta = 256
                    axes_dims = (32, 48, 48)
                    head_dim = sum(axes_dims)
                    rope_embedder = getattr(parent, "_comfyui_rope_embedder", None)
                    if rope_embedder is None or getattr(rope_embedder, "theta", None) != rope_theta:
                        rope_embedder = EmbedND(dim=head_dim, theta=rope_theta, axes_dim=list(axes_dims))
                        parent._comfyui_rope_embedder = rope_embedder

                    b = int(patch_x.shape[0]) if hasattr(patch_x, "shape") else 1
                    ids = torch.zeros((b, img_len, 3), dtype=torch.float32, device=unified.device)

                    x_list = getattr(parent, "_comfyui_original_x_list", None)
                    h_tokens = 0
                    w_tokens = 0
                    try:
                        if isinstance(x_list, list) and len(x_list) > 0 and hasattr(x_list[0], "shape"):
                            _, f, h, w = x_list[0].shape
                            patch_size = int(getattr(parent, "_comfyui_patch_size", 2))
                            h_tokens = h // patch_size
                            w_tokens = w // patch_size
                    except Exception:
                        h_tokens = 0
                        w_tokens = 0

                    if h_tokens > 0 and w_tokens > 0:
                        n = min(img_len, h_tokens * w_tokens)
                        cap_offset = float(cap_len + 1)
                        ids[:, :n, 0] = cap_offset
                        ys = (torch.arange(h_tokens, dtype=torch.float32, device=unified.device) * h_scale + h_start).view(-1, 1).repeat(1, w_tokens).flatten()
                        xs = (torch.arange(w_tokens, dtype=torch.float32, device=unified.device) * w_scale + w_start).view(1, -1).repeat(h_tokens, 1).flatten()
                        ids[:, :n, 1] = ys[:n]
                        ids[:, :n, 2] = xs[:n]

                    pe_img = rope_embedder(ids).movedim(1, 2)  # (B, seq, 1, head_dim/2, 2, 2)
                    pe_img = pe_img.to(dtype=unified.dtype).contiguous()
                    parent._comfyui_pe_img = pe_img
                    parent._comfyui_pe_key = want_key
                    pe_cached = pe_img

                patch_in["pe"] = pe_cached
                patch_in["vec"] = adaln_input
                patch_in["block_type"] = transformer_options.get("block_type", "double")

                patch_out = p(patch_in)
                if isinstance(patch_out, dict) and unified is not None and hasattr(unified, "shape"):
                    if "img" in patch_out:
                        unified[:, :img_len] = patch_out["img"]
                    if "txt" in patch_out:
                        unified[:, img_len:img_len + cap_len] = patch_out["txt"]

            return unified

        def _ensure_layers_patched_in_place(parent: "NunchakuZImageTransformer2DModel"):
            if getattr(parent, "_comfyui_layers_patched_in_place", False):
                return
            try:
                layers = getattr(parent, "layers", None)
            except Exception:
                layers = None
            if layers is None:
                return
            try:
                for idx, layer in enumerate(layers):
                    if getattr(layer, "_comfyui_double_block_patched", False):
                        continue
                    orig_layer_forward = layer.forward

                    def _layer_forward_patched(*args, __orig=orig_layer_forward, __idx=idx, __parent=parent, **kwargs):
                        unified_in = args[0] if len(args) > 0 else None
                        adaln_input = args[3] if len(args) > 3 else None
                        out = __orig(*args, **kwargs)
                        return _apply_double_block_patches(__parent, __idx, unified_in, out, adaln_input)

                    layer.forward = _layer_forward_patched
                    layer._comfyui_double_block_patched = True
                parent._comfyui_layers_patched_in_place = True
                logger.info("[ZImageTurbo] Patched transformer block forwards (in-place) for double_block ControlNet patches")
            except Exception as e:
                logger.exception(f"[ZImageTurbo] Failed to patch layer forwards in-place: {e}")

        def _patched_forward(self, x, t, cap_feats=None, *args, control=None, transformer_options=None, **kwargs):
            # Store for block wrappers
            if isinstance(transformer_options, dict):
                self._comfyui_transformer_options = transformer_options
            else:
                self._comfyui_transformer_options = {}
            self._comfyui_original_x_list = x
            self._comfyui_cap_feats = cap_feats
            try:
                if isinstance(cap_feats, list) and len(cap_feats) > 0 and hasattr(cap_feats[0], "shape"):
                    self._comfyui_cap_len = int(cap_feats[0].shape[0])
            except Exception:
                self._comfyui_cap_len = 0

            # Z-Image (diffusers) unified token order is: img tokens first, then txt tokens.
            # Compute img token length from the input x list shape and patch sizes (default: 2 / 1).
            try:
                patch_size = int(kwargs.get("patch_size", 2))
            except Exception:
                patch_size = 2
            try:
                f_patch_size = int(kwargs.get("f_patch_size", 1))
            except Exception:
                f_patch_size = 1

            try:
                if isinstance(x, list) and len(x) > 0 and hasattr(x[0], "shape"):
                    # x element shape: (C, F, H, W)
                    _, f, h, w = x[0].shape
                    img_tokens = (h // patch_size) * (w // patch_size) * (f // f_patch_size)
                    # SEQ_MULTI_OF in diffusers ZImage is 32
                    img_tokens = int(((img_tokens + 31) // 32) * 32)
                    self._comfyui_img_len = img_tokens
                else:
                    self._comfyui_img_len = 0
            except Exception:
                self._comfyui_img_len = 0

            _ensure_layers_patched_in_place(self)

            # Log once per forward if patches exist (avoid spam)
            try:
                patches = self._comfyui_transformer_options.get("patches", {})
                dbl = patches.get("double_block", []) if isinstance(patches, dict) else []
                if dbl and not getattr(self, "_comfyui_logged_double_block", False):
                    logger.info(f"[ZImageTurbo] double_block patches detected: {len(dbl)} (will apply after each block)")
                    self._comfyui_logged_double_block = True
            except Exception:
                pass

            # Call original forward with only supported kwargs
            try:
                sig = inspect.signature(_orig_forward)
                allowed = set(sig.parameters.keys())
                call_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
            except Exception:
                call_kwargs = kwargs
            return _orig_forward(self, x, t, cap_feats=cap_feats, **call_kwargs)

        NunchakuZImageTransformer2DModel.forward = _patched_forward
        NunchakuZImageTransformer2DModel._comfyui_mp_patched = True
        logger.info("[ZImageTurbo] Installed ModelPatcher-compatible forward() patch")
except ImportError as e:
    logger.warning(f"Could not import NunchakuZImageTransformer2DModel to check _patch_model: {e}")

nunchaku_full_version = get_package_version("nunchaku").split("+")[0].strip()

logger.info(f"Nunchaku version: {nunchaku_full_version}")
logger.info(f"ComfyUI-nunchaku version: {get_plugin_version()}")


min_nunchaku_version = "1.0.0"
nunchaku_version = nunchaku_full_version.split("+")[0].strip()
nunchaku_major_minor_patch_version = ".".join(nunchaku_version.split(".")[:3])

try:
    if Version(nunchaku_major_minor_patch_version) < Version(min_nunchaku_version):
        logger.warning(
            f"ComfyUI-nunchaku {get_plugin_version()} requires nunchaku >= v{min_nunchaku_version}, "
            f"but found nunchaku {nunchaku_full_version}. Please update nunchaku."
        )
except InvalidVersion:
    logger.warning(
        f"Could not parse nunchaku version: {nunchaku_full_version}. "
        f"Please ensure you have at least v{min_nunchaku_version}."
    )

NODE_CLASS_MAPPINGS = {}

# SDXL Integrated Loader / DualCLIP loaders removed (owner order).
# Nunchaku Ultimate SD Upscale remains.

try:
    from .nodes.nunchaku_usdu import (
        HSWQUltimateSDUpscale,
    )
    NODE_CLASS_MAPPINGS["HSWQUltimateSDUpscale"] = HSWQUltimateSDUpscale
    logger.info("HSWQ Ultimate SD Upscale nodes registered successfully")
except Exception as e:
    logger.error(f"Failed to register HSWQ Ultimate SD Upscale nodes: {e}", exc_info=True)

try:
    from .nodes.hswq_save_image import HSWQSaveImage

    NODE_CLASS_MAPPINGS["HSWQSaveImage"] = HSWQSaveImage
    logger.info("HSWQ Save Image node registered successfully")
except Exception as e:
    logger.error(f"Failed to register HSWQ Save Image node: {e}", exc_info=True)

# SDXL MultiGPU node registration (UNET + CLIP, ref CheckpointLoaderSimple)
try:
    import folder_paths
    import comfy.sd
    from nodes import NODE_CLASS_MAPPINGS as GLOBAL_NODE_CLASS_MAPPINGS
    from .device_utils import get_device_list

    _UNETLoaderBase = GLOBAL_NODE_CLASS_MAPPINGS.get("UNETLoader")
    if _UNETLoaderBase is None:
        sdxl_logger.warning("[SDXL] UNETLoader not found in GLOBAL_NODE_CLASS_MAPPINGS")
    else:

        class HSWQCheckpointLoaderSDXL(_UNETLoaderBase):
            """HSWQ Checkpoint Loader (SDXL) with device selection. Ref: CheckpointLoaderSimple."""

            @classmethod
            def INPUT_TYPES(cls):
                base = _UNETLoaderBase.INPUT_TYPES()
                base_req = dict(base.get("required", {}))
                base_opt = dict(base.get("optional", {}))
                devices = get_device_list()
                default_dev = devices[1] if len(devices) > 1 else devices[0]
                req = {
                    "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "SDXL checkpoint to load MODEL and CLIP from (same as standard Load Checkpoint)."}),
                    "weight_dtype": ([
                        "default",
                        "fp8_e4m3fn",
                        "fp8_e4m3fn_fast",
                        "fp8_e5m2",
                        "int8_tensorwise",
                        # SDXL only — never reuse Z Image / Krea "Z Image ConvRot NVFP4".
                        "ConvRot NVFP4",
                    ],),
                }
                opt = {"device": (devices, {"default": default_dev})}
                return {"required": req, "optional": opt}

            RETURN_TYPES = ("MODEL", "CLIP")
            OUTPUT_TOOLTIPS = ("The UNet diffusion model from checkpoint.", "The CLIP model from the SDXL checkpoint.")
            FUNCTION = "load_checkpoint"
            CATEGORY = "loaders"
            TITLE = "HSWQ Checkpoint Loader (SDXL)"

            def load_checkpoint(self, ckpt_name, weight_dtype, device=None):
                original_device = get_current_device()
                if device is not None:
                    set_current_device(device)
                try:
                    ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
                    model_options = {}
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

        NODE_CLASS_MAPPINGS["HSWQCheckpointLoaderSDXL"] = HSWQCheckpointLoaderSDXL
        sdxl_logger.info("[SDXL] Registered HSWQCheckpointLoaderSDXL node (MODEL + CLIP)")

        # Z Image Turbo checkpoint loader removed (Nunchaku Z Image Turbo support dropped).
except Exception as e:
    sdxl_logger.exception(f"[SDXL] Failed to register HSWQCheckpointLoaderSDXL: {e}")

# Z Image FP8 E4M3 UNet Loader (DiT Loader excluded from init)
try:
    from .hswq.zimage_fp8_e4m3_unet import HSWQFP8E4M3UNetLoader
    NODE_CLASS_MAPPINGS["HSWQFP8E4M3UNetLoader"] = HSWQFP8E4M3UNetLoader
    logger.info("Registered HSWQ FP8 E4M3 UNet Loader (ComfyUI standard UNet loader wrapper)")
except (ImportError, ModuleNotFoundError) as e:
    logger.debug("HSWQ FP8 E4M3 UNet Loader not registered: %s", e)

try:
    from .patches.comfy_quant_int8 import install_int8_option_dispatch
    install_int8_option_dispatch(NODE_CLASS_MAPPINGS)
except Exception as e:
    logger.exception("[HSWQ INT8] install_int8_option_dispatch: %s", e)

# After INT8 dispatch: NVFP4 ckpts also have int8 Conv markers; NVFP4 must win first.
try:
    from .nodes.nvfp4.comfy_quant_nvfp4 import install_nvfp4_option_dispatch
    install_nvfp4_option_dispatch(NODE_CLASS_MAPPINGS)
except Exception as e:
    logger.exception("[HSWQ NVFP4] install_nvfp4_option_dispatch: %s", e)

# SDXL LoRA Stack V3 (INT8 / NVFP4 / standard via load_lora_for_models bake path)
try:
    from .nodes.lora.sdxl_v3 import GENERATED_NODES as _SDXL_LORA_V3_NODES
    NODE_CLASS_MAPPINGS.update(_SDXL_LORA_V3_NODES)
    logger.info("Registered HSWQSDXLLoraStackV3")
except Exception as e:
    logger.exception("[SDXL LoRA] HSWQSDXLLoraStackV3: %s", e)

# HSWQ Batched Detailer (SEGS) - phase-split version to minimize model switching
try:
    from .nodes.hswq_batched_detailer import HSWQBatchedDetailer
    NODE_CLASS_MAPPINGS["HSWQBatchedDetailer"] = HSWQBatchedDetailer
    logger.info("Registered HSWQ Batched Detailer (SEGS) (phase-split model-switch optimization)")
except (ImportError, ModuleNotFoundError) as e:
    logger.debug("HSWQ Batched Detailer not registered: %s", e)

try:
    from .nodes.hswq_sampler import HSWQSampler
    NODE_CLASS_MAPPINGS["HSWQSampler"] = HSWQSampler
    logger.info("Registered HSWQ Sampler")
except (ImportError, ModuleNotFoundError) as e:
    logger.debug("HSWQ Sampler not registered: %s", e)

try:
    from .nodes.hswq_vae_decode_tiled import HSWQVAEDecodeTiled
    NODE_CLASS_MAPPINGS["HSWQVAEDecodeTiled"] = HSWQVAEDecodeTiled
    logger.info("Registered HSWQ VAE Decode Tiled")
except (ImportError, ModuleNotFoundError) as e:
    logger.debug("HSWQ VAE Decode Tiled not registered: %s", e)

try:
    from .nodes.hswq_torch_compile import HSWQTorchCompileModel
    NODE_CLASS_MAPPINGS["HSWQTorchCompileModel"] = HSWQTorchCompileModel
    logger.info("Registered HSWQ Torch Compile")
except (ImportError, ModuleNotFoundError) as e:
    logger.debug("HSWQ Torch Compile not registered: %s", e)

NODE_DISPLAY_NAME_MAPPINGS = {k: getattr(v, "TITLE", k) for k, v in NODE_CLASS_MAPPINGS.items()}
# Explicit override — TITLE alone is not enough for ComfyUI graph node headers.
NODE_DISPLAY_NAME_MAPPINGS["HSWQCheckpointLoaderSDXL"] = "HSWQ Checkpoint Loader (SDXL)"
WEB_DIRECTORY = "js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
logger.info("=" * (80 + len(" ComfyUI-nunchaku Initialization ")))
