"""HSWQ Model Patch Loader (port of ``ModelPatchLoaderCustom``).

Ported from ``ComfyUI-NunchakuFluxLoraStacker/nodes/misc_v2.py``
(``ModelPatchLoaderCustom``, display name "Model Patch Loader").

Adds over the stock ``comfy_extras.nodes_model_patch.ModelPatchLoader``:

1. ``cpu_offload`` boolean: build the model graph + ``ModelPatcher`` on CPU
   (load_device / offload_device / model_device all CPU) so the model patch
   never occupies VRAM.
2. ConvRot INT8 support: when the checkpoint carries ``int8_tensorwise``
   ``comfy_quant`` layers, the module graph is built with
   ``mixed_precision_ops`` so ``<layer>.comfy_quant`` / ``<layer>.weight_scale``
   are consumed and weights stay INT8 in memory (TensorWiseINT8Layout,
   comfy-kitchen ``int8_linear`` kernel with online ConvRot rotation).

Supported model-patch architectures (same dispatch as the source):

- Qwen Image block-wise ControlNet (``controlnet_blocks.0.y_rms.weight``)
- SigLIP multi-feature projector (``feature_embedder.mid_layer_norm.bias``)
- Z-Image Fun ControlNet (``control_all_x_embedder.2-1.weight``)

The output is a normal ``MODEL_PATCH``; apply with the stock apply nodes
(``QwenImageDiffsynthControlnet`` / ``ZImageFunControlnet`` /
``USOStyleReference``).

Note: the ``cpu_offload`` option is honored end-to-end only for the Z-Image
Fun ControlNet path, via ``patches/model_patch_cpu_offload.py`` (ports the
source's ``_model_patch_cpu_offload_apply``), which keeps the stock
``ZImageControlPatch`` running on CPU when the patch was loaded with
``load_device=cpu``.
"""
from __future__ import annotations

import json
import contextlib
import logging

import torch
from torch import nn

import comfy.ldm.common_dit
import comfy.latent_formats
import comfy.ldm.lumina.controlnet
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils
from comfy.quant_ops import QUANT_ALGOS

from .utils import get_filename_list, get_full_path_or_raise

logger = logging.getLogger(__name__)

_MP_DIR = "model_patches"


class BlockWiseControlBlock(torch.nn.Module):
    # [linear, gelu, linear]
    def __init__(self, dim: int = 3072, device=None, dtype=None, operations=None):
        super().__init__()
        self.x_rms = operations.RMSNorm(dim, eps=1e-6)
        self.y_rms = operations.RMSNorm(dim, eps=1e-6)
        self.input_proj = operations.Linear(dim, dim)
        self.act = torch.nn.GELU()
        self.output_proj = operations.Linear(dim, dim)

    def forward(self, x, y):
        x, y = self.x_rms(x), self.y_rms(y)
        x = self.input_proj(x + y)
        x = self.act(x)
        x = self.output_proj(x)
        return x


class QwenImageBlockWiseControlNet(torch.nn.Module):
    def __init__(
        self,
        num_layers: int = 60,
        in_dim: int = 64,
        additional_in_dim: int = 0,
        dim: int = 3072,
        device=None, dtype=None, operations=None
    ):
        super().__init__()
        self.additional_in_dim = additional_in_dim
        self.img_in = operations.Linear(in_dim + additional_in_dim, dim, device=device, dtype=dtype)
        self.controlnet_blocks = torch.nn.ModuleList(
            [
                BlockWiseControlBlock(dim, device=device, dtype=dtype, operations=operations)
                for _ in range(num_layers)
            ]
        )

    def process_input_latent_image(self, latent_image):
        latent_image[:, :16] = comfy.latent_formats.Wan21().process_in(latent_image[:, :16])
        patch_size = 2
        hidden_states = comfy.ldm.common_dit.pad_to_patch_size(latent_image, (1, patch_size, patch_size))
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(orig_shape[0], orig_shape[1], orig_shape[-2] // 2, 2, orig_shape[-1] // 2, 2)
        hidden_states = hidden_states.permute(0, 2, 4, 1, 3, 5)
        hidden_states = hidden_states.reshape(orig_shape[0], (orig_shape[-2] // 2) * (orig_shape[-1] // 2), orig_shape[1] * 4)
        return self.img_in(hidden_states)

    def control_block(self, img, controlnet_conditioning, block_id):
        return self.controlnet_blocks[block_id](img, controlnet_conditioning)


class SigLIPMultiFeatProjModel(torch.nn.Module):
    """SigLIP Multi-Feature Projection Model for processing style features from
    different layers and projecting them into a unified hidden space."""

    def __init__(
        self,
        siglip_token_nums: int = 729,
        style_token_nums: int = 64,
        siglip_token_dims: int = 1152,
        hidden_size: int = 3072,
        context_layer_norm: bool = True,
        device=None, dtype=None, operations=None
    ):
        super().__init__()

        # High-level feature processing (layer -2)
        self.high_embedding_linear = nn.Sequential(
            operations.Linear(siglip_token_nums, style_token_nums),
            nn.SiLU()
        )
        self.high_layer_norm = (
            operations.LayerNorm(siglip_token_dims) if context_layer_norm else nn.Identity()
        )
        self.high_projection = operations.Linear(siglip_token_dims, hidden_size, bias=True)

        # Mid-level feature processing (layer -11)
        self.mid_embedding_linear = nn.Sequential(
            operations.Linear(siglip_token_nums, style_token_nums),
            nn.SiLU()
        )
        self.mid_layer_norm = (
            operations.LayerNorm(siglip_token_dims) if context_layer_norm else nn.Identity()
        )
        self.mid_projection = operations.Linear(siglip_token_dims, hidden_size, bias=True)

        # Low-level feature processing (layer -20)
        self.low_embedding_linear = nn.Sequential(
            operations.Linear(siglip_token_nums, style_token_nums),
            nn.SiLU()
        )
        self.low_layer_norm = (
            operations.LayerNorm(siglip_token_dims) if context_layer_norm else nn.Identity()
        )
        self.low_projection = operations.Linear(siglip_token_dims, hidden_size, bias=True)

    def forward(self, siglip_outputs):
        dtype = next(self.high_embedding_linear.parameters()).dtype

        high_embedding = self._process_layer_features(
            siglip_outputs[2],
            self.high_embedding_linear,
            self.high_layer_norm,
            self.high_projection,
            dtype
        )
        mid_embedding = self._process_layer_features(
            siglip_outputs[1],
            self.mid_embedding_linear,
            self.mid_layer_norm,
            self.mid_projection,
            dtype
        )
        low_embedding = self._process_layer_features(
            siglip_outputs[0],
            self.low_embedding_linear,
            self.low_layer_norm,
            self.low_projection,
            dtype
        )
        return torch.cat((high_embedding, mid_embedding, low_embedding), dim=1)

    def _process_layer_features(
        self,
        hidden_states: torch.Tensor,
        embedding_linear: nn.Module,
        layer_norm: nn.Module,
        projection: nn.Module,
        dtype: torch.dtype
    ) -> torch.Tensor:
        embedding = embedding_linear(
            hidden_states.to(dtype).transpose(1, 2)
        ).transpose(1, 2)
        embedding = layer_norm(embedding)
        embedding = projection(embedding)
        return embedding


def z_image_convert(sd):
    replace_keys = {".attention.to_out.0.bias": ".attention.out.bias",
                    ".attention.norm_k.weight": ".attention.k_norm.weight",
                    ".attention.norm_q.weight": ".attention.q_norm.weight",
                    ".attention.to_out.0.weight": ".attention.out.weight"
                    }

    out_sd = {}
    for k in sorted(sd.keys()):
        w = sd[k]

        k_out = k
        if k_out.endswith(".attention.to_k.weight"):
            cc = [w]
            continue
        if k_out.endswith(".attention.to_q.weight"):
            cc = [w] + cc
            continue
        if k_out.endswith(".attention.to_v.weight"):
            cc = cc + [w]
            w = torch.cat(cc, dim=0)
            k_out = k_out.replace(".attention.to_v.weight", ".attention.qkv.weight")

        for r, rr in replace_keys.items():
            k_out = k_out.replace(r, rr)
        out_sd[k_out] = w

    return out_sd


def _decode_comfy_quant(raw) -> dict:
    try:
        return json.loads(raw.numpy().tobytes())
    except Exception:  # noqa: BLE001
        return {}


def _has_int8_comfy_quant(sd) -> bool:
    """True if the checkpoint carries >=1 int8_tensorwise comfy_quant layer (ConvRot INT8)."""
    for key in sd.keys():
        if not key.endswith(".comfy_quant"):
            continue
        conf = _decode_comfy_quant(sd[key])
        if conf.get("format") == "int8_tensorwise":
            return True
    return False


def _get_default_compute_dtype(device: torch.device | None = None) -> torch.dtype:
    """Select BF16 on modern GPUs (Ampere/Ada/Blackwell) or FP16 on Turing/older GPUs."""
    if device is None:
        try:
            device = comfy.model_management.get_torch_device()
        except Exception:  # noqa: BLE001
            device = None
    if comfy.model_management.should_use_bf16(device=device):
        return torch.bfloat16
    return torch.float16


def _int8_mixed_precision_ops(compute_dtype: torch.dtype | None = None):
    """MixedPrecisionOps supporting int8_tensorwise (ConvRot included).

    Same approach as the HSWQ ControlNet loader: build the module graph in a
    float dtype (BF16 on Ampere+, FP16 on Turing/older) and let
    MixedPrecisionOps.Linear._load_from_state_dict consume
    "<layer>.comfy_quant" / "<layer>.weight_scale", attaching an INT8
    QuantizedTensor (TensorWiseINT8Layout) to every quantized Linear.
    """
    if compute_dtype is None:
        compute_dtype = _get_default_compute_dtype()
    quant_config = {
        "int8_tensorwise": QUANT_ALGOS["int8_tensorwise"],
    }
    return comfy.ops.mixed_precision_ops(
        quant_config,
        compute_dtype,
        full_precision_mm=False,
        disabled=[],
    )


@contextlib.contextmanager
def _silence_lazy_op_state_dict_warnings():
    """Suppress the benign core warning while introspecting shapes of lazy layers.

    MixedPrecisionOps.Linear carries no ``weight`` attribute until
    ``_load_from_state_dict`` runs. Calling ``model.state_dict()`` on such a
    freshly built controlnet therefore hits the informational guard in
    comfy/ops.py ``_quantized_weight_state_dict`` ("state dict on uninitialized
    op ...") once per lazy layer. That storm is noise for a shape-introspection
    pass: every reported layer is about to be initialized by the subsequent
    load_state_dict. The filter matches exactly this one message, lives only
    for the introspection call, and any real warning still passes through.
    """
    class _OnlyLazyOpNoiseFilter(logging.Filter):
        def filter(self, record):  # noqa: D102
            try:
                return "state dict on uninitialized op" not in record.getMessage()
            except Exception:  # noqa: BLE001
                return True

    root = logging.getLogger()
    flt = _OnlyLazyOpNoiseFilter()
    root.addFilter(flt)
    try:
        yield
    finally:
        root.removeFilter(flt)


class HSWQModelPatchLoaderCustom:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "name": (get_filename_list(_MP_DIR),),
            "cpu_offload": ("BOOLEAN", {"default": True, "tooltip": "Load model to CPU (main memory). Does not use VRAM."}),
        }}

    RETURN_TYPES = ("MODEL_PATCH",)
    FUNCTION = "load_model_patch"
    CATEGORY = "loaders"
    TITLE = "HSWQ Model Patch Loader (ConvRot INT8 / CPU offload)"

    def load_model_patch(self, name, cpu_offload):
        model_patch_path = get_full_path_or_raise(_MP_DIR, name)
        sd = comfy.utils.load_torch_file(model_patch_path, safe_load=True)
        dtype = comfy.utils.weight_dtype(sd)

        if cpu_offload:
            load_device = torch.device("cpu")
            offload_device = torch.device("cpu")
            model_device = torch.device("cpu")
        else:
            load_device = comfy.model_management.get_torch_device()
            offload_device = comfy.model_management.unet_offload_device()
            model_device = comfy.model_management.unet_offload_device()

        int8_checkpoint = _has_int8_comfy_quant(sd)

        if int8_checkpoint:
            try:
                from ..patches.comfy_quant_int8 import (
                    _patch_comfy_kitchen_rms_rope_fallback,
                    _patch_comfy_kitchen_int8_gemm_fallback,
                )
                _patch_comfy_kitchen_rms_rope_fallback()
                _patch_comfy_kitchen_int8_gemm_fallback()
            except Exception:
                pass
            dtype = _get_default_compute_dtype()
            operations = _int8_mixed_precision_ops(compute_dtype=dtype)
            logger.info(
                "[HSWQ ModelPatch] INT8 ComfyQuant detected in '%s': loading with "
                "MixedPrecisionOps (weights stay INT8 / ConvRot)", name,
            )
        else:
            operations = comfy.ops.manual_cast

        if 'controlnet_blocks.0.y_rms.weight' in sd:
            additional_in_dim = sd["img_in.weight"].shape[1] - 64
            model = QwenImageBlockWiseControlNet(additional_in_dim=additional_in_dim, device=model_device, dtype=dtype, operations=operations)
        elif 'feature_embedder.mid_layer_norm.bias' in sd:
            sd = comfy.utils.state_dict_prefix_replace(sd, {"feature_embedder.": ""}, filter_keys=True)
            model = SigLIPMultiFeatProjModel(device=model_device, dtype=dtype, operations=operations)
        elif 'control_all_x_embedder.2-1.weight' in sd:  # alipai z image fun controlnet
            if not int8_checkpoint:
                sd = z_image_convert(sd)
            config = {}
            # Check for 2.0 or 2.1 by counting control_layers
            n_control_layers = 0
            for k in sd.keys():
                if k.startswith('control_layers.') and '.adaLN_modulation.0.weight' in k:
                    layer_idx = int(k.split('.')[1])
                    n_control_layers = max(n_control_layers, layer_idx + 1)

            # Fallback to 2.0 detection if dynamic count fails
            if n_control_layers == 0 and 'control_layers.14.adaLN_modulation.0.weight' in sd:
                n_control_layers = 15

            if n_control_layers > 0:
                config['n_control_layers'] = n_control_layers
                config['additional_in_dim'] = 17
                config['refiner_control'] = True
                ref_weight = sd.get("control_noise_refiner.0.after_proj.weight", None)
                if ref_weight is not None:
                    if torch.count_nonzero(ref_weight) == 0:
                        config['broken'] = True

            # Infer control_in_dim from checkpoint so embedder input dim matches
            # (avoids matmul shape error).
            embedder_in = sd["control_all_x_embedder.2-1.weight"].shape[1]
            expected_channels = embedder_in // 4  # f_patch_size * patch_size * patch_size = 4
            if 'additional_in_dim' in config:
                config['control_in_dim'] = expected_channels - config['additional_in_dim']
            else:
                config['control_in_dim'] = expected_channels

            model = comfy.ldm.lumina.controlnet.ZImage_Control(device=model_device, dtype=dtype, operations=operations, **config)

            # Filter only size mismatches; keep keys that are in checkpoint but
            # not in model.state_dict() (e.g. lazy-init Linear layers that only
            # appear once load_state_dict / _load_from_state_dict runs).
            with _silence_lazy_op_state_dict_warnings():
                model_state_dict = model.state_dict()
            filtered_sd = {}
            size_mismatch_keys = []
            keys_not_in_model = []

            for k, v in sd.items():
                if k in model_state_dict:
                    if v.shape == model_state_dict[k].shape:
                        filtered_sd[k] = v
                    else:
                        size_mismatch_keys.append(f"{k}: checkpoint shape {v.shape} vs model shape {model_state_dict[k].shape}")
                else:
                    filtered_sd[k] = v
                    keys_not_in_model.append(k)

            if keys_not_in_model:
                logger.info("[HSWQ ModelPatch] Info: %d keys loaded via state_dict (e.g. lazy-init layers)", len(keys_not_in_model))
            if size_mismatch_keys:
                logger.warning("[HSWQ ModelPatch] Warning: %d keys have size mismatches (excluded)", len(size_mismatch_keys))
                for key_info in size_mismatch_keys[:5]:
                    logger.warning("  - %s", key_info)
                if len(size_mismatch_keys) > 5:
                    logger.warning("  ... and %d more", len(size_mismatch_keys) - 5)

            sd = filtered_sd
        else:
            raise ValueError(
                f"[HSWQ ModelPatch] {name}: could not detect a known model patch "
                "architecture (no supported discriminator key found)."
            )

        model.load_state_dict(sd, strict=False)
        model = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)
        return (model,)


NODE_CLASS_MAPPINGS = {
    "HSWQModelPatchLoaderCustom": HSWQModelPatchLoaderCustom,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HSWQModelPatchLoaderCustom": "HSWQ Model Patch Loader (ConvRot INT8 / CPU offload)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
