"""Convert float diffusion weights to ComfyUI-native int8_tensorwise (optional ConvRot).

ComfyUI MixedPrecisionOps + comfy_kitchen read this layout:

  <layer>.weight           int8
  <layer>.weight_scale     float32
      plain INT8:          scalar (tensorwise)
      ConvRot Linear:      [out, 1] (row-wise) — kitchen online act rotate
      ConvRot Conv2d:      [out, 1, 1, 1] (per-out-channel)
  <layer>.comfy_quant      uint8 JSON (compact)
      plain:  {"format":"int8_tensorwise"}
      ConvRot:{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":N}

Full ConvRot accuracy path (--convrot):
  Linear 2D:  W_rot = W @ H^T, row-wise INT8, stamp (kitchen int8_linear online).
  Conv2d 4D:  rotate along in_channels, channelwise INT8, stamp.
              Online activation rotate is done by HSWQ INT8 Conv2d patches
              (kitchen has no int8_conv; Params.convrot dequant is 2D-only).
  If in_features / in_channels is not divisible by a power-of-4 group size,
  that layer stays plain tensorwise (cannot apply Hadamard).

Default --scope diffusion only touches UNet/DiT keys. Never quantizes CLIP / VAE.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

_DEFAULT_GROUPSIZE = 256
_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}
# GPU-side cache: avoids CPU→GPU transfer on every rotate_activation call.
# Keyed by (size, device_str, dtype) – same as CPU cache but on target device.
_HADAMARD_GPU_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}

_SKIP_PREFIXES = (
    "conditioner.",
    "first_stage_model.",
    "text_encoders.",
    "clip_g.",
    "clip_l.",
    "vae.",
)


def build_hadamard(
    size: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Normalized regular Hadamard (power-of-4), same as comfy_kitchen ConvRot."""
    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]

    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")

    h4 = torch.tensor(
        [
            [1, 1, 1, -1],
            [1, 1, -1, 1],
            [1, -1, 1, 1],
            [-1, 1, 1, 1],
        ],
        dtype=dtype,
        device=device,
    )
    h_matrix = h4
    current_size = 4
    while current_size < size:
        h_matrix = torch.kron(h_matrix, h4)
        current_size *= 4
    h_matrix = h_matrix / (size**0.5)
    _HADAMARD_CACHE[cache_key] = h_matrix
    return h_matrix


def get_hadamard_on_device(
    size: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return Hadamard matrix on target device, with GPU-side caching.

    Builds on CPU (via build_hadamard) once, then transfers to the target
    device and caches there. Subsequent calls with the same
    (size, device, dtype) hit the GPU cache and skip CPU->GPU transfer.
    """
    cache_key = (size, str(device), dtype)
    cached = _HADAMARD_GPU_CACHE.get(cache_key)
    if cached is not None:
        return cached
    h = build_hadamard(size, device="cpu", dtype=torch.float32)
    h = h.to(dtype=dtype, device=device)
    _HADAMARD_GPU_CACHE[cache_key] = h
    return h

def convrot_group_size_for_features(n: int, preferred: int = _DEFAULT_GROUPSIZE) -> int | None:
    """Largest power-of-4 group size <= preferred that divides n (or None)."""
    if n < 4:
        return None
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def rotate_weight(weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int) -> torch.Tensor:
    """Offline Linear: W_rot = W @ H^T (group-wise). Matches comfy_kitchen._rotate_weight."""
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features {in_features} not divisible by group_size {group_size}"
        )
    group_count = in_features // group_size
    weight_grouped = weight.view(out_features, group_count, group_size)
    return torch.matmul(
        weight_grouped, h_matrix.T.to(dtype=weight.dtype, device=weight.device)
    ).reshape(weight.shape)


def unrotate_weight(weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int) -> torch.Tensor:
    """Inverse of rotate_weight: W = W_rot @ H (H orthogonal ⇒ H^{-1}=H^T)."""
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features {in_features} not divisible by group_size {group_size}"
        )
    group_count = in_features // group_size
    weight_grouped = weight.view(out_features, group_count, group_size)
    return torch.matmul(
        weight_grouped, h_matrix.to(dtype=weight.dtype, device=weight.device)
    ).reshape(weight.shape)


def rotate_weight_conv2d(
    weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Offline Conv2d: rotate along in_channels. weight (O, I, kH, kW)."""
    if weight.ndim != 4:
        raise ValueError(f"Conv2d weight must be 4D, got ndim={weight.ndim}")
    out_c, in_c, k_h, k_w = weight.shape
    flat = weight.permute(0, 2, 3, 1).contiguous().view(-1, in_c)
    flat_rot = rotate_weight(flat, h_matrix, group_size)
    return flat_rot.view(out_c, k_h, k_w, in_c).permute(0, 3, 1, 2).contiguous()


def unrotate_weight_conv2d(
    weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Inverse of rotate_weight_conv2d (for LoRA float space)."""
    if weight.ndim != 4:
        raise ValueError(f"Conv2d weight must be 4D, got ndim={weight.ndim}")
    out_c, in_c, k_h, k_w = weight.shape
    flat = weight.permute(0, 2, 3, 1).contiguous().view(-1, in_c)
    flat_un = unrotate_weight(flat, h_matrix, group_size)
    return flat_un.view(out_c, k_h, k_w, in_c).permute(0, 3, 1, 2).contiguous()


def rotate_activation(
    x: torch.Tensor, h_matrix: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Online Linear: x_rot = x @ H (last dim = features)."""
    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} not divisible by group_size {group_size}")
    group_count = features // group_size
    x_grouped = x.reshape(-1, group_count, group_size)
    # GPU-cached Hadamard: build/transfer once, reuse on every call
    h = get_hadamard_on_device(group_size, device=x.device, dtype=x.dtype)
    return torch.matmul(x_grouped, h).reshape(orig_shape)


def rotate_activation_nchw(
    x: torch.Tensor, h_matrix: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Online Conv2d: rotate channel dim of NCHW activation."""
    if x.ndim != 4:
        raise ValueError(f"NCHW activation must be 4D, got ndim={x.ndim}")
    # (N, C, H, W) → (N, H, W, C) → rotate last → back
    x_perm = x.permute(0, 2, 3, 1).contiguous()
    x_rot = rotate_activation(x_perm, h_matrix, group_size)
    return x_rot.permute(0, 3, 1, 2).contiguous()


def quantize_int8_tensorwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Tensorwise INT8: scalar weight_scale (plain ComfyUI int8_tensorwise)."""
    amax = max(w.abs().max().item(), 1e-6)
    scale = torch.tensor(amax / 127.0, dtype=torch.float32)
    q = (w / scale.item()).round().clamp(-127, 127).to(torch.int8)
    return q, scale


def quantize_int8_rowwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel INT8 for Linear: weight_scale [out, 1]."""
    abs_max = w.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
    scale = abs_max / 127.0
    q = (w / scale.to(dtype=w.dtype)).round().clamp(-127, 127).to(torch.int8)
    return q, scale.to(dtype=torch.float32)


def quantize_int8_channelwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-out-channel INT8 for Linear/Conv2d (broadcast-safe scale shape)."""
    reduce_dims = tuple(range(1, w.dim()))
    amax = torch.clamp(w.abs().amax(dim=reduce_dims).reshape(-1), min=1e-6)
    scale = amax / 127.0
    if w.dim() == 4:
        scale_view = scale.view(-1, 1, 1, 1)
    elif w.dim() == 2:
        scale_view = scale.view(-1, 1)
    else:
        raise ValueError(f"unsupported weight ndim={w.dim()} for channelwise INT8")
    q = (w / scale_view.to(dtype=w.dtype)).round().clamp(-127, 127).to(torch.int8)
    return q, scale_view.to(dtype=torch.float32)


def _encode_comfy_quant(config: dict) -> torch.Tensor:
    """Compact JSON bytes — same encoding as ComfyUI / convert_to_comfy.py."""
    return torch.tensor(
        list(json.dumps(config, separators=(",", ":")).encode("utf-8")),
        dtype=torch.uint8,
    )


def _is_float_matmul_weight(key: str, tensor: torch.Tensor) -> bool:
    if not key.endswith(".weight"):
        return False
    if tensor.ndim < 2:
        return False
    return tensor.dtype in (torch.float16, torch.float32, torch.bfloat16)


def _in_diffusion_scope(key: str) -> bool:
    """True for SDXL UNet / standalone DiT keys; False for CLIP/VAE."""
    if key.startswith(_SKIP_PREFIXES):
        return False
    if key.startswith("model.diffusion_model.") or key.startswith("diffusion_model."):
        return True
    # Standalone UNet/DiT safetensors (no model. / conditioner. prefixes).
    if not key.startswith("model.") and not any(key.startswith(p) for p in _SKIP_PREFIXES):
        return True
    return False


def convert_to_int8(
    input_path: str,
    output_path: str,
    group_size: int = _DEFAULT_GROUPSIZE,
    enable_convrot: bool = False,
    scope: str = "diffusion",
) -> None:
    print(f"Loading model: {input_path}")
    state_dict = load_file(input_path)

    new_state_dict: dict = {}
    quant_meta_layers: dict = {}
    convrot_linear = 0
    convrot_conv2d = 0
    plain_int8_count = 0
    skipped_count = 0
    out_of_scope = 0
    conv2d_plain = 0

    mode = "ConvRot FULL (Linear + Conv2d)" if enable_convrot else "plain INT8 (no ConvRot)"
    print(f"Converting matmul weights to INT8 [{mode}], scope={scope}, groupsize={group_size}")
    if enable_convrot:
        print(
            "  Linear: W@H^T + row-wise [out,1] + stamp (kitchen online act rotate)"
        )
        print(
            "  Conv2d: rotate in_channels + channelwise [out,1,1,1] + stamp "
            "(HSWQ Conv2d online act rotate)"
        )

    for key, tensor in tqdm(state_dict.items()):
        if scope == "diffusion" and not _in_diffusion_scope(key):
            new_state_dict[key] = tensor
            out_of_scope += 1
            continue

        if not _is_float_matmul_weight(key, tensor):
            new_state_dict[key] = tensor
            skipped_count += 1
            continue

        w = tensor.float()
        module_key = key[: -len(".weight")]
        quant_config: dict
        q: torch.Tensor
        scale: torch.Tensor

        if enable_convrot and tensor.ndim == 2:
            gs = convrot_group_size_for_features(w.shape[1], group_size)
            if gs is not None:
                h_matrix = build_hadamard(gs, device="cpu", dtype=torch.float32)
                w = rotate_weight(w, h_matrix, gs)
                q, scale = quantize_int8_rowwise(w)
                quant_config = {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": int(gs),
                }
                convrot_linear += 1
            else:
                q, scale = quantize_int8_tensorwise(w)
                quant_config = {"format": "int8_tensorwise"}
                plain_int8_count += 1
        elif enable_convrot and tensor.ndim == 4:
            gs = convrot_group_size_for_features(w.shape[1], group_size)
            if gs is not None:
                h_matrix = build_hadamard(gs, device="cpu", dtype=torch.float32)
                w = rotate_weight_conv2d(w, h_matrix, gs)
                q, scale = quantize_int8_channelwise(w)
                quant_config = {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": int(gs),
                }
                convrot_conv2d += 1
            else:
                q, scale = quantize_int8_tensorwise(w)
                quant_config = {"format": "int8_tensorwise"}
                plain_int8_count += 1
                conv2d_plain += 1
        else:
            q, scale = quantize_int8_tensorwise(w)
            quant_config = {"format": "int8_tensorwise"}
            plain_int8_count += 1
            if tensor.ndim == 4:
                conv2d_plain += 1

        new_state_dict[key] = q
        new_state_dict[f"{module_key}.weight_scale"] = scale
        new_state_dict[f"{module_key}.comfy_quant"] = _encode_comfy_quant(quant_config)
        quant_meta_layers[module_key] = dict(quant_config)

    metadata = {
        "_quantization_metadata": json.dumps(
            {"format_version": "1.0", "layers": quant_meta_layers},
            separators=(",", ":"),
        )
    }

    print(f"Saving to: {output_path}")
    print(
        f"ConvRot Linear: {convrot_linear}, ConvRot Conv2d: {convrot_conv2d}, "
        f"plain INT8: {plain_int8_count} (Conv2d plain: {conv2d_plain}), "
        f"kept_non_matmul: {skipped_count}, left_float_out_of_scope: {out_of_scope}"
    )
    save_file(new_state_dict, output_path, metadata=metadata)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Float SDXL/DiT → ComfyUI int8_tensorwise (+ optional ConvRot)"
    )
    parser.add_argument("--model", type=str, required=True, help="Input .safetensors")
    parser.add_argument("--output", type=str, required=True, help="Output .safetensors")
    parser.add_argument(
        "--groupsize",
        type=int,
        default=_DEFAULT_GROUPSIZE,
        help=f"Preferred ConvRot Hadamard group size (power of 4, default {_DEFAULT_GROUPSIZE})",
    )
    parser.add_argument(
        "--convrot",
        action="store_true",
        help=(
            "Full ConvRot accuracy: Linear + Conv2d when in_dim divisible by a "
            "power-of-4 group size — rotate + per-channel scale + comfy_quant.convrot. "
            "Linear uses kitchen online rotate; Conv2d needs HSWQ INT8 Conv2d patches."
        ),
    )
    parser.add_argument(
        "--scope",
        type=str,
        choices=("diffusion", "all"),
        default="diffusion",
        help="diffusion=UNet/DiT only (default); all=also CLIP/VAE (dangerous)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        raise SystemExit(1)
    if args.groupsize < 4 or (args.groupsize & (args.groupsize - 1)) != 0:
        print(f"Error: --groupsize must be a power of 4 (>=4), got {args.groupsize}")
        raise SystemExit(1)
    if math.log(args.groupsize, 4) % 1 != 0:
        print(f"Error: --groupsize must be a power of 4, got {args.groupsize}")
        raise SystemExit(1)

    convert_to_int8(
        args.model,
        args.output,
        group_size=args.groupsize,
        enable_convrot=args.convrot,
        scope=args.scope,
    )
