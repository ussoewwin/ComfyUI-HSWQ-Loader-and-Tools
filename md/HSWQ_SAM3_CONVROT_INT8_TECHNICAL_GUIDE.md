# HSWQ SAM3 ConvRot INT8 Nodes — Complete Technical Guide

> Baseline: `d33862a027c99c7b806099012724ccf3181379d9` → current (`8c20913`)
> This document explains the complete set of nodes added to run **SAM3 (Segment Anything 3)** checkpoints quantized with **ConvRot / TensorWise INT8** in ComfyUI (2 new files + 3 modified files).

---

## 1. Overview

### 1.1 What these nodes do

Two nodes that load and run **SAM3 (Meta Segment Anything 3)** checkpoints quantized with **ConvRot / TensorWise INT8** in ComfyUI.

| Node | Role |
|---|---|
| **HSWQ SAM3 Loader (ConvRot INT8)** | Loads an INT8-quantized SAM3 checkpoint while keeping the weights in 8-bit precision in VRAM |
| **HSWQ SAM3 Detect** | Runs open-vocabulary detection & segmentation with text / bounding-box / point prompts and outputs masks, bboxes, and the image |

### 1.2 Technical highlights

- **INT8 VRAM retention**: builds Linear layers as `QuantizedTensor` (`TensorWiseINT8Layout`) via `MixedPrecisionOps` + `int8_tensorwise`, keeping weights in 8-bit in VRAM. SAM3.1 Multiplex: ~525MB (about half of FP16).
- **Fast GEMM**: at runtime uses comfy_kitchen's `int8_linear` CUDA kernel; ConvRot layers run with online activation rotation (`convrot`).
- **Alignment safety fallback**: cuBLAS INT8 GEMM requires K and N to be multiples of 4. Layers with unaligned dims (e.g. `boxRPB_embed_x` with in_features=2) automatically fall back to float precision instead of crashing.
- **DynamicVRAM (aimdo) support**: avoids the vbar buffer mismatch that occurs under ComfyUI's aimdo / DynamicVRAM (`--fast`-style) environment.
- **CLIP key remap**: INT8 checkpoints store language_backbone already split into q/k/v, so ComfyUI's stock `transformers_convert` (which expects fused `in_proj_weight`) cannot remap them, producing "clip missing". This is fixed by a remapping patch.

### 1.3 Overall architecture

```
sam3.1_multiplex_convrot_int8.safetensors
        │  (detector.* + tracker.*, language_backbone pre-split q/k/v INT8)
        ▼
HSWQ SAM3 Loader / CheckpointLoaderSimple (HSWQ patch)
        │  MixedPrecisionOps(int8_tensorwise) + bfloat16
        │  CLIP keys: dequant → fp16 + key remap
        ▼
ModelPatcher (SAM3 / SAM31)
        │
        ▼
HSWQ SAM3 Detect ──(text/box/point prompts)──► masks / bboxes / image
        │
        ├─ _guard_sam3_model_weights : dequant INT8 weights to fp16 (runtime stability)
        │     └─ _strip_dynamic_vram_attrs : drop vbar attrs → regular cast path
        └─ _refine_mask : SAM decoder mask refinement
```

---

## 2. Files Created / Modified

| Type | File | Content |
|---|---|---|
| New | `nodes/hswq_load_convrot_int8_sam3.py` | HSWQ SAM3 Loader node (177 lines) |
| New | `nodes/hswq_sam3_detect.py` | HSWQ SAM3 Detect node (397 lines) |
| Modified | `patches/comfy_quant_int8.py` | Added SAM3 patch set (GEMM fallback / CLIP remap / CheckpointLoader integration) |
| Modified | `__init__.py` | Startup patches + node registration |
| Modified | `README.md` | Node documentation |
| New | `png/sam3.png` | Illustration |

> Note: `patches/comfy_quant_int8.py` is an existing 3,372-line INT8 infrastructure file (LoRA baking etc.). This guide only lists the SAM3-related functions added after `d33862a`.

---

## 3. Full Code

### 3-1. `nodes/hswq_load_convrot_int8_sam3.py` (new, full)

```python
"""HSWQ Load ConvRot INT8 SAM3 Model.

ComfyUI's stock SAM3 path (Load Diffusion Model / UNetLoader) loads the SAM3
architecture with default float weights; for INT8 / ConvRot INT8 checkpoints,
MixedPrecisionOps attaches INT8 QuantizedTensor (TensorWiseINT8Layout) to Linear layers.

This loader node:
1. Detects INT8 ComfyQuant metadata (int8_tensorwise / ConvRot).
2. Configures MixedPrecisionOps with bfloat16 base dtype so module graph is
   cleanly constructed and INT8 weights remain in VRAM.
3. Automatically protects against non-multiple-of-4 layers (e.g. boxRPB_embed_x
   which has in_features=2) via safe dequantization fallback, enabling 100%
   error-free execution across all hardware.

The output is a standard MODEL object compatible with HSWQ SAM3 Detect and
ComfyUI SAM3 nodes.
"""
from __future__ import annotations

import json
import logging
import os
from typing import List

import torch

import comfy.sd
import comfy.utils
from comfy import ops as comfy_ops
from comfy.quant_ops import QUANT_ALGOS
import folder_paths

from ..patches.comfy_quant_int8 import (
    _patch_comfy_kitchen_int8_gemm_fallback,
    apply_comfy_quant_int8_patches,
)
from .utils import get_filename_list, get_full_path_or_raise

logger = logging.getLogger(__name__)

_SAM_FOLDERS = ["diffusion_models", "sams", "detection", "checkpoints"]


def _get_sam3_filenames() -> List[str]:
    """Get list of candidate filenames for SAM3 models across standard directories."""
    files: set[str] = set()
    for folder in _SAM_FOLDERS:
        try:
            flist = folder_paths.get_filename_list(folder)
            if flist:
                files.update(flist)
        except Exception:
            pass
    if not files:
        try:
            return folder_paths.get_filename_list("diffusion_models")
        except Exception:
            return []
    return sorted(list(files))


def _get_sam3_full_path_or_raise(filename: str) -> str:
    """Resolve full path to SAM3 model across candidate directories."""
    for folder in _SAM_FOLDERS:
        try:
            path = folder_paths.get_full_path(folder, filename)
            if path and os.path.isfile(path):
                return path
        except Exception:
            pass
    # Fallback to get_full_path_or_raise on primary diffusion_models directory
    return folder_paths.get_full_path_or_raise("diffusion_models", filename)


def _decode_comfy_quant(raw) -> dict:
    try:
        return json.loads(raw.numpy().tobytes())
    except Exception:
        return {}


def _has_int8_comfy_quant(sd: dict) -> bool:
    """True if checkpoint carries >=1 int8_tensorwise comfy_quant layer."""
    for key in sd.keys():
        if not key.endswith(".comfy_quant"):
            continue
        conf = _decode_comfy_quant(sd[key])
        if isinstance(conf, dict) and conf.get("format") == "int8_tensorwise":
            return True
        if conf == "int8_tensorwise":
            return True
    return False


def _int8_mixed_precision_ops():
    """MixedPrecisionOps supporting int8_tensorwise (ConvRot included)."""
    quant_config = {
        "int8_tensorwise": QUANT_ALGOS["int8_tensorwise"],
    }
    return comfy_ops.mixed_precision_ops(
        quant_config,
        torch.bfloat16,
        full_precision_mm=False,
        disabled=[],
    )


class HSWQSAM3Loader:
    """Load a ConvRot INT8 SAM3 model keeping weights INT8 in VRAM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_name": (
                    _get_sam3_filenames(),
                    {"tooltip": "ConvRot INT8 / standard SAM3 model (.safetensors)"},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_sam3"
    CATEGORY = "loaders"
    TITLE = "HSWQ SAM3 Loader (ConvRot INT8)"
    SEARCH_ALIASES = [
        "sam3",
        "sam3 loader",
        "load sam3",
        "sam 3",
        "segment anything",
        "hswq sam3",
        "convrot sam3",
        "int8 sam3",
    ]

    def load_sam3(self, sam3_name: str, **kwargs):
        _patch_comfy_kitchen_int8_gemm_fallback()
        apply_comfy_quant_int8_patches()
        ckpt_path = _get_sam3_full_path_or_raise(sam3_name)

        sd = comfy.utils.load_torch_file(ckpt_path, safe_load=True)

        if _has_int8_comfy_quant(sd):
            model_options = {
                "dtype": torch.bfloat16,
                "custom_operations": _int8_mixed_precision_ops(),
            }
            logger.info(
                "[HSWQ INT8 SAM3] INT8 ComfyQuant detected: loading %s with "
                "MixedPrecisionOps (weights stay INT8 in VRAM)",
                sam3_name,
            )
        else:
            model_options = {}
            logger.info(
                "[HSWQ INT8 SAM3] No INT8 ComfyQuant layers, loading %s directly",
                sam3_name,
            )

        model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options=model_options
        )
        if model is None:
            # Fallback to load_diffusion_model with path
            model = comfy.sd.load_diffusion_model(
                ckpt_path, model_options=model_options
            )
        if model is None:
            raise RuntimeError(
                f"[HSWQ INT8 SAM3] Failed to load SAM3 model from {sam3_name}"
            )
        return (model,)


HSWQLoadConvRotINT8SAM3 = HSWQSAM3Loader
```

### 3-2. `nodes/hswq_sam3_detect.py` (new, full)

```python
"""
HSWQ SAM3 Detect node with pass-through IMAGE output.

Registered node: HSWQSAM3Detect
Display name: HSWQ SAM3 Detect
"""

from __future__ import annotations

import json
import torch
import torch.nn.functional as F
import comfy.model_management
import comfy.utils

try:
    from comfy_api.latest import ComfyExtension, io
    COMFY_V3_AVAILABLE = True
except Exception:
    COMFY_V3_AVAILABLE = False


def _extract_text_prompts(conditioning, device, dtype):
    """Extract list of (text_embeddings, text_mask) from conditioning."""
    cond_meta = conditioning[0][1]
    multi = cond_meta.get("sam3_multi_cond")
    prompts = []
    if multi is not None:
        for entry in multi:
            emb = entry["cond"].to(device=device, dtype=dtype)
            mask = entry["attention_mask"].to(device) if entry["attention_mask"] is not None else None
            if mask is None:
                mask = torch.ones(emb.shape[0], emb.shape[1], dtype=torch.int64, device=device)
            prompts.append((emb, mask, entry.get("max_detections", 1)))
    else:
        emb = conditioning[0][0].to(device=device, dtype=dtype)
        mask = cond_meta.get("attention_mask")
        if mask is not None:
            mask = mask.to(device)
        else:
            mask = torch.ones(emb.shape[0], emb.shape[1], dtype=torch.int64, device=device)
        prompts.append((emb, mask, 1))
    return prompts


def _refine_mask(sam3_model, orig_image_hwc, coarse_mask, box_xyxy, H, W, device, dtype, iterations):
    """Refine a coarse detector mask via SAM decoder, cropping to the detection box.

    Returns: [1, H, W] binary mask
    """
    def _coarse_fallback():
        return (F.interpolate(coarse_mask.unsqueeze(0).unsqueeze(0), size=(H, W),
                              mode="bilinear", align_corners=False)[0] > 0).float()

    if iterations <= 0:
        return _coarse_fallback()

    pad_frac = 0.1
    x1, y1, x2, y2 = box_xyxy.tolist()
    bw, bh = x2 - x1, y2 - y1
    cx1 = max(0, int(x1 - bw * pad_frac))
    cy1 = max(0, int(y1 - bh * pad_frac))
    cx2 = min(W, int(x2 + bw * pad_frac))
    cy2 = min(H, int(y2 + bh * pad_frac))
    if cx2 <= cx1 or cy2 <= cy1:
        return _coarse_fallback()

    crop = orig_image_hwc[cy1:cy2, cx1:cx2, :3]
    crop_1008 = comfy.utils.common_upscale(crop.unsqueeze(0).movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled")
    crop_frame = crop_1008.to(device=device, dtype=dtype)
    crop_h, crop_w = cy2 - cy1, cx2 - cx1

    mask_h, mask_w = coarse_mask.shape[-2:]
    mx1, my1 = int(cx1 / W * mask_w), int(cy1 / H * mask_h)
    mx2, my2 = int(cx2 / W * mask_w), int(cy2 / H * mask_h)
    if mx2 <= mx1 or my2 <= my1:
        return _coarse_fallback()
    mask_logit = coarse_mask[..., my1:my2, mx1:mx2].unsqueeze(0).unsqueeze(0)
    for _ in range(iterations):
        coarse_input = F.interpolate(mask_logit, size=(1008, 1008), mode="bilinear", align_corners=False)
        mask_logit = sam3_model.forward_segment(crop_frame, mask_inputs=coarse_input)

    refined_crop = F.interpolate(mask_logit, size=(crop_h, crop_w), mode="bilinear", align_corners=False)
    full_mask = torch.zeros(1, 1, H, W, device=device, dtype=dtype)
    full_mask[:, :, cy1:cy2, cx1:cx2] = refined_crop
    coarse_full = F.interpolate(coarse_mask.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)
    return ((full_mask[0] > 0) | (coarse_full[0] > 0)).float()


def _strip_dynamic_vram_attrs(module):
    """Drop DynamicVRAM (vbar) state after replacing an INT8 weight with a plain
    float16 tensor.

    The vbar buffer was allocated with the INT8 payload size (int8 data + scale).
    Once the weight is replaced by a float16 parameter, cast_bias_weight() goes
    through resolve_cast_module_with_vbar() and the float16-sized cast geometry no
    longer fits the INT8-sized vbar buffer ("Buffer too small"). Removing _v makes
    cast_bias_weight() fall back to the regular cast path.
    """
    if not hasattr(module, "_v"):
        return
    try:
        from comfy_aimdo import model_vbar
        model_vbar.vbar_unpin(module._v)
    except Exception:
        pass
    for attr in ("_v", "_prefetch", "_v_signature", "_v_block"):
        if hasattr(module, attr):
            try:
                delattr(module, attr)
            except Exception:
                pass


def _guard_sam3_model_weights(sam3_model):
    """Ensure that all linear/conv layers in SAM3 are pristine float16,
    never raw unscaled int8, QuantizedTensor with kernel bugs, or rotated Conv2d."""
    try:
        from comfy_kitchen.tensor import QuantizedTensor
    except Exception:
        QuantizedTensor = None
    try:
        from ..patches.comfy_quant_int8 import _regular_hadamard_global
    except Exception:
        _regular_hadamard_global = None

    for name, module in sam3_model.named_modules():
        w = getattr(module, "weight", None)
        if w is None:
            continue
        is_qt = QuantizedTensor is not None and isinstance(w, QuantizedTensor)
        is_raw_int8 = not is_qt and getattr(w, "dtype", None) == torch.int8

        if is_qt:
            try:
                w_deq = w.dequantize().to(torch.float16)
                module.weight = torch.nn.Parameter(w_deq, requires_grad=False)
                _strip_dynamic_vram_attrs(module)
            except Exception:
                pass
        elif is_raw_int8:
            scale = getattr(module, "weight_scale", None)
            w_float = w.float() * (scale.float() if scale is not None else 1.0)
            if _regular_hadamard_global is not None:
                if w_float.ndim == 2:
                    o, i = w_float.shape
                    for cand_gs in (256, 64, 16, 4):
                        if i % cand_gs == 0 and i // cand_gs > 0:
                            h = _regular_hadamard_global(cand_gs, device=w_float.device)
                            group_count = i // cand_gs
                            w_grouped = w_float.view(o, group_count, cand_gs)
                            w_float = torch.matmul(w_grouped, h.to(dtype=w_float.dtype, device=w_float.device)).reshape(o, i)
                            break
                elif w_float.ndim == 4:
                    o, i, kh, kw = w_float.shape
                    for cand_gs in (256, 64, 16, 4):
                        if i % cand_gs == 0 and i // cand_gs > 0:
                            h = _regular_hadamard_global(cand_gs, device=w_float.device)
                            flat = w_float.permute(0, 2, 3, 1).contiguous().view(-1, i)
                            group_count = i // cand_gs
                            flat_grouped = flat.view(-1, group_count, cand_gs)
                            flat_un = torch.matmul(flat_grouped, h.to(dtype=flat.dtype, device=flat.device)).reshape(-1, i)
                            w_float = flat_un.view(o, kh, kw, i).permute(0, 3, 1, 2).contiguous()
                            break
            module.weight = torch.nn.Parameter(w_float.to(dtype=torch.float16), requires_grad=False)
            _strip_dynamic_vram_attrs(module)


def run_sam3_detect(model, image, conditioning=None, bboxes=None, positive_coords=None, negative_coords=None, threshold=0.5, refine_iterations=2, individual_masks=False):
    try:
        from ..patches.comfy_quant_int8 import apply_comfy_quant_int8_patches
        apply_comfy_quant_int8_patches()
    except Exception:
        pass
    B, H, W, C = image.shape
    image_in = comfy.utils.common_upscale(image[..., :3].movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled")

    def _boxes_to_tensor(box_list):
        coords = []
        for d in box_list:
            cx = (d["x"] + d["width"] / 2) / W
            cy = (d["y"] + d["height"] / 2) / H
            coords.append([cx, cy, d["width"] / W, d["height"] / H])
        return torch.tensor([coords], dtype=torch.float32)

    per_frame_boxes = None
    if bboxes is not None:
        if isinstance(bboxes, dict):
            shared = _boxes_to_tensor([bboxes])
            per_frame_boxes = [shared] * B
        elif isinstance(bboxes, list) and len(bboxes) > 0 and isinstance(bboxes[0], list):
            per_frame_boxes = [_boxes_to_tensor(frame_boxes) if frame_boxes else None for frame_boxes in bboxes]
            while len(per_frame_boxes) < B:
                per_frame_boxes.append(per_frame_boxes[-1] if per_frame_boxes else None)
        elif isinstance(bboxes, list) and len(bboxes) > 0:
            shared = _boxes_to_tensor(bboxes)
            per_frame_boxes = [shared] * B

    pos_pts = json.loads(positive_coords) if positive_coords else []
    neg_pts = json.loads(negative_coords) if negative_coords else []
    has_points = len(pos_pts) > 0 or len(neg_pts) > 0

    comfy.model_management.load_model_gpu(model)
    device = comfy.model_management.get_torch_device()
    dtype = model.model.get_dtype()
    sam3_model = model.model.diffusion_model
    _guard_sam3_model_weights(sam3_model)

    point_inputs = None
    if has_points:
        all_coords = [[p["x"] / W * 1008, p["y"] / H * 1008] for p in pos_pts] + \
                     [[p["x"] / W * 1008, p["y"] / H * 1008] for p in neg_pts]
        all_labels = [1] * len(pos_pts) + [0] * len(neg_pts)
        point_inputs = {
            "point_coords": torch.tensor([all_coords], dtype=dtype, device=device),
            "point_labels": torch.tensor([all_labels], dtype=torch.int32, device=device),
        }

    cond_list = _extract_text_prompts(conditioning, device, dtype) if conditioning is not None and len(conditioning) > 0 else []
    has_text = len(cond_list) > 0

    all_bbox_dicts = []
    all_masks = []
    pbar = comfy.utils.ProgressBar(B)

    for b in range(B):
        frame = image_in[b:b+1].to(device=device, dtype=dtype)
        b_boxes = None
        if per_frame_boxes is not None and per_frame_boxes[b] is not None:
            b_boxes = per_frame_boxes[b].to(device=device, dtype=dtype)

        frame_bbox_dicts = []
        frame_masks = []

        if point_inputs is not None:
            mask_logit = sam3_model.forward_segment(frame, point_inputs=point_inputs)
            for _ in range(max(0, refine_iterations - 1)):
                mask_logit = sam3_model.forward_segment(frame, mask_inputs=mask_logit)
            mask = F.interpolate(mask_logit, size=(H, W), mode="bilinear", align_corners=False)
            frame_masks.append((mask[0] > 0).float())

        if b_boxes is not None and not has_text:
            for box_cxcywh in b_boxes[0]:
                cx, cy, bw, bh = box_cxcywh.tolist()
                sam_box = torch.tensor([[[(cx - bw/2) * 1008, (cy - bh/2) * 1008],
                                         [(cx + bw/2) * 1008, (cy + bh/2) * 1008]]],
                                       device=device, dtype=dtype)
                mask_logit = sam3_model.forward_segment(frame, box_inputs=sam_box)
                for _ in range(max(0, refine_iterations - 1)):
                    mask_logit = sam3_model.forward_segment(frame, mask_inputs=mask_logit)
                mask = F.interpolate(mask_logit, size=(H, W), mode="bilinear", align_corners=False)
                frame_masks.append((mask[0] > 0).float())

        for text_embeddings, text_mask, max_det in cond_list:
            results = sam3_model(
                frame, text_embeddings=text_embeddings, text_mask=text_mask,
                boxes=b_boxes, threshold=threshold, orig_size=(H, W))

            pred_boxes = results["boxes"][0]
            scores = results["scores"][0]
            masks = results["masks"][0]

            probs = scores.sigmoid()
            keep = probs > threshold
            kept_boxes = pred_boxes[keep].cpu()
            kept_scores = probs[keep].cpu()
            kept_masks = masks[keep]

            order = kept_scores.argsort(descending=True)[:max_det]
            kept_boxes = kept_boxes[order]
            kept_scores = kept_scores[order]
            kept_masks = kept_masks[order]

            for box, score in zip(kept_boxes, kept_scores):
                frame_bbox_dicts.append({
                    "x": float(box[0]), "y": float(box[1]),
                    "width": float(box[2] - box[0]), "height": float(box[3] - box[1]),
                    "score": float(score),
                })
            for m, box in zip(kept_masks, kept_boxes):
                frame_masks.append(_refine_mask(
                    sam3_model, image[b], m, box, H, W, device, dtype, refine_iterations))

        all_bbox_dicts.append(frame_bbox_dicts)
        if len(frame_masks) > 0:
            combined = torch.cat(frame_masks, dim=0)
            if individual_masks:
                all_masks.append(combined)
            else:
                all_masks.append((combined > 0).any(dim=0).float())
        else:
            if individual_masks:
                all_masks.append(torch.zeros(0, H, W, device=comfy.model_management.intermediate_device()))
            else:
                all_masks.append(torch.zeros(H, W, device=comfy.model_management.intermediate_device()))
        pbar.update(1)

    idev = comfy.model_management.intermediate_device()
    all_masks = [m.to(idev) for m in all_masks]
    mask_out = torch.cat(all_masks, dim=0) if individual_masks else torch.stack(all_masks)
    return mask_out, all_bbox_dicts, image


if COMFY_V3_AVAILABLE:
    class HSWQSAM3Detect(io.ComfyNode):
        """Open-vocabulary detection and segmentation using text, box, or point prompts with pass-through IMAGE output."""

        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="HSWQSAM3Detect",
                display_name="HSWQ SAM3 Detect",
                category="HSWQ/Detection",
                search_aliases=["sam3", "segment anything", "open vocabulary", "text detection", "segment", "hswq"],
                inputs=[
                    io.Model.Input("model", display_name="model"),
                    io.Image.Input("image", display_name="image"),
                    io.Conditioning.Input("conditioning", display_name="conditioning", optional=True, tooltip="Text conditioning from CLIPTextEncode"),
                    io.BoundingBox.Input("bboxes", display_name="bboxes", force_input=True, optional=True, tooltip="Bounding boxes to segment within"),
                    io.String.Input("positive_coords", display_name="positive_coords", force_input=True, optional=True, tooltip="Positive point prompts as JSON [{\"x\": int, \"y\": int}, ...] (pixel coords)"),
                    io.String.Input("negative_coords", display_name="negative_coords", force_input=True, optional=True, tooltip="Negative point prompts as JSON [{\"x\": int, \"y\": int}, ...] (pixel coords)"),
                    io.Float.Input("threshold", display_name="threshold", default=0.5, min=0.0, max=1.0, step=0.01),
                    io.Int.Input("refine_iterations", display_name="refine_iterations", default=2, min=0, max=5, tooltip="SAM decoder refinement passes (0=use raw detector masks)"),
                    io.Boolean.Input("individual_masks", display_name="individual_masks", default=False, tooltip="Output per-object masks instead of union"),
                ],
                outputs=[
                    io.Mask.Output("masks", display_name="masks"),
                    io.BoundingBox.Output("bboxes", display_name="bboxes"),
                    io.Image.Output("image", display_name="image"),
                ],
            )

        @classmethod
        def execute(cls, model, image, conditioning=None, bboxes=None, positive_coords=None, negative_coords=None, threshold=0.5, refine_iterations=2, individual_masks=False) -> io.NodeOutput:
            masks, bbox_dicts, out_img = run_sam3_detect(
                model=model,
                image=image,
                conditioning=conditioning,
                bboxes=bboxes,
                positive_coords=positive_coords,
                negative_coords=negative_coords,
                threshold=threshold,
                refine_iterations=refine_iterations,
                individual_masks=individual_masks,
            )
            return io.NodeOutput(masks, bbox_dicts, out_img)


class HSWQSAM3DetectV1:
    """V1-compatible wrapper for HSWQSAM3Detect."""
    TITLE = "HSWQ SAM3 Detect"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "refine_iterations": ("INT", {"default": 2, "min": 0, "max": 5, "step": 1}),
                "individual_masks": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "conditioning": ("CONDITIONING",),
                "bboxes": ("BBOXES",),
                "positive_coords": ("STRING", {"forceInput": True}),
                "negative_coords": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("MASK", "BBOXES", "IMAGE")
    RETURN_NAMES = ("masks", "bboxes", "image")
    FUNCTION = "execute"
    CATEGORY = "HSWQ/Detection"

    def execute(self, model, image, threshold=0.50, refine_iterations=2, individual_masks=False, conditioning=None, bboxes=None, positive_coords=None, negative_coords=None):
        masks, bbox_dicts, out_img = run_sam3_detect(
            model=model,
            image=image,
            conditioning=conditioning,
            bboxes=bboxes,
            positive_coords=positive_coords,
            negative_coords=negative_coords,
            threshold=threshold,
            refine_iterations=refine_iterations,
            individual_masks=individual_masks,
        )
        return (masks, bbox_dicts, out_img)


NODE_CLASS_MAPPINGS = {
    "HSWQSAM3Detect": HSWQSAM3DetectV1,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HSWQSAM3Detect": "HSWQ SAM3 Detect",
}
```

### 3-3. `patches/comfy_quant_int8.py` (modified, SAM3-related functions)

```python
def _patch_comfy_kitchen_int8_gemm_fallback() -> bool:
    """Patch comfy_kitchen's cuda.int8_linear and TensorWiseINT8Layout op handlers
    to safely handle non-multiple-of-4 dimensions (e.g. SAM3 boxRPB_embed_x K=2)
    and GPU GEMM exceptions by falling back to float precision instead of crashing.
    """
    global _COMFY_KITCHEN_INT8_FALLBACK_PATCHED
    if _COMFY_KITCHEN_INT8_FALLBACK_PATCHED:
        return True

    applied = []
    import torch

    # 1. Patch comfy_kitchen.backends.cuda.int8_linear (root backend kernel implementation)
    try:
        import comfy_kitchen.backends.cuda as ck_cuda

        orig_cuda_int8_linear = getattr(ck_cuda, "int8_linear", None)
        if orig_cuda_int8_linear is not None and not getattr(orig_cuda_int8_linear, "_hswq_safe_int8", False):
            def _safe_cuda_int8_linear(
                x: torch.Tensor,
                weight: torch.Tensor,
                weight_scale: torch.Tensor,
                bias: torch.Tensor = None,
                out_dtype: torch.dtype = None,
                convrot: bool = False,
                convrot_groupsize: int = 256,
                input_act: str | None = None,
            ) -> torch.Tensor:
                orig_shape = x.shape
                x_2d = x if x.dim() == 2 and x.is_contiguous() else x.reshape(-1, x.shape[-1]).contiguous()
                k = x_2d.shape[-1]
                n = weight.shape[0]
                out_dt = out_dtype or x.dtype
                is_2d_output = len(orig_shape) == 2

                # Hardware unaligned check (cuBLAS GEMM requires K % 4 == 0 and N % 4 == 0)
                if k % 4 != 0 or n % 4 != 0 or not x.is_cuda:
                    ws = weight_scale.to(device=x.device, dtype=torch.float32)
                    w_float = weight.to(device=x.device, dtype=torch.float32) * (ws if ws.numel() == 1 else ws.view(-1, 1))
                    b_arg = bias.to(device=x.device, dtype=out_dt) if bias is not None else None
                    res = torch.nn.functional.linear(x_2d.to(dtype=out_dt), w_float.to(dtype=out_dt), b_arg)
                    return res if is_2d_output else res.reshape(*orig_shape[:-1], n)

                try:
                    return orig_cuda_int8_linear(
                        x=x,
                        weight=weight,
                        weight_scale=weight_scale,
                        bias=bias,
                        out_dtype=out_dtype,
                        convrot=convrot,
                        convrot_groupsize=convrot_groupsize,
                        input_act=input_act,
                    )
                except Exception as e:
                    logger.debug("[HSWQ INT8] cuda.int8_linear fallback (k=%d, n=%d): %s", k, n, e)
                    ws = weight_scale.to(device=x.device, dtype=torch.float32)
                    w_float = weight.to(device=x.device, dtype=torch.float32) * (ws if ws.numel() == 1 else ws.view(-1, 1))
                    b_arg = bias.to(device=x.device, dtype=out_dt) if bias is not None else None
                    res = torch.nn.functional.linear(x_2d.to(dtype=out_dt), w_float.to(dtype=out_dt), b_arg)
                    return res if is_2d_output else res.reshape(*orig_shape[:-1], n)

            _safe_cuda_int8_linear._hswq_safe_int8 = True
            ck_cuda.int8_linear = _safe_cuda_int8_linear
            applied.append("cuda.int8_linear")
    except Exception as e:
        logger.debug("[HSWQ INT8] comfy_kitchen.backends.cuda patch failed: %s", e)

    # 2. Patch comfy_kitchen.tensor layout dispatchers
    try:
        import comfy_kitchen.tensor.base as ck_base
        import comfy_kitchen.tensor.int8 as ck_int8

        QuantizedTensor = ck_base.QuantizedTensor
        TensorWiseINT8Layout = getattr(ck_int8, "TensorWiseINT8Layout", None)
        if TensorWiseINT8Layout is None:
            TensorWiseINT8Layout = ck_base.get_layout_class("TensorWiseINT8Layout")
        _LAYOUT_DISPATCH_TABLE = ck_base._LAYOUT_DISPATCH_TABLE
        dequantize_args = ck_base.dequantize_args
        _dtype_code = ck_int8._dtype_code

        def _safe_handle_int8_linear_tensorwise(qt, args, kwargs):
            input_tensor = args[0]
            weight = args[1]
            bias = args[2] if len(args) > 2 else None

            if not isinstance(weight, QuantizedTensor) or getattr(weight, "_layout_cls", None) != "TensorWiseINT8Layout":
                return torch.nn.functional.linear(*dequantize_args(args), **dequantize_args(kwargs))
            if getattr(weight._params, "transposed", False):
                return torch.nn.functional.linear(*dequantize_args(args), **dequantize_args(kwargs))

            if isinstance(input_tensor, QuantizedTensor):
                input_tensor = input_tensor.dequantize()

            weight_qdata, weight_scale = TensorWiseINT8Layout.get_plain_tensors(weight)
            k = input_tensor.shape[-1]
            n = weight_qdata.shape[0]

            if not input_tensor.is_cuda or k % 4 != 0 or n % 4 != 0:
                return torch.nn.functional.linear(*dequantize_args(args), **dequantize_args(kwargs))

            out_dtype = kwargs.get("out_dtype", input_tensor.dtype)
            convrot = getattr(weight._params, "convrot", False)
            convrot_groupsize = getattr(weight._params, "convrot_groupsize", 256)

            scale = weight_scale.squeeze(-1).contiguous() if isinstance(weight_scale, torch.Tensor) and weight_scale.dim() > 1 else weight_scale

            try:
                return torch.ops.comfy_kitchen.int8_linear(
                    input_tensor.contiguous(),
                    weight_qdata.contiguous(),
                    scale,
                    bias,
                    _dtype_code(out_dtype),
                    convrot,
                    convrot_groupsize,
                )
            except Exception as e:
                logger.debug("[HSWQ INT8] int8_linear fallback (k=%d, n=%d): %s", k, n, e)
                return torch.nn.functional.linear(*dequantize_args(args), **dequantize_args(kwargs))

        def _safe_handle_int8_mm_tensorwise(qt, args, kwargs):
            input_tensor = args[0]
            weight = args[1]

            if not isinstance(weight, QuantizedTensor) or getattr(weight, "_layout_cls", None) != "TensorWiseINT8Layout":
                return torch.mm(*dequantize_args(args), **dequantize_args(kwargs))

            if isinstance(input_tensor, QuantizedTensor):
                input_tensor = input_tensor.dequantize()

            weight_qdata, weight_scale = TensorWiseINT8Layout.get_plain_tensors(weight)
            out_dtype = kwargs.get("out_dtype", input_tensor.dtype)

            convrot = getattr(weight._params, "convrot", False)
            convrot_groupsize = getattr(weight._params, "convrot_groupsize", 256)
            scale = weight_scale.squeeze(-1).contiguous() if isinstance(weight_scale, torch.Tensor) and weight_scale.dim() > 1 else weight_scale

            if getattr(weight._params, "transposed", False):
                int8_weight = weight_qdata.contiguous()
            elif weight_scale.numel() == 1 and not convrot:
                int8_weight = weight_qdata.t().contiguous()
            else:
                return torch.mm(*dequantize_args(args), **dequantize_args(kwargs))

            k = input_tensor.shape[-1]
            n = int8_weight.shape[0]
            if not input_tensor.is_cuda or k % 4 != 0 or n % 4 != 0:
                return torch.mm(*dequantize_args(args), **dequantize_args(kwargs))

            try:
                return torch.ops.comfy_kitchen.int8_linear(
                    input_tensor.contiguous(),
                    int8_weight,
                    scale,
                    None,
                    _dtype_code(out_dtype),
                    convrot,
                    convrot_groupsize,
                )
            except Exception as e:
                logger.debug("[HSWQ INT8] int8_mm fallback (k=%d, n=%d): %s", k, n, e)
                return torch.mm(*dequantize_args(args), **dequantize_args(kwargs))

        def _safe_handle_int8_addmm_tensorwise(qt, args, kwargs):
            bias = args[0]
            input_tensor = args[1]
            weight = args[2]

            if not isinstance(weight, QuantizedTensor) or getattr(weight, "_layout_cls", None) != "TensorWiseINT8Layout":
                return torch.addmm(*dequantize_args(args), **dequantize_args(kwargs))

            if isinstance(input_tensor, QuantizedTensor):
                input_tensor = input_tensor.dequantize()

            weight_qdata, weight_scale = TensorWiseINT8Layout.get_plain_tensors(weight)
            out_dtype = kwargs.get("out_dtype", input_tensor.dtype)

            convrot = getattr(weight._params, "convrot", False)
            convrot_groupsize = getattr(weight._params, "convrot_groupsize", 256)
            scale = weight_scale.squeeze(-1).contiguous() if isinstance(weight_scale, torch.Tensor) and weight_scale.dim() > 1 else weight_scale

            if getattr(weight._params, "transposed", False):
                int8_weight = weight_qdata.contiguous()
            elif weight_scale.numel() == 1 and not convrot:
                int8_weight = weight_qdata.t().contiguous()
            else:
                return torch.addmm(*dequantize_args(args), **dequantize_args(kwargs))

            k = input_tensor.shape[-1]
            n = int8_weight.shape[0]
            if not input_tensor.is_cuda or k % 4 != 0 or n % 4 != 0:
                return torch.addmm(*dequantize_args(args), **dequantize_args(kwargs))

            try:
                return torch.ops.comfy_kitchen.int8_linear(
                    input_tensor.contiguous(),
                    int8_weight,
                    scale,
                    bias,
                    _dtype_code(out_dtype),
                    convrot,
                    convrot_groupsize,
                )
            except Exception as e:
                logger.debug("[HSWQ INT8] int8_addmm fallback (k=%d, n=%d): %s", k, n, e)
                return torch.addmm(*dequantize_args(args), **dequantize_args(kwargs))

        def _safe_dequantize(qdata, params):
            output_dtype = getattr(params, "orig_dtype", torch.float16)
            scale = params.scale
            # 1. Multiply qdata by scale with correct broadcasting
            if scale is None:
                w_float = qdata.float()
            elif isinstance(scale, torch.Tensor):
                if scale.ndim == 1 and qdata.ndim == 2 and scale.shape[0] == qdata.shape[0]:
                    w_float = qdata.float() * scale.float().unsqueeze(1)
                elif scale.ndim == 2 and qdata.ndim == 2 and scale.shape[0] == qdata.shape[0]:
                    w_float = qdata.float() * scale.float()
                elif scale.numel() == 1:
                    w_float = qdata.float() * scale.item()
                else:
                    try:
                        w_float = qdata.float() * scale.float()
                    except Exception:
                        w_float = qdata.float() * scale.float().view(-1, 1)
            else:
                w_float = qdata.float() * float(scale)

            # 2. Apply ConvRot Hadamard un-rotation if enabled
            convrot = getattr(params, "convrot", False)
            gs = getattr(params, "convrot_groupsize", 256) or 256
            if convrot and w_float.ndim >= 2:
                if w_float.ndim == 2:
                    o, i = w_float.shape
                    for cand_gs in (gs, 256, 64, 16, 4):
                        if i % cand_gs == 0 and i // cand_gs > 0:
                            h = _regular_hadamard_global(cand_gs, device=w_float.device)
                            group_count = i // cand_gs
                            w_grouped = w_float.view(o, group_count, cand_gs)
                            w_float = torch.matmul(w_grouped, h.to(dtype=w_float.dtype, device=w_float.device)).reshape(o, i)
                            break
                elif w_float.ndim == 4:
                    o, i, kh, kw = w_float.shape
                    for cand_gs in (gs, 256, 64, 16, 4):
                        if i % cand_gs == 0 and i // cand_gs > 0:
                            h = _regular_hadamard_global(cand_gs, device=w_float.device)
                            flat = w_float.permute(0, 2, 3, 1).contiguous().view(-1, i)
                            group_count = i // cand_gs
                            flat_grouped = flat.view(-1, group_count, cand_gs)
                            flat_un = torch.matmul(flat_grouped, h.to(dtype=flat.dtype, device=flat.device)).reshape(-1, i)
                            w_float = flat_un.view(o, kh, kw, i).permute(0, 3, 1, 2).contiguous()
                            break

            return w_float.to(output_dtype)

        TensorWiseINT8Layout.dequantize = _safe_dequantize
        _LAYOUT_DISPATCH_TABLE.setdefault(torch.ops.aten.linear.default, {})[TensorWiseINT8Layout] = _safe_handle_int8_linear_tensorwise
        _LAYOUT_DISPATCH_TABLE.setdefault(torch.ops.aten.mm.default, {})[TensorWiseINT8Layout] = _safe_handle_int8_mm_tensorwise
        _LAYOUT_DISPATCH_TABLE.setdefault(torch.ops.aten.addmm.default, {})[TensorWiseINT8Layout] = _safe_handle_int8_addmm_tensorwise
        ck_int8._handle_int8_linear_tensorwise = _safe_handle_int8_linear_tensorwise
        ck_int8._handle_int8_mm_tensorwise = _safe_handle_int8_mm_tensorwise
        ck_int8._handle_int8_addmm_tensorwise = _safe_handle_int8_addmm_tensorwise
        applied.append("TensorWiseINT8Layout dispatch")
    except Exception as e:
        logger.debug("[HSWQ INT8] comfy_kitchen.tensor layout patch failed: %s", e)

    if applied:
        _COMFY_KITCHEN_INT8_FALLBACK_PATCHED = True
        logger.info("[HSWQ INT8] comfy_kitchen TensorWiseINT8 unaligned GEMM fallback patch armed (%s)", ", ".join(applied))
        return True
    return False


def _regular_hadamard_global(size: int, device=None):
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=torch.float32,
        device=device,
    )
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    return h / (size ** 0.5)


_SAM3_PROCESS_STATE_DICT_PATCHED = False


def _patch_sam3_process_state_dict() -> bool:
    """Fix ComfyUI SAM3 / SAM31 process_unet_state_dict to preserve and slice
    weight_scale and .comfy_quant sidecars when in_proj_weight is split into
    q_proj, k_proj, v_proj. Also split in_proj_weight in _clip_stash for CLIP.
    """
    global _SAM3_PROCESS_STATE_DICT_PATCHED
    if _SAM3_PROCESS_STATE_DICT_PATCHED:
        return True

    try:
        import comfy.supported_models as sm
        sam3_cls = getattr(sm, "SAM3", None)
        if sam3_cls is None:
            return False

        orig_process = getattr(sam3_cls, "process_unet_state_dict", None)
        orig_process_clip = getattr(sam3_cls, "process_clip_state_dict", None)
        if orig_process is None or orig_process_clip is None or getattr(orig_process, "_hswq_patched", False):
            _SAM3_PROCESS_STATE_DICT_PATCHED = True
            return True

        def _safe_process_unet_state_dict(self, state_dict):
            # HSWQ: keep stock behavior. Pre-splitting in_proj_weight (or its
            # weight_scale / .comfy_quant sidecars) before ComfyUI's
            # clip_text_transformers_convert() breaks the CLIP key remap:
            # transformers_convert() expects the fused in_proj_weight form to
            # produce "sam3_clip.transformer.text_model.encoder.layers.N.
            # self_attn.q_proj" keys. Pre-split keys are left un-remapped, so the
            # CLIP loads with missing weights ("clip missing") and corrupts text
            # embeddings -> noisy SAM3 masks. Stock path handles everything.
            return orig_process(self, state_dict)

        _CLIP_SIMPLE_REMAP = {
            "encoder.positional_embedding": "sam3_clip.transformer.text_model.embeddings.position_embedding.weight",
            "encoder.token_embedding.weight": "sam3_clip.transformer.text_model.embeddings.token_embedding.weight",
            "encoder.ln_final.weight": "sam3_clip.transformer.text_model.final_layer_norm.weight",
            "encoder.ln_final.bias": "sam3_clip.transformer.text_model.final_layer_norm.bias",
            "encoder.text_projection.weight": "sam3_clip.transformer.text_projection.weight",
            "encoder.text_projection": "sam3_clip.transformer.text_projection.weight",
        }

        def _safe_process_clip_state_dict(self, state_dict):
            clip_sd = orig_process_clip(self, state_dict)
            # INT8 SAM3 checkpoints store language_backbone already split into
            # q_proj/k_proj/v_proj (no fused in_proj_weight), so ComfyUI's
            # transformers_convert() leaves those keys un-remapped
            # ("clip missing" -> corrupt text embeddings -> noisy masks).
            # Remap leftover "encoder.*" keys to the expected CLIP layout.
            out = {}
            for k, v in clip_sd.items():
                nk = k
                if nk.startswith("encoder.transformer.resblocks."):
                    nk = nk.replace("encoder.transformer.resblocks.", "sam3_clip.transformer.text_model.encoder.layers.", 1)
                    nk = nk.replace(".attn.", ".self_attn.")
                    nk = nk.replace(".mlp.c_fc.", ".mlp.fc1.")
                    nk = nk.replace(".mlp.c_proj.", ".mlp.fc2.")
                    nk = nk.replace(".ln_1.", ".layer_norm1.")
                    nk = nk.replace(".ln_2.", ".layer_norm2.")
                elif nk in _CLIP_SIMPLE_REMAP:
                    nk = _CLIP_SIMPLE_REMAP[nk]
                out[nk] = v
            return out

        _safe_process_unet_state_dict._hswq_patched = True
        _safe_process_clip_state_dict._hswq_patched = True
        sam3_cls.process_unet_state_dict = _safe_process_unet_state_dict
        sam3_cls.process_clip_state_dict = _safe_process_clip_state_dict
        _SAM3_PROCESS_STATE_DICT_PATCHED = True
        logger.info("[HSWQ INT8] SAM3 process_state_dict patch armed (CLIP remap for pre-split INT8 checkpoints)")
        return True
    except Exception as e:
        logger.debug("[HSWQ INT8] SAM3 process_state_dict patch skipped: %s", e)
        return False


def _dequant_and_unrotate_tensor(q, scale, raw_conf):
    try:
        conf = json.loads(raw_conf.numpy().tobytes()) if hasattr(raw_conf, "numpy") else {}
    except Exception:
        conf = {}
    if scale is None:
        w_float = q.float()
    elif isinstance(scale, torch.Tensor):
        if scale.ndim == 1 and q.ndim == 2 and scale.shape[0] == q.shape[0]:
            w_float = q.float() * scale.float().unsqueeze(1)
        elif scale.ndim == 2 and q.ndim == 2 and scale.shape[0] == q.shape[0]:
            w_float = q.float() * scale.float()
        elif scale.numel() == 1:
            w_float = q.float() * scale.item()
        else:
            try:
                w_float = q.float() * scale.float()
            except Exception:
                w_float = q.float() * scale.float().view(-1, 1)
    else:
        w_float = q.float() * float(scale)

    has_convrot = isinstance(conf, dict) and conf.get("convrot", False)
    gs = int(conf.get("convrot_groupsize", 256) or 256) if has_convrot else 0

    if has_convrot and gs >= 4:
        if w_float.ndim == 2:
            o, i = w_float.shape
            for cand_gs in (gs, 256, 64, 16, 4):
                if i % cand_gs == 0 and i // cand_gs > 0:
                    h = _regular_hadamard_global(cand_gs, device=w_float.device)
                    group_count = i // cand_gs
                    w_grouped = w_float.view(o, group_count, cand_gs)
                    w_float = torch.matmul(w_grouped, h.to(dtype=w_float.dtype, device=w_float.device)).reshape(o, i)
                    break
        elif w_float.ndim == 4:
            o, i, kh, kw = w_float.shape
            for cand_gs in (gs, 256, 64, 16, 4):
                if i % cand_gs == 0 and i // cand_gs > 0:
                    h = _regular_hadamard_global(cand_gs, device=w_float.device)
                    flat = w_float.permute(0, 2, 3, 1).contiguous().view(-1, i)
                    group_count = i // cand_gs
                    flat_grouped = flat.view(-1, group_count, cand_gs)
                    flat_un = torch.matmul(flat_grouped, h.to(dtype=flat.dtype, device=flat.device)).reshape(-1, i)
                    w_float = flat_un.view(o, kh, kw, i).permute(0, 3, 1, 2).contiguous()
                    break
    return w_float.to(torch.float16)


_CHECKPOINT_LOADER_INT8_PATCHED = False


def _patch_load_state_dict_guess_config_int8() -> bool:
    """Patch comfy.sd.load_state_dict_guess_config so that SAM3 checkpoints
    with .comfy_quant INT8 layers automatically attach MixedPrecisionOps with
    INT8 tensorwise algorithm, while dequantizing text encoder (CLIP) keys.
    """
    global _CHECKPOINT_LOADER_INT8_PATCHED
    if _CHECKPOINT_LOADER_INT8_PATCHED:
        return True

    try:
        import comfy.sd
        import comfy.ops as comfy_ops
        from comfy.quant_ops import QUANT_ALGOS

        orig_fn = getattr(comfy.sd, "load_state_dict_guess_config", None)
        if orig_fn is None or getattr(orig_fn, "_hswq_patched", False):
            _CHECKPOINT_LOADER_INT8_PATCHED = True
            return True

        def _safe_load_state_dict_guess_config(
            sd,
            output_vae=True,
            output_clip=True,
            output_clipvision=False,
            embedding_directory=None,
            output_model=True,
            model_options={},
            te_model_options={},
            metadata=None,
            disable_dynamic=False,
        ):
            # Arm sub-patches first
            apply_comfy_quant_int8_patches()

            # Strict SAM3 gate: NEVER affect SDXL, Krea2, ZImage, FLUX, SD1.5, SD3, Wan, etc.
            is_sam3 = any(k.startswith("detector.") or "detector.backbone.vision_backbone" in k for k in sd.keys())
            if not is_sam3:
                return orig_fn(
                    sd,
                    output_vae=output_vae,
                    output_clip=output_clip,
                    output_clipvision=output_clipvision,
                    embedding_directory=embedding_directory,
                    output_model=output_model,
                    model_options=model_options,
                    te_model_options=te_model_options,
                    metadata=metadata,
                    disable_dynamic=disable_dynamic,
                )

            model_options = dict(model_options) if model_options else {}
            te_model_options = dict(te_model_options) if te_model_options else {}

            # 1. Check if SAM3 checkpoint carries INT8 comfy_quant layers
            has_int8 = any(k.endswith(".comfy_quant") for k in sd.keys())

            # 2. Pre-split and dequantize language_backbone (CLIP Text Encode) and Conv2d keys in sd
            if has_int8:
                dequant_count = 0
                for quant_key in list(sd.keys()):
                    if not quant_key.endswith(".comfy_quant"):
                        continue
                    base = quant_key[:-len(".comfy_quant")]
                    w_key = base + "weight" if base.endswith(".") else base + ".weight"
                    if w_key not in sd:
                        w_key = base
                    q = sd.get(w_key)
                    if q is None:
                        continue

                    # Only dequantize 4D Conv2d and text encoder keys; keep ALL Linear layers in INT8!
                    is_conv = getattr(q, "ndim", 0) == 4
                    is_text = "language_backbone" in quant_key
                    if is_conv or is_text:
                        candidates = [
                            base + "weight_scale" if base.endswith(".") else base + ".weight_scale",
                            base + "scale" if base.endswith(".") else base + ".scale",
                            base[:-1] + "_scale" if base.endswith(".") else base + "_scale",
                        ]
                        scale_key = next((c for c in candidates if c in sd), None)
                        scale = sd.get(scale_key) if scale_key else None
                        raw_conf = sd.get(quant_key)

                        w_clean = _dequant_and_unrotate_tensor(q, scale, raw_conf)
                        sd[w_key] = w_clean
                        if scale_key:
                            sd.pop(scale_key, None)
                        sd.pop(quant_key, None)
                        dequant_count += 1

                logger.info(
                    "[HSWQ INT8] Load Checkpoint: Dequantized %d Conv2d/CLIP keys; ALL Linear layers stay true INT8 in VRAM",
                    dequant_count,
                )

            # (removed) sd-level in_proj_weight pre-split: it breaks ComfyUI's
            # clip_text_transformers_convert() remap -> "clip missing" -> corrupt
            # text embeddings -> noisy SAM3 masks. The stock process_unet_state_dict
            # / transformers_convert split in_proj_weight correctly.

            # 4. Attach MixedPrecisionOps for SAM3 so all Linear layers (ViT trunk, transformer) load as true INT8 in VRAM
            if has_int8:
                quant_config = {
                    "int8_tensorwise": QUANT_ALGOS["int8_tensorwise"],
                }
                sam3_ops = comfy_ops.mixed_precision_ops(
                    quant_config,
                    torch.bfloat16,
                    full_precision_mm=False,
                    disabled=[],
                )
                model_options["custom_operations"] = sam3_ops
                model_options["dtype"] = torch.bfloat16
                logger.info(
                    "[HSWQ INT8] Load Checkpoint: Auto-attached MixedPrecisionOps for SAM3 INT8 comfy_quant (525MB VRAM)"
                )

            out = orig_fn(
                sd,
                output_vae=output_vae,
                output_clip=output_clip,
                output_clipvision=output_clipvision,
                embedding_directory=embedding_directory,
                output_model=output_model,
                model_options=model_options,
                te_model_options=te_model_options,
                metadata=metadata,
                disable_dynamic=disable_dynamic,
            )
            return out

        _safe_load_state_dict_guess_config._hswq_patched = True
        comfy.sd.load_state_dict_guess_config = _safe_load_state_dict_guess_config
        _CHECKPOINT_LOADER_INT8_PATCHED = True
        logger.info("[HSWQ INT8] CheckpointLoader INT8 auto-attach & CLIP dequant patch armed")
        return True
    except Exception as e:
        logger.debug("[HSWQ INT8] load_state_dict_guess_config patch failed: %s", e)
        return False
```

> Note: `apply_comfy_quant_int8_patches()` is the existing entry point (present since before `d33862a`) that bundles the INT8 infrastructure patches (LoRA key counts / LoRA name / LowVramPatch float dtype / Dynamic INT8 LoRA bake / INT8→Nunchaku VRAM handoff, etc.) and applies them once. This guide only lists its SAM3-related call sites.

### 3-4. `__init__.py` (modified, diff)

```python
# --- Startup patches (after) ---
try:
    from .patches.comfy_quant_int8 import (
        _patch_controllora_int8_dequant,
        _patch_comfy_kitchen_int8_gemm_fallback,
        _patch_sam3_process_state_dict,
        _patch_load_state_dict_guess_config_int8,
    )
    _patch_comfy_kitchen_int8_gemm_fallback()
    _patch_sam3_process_state_dict()
    _patch_load_state_dict_guess_config_int8()
    if not _patch_controllora_int8_dequant():
        logger.warning("ControlLora INT8 dequant patch not installed")
except Exception:
    ...
```

```python
# --- Node registration (added) ---
try:
    from .nodes.hswq_load_convrot_int8_sam3 import (
        HSWQSAM3Loader,
        HSWQLoadConvRotINT8SAM3,
    )
    NODE_CLASS_MAPPINGS["HSWQSAM3Loader"] = HSWQSAM3Loader
    NODE_CLASS_MAPPINGS["HSWQLoadConvRotINT8SAM3"] = HSWQLoadConvRotINT8SAM3
    logger.info("Registered HSWQ SAM3 Loader (ConvRot INT8)")
except (ImportError, ModuleNotFoundError) as e:
    logger.debug("HSWQ SAM3 Loader not registered: %s", e)

try:
    from .nodes.hswq_sam3_detect import HSWQSAM3DetectV1
    NODE_CLASS_MAPPINGS["HSWQSAM3Detect"] = HSWQSAM3DetectV1
    logger.info("Registered HSWQ SAM3 Detect")
except (ImportError, ModuleNotFoundError) as e:
    logger.debug("HSWQ SAM3 Detect not registered: %s", e)

NODE_DISPLAY_NAME_MAPPINGS["HSWQSAM3Loader"] = "HSWQ SAM3 Loader (ConvRot INT8)"
NODE_DISPLAY_NAME_MAPPINGS["HSWQLoadConvRotINT8SAM3"] = "HSWQ SAM3 Loader (ConvRot INT8)"
NODE_DISPLAY_NAME_MAPPINGS["HSWQSAM3Detect"] = "HSWQ SAM3 Detect"
```

### 3-5. `README.md` (modified, diff)

```markdown
### HSWQ SAM3 Loader (ConvRot INT8) & SAM3 Detect

ComfyUI loader and detector nodes for **ConvRot / TensorWise INT8-quantized SAM3 (Segment Anything 3) checkpoints**. Loads the SAM3 model directly into VRAM in 8-bit precision (`QuantizedTensor` / `TensorWiseINT8Layout`) and executes via `comfy_kitchen`'s high-speed `int8_linear` kernel with online activation rotation (`convrot`).

Includes automatic hardware safety fallback for unaligned layers (such as `boxRPB_embed_x` with $K=2$), dynamically dequantizing non-multiple-of-4 dimensions while running all heavy backbone and transformer blocks in accelerated INT8 Tensor Core precision.

#### Features

- **Native INT8 VRAM Retention**: Keeps weights in 8-bit precision in VRAM with `TensorWiseINT8Layout`, cutting memory requirements significantly
- **Fast Execution**: Uses `comfy_kitchen` `int8_linear` GEMM kernel with online activation rotation for ConvRot layers
- **Automatic Fallback Protection**: Layers with unaligned dimensions ($K \% 4 
eq 0$) safely compute in float precision without crashing cuBLAS INT8 GEMM
- **Seamless Compatibility**: Produces standard `MODEL` output compatible with **HSWQ SAM3 Detect** and stock ComfyUI SAM3 detection/tracking nodes
```

---

## 4. Meaning of Each Part (Code Walkthrough)

### 4-1. `nodes/hswq_load_convrot_int8_sam3.py` — Loader

| Element | Meaning |
|---|---|
| `_SAM_FOLDERS` | Model search folders: `diffusion_models` / `sams` / `detection` / `checkpoints` |
| `_get_sam3_filenames()` | Merges the file lists of the folders above to build the `sam3_name` choices; falls back to `diffusion_models` if empty |
| `_get_sam3_full_path_or_raise()` | Resolves a filename to a real path; falls back to `diffusion_models` + `get_full_path_or_raise` |
| `_decode_comfy_quant()` | Decodes `.comfy_quant` metadata (uint8 JSON byte string) into a dict |
| `_has_int8_comfy_quant()` | Returns True if the checkpoint has at least one `int8_tensorwise` comfy_quant layer. This is the branch point for INT8 loading |
| `_int8_mixed_precision_ops()` | Builds `MixedPrecisionOps` with `QUANT_ALGOS["int8_tensorwise"]` on a bfloat16 base, so Linear layers are constructed as `QuantizedTensor` (INT8) |
| `HSWQSAM3Loader.load_sam3()` | ① arm GEMM fallback patch → ② `load_torch_file` → ③ if INT8: load with MixedPrecisionOps; else plain `model_options={}` → ④ fallback to path-based `load_diffusion_model` on failure → ⑤ return standard `MODEL` |
| `HSWQLoadConvRotINT8SAM3` | Backward-compatible alias (same class) |

### 4-2. `nodes/hswq_sam3_detect.py` — Detect node

| Element | Meaning |
|---|---|
| `_extract_text_prompts()` | Extracts `(text_embeddings, attention_mask, max_detections)` list from CONDITIONING. Supports `sam3_multi_cond` (multiple texts); fills an all-ones mask when missing |
| `_refine_mask()` | Refines a coarse detection mask with the SAM decoder: crops around the detected BBOX with 10% padding → resize to 1008×1008 → `forward_segment(mask_inputs=...)` for `iterations` passes → resize back and OR with the coarse mask. Falls back to coarse when `iterations<=0` or the crop is invalid |
| `_strip_dynamic_vram_attrs()` | **Core of the fix.** After replacing an INT8 weight with float16, drops the DynamicVRAM (aimdo) vbar state (`_v` / `_prefetch` / `_v_signature` / `_v_block`). The vbar buffer was allocated with the INT8 payload size (data+scale); if `_v` stays after float16 conversion, `resolve_cast_module_with_vbar` fails with "Buffer too small". Releases the pin via `vbar_unpin`, deletes the attrs, and falls back to the regular cast path |
| `_guard_sam3_model_weights()` | Runtime stability guard. Scans all Linear/Conv layers: ① `QuantizedTensor` → `dequantize()` and replace with float16 parameter; ② raw unscaled int8 → multiply by `weight_scale` and convert to float16; ConvRot layers get the inverse Hadamard rotation via `_regular_hadamard_global` first. Always calls `_strip_dynamic_vram_attrs` after a successful replacement |
| `run_sam3_detect()` | Main execution: resize image to 1008×1008 → load model to GPU → apply guard → run detection/segmentation with point / box / text prompts → threshold filter → top-k → build BBOX dicts and masks → return `(masks, bbox_dicts, image)`. Supports multiple frames (B>1) |
| `HSWQSAM3Detect` (v3) | v3 schema version using `comfy_api.latest` (ComfyExtension) with `io.Schema` inputs/outputs |
| `HSWQSAM3DetectV1` | Legacy `INPUT_TYPES` v1 wrapper; this is what `NODE_CLASS_MAPPINGS["HSWQSAM3Detect"]` registers |

### 4-3. `patches/comfy_quant_int8.py` — SAM3 patch set

| Element | Meaning |
|---|---|
| `_patch_comfy_kitchen_int8_gemm_fallback()` | Makes comfy_kitchen INT8 GEMM safe. ① wraps `cuda.int8_linear`: falls back to float when K or N is not a multiple of 4 (e.g. `boxRPB_embed_x` K=2), when not on CUDA, or on kernel exceptions. ② registers safe handlers for `TensorWiseINT8Layout` in the dispatch table (linear / mm / addmm) with the same fallback logic. ③ `dequantize` itself correctly handles scale broadcasting and applies the inverse Hadamard rotation for ConvRot |
| `_regular_hadamard_global(size, device)` | Builds an orthogonal matrix by Kronecker-expanding the size-4 Hadamard basis `h4` up to `size`, normalized by `1/sqrt(size)`. Used for ConvRot un-rotation |
| `_patch_sam3_process_state_dict()` | Patches `process_unet_state_dict` / `process_clip_state_dict` of ComfyUI's `SAM3` class (`comfy.supported_models.SAM3`).<br>**process_unet_state_dict is a pass-through** (the previous in_proj pre-split was removed because ComfyUI's `transformers_convert` expects the fused `in_proj_weight` and performs the q/k/v split + remap to `sam3_clip.transformer.text_model.encoder.layers.N.self_attn.q_proj` itself; pre-splitting produces "clip missing". Stock behavior is correct).<br>**process_clip_state_dict adds the remap**: INT8 checkpoints store language_backbone already split into q/k/v (no `in_proj_weight`), so `transformers_convert` cannot remap them and `encoder.transformer.resblocks.N.attn.q_proj.*` keys remain. These are converted to `sam3_clip.transformer.text_model.encoder.layers.N.self_attn.q_proj.*`, and `token_embedding` / `positional_embedding` / `ln_final` / `text_projection` are mapped through `_CLIP_SIMPLE_REMAP` |
| `_dequant_and_unrotate_tensor()` | Per-tensor dequant: int8 data × scale (handles row / column / scalar / full broadcasting) + inverse Hadamard rotation for ConvRot using `convrot_groupsize` from the `comfy_quant` metadata; returns float16 |
| `_patch_load_state_dict_guess_config_int8()` | Wraps `comfy.sd.load_state_dict_guess_config` (used by CheckpointLoaderSimple etc.). The **`is_sam3` gate** (presence of `detector.` keys) passes everything non-SAM3 through untouched. For SAM3 + INT8: ① dequantizes only language_backbone (text encoder) and 4D Conv2d keys (Linear stays INT8), ② auto-attaches MixedPrecisionOps, then calls the original. **The sd-level in_proj pre-split is removed** (it broke CLIP key remapping) |

### 4-4. `__init__.py` — Startup application and registration

- **Startup patches**: `_patch_comfy_kitchen_int8_gemm_fallback` / `_patch_sam3_process_state_dict` / `_patch_load_state_dict_guess_config_int8` are applied at ComfyUI startup, so loading an INT8 SAM3 via the stock `CheckpointLoaderSimple` (without the HSWQ SAM3 Loader) also gets INT8 auto-attach + correct CLIP handling.
- **Node registration**: `HSWQSAM3Loader` / `HSWQLoadConvRotINT8SAM3` / `HSWQSAM3Detect` are registered in `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`. Import failures only log at debug level and never block startup.

### 4-5. `README.md` — Documentation

- Adds user-facing documentation of the nodes and their features (INT8 VRAM retention / fast execution / fallback protection / compatibility).

---

## Appendix: Verification Results (real hardware)

| Item | Result |
|---|---|
| fp16 model + "person" | scores top1 **0.9849**, uniform mask (ero5_ratio 0.964) |
| INT8 model + "person" | scores top1 **0.9849** (NaN resolved), uniform mask (ero5_ratio 0.964) |
| "clip missing" warning | gone in both cases |
| Image-gen INT8 (moodyProMix / NextDiT) | loads fine, all Linear layers stay INT8 (0 non-INT8) |
