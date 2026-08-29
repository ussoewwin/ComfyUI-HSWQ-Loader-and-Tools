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


def run_sam3_detect(model, image, conditioning=None, bboxes=None, positive_coords=None, negative_coords=None, threshold=0.5, refine_iterations=2, individual_masks=False):
    try:
        from ..patches.comfy_quant_int8 import _patch_comfy_kitchen_int8_gemm_fallback
        _patch_comfy_kitchen_int8_gemm_fallback()
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
