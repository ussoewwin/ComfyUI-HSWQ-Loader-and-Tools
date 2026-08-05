"""
SEGS / scheduler helpers for Batched Detailer.

Developed based on ComfyUI-Impact-Pack modules/impact/core.py (ltdrdata, GPL-3.0).
"""

from __future__ import annotations

from collections import namedtuple

import comfy.samplers
import numpy as np
import torch

from . import utils

SEG = namedtuple(
    "SEG",
    [
        "cropped_image",
        "cropped_mask",
        "confidence",
        "crop_region",
        "bbox",
        "label",
        "control_net_wrapper",
    ],
    defaults=[None],
)

# Extra schedulers historically exposed by Impact Pack Detailer UI.
_ADDITIONAL_SCHEDULERS = [
    "AYS SD1",
    "AYS SDXL",
    "AYS SVD",
    "GITS[coeff=1.2]",
    "LTXV[default]",
    "OSS FLUX",
    "OSS Wan",
    "OSS Chroma",
]


def get_schedulers():
    return list(comfy.samplers.SCHEDULER_HANDLERS) + list(_ADDITIONAL_SCHEDULERS)


def segs_scale_match(segs, target_shape):
    h = segs[0][0]
    w = segs[0][1]

    th = target_shape[1]
    tw = target_shape[2]

    if (h == th and w == tw) or h == 0 or w == 0:
        return segs

    rh = th / h
    rw = tw / w

    new_segs = []
    for seg in segs[1]:
        cropped_image = seg.cropped_image
        cropped_mask = seg.cropped_mask
        x1, y1, x2, y2 = seg.crop_region
        bx1, by1, bx2, by2 = seg.bbox

        crop_region = int(x1 * rw), int(y1 * rw), int(x2 * rh), int(y2 * rh)
        bbox = int(bx1 * rw), int(by1 * rw), int(bx2 * rh), int(by2 * rh)
        new_w = crop_region[2] - crop_region[0]
        new_h = crop_region[3] - crop_region[1]

        if isinstance(cropped_mask, np.ndarray):
            cropped_mask = torch.from_numpy(cropped_mask)

        if isinstance(cropped_mask, torch.Tensor) and len(cropped_mask.shape) == 3:
            cropped_mask = torch.nn.functional.interpolate(
                cropped_mask.unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            )
            cropped_mask = cropped_mask.squeeze(0)
        else:
            cropped_mask = torch.nn.functional.interpolate(
                cropped_mask.unsqueeze(0).unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            )
            cropped_mask = cropped_mask.squeeze(0).squeeze(0).numpy()

        if cropped_image is not None:
            cropped_image = utils.tensor_resize(
                cropped_image
                if isinstance(cropped_image, torch.Tensor)
                else torch.from_numpy(cropped_image),
                new_w,
                new_h,
            )
            cropped_image = cropped_image.numpy()

        new_seg = SEG(
            cropped_image,
            cropped_mask,
            seg.confidence,
            crop_region,
            bbox,
            seg.label,
            seg.control_net_wrapper,
        )
        new_segs.append(new_seg)

    return (th, tw), new_segs


def crop_condition_mask(mask, image, crop_region):
    cond_scale = (mask.shape[1] / image.shape[1], mask.shape[2] / image.shape[2])
    mask_region = [round(v * cond_scale[i % 2]) for i, v in enumerate(crop_region)]
    return utils.crop_ndarray3(mask, mask_region)
