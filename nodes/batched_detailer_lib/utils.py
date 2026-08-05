"""
Image / mask helpers for Batched Detailer.

Developed based on ComfyUI-Impact-Pack modules/impact/utils.py (ltdrdata, GPL-3.0).
"""

from __future__ import annotations

import logging
import time

import comfy
import comfy.model_management
import nodes
import numpy as np
import torch
import torchvision
from PIL import Image


class TensorBatchBuilder:
    def __init__(self):
        self.tensor = None

    def concat(self, new_tensor):
        if self.tensor is None:
            self.tensor = new_tensor
        else:
            self.tensor = torch.concat((self.tensor, new_tensor), dim=0)


def tensor_convert_rgba(image, prefer_copy=True):
    _tensor_check_image(image)
    n_channel = image.shape[-1]
    if n_channel == 4:
        return image
    if n_channel == 3:
        alpha = torch.ones((*image.shape[:-1], 1))
        return torch.cat((image, alpha), axis=-1)
    if n_channel == 1:
        if prefer_copy:
            image = image.repeat(1, -1, -1, 4)
        else:
            image = image.expand(1, -1, -1, 3)
        return image
    raise ValueError(f"illegal conversion (channels: {n_channel} -> 4)")


def tensor_convert_rgb(image, prefer_copy=True):
    _tensor_check_image(image)
    n_channel = image.shape[-1]
    if n_channel == 3:
        return image
    if n_channel == 4:
        image = image[..., :3]
        if prefer_copy:
            # torch.Tensor has no ndarray.copy(); use clone().
            image = image.clone() if isinstance(image, torch.Tensor) else image.copy()
        return image
    if n_channel == 1:
        if prefer_copy:
            image = image.repeat(1, -1, -1, 4)
        else:
            image = image.expand(1, -1, -1, 3)
        return image
    raise ValueError(f"illegal conversion (channels: {n_channel} -> 3)")


def general_tensor_resize(image, w: int, h: int):
    _tensor_check_image(image)
    image = image.permute(0, 3, 1, 2)
    image = torch.nn.functional.interpolate(image, size=(h, w), mode="bilinear")
    image = image.permute(0, 2, 3, 1)
    return image


LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


def tensor_resize(image, w: int, h: int):
    _tensor_check_image(image)
    if image.shape[3] >= 3:
        scaled_images = TensorBatchBuilder()
        for single_image in image:
            single_image = single_image.unsqueeze(0)
            single_pil = tensor2pil(single_image)
            scaled_pil = single_pil.resize((w, h), resample=LANCZOS)
            single_image = pil2tensor(scaled_pil)
            scaled_images.concat(single_image)
        return scaled_images.tensor
    return general_tensor_resize(image, w, h)


def tensor_get_size(image):
    _tensor_check_image(image)
    _, h, w, _ = image.shape
    return (w, h)


def tensor2pil(image):
    _tensor_check_image(image)
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(0), 0, 255).astype(np.uint8)
    )


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def to_tensor(image):
    if isinstance(image, Image.Image):
        return torch.from_numpy(np.array(image)) / 255.0
    if isinstance(image, torch.Tensor):
        return image
    if isinstance(image, np.ndarray):
        return torch.from_numpy(image)
    raise ValueError(f"Cannot convert {type(image)} to torch.Tensor")


def tensor_putalpha(image, mask):
    _tensor_check_image(image)
    _tensor_check_mask(mask)
    image[..., -1] = mask[..., 0]


def _tensor_check_image(image):
    if image.ndim != 4:
        raise ValueError(f"Expected NHWC tensor, but found {image.ndim} dimensions")
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError(
            f"Expected 1, 3 or 4 channels for image, but found {image.shape[-1]} channels"
        )


def _tensor_check_mask(mask):
    if mask.ndim != 4:
        raise ValueError(f"Expected NHWC tensor, but found {mask.ndim} dimensions")
    if mask.shape[-1] != 1:
        raise ValueError(
            f"Expected 1 channel for mask, but found {mask.shape[-1]} channels"
        )


def make_4d_mask(mask):
    if len(mask.shape) == 2:
        return mask.unsqueeze(0).unsqueeze(0)
    if len(mask.shape) == 3:
        return mask.unsqueeze(0)
    return mask


def resize_mask(mask, size):
    mask = make_4d_mask(mask)
    resized_mask = torch.nn.functional.interpolate(
        mask, size=size, mode="bilinear", align_corners=False
    )
    return resized_mask.squeeze(0)


def tensor_paste(image1, image2, left_top, mask):
    _tensor_check_image(image1)
    _tensor_check_image(image2)
    _tensor_check_mask(mask)

    if image2.shape[1:3] != mask.shape[1:3]:
        mask = resize_mask(mask.squeeze(dim=3), image2.shape[1:3]).unsqueeze(dim=3)

    x, y = left_top
    _, h1, w1, c1 = image1.shape
    _, h2, w2, c2 = image2.shape
    w = min(w1, x + w2) - x
    h = min(h1, y + h2) - y
    if w <= 0 or h <= 0:
        return

    mask = mask[:, :h, :w, :]
    region1 = image1[:, y : y + h, x : x + w, :]
    region2 = image2[:, :h, :w, :]

    if c1 == 3 and c2 == 3:
        image1[:, y : y + h, x : x + w, :] = (1 - mask) * region1 + mask * region2
    elif c1 == 4 and c2 == 4:
        image1[:, y : y + h, x : x + w, :3] = (1 - mask) * region1[:, :, :, :3] + mask * region2[
            :, :, :, :3
        ]
        a1 = region1[:, :, :, 3:4]
        a2 = region2[:, :, :, 3:4] * mask
        image1[:, y : y + h, x : x + w, 3:4] = a1 + a2 * (1 - a1)
    elif c1 == 4 and c2 == 3:
        image1[:, y : y + h, x : x + w, :3] = (1 - mask) * region1[:, :, :, :3] + mask * region2
        image1[:, y : y + h, x : x + w, 3:4] = region1[:, :, :, 3:4] * (1 - mask) + mask
    elif c1 == 3 and c2 == 4:
        effective_mask = mask * region2[:, :, :, 3:4]
        image1[:, y : y + h, x : x + w, :] = (1 - effective_mask) * region1 + effective_mask * region2[
            :, :, :, :3
        ]


def tensor_gaussian_blur_mask(mask, kernel_size, sigma=10.0):
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    if mask.ndim == 2:
        mask = mask[None, ..., None]
    elif mask.ndim == 3:
        mask = mask[..., None]
    _tensor_check_mask(mask)
    if kernel_size <= 0:
        return mask

    kernel_size = kernel_size * 2 + 1
    shortest = min(mask.shape[1], mask.shape[2])
    if shortest <= kernel_size:
        kernel_size = int(shortest / 2)
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            return mask

    prev_device = mask.device
    device = comfy.model_management.get_torch_device()
    mask = mask.to(device)
    mask = mask[:, None, ..., 0]
    blurred_mask = torchvision.transforms.GaussianBlur(
        kernel_size=kernel_size, sigma=sigma
    )(mask)
    blurred_mask = blurred_mask[:, 0, ..., None]
    return blurred_mask.to(prev_device)


def crop_ndarray4(npimg, crop_region):
    x1, y1, x2, y2 = crop_region
    return npimg[:, y1:y2, x1:x2, :]


def crop_ndarray3(npimg, crop_region):
    x1, y1, x2, y2 = crop_region
    return npimg[:, y1:y2, x1:x2]


def to_latent_image(pixels, vae, vae_tiled_encode=False):
    start = time.time()
    if vae_tiled_encode:
        encoded = nodes.VAEEncodeTiled().encode(vae, pixels, 512, overlap=64)[0]
        logging.info(
            "[HSWQ BatchedDetailer] vae encoded (tiled) in %.1fs", time.time() - start
        )
    else:
        encoded = nodes.VAEEncode().encode(vae, pixels)[0]
        logging.info(
            "[HSWQ BatchedDetailer] vae encoded in %.1fs", time.time() - start
        )
    return encoded
