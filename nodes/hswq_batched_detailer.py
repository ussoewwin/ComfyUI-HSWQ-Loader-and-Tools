"""
HSWQ Batched Detailer (SEGS) - Phase-split version of DetailerForEach.

Original DetailerForEach calls core.enhance_detail() per segment, which
does VAE encode -> KSampler -> VAE decode sequentially. For n segments this
causes 3n model switches, each destroying and recreating all pin buffers.

This node restructures the loop into 3 phases:
  Phase 1: crop + upscale + VAE encode   (all segments, VAE stays loaded)
  Phase 2: KSampler                       (all segments, UNet stays loaded)
  Phase 3: VAE decode + downscale + paste (all segments, VAE stays loaded)

Result: only 2 model switches total (VAE->UNet, UNet->VAE) regardless of
how many segments are detected.

NOTE: For overlapping segments, the original processes them sequentially so
later segments crop from the already-pasted image.  This batched version crops
all segments from the original image before any pastes, which may produce
slightly different results in overlapping regions.  For face detection
(non-overlapping) the output is identical.
"""

import inspect
import logging
import math
import time

import comfy.samplers
import nodes
import torch
from nodes import MAX_RESOLUTION

from comfy_extras import nodes_differential_diffusion

from .batched_detailer_lib import (
    SEG,
    impact_sampling,
    utils,
    wildcards,
)
from .batched_detailer_lib import core

logger = logging.getLogger("HSWQ_BatchedDetailer")


class HSWQBatchedDetailer:
    """Batched Detailer (SEGS) that minimises model switching overhead."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "segs": ("SEGS",),
                "model": (
                    "MODEL",
                    {
                        "tooltip": "If the `ImpactDummyInput` is connected to the model, the inference stage is skipped."
                    },
                ),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "guide_size": (
                    "FLOAT",
                    {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "guide_size_for": (
                    "BOOLEAN",
                    {"default": True, "label_on": "bbox", "label_off": "crop_region"},
                ),
                "max_size": (
                    "FLOAT",
                    {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (core.get_schedulers(),),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "denoise": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01},
                ),
                "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                "noise_mask": (
                    "BOOLEAN",
                    {"default": True, "label_on": "enabled", "label_off": "disabled"},
                ),
                "force_inpaint": (
                    "BOOLEAN",
                    {"default": True, "label_on": "enabled", "label_off": "disabled"},
                ),
                "wildcard": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "cycle": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
            "optional": {
                "detailer_hook": ("DETAILER_HOOK",),
                "inpaint_model": (
                    "BOOLEAN",
                    {"default": False, "label_on": "enabled", "label_off": "disabled"},
                ),
                "noise_mask_feather": (
                    "INT",
                    {"default": 20, "min": 0, "max": 100, "step": 1},
                ),
                "scheduler_func_opt": ("SCHEDULER_FUNC",),
                "tiled_encode": (
                    "BOOLEAN",
                    {"default": False, "label_on": "enabled", "label_off": "disabled"},
                ),
                "tiled_decode": (
                    "BOOLEAN",
                    {"default": False, "label_on": "enabled", "label_off": "disabled"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "doit"
    CATEGORY = "HSWQ/Detailer"
    TITLE = "HSWQ Batched Detailer (SEGS)"
    DESCRIPTION = (
        "Phase-split version of Detailer (SEGS). Groups all VAE encodes, "
        "all KSampler calls, and all VAE decodes into separate phases to "
        "minimize model switching under Dynamic VRAM Loading."
    )

    @staticmethod
    def do_detail_batched(
        image,
        segs,
        model,
        clip,
        vae,
        guide_size,
        guide_size_for_bbox,
        max_size,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        denoise,
        feather,
        noise_mask,
        force_inpaint,
        wildcard_opt=None,
        detailer_hook=None,
        refiner_ratio=None,
        refiner_model=None,
        refiner_clip=None,
        refiner_positive=None,
        refiner_negative=None,
        cycle=1,
        inpaint_model=False,
        noise_mask_feather=0,
        scheduler_func_opt=None,
        tiled_encode=False,
        tiled_decode=False,
        force_fixed_latent_size=False,
    ):
        if len(image) > 1:
            raise Exception(
                "[Impact Pack] ERROR: DetailerForEach does not allow image batches.\n"
                "Please refer to https://github.com/ltdrdata/ComfyUI-extension-tutorials/"
                "blob/Main/ComfyUI-Impact-Pack/tutorial/batching-detailer.md for more information."
            )

        image = image.clone()
        enhanced_alpha_list = []
        enhanced_list = []
        cropped_list = []
        cnet_pil_list = []

        segs = core.segs_scale_match(segs, image.shape)
        new_segs = []

        wildcard_concat_mode = None
        if wildcard_opt is not None:
            if wildcard_opt.startswith("[CONCAT]"):
                wildcard_concat_mode = "concat"
                wildcard_opt = wildcard_opt[8:]
            wmode, wildcard_chooser = wildcards.process_wildcard_for_segs(wildcard_opt)
        else:
            wmode, wildcard_chooser = None, None

        if wmode in ["ASC", "DSC", "ASC-SIZE", "DSC-SIZE"]:
            if wmode == "ASC":
                ordered_segs = sorted(segs[1], key=lambda x: (x.bbox[0], x.bbox[1]))
            elif wmode == "DSC":
                ordered_segs = sorted(
                    segs[1], key=lambda x: (x.bbox[0], x.bbox[1]), reverse=True
                )
            elif wmode == "ASC-SIZE":
                ordered_segs = sorted(
                    segs[1],
                    key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                )
            else:
                ordered_segs = sorted(
                    segs[1],
                    key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                    reverse=True,
                )
        else:
            ordered_segs = segs[1]

        is_dummy = isinstance(model, str) and model == "DUMMY"

        if not is_dummy and noise_mask_feather > 0 and "denoise_mask_function" not in model.model_options:
            model = nodes_differential_diffusion.DifferentialDiffusion().execute(model)[0]

        n_segs = len(ordered_segs)
        logger.info(
            "[HSWQ BatchedDetailer] %d segments detected, running 3-phase pipeline",
            n_segs,
        )

        # =====================================================================
        # Phase 1: pre-process + VAE encode  (VAE stays loaded throughout)
        # =====================================================================
        prepared = []
        for i, seg in enumerate(ordered_segs):
            cropped_image = utils.crop_ndarray4(image.cpu().numpy(), seg.crop_region)
            cropped_image = utils.to_tensor(cropped_image)
            blur_mask = utils.to_tensor(seg.cropped_mask)
            blur_mask = utils.tensor_gaussian_blur_mask(blur_mask, feather)

            is_mask_all_zeros = (seg.cropped_mask == 0).all().item()
            if is_mask_all_zeros:
                logging.info("Detailer: segment skip [empty mask]")
                continue

            cropped_mask = seg.cropped_mask if noise_mask else None

            if wildcard_chooser is not None and wmode != "LAB":
                seg_seed, wildcard_item = wildcard_chooser.get(seg)
            elif wildcard_chooser is not None and wmode == "LAB":
                seg_seed, wildcard_item = None, wildcard_chooser.get(seg)
            else:
                seg_seed, wildcard_item = None, None

            seg_seed = seed + i if seg_seed is None else seg_seed

            if not isinstance(positive, str):
                cropped_positive = [
                    [
                        condition,
                        {
                            k: core.crop_condition_mask(v, image, seg.crop_region)
                            if k == "mask"
                            else v
                            for k, v in details.items()
                        },
                    ]
                    for condition, details in positive
                ]
            else:
                cropped_positive = positive

            if not isinstance(negative, str):
                cropped_negative = [
                    [
                        condition,
                        {
                            k: core.crop_condition_mask(v, image, seg.crop_region)
                            if k == "mask"
                            else v
                            for k, v in details.items()
                        },
                    ]
                    for condition, details in negative
                ]
            else:
                cropped_negative = negative

            if wildcard_item and wildcard_item.strip() == "[SKIP]":
                continue
            if wildcard_item and wildcard_item.strip() == "[STOP]":
                break

            orig_cropped_image = cropped_image.clone()

            if is_dummy:
                prepared.append(
                    {
                        "seg": seg,
                        "orig_cropped_image": orig_cropped_image,
                        "blur_mask": blur_mask,
                        "enhanced_image": cropped_image,
                        "cnet_pils": None,
                        "is_dummy": True,
                    }
                )
                continue

            # --- inline core.enhance_detail Phase-1 logic ---
            seg_model = model
            seg_positive = cropped_positive
            seg_negative = cropped_negative

            seg_noise_mask = cropped_mask
            if seg_noise_mask is not None:
                seg_noise_mask = utils.tensor_gaussian_blur_mask(seg_noise_mask, noise_mask_feather)
                seg_noise_mask = seg_noise_mask.squeeze(3)
                if noise_mask_feather > 0 and "denoise_mask_function" not in seg_model.model_options:
                    seg_model = nodes_differential_diffusion.DifferentialDiffusion().execute(seg_model)[0]

            if wildcard_item is not None and wildcard_item != "":
                seg_model, _, wildcard_positive = wildcards.process_with_loras(
                    wildcard_item, seg_model, clip
                )
                if wildcard_concat_mode == "concat":
                    seg_positive = nodes.ConditioningConcat().concat(seg_positive, wildcard_positive)[0]
                else:
                    seg_positive = wildcard_positive
                    seg_positive = [seg_positive[0].copy()]
                    if "pooled_output" in wildcard_positive[0][1]:
                        seg_positive[0][1]["pooled_output"] = wildcard_positive[0][1]["pooled_output"]
                    elif "pooled_output" in seg_positive[0][1]:
                        del seg_positive[0][1]["pooled_output"]

            h = cropped_image.shape[1]
            w = cropped_image.shape[2]
            bbox = seg.bbox
            bbox_h = bbox[3] - bbox[1]
            bbox_w = bbox[2] - bbox[0]

            if not force_inpaint and bbox_h >= guide_size and bbox_w >= guide_size:
                logging.info("Detailer: segment skip (enough big)")
                prepared.append(
                    {
                        "seg": seg,
                        "orig_cropped_image": orig_cropped_image,
                        "blur_mask": blur_mask,
                        "enhanced_image": None,
                        "cnet_pils": None,
                        "is_dummy": False,
                        "skipped": True,
                    }
                )
                continue

            if guide_size_for_bbox:
                upscale = guide_size / min(bbox_w, bbox_h)
            else:
                upscale = guide_size / min(w, h)

            new_w = int(w * upscale)
            new_h = int(h * upscale)

            if "aitemplate_keep_loaded" in seg_model.model_options:
                max_size = min(4096, max_size)

            if new_w > max_size or new_h > max_size:
                upscale *= max_size / max(new_w, new_h)
                new_w = int(w * upscale)
                new_h = int(h * upscale)

            if not force_inpaint:
                if upscale <= 1.0:
                    logging.info(f"Detailer: segment skip [determined upscale factor={upscale}]")
                    prepared.append(
                        {
                            "seg": seg,
                            "orig_cropped_image": orig_cropped_image,
                            "blur_mask": blur_mask,
                            "enhanced_image": None,
                            "cnet_pils": None,
                            "is_dummy": False,
                            "skipped": True,
                        }
                    )
                    continue
                if new_w == 0 or new_h == 0:
                    logging.info(f"Detailer: segment skip [zero size={new_w, new_h}]")
                    prepared.append(
                        {
                            "seg": seg,
                            "orig_cropped_image": orig_cropped_image,
                            "blur_mask": blur_mask,
                            "enhanced_image": None,
                            "cnet_pils": None,
                            "is_dummy": False,
                            "skipped": True,
                        }
                    )
                    continue
            else:
                if upscale <= 1.0 or new_w == 0 or new_h == 0:
                    logging.info("Detailer: force inpaint")
                    upscale = 1.0
                    new_w = w
                    new_h = h

            if detailer_hook is not None:
                new_w, new_h = detailer_hook.touch_scaled_size(new_w, new_h)

            logging.info(
                f"Detailer: segment upscale for ({bbox_w, bbox_h}) | crop region {w, h} x {upscale} -> {new_w, new_h}"
            )

            upscaled_image = utils.tensor_resize(cropped_image, new_w, new_h)

            if force_fixed_latent_size:
                fixed_side = int(min(guide_size, max_size))
                fixed_side = max(8, (fixed_side // 8) * 8)
                upscaled_image = utils.tensor_resize(upscaled_image, fixed_side, fixed_side)
                if seg_noise_mask is not None:
                    ndim = seg_noise_mask.ndim
                    if ndim == 2:
                        seg_noise_mask = seg_noise_mask.unsqueeze(0).unsqueeze(-1)
                    elif ndim == 3:
                        seg_noise_mask = seg_noise_mask.unsqueeze(-1)
                    seg_noise_mask = utils.tensor_resize(seg_noise_mask, fixed_side, fixed_side)
                    seg_noise_mask = seg_noise_mask.squeeze(-1)
                    if ndim == 2:
                        seg_noise_mask = seg_noise_mask.squeeze(0)
                logging.info(f"Detailer: force_fixed_latent_size -> {fixed_side}x{fixed_side}")

            if detailer_hook is not None:
                upscaled_image = detailer_hook.post_upscale(upscaled_image, seg_noise_mask)

            seg_cnet_pils = None
            if seg.control_net_wrapper is not None:
                seg_positive, seg_negative, seg_cnet_pils = seg.control_net_wrapper.apply(
                    seg_positive, seg_negative, upscaled_image, seg_noise_mask
                )
                seg_model, cnet_pils2 = seg.control_net_wrapper.doit_ipadapter(seg_model)
                seg_cnet_pils.extend(cnet_pils2)

            skip_sampling = detailer_hook is not None and detailer_hook.get_skip_sampling()

            latent_image = None
            if not skip_sampling:
                if seg_noise_mask is not None and inpaint_model:
                    imc_encode = nodes.InpaintModelConditioning().encode
                    if "noise_mask" in inspect.signature(imc_encode).parameters:
                        seg_positive, seg_negative, latent_image = imc_encode(
                            seg_positive, seg_negative, upscaled_image, vae,
                            mask=seg_noise_mask, noise_mask=True,
                        )
                    else:
                        logging.warning("[Impact Pack] ComfyUI is an outdated version.")
                        seg_positive, seg_negative, latent_image = imc_encode(
                            seg_positive, seg_negative, upscaled_image, vae, seg_noise_mask,
                        )
                else:
                    latent_image = utils.to_latent_image(
                        upscaled_image, vae, vae_tiled_encode=tiled_encode
                    )
                    if seg_noise_mask is not None:
                        latent_image["noise_mask"] = seg_noise_mask

                if detailer_hook is not None:
                    latent_image = detailer_hook.post_encode(latent_image)

            prepared.append(
                {
                    "seg": seg,
                    "seg_index": i,
                    "seg_seed": seg_seed,
                    "orig_cropped_image": orig_cropped_image,
                    "blur_mask": blur_mask,
                    "w": w,
                    "h": h,
                    "upscaled_image": upscaled_image,
                    "latent_image": latent_image,
                    "model": seg_model,
                    "positive": seg_positive,
                    "negative": seg_negative,
                    "cnet_pils": seg_cnet_pils,
                    "skip_sampling": skip_sampling,
                    "is_dummy": False,
                    "skipped": False,
                }
            )

        logger.info(
            "[HSWQ BatchedDetailer] Phase 1 complete: %d segments prepared (%d active)",
            len(prepared),
            sum(1 for p in prepared if not p.get("is_dummy") and not p.get("skipped")),
        )

        # =====================================================================
        # Phase 2: KSampler  (UNet stays loaded throughout)
        # =====================================================================
        for p in prepared:
            if p.get("is_dummy") or p.get("skipped") or p["skip_sampling"]:
                continue

            seg_model = p["model"]
            seg_positive = p["positive"]
            seg_negative = p["negative"]
            latent_image = p["latent_image"]
            seg_seed = p["seg_seed"]

            sampler_opt = None
            if detailer_hook is not None:
                sampler_opt = detailer_hook.get_custom_sampler()

            refined_latent = latent_image
            for c in range(cycle):
                if detailer_hook is not None:
                    detailer_hook.set_steps((c, cycle))
                    refined_latent = detailer_hook.cycle_latent(refined_latent)

                    model2, seed2, steps2, cfg2, sampler_name2, scheduler2, positive2, negative2, upscaled_latent2, denoise2 = (
                        detailer_hook.pre_ksample(
                            seg_model, seg_seed + c, steps, cfg, sampler_name, scheduler,
                            seg_positive, seg_negative, latent_image, denoise,
                        )
                    )
                    noise, is_touched = detailer_hook.get_custom_noise(
                        seg_seed + c,
                        torch.zeros(latent_image["samples"].size()),
                        is_touched=False,
                    )
                    if not is_touched:
                        noise = None
                else:
                    model2, seed2, steps2, cfg2, sampler_name2, scheduler2, positive2, negative2 = (
                        seg_model, seg_seed + c, steps, cfg, sampler_name, scheduler,
                        seg_positive, seg_negative,
                    )
                    denoise2 = denoise
                    noise = None

                refined_latent = impact_sampling.ksampler_wrapper(
                    model2, seed2, steps2, cfg2, sampler_name2, scheduler2,
                    positive2, negative2, refined_latent, denoise2,
                    refiner_ratio=refiner_ratio,
                    refiner_model=refiner_model,
                    refiner_clip=refiner_clip,
                    refiner_positive=refiner_positive,
                    refiner_negative=refiner_negative,
                    noise=noise,
                    scheduler_func=scheduler_func_opt,
                    sampler_opt=sampler_opt,
                )

            p["refined_latent"] = refined_latent

        logger.info("[HSWQ BatchedDetailer] Phase 2 complete: KSampler done")

        # =====================================================================
        # Phase 3: VAE decode + downscale + paste  (VAE stays loaded throughout)
        # =====================================================================
        for p in prepared:
            seg = p["seg"]
            orig_cropped_image = p["orig_cropped_image"]
            blur_mask = p["blur_mask"]

            if p.get("is_dummy"):
                enhanced_image = p["enhanced_image"]
                cnet_pils_seg = p.get("cnet_pils")
            elif p.get("skipped"):
                enhanced_image = p.get("enhanced_image")
                cnet_pils_seg = p.get("cnet_pils")
            elif p["skip_sampling"]:
                refined_image = p["upscaled_image"]
                if detailer_hook is not None:
                    refined_image = detailer_hook.post_decode(refined_image)
                if len(refined_image.shape) == 5:
                    refined_image = refined_image.squeeze(0)
                refined_image = utils.tensor_resize(refined_image, p["w"], p["h"])
                refined_image = refined_image.cpu()
                enhanced_image = refined_image
                cnet_pils_seg = p.get("cnet_pils")
            else:
                refined_latent = p["refined_latent"]

                if detailer_hook is not None:
                    refined_latent = detailer_hook.pre_decode(refined_latent)

                start = time.time()
                if tiled_decode:
                    (refined_image,) = nodes.VAEDecodeTiled().decode(vae, refined_latent, 512)
                    logging.info(f"[Impact Pack] vae decoded (tiled) in {time.time() - start:.1f}s")
                else:
                    try:
                        refined_image = vae.decode(refined_latent["samples"])
                    except Exception:
                        logging.warning(
                            f"[Impact Pack] failed after {time.time() - start:.1f}s, doing vae.decode_tiled 64..."
                        )
                        refined_image = vae.decode_tiled(refined_latent["samples"], tile_x=64, tile_y=64)
                    logging.info(f"[Impact Pack] vae decoded in {time.time() - start:.1f}s")

                if detailer_hook is not None:
                    refined_image = detailer_hook.post_decode(refined_image)

                if len(refined_image.shape) == 5:
                    refined_image = refined_image.squeeze(0)

                refined_image = utils.tensor_resize(refined_image, p["w"], p["h"])
                refined_image = refined_image.cpu()
                enhanced_image = refined_image
                cnet_pils_seg = p.get("cnet_pils")

            if cnet_pils_seg is not None:
                cnet_pil_list.extend(cnet_pils_seg)

            if enhanced_image is not None:
                image = image.cpu()
                enhanced_image = enhanced_image.cpu()
                utils.tensor_paste(
                    image, enhanced_image,
                    (seg.crop_region[0], seg.crop_region[1]),
                    blur_mask,
                )
                enhanced_list.append(enhanced_image)

                if detailer_hook is not None:
                    image = detailer_hook.post_paste(image)

            if enhanced_image is not None:
                enhanced_image_alpha = utils.tensor_convert_rgba(enhanced_image)
                new_seg_image = enhanced_image.numpy()
                mask_resized = utils.tensor_resize(blur_mask, *utils.tensor_get_size(enhanced_image))
                utils.tensor_putalpha(enhanced_image_alpha, mask_resized)
                enhanced_alpha_list.append(enhanced_image_alpha)
            else:
                new_seg_image = None

            cropped_list.append(orig_cropped_image)
            new_seg = SEG(
                new_seg_image, seg.cropped_mask, seg.confidence,
                seg.crop_region, seg.bbox, seg.label, seg.control_net_wrapper,
            )
            new_segs.append(new_seg)

        logger.info("[HSWQ BatchedDetailer] Phase 3 complete: all segments decoded and pasted")

        # Impact Pack utils.tensor_convert_rgb(..., prefer_copy=True) calls
        # ndarray.copy() on the RGBA→RGB path, which raises on torch.Tensor
        # ("'Tensor' object has no attribute 'copy'"). Keep torch-safe conversion.
        n_ch = int(image.shape[-1])
        if n_ch == 3:
            image_tensor = image
        elif n_ch == 4:
            image_tensor = image[..., :3].contiguous().clone()
        else:
            image_tensor = utils.tensor_convert_rgb(image, prefer_copy=False)

        cropped_list.sort(key=lambda x: x.shape, reverse=True)
        enhanced_list.sort(key=lambda x: x.shape, reverse=True)
        enhanced_alpha_list.sort(key=lambda x: x.shape, reverse=True)

        return (
            image_tensor,
            cropped_list,
            enhanced_list,
            enhanced_alpha_list,
            cnet_pil_list,
            (segs[0], new_segs),
        )

    def doit(
        self,
        image,
        segs,
        model,
        clip,
        vae,
        guide_size,
        guide_size_for,
        max_size,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        denoise,
        feather,
        noise_mask,
        force_inpaint,
        wildcard,
        cycle=1,
        detailer_hook=None,
        inpaint_model=False,
        noise_mask_feather=0,
        scheduler_func_opt=None,
        tiled_encode=False,
        tiled_decode=False,
    ):
        try:
            enhanced_img, *_ = HSWQBatchedDetailer.do_detail_batched(
                image, segs, model, clip, vae,
                guide_size, guide_size_for, max_size, seed, steps, cfg,
                sampler_name, scheduler, positive, negative, denoise, feather,
                noise_mask, force_inpaint, wildcard, detailer_hook,
                cycle=cycle, inpaint_model=inpaint_model,
                noise_mask_feather=noise_mask_feather,
                scheduler_func_opt=scheduler_func_opt,
                tiled_encode=tiled_encode, tiled_decode=tiled_decode,
                force_fixed_latent_size=False,
            )
        except RuntimeError as e:
            err_msg = str(e)
            if "QuantizedTensor" in err_msg and (
                "copy_" in err_msg or "size mismatch" in err_msg
            ):
                logging.warning(
                    "[HSWQ] BatchedDetailer: QuantizedTensor copy_ mismatch. "
                    "Retrying with fixed latent size for all segments: %s",
                    err_msg[:200],
                )
                enhanced_img, *_ = HSWQBatchedDetailer.do_detail_batched(
                    image, segs, model, clip, vae,
                    guide_size, guide_size_for, max_size, seed, steps, cfg,
                    sampler_name, scheduler, positive, negative, denoise, feather,
                    noise_mask, force_inpaint, wildcard, detailer_hook,
                    cycle=cycle, inpaint_model=inpaint_model,
                    noise_mask_feather=noise_mask_feather,
                    scheduler_func_opt=scheduler_func_opt,
                    tiled_encode=tiled_encode, tiled_decode=tiled_decode,
                    force_fixed_latent_size=True,
                )
            else:
                raise

        return (enhanced_img,)
