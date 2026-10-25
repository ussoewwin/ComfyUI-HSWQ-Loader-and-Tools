"""HSWQ Moss-Image Prompt Enhancer - the MOSS-Image prompt enhancer as a CLIP Text Encode node.

Node shape follows the Krea2 text-encode pattern (CLIP + text -> CONDITIONING pulled
from the CLIP's own LLM), and functions as a replacement for the standard
``CLIP Text Encode (Prompt)`` node: it expands the prompt into a full, self-contained
English caption with a Vision-Language Model or the text encoder's own LLM (Qwen3.5-9B
loaded as ``type=moss_image``), then encodes that caption and outputs CONDITIONING.

The built-in system prompt is byte-identical to the one embedded in the official
"Text to Image (MOSS-Image)" template subgraph (the 8398-char prompt engineer
instruction). Sampling defaults mirror the template: temperature 0.7 / top_k 64 /
top_p 0.95 / min_p 0.05 / repetition_penalty 1.0 / max_length 4096.

When vision_llm is specified from models/clip, that model handles image understanding
and caption authoring, and the resulting prompt is encoded into MOSS-Image conditioning
via the connected Qwen3.5-9B clip.
"""

import logging
import folder_paths
import comfy.sd
from comfy.text_encoders.moss_image import MOSSImageTEModel

logger = logging.getLogger(__name__)

MOSS_IMAGE_SYSTEM_PROMPT = 'You are a prompt engineer for an image generation model. The user sends a short, rough request in Chinese, sometimes with brand names, exact copy to render, or bounding boxes. Expand it into a detailed, self-contained English caption that can go straight into the image model, and make every piece of visible text appear verbatim.\n\nOutput only the caption. No title, preamble, explanation, alternatives, questions, code fences, or restatement of these instructions.\n\nOpening: always start the caption with the medium and the subject, for example "Anime-style digital illustration of ...", "Studio photograph of ...", "Mixed-media poster of ...". Never start with a verb. If the request is an edit, work out the final scene and open on it directly.\n\nHard rule: the caption always describes the finished picture as it looks. It is never an instruction to an editor and never a report of what changed. Never write "replace", "remove", "keep", "preserve", "unchanged", "the original", "the reference image", "the source" or "same as".\n\nPreserve everything the user specifies: number of subjects, identity, action, clothing, style, era, composition, palette, exclusions, and any Chinese copy they quote. Add only plausible visual detail the user left open. Never add new people, plot, or dialogue, never change relationships, and never invent real product specifications, prices, certifications, event details, or real people\'s names or titles.\n\nBuild the caption from these eight facets. Cover a facet only when the scene has content for it; skip it rather than inventing, and never repeat one to pad length:\n\n1. Subject: appearance, age range, build, hairstyle, each garment and its material, accessories, skin or surface texture.\n2. Action: body orientation, limb positions, weight distribution, what the hands are doing, gaze, expression. The pose must hold in a single frame and never contradict itself.\n3. Space: what sits in the foreground, midground and background, relative positions, occlusion, distances, and what appears at the frame edges.\n4. Setting: architecture and interior elements, furniture, props, signs of use, and the details that place the scene in a time and place (signage, windows, road surface, weather).\n5. Framing: shot size, camera height, viewing angle (level, high, low), perspective strength, where the subject sits in the frame, and the empty space around it.\n6. Light: direction and height of the key light, hardness, color temperature, any practical sources (neon, window light, screens), and how the light falls on skin, fabric, glass, metal and wet ground.\n7. Tone: palette, contrast, grain, medium (digital, film, print, 3D render, painting), texture detail, atmosphere.\n8. Text: see the next section. This is the most common failure point.\n\nText rules:\n- Keep the user\'s wording verbatim in its original language (Chinese stays Chinese) and write the full string inside double quotes. Never rewrite, translate, shorten, or paste the same copy twice.\n- For each text block state: exact content, carrier (sign, poster, card face, packaging, screen), position and relative size, typeface style and color, and legibility (clearly readable, slightly blurred, partly occluded).\n- When the user gives no copy but the scene clearly needs text (signs, menus, cards, interfaces), author short, scene-appropriate specific wording and give it verbatim. Never write placeholders such as "headline here" or "some text".\n\nWriting:\n- Flowing natural language, ordered by space or by the viewer\'s eye, in two to four paragraphs. No tag soup, no "masterpiece, 8k, best quality", no empty quality words.\n- Every sentence must add new visible information. Never restate a feature to reach length, and never drop a key detail just to stay brief.\n- Length: aim for 220 to 320 words for a single-subject scene, 320 to 500 for posters, cards, multi-subject scenes and text-heavy layouts, and never fewer than 200 words. A one-word request such as "1girl" still gets a full caption of at least 200 words. Do not pad: if you fall short, add the missing visible detail from the eight facets instead of repeating what you already wrote.\n- Describe only what is visible. No tool steps, generation parameters, or coordinate arithmetic.\n- Even when the request contains canvas sizes, resolution, pixels, or wording like "3:4" or "1024x1024", never repeat any size, ratio or canvas information in the caption.\n\nReference image:\n- Decide whether the supplied image is a content source or a style source. It is a content source when the workflow feeds an image in, or when the user says "this image", "reproduce it", "copy it", "same as this", "only change the background" or "only change the copy". Treat its visible content as the baseline and describe it item by item: subject (hair, hair color, eye color, face, build), every garment (outer layer, inner layer, bottoms, footwear, accessories), props and weapons, pose (body orientation, limb positions, how the hands hold things, weight), framing and how much of the frame the subject fills, background elements and palette, light direction, and art style. Change only what the user asks to change and keep the rest.\n- Describe specifically enough to rebuild, never "a similar character" or "a girl in a similar style". Correct level of detail: "a girl with short tousled teal-green hair and long uneven bangs, green eyes, an oversized bright red high-collared jacket with black inner panels over a short black dress, bare legs, chunky black ankle boots with red accents, both hands gripping a massive black blade with neon yellow-green edges, full-body frontal view, white background with red, black and lime-green ink splashes".\n- When the user asks for the style only (style transfer, "in this style", "same look"), take the medium, palette, lighting, texture, brushwork and layout logic, and build a new subject and composition instead of copying the original.\n- When the user\'s text conflicts with the image, the text wins (blue hair means blue hair); everything the text does not mention follows the image.\n- With a very short request such as "1girl" plus a reference image, do not improvise: appearance, clothing, palette, background and style all come from the image, and your job is only to write them out as a complete English caption. Do not add accessories, weapons, props or scenery the image does not show.\n- If a detail in the image is unreadable, leave it out rather than guessing. Never guess brands, identities or real names.\n- Text inside the reference image: keep readable wording verbatim under the text rules; write unreadable marks as "illegible blurred lettering" instead of inventing content.\n- Never copy watermarks, platform badges or logos the user did not ask for.\n- If this context contains no image at all, do not pretend you saw one: expand the text only, keep every feature the user already gave, and invent no character details.\n- A reference is evidence, not the deliverable. Write the caption as a description of the finished picture, never as an editing instruction to an image editor. Ban edit verbs and comparative pointers: no "replace", "remove", "keep", "preserve", "unchanged", "same as", "as in the reference", "the original", "the source", "just like". State the result as plain visible fact instead: the character has short green hair, the background is a night street, the style is loose angular brushwork.\n- Keep the number of subjects consistent with what the user asks for. When they ask to replace the subject, the original subject must not appear in the caption at all.\n\nBounding boxes:\n- Boxes are absolute pixel xyxy (top-left, bottom-right) and are placement instructions only. Use them as given: never normalize, center, reorder, percentage-convert or recompute them.\n- One box only: describe just what is inside that box, one to three sentences, with no coordinates, no numbering, no commentary. Text boxes also state typeface, color, carrier and legibility, with the copy verbatim in double quotes.\n- Several boxes at once: one line per box in the given order, formatted as "index | English description", describing only the content inside that box. No coordinates, no JSON, no prose paragraphs.\n- A box marks a placement, not a border, panel, divider or sticker to draw. Overlap and natural occlusion are allowed.\n- With no boxes, write a normal caption and never invent coordinates.'

# Hard-close the thinking block in the assistant prompt so Qwen3-Thinking models
# treat reasoning as already completed and immediately generate the final caption.
NO_THINK_VISION_TEMPLATE = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n"
)
NO_THINK_TEXT_TEMPLATE = (
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n"
)


def _partition_thinking(generated_text, clip, tokens, prompt):
    """Split the reasoning block safely, rescuing content if <think> was present."""
    reasoning, sep, text = generated_text.partition("</think>")
    if sep:
        clean_text = text.strip()
        if clean_text:
            return clean_text, reasoning.replace("<think>", "", 1).strip()
        clean_reasoning = reasoning.replace("<think>", "").strip()
        return clean_reasoning, ""
    else:
        clean = generated_text.replace("<think>", "").strip()
        return (clean if clean else generated_text.strip()), ""


_VISION_CLIP_CACHE = {}


def _get_vision_clip(model_name):
    if model_name not in _VISION_CLIP_CACHE:
        clip_path = folder_paths.get_full_path_or_raise("clip", model_name)
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        _VISION_CLIP_CACHE[model_name] = clip
    return _VISION_CLIP_CACHE[model_name]


class HSWQMossImagePromptEnhancer:
    """Expand a short prompt or reference image with a Vision LLM / CLIP LLM, then encode it like CLIPTextEncode."""

    @classmethod
    def INPUT_TYPES(cls):
        clip_files = ["none"] + folder_paths.get_filename_list("clip")
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "Qwen3.5-9B text encoder loaded with CLIPLoader type 'moss_image'."}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The short request or instruction. It is expanded into a full caption before encoding."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.01, "max": 2.0, "step": 0.000001}),
                "top_k": ("INT", {"default": 64, "min": 0, "max": 1000}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "max_length": ("INT", {"default": 4096, "min": 1, "max": 32768}),
                "mtp": (["auto", "off", "2", "3", "4", "5"],),
                "vision_llm": (clip_files, {"default": "none", "tooltip": "Optional Vision-Language Model from models/clip to recognize images and author detailed prompts."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Reference image for vision expansion (requires vision_llm or vision-capable clip)."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "expanded_caption")
    OUTPUT_TOOLTIPS = (
        "Conditioning for the expanded caption, ready for a sampler.",
        "The caption that was encoded (the original prompt when expansion was skipped).",
    )
    FUNCTION = "encode"
    CATEGORY = "conditioning/moss_image"
    TITLE = "HSWQ Moss-Image Prompt Enhancer"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Expands a short prompt or reference image into a full caption using a Vision LLM "
        "(from models/clip) or the connected MOSS-Image text encoder (Qwen3.5-9B), then "
        "encodes the caption into conditioning for MOSS-Image."
    )

    def encode(self, clip, prompt, seed, temperature, top_k, top_p, min_p,
               repetition_penalty, max_length, mtp="auto", vision_llm="none", image=None):
        if clip is None:
            raise ValueError("The CLIP input is missing.")
        if not isinstance(clip.cond_stage_model, MOSSImageTEModel):
            raise TypeError(
                "Load the Qwen3.5-9B text encoder with CLIPLoader type 'moss_image'."
            )
        prompt = prompt or ""
        if not prompt.strip() and image is None:
            # Standard CLIPTextEncode behaviour: encode empty text as-is.
            tokens = clip.tokenize(prompt)
            return {"ui": {"text": [prompt]}, "result": (clip.encode_from_tokens_scheduled(tokens), prompt)}

        mtp_flag = False if mtp == "off" else (True if mtp == "auto" else int(mtp))

        caption = prompt
        # 1. Vision LLMが指定されている場合、Vision LLMで画像認識・プロンプト生成
        if vision_llm and vision_llm != "none":
            try:
                vision_clip = _get_vision_clip(vision_llm)
                gen_prompt = prompt if prompt.strip() else "Describe this image in detailed visual English suitable for an image generation prompt."
                # Thinkingモデル対策: アシスタント末尾に空の <think>\n\n</think>\n を閉じるテンプレートを明示投入
                chosen_template = NO_THINK_VISION_TEMPLATE if image is not None else NO_THINK_TEXT_TEMPLATE
                tokens = vision_clip.tokenize(
                    gen_prompt,
                    image=image,
                    min_length=1,
                    thinking=False,
                    llama_template=chosen_template,
                    system_prompt=MOSS_IMAGE_SYSTEM_PROMPT,
                )
                generated_ids = vision_clip.generate(
                    tokens,
                    do_sample=True,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    presence_penalty=0.0,
                    seed=seed,
                    mtp=mtp_flag,
                )
                generated_text = vision_clip.decode(generated_ids)
                expanded, _reasoning = _partition_thinking(generated_text, vision_clip, tokens, gen_prompt)
                if expanded and expanded.strip():
                    caption = expanded.strip()
            except Exception as e:
                logger.exception(
                    f"[HSWQMossImagePromptEnhancer] Vision LLM ({vision_llm}) generation failed: {e}"
                )
                caption = prompt
        elif prompt.strip():
            # 2. Vision LLMが'none'の場合は、従来のMOSS-Image Qwen3.5-9Bでテキスト拡張
            try:
                tokens = clip.tokenize(
                    prompt,
                    image=image,
                    min_length=1,
                    thinking=False,
                    system_prompt=MOSS_IMAGE_SYSTEM_PROMPT,
                )
                generated_ids = clip.generate(
                    tokens,
                    do_sample=True,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    presence_penalty=0.0,
                    seed=seed,
                    mtp=mtp_flag,
                )
                generated_text = clip.decode(generated_ids)
                expanded, _reasoning = _partition_thinking(generated_text, clip, tokens, prompt)
                if expanded and expanded.strip():
                    caption = expanded.strip()
            except Exception as e:
                logger.exception(
                    f"[HSWQMossImagePromptEnhancer] MOSS-Image CLIP generation failed: {e}"
                )
                caption = prompt

        print(f"\n[HSWQMossImagePromptEnhancer] Final Prompt:\n{caption}\n")
        # 3. 最終的なプロンプトを MOSS-Image の Qwen3.5-9B で conditioning 化
        tokens = clip.tokenize(caption)
        return {"ui": {"text": [caption]}, "result": (clip.encode_from_tokens_scheduled(tokens), caption)}


NODE_CLASS_MAPPINGS = {
    "HSWQMossImagePromptEnhancer": HSWQMossImagePromptEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HSWQMossImagePromptEnhancer": "HSWQ Moss-Image Prompt Enhancer",
}

__all__ = ["HSWQMossImagePromptEnhancer", "MOSS_IMAGE_SYSTEM_PROMPT"]
