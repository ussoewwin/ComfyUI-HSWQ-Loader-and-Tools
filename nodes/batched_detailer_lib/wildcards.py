"""
Wildcard / LoRA prompt helpers for Batched Detailer.

Developed based on ComfyUI-Impact-Pack modules/impact/wildcards.py
(ltdrdata, GPL-3.0).

Full Impact Pack wildcard-file expansion (__name__) is not shipped here.
`process()` is an identity pass so prompts without Impact wildcard files
still work; `[LAB]` / `[ASC]` / `[SEP]` segment modes and `<lora:...>`
loading are preserved.
"""

from __future__ import annotations

import logging
import os
import random
import re

import folder_paths
import nodes


def process(text, seed=None):
    """Identity stand-in for Impact Pack wildcard file expansion."""
    _ = seed
    return text if text is not None else ""


def is_numeric_string(input_str):
    return re.match(r"^-?(\d*\.?\d+|\d+\.?\d*)$", input_str) is not None


def safe_float(x):
    if is_numeric_string(x):
        return float(x)
    return 1.0


def extract_lora_values(string):
    pattern = r"<lora:([^>]+)>"
    matches = re.findall(pattern, string)

    def touch_lbw(text):
        return re.sub(r"LBW=[A-Za-z][A-Za-z0-9_-]*:", r"LBW=", text)

    items = [touch_lbw(match.strip(":")) for match in matches]

    added = set()
    result = []
    for item in items:
        item = item.split(":")

        lora = None
        a = None
        b = None
        lbw = None
        lbw_a = None
        lbw_b = None
        loader = None

        if len(item) > 0:
            lora = item[0]

            for sub_item in item[1:]:
                if is_numeric_string(sub_item):
                    if a is None:
                        a = float(sub_item)
                    elif b is None:
                        b = float(sub_item)
                elif sub_item.startswith("LBW="):
                    for lbw_item in sub_item[4:].split(";"):
                        if lbw_item.startswith("A="):
                            lbw_a = safe_float(lbw_item[2:].strip())
                        elif lbw_item.startswith("B="):
                            lbw_b = safe_float(lbw_item[2:].strip())
                        elif lbw_item.strip() != "":
                            lbw = lbw_item
                elif sub_item.startswith("LOADER="):
                    loader = sub_item[7:]

        if a is None:
            a = 1.0
        if b is None:
            b = a

        if lora is not None and lora not in added:
            result.append((lora, a, b, lbw, lbw_a, lbw_b, loader))
            added.add(lora)

    return result


def remove_lora_tags(string):
    pattern = r"<lora:[^>]+>"
    return re.sub(pattern, "", string)


def resolve_lora_name(lora_name_cache, name):
    if os.path.exists(name):
        return name
    if len(lora_name_cache) == 0:
        lora_name_cache.extend(folder_paths.get_filename_list("loras"))
    for x in lora_name_cache:
        if x.endswith(name):
            return x
    return None


def process_with_loras(
    wildcard_opt, model, clip, clip_encoder=None, seed=None, processed=None
):
    lora_name_cache = []

    pass1 = process(wildcard_opt, seed)
    loras = extract_lora_values(pass1)
    pass2 = remove_lora_tags(pass1)

    for lora_name, model_weight, clip_weight, lbw, lbw_a, lbw_b, loader in loras:
        lora_name_ext = lora_name.split(".")
        if ("." + lora_name_ext[-1]) not in folder_paths.supported_pt_extensions:
            lora_name = lora_name + ".safetensors"

        orig_lora_name = lora_name
        lora_name = resolve_lora_name(lora_name_cache, lora_name)

        if lora_name is not None:
            path = folder_paths.get_full_path("loras", lora_name)
        else:
            path = None

        if path is not None:
            logging.info(
                "LOAD LORA: %s: %s, %s, LBW=%s, A=%s, B=%s, LOADER=%s",
                lora_name,
                model_weight,
                clip_weight,
                lbw,
                lbw_a,
                lbw_b,
                loader,
            )

            if loader is not None:
                if loader == "nunchaku":
                    if "NunchakuFluxLoraLoader" not in nodes.NODE_CLASS_MAPPINGS:
                        logging.warning(
                            "To use `LOADER=nunchaku`, 'ComfyUI-nunchaku' is required. "
                            "The LOADER= attribute is being ignored."
                        )
                    else:
                        cls = nodes.NODE_CLASS_MAPPINGS["NunchakuFluxLoraLoader"]
                        model = cls().load_lora(model, lora_name, model_weight)[0]
                else:
                    logging.warning("LORA LOADER NOT FOUND: '%s'", loader)
            else:

                def default_lora():
                    return nodes.LoraLoader().load_lora(
                        model, clip, lora_name, model_weight, clip_weight
                    )

                if lbw is not None:
                    if "LoraLoaderBlockWeight //Inspire" not in nodes.NODE_CLASS_MAPPINGS:
                        logging.warning(
                            "'LBW(Lora Block Weight)' is given, but the 'Inspire Pack' "
                            "is not installed. The LBW= attribute is being ignored."
                        )
                        model, clip = default_lora()
                    else:
                        cls = nodes.NODE_CLASS_MAPPINGS[
                            "LoraLoaderBlockWeight //Inspire"
                        ]
                        model, clip, _ = cls().doit(
                            model,
                            clip,
                            lora_name,
                            model_weight,
                            clip_weight,
                            False,
                            0,
                            lbw_a,
                            lbw_b,
                            "",
                            lbw,
                        )
                else:
                    model, clip = default_lora()
        else:
            logging.warning("LORA NOT FOUND: %s", orig_lora_name)

    pass3 = [x.strip() for x in pass2.split("BREAK")]
    pass3 = [x for x in pass3 if x != ""]

    if len(pass3) == 0:
        pass3 = [""]

    pass3_str = [f"[{x}]" for x in pass3]
    logging.info("CLIP: %s", str.join(" + ", pass3_str))

    result = None

    for prompt in pass3:
        if clip_encoder is None:
            cur = nodes.CLIPTextEncode().encode(clip, prompt)[0]
        else:
            cur = clip_encoder.encode(clip, prompt)[0]

        if result is not None:
            result = nodes.ConditioningConcat().concat(result, cur)[0]
        else:
            result = cur

    if processed is not None:
        processed.append(pass1)
        processed.append(pass2)
        processed.append(pass3)

    return model, clip, result


def starts_with_regex(pattern, text):
    regex = re.compile(pattern)
    return regex.match(text)


def split_to_dict(text):
    pattern = r"\[([A-Za-z0-9_. ]+)\]([^\[]+)(?=\[|$)"
    matches = re.findall(pattern, text)
    return {key: value.strip() for key, value in matches}


class WildcardChooser:
    def __init__(self, items, randomize_when_exhaust):
        self.i = 0
        self.items = items
        self.randomize_when_exhaust = randomize_when_exhaust

    def get(self, seg):
        _ = seg
        if self.i >= len(self.items):
            self.i = 0
            if self.randomize_when_exhaust:
                random.shuffle(self.items)

        item = self.items[self.i]
        self.i += 1
        return item


class WildcardChooserDict:
    def __init__(self, items):
        self.items = items

    def get(self, seg):
        text = ""
        if "ALL" in self.items:
            text = self.items["ALL"]
        if seg.label in self.items:
            text += self.items[seg.label]
        return text


def split_string_with_sep(input_string):
    sep_pattern = r"\[SEP(?:\:\w+)?\]"
    substrings = re.split(sep_pattern, input_string)

    result_list = [None]
    matches = re.findall(sep_pattern, input_string)
    for i, substring in enumerate(substrings):
        result_list.append(substring)
        if i < len(matches):
            if matches[i] == "[SEP]":
                result_list.append(None)
            elif matches[i] == "[SEP:R]":
                result_list.append(random.randint(0, 1125899906842624))
            else:
                try:
                    seed = int(matches[i][5:-1])
                except Exception:
                    seed = None
                result_list.append(seed)

    iterable = iter(result_list)
    return list(zip(iterable, iterable))


def process_wildcard_for_segs(wildcard):
    if wildcard.startswith("[LAB]"):
        raw_items = split_to_dict(wildcard)
        items = {}
        for k, v in raw_items.items():
            v = v.strip()
            if v != "":
                items[k] = v
        return "LAB", WildcardChooserDict(items)

    match = starts_with_regex(r"\[(ASC-SIZE|DSC-SIZE|ASC|DSC|RND)\]", wildcard)
    if match:
        mode = match[1]
        items = split_string_with_sep(wildcard[len(match[0]) :])
        if mode == "RND":
            random.shuffle(items)
            return mode, WildcardChooser(items, True)
        return mode, WildcardChooser(items, False)

    return None, WildcardChooser([(None, wildcard)], False)
