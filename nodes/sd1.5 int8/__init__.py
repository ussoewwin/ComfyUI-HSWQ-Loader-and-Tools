"""HSWQ SD1.5 ConvRot INT8 production loader (vendored inside nodes/sd1.5 int8).

Ported from the hswq-repo bench stack (benchmark/sd1.5_int8_traj_compare.py)
measured on 2026-10-16/17:
- spec-from-path bootstrap of the vendored ``comfy_quant_int8`` patch module
  (folder name ``sd1.5 int8`` contains a space and a dot, so regular imports
  are impossible)
- INT8 probe via ``checkpoint_looks_like_comfy_quant_int8``
- ``load_checkpoint_guess_config`` executed inside ``_int8_quant_conv_scope``
  so the Conv2d INT8 path loads with ``int8_tensorwise`` metadata (native
  integrate [{'weight_scale':1}] reading, comfortable with both full ConvRot
  and plain tensorwise packs)

Interface mirrors the existing HSWQ Checkpoint Loader (SDXL) contract:
``(ckpt_name, weight_dtype, device) -> (MODEL, CLIP)``.
"""

from __future__ import annotations

import os
import sys
import types


def _load_int8_patch_module():
    """Spec-from-path bootstrap of the vendored comfy_quant_int8 patch module.

    Mirrored verbatim from benchmark/sd1.5_int8_traj_compare.py - the vendored
    folder ``nodes/sd1.5 int8`` is the only authoritative location for the
    SD1.5 ConvRot INT8 patch (benchmark/int8 is NOT used).
    """
    import importlib.util

    folder = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(folder, "comfy_quant_int8.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"INT8 patch module not found: {path}")
    mod_name = "hswq_sd15_int8_comfy_quant_int8"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec for {path}")
    mod = sys.modules.get(mod_name)
    if mod is None:
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_checkpoint_sd15_hswq_weight_dtype(ckpt_path, device=None):
    """Load an SD1.5 ConvRot INT8 checkpoint (or fall back to stock for FP16).

    Ported from benchmark/sd1.5_int8_traj_compare.py::_load_checkpoint
    (measured on 2026-10-16/17): probe -> INT8 load scope only when the file
    looks like a comfy_quant INT8 pack; FP16 baseline loads stock.
    """
    import torch
    import folder_paths
    import comfy.sd

    _cq_int8 = _load_int8_patch_module()
    checkpoint_looks_like_comfy_quant_int8 = _cq_int8.checkpoint_looks_like_comfy_quant_int8
    _int8_quant_conv_scope = _cq_int8._int8_quant_conv_scope

    path = os.path.abspath(ckpt_path)
    use_int8_scope = checkpoint_looks_like_comfy_quant_int8(path)
    print(f"  [SD1.5 INT8] Conv2d load scope: {use_int8_scope}")
    if use_int8_scope:
        with _int8_quant_conv_scope():
            out = comfy.sd.load_checkpoint_guess_config(
                path,
                output_vae=False,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
            )
    else:
        out = comfy.sd.load_checkpoint_guess_config(
            path,
            output_vae=False,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
    model, clip, _vae = out[:3]
    return (model, clip)
