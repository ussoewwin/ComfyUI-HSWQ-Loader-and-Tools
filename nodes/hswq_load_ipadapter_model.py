"""HSWQ Load IPAdapter Model.

Port of ComfyUI IPAdapter Plus's "IPAdapter Model Loader" node
(comfyui_ipadapter_plus ``IPAdapterModelLoader``) into the HSWQ loader
package, with the ``HSWQ`` name prefix and ConvRot INT8 pack support.

Inputs:
  * ``ipadapter_file`` - model file name, as typed.
  * ``ipadapter_dir`` - optional folder the model is loaded from. Empty means
    the ComfyUI-registered ``models/ipadapter`` folder.

The folder given in ``ipadapter_dir`` is the only place the file is loaded
from - with no folder the registered ``models/ipadapter`` folder is used.
Both inputs are plain text, so what the user types is exactly what is loaded.

Float packs take the stock path (inlined ``ipadapter_model_loader``:
key split into ``image_proj`` / ``ip_adapter`` dicts, ``plusv2`` /
``unnorm`` filename flags). ConvRot INT8 packs keep their int8 codes and
are executed through the ConvRot INT8 GEMM - the runtime patch lives in
``patches/ipadapter_convrot_int8.py`` and is installed by this node when a
ConvRot INT8 pack is loaded. Nothing is dequantized in the runtime path.
"""
from __future__ import annotations

import logging
import os

import comfy.utils
import folder_paths

logger = logging.getLogger(__name__)


def _ipadapter_model_loader(file):
    """Inline of comfyui_ipadapter_plus.utils.ipadapter_model_loader."""
    model = comfy.utils.load_torch_file(file, safe_load=True)

    if file.lower().endswith(".safetensors"):
        st_model = {"image_proj": {}, "ip_adapter": {}}
        for key in model.keys():
            if key.startswith("image_proj."):
                st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
            elif key.startswith("ip_adapter."):
                st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
        model = st_model
        del st_model

    if not "ip_adapter" in model.keys() or not model["ip_adapter"]:
        raise Exception("invalid IPAdapter model {}".format(file))

    if "plusv2" in file.lower():
        model["faceidplusv2"] = True

    if "unnorm" in file.lower():
        model["portraitunnorm"] = True

    return model


# ---- ConvRot INT8 execution ----

try:
    from ..hswq import ipadapter_convrot_int8 as _ipa_convrot
except ImportError:
    try:
        from hswq import ipadapter_convrot_int8 as _ipa_convrot
    except ImportError:
        _ipa_convrot = None


def _default_dir():
    """Registered ipadapter model folder (default of the folder field)."""
    try:
        paths = folder_paths.folder_names_and_paths.get("ipadapter")
        if paths and paths[0]:
            return paths[0][0]
    except Exception:
        pass
    return os.path.join(folder_paths.models_dir, "ipadapter")


def _registered_files():
    """Model file names offered by the file dropdown.

    The ``ipadapter`` category of ``folder_paths``. A folder outside
    ``models/ipadapter`` is added to that category in memory when the user
    points at it, so its files become valid dropdown options in this process
    without writing anything to disk and without depending on what any other
    machine has loaded before.
    """
    try:
        return sorted(folder_paths.get_filename_list("ipadapter"))
    except Exception:
        return []


def _resolve_path(file_name, directory):
    """Full path of the selected model.

    ``directory`` given -> the file is loaded from that folder and nowhere
    else. Empty -> the registered ``ipadapter`` model folder (stock).
    """
    name = str(file_name or "").strip()
    if not name:
        raise ValueError("IPAdapter: no file selected")
    if os.path.isabs(name):
        return os.path.normpath(name)

    if not directory or not str(directory).strip():
        raise ValueError("IPAdapter: no folder given")
    base = os.path.abspath(os.path.expanduser(os.path.expandvars(str(directory).strip())))
    try:
        from hswq.ipadapter_paths import register_directory

        register_directory(base)
    except Exception:
        pass
    return os.path.normpath(os.path.join(base, name))


def _split_prefixed(model):
    """Split a flat state dict into the ``image_proj`` / ``ip_adapter`` dicts."""
    st_model = {"image_proj": {}, "ip_adapter": {}}
    for key in model.keys():
        if key.startswith("image_proj."):
            st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
        elif key.startswith("ip_adapter."):
            st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
    return st_model


def _load_with_convrot_int8(file):
    """Float packs take the stock path; ConvRot INT8 packs the quantised one."""
    if _ipa_convrot is None:
        return _ipadapter_model_loader(file)

    state_dict = comfy.utils.load_torch_file(file, safe_load=True)

    unsupported = _ipa_convrot.unsupported_quant_formats(state_dict)
    if unsupported:
        raise RuntimeError(
            "IPAdapter: unsupported quantised format(s) %s in %s - only %s is "
            "supported; refusing to load quantised codes into float weights"
            % (sorted(unsupported), file, "int8_tensorwise")
        )

    if not _ipa_convrot.has_convrot_int8(state_dict):
        # No ConvRot INT8: stock behaviour (inline of ipadapter_model_loader)
        return _ipadapter_model_loader(file)

    # ConvRot INT8 pack: keep the int8 codes, install the runtime patch so the
    # graph evaluates them with the ConvRot INT8 GEMM instead of reading the
    # codes as float weights.
    from ..patches.ipadapter_convrot_int8 import patch_ipadapter_plus

    if not patch_ipadapter_plus():
        raise RuntimeError(
            "IPAdapter: ConvRot INT8 pack %s needs the comfyui_ipadapter_plus "
            "extension for the runtime patch; it is not loaded." % file
        )

    st_model = _split_prefixed(state_dict)
    if "plusv2" in file.lower():
        st_model["faceidplusv2"] = True
    if "unnorm" in file.lower():
        st_model["portraitunnorm"] = True

    n_int8 = 0
    for sub in ("image_proj", "ip_adapter"):
        n_int8 += len(_ipa_convrot.split_convrot_state_dict(st_model[sub])[0])
    logger.info(
        "[HSWQ IPAdapter] ConvRot INT8 pack loaded: %d int8 layers (codes kept INT8)",
        n_int8,
    )
    return st_model


class HSWQLoadIPAdapterModel:
    """Load an IP-Adapter model from the given folder (or models/ipadapter)."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # COMBO: the dropdown of IP-Adapter model files. Its options
                # follow ipadapter_dir (route /hswq/ipadapter/list).
                "ipadapter_file": (
                    _registered_files() if _registered_files() else [""],
                    {"tooltip": "IP-Adapter model file (dropdown follows ipadapter_dir)."},
                ),
                "ipadapter_dir": (
                    "STRING",
                    {
                        "default": _default_dir(),
                        "multiline": False,
                        "tooltip": "Folder the IP-Adapter model is loaded from.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IPADAPTER",)
    FUNCTION = "load_ipadapter_model"
    CATEGORY = "loaders"
    TITLE = "HSWQ Load IPAdapter Model"

    @classmethod
    def VALIDATE_INPUTS(s, ipadapter_file, ipadapter_dir=""):
        """Resolve the path and report a missing file before the run."""
        try:
            path = _resolve_path(ipadapter_file, ipadapter_dir)
        except Exception as exc:
            return "IPAdapter: %s" % exc
        if not os.path.isfile(path):
            return "IPAdapter: model file not found: %s" % path
        return True

    def load_ipadapter_model(self, ipadapter_file, ipadapter_dir=""):
        path = _resolve_path(ipadapter_file, ipadapter_dir)
        if not os.path.isfile(path):
            raise FileNotFoundError("IPAdapter: model file not found: %s" % path)
        return (_load_with_convrot_int8(path),)
