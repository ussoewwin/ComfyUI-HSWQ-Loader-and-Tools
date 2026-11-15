"""HSWQ Load IPAdapter Model.

Port of ComfyUI IPAdapter Plus's "IPAdapter Model Loader" node
(comfyui_ipadapter_plus ``IPAdapterModelLoader``) into the HSWQ loader
package, with the ``HSWQ`` name prefix, ConvRot INT8 pack support and an
arbitrary model folder.

Inputs:
  * ``ipadapter_file`` - model file, a dropdown like the stock node. Its
    options are the registered ``models/ipadapter`` folder at registration
    time; the frontend replaces them with the files of ``ipadapter_dir``
    when that folder is given (route ``/hswq/ipadapter/list``).
  * ``ipadapter_dir`` - optional folder the model is loaded from. Empty means
    the ComfyUI-registered ``models/ipadapter`` folder (stock behaviour).
    An absolute ``ipadapter_file`` is used as-is.

Why ``VALIDATE_INPUTS`` exists: ComfyUI validates a COMBO value against the
option list the node registered (``execution.py`` -> ``value_not_in_list``).
Because that list is fixed at registration, a file of a folder the user
types afterwards is not in it. A node that declares the input in its own
``VALIDATE_INPUTS`` signature owns that input's validation instead - ComfyUI
then skips the generic list check for it (``execution.py``: the check is
inside ``if x not in validate_function_inputs``). The node validates the
resolved path itself, so the dropdown still works while any folder is
accepted. Nothing of ComfyUI is modified.

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

# ---- arbitrary load location ----

try:
    from ..hswq import ipadapter_paths as _ipa_paths
except ImportError:
    try:
        from hswq import ipadapter_paths as _ipa_paths
    except ImportError:
        _ipa_paths = None


def _registered_files():
    """File list of the registered ``ipadapter`` model folder."""
    if _ipa_paths is not None:
        files = _ipa_paths.list_ipadapter_files("")
        if files:
            return files
    try:
        return sorted(folder_paths.get_filename_list("ipadapter"))
    except Exception:
        pass
    # The "ipadapter" folder is registered by comfyui_ipadapter_plus; if it is
    # not loaded, scan models/ipadapter directly with ComfyUI's model
    # extensions so the dropdown is never empty.
    base = os.path.join(folder_paths.models_dir, "ipadapter")
    exts = tuple(folder_paths.supported_pt_extensions)
    files = []
    try:
        for entry in os.listdir(base):
            full = os.path.join(base, entry)
            if os.path.isfile(full) and os.path.splitext(entry)[1].lower() in exts:
                files.append(entry)
    except OSError:
        pass
    return sorted(files)


def _resolve_path(file_name, directory):
    """Full path of the selected model."""
    if _ipa_paths is not None:
        return _ipa_paths.resolve_ipadapter_path(file_name, directory)

    name = str(file_name or "").strip()
    if not name:
        raise ValueError("IPAdapter: no file selected")
    if os.path.isabs(name):
        return os.path.normpath(name)
    if not directory or not str(directory).strip():
        return folder_paths.get_full_path_or_raise("ipadapter", name)
    base = os.path.abspath(os.path.expanduser(os.path.expandvars(str(directory).strip())))
    return os.path.join(base, name)


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
    """Load an IP-Adapter model from any folder (float + ConvRot INT8)."""

    @classmethod
    def INPUT_TYPES(s):
        files = _registered_files()
        return {
            "required": {
                # COMBO: the frontend renders a dropdown, exactly like the
                # stock IPAdapter loader. Its options follow ipadapter_dir
                # at runtime (frontend refresh through /hswq/ipadapter/list).
                "ipadapter_file": (
                    files if files else [""],
                    {"tooltip": "IP-Adapter model file. The list follows ipadapter_dir."},
                ),
            },
            "optional": {
                "ipadapter_dir": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Folder the IP-Adapter model is loaded from. Leave empty "
                            "for the ComfyUI models/ipadapter folder."
                        ),
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
        """Own validation of ``ipadapter_file`` (see module docstring).

        Declaring the input here makes ComfyUI skip its registration-time
        COMBO list check for it, so a file of ``ipadapter_dir`` is accepted.
        The path is resolved and checked instead.
        """
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
