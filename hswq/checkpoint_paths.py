# -*- coding: utf-8 -*-
"""Arbitrary-directory loading for the HSWQ Checkpoint Loader node.

ComfyUI resolves a loader's file combo from ``folder_paths`` when the node is
registered, so a widget-combo alone cannot point at a directory the user types
afterwards. This module provides the two halves:

  * ``list_checkpoint_files(directory)`` - the checkpoint file names of one
    directory.
  * ``register_checkpoint_list_route()`` - a GET route that returns that list
    as JSON, so the frontend can refresh the node's file dropdown when the
    directory widget changes.

Checkpoints live in different folders per installation, so they are collected
into a dedicated ``folder_paths`` category. A folder the user points at is
added to that category in memory while the process runs, which makes its files
valid options of the node definition; nothing is written to disk.

Scope: the HSWQ Checkpoint Loader only. No ComfyUI core file is modified, and
the file lists of any other node stay untouched.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_ROUTE_PATH = "/hswq/checkpoint/list"
_ROUTE_REGISTERED = False

# Folders a checkpoint may live in, in the order the node searches them.
CHECKPOINT_FOLDERS = ("checkpoints",)

# Dedicated category: adding a folder here must not change the file list of
# any other node (stock CheckpointLoaderSimple, UNETLoader, ...).
CATEGORY = "hswq_checkpoint"


def _model_extensions():
    """Model file extensions ComfyUI accepts, with a safe fallback."""
    try:
        import folder_paths

        exts = tuple(folder_paths.supported_pt_extensions)
        if exts:
            return exts
    except Exception:
        pass
    return (".safetensors", ".bin", ".ckpt", ".pt", ".pth")


def _registered_paths(folder_name):
    """Registered paths of one ComfyUI model folder (empty when unknown)."""
    try:
        import folder_paths

        paths = folder_paths.folder_names_and_paths.get(folder_name)
        if not paths:
            return []
        return [p for p in paths[0] if os.path.isdir(p)]
    except Exception:
        return []


def _category_paths():
    """Paths of the dedicated checkpoint category, seeded from the folders."""
    paths = []
    for name in CHECKPOINT_FOLDERS:
        for path in _registered_paths(name):
            if path not in paths:
                paths.append(path)
    return paths


def ensure_checkpoint_category():
    """Create (or refresh) the dedicated ``hswq_checkpoint`` category.

    Returns the list of registered paths of that category.
    """
    import folder_paths

    paths = folder_paths.folder_names_and_paths.get(CATEGORY)
    if paths is None:
        folder_paths.folder_names_and_paths[CATEGORY] = (
            _category_paths(),
            folder_paths.supported_pt_extensions,
        )
        return folder_paths.folder_names_and_paths[CATEGORY][0]
    known = paths[0]
    for path in _category_paths():
        if path not in known:
            known.append(path)
    return known


def register_checkpoint_directory(directory):
    """Add ``directory`` to the checkpoint category (in memory only).

    The frontend validates a combo value against the node definition's option
    list, which is built from the category. Adding the folder the user points
    at makes its files valid options of that node.

    Returns the normalised path, or None when it is not a directory.
    """
    if not directory or not str(directory).strip():
        return None
    path = os.path.abspath(os.path.expanduser(os.path.expandvars(str(directory).strip())))
    if not os.path.isdir(path):
        return None
    known = ensure_checkpoint_category()
    if path not in known:
        known.append(path)
        logger.info("[HSWQ Checkpoint] registered extra checkpoint folder: %s", path)
    return path


def list_checkpoint_files(directory):
    """File names of the checkpoint files in ``directory``.

    ``directory`` empty -> the files of the registered checkpoint category.
    Returns a sorted list of file names (no directories, no full paths).
    """
    if not directory or not str(directory).strip():
        ensure_checkpoint_category()
        try:
            import folder_paths

            return sorted(folder_paths.get_filename_list(CATEGORY))
        except Exception:
            return []

    path = os.path.abspath(os.path.expanduser(os.path.expandvars(str(directory).strip())))
    if not os.path.isdir(path):
        return []

    exts = _model_extensions()
    names = []
    try:
        for entry in os.listdir(path):
            full = os.path.join(path, entry)
            if not os.path.isfile(full):
                continue
            if os.path.splitext(entry)[1].lower() in exts:
                names.append(entry)
    except OSError as exc:
        logger.warning("[HSWQ Checkpoint] cannot list %s: %s", path, exc)
        return []
    return sorted(names)


def resolve_checkpoint_path(file_name, directory):
    """Full path of ``file_name`` inside ``directory``.

    ``directory`` is required: the checkpoint is loaded from that folder and
    nowhere else. An absolute ``file_name`` is used as-is.
    """
    name = str(file_name or "").strip()
    if not name:
        raise ValueError("Checkpoint: no file selected")

    if os.path.isabs(name):
        return os.path.normpath(name)

    if not directory or not str(directory).strip():
        raise ValueError("Checkpoint: no folder given")

    base = os.path.abspath(os.path.expanduser(os.path.expandvars(str(directory).strip())))
    register_checkpoint_directory(base)
    return os.path.join(base, name)


def default_checkpoint_dir():
    """The registered folder that holds checkpoint files, else the primary one."""
    exts = _model_extensions()
    for path in _registered_paths("checkpoints"):
        try:
            entries = os.listdir(path)
        except OSError:
            continue
        for entry in entries:
            if os.path.splitext(entry)[1].lower() in exts:
                return path
    for path in _registered_paths("checkpoints"):
        return path
    try:
        import folder_paths

        return os.path.join(folder_paths.models_dir, "checkpoints")
    except Exception:
        return ""


def register_checkpoint_list_route():
    """Register the listing route on the running PromptServer.

    Returns True when the route is active, False when no PromptServer is
    available (the node then keeps the registered file list only).
    """
    global _ROUTE_REGISTERED
    if _ROUTE_REGISTERED:
        return True

    try:
        from aiohttp import web
        from server import PromptServer
    except Exception as exc:
        logger.debug("[HSWQ Checkpoint] list route not registered (import failed): %s", exc)
        return False

    prompt_server = getattr(PromptServer, "instance", None)
    if prompt_server is None:
        logger.debug("[HSWQ Checkpoint] list route not registered (no PromptServer)")
        return False

    try:
        routes = prompt_server.routes
    except Exception as exc:
        logger.debug("[HSWQ Checkpoint] list route not registered (no routes table): %s", exc)
        return False

    if getattr(routes, "_hswq_checkpoint_list_route", False):
        _ROUTE_REGISTERED = True
        return True

    @routes.get(_ROUTE_PATH)
    async def hswq_checkpoint_list(request):
        directory = request.rel_url.query.get("dir", "")
        return web.json_response(list_checkpoint_files(directory))

    try:
        routes._hswq_checkpoint_list_route = True
    except Exception:
        pass

    _ROUTE_REGISTERED = True
    logger.info("[HSWQ Checkpoint] directory listing route active at %s", _ROUTE_PATH)
    return True
