"""Arbitrary-directory listing for the HSWQ IP-Adapter loader.

ComfyUI resolves a loader's file combo from ``folder_paths`` when the node is
registered, so a widget-combo alone cannot point at a directory the user
types afterwards. This module provides the two halves:

  * ``list_ipadapter_files(directory)`` - the model file names of one
    directory (the registered ``ipadapter`` folder when the directory is
    empty).
  * ``register_ipadapter_list_route()`` - a GET route that returns that
    list as JSON, so the frontend can refresh the node's file dropdown
    when the directory widget changes.

Only model files are listed (the extensions ComfyUI itself accepts for
model files), and only the file names are returned - never a full path.

Scope: IP-Adapter route only. No ComfyUI core file and no file of the
IP-Adapter extension is modified.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_ROUTE_PATH = "/hswq/ipadapter/list"
_ROUTE_REGISTERED = False


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


def list_ipadapter_files(directory):
    """File names of the model files in ``directory``.

    ``directory`` empty -> the registered ``ipadapter`` model folder.
    Returns a sorted list of file names (no directories, no full paths).
    """
    if not directory or not str(directory).strip():
        return _list_registered_ipadapter_folder()

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
        logger.warning("[HSWQ IPAdapter] cannot list %s: %s", path, exc)
        return []
    return sorted(names)


def _list_registered_ipadapter_folder():
    """File names of the ComfyUI-registered ``ipadapter`` model folder."""
    try:
        import folder_paths

        return sorted(folder_paths.get_filename_list("ipadapter"))
    except Exception as exc:
        logger.debug("[HSWQ IPAdapter] registered ipadapter folder not available: %s", exc)
        return []


def resolve_ipadapter_path(file_name, directory):
    """Full path of ``file_name`` inside ``directory``.

    ``directory`` empty -> the registered ``ipadapter`` model folder
    (stock behaviour). Otherwise the file is looked up in that directory;
    an absolute ``file_name`` is used as-is.
    """
    name = str(file_name or "").strip()
    if not name:
        raise ValueError("IPAdapter: no file selected")

    if os.path.isabs(name):
        return os.path.normpath(name)

    if not directory or not str(directory).strip():
        import folder_paths

        return folder_paths.get_full_path_or_raise("ipadapter", name)

    base = os.path.abspath(os.path.expanduser(os.path.expandvars(str(directory).strip())))
    return os.path.join(base, name)


def register_ipadapter_list_route():
    """Register the listing route on the running PromptServer.

    Returns True when the route is active, False when no PromptServer is
    available (the node then keeps the stock file list only).
    """
    global _ROUTE_REGISTERED
    if _ROUTE_REGISTERED:
        return True

    try:
        from aiohttp import web
        from server import PromptServer
    except Exception as exc:
        logger.debug("[HSWQ IPAdapter] list route not registered (import failed): %s", exc)
        return False

    prompt_server = getattr(PromptServer, "instance", None)
    if prompt_server is None:
        logger.debug("[HSWQ IPAdapter] list route not registered (no PromptServer)")
        return False

    try:
        routes = prompt_server.routes
    except Exception as exc:
        logger.debug("[HSWQ IPAdapter] list route not registered (no routes table): %s", exc)
        return False

    if getattr(routes, "_hswq_ipadapter_list_route", False):
        _ROUTE_REGISTERED = True
        return True

    @routes.get(_ROUTE_PATH)
    async def hswq_ipadapter_list(request):
        directory = request.rel_url.query.get("dir", "")
        return web.json_response(list_ipadapter_files(directory))

    try:
        routes._hswq_ipadapter_list_route = True
    except Exception:
        pass

    _ROUTE_REGISTERED = True
    logger.info("[HSWQ IPAdapter] directory listing route active at %s", _ROUTE_PATH)
    return True
