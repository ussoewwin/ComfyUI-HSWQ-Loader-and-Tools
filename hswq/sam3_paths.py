"""Arbitrary-directory loading for the HSWQ SAM3 loader.

ComfyUI resolves a loader's file combo from ``folder_paths`` when the node is
registered, so a widget-combo alone cannot point at a directory the user types
afterwards. This module provides the same two halves the IP-Adapter loader
uses, for SAM3 models:

  * ``list_sam3_files(directory)`` - the model file names of one directory.
  * ``register_sam3_list_route()`` - a GET route that returns that list as
    JSON, so the frontend can refresh the node's file dropdown when the
    directory widget changes.

SAM3 models live in different folders per installation (``diffusion_models``,
``sams``, ``detection``, ``checkpoints``), so they are collected into a
dedicated ``folder_paths`` category. A folder the user points at is added to
that category in memory while the process runs, which makes its files valid
options of the node definition; nothing is written to disk.

Scope: SAM3 only. No ComfyUI core file and no file of the SAM3 extension is
modified.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_ROUTE_PATH = "/hswq/sam3/list"
_ROUTE_REGISTERED = False

# Folders a SAM3 model may live in, in the order the loader searches them.
SAM3_FOLDERS = ("diffusion_models", "sams", "detection", "checkpoints")

# Dedicated category: adding a folder here must not change the file list of
# any other node (stock UNETLoader, CheckpointLoaderSimple, ...).
CATEGORY = "hswq_sam3"


def _clean_path(value):
    """Strip whitespace and surrounding quotes from a path written by hand."""
    text = str(value or "").strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in ("\"", "'"):
        text = text[1:-1].strip()
    return text


def _workdir_entries(base, max_depth=4, max_entries=20000):
    """Model files inside ``base``, looked at through its subfolders too.

    Yields (full_path, name) pairs, shallow levels first. Hidden and build
    folders are skipped, and the walk stops at ``max_entries`` so a large
    checkout cannot stall the listing.
    """
    skip = {
        ".git", ".hg", ".svn", "__pycache__", "node_modules", "target",
        ".cargo", ".venv", "venv", "site-packages",
    }
    base_depth = base.rstrip("\\/").count(os.sep)
    seen = 0
    try:
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d.lower() not in skip and not d.startswith(".")]
            if root.rstrip("\\/").count(os.sep) - base_depth >= max_depth:
                dirs[:] = []
            for entry in files:
                seen += 1
                if seen > max_entries:
                    return
                yield os.path.join(root, entry), entry
    except OSError:
        return


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
    """Paths of the dedicated SAM3 category, seeded from the known folders."""
    paths = []
    for name in SAM3_FOLDERS:
        for path in _registered_paths(name):
            if path not in paths:
                paths.append(path)
    return paths


def ensure_sam3_category():
    """Create (or refresh) the dedicated ``hswq_sam3`` folder category.

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


def register_sam3_directory(directory):
    """Add ``directory`` to the SAM3 category (in memory only).

    The frontend validates a combo value against the node definition's option
    list, which is built from the category. Adding the folder the user points
    at makes its files valid options of that node.

    Returns the normalised path, or None when it is not a directory.
    """
    cleaned = _clean_path(directory)
    if not cleaned:
        return None
    path = os.path.abspath(os.path.expanduser(os.path.expandvars(cleaned)))
    if not os.path.isdir(path):
        return None
    known = ensure_sam3_category()
    if path not in known:
        known.append(path)
        logger.info("[HSWQ SAM3] registered extra model folder: %s", path)
    return path


def list_sam3_files(directory):
    """File names of the model files in ``directory``.

    ``directory`` empty -> the files of the registered SAM3 category.
    Returns a sorted list of file names (no directories, no full paths).
    """
    if not directory or not str(directory).strip():
        ensure_sam3_category()
        try:
            import folder_paths

            return sorted(folder_paths.get_filename_list(CATEGORY))
        except Exception:
            return []

    path = os.path.abspath(os.path.expanduser(os.path.expandvars(_clean_path(directory))))
    if not os.path.isdir(path):
        return []

    exts = _model_extensions()
    names = []
    for full, entry in _workdir_entries(path):
        if os.path.splitext(entry)[1].lower() not in exts:
            continue
        rel = os.path.relpath(full, path)
        names.append(entry if rel == entry else rel.replace(os.sep, "\\"))
    return sorted(names)


def resolve_sam3_path(file_name, directory):
    """Full path of ``file_name`` inside ``directory``.

    ``directory`` is required: the model is loaded from that folder and
    nowhere else. An absolute ``file_name`` is used as-is.
    """
    name = _clean_path(file_name)
    if not name:
        raise ValueError("SAM3: no file selected")

    if os.path.isabs(name):
        return os.path.normpath(name)

    base = _clean_path(directory)
    if not base:
        raise ValueError("SAM3: no folder given")

    base = os.path.abspath(os.path.expanduser(os.path.expandvars(base)))
    register_sam3_directory(base)

    direct = os.path.join(base, name)
    if os.path.isfile(direct):
        return direct

    # A pack is often kept one level down inside the folder that was given.
    # Look through the same folder's subfolders before reporting it missing;
    # nothing outside the given folder is searched.
    wanted = os.path.basename(name).lower()
    for full, entry in _workdir_entries(base):
        if entry.lower() == wanted:
            logger.info("[HSWQ SAM3] resolved %s inside %s", name, base)
            return full
    return direct


def default_sam3_dir():
    """The registered folder that holds a SAM3 model, else the primary one.

    SAM3 models are shipped under different names and folders, so the folder
    is picked from the files that are actually present: the first registered
    folder that contains a file whose name mentions SAM3.
    """
    for name in SAM3_FOLDERS:
        for path in _registered_paths(name):
            try:
                entries = os.listdir(path)
            except OSError:
                continue
            for entry in entries:
                if "sam3" in entry.lower():
                    return path
    for path in _registered_paths("diffusion_models"):
        return path
    try:
        import folder_paths

        return os.path.join(folder_paths.models_dir, "diffusion_models")
    except Exception:
        return ""


def register_sam3_list_route():
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
        logger.debug("[HSWQ SAM3] list route not registered (import failed): %s", exc)
        return False

    prompt_server = getattr(PromptServer, "instance", None)
    if prompt_server is None:
        logger.debug("[HSWQ SAM3] list route not registered (no PromptServer)")
        return False

    try:
        routes = prompt_server.routes
    except Exception as exc:
        logger.debug("[HSWQ SAM3] list route not registered (no routes table): %s", exc)
        return False

    if getattr(routes, "_hswq_sam3_list_route", False):
        _ROUTE_REGISTERED = True
        return True

    @routes.get(_ROUTE_PATH)
    async def hswq_sam3_list(request):
        directory = request.rel_url.query.get("dir", "")
        return web.json_response(list_sam3_files(directory))

    try:
        routes._hswq_sam3_list_route = True
    except Exception:
        pass

    _ROUTE_REGISTERED = True
    logger.info("[HSWQ SAM3] directory listing route active at %s", _ROUTE_PATH)
    return True
