"""ComfyUI v0.38 fast-disk recovery guard (aimdo direct file read).

Root cause (confirmed against v0.38.1 logs + source):

ComfyUI v0.38 added automatic --fast-disk detection to every model loader
(commits 7a0b5eede / e638023d5 / 5ba116a40). For a disk-detected model the
state-dict tensors keep a read-only mmap plus a _comfy_tensor_file_slice
descriptor, and the weight transfer path
(comfy.model_management.cast_to_gathered ->
comfy.memory_management.read_tensor_file_slice_into) does the copy through the
native comfy_aimdo file reader instead of RAM pinning.

comfy_aimdo.host_buffer.read_file_to_device raises RuntimeError when the native
hostbuf_file_reader_read returns false. During sampling this unwinds into the
HSWQ JointAttention guard, whose eager fallback re-reads the same weights and
hits the native reader again -> Fatal Python error: Aborted (unrecoverable
C-level abort, cannot be caught from Python).

ComfyUI already defines a safe fallback: cast_to_gathered treats a falsy return
of read_tensor_file_slice_into as "use the plain mmap copy"
(dest_view.copy_(tensor)). The only defect is that the aimdo read *raises*
instead of signalling failure, so the fallback never runs.

This guard wraps ONLY comfy.memory_management.read_tensor_file_slice_into: a
RuntimeError from the native reader becomes a False return, so ComfyUI's own
mmap copy fallback takes over instead of aborting. On success nothing changes.

Important: the aimdo helpers themselves are deliberately left raising. If
read_file_to_device were made to silently return False it would bypass the
exception and hit ComfyUI's `if destination is None: ... return True` branch,
which ignores the result and would skip the copy -> silent bad weights. Catching
at read_tensor_file_slice_into keeps the fallback on the correct path for both
the device-destination and host-buffer branches.

Scope: only the file-read signalling. Does not disable fast-disk, does not touch
the native library, does not alter any quantized weight path.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_APPLIED = False


def _patch_read_tensor_file_slice_into() -> bool:
    """Wrap comfy.memory_management.read_tensor_file_slice_into.

    A RuntimeError from the native aimdo reader is converted to a False return
    so the caller (cast_to_gathered) falls back to the mmap copy.
    """
    try:
        import comfy.memory_management as mem
    except ImportError:
        return False

    original = getattr(mem, "read_tensor_file_slice_into", None)
    if original is None:
        return False
    if getattr(original, "_hswq_fastdisk_recoverable", False):
        return True

    def read_tensor_file_slice_into_recoverable(tensor, destination, stream=None, destination2=None):
        try:
            return original(tensor, destination, stream=stream, destination2=destination2)
        except RuntimeError as exc:
            logger.warning(
                "[HSWQ FastDisk] native read failed, using mmap fallback: %s", exc
            )
            return False

    read_tensor_file_slice_into_recoverable._hswq_fastdisk_recoverable = True
    read_tensor_file_slice_into_recoverable._hswq_fastdisk_original = original
    mem.read_tensor_file_slice_into = read_tensor_file_slice_into_recoverable
    return True


def apply_comfy_aimdo_fastdisk_guard() -> bool:
    """Install the fast-disk recovery guard. Safe to call repeatedly."""
    global _APPLIED
    if _APPLIED:
        return True

    ok = _patch_read_tensor_file_slice_into()
    if ok:
        _APPLIED = True
        logger.info(
            "[HSWQ FastDisk] v0.38 fast-disk recovery guard armed "
            "(read_tensor_file_slice_into -> mmap fallback on native read failure)"
        )
        return True
    return False