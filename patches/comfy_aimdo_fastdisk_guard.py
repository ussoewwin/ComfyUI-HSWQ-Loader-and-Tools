"""ComfyUI v0.38 fast-disk support: serialise comfy-aimdo native calls.

Root cause (confirmed against the comfy-aimdo native sources and by
measurement on this machine):

comfy-aimdo was written for a single thread. Its device context
(``g_devctx``, src/control.c) is thread-local, the native file reader keeps
ONE completion slot that the previous read must retire before the next one
activates (src/hostbuf-file-reader.c: "active slot N already has a completion
event"), and stream resolution inside the native code uses the CURRENT stream
of the CALLING thread. Every native call also maps/unmaps/evicts VRAM virtual
pages (VBAR/HostBuffer), which must not overlap with kernels still running on
other streams or the pages become illegal (CUDA_ERROR_ILLEGAL_ADDRESS,
sticky result=700 -> Fatal Python error: Aborted).

ComfyUI v0.38 auto-enables fast-disk for every loader (7a0b5eede / e638023d5 /
5ba116a40) and its async offload, parallel load and prefetch drive these
native calls from several threads, breaking all three invariants at once:

  * overlapping reads        -> "hostbuf_file_reader_read failed" cascade
  * page maps under in-flight kernels -> result=700 illegal access
  * failed read leaves the slot dirty -> every later read fails too

Moving the calls to a dedicated owner thread is WRONG here: the native code
resolves the caller's current stream, so the copy lands on the wrong stream
while comfy waits on its own offload stream (observed as a sticky
cuEventCreate/cuStreamIsCapturing result=700), and it breaks CUDA graph
capture (comfy-aimdo malloc graphs, RES4LYF) whose members must be recorded
on the capturing thread.

Fix (make fast-disk work; no mmap fallback on the happy path):

  * every native comfy-aimdo call runs INLINE on the calling thread (so the
    native thread-local context, current-stream resolution and graph capture
    all keep working exactly as the native code expects)
  * a process-wide RLock serialises all native calls against each other
  * a device-wide sync runs before page-mapping operations and after
    device-destination copies - never while a CUDA graph capture is active
    (a sync during capture would invalidate it; those calls run lock-only and
    record into the capture like stock ComfyUI)
  * a failed read resets the native reader slot (cleanup_file_reader) and
    retries once; if it still fails, the error converts to a recoverable
    False at the comfy entry so ComfyUI's mmap copy fallback materialises the
    data instead of cascading into an abort

Disabling fast-disk is NOT done; the native library is NOT replaced; no
quantized weight path is altered.
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_APPLIED = False

# comfy-aimdo native calls are single-thread by design: serialise them all.
_NATIVE_LOCK = threading.RLock()

_MARK = "_hswq_aimdo_serialised"


def _capturing_here() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        fn = getattr(torch.cuda, "is_current_stream_capturing", None)
        if fn is None:
            return False
        return bool(fn())
    except Exception:  # noqa: BLE001
        return False


def _device_sync() -> None:
    """Device-wide wait, skipped while capturing (sync would invalidate)."""
    if _capturing_here():
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass


def _wrap(owner_mod, name, original, pre_sync=False, post_sync_args=None,
          best_effort=False):
    """Run original inline on the caller thread, under the native lock."""
    def routed(*args, **kwargs):
        try:
            with _NATIVE_LOCK:
                if pre_sync:
                    _device_sync()
                result = original(*args, **kwargs)
                if post_sync_args is not None:
                    try:
                        idx, default = post_sync_args
                        allargs = list(args) + [
                            kwargs.get(k) for k in sorted(kwargs)
                        ]
                        val = None
                        if idx < len(args):
                            val = args[idx]
                        else:
                            ordered = sorted(kwargs)
                            off = idx - len(args)
                            if off < len(ordered) and kwargs.get(ordered[off]) is not None:
                                val = kwargs[ordered[off]]
                        if val:
                            _device_sync()
                    except Exception:  # noqa: BLE001
                        pass
                return result
        except Exception as exc:
            if best_effort:
                logger.debug(
                    "[HSWQ FastDisk] best-effort native call %s skipped: %s",
                    name, exc,
                )
                return None
            raise

    setattr(routed, _MARK, True)
    routed.__name__ = getattr(original, "__name__", name)
    routed._hswq_original = original
    return routed


def _patch_host_buffer() -> bool:
    try:
        import comfy_aimdo.host_buffer as hb
    except ImportError:
        return False

    changed = False

    # THE file->VRAM read: serialise, drain afterwards, reset+retry once on a
    # dirty reader slot.
    orig_rfd = getattr(hb, "read_file_to_device", None)
    if orig_rfd is not None and not getattr(orig_rfd, _MARK, False):
        def read_file_to_device_serialised(file_obj, file_offset, size, stream,
                                           device_ptr, device, mark_cold=True):
            with _NATIVE_LOCK:
                _device_sync()
                try:
                    orig_rfd(file_obj, file_offset, size, stream, device_ptr,
                             device, mark_cold)
                except RuntimeError as exc:
                    name = getattr(file_obj, "name", "?")
                    logger.warning(
                        "[HSWQ FastDisk] read failed file=%s off=%s size=%s (%s); "
                        "resetting reader slot and retrying once",
                        name, file_offset, size, exc,
                    )
                    try:
                        hb.cleanup_file_reader()
                    except Exception:  # noqa: BLE001
                        pass
                    _device_sync()
                    orig_rfd(file_obj, file_offset, size, stream, device_ptr,
                             device, mark_cold)
                # wait for the async copy so the slot retires before the next
                # read and before any kernel consumes the destination
                _device_sync()

        setattr(read_file_to_device_serialised, _MARK, True)
        read_file_to_device_serialised._hswq_original = orig_rfd
        hb.read_file_to_device = read_file_to_device_serialised
        changed = True

    orig_cleanup = getattr(hb, "cleanup_file_reader", None)
    if orig_cleanup is not None and not getattr(orig_cleanup, _MARK, False):
        hb.cleanup_file_reader = _wrap(hb, "cleanup_file_reader", orig_cleanup)

    HostBuffer = getattr(hb, "HostBuffer", None)
    if HostBuffer is not None:
        for name in ("__init__", "extend", "get_raw_address", "register",
                     "unregister", "truncate"):
            meth = getattr(HostBuffer, name, None)
            if meth is None or getattr(meth, _MARK, False):
                continue
            setattr(HostBuffer, name,
                    _wrap(HostBuffer, name, meth, pre_sync=(name == "extend")))
            changed = True

        meth_del = getattr(HostBuffer, "__del__", None)
        if meth_del is not None and not getattr(meth_del, _MARK, False):
            setattr(HostBuffer, "__del__",
                    _wrap(HostBuffer, "__del__", meth_del, best_effort=True))
            changed = True

        orig_rfs = getattr(HostBuffer, "read_file_slice", None)
        if orig_rfs is not None and not getattr(orig_rfs, _MARK, False):
            def read_file_slice_serialised(self, file_obj, file_offset, size,
                                           offset=0, stream=0, device_ptr=0,
                                           device=-1):
                with _NATIVE_LOCK:
                    _device_sync()
                    result = orig_rfs(self, file_obj, file_offset, size,
                                      offset=offset, stream=stream,
                                      device_ptr=device_ptr, device=device)
                    if device_ptr:
                        _device_sync()
                    return result

            setattr(read_file_slice_serialised, _MARK, True)
            read_file_slice_serialised._hswq_original = orig_rfs
            HostBuffer.read_file_slice = read_file_slice_serialised
            changed = True

    return changed


def _patch_model_vbar() -> bool:
    try:
        import comfy_aimdo.model_vbar as mvbar
    except ImportError:
        return False

    changed = False

    for name in ("vbar_fault", "vbar_unpin", "vbars_reset_watermark_limits",
                 "vbars_analyze"):
        original = getattr(mvbar, name, None)
        if original is None or getattr(original, _MARK, False):
            continue
        # fault/unpin map & evict pages: wait for in-flight kernels first
        setattr(mvbar, name, _wrap(mvbar, name, original, pre_sync=True))
        changed = True

    VBAR = getattr(mvbar, "ModelVBAR", None)
    if VBAR is not None:
        for name in ("__init__", "prioritize", "deprioritize", "alloc",
                     "fault", "unpin", "loaded_size", "set_watermark_limit",
                     "set_watermark", "free_memory", "get_nr_pages",
                     "get_watermark", "get_residency"):
            original = getattr(VBAR, name, None)
            if original is None or getattr(original, _MARK, False):
                continue
            setattr(VBAR, name, _wrap(VBAR, name, original,
                                      pre_sync=name in ("prioritize", "fault",
                                                        "unpin", "__init__")))
            changed = True

        original = getattr(VBAR, "__del__", None)
        if original is not None and not getattr(original, _MARK, False):
            setattr(VBAR, "__del__",
                    _wrap(VBAR, "__del__", original, best_effort=True))
            changed = True

    return changed


def _patch_vram_buffer() -> bool:
    try:
        import comfy_aimdo.vram_buffer as vb
    except ImportError:
        return False

    VRAM = getattr(vb, "VRAMBuffer", None)
    if VRAM is None:
        return False

    changed = False
    for name in ("__init__", "get", "size"):
        original = getattr(VRAM, name, None)
        if original is None or getattr(original, _MARK, False):
            continue
        setattr(VRAM, name, _wrap(VRAM, name, original, pre_sync=(name == "get")))
        changed = True

    original = getattr(VRAM, "__del__", None)
    if original is not None and not getattr(original, _MARK, False):
        setattr(VRAM, "__del__", _wrap(VRAM, "__del__", original, best_effort=True))
        changed = True
    return changed


def _patch_control() -> bool:
    try:
        import comfy_aimdo.control as control
    except ImportError:
        return False

    changed = False
    for name in ("init_devices", "init_device"):
        original = getattr(control, name, None)
        if original is None or getattr(original, _MARK, False):
            continue
        setattr(control, name, _wrap(control, name, original))
        changed = True
    original = getattr(control, "get_devctx", None)
    if original is not None and not getattr(original, _MARK, False):
        setattr(control, "get_devctx", _wrap(control, "get_devctx", original))
        changed = True
    return changed


def _patch_mem_slice() -> bool:
    """Recoverable at comfy's entry: a failed native read must not abort.

    With serialisation the slot stays clean; if a read still fails, return
    False so cast_to_gathered falls back to the mmap copy (dest_view.copy_).
    """
    try:
        import comfy.memory_management as mem
    except ImportError:
        return False

    orig_mem = getattr(mem, "read_tensor_file_slice_into", None)
    if orig_mem is None or getattr(orig_mem, "_hswq_fastdisk_recoverable", False):
        return bool(orig_mem is not None)

    def read_tensor_file_slice_into_recoverable(tensor, destination, stream=None,
                                                destination2=None):
        with _NATIVE_LOCK:
            try:
                return orig_mem(tensor, destination, stream=stream,
                                destination2=destination2)
            except RuntimeError as exc:
                logger.warning("[HSWQ FastDisk] native read failed: %s", exc)
                return False

    read_tensor_file_slice_into_recoverable._hswq_fastdisk_recoverable = True
    read_tensor_file_slice_into_recoverable._hswq_fastdisk_original = orig_mem
    mem.read_tensor_file_slice_into = read_tensor_file_slice_into_recoverable
    return True


def apply_comfy_aimdo_fastdisk_guard() -> bool:
    """Install fast-disk support (serialised native calls on the caller thread)."""
    global _APPLIED
    if _APPLIED:
        return True

    ok_control = _patch_control()
    ok_hb = _patch_host_buffer()
    ok_vbar = _patch_model_vbar()
    ok_vram = _patch_vram_buffer()
    ok_mem = _patch_mem_slice()

    if ok_control or ok_hb or ok_vbar or ok_vram or ok_mem:
        _APPLIED = True
        logger.info(
            "[HSWQ FastDisk] v0.38 fast-disk support armed "
            "(native calls serialised: control=%s host_buffer=%s model_vbar=%s "
            "vram_buffer=%s mem_slice=%s)",
            ok_control, ok_hb, ok_vbar, ok_vram, ok_mem,
        )
        return True
    return False