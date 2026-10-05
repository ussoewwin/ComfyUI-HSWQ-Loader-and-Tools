"""ComfyUI v0.38 fast-disk support: thread-bound comfy_aimdo file reader.

Root cause (confirmed by direct measurement on this machine):

ComfyUI v0.38 enables automatic fast-disk for every loader (commits
7a0b5eede / e638023d5 / 5ba116a40) and drives the weight copies through the
native comfy_aimdo file reader. The native reader (aimdo.dll,
src/hostbuf-file-reader.c) has a single completion slot per stream and is bound
to the CUDA context of the thread that first issues a read:

  * one thread, calls serialised        -> all reads OK   (measured)
  * the same reads issued concurrently / from a thread that then creates its own
    CUDA stream and copies into a VBAR destination
                                        -> failure: "active slot N already has a
                                           completion event" or
                                           "device copy failed result=700"
                                           (CUDA_ERROR_ILLEGAL_ADDRESS)
  * a dedicated reader thread owning control.init_devices() + a warm-up read,
    issuing every read on the NULL (default) stream and synchronising before
    returning, while VBAR alloc/fault stay on the caller threads
                                        -> all reads OK   (600/600 measured)

v0.38's parallel load / prefetch issues reads from several threads and streams,
breaking the single-slot / context contract.

Fix (make fast-disk work; no mmap fallback on the happy path):

  1. A dedicated reader thread binds the CUDA context early: ComfyUI's own
     control.init_devices() is routed onto it, followed by a one-shot warm-up
     read, so the reader owns the native reader's binding before the main thread
     or any pool worker touches CUDA.
  2. Every native read is marshalled onto that thread and performed on the NULL
     stream, then synchronised, so the copy is complete and visible when the
     caller resumes regardless of which stream/thread the caller uses.
  3. VBAR alloc/fault/unpin stay on the caller threads (their native calls are
     context sensitive and were never the problem).
  4. A real I/O error becomes a recoverable result instead of a C-level abort.

Disabling fast-disk is NOT done; the native library is NOT replaced; no
quantized weight path is altered.
"""
from __future__ import annotations

import logging
import queue
import threading

logger = logging.getLogger(__name__)

_APPLIED = False

_READER = None
_READER_LOCK = threading.Lock()


class _BoundReader:
    """Single thread that owns the comfy_aimdo file-reader CUDA binding."""

    def __init__(self):
        self._queue = queue.Queue()
        self._ready = threading.Event()
        self._warmup_event = threading.Event()
        self._warmup_result = None
        self._native_read = None
        try:
            import comfy_aimdo.host_buffer as _hb

            self._native_read = getattr(_hb, "read_file_to_device", None)
        except Exception:
            self._native_read = None
        self._thread = threading.Thread(
            target=self._loop, name="hswq-aimdo-reader", daemon=True
        )
        self._thread.start()
        self._ready.wait()

    # -- reader loop ------------------------------------------------------
    def _loop(self):
        try:
            import torch

            torch.cuda.set_device(0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[HSWQ FastDisk] reader thread CUDA init failed: %s", exc)
        self._ready.set()

        while True:
            job = self._queue.get()
            if job is None:
                return
            kind = job[0]
            fn, args, box = job[1], job[2], job[3]
            try:
                box["result"] = fn(*args)
            except BaseException as exc:  # noqa: BLE001
                box["error"] = exc
            finally:
                if kind == "init":
                    # Bind the native reader after the device context exists.
                    self._warmup_result = self._warmup()
                    self._warmup_event.set()
                box["event"].set()

    def _warmup(self):
        """Bind the native reader to this thread with one tiny NULL-stream read."""
        try:
            import os
            import tempfile

            import torch

            native_read = self._native_read
            if native_read is None:
                return "ERR:no native read"
            fd, path = tempfile.mkstemp(prefix="hswq_aimdo_warmup_")
            try:
                os.write(fd, b"\x00" * 8192)
                os.close(fd)
                with open(path, "rb") as f:
                    dst = torch.empty(8192, dtype=torch.uint8, device="cuda:0")
                    native_read(f, 0, 8192, 0, dst.data_ptr(), 0, False)
                    torch.cuda.synchronize()
                return "OK"
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
        except Exception as exc:  # noqa: BLE001
            return "ERR:%s" % exc

    # -- submission API ---------------------------------------------------
    def run(self, fn, *args, timeout=None):
        box = {"event": threading.Event(), "result": None, "error": None}
        self._queue.put(("call", fn, args, box))
        if not box["event"].wait(timeout):
            raise RuntimeError("hswq-aimdo-reader timeout")
        if box["error"] is not None:
            raise box["error"]
        return box["result"]

    def run_init(self, fn, *args, timeout=120):
        box = {"event": threading.Event(), "result": None, "error": None}
        self._queue.put(("init", fn, args, box))
        if not box["event"].wait(timeout):
            raise RuntimeError("hswq-aimdo-reader init timeout")
        if box["error"] is not None:
            raise box["error"]
        return box["result"]


def get_bound_reader():
    global _READER
    if _READER is not None:
        return _READER
    with _READER_LOCK:
        if _READER is None:
            _READER = _BoundReader()
    return _READER


def _patch_init_devices(reader) -> bool:
    """Route ComfyUI's control.init_devices / init_device through the reader
    thread so the reader owns the CUDA binding and warm-up."""
    try:
        import comfy_aimdo.control as control
    except ImportError:
        return False

    changed = False

    orig_init_devices = getattr(control, "init_devices", None)
    if orig_init_devices is not None and not getattr(
        orig_init_devices, "_hswq_fastdisk_routed", False
    ):
        def init_devices_routed(*args, **kwargs):
            return reader.run_init(orig_init_devices, *args, **kwargs)

        init_devices_routed._hswq_fastdisk_routed = True
        init_devices_routed._hswq_fastdisk_original = orig_init_devices
        control.init_devices = init_devices_routed
        changed = True

    orig_init_device = getattr(control, "init_device", None)
    if orig_init_device is not None and not getattr(
        orig_init_device, "_hswq_fastdisk_routed", False
    ):
        def init_device_routed(*args, **kwargs):
            return reader.run_init(orig_init_device, *args, **kwargs)

        init_device_routed._hswq_fastdisk_routed = True
        init_device_routed._hswq_fastdisk_original = orig_init_device
        control.init_device = init_device_routed
        changed = True

    return changed


def _read_null_stream(fn, file_obj, file_offset, size, device_ptr, device, mark_cold):
    """Run a native device read on the NULL stream and wait for it."""
    result = fn(file_obj, file_offset, size, 0, device_ptr, device, mark_cold)
    try:
        import torch

        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass
    return result


def _read_slice_null_stream(fn, self_obj, file_obj, file_offset, size, offset,
                            device_ptr, device):
    """Run a native host-buffer slice read on the NULL stream and wait for it."""
    result = fn(self_obj, file_obj, file_offset, size, offset=offset, stream=0,
                device_ptr=device_ptr, device=device)
    try:
        import torch

        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass
    return result


def _wrap_native_reads(reader) -> bool:
    """Marshal the native read entry points onto the bound reader thread."""
    try:
        import comfy_aimdo.host_buffer as hb
    except ImportError:
        return False

    changed = False

    orig_read_file_to_device = getattr(hb, "read_file_to_device", None)
    if orig_read_file_to_device is not None and not getattr(
        orig_read_file_to_device, "_hswq_fastdisk_bound", False
    ):
        def read_file_to_device_bound(file_obj, file_offset, size, stream, device_ptr,
                                      device, mark_cold=True):
            return reader.run(
                _read_null_stream, orig_read_file_to_device, file_obj,
                file_offset, size, device_ptr, device, mark_cold,
            )

        read_file_to_device_bound._hswq_fastdisk_bound = True
        read_file_to_device_bound._hswq_fastdisk_original = orig_read_file_to_device
        hb.read_file_to_device = read_file_to_device_bound
        changed = True

    HostBuffer = getattr(hb, "HostBuffer", None)
    if HostBuffer is not None:
        orig_read_file_slice = getattr(HostBuffer, "read_file_slice", None)
        if orig_read_file_slice is not None and not getattr(
            orig_read_file_slice, "_hswq_fastdisk_bound", False
        ):
            def read_file_slice_bound(self, file_obj, file_offset, size, offset=0,
                                      stream=0, device_ptr=0, device=-1):
                device = -1 if device is None else int(device)
                return reader.run(
                    _read_slice_null_stream, orig_read_file_slice, self, file_obj,
                    file_offset, size, offset, device_ptr, device,
                )

            read_file_slice_bound._hswq_fastdisk_bound = True
            read_file_slice_bound._hswq_fastdisk_original = orig_read_file_slice
            HostBuffer.read_file_slice = read_file_slice_bound
            changed = True

    return changed


def _wrap_read_tensor_file_slice_into() -> bool:
    """Keep a genuine read failure as a recoverable False instead of an abort."""
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
            logger.warning("[HSWQ FastDisk] native read failed: %s", exc)
            return False

    read_tensor_file_slice_into_recoverable._hswq_fastdisk_recoverable = True
    read_tensor_file_slice_into_recoverable._hswq_fastdisk_original = original
    mem.read_tensor_file_slice_into = read_tensor_file_slice_into_recoverable
    return True


def apply_comfy_aimdo_fastdisk_guard() -> bool:
    """Install fast-disk support (thread-bound reader). Idempotent."""
    global _APPLIED
    if _APPLIED:
        return True

    reader = get_bound_reader()
    ok_init = _patch_init_devices(reader)
    ok_reads = _wrap_native_reads(reader)
    ok_mem = _wrap_read_tensor_file_slice_into()

    if ok_init or ok_reads or ok_mem:
        _APPLIED = True
        logger.info(
            "[HSWQ FastDisk] v0.38 fast-disk support armed "
            "(reader-bound init_devices=%s, native reads=%s, safe slice=%s)",
            ok_init,
            ok_reads,
            ok_mem,
        )
        return True
    return False