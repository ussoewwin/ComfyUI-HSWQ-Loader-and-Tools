"""ComfyUI v0.38 fast-disk support: single native aimdo owner thread.

Root cause (confirmed against the native source and by direct measurement):

comfy_aimdo keeps its device context in a THREAD-LOCAL (``g_devctx``,
src/control.c) and the file reader plus the VBAR slot table live inside it:

  * the first thread to drive a native aimdo op wins the binding; a second
    thread then fails - even with a lock serialising the calls - because its
    thread-local context has a different slot table ("hostbuf_file_reader_read
    failed", "active slot N already has a completion event") and VBAR
    (cuMemMap) operations report "invalid argument".
  * issuing every native aimdo op from ONE thread succeeds (measured 600/600 -
    1200/1200, including several streams).

v0.38 enables fast-disk for every loader and ComfyUI's parallel load / prefetch
drives reads and VBAR faults from several threads, so the non-bound threads
fail. During sampling that surfaces as a CUDA illegal memory access and
"Fatal Python error: Aborted".

Fix (make fast-disk work; no mmap fallback on the happy path):

  * a dedicated owner thread performs control.init_devices() plus a warm-up read
    at prestartup, so it binds aimdo before any other thread touches it.
  * every native aimdo op - VBAR alloc/fault/unpin/prioritize and the three
    file-read entry points - is marshalled onto that thread.
  * the read is issued with the CALLER's stream passed through unchanged, so it
    stays correct under CUDA graph capture; no forced synchronise.
  * a genuine read failure is converted to a recoverable result instead of an
    unrecoverable C-level abort.

Disabling fast-disk is NOT done; the native library is NOT replaced; no
quantized weight path is altered.
"""
from __future__ import annotations

import logging
import queue
import threading

logger = logging.getLogger(__name__)

_APPLIED = False

_OWNER = None
_OWNER_LOCK = threading.Lock()


class _Owner:
    """Single thread that owns every native comfy_aimdo call."""

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
            target=self._loop, name="hswq-aimdo-owner", daemon=True
        )
        self._thread.start()
        self._ready.wait()

    def _loop(self):
        try:
            import torch

            torch.cuda.set_device(0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[HSWQ FastDisk] owner thread CUDA init failed: %s", exc)
        self._ready.set()

        while True:
            job = self._queue.get()
            if job is None:
                return
            kind = job[0]
            fn, args, kwargs, box = job[1], job[2], job[3], job[4]
            try:
                box["result"] = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001
                box["error"] = exc
            finally:
                if kind == "init":
                    self._warmup_result = self._warmup()
                    self._warmup_event.set()
                box["event"].set()

    def _warmup(self):
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

    def run(self, fn, *args, timeout=None, **kwargs):
        box = {"event": threading.Event(), "result": None, "error": None}
        self._queue.put(("call", fn, args, kwargs, box))
        if not box["event"].wait(timeout):
            raise RuntimeError("hswq-aimdo-owner timeout")
        if box["error"] is not None:
            raise box["error"]
        return box["result"]

    def run_init(self, fn, *args, timeout=120, **kwargs):
        box = {"event": threading.Event(), "result": None, "error": None}
        self._queue.put(("init", fn, args, kwargs, box))
        if not box["event"].wait(timeout):
            raise RuntimeError("hswq-aimdo-owner init timeout")
        if box["error"] is not None:
            raise box["error"]
        return box["result"]


def get_owner():
    global _OWNER
    if _OWNER is not None:
        return _OWNER
    with _OWNER_LOCK:
        if _OWNER is None:
            _OWNER = _Owner()
    return _OWNER


def _wrap_control_init(owner) -> bool:
    try:
        import comfy_aimdo.control as control
    except ImportError:
        return False

    changed = False
    for name in ("init_devices", "init_device"):
        original = getattr(control, name, None)
        if original is None or getattr(original, "_hswq_owner_routed", False):
            continue

        def make(orig=original):
            def routed(*args, **kwargs):
                return owner.run_init(orig, *args, **kwargs)

            routed._hswq_owner_routed = True
            routed._hswq_owner_original = orig
            return routed

        setattr(control, name, make())
        changed = True
    return changed


def _wrap_vbar(owner) -> bool:
    """Route VBAR operations onto the owner thread."""
    try:
        import comfy_aimdo.model_vbar as mvbar
    except ImportError:
        return False

    changed = False

    for name in ("vbar_fault", "vbar_unpin"):
        original = getattr(mvbar, name, None)
        if original is None or getattr(original, "_hswq_owner_routed", False):
            continue

        def make(orig=original):
            def routed(*args, **kwargs):
                return owner.run(orig, *args, **kwargs)

            routed._hswq_owner_routed = True
            routed._hswq_owner_original = orig
            return routed

        setattr(mvbar, name, make())
        changed = True

    VBAR = getattr(mvbar, "ModelVBAR", None)
    if VBAR is not None:
        # Do NOT wrap ModelVBAR.fault / ModelVBAR.unpin: the module-level
        # vbar_fault / vbar_unpin (wrapped above) call those methods, so
        # wrapping both would re-enter the owner queue from the owner thread.
        for name in ("alloc", "prioritize", "free_memory", "loaded_size",
                     "set_watermark", "set_watermark_limit"):
            original = getattr(VBAR, name, None)
            if original is None or getattr(original, "_hswq_owner_routed", False):
                continue

            def make(orig=original):
                def routed(self, *args, **kwargs):
                    return owner.run(orig, self, *args, **kwargs)

                routed._hswq_owner_routed = True
                routed._hswq_owner_original = orig
                return routed

            setattr(VBAR, name, make())
            changed = True

    return changed


def _wrap_reads(owner) -> bool:
    """Route the native read entry points onto the owner thread.

    The caller's stream is passed through unchanged (CUDA-graph safe); the owner
    thread runs the native read and returns once the copy is enqueued.
    """
    try:
        import comfy_aimdo.host_buffer as hb
    except ImportError:
        return False

    changed = False

    orig_read_file_to_device = getattr(hb, "read_file_to_device", None)
    if orig_read_file_to_device is not None and not getattr(
        orig_read_file_to_device, "_hswq_owner_routed", False
    ):
        def read_file_to_device_routed(file_obj, file_offset, size, stream, device_ptr,
                                       device, mark_cold=True):
            return owner.run(orig_read_file_to_device, file_obj, file_offset, size,
                             stream, device_ptr, device, mark_cold)

        read_file_to_device_routed._hswq_owner_routed = True
        read_file_to_device_routed._hswq_owner_original = orig_read_file_to_device
        hb.read_file_to_device = read_file_to_device_routed
        changed = True

    HostBuffer = getattr(hb, "HostBuffer", None)
    if HostBuffer is not None:
        orig_read_file_slice = getattr(HostBuffer, "read_file_slice", None)
        if orig_read_file_slice is not None and not getattr(
            orig_read_file_slice, "_hswq_owner_routed", False
        ):
            def read_file_slice_routed(self, file_obj, file_offset, size, offset=0,
                                       stream=0, device_ptr=0, device=-1):
                device = -1 if device is None else int(device)
                return owner.run(orig_read_file_slice, self, file_obj, file_offset,
                                 size, offset=offset, stream=stream,
                                 device_ptr=device_ptr, device=device)

            read_file_slice_routed._hswq_owner_routed = True
            read_file_slice_routed._hswq_owner_original = orig_read_file_slice
            HostBuffer.read_file_slice = read_file_slice_routed
            changed = True

    return changed


def _wrap_mem_slice() -> bool:
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
    """Install fast-disk support (single native aimdo owner thread). Idempotent."""
    global _APPLIED
    if _APPLIED:
        return True

    owner = get_owner()
    ok_init = _wrap_control_init(owner)
    ok_vbar = _wrap_vbar(owner)
    ok_reads = _wrap_reads(owner)
    ok_mem = _wrap_mem_slice()

    if ok_init or ok_vbar or ok_reads or ok_mem:
        _APPLIED = True
        logger.info(
            "[HSWQ FastDisk] v0.38 fast-disk support armed "
            "(owner-thread init=%s vbar=%s reads=%s, safe slice=%s)",
            ok_init, ok_vbar, ok_reads, ok_mem,
        )
        return True
    return False