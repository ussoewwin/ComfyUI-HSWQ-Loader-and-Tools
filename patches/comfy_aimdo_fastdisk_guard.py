"""ComfyUI v0.38 fast-disk support: one owner thread for all comfy_aimdo native ops.

Root cause (confirmed against the comfy-aimdo native sources and by measurement):

comfy-aimdo keeps its per-device context in a THREAD-LOCAL (``g_devctx``,
src/control.c). The file reader slot table, the VBAR/HostBuffer/VRAM reservation
machinery and the fault path (which calls ``plat_current_stream()``) all resolve
that thread-local context. Consequences on this machine (RTX 5060 Ti, Windows):

  * the first thread that drives the native reader owns its single completion
    slot. A second thread that overlaps a read trips
    "hostbuf_file_reader_retire_active: active slot N already has a completion
    event"; a device copy then fails with CUDA_ERROR_ILLEGAL_ADDRESS
    (result=700) and the process dies with "Fatal Python error: Aborted".
  * allocating/extending a HostBuffer or a ModelVBAR/VRAMBuffer from a thread
    that never bound the context fails outright
    ("HostBuffer.extend failed", "cuMemMap ... invalid argument",
    "VRAM Allocation failed (non OOM)") - measured.
  * routing only the *reads* is not enough: ComfyUI's parallel load / prefetch
    also creates and faults the buffers on worker threads.

So every native comfy-aimdo call - device init, buffer create/extend/free, VBAR
prioritize/fault/unpin and the file reads - must run on ONE thread.

Fix (make fast-disk work; no mmap fallback on the happy path):

  * a dedicated owner thread performs control.init_devices() plus a warm-up read
    at prestartup, so it binds the thread-local context before any other thread
    touches comfy-aimdo.
  * every native comfy-aimdo entry point is marshalled onto that thread. Calls
    already running on the owner execute inline (thread-id bypass) so wrapped
    functions that call wrapped functions never re-enter the queue.
  * the caller's stream is passed through unchanged (CUDA-graph capture safe);
    no forced stream change and no forced global sync.
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


_INFERENCE_MODE = None


def _run_in_caller_modes(fn, args, kwargs):
    """Execute a routed native comfy-aimdo call with comfy-caller semantics.

    PyTorch inference/grad state is thread-local. ComfyUI creates weight
    tensors under torch.inference_mode() on its worker threads; in-place
    updates on those tensors (the copy inside read paths) are only allowed
    while inference mode is active on the executing thread. Enter
    inference_mode on the owner thread so those copies succeed. New tensors
    created during the call are staging buffers only, so inference tensors
    here are safe.
    """
    global _INFERENCE_MODE
    if _INFERENCE_MODE is None:
        try:
            import torch

            _INFERENCE_MODE = getattr(torch, "inference_mode", None)
            if _INFERENCE_MODE is False:
                _INFERENCE_MODE = None
        except Exception:
            _INFERENCE_MODE = False
    if _INFERENCE_MODE:
        with _INFERENCE_MODE():
            return fn(*args, **kwargs)
    return fn(*args, **kwargs)


def _drain_device_copy(device_ptr):
    """Wait for the async file->VRAM copy to finish before the next read.

    The native reader keeps ONE slot whose completion event belongs to the
    previous read; issuing a new read while the prior copy is still in flight
    trips "active slot N already has a completion event" and then an illegal
    device access. Blocking here makes the serialisation correct, not just
    interleaving-safe.
    """
    if not device_ptr:
        return
    try:
        import torch

        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass


class _Owner:
    """Single thread that owns every native comfy_aimdo call."""

    def __init__(self):
        self._queue = queue.Queue()
        self._ready = threading.Event()
        self._warmup_event = threading.Event()
        self._warmup_result = None
        self._tid = None
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
        self._tid = threading.get_ident()
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
                box["result"] = _run_in_caller_modes(fn, args, kwargs)
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

    def on_owner(self):
        return self._tid is not None and threading.get_ident() == self._tid

    def call(self, fn, *args, timeout=None, **kwargs):
        """Run fn on the owner thread unless we are already on it."""
        if self.on_owner():
            return fn(*args, **kwargs)
        box = {"event": threading.Event(), "result": None, "error": None}
        self._queue.put(("call", fn, args, kwargs, box))
        if not box["event"].wait(timeout):
            raise RuntimeError("hswq-aimdo-owner timeout")
        if box["error"] is not None:
            raise box["error"]
        return box["result"]

    def call_best_effort(self, fn, *args, **kwargs):
        """Like call() but falls back to inline execution when the owner is
        gone, and never raises (used for __del__ finalizers)."""
        try:
            if self.on_owner() or not self._thread.is_alive():
                return fn(*args, **kwargs)
            return self.call(fn, *args, timeout=30, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[HSWQ FastDisk] best-effort native call skipped: %s", exc)
            return None

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


_MARK = "_hswq_aimdo_owner_routed"


def _route_methods(owner, cls, method_names, best_effort_names=()):
    """Wrap the named methods of cls so they run on the owner thread."""
    changed = False
    for name in method_names:
        original = getattr(cls, name, None)
        if original is None or getattr(original, _MARK, False):
            continue

        def make(orig=original):
            def routed(self, *args, **kwargs):
                return owner.call(orig, self, *args, **kwargs)

            setattr(routed, _MARK, True)
            routed.__name__ = name
            routed._hswq_original = orig
            return routed

        setattr(cls, name, make())
        changed = True

    for name in best_effort_names:
        original = getattr(cls, name, None)
        if original is None or getattr(original, _MARK, False):
            continue

        def make(orig=original):
            def routed(self, *args, **kwargs):
                return owner.call_best_effort(orig, self, *args, **kwargs)

            setattr(routed, _MARK, True)
            routed.__name__ = name
            routed._hswq_original = orig
            return routed

        setattr(cls, name, make())
        changed = True

    return changed


def _route_functions(owner, module, func_names):
    """Wrap module-level callables so they run on the owner thread."""
    changed = False
    for name in func_names:
        original = getattr(module, name, None)
        if original is None or not callable(original) or getattr(original, _MARK, False):
            continue

        def make(orig=original):
            def routed(*args, **kwargs):
                return owner.call(orig, *args, **kwargs)

            setattr(routed, _MARK, True)
            routed.__name__ = name
            routed._hswq_original = orig
            return routed

        setattr(module, name, make())
        changed = True
    return changed


def _patch_control(owner) -> bool:
    try:
        import comfy_aimdo.control as control
    except ImportError:
        return False

    changed = False

    # init_devices / init_device: run on the owner AND trigger warm-up.
    for name in ("init_devices", "init_device"):
        original = getattr(control, name, None)
        if original is None or getattr(original, _MARK, False):
            continue

        def make(orig=original):
            def routed(*args, **kwargs):
                return owner.run_init(orig, *args, **kwargs)

            setattr(routed, _MARK, True)
            routed.__name__ = name
            routed._hswq_original = orig
            return routed

        setattr(control, name, make())
        changed = True

    # get_devctx returns the thread-local context; it must be read on the owner.
    changed |= _route_functions(owner, control, ("get_devctx",))
    return changed


def _patch_host_buffer(owner) -> bool:
    try:
        import comfy_aimdo.host_buffer as hb
    except ImportError:
        return False

    changed = False

    orig_read_file_to_device = getattr(hb, "read_file_to_device", None)
    if orig_read_file_to_device is not None and not getattr(
        orig_read_file_to_device, _MARK, False
    ):
        def read_file_to_device_routed(file_obj, file_offset, size, stream, device_ptr,
                                       device, mark_cold=True):
            def _do():
                result = orig_read_file_to_device(file_obj, file_offset, size, stream,
                                                  device_ptr, device, mark_cold)
                # drain async file->VRAM copy before returning so the next read
                # cannot hit the still-occupied reader slot
                _drain_device_copy(device_ptr)
                return result

            return owner.call(_do)

        setattr(read_file_to_device_routed, _MARK, True)
        read_file_to_device_routed._hswq_original = orig_read_file_to_device
        hb.read_file_to_device = read_file_to_device_routed
        changed = True

    changed |= _route_functions(
        owner, hb, ("cleanup_file_reader",)
    )

    HostBuffer = getattr(hb, "HostBuffer", None)
    if HostBuffer is not None:
        changed |= _route_methods(
            owner, HostBuffer,
            ("__init__", "extend", "get_raw_address",
             "register", "unregister", "truncate"),
            best_effort_names=("__del__",),
        )

        orig_slice = getattr(HostBuffer, "read_file_slice", None)
        if orig_slice is not None and not getattr(orig_slice, _MARK, False):
            def read_file_slice_routed(self, file_obj, file_offset, size, offset=0,
                                       stream=0, device_ptr=0, device=-1):
                def _do():
                    result = orig_slice(self, file_obj, file_offset, size, offset=offset,
                                       stream=stream, device_ptr=device_ptr, device=device)
                    _drain_device_copy(device_ptr)
                    return result

                return owner.call(_do)

            setattr(read_file_slice_routed, _MARK, True)
            read_file_slice_routed._hswq_original = orig_slice
            HostBuffer.read_file_slice = read_file_slice_routed
            changed = True
    return changed


def _patch_model_vbar(owner) -> bool:
    try:
        import comfy_aimdo.model_vbar as mvbar
    except ImportError:
        return False

    changed = False

    changed |= _route_functions(
        owner, mvbar,
        ("vbar_fault", "vbar_unpin", "vbars_reset_watermark_limits", "vbars_analyze"),
    )

    VBAR = getattr(mvbar, "ModelVBAR", None)
    if VBAR is not None:
        # alloc mutates self.offset, so route it too (keeps multi-model
        # workers from handing out overlapping ranges).
        # fault / unpin are called by the routed module-level vbar_fault /
        # vbar_unpin; wrapping them too is safe because the thread-id bypass
        # runs them inline once we are on the owner.
        changed |= _route_methods(
            owner, VBAR,
            ("__init__", "prioritize", "deprioritize", "alloc", "fault", "unpin",
             "loaded_size", "set_watermark_limit", "set_watermark",
             "free_memory", "get_nr_pages", "get_watermark", "get_residency"),
            best_effort_names=("__del__",),
        )
    return changed


def _patch_vram_buffer(owner) -> bool:
    try:
        import comfy_aimdo.vram_buffer as vb
    except ImportError:
        return False

    VRAM = getattr(vb, "VRAMBuffer", None)
    if VRAM is None:
        return False

    # VRAMBuffer.get grows the reservation (native) so it must run on the owner.
    return _route_methods(owner, VRAM, ("__init__", "get", "size"),
                          best_effort_names=("__del__",))


def _wrap_mem_slice(owner) -> bool:
    """Recoverable + owner-routed comfy.memory_management.read_tensor_file_slice_into.

    This is the entry point ComfyUI's weight cast actually calls
    (comfy.model_management.cast_to_gathered -> read_tensor_file_slice_into).
    It can run on prefetch / parallel-load worker threads, so it must be
    marshalled onto the owner thread like the other native reads; otherwise a
    non-bound thread drives the single reader slot.
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
        def _do():
            try:
                return original(tensor, destination, stream=stream, destination2=destination2)
            except RuntimeError as exc:
                logger.warning("[HSWQ FastDisk] native read failed: %s", exc)
                return False

        return owner.call(_do)

    read_tensor_file_slice_into_recoverable._hswq_fastdisk_recoverable = True
    read_tensor_file_slice_into_recoverable._hswq_fastdisk_original = original
    mem.read_tensor_file_slice_into = read_tensor_file_slice_into_recoverable
    return True


def apply_comfy_aimdo_fastdisk_guard() -> bool:
    """Install fast-disk support (single owner thread for all aimdo native ops)."""
    global _APPLIED
    if _APPLIED:
        return True

    owner = get_owner()
    ok_control = _patch_control(owner)
    ok_hb = _patch_host_buffer(owner)
    ok_vbar = _patch_model_vbar(owner)
    ok_vram = _patch_vram_buffer(owner)
    ok_mem = _wrap_mem_slice(owner)

    if ok_control or ok_hb or ok_vbar or ok_vram or ok_mem:
        _APPLIED = True
        logger.info(
            "[HSWQ FastDisk] v0.38 fast-disk support armed "
            "(owner-thread: control=%s host_buffer=%s model_vbar=%s vram_buffer=%s "
            "mem_slice=%s)",
            ok_control, ok_hb, ok_vbar, ok_vram, ok_mem,
        )
        return True
    return False
