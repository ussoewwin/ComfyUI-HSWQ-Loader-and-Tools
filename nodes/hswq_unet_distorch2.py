"""HSWQ UNet Loader (DisTorch2).

Upstream DisTorch2 wrapper (same widgets/behaviour as ComfyUI-MultiGPU's
``UNETLoaderDisTorch2MultiGPU``) applied to the HSWQ UNet loader registered in
``NODE_CLASS_MAPPINGS``, plus two fixes on the load/placement path:

1. Placement: DisTorch assigns blocks inside the patched
   ``ModelPatcher.partially_load``. Under DynamicVRAM the loaded model would be a
   ``ModelPatcherDynamic``, whose own ``partially_load``/``load`` win, so that
   placement never runs. ``clone(disable_dynamic=True)`` (same inner model) puts
   the load back on the standard path where the placement runs.

2. Host pinning: on that path ComfyUI pins offloaded weights via
   ``comfy.model_management.pin_memory()`` -> ``cudaHostRegister``. ComfyUI's own
   guard (``tensor.is_pinned()``) cannot see registrations made outside the torch
   allocator (e.g. comfy-aimdo's host buffers), so the call returns
   ``cudaErrorHostMemoryAlreadyRegistered`` (712) for every such weight and
   ComfyUI logs ``Pin error.`` for each one. That is the already-pinned case its
   own comment describes; this patch handles the error code properly instead of
   spamming warnings. Genuine failures are still reported (with the real CUDA
   error string and the size), no message is hidden.
"""

import logging

import torch

from ..wrappers import override_class_with_distorch_safetensor_v2

logger = logging.getLogger("SDXL")

_CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712
_installed = False
_reported = set()


def _real_error_string(rc):
    try:
        res = torch.cuda.cudart().cudaGetErrorString(rc)
        return res[1] if isinstance(res, tuple) else str(res)
    except Exception:
        return "rc=%s" % rc


def _install_pin_memory_fix():
    """Make pin_memory handle the 'already registered' case instead of warning per weight."""
    global _installed
    if _installed:
        return True
    try:
        import comfy.memory_management
        import comfy.model_management as mm

        original = mm.pin_memory

        def pin_memory_fixed(tensor):
            if mm.MAX_PINNED_MEMORY <= 0:
                return False
            if type(tensor).__name__ not in mm.PINNING_ALLOWED_TYPES:
                return False
            if not mm.is_device_cpu(tensor.device):
                return False
            if tensor.is_pinned():
                return False
            if not tensor.is_contiguous():
                return False
            size = tensor.nbytes
            comfy.memory_management.extra_ram_release(comfy.memory_management.RAM_CACHE_HEADROOM)
            mm.ensure_pin_registerable(size)
            ptr = tensor.data_ptr()
            if ptr == 0:
                return False
            rc = int(torch.cuda.cudart().cudaHostRegister(ptr, size, 1))
            if rc == 0:
                mm.PINNED_MEMORY[ptr] = size
                mm.TOTAL_PINNED_MEMORY += size
                return True
            if rc == _CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
                # Already host-registered by another subsystem (comfy-aimdo host buffers).
                # The memory is pinnable; do not track it (so it is never unregistered
                # here) and do not warn - ComfyUI's is_pinned() guard cannot see this.
                mm.discard_cuda_async_error()
                key = "already"
                if key not in _reported:
                    _reported.add(key)
                    logger.info(
                        "[HSWQ] host-pin: weights are already host-registered (cudaErrorHostMemoryAlreadyRegistered); "
                        "skipping re-registration (normal with comfy-aimdo host buffers)"
                    )
                return False
            mm.discard_cuda_async_error()
            key = _real_error_string(rc)
            if key not in _reported:
                _reported.add(key)
                logger.warning(
                    "[HSWQ] host-pin failed: %s (size=%d bytes, rc=%s) - pinning is optional, continuing",
                    key, size, rc,
                )
            return False

        pin_memory_fixed._hswq_orig = original
        mm.pin_memory = pin_memory_fixed
        _installed = True
        return True
    except Exception:
        logger.exception("[HSWQ] could not install the host-pin fix")
        return False


_install_pin_memory_fix()


def build_distorch2_unet_loader(base_cls):
    """Return the DisTorch2 override class for ``base_cls`` (the HSWQ UNet loader)."""
    wrapped = override_class_with_distorch_safetensor_v2(base_cls)

    class HSWQUNetLoaderDisTorch2(wrapped):
        """DisTorch2 wrapper; forces the placement-capable patcher for the load."""

        def override(self, *args, **kwargs):
            out = super().override(*args, **kwargs)
            try:
                patcher = out[0]
                if patcher is not None and getattr(patcher, "is_dynamic", None) and patcher.is_dynamic():
                    patcher = patcher.clone(disable_dynamic=True)
                    logger.info(
                        "[HSWQ DisTorch2] legacy ModelPatcher for this load: DisTorch block "
                        "placement / GPU-addressable offload is otherwise skipped by DynamicVRAM"
                    )
                    return (patcher,) + tuple(out[1:])
            except Exception:
                logger.exception("[HSWQ DisTorch2] legacy-patcher switch failed; keeping the dynamic patcher")
            return out

    return HSWQUNetLoaderDisTorch2
