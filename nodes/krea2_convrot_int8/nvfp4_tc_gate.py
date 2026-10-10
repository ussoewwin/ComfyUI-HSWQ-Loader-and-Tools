"""NVFP4 TensorCore availability gate (shared by addmm patch + TC forward).

cuBLAS NVFP4 GEMM needs compute capability >= 10.0 (Blackwell). Cloud hosts are
often Ada / Hopper / Ampere — every ``scaled_mm_nvfp4`` then raises
``CUBLAS_STATUS_NOT_SUPPORTED`` and kitchen / addmm log WARNING per Linear.

This module:
  1) probes CC once
  2) after first NOT_SUPPORTED (or CC < 10.0), disables further TC attempts
  3) emits a single clear line; mutes kitchen nvfp4 WARNING spam

Never edits ComfyUI-master.
"""
from __future__ import annotations

import logging

_PROBED = False
_TC_OK: bool | None = None
_DISABLED = False
_WARNED = False
_DISABLE_REASON = ""

_KITCHEN_NVFP4_LOG = "comfy_kitchen.tensor.nvfp4"
_ADDMM_LOG = "nvfp4.nvfp4_addmm_patch"
_FORWARD_LOG = "nvfp4.nvfp4_forward"


def _mute_nvfp4_warning_spam() -> None:
    for name in (_KITCHEN_NVFP4_LOG, _ADDMM_LOG, _FORWARD_LOG):
        logging.getLogger(name).setLevel(logging.ERROR)


def probe_nvfp4_tc_support(device_index: int = 0) -> bool:
    """Return True if GPU CC is Blackwell NVFP4-TC capable (CC >= 10.0: RTX 5050-5090 / B200)."""
    global _PROBED, _TC_OK
    if _PROBED and _TC_OK is not None:
        return bool(_TC_OK)
    _PROBED = True
    try:
        import torch

        if not torch.cuda.is_available():
            _TC_OK = False
            return False
        major, minor = torch.cuda.get_device_capability(device_index)
        # All Blackwell family GPUs (RTX 5050..5090, B200) have CC >= 10.0
        _TC_OK = (int(major), int(minor)) >= (10, 0)
        return bool(_TC_OK)
    except Exception:
        _TC_OK = False
        return False


def nvfp4_tc_enabled() -> bool:
    if _DISABLED:
        return False
    return probe_nvfp4_tc_support()


def disable_nvfp4_tc(reason: str, *, announce: bool = True) -> None:
    """Permanent disable for this process; warn once then mute spam loggers."""
    global _DISABLED, _WARNED, _DISABLE_REASON
    _DISABLED = True
    _DISABLE_REASON = str(reason) if reason else "unknown"
    _mute_nvfp4_warning_spam()
    if announce and not _WARNED:
        _WARNED = True
        name = "?"
        cc = "?"
        try:
            import torch

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                major, minor = torch.cuda.get_device_capability(0)
                cc = f"{major}.{minor}"
        except Exception:
            pass
        print(
            f"[HSWQ NVFP4] TensorCore scaled_mm disabled for this run "
            f"(GPU={name}, CC={cc}): {_DISABLE_REASON}. "
            f"Using dequant mm; further CUBLAS/kitchen WARNINGs suppressed.",
            flush=True,
        )


def note_scaled_mm_failure(exc: BaseException) -> bool:
    """If failure is permanent (NOT_SUPPORTED / unsupported), disable TC.

    Returns True if TC is now disabled (caller should dequant without retry storm).
    """
    msg = str(exc)
    permanent = (
        "CUBLAS_STATUS_NOT_SUPPORTED" in msg
        or "NOT_SUPPORTED" in msg
        or "not supported" in msg.lower()
    )
    if permanent:
        disable_nvfp4_tc(msg.split("\n", 1)[0][:240])
        return True
    return _DISABLED


def announce_tc_status_at_register() -> None:
    """One-line status when addmm / full stack is registered (cloud-visible)."""
    ok = probe_nvfp4_tc_support()
    try:
        import torch

        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            cc = f"{major}.{minor}"
        else:
            cc = "n/a"
    except Exception:
        name, cc = "?", "?"
    if ok:
        print(
            f"[HSWQ NVFP4] TC probe: GPU={name} CC={cc} — "
            f"scaled_mm_nvfp4 enabled (min CC 10.0)",
            flush=True,
        )
    else:
        disable_nvfp4_tc(
            f"compute capability {cc} < 10.0 (NVFP4 TensorCore requires Blackwell+)",
            announce=True,
        )
