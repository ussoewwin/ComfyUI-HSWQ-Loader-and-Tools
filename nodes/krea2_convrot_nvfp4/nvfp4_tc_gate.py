"""NVFP4 path gate (shared by addmm patch + ConvRot Linear forward).

HSWQ Linear hot path: ConvRot act rotate → pooled act NVFP4 quant → cuBLAS
FP4 Tensor-Core GEMM (raw kitchen ``_C`` primitive; weight stays packed).
Bake→float + ``F.linear`` is the per-call fallback only. Residual QT×QT
edges: ``hswq_scaled_mm_nvfp4`` (dequant both → float mm).
Never torch native ``F.scaled_mm`` FP4 (SM120: CUBLAS NOT_SUPPORTED → sticky
CUDA → illegal memory access); the direct ``_C`` call is the verified path.

Gate contract:
  1) probe CC once (Blackwell family: CC >= 10.0)
  2) permanent disable ONLY when hardware cannot do NVFP4 TC (CC < 10.0)
  3) rare RuntimeError → sticky clear + per-call float path (no process-wide kill)

Never edits ComfyUI-master.
"""
from __future__ import annotations

import logging

_PROBED = False
_TC_OK: bool | None = None
_DISABLED = False
_WARNED = False
_DISABLE_REASON = ""
_CALL_FAIL_WARNED = False

_KITCHEN_NVFP4_LOG = "comfy_kitchen.tensor.nvfp4"
_ADDMM_LOG = "nvfp4.nvfp4_addmm_patch"
_FORWARD_LOG = "nvfp4.nvfp4_forward"


def _mute_nvfp4_warning_spam() -> None:
    for name in (_KITCHEN_NVFP4_LOG, _ADDMM_LOG, _FORWARD_LOG):
        logging.getLogger(name).setLevel(logging.ERROR)


def clear_cuda_sticky_error() -> None:
    """Clear sticky CUDA / cuBLAS error so follow-up kernels can run."""
    try:
        import torch

        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
    except Exception:
        pass


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
    """Permanent disable for this process (CC < 10.0 only).

    Do NOT call this for CUBLAS_STATUS_NOT_SUPPORTED — stock kitchen dequants
    that call only and keeps TC enabled for later Linears.
    """
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
            f"[HSWQ NVFP4] TensorCore path disabled for this run "
            f"(GPU={name}, CC={cc}): {_DISABLE_REASON}. "
            f"Using float dequant mm; further kitchen WARNINGs suppressed.",
            flush=True,
        )


def note_scaled_mm_failure(exc: BaseException) -> bool:
    """Per-call failure: sticky-clear + do NOT kill TC for the process.

    HSWQ GEMM / weight-dequant edges (OOM/import). Always clear sticky CUDA first.
    Permanent disable remains only via ``disable_nvfp4_tc`` (CC < 10.0 probe).
    """
    global _CALL_FAIL_WARNED
    clear_cuda_sticky_error()
    if _DISABLED:
        return True
    if not _CALL_FAIL_WARNED:
        _CALL_FAIL_WARNED = True
        msg = str(exc).split("\n", 1)[0][:240]
        print(
            f"[HSWQ NVFP4] GEMM path exception (sticky cleared; per-call "
            f"float dequant; TC stays enabled): {msg}",
            flush=True,
        )
    return False


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
            f"[HSWQ NVFP4] TC probe: GPU={name} CC={cc} - "
            f"HSWQ path enabled: ConvRot act + pooled NVFP4 quant + cuBLAS FP4 "
            f"TC GEMM (weight packed resident; bake fallback; never F.scaled_mm)",
            flush=True,
        )
    else:
        disable_nvfp4_tc(
            f"compute capability {cc} < 10.0 (NVFP4 TensorCore requires Blackwell+)",
            announce=True,
        )
