"""CLIP / text-encoder INT8 GEMM arm for ComfyUI.

ComfyUI's ``int8_tensorwise`` algorithm is declared with ``quantize_input: False``,
so it is a **weight-only** format: the weight stays int8 in memory but is
dequantized for every matmul, and the real INT8 GEMM (``comfy_kitchen``'s
``int8_linear``) is never reached.  ``comfy/sd1_clip.py`` additionally builds the
text encoder with ``full_precision_mm=True``, which disables the quantized matmul
outright.

This arm enables the real INT8 GEMM at runtime.  ComfyUI core is not modified.
See ``arm.py`` for the implementation.
"""

from .arm import (  # noqa: F401
    _enable_int8_gemm_for_clip_text,
    disable_int8_gemm_for_clip_text,
    int8_gemm_stats,
    reset_int8_gemm_stats,
)

__all__ = [
    "_enable_int8_gemm_for_clip_text",
    "disable_int8_gemm_for_clip_text",
    "int8_gemm_stats",
    "reset_int8_gemm_stats",
]
