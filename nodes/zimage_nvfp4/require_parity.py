"""Fail closed if ConvRot act-rotate forward is not armed.

Re-exports the canonical guard from ``nodes.nvfp4.nvfp4_comfy_parity`` so
Z Image and nvfp4 package entry points cannot drift.
"""
from __future__ import annotations

from ..nvfp4.nvfp4_comfy_parity import require_convrot_parity_forward

__all__ = ["require_convrot_parity_forward"]
