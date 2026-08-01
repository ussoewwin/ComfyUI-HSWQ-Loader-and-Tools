"""Z Image entry — same module as ``nodes.nvfp4.nvfp4_comfy_parity`` (single source).

Do not fork logic here. INT8 protect ConvRot act-rotate and NVFP4 parity live only
in ``nodes/nvfp4/nvfp4_comfy_parity.py``; this path re-exports so both import sites
stay identical.
"""
from __future__ import annotations

from ..nvfp4.nvfp4_comfy_parity import (  # noqa: F401
    apply_nvfp4_comfy_parity,
    is_nvfp4_comfy_parity_active,
    log_nvfp4_parity_load_summary,
    remember_nvfp4_tc_product_stack,
    require_convrot_parity_forward,
    reset_nvfp4_parity_load_counters,
    restore_nvfp4_tc_product_stack,
    summarize_nvfp4_parity_modules,
)

__all__ = [
    "apply_nvfp4_comfy_parity",
    "is_nvfp4_comfy_parity_active",
    "log_nvfp4_parity_load_summary",
    "remember_nvfp4_tc_product_stack",
    "require_convrot_parity_forward",
    "reset_nvfp4_parity_load_counters",
    "restore_nvfp4_tc_product_stack",
    "summarize_nvfp4_parity_modules",
]
