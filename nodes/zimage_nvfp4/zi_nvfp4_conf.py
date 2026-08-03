"""Z Image INT8 protect ConvRot conf helpers (branch-only; not in nodes/nvfp4)."""
from __future__ import annotations

from typing import Optional


def is_int8_tensorwise_conf(conf: Optional[dict]) -> bool:
    return isinstance(conf, dict) and str(conf.get("format") or "").lower() == "int8_tensorwise"


def int8_convrot_flags_from_conf(conf: Optional[dict]) -> tuple[bool, int]:
    """Return (enabled, groupsize) for INT8 protect ConvRot comfy_quant.

    Do **not** reuse ``convrot_flags_from_conf`` — that helper is NVFP4-only and
    always returns False for ``int8_tensorwise``. Used by load arm to set
    ``_hswq_int8_convrot`` and clear kitchen ``Params.convrot`` (Conv2d twin).
    """
    if not is_int8_tensorwise_conf(conf):
        return False, 256
    params_conf = conf.get("params", {})
    if not isinstance(params_conf, dict):
        params_conf = {}
    enabled = bool(conf.get("convrot", False)) or bool(params_conf.get("convrot", False))
    if not enabled:
        return False, 256
    gs = int(conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256)) or 256)
    return True, gs
