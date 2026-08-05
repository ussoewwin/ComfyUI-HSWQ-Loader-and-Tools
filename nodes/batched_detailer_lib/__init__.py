"""
In-tree helpers for HSWQ Batched Detailer (SEGS).

Developed based on ComfyUI-Impact-Pack (ltdrdata, GPL-3.0):
  https://github.com/ltdrdata/ComfyUI-Impact-Pack
with HSWQ-specific packaging so this node does not import the Impact Pack
package at runtime. See IMPACT_PACK_LICENSE.txt and NOTICE.
"""

from .core import SEG, crop_condition_mask, get_schedulers, segs_scale_match
from . import sampling as impact_sampling
from . import utils
from . import wildcards

__all__ = [
    "SEG",
    "crop_condition_mask",
    "get_schedulers",
    "segs_scale_match",
    "impact_sampling",
    "utils",
    "wildcards",
]
