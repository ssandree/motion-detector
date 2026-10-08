"""Stage 3: attractive Union-Find → AABB composition (official)."""

from __future__ import annotations

from typing import Any

from stage3.roi_tube import list_videos_in_fusion_root

__all__ = [
    "list_videos_in_fusion_root",
    "process_video",
]


def process_video(*args: Any, **kwargs: Any):
    from stage3.v2.light_pipeline import process_video as _process

    return _process(*args, **kwargs)
