"""Minimal container probing, without pulling in torch or the VLM stack.

Needed because Stage3 tube times are indices into the 5 fps sampled sequence while
the VIRAT GT object tracks are indexed by original video frame.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import av


@lru_cache(maxsize=512)
def native_fps(video_path: str) -> float:
    """Frame rate of the source container, or 30.0 when it cannot be read."""
    if not Path(video_path).is_file():
        return 30.0
    try:
        container = av.open(video_path)
    except Exception:  # noqa: BLE001
        return 30.0
    try:
        stream = container.streams.video[0]
        rate = float(stream.average_rate) if stream.average_rate else 0.0
    except Exception:  # noqa: BLE001
        rate = 0.0
    finally:
        container.close()
    return rate if rate > 1e-6 else 30.0
