"""Debug visualization for motion maps (internal use only)."""

from __future__ import annotations

import cv2
import numpy as np


def motion_map_to_heatmap(motion_map: np.ndarray) -> np.ndarray:
    """Convert a normalized motion map to a BGR heatmap image."""
    clipped = np.clip(motion_map, 0.0, 1.0)
    uint8 = (clipped * 255).astype(np.uint8)
    return cv2.applyColorMap(uint8, cv2.COLORMAP_JET)
