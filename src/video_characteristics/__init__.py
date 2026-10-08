"""Video characteristic annotation (AD/OD/BM/IE/DE) for VIRAT and MEVA.

Reads existing Stage3 ROI tube results and existing GT event annotations.
Never rewrites Stage1-3 outputs or earlier inference results.
"""

from __future__ import annotations

LABELS = ("AD", "OD", "BM", "IE", "DE")

LABEL_NAMES = {
    "AD": "action_detail",
    "OD": "object_detail",
    "BM": "background_motion",
    "IE": "integrated_event",
    "DE": "different_event",
}

__all__ = ["LABELS", "LABEL_NAMES"]
