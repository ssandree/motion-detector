"""Bounding-box geometry helpers for motion-object association."""

from __future__ import annotations

from typing import Any

import numpy as np

EPS = 1e-8


def bbox_from_dict(d: dict[str, Any]) -> tuple[float, float, float, float]:
    return float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"])


def bbox_area(x1: float, y1: float, x2: float, y2: float) -> float:
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = bbox_from_dict(a)
    bx1, by1, bx2, by2 = bbox_from_dict(b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = bbox_area(ax1, ay1, ax2, ay2) + bbox_area(bx1, by1, bx2, by2) - inter
    return inter / max(union, EPS)


def center_inside(inner: dict[str, Any], outer: dict[str, Any]) -> bool:
  cx = float(inner.get("center_x", (inner["x1"] + inner["x2"]) / 2))
  cy = float(inner.get("center_y", (inner["y1"] + inner["y2"]) / 2))
  ox1, oy1, ox2, oy2 = bbox_from_dict(outer)
  return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


def center_distance_norm(a: dict[str, Any], b: dict[str, Any], frame_w: int, frame_h: int) -> float:
    acx = float(a.get("center_x", (a["x1"] + a["x2"]) / 2))
    acy = float(a.get("center_y", (a["y1"] + a["y2"]) / 2))
    bcx = float(b.get("center_x", (b["x1"] + b["x2"]) / 2))
    bcy = float(b.get("center_y", (b["y1"] + b["y2"]) / 2))
    dist = float(np.hypot(acx - bcx, acy - bcy))
    diag = float(np.hypot(frame_w, frame_h))
    return dist / max(diag, EPS)


def expand_bbox(
    x1: float, y1: float, x2: float, y2: float,
    margin_ratio: float,
    frame_w: int,
    frame_h: int,
) -> list[int]:
    w = x2 - x1
    h = y2 - y1
    mx = w * margin_ratio
    my = h * margin_ratio
    return [
        int(max(0, x1 - mx)),
        int(max(0, y1 - my)),
        int(min(frame_w, x2 + mx)),
        int(min(frame_h, y2 + my)),
    ]


def union_bbox(boxes: list[dict[str, Any]]) -> dict[str, float]:
    if not boxes:
        return {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0}
    x1 = min(float(b["x1"]) for b in boxes)
    y1 = min(float(b["y1"]) for b in boxes)
    x2 = max(float(b["x2"]) for b in boxes)
    y2 = max(float(b["y2"]) for b in boxes)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "center_x": cx, "center_y": cy,
        "bbox_width": x2 - x1,
        "bbox_height": y2 - y1,
    }


def size_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    area_a = max(float(a.get("bbox_width", a["x2"] - a["x1"])) * float(a.get("bbox_height", a["y2"] - a["y1"])), 1.0)
    area_b = max(float(b.get("bbox_width", b["x2"] - b["x1"])) * float(b.get("bbox_height", b["y2"] - b["y1"])), 1.0)
    return min(area_a, area_b) / max(area_a, area_b)


def min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < EPS:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]
