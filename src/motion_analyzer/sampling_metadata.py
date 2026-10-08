"""Build and write VLM frame-sampling ``metadata.json`` (shared schema).

Canonical form matches ``proposed_ROI_w_fps/{gaptype}/{VIDEO_ID}/metadata.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class SavedFrameRef:
    """One saved JPEG relative to the video root directory."""

    timestamp: float
    file: str  # e.g. "global/global_frame_000001_t00.0.jpg"
    frame_index: int | None = None  # optional; not written to JSON


@dataclass(frozen=True)
class RoiMetaInput:
    roi_id: int
    normalized_bbox: Sequence[float]  # [x_min, y_min, x_max, y_max] in [0, 1]
    frames: Sequence[SavedFrameRef]
    source_region: str | None = None
    # Ignored when frames are non-empty; time_range is derived from frames.
    time_range: tuple[float, float] | None = None


def normalize_bbox(
    pixel_bbox: Sequence[float],
    *,
    frame_width: int,
    frame_height: int,
) -> list[float]:
    """Pixel AABB → [x_min, y_min, x_max, y_max] in [0, 1] on full frame."""
    w = max(int(frame_width), 1)
    h = max(int(frame_height), 1)
    x0, y0, x1, y1 = (float(v) for v in pixel_bbox)
    nx0 = float(min(max(x0 / w, 0.0), 1.0))
    ny0 = float(min(max(y0 / h, 0.0), 1.0))
    nx1 = float(min(max(x1 / w, 0.0), 1.0))
    ny1 = float(min(max(y1 / h, 0.0), 1.0))
    if nx1 < nx0:
        nx0, nx1 = nx1, nx0
    if ny1 < ny0:
        ny0, ny1 = ny1, ny0
    return [nx0, ny0, nx1, ny1]


def union_normalized_bboxes(
    bboxes: Sequence[Sequence[float]],
) -> list[float]:
    """Axis-aligned union of normalized boxes; empty → full frame."""
    if not bboxes:
        return [0.0, 0.0, 1.0, 1.0]
    xs0, ys0, xs1, ys1 = [], [], [], []
    for box in bboxes:
        x0, y0, x1, y1 = (float(v) for v in box)
        xs0.append(x0)
        ys0.append(y0)
        xs1.append(x1)
        ys1.append(y1)
    return [
        float(min(xs0)),
        float(min(ys0)),
        float(max(xs1)),
        float(max(ys1)),
    ]


def _cell_label(row: int, col: int) -> str:
    vert = ("top", "center", "bottom")[row]
    horiz = ("left", "center", "right")[col]
    if vert == "center" and horiz == "center":
        return "center"
    return f"{vert}-{horiz}"


def _overlap_area(
    a: Sequence[float],
    b: Sequence[float],
) -> float:
    ax0, ay0, ax1, ay1 = (float(v) for v in a)
    bx0, by0, bx1, by1 = (float(v) for v in b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return float((ix1 - ix0) * (iy1 - iy0))


def source_region_from_normalized_bbox(normalized_bbox: Sequence[float]) -> str:
    """Label from 3×3 full-frame cells that the bbox overlaps.

    Single cell → ``top-left`` / ``center-right`` / ``center`` …
    Multiple cells → joined with `` to `` in reading order, e.g.
    ``center-right to bottom-right``.
    """
    x0, y0, x1, y1 = (float(v) for v in normalized_bbox)
    box = [x0, y0, x1, y1]
    box_area = max((x1 - x0) * (y1 - y0), 0.0)

    hits: list[str] = []
    for row in range(3):
        for col in range(3):
            cell = [col / 3.0, row / 3.0, (col + 1) / 3.0, (row + 1) / 3.0]
            area = _overlap_area(box, cell)
            if area <= 1e-9:
                continue
            cell_area = (1.0 / 3.0) * (1.0 / 3.0)
            frac_cell = area / cell_area
            frac_box = area / box_area if box_area > 0 else 0.0
            # Keep cells that cover ≥2% of the cell or ≥2% of the ROI.
            if frac_cell >= 0.02 or frac_box >= 0.02:
                hits.append(_cell_label(row, col))

    if not hits:
        # Degenerate / empty box — fall back to center cell of the midpoint.
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        col = 0 if cx < 1 / 3 else (1 if cx < 2 / 3 else 2)
        row = 0 if cy < 1 / 3 else (1 if cy < 2 / 3 else 2)
        return _cell_label(row, col)
    if len(hits) == 1:
        return hits[0]
    return " to ".join(hits)


def _round_ts(value: float) -> float:
    """Match on-disk timestamp tokens (1 decimal)."""
    return float(round(float(value), 1))


def _frame_dicts(frames: Iterable[SavedFrameRef]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fr in frames:
        rows.append(
            {
                "timestamp": _round_ts(fr.timestamp),
                "file": str(fr.file).replace("\\", "/"),
            }
        )
    return rows


def build_sampling_metadata(
    *,
    video_id: str,
    global_frames: Sequence[SavedFrameRef] | None,
    rois: Sequence[RoiMetaInput],
    global_coverage: str = "full scene",
) -> dict[str, Any]:
    """Assemble per-video metadata.

    * ``global_frames is None`` → ``"global": null`` (ROI-only datasets).
    * ``global_frames == []`` → empty global block with ``num_frames: 0``.
    * ROIs with zero sampled frames are **omitted** (no empty stubs).
    * ``time_range`` / ``num_frames`` always follow the sampled ``frames`` list.
    """
    roi_rows: list[dict[str, Any]] = []
    for roi in rois:
        frames = _frame_dicts(roi.frames)
        if not frames:
            continue
        bbox = [round(float(v), 6) for v in roi.normalized_bbox]
        # Always derive spatial label from the (possibly union) bbox coverage.
        region = source_region_from_normalized_bbox(bbox)
        if roi.source_region:
            # Explicit override only when caller already computed overlap-style label.
            region = str(roi.source_region)
        timestamps = [float(f["timestamp"]) for f in frames]
        t0, t1 = min(timestamps), max(timestamps)
        roi_rows.append(
            {
                "roi_id": int(roi.roi_id),
                "time_range": [_round_ts(t0), _round_ts(t1)],
                "source_region": region,
                "normalized_bbox": bbox,
                "num_frames": len(frames),
                "frames": frames,
            }
        )

    if global_frames is None:
        global_block: Any = None
    else:
        g_frames = _frame_dicts(global_frames)
        global_block = {
            "coverage": str(global_coverage),
            "num_frames": len(g_frames),
            "frames": g_frames,
        }

    return {
        "video_id": str(video_id),
        "global": global_block,
        "rois": roi_rows,
    }


def write_sampling_metadata(path: Path | str, metadata: dict[str, Any]) -> Path:
    """Write UTF-8 indented JSON; returns the written path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out


def validate_sampling_metadata(
    metadata: dict[str, Any],
    *,
    video_dir: Path,
) -> list[str]:
    """Return a list of validation error strings (empty = OK)."""
    errors: list[str] = []
    root = Path(video_dir)

    global_block = metadata.get("global")
    if global_block is not None:
        if not isinstance(global_block, dict):
            errors.append("global must be object or null")
        else:
            frames = global_block.get("frames") or []
            n = int(global_block.get("num_frames", -1))
            if n != len(frames):
                errors.append(f"global.num_frames={n} != len(frames)={len(frames)}")
            for fr in frames:
                rel = fr.get("file")
                if not rel or not (root / rel).is_file():
                    errors.append(f"missing global file: {rel}")

    rois = metadata.get("rois")
    if rois is None:
        errors.append("rois missing")
        return errors
    if not isinstance(rois, list):
        errors.append("rois must be a list")
        return errors

    for roi in rois:
        n = int(roi.get("num_frames", -1))
        frames = roi.get("frames") or []
        if n <= 0:
            errors.append(f"roi_id={roi.get('roi_id')}: num_frames must be > 0")
        if n != len(frames):
            errors.append(
                f"roi_id={roi.get('roi_id')}: num_frames={n} != len(frames)={len(frames)}"
            )
        if not frames:
            errors.append(f"roi_id={roi.get('roi_id')}: empty frames stub")
            continue
        ts = [float(f["timestamp"]) for f in frames]
        tr = roi.get("time_range") or []
        if len(tr) != 2 or float(tr[0]) != min(ts) or float(tr[1]) != max(ts):
            errors.append(
                f"roi_id={roi.get('roi_id')}: time_range={tr} != "
                f"[{min(ts)}, {max(ts)}]"
            )
        bbox = roi.get("normalized_bbox") or []
        if len(bbox) != 4 or any(float(v) < 0.0 or float(v) > 1.0 for v in bbox):
            errors.append(f"roi_id={roi.get('roi_id')}: bad normalized_bbox={bbox}")
        for fr in frames:
            rel = fr.get("file")
            if not rel or not (root / rel).is_file():
                errors.append(f"missing roi file: {rel}")

    return errors


__all__ = [
    "RoiMetaInput",
    "SavedFrameRef",
    "build_sampling_metadata",
    "normalize_bbox",
    "source_region_from_normalized_bbox",
    "union_normalized_bboxes",
    "validate_sampling_metadata",
    "write_sampling_metadata",
]
