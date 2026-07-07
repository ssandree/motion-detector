"""Load video frames and write VLM input images from resolved plan."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from motion_analyzer.adaptive_input.plan_resolver import (
    EventAnchorInput,
    ResolvedInputPlan,
    RoiFrameInput,
    SparseGlobalInput,
)

logger = logging.getLogger(__name__)

ROI_COLORS: dict[int, tuple[int, int, int]] = {
    1: (0, 255, 0),
    2: (0, 165, 255),
}


@dataclass
class SavedInputFrame:
    category: str
    frame_idx: int
    timestamp_sec: float
    path: Path
    roi_id: int | None = None
    bbox: list[int] | None = None
    label: str | None = None


def _input_filename(frame_idx: int, timestamp_sec: float) -> str:
    return f"{frame_idx:06d}_{timestamp_sec:.3f}s.jpg"


def _clamp_bbox(bbox: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def _read_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    if not ret:
        logger.warning("Could not read frame %d from video", frame_idx)
        return None
    return frame


def _save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def _draw_roi_bbox(
    frame: np.ndarray,
    bbox: list[int],
    *,
    roi_id: int,
    label: str | None = None,
) -> np.ndarray:
    vis = frame.copy()
    color = ROI_COLORS.get(roi_id, (255, 255, 0))
    x1, y1, x2, y2 = bbox
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
    text = label or f"ROI{roi_id}"
    cv2.putText(
        vis,
        text,
        (x1 + 2, max(y1 - 8, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
    return vis


class VideoFrameCache:
    """Seek-read cache for repeated frame access during export."""

    def __init__(self, video_path: str) -> None:
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cache: dict[int, np.ndarray] = {}

    def close(self) -> None:
        self.cap.release()

    def get(self, frame_idx: int) -> np.ndarray | None:
        fi = int(frame_idx)
        if fi in self._cache:
            return self._cache[fi].copy()
        frame = _read_frame(self.cap, fi)
        if frame is not None:
            self._cache[fi] = frame
        return frame.copy() if frame is not None else None


def extract_vlm_frames(
    video_path: str,
    resolved: ResolvedInputPlan,
    out_dir: Path,
) -> list[SavedInputFrame]:
    """Write sparse_global/, roi1/, roi2/, event_anchor/ images."""
    cache = VideoFrameCache(video_path)
    saved: list[SavedInputFrame] = []

    try:
        sg_dir = out_dir / "sparse_global"
        for entry in resolved.sparse_global:
            frame = cache.get(entry.frame_idx)
            if frame is None:
                continue
            path = sg_dir / _input_filename(entry.frame_idx, entry.timestamp_sec)
            _save_image(path, frame)
            saved.append(
                SavedInputFrame(
                    category="sparse_global",
                    frame_idx=entry.frame_idx,
                    timestamp_sec=entry.timestamp_sec,
                    path=path,
                )
            )

        roi_by_id: dict[int, list[RoiFrameInput]] = {}
        for roi_frame in resolved.roi_frames:
            roi_by_id.setdefault(roi_frame.roi_id, []).append(roi_frame)

        for roi_id, roi_frames in sorted(roi_by_id.items()):
            roi_dir = out_dir / f"roi{roi_id}"
            for roi_frame in roi_frames:
                frame = cache.get(roi_frame.frame_idx)
                if frame is None:
                    continue
                x1, y1, x2, y2 = _clamp_bbox(
                    roi_frame.bbox, cache.width, cache.height
                )
                crop = frame[y1:y2, x1:x2].copy()
                if crop.size == 0:
                    logger.warning(
                        "Empty ROI crop roi%d frame %d bbox %s",
                        roi_id,
                        roi_frame.frame_idx,
                        roi_frame.bbox,
                    )
                    continue
                path = roi_dir / _input_filename(
                    roi_frame.frame_idx, roi_frame.timestamp_sec
                )
                _save_image(path, crop)
                saved.append(
                    SavedInputFrame(
                        category=f"roi{roi_id}",
                        frame_idx=roi_frame.frame_idx,
                        timestamp_sec=roi_frame.timestamp_sec,
                        path=path,
                        roi_id=roi_id,
                        bbox=roi_frame.bbox,
                    )
                )

        anchor_dir = out_dir / "event_anchor"
        for anchor in resolved.event_anchors:
            frame = cache.get(anchor.frame_idx)
            if frame is None:
                continue
            path = anchor_dir / _input_filename(anchor.frame_idx, anchor.timestamp_sec)
            _save_image(path, frame)
            saved.append(
                SavedInputFrame(
                    category="event_anchor",
                    frame_idx=anchor.frame_idx,
                    timestamp_sec=anchor.timestamp_sec,
                    path=path,
                    label=anchor.label,
                )
            )
    finally:
        cache.close()

    saved.sort(key=lambda s: (s.timestamp_sec, s.category, s.frame_idx))
    return saved


def load_frame_with_roi_overlay(
    cache: VideoFrameCache,
    roi_frame: RoiFrameInput,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (source with bbox, crop) for visualization."""
    frame = cache.get(roi_frame.frame_idx)
    if frame is None:
        return None, None
    x1, y1, x2, y2 = _clamp_bbox(roi_frame.bbox, cache.width, cache.height)
    overlay = _draw_roi_bbox(frame, [x1, y1, x2, y2], roi_id=roi_frame.roi_id)
    crop = frame[y1:y2, x1:x2].copy()
    return overlay, crop if crop.size else None


def saved_inputs_to_report_entries(saved: list[SavedInputFrame]) -> list[dict[str, Any]]:
    return [
        {
            "category": s.category,
            "frame_idx": s.frame_idx,
            "timestamp_sec": round(s.timestamp_sec, 4),
            "path": s.path.name,
            "roi_id": s.roi_id,
            "bbox": s.bbox,
            "label": s.label,
        }
        for s in saved
    ]
