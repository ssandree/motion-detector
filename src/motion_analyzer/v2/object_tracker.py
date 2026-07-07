"""YOLO + ByteTrack object tracking on sampled frames (ultralytics track API)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from motion_analyzer.v2.object_detector import DEFAULT_TARGET_CLASSES

logger = logging.getLogger(__name__)


@dataclass
class TrackerConfig:
    model_name: str = "yolov8n.pt"
    conf_threshold: float = 0.15
    tracker: str = "bytetrack"
    tracker_config: str = "bytetrack.yaml"
    target_classes: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_CLASSES))
    device: str | None = None


def _resolve_tracker_yaml(tracker_config: str) -> str:
    """Resolve tracker yaml path for ultralytics."""
    if Path(tracker_config).is_file():
        return str(tracker_config)
    return tracker_config


def run_object_tracking(
    video_path: str,
    frame_df: pd.DataFrame,
    *,
    config: TrackerConfig,
    video_name: str,
    frame_w: int,
    frame_h: int,
) -> pd.DataFrame:
    """
    Run YOLO + ByteTrack on sampled frames in temporal order.

    Uses ``model.track(persist=True)`` so track_id continuity is maintained
  across the 5fps sampled sequence (not native video FPS).
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required. Install with: pip install ultralytics"
        ) from exc

    model = YOLO(config.model_name)
    target_set = set(config.target_classes)
    frame_area = float(frame_w * frame_h)
    tracker_yaml = _resolve_tracker_yaml(config.tracker_config)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    rows: list[dict[str, Any]] = []
    indices = list(range(len(frame_df)))

    for sampled_idx, (_, row) in enumerate(frame_df.iterrows()):
        frame_idx = int(row["frame_idx"])
        timestamp_sec = float(row["timestamp_sec"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            logger.warning("Could not read frame %d", frame_idx)
            continue

        predict_kwargs: dict[str, Any] = {
            "conf": config.conf_threshold,
            "persist": True,
            "tracker": tracker_yaml,
            "verbose": False,
        }
        if config.device:
            predict_kwargs["device"] = config.device

        results = model.track(frame, **predict_kwargs)
        if not results:
            continue

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue

        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls.item())
            class_name = names.get(cls_id, str(cls_id))
            if class_name not in target_set:
                continue

            track_id = -1
            if box.id is not None:
                track_id = int(box.id.item())

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf.item())
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            area_ratio = ((x2 - x1) * (y2 - y1)) / max(frame_area, 1.0)

            rows.append({
                "video_name": video_name,
                "sampled_idx": sampled_idx,
                "frame_idx": frame_idx,
                "timestamp_sec": round(timestamp_sec, 4),
                "track_id": track_id,
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": round(conf, 6),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "bbox_area_ratio": round(area_ratio, 8),
                "center_x": round(cx, 2),
                "center_y": round(cy, 2),
            })

        if (sampled_idx + 1) % 20 == 0:
            logger.info("Tracker: %d/%d sampled frames", sampled_idx + 1, len(indices))

    cap.release()

    if not rows:
        logger.warning("No tracked objects for %s", video_name)
        return pd.DataFrame(columns=[
            "video_name", "sampled_idx", "frame_idx", "timestamp_sec",
            "track_id", "class_id", "class_name", "confidence",
            "x1", "y1", "x2", "y2", "bbox_area_ratio", "center_x", "center_y",
        ])

    df = pd.DataFrame(rows)
    n_tracks = df[df["track_id"] >= 0]["track_id"].nunique()
    logger.info("Tracked %d detections, %d unique track_ids", len(df), n_tracks)
    return df
