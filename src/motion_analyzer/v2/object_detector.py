"""YOLO-based object detection on sampled video frames."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_TARGET_CLASSES = [
    "person", "car", "bicycle", "motorcycle", "bus", "truck",
    "backpack", "handbag", "suitcase",
]
DEFAULT_STATIC_SUPPRESSION_CLASSES = [
    "chair", "bench", "umbrella", "potted plant", "dining table",
]

CLASS_WEIGHTS: dict[str, float] = {
    "person": 1.0,
    "car": 0.95,
    "bicycle": 0.85,
    "motorcycle": 0.85,
    "bus": 0.90,
    "truck": 0.90,
    "backpack": 0.70,
    "handbag": 0.65,
    "suitcase": 0.65,
}


@dataclass
class DetectorConfig:
    model_name: str = "yolov8n.pt"
    conf_threshold: float = 0.25
    target_classes: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_CLASSES))
    static_suppression_classes: list[str] = field(
        default_factory=lambda: list(DEFAULT_STATIC_SUPPRESSION_CLASSES)
    )
    device: str | None = None


def _load_model(model_name: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for object detection. "
            "Install with: pip install ultralytics"
        ) from exc
    return YOLO(model_name)


def _read_frame_at(video_path: str, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def run_object_detector(
    video_path: str,
    frame_df: pd.DataFrame,
    *,
    config: DetectorConfig,
    video_name: str,
    frame_w: int,
    frame_h: int,
) -> pd.DataFrame:
    """
    Run YOLO on sampled frames and return detection rows.

    Detects both target and static-suppression classes; flags each row accordingly.
    """
    all_classes = set(config.target_classes) | set(config.static_suppression_classes)
    model = _load_model(config.model_name)
    frame_area = float(frame_w * frame_h)

    rows: list[dict[str, Any]] = []
    indices = list(range(len(frame_df)))

    for sampled_idx, (_, row) in enumerate(frame_df.iterrows()):
        frame_idx = int(row["frame_idx"])
        timestamp_sec = float(row["timestamp_sec"])
        frame = _read_frame_at(video_path, frame_idx)
        if frame is None:
            logger.warning("Could not read frame %d from %s", frame_idx, video_path)
            continue

        predict_kwargs: dict[str, Any] = {
            "conf": config.conf_threshold,
            "verbose": False,
        }
        if config.device:
            predict_kwargs["device"] = config.device

        results = model.predict(frame, **predict_kwargs)
        if not results:
            continue

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue

        names = result.names
        det_id = 0
        for box in result.boxes:
            cls_id = int(box.cls.item())
            class_name = names.get(cls_id, str(cls_id))
            if class_name not in all_classes:
                continue

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
                "det_id_in_frame": det_id,
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
                "is_target_class": class_name in config.target_classes,
                "is_static_suppression_class": class_name in config.static_suppression_classes,
            })
            det_id += 1

        if (sampled_idx + 1) % 10 == 0:
            logger.info("Detector: %d/%d sampled frames", sampled_idx + 1, len(indices))

    if not rows:
        logger.warning("No detections found for %s", video_name)
        return pd.DataFrame(columns=[
            "video_name", "sampled_idx", "frame_idx", "timestamp_sec",
            "det_id_in_frame", "class_id", "class_name", "confidence",
            "x1", "y1", "x2", "y2", "bbox_area_ratio", "center_x", "center_y",
            "is_target_class", "is_static_suppression_class",
        ])

    return pd.DataFrame(rows)
