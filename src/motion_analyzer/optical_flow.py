"""Video frame sampling helpers for the motion pipeline."""

from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np


def sample_frame_indices(native_fps: float, target_fps: float, frame_count: int) -> list[int]:
    """Return frame indices to sample at approximately *target_fps*."""
    if native_fps <= 0 or frame_count <= 0:
        return []
    step = max(1, int(round(native_fps / target_fps)))
    return list(range(0, frame_count, step))


def iter_sampled_frames(
    video_path: str,
    target_fps: float,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_idx, timestamp_sec, bgr_frame) sampled at *target_fps*."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(sample_frame_indices(native_fps, target_fps, frame_count))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in indices:
            timestamp = frame_idx / native_fps
            yield frame_idx, timestamp, frame
        frame_idx += 1

    cap.release()
