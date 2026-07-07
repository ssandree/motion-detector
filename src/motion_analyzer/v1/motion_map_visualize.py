"""Visualization for motion compensation layers."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

LAYER_FILENAMES = {
    "original": "original_motion_overlay.mp4",
    "global": "global_motion_overlay.mp4",
    "residual": "residual_motion_overlay.mp4",
    "suppressed": "suppressed_motion_overlay.mp4",
}


def colorize_motion_on_frame(
    frame_bgr: np.ndarray,
    motion_map: np.ndarray,
    *,
    alpha: float = 0.55,
) -> np.ndarray:
    heat = cv2.applyColorMap(
        np.clip(motion_map * 255.0, 0, 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, heat, alpha, 0)


def save_motion_layer_overlays(
    video_path: str,
    frame_indices: list[int],
    layer_maps: dict[str, list[np.ndarray]],
    out_dir: Path,
    *,
    output_fps: float = 5.0,
) -> dict[str, Path]:
    """Write original/global/residual/suppressed motion overlay MP4s."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writers: dict[str, cv2.VideoWriter] = {}
    paths: dict[str, Path] = {}

    for key, filename in LAYER_FILENAMES.items():
        maps = layer_maps.get(key, [])
        if not maps:
            continue
        out_path = out_dir / filename
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to create video writer: {out_path}")
        writers[key] = writer
        paths[key] = out_path

    for fi, fidx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ret, frame = cap.read()
        if not ret:
            logger.warning("motion overlay: could not read frame %d", fidx)
            continue
        for key, writer in writers.items():
            maps = layer_maps[key]
            if fi >= len(maps):
                continue
            vis = colorize_motion_on_frame(frame, maps[fi])
            writer.write(vis)

    cap.release()
    for writer in writers.values():
        writer.release()

    for key, path in paths.items():
        logger.info("Wrote %s (%d frames)", path.name, len(layer_maps.get(key, [])))
    return paths
