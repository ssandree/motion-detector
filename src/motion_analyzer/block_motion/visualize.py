"""Visualization for residual-flow block representation heatmaps."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from motion_analyzer.block_motion.representation import (
    BlockRepresentationConfig,
    FrameBlockRepresentation,
)

logger = logging.getLogger(__name__)


def _normalize_map(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def heatmap_overlay(
    frame_bgr: np.ndarray,
    heatmap: np.ndarray,
    *,
    block_size: int,
    alpha: float = 0.55,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    norm = _normalize_map(heatmap)
    heat_small = cv2.applyColorMap(
        np.clip(norm * 255.0, 0, 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    heat = cv2.resize(heat_small, (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, heat, alpha, 0.0)


def _annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    out = frame.copy()
    y = 22
    for line in lines:
        cv2.putText(
            out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 3, cv2.LINE_AA
        )
        cv2.putText(
            out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA
        )
        y += 20
    return out


def save_block_representation_overlays(
    *,
    video_path: str,
    representations: Sequence[FrameBlockRepresentation],
    out_dir: str | Path,
    config: BlockRepresentationConfig,
    output_fps: float = 5.0,
) -> dict[str, str]:
    """Write heatmap_overlay.mp4 for block representation scores."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not representations:
        raise ValueError("No representations to visualize")

    cfg = config
    heat_path = out / "heatmap_overlay.mp4"
    legacy_heat_path = out / "block_heatmap_overlay.mp4"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    heat_writer = cv2.VideoWriter(str(heat_path), fourcc, max(output_fps, 0.1), (frame_w, frame_h))
    if not heat_writer.isOpened():
        cap.release()
        raise RuntimeError("Failed to open heatmap overlay writer")

    for rep in representations:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(rep.frame_idx))
        ret, frame = cap.read()
        if not ret:
            logger.warning("Could not read frame %d for heatmap overlay", rep.frame_idx)
            continue
        hud = [
            f"f={rep.frame_idx}  block={cfg.block_size}x{cfg.block_size}",
            f"score={cfg.heatmap_feature}  (representation only)",
        ]
        heat_vis = heatmap_overlay(
            frame,
            rep.score_map,
            block_size=cfg.block_size,
            alpha=cfg.overlay_alpha,
        )
        heat_writer.write(_annotate(heat_vis, hud))

    cap.release()
    heat_writer.release()
    shutil.copy2(heat_path, legacy_heat_path)
    logger.info("Wrote representation heatmap: %s", heat_path.name)
    return {
        "heatmap_overlay_mp4": str(heat_path),
        "block_heatmap_overlay_mp4": str(legacy_heat_path),
    }
