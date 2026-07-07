"""Visualization for merged grid motion regions as overlay video."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _importance_color(importance: float, max_importance: float) -> tuple[int, int, int]:
    if max_importance <= 0:
        t = 0.0
    else:
        t = min(1.0, importance / max_importance)
    b = 0
    g = int(255 * (1.0 - t))
    r = int(255 * t)
    return (b, g, r)


def draw_blobs_on_frame(
    frame: np.ndarray,
    blobs: list[dict[str, Any]],
    *,
    max_labels: int = 30,
    fill_alpha: float = 0.15,
    line_thickness: int | None = None,
) -> np.ndarray:
    """Draw merged connected-region bboxes on a copy of *frame*."""
    vis = frame.copy()
    if not blobs:
        return vis

    ranked = sorted(blobs, key=lambda b: float(b.get("blob_importance", 0.0)), reverse=True)
    to_draw = ranked[:max_labels]
    max_imp = max(float(b.get("blob_importance", 0.0)) for b in to_draw)
    overlay = vis.copy()

    for blob in to_draw:
        x1, y1, x2, y2 = blob["x1"], blob["y1"], blob["x2"], blob["y2"]
        color = _importance_color(float(blob.get("blob_importance", 0.0)), max_imp)
        n_cells = blob.get("num_grid_cells", 1)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        thickness = line_thickness if line_thickness is not None else (3 if n_cells > 1 else 2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        coh = blob.get("flow_direction_coherence")
        coh_str = f" c{coh:.2f}" if coh is not None else ""
        imp = float(blob.get("blob_importance", 0.0))
        tex = blob.get("texture_flicker_score", 0.0)
        gm = blob.get("global_motion_score", 0.0)
        label = (
            f"R{blob.get('rank_by_importance', '?')} n={n_cells} "
            f"imp={imp:.2f} tex={tex:.2f} gm={gm:.2f}{coh_str}"
        )

        font_scale = 0.5
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), (0, 0, 0), -1)
        cv2.putText(
            vis, label, (x1 + 2, ty),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, fill_alpha, vis, 1.0 - fill_alpha, 0, vis)
    return vis


def save_blob_overlay_video(
    video_path: str,
    frame_blob_map: dict[int, list[dict[str, Any]]],
    out_dir: Path,
    *,
    output_fps: float = 5.0,
    max_blobs_per_frame: int = 30,
    filename: str = "motion_blobs_overlay.mp4",
    debug: bool = False,
) -> Path:
    """Write MP4 with motion-region bboxes on sampled frames (debug mode only)."""
    if not debug:
        raise ValueError(
            "Blob overlay requires debug=True; use input_plan_overlay for production viz"
        )
    frame_indices = sorted(frame_blob_map.keys())
    if not frame_indices:
        raise ValueError("No frames with blobs to visualize")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    max_target = frame_indices[-1]
    target_set = set(frame_indices)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, output_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create video writer: {out_path}")

    frame_idx = 0
    while frame_idx <= max_target:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in target_set:
            blobs = frame_blob_map.get(frame_idx, [])
            vis = draw_blobs_on_frame(frame, blobs, max_labels=max_blobs_per_frame)
            writer.write(vis)
        frame_idx += 1

    writer.release()
    cap.release()
    return out_path
