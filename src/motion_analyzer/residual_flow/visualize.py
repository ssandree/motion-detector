"""Visualization for Farneback residual-flow artifacts."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from motion_analyzer.residual_flow.flow import ResidualFlowVideoResult

logger = logging.getLogger(__name__)


def flow_to_bgr(flow: np.ndarray, *, saturate_percentile: float = 95.0) -> np.ndarray:
    """Convert HxWx2 flow to HSV color-wheel BGR image."""
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).astype(np.float32)
    ang = np.arctan2(flow[..., 1], flow[..., 0]).astype(np.float32)
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ((ang + np.pi) / (2.0 * np.pi) * 179.0).astype(np.uint8)
    hsv[..., 1] = 255
    vmax = float(np.percentile(mag, saturate_percentile)) if mag.size else 1.0
    vmax = max(vmax, 1e-3)
    hsv[..., 2] = np.clip(mag / vmax * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def overlay_flow_on_frame(
    frame_bgr: np.ndarray,
    flow: np.ndarray,
    *,
    alpha: float = 0.55,
) -> np.ndarray:
    color = flow_to_bgr(flow)
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, color, alpha, 0.0)


def draw_ransac_inliers(
    frame_bgr: np.ndarray,
    sample_xy: np.ndarray,
    inlier_mask: np.ndarray | None,
    *,
    flow: np.ndarray | None = None,
    arrow_scale: float = 1.0,
) -> np.ndarray:
    """Draw sampled correspondences; green=inlier, red=outlier/unknown."""
    out = frame_bgr.copy()
    if sample_xy is None or sample_xy.size == 0:
        return out

    for i, (x, y) in enumerate(sample_xy):
        x_i, y_i = int(x), int(y)
        is_inlier = bool(inlier_mask[i]) if inlier_mask is not None and i < len(inlier_mask) else False
        color = (0, 220, 0) if is_inlier else (0, 0, 220)
        cv2.circle(out, (x_i, y_i), 2, color, -1, lineType=cv2.LINE_AA)
        if flow is not None and 0 <= y_i < flow.shape[0] and 0 <= x_i < flow.shape[1]:
            dx, dy = float(flow[y_i, x_i, 0]), float(flow[y_i, x_i, 1])
            x2 = int(round(x_i + dx * arrow_scale))
            y2 = int(round(y_i + dy * arrow_scale))
            cv2.arrowedLine(out, (x_i, y_i), (x2, y2), color, 1, tipLength=0.3)
    return out


def _annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    out = frame.copy()
    y = 22
    for line in lines:
        cv2.putText(
            out,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        y += 20
    return out


def save_residual_flow_overlays(
    result: ResidualFlowVideoResult,
    out_dir: str | Path,
    *,
    output_fps: float | None = None,
    alpha: float = 0.55,
) -> dict[str, str]:
    """Write raw/residual flow overlays and RANSAC inlier overlay MP4s."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not result.raw_flows:
        raise ValueError("No residual-flow pairs to visualize")

    fps = float(output_fps if output_fps is not None else result.config.process_fps)
    h, w = result.raw_flows[0].shape[:2]

    paths = {
        "raw": out / "raw_flow_overlay.mp4",
        "residual": out / "residual_flow_overlay.mp4",
        "ransac": out / "ransac_inlier_overlay.mp4",
    }
    writers = {
        key: cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(fps, 0.1),
            (w, h),
        )
        for key, path in paths.items()
    }
    for key, writer in writers.items():
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {paths[key]}")

    for i, frame_idx in enumerate(result.frame_indices_curr):
        frame = result.sampled_bgr.get(frame_idx)
        if frame is None:
            logger.warning("Missing sampled frame %d for overlay", frame_idx)
            continue
        if frame.shape[0] != h or frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        meta = result.global_motion_rows[i]
        raw_vis = overlay_flow_on_frame(frame, result.raw_flows[i], alpha=alpha)
        residual_vis = overlay_flow_on_frame(frame, result.residual_flows[i], alpha=alpha)
        ransac_vis = draw_ransac_inliers(
            frame,
            result.sample_points[i],
            result.inlier_masks[i],
            flow=result.raw_flows[i],
            arrow_scale=1.0,
        )

        tag = "COMP" if meta.compensation_applied else "RAW"
        info = [
            f"f={meta.frame_idx_curr}  {tag}  inlier={meta.inlier_ratio:.2f}",
            f"t={meta.translation_px:.2f}px  rot={meta.rotation_deg:.3f}deg  {meta.reason}",
        ]
        writers["raw"].write(_annotate(raw_vis, info))
        writers["residual"].write(_annotate(residual_vis, info))
        writers["ransac"].write(_annotate(ransac_vis, info))

    for writer in writers.values():
        writer.release()

    written = {k: str(v) for k, v in paths.items()}
    for key, path in paths.items():
        logger.info("Wrote %s (%d frames)", path.name, len(result.raw_flows))
    return {
        "raw_flow_overlay_mp4": written["raw"],
        "residual_flow_overlay_mp4": written["residual"],
        "ransac_inlier_overlay_mp4": written["ransac"],
    }
