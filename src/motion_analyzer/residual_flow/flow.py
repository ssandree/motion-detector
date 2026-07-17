"""Farneback-based residual flow with RANSAC global affine compensation.

Pipeline (per process_fps frame pair):
  1. Farneback dense optical flow  → raw_flow
  2. Grid-sample correspondences (stride 8~16 px)
  3. estimateAffinePartial2D(RANSAC) → global camera affine
  4. camera_flow from affine; residual_flow = raw_flow - camera_flow
  5. Apply compensation only when shake is sufficient; else residual = raw
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from motion_analyzer.residual_flow.global_motion import affine_flow_field
from motion_analyzer.residual_flow.pixel_motion import compute_optical_flow, iter_sampled_frames

logger = logging.getLogger(__name__)

EPS = 1e-8
DEG_PER_RAD = 180.0 / np.pi


@dataclass
class ResidualFlowConfig:
    process_fps: float = 5.0
    flow_scale: float = 1.0
    sample_stride: int = 12
    ransac_reproj_threshold: float = 3.0
    min_sample_points: int = 16
    min_inlier_ratio: float = 0.3
    min_translation_px: float = 1.5
    min_rotation_deg: float = 0.2


@dataclass
class FramePairGlobalMotion:
    frame_idx_prev: int
    frame_idx_curr: int
    timestamp_sec_prev: float
    timestamp_sec_curr: float
    dx: float
    dy: float
    translation_px: float
    scale: float
    rotation_rad: float
    rotation_deg: float
    inlier_ratio: float
    num_samples: int
    num_inliers: int
    compensation_applied: bool
    reason: str
    affine_matrix: list[list[float]] | None = None


@dataclass
class ResidualFlowVideoResult:
    raw_flows: list[np.ndarray] = field(default_factory=list)
    residual_flows: list[np.ndarray] = field(default_factory=list)
    camera_flows: list[np.ndarray | None] = field(default_factory=list)
    sample_points: list[np.ndarray] = field(default_factory=list)
    inlier_masks: list[np.ndarray | None] = field(default_factory=list)
    global_motion_rows: list[FramePairGlobalMotion] = field(default_factory=list)
    frame_indices_curr: list[int] = field(default_factory=list)
    timestamps_curr: list[float] = field(default_factory=list)
    sampled_bgr: dict[int, np.ndarray] = field(default_factory=dict)
    config: ResidualFlowConfig = field(default_factory=ResidualFlowConfig)


def _decompose_partial_affine(M: np.ndarray) -> tuple[float, float, float, float]:
    a, b, tx = float(M[0, 0]), float(M[0, 1]), float(M[0, 2])
    c, d, ty = float(M[1, 0]), float(M[1, 1]), float(M[1, 2])
    dx, dy = tx, ty
    scale = float(np.sqrt(max(a * a + c * c, EPS)))
    rotation = float(np.arctan2(c, a))
    return dx, dy, scale, rotation


def sample_flow_correspondences(
    flow: np.ndarray,
    stride: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample (src, dst) correspondences from dense Farneback flow on a grid.

    Returns:
        src: (N, 2) float32 previous-frame points (x, y)
        dst: (N, 2) float32 current-frame points
        sample_xy: (N, 2) int32 grid sample locations used for indexing
    """
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"flow must be HxWx2, got {flow.shape}")
    h, w = flow.shape[:2]
    stride = max(1, int(stride))
    ys = np.arange(stride // 2, h, stride, dtype=np.int32)
    xs = np.arange(stride // 2, w, stride, dtype=np.int32)
    if ys.size == 0 or xs.size == 0:
        empty = np.zeros((0, 2), dtype=np.float32)
        return empty, empty, empty.astype(np.int32)

    xx, yy = np.meshgrid(xs, ys)
    sample_xy = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.int32)
    src = sample_xy.astype(np.float32)
    vec = flow[sample_xy[:, 1], sample_xy[:, 0]]
    dst = src + vec.astype(np.float32)
    return src, dst, sample_xy


def estimate_global_affine_from_flow(
    flow: np.ndarray,
    *,
    stride: int = 12,
    ransac_reproj_threshold: float = 3.0,
    min_sample_points: int = 16,
) -> tuple[np.ndarray | None, float, int, int, np.ndarray, np.ndarray | None]:
    """Estimate partial affine from grid-sampled Farneback correspondences.

    Returns:
        M, inlier_ratio, num_samples, num_inliers, sample_xy, inlier_mask
    """
    src, dst, sample_xy = sample_flow_correspondences(flow, stride=stride)
    num_samples = int(src.shape[0])
    if num_samples < max(3, int(min_sample_points)):
        return None, 0.0, num_samples, 0, sample_xy, None

    M, inliers = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_reproj_threshold),
        maxIters=2000,
        confidence=0.99,
    )
    if M is None or inliers is None:
        return None, 0.0, num_samples, 0, sample_xy, None

    inlier_mask = inliers.ravel().astype(bool)
    num_inliers = int(inlier_mask.sum())
    inlier_ratio = float(num_inliers) / max(num_samples, 1)
    return M.astype(np.float64), inlier_ratio, num_samples, num_inliers, sample_xy, inlier_mask


def should_apply_compensation(
    inlier_ratio: float,
    translation_px: float,
    rotation_deg: float,
    *,
    min_inlier_ratio: float = 0.3,
    min_translation_px: float = 1.5,
    min_rotation_deg: float = 0.2,
) -> tuple[bool, str]:
    """Gate: apply residual compensation only when shake is reliable + large enough."""
    if inlier_ratio < min_inlier_ratio:
        return False, f"inlier_ratio<{min_inlier_ratio}"
    motion_ok = translation_px >= min_translation_px or abs(rotation_deg) >= min_rotation_deg
    if not motion_ok:
        return (
            False,
            f"translation<{min_translation_px} and rotation<{min_rotation_deg}",
        )
    return True, "ok"


def compute_residual_flow_pair(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    *,
    config: ResidualFlowConfig | None = None,
    frame_idx_prev: int = 0,
    frame_idx_curr: int = 1,
    timestamp_sec_prev: float = 0.0,
    timestamp_sec_curr: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None, FramePairGlobalMotion]:
    """Compute raw/residual flow and global-motion metadata for one frame pair."""
    cfg = config or ResidualFlowConfig()
    h, w = prev_gray.shape[:2]

    raw_flow = compute_optical_flow(prev_gray, curr_gray, scale=cfg.flow_scale)
    M, inlier_ratio, num_samples, num_inliers, sample_xy, inlier_mask = (
        estimate_global_affine_from_flow(
            raw_flow,
            stride=cfg.sample_stride,
            ransac_reproj_threshold=cfg.ransac_reproj_threshold,
            min_sample_points=cfg.min_sample_points,
        )
    )

    dx = dy = 0.0
    scale = 1.0
    rotation_rad = 0.0
    translation_px = 0.0
    affine_list: list[list[float]] | None = None
    camera_flow: np.ndarray | None = None

    if M is not None:
        dx, dy, scale, rotation_rad = _decompose_partial_affine(M)
        translation_px = float(np.hypot(dx, dy))
        affine_list = M.tolist()
        camera_flow = affine_flow_field(h, w, M)

    rotation_deg = float(rotation_rad * DEG_PER_RAD)
    apply, reason = should_apply_compensation(
        inlier_ratio,
        translation_px,
        rotation_deg,
        min_inlier_ratio=cfg.min_inlier_ratio,
        min_translation_px=cfg.min_translation_px,
        min_rotation_deg=cfg.min_rotation_deg,
    )
    if M is None:
        reason = "affine_estimation_failed"
        apply = False

    if apply and camera_flow is not None:
        residual_flow = (raw_flow - camera_flow).astype(np.float32)
    else:
        residual_flow = raw_flow.copy()
        apply = False

    meta = FramePairGlobalMotion(
        frame_idx_prev=int(frame_idx_prev),
        frame_idx_curr=int(frame_idx_curr),
        timestamp_sec_prev=float(timestamp_sec_prev),
        timestamp_sec_curr=float(timestamp_sec_curr),
        dx=round(dx, 6),
        dy=round(dy, 6),
        translation_px=round(translation_px, 6),
        scale=round(scale, 6),
        rotation_rad=round(rotation_rad, 8),
        rotation_deg=round(rotation_deg, 6),
        inlier_ratio=round(float(inlier_ratio), 6),
        num_samples=int(num_samples),
        num_inliers=int(num_inliers),
        compensation_applied=bool(apply),
        reason=reason,
        affine_matrix=affine_list,
    )
    return raw_flow, residual_flow, camera_flow, sample_xy, inlier_mask, meta


def process_video_residual_flow(
    video_path: str,
    config: ResidualFlowConfig | None = None,
) -> ResidualFlowVideoResult:
    """Run residual-flow generation over a video at process_fps."""
    cfg = config or ResidualFlowConfig()
    result = ResidualFlowVideoResult(config=cfg)

    prev_bgr: np.ndarray | None = None
    prev_idx: int | None = None
    prev_ts: float | None = None

    for frame_idx, timestamp, frame in iter_sampled_frames(video_path, cfg.process_fps):
        result.sampled_bgr[frame_idx] = frame
        if prev_bgr is not None and prev_idx is not None and prev_ts is not None:
            prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            raw, residual, camera, sample_xy, inlier_mask, meta = compute_residual_flow_pair(
                prev_gray,
                curr_gray,
                config=cfg,
                frame_idx_prev=prev_idx,
                frame_idx_curr=frame_idx,
                timestamp_sec_prev=prev_ts,
                timestamp_sec_curr=timestamp,
            )
            result.raw_flows.append(raw)
            result.residual_flows.append(residual)
            result.camera_flows.append(camera)
            result.sample_points.append(sample_xy)
            result.inlier_masks.append(inlier_mask)
            result.global_motion_rows.append(meta)
            result.frame_indices_curr.append(frame_idx)
            result.timestamps_curr.append(timestamp)

        prev_bgr = frame
        prev_idx = frame_idx
        prev_ts = timestamp

    return result


def global_motion_to_dict(result: ResidualFlowVideoResult) -> dict[str, Any]:
    cfg = result.config
    applied = sum(1 for r in result.global_motion_rows if r.compensation_applied)
    return {
        "process_fps": cfg.process_fps,
        "flow_scale": cfg.flow_scale,
        "sample_stride": cfg.sample_stride,
        "ransac_reproj_threshold": cfg.ransac_reproj_threshold,
        "min_inlier_ratio": cfg.min_inlier_ratio,
        "min_translation_px": cfg.min_translation_px,
        "min_rotation_deg": cfg.min_rotation_deg,
        "num_frame_pairs": len(result.global_motion_rows),
        "num_compensation_applied": applied,
        "frames": [asdict(row) for row in result.global_motion_rows],
    }


def save_residual_flow_artifacts(
    result: ResidualFlowVideoResult,
    out_dir: str | Path,
) -> dict[str, str]:
    """Save raw_flow.npy, residual_flow.npy, global_motion.json under *out_dir*."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not result.raw_flows:
        raise ValueError("No residual-flow frame pairs to save")

    raw_path = out / "raw_flow.npy"
    residual_path = out / "residual_flow.npy"
    meta_path = out / "global_motion.json"

    np.save(raw_path, np.stack(result.raw_flows, axis=0).astype(np.float32))
    np.save(residual_path, np.stack(result.residual_flows, axis=0).astype(np.float32))
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(global_motion_to_dict(result), f, indent=2, ensure_ascii=False)

    logger.info(
        "Saved residual-flow artifacts: %s, %s, %s (%d pairs)",
        raw_path.name,
        residual_path.name,
        meta_path.name,
        len(result.raw_flows),
    )
    return {
        "raw_flow_npy": str(raw_path),
        "residual_flow_npy": str(residual_path),
        "global_motion_json": str(meta_path),
    }
