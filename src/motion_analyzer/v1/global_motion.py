"""Global (camera) motion estimation, compensation, and layer decomposition."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from motion_analyzer.v1.background_jitter import (
    JitterSuppressConfig,
    flow_direction_coherence,
    suppress_background_jitter,
)

EPS = 1e-8


@dataclass
class GlobalMotionConfig:
    max_corners: int = 500
    quality_level: float = 0.01
    min_distance: int = 8
    ransac_threshold: float = 3.0
    min_inliers: int = 8


@dataclass
class GlobalMotionResult:
    dx: float
    dy: float
    scale: float
    rotation: float
    global_motion_score: float
    residual_motion_mean: float
    residual_motion_area_ratio: float
    camera_motion_flag: bool
    affine_matrix: np.ndarray | None
    inlier_ratio: float
    global_flow_method: str = "none"
    original_motion_mean: float = 0.0
    global_motion_mean: float = 0.0
    flow_direction_coherence: float = 0.0
    background_jitter_flag: bool = False
    background_jitter_score: float = 0.0
    jitter_suppression_applied: bool = False


@dataclass
class MotionCompensationLayers:
    """Per frame-pair motion maps used for blob selection and debug overlays."""

    original_map: np.ndarray
    global_motion_map: np.ndarray
    residual_map: np.ndarray
    suppressed_map: np.ndarray
    original_flow: np.ndarray | None = None
    global_flow: np.ndarray | None = None
    residual_flow: np.ndarray | None = None
    metadata: GlobalMotionResult = field(default_factory=GlobalMotionResult)


def _decompose_partial_affine(M: np.ndarray) -> tuple[float, float, float, float]:
    a, b, tx = float(M[0, 0]), float(M[0, 1]), float(M[0, 2])
    c, d, ty = float(M[1, 0]), float(M[1, 1]), float(M[1, 2])
    dx, dy = tx, ty
    scale = float(np.sqrt(max(a * a + c * c, EPS)))
    rotation = float(np.arctan2(c, a))
    return dx, dy, scale, rotation


def affine_flow_field(h: int, w: int, M: np.ndarray) -> np.ndarray:
    """Dense (H×W×2) flow induced by applying partial affine *M* to each pixel."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = M[0, 0] * xx + M[0, 1] * yy + M[0, 2]
    ny = M[1, 0] * xx + M[1, 1] * yy + M[1, 2]
    return np.stack([nx - xx, ny - yy], axis=-1).astype(np.float32)


_affine_flow_field = affine_flow_field


def estimate_global_affine(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    config: GlobalMotionConfig | None = None,
) -> tuple[np.ndarray | None, float]:
    cfg = config or GlobalMotionConfig()
    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=cfg.max_corners,
        qualityLevel=cfg.quality_level,
        minDistance=cfg.min_distance,
    )
    if prev_pts is None or len(prev_pts) < cfg.min_inliers:
        return None, 0.0

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    if curr_pts is None or status is None:
        return None, 0.0

    mask = status.ravel() == 1
    if mask.sum() < cfg.min_inliers:
        return None, float(mask.sum()) / max(len(prev_pts), 1)

    src = prev_pts[mask].reshape(-1, 2)
    dst = curr_pts[mask].reshape(-1, 2)
    M, inliers = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=cfg.ransac_threshold,
    )
    if M is None:
        return None, float(mask.sum()) / max(len(prev_pts), 1)

    inlier_ratio = float(inliers.sum()) / max(len(src), 1) if inliers is not None else 0.0
    return M.astype(np.float64), inlier_ratio


def _median_translation_flow(flow: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Fallback global motion: robust median flow vector (camera shake)."""
    dx = float(np.median(flow[..., 0]))
    dy = float(np.median(flow[..., 1]))
    h, w = flow.shape[:2]
    global_flow = np.zeros((h, w, 2), dtype=np.float32)
    global_flow[..., 0] = dx
    global_flow[..., 1] = dy
    return global_flow, dx, dy


def _global_motion_score(
    dx: float,
    dy: float,
    scale: float,
    rotation: float,
    inlier_ratio: float,
    frame_h: int,
    frame_w: int,
) -> float:
    diag = float(np.hypot(frame_h, frame_w))
    trans_norm = float(np.hypot(dx, dy)) / max(diag, 1.0)
    scale_dev = abs(scale - 1.0)
    rot_dev = abs(rotation)
    motion_mag = min(1.0, trans_norm * 8.0 + scale_dev * 4.0 + rot_dev * 2.0)
    return round(min(1.0, 0.55 * inlier_ratio + 0.45 * motion_mag), 6)


def _residual_stats(residual_map: np.ndarray, threshold: float = 0.05) -> tuple[float, float]:
    mean = float(residual_map.mean())
    area_ratio = float((residual_map > threshold).sum()) / max(residual_map.size, 1)
    return round(mean, 6), round(area_ratio, 6)


def _flow_magnitude_map(flow: np.ndarray, blur_ksize: int = 7, blur_sigma: float = 1.5) -> np.ndarray:
    from motion_analyzer.v1.pixel_motion import _normalize_map

    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).astype(np.float32)
    if blur_ksize > 1:
        mag = cv2.GaussianBlur(mag, (blur_ksize, blur_ksize), blur_sigma)
    return _normalize_map(mag)


def compute_motion_compensation_layers(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    *,
    config: GlobalMotionConfig | None = None,
    jitter_config: JitterSuppressConfig | None = None,
    motion_threshold: float = 0.05,
    camera_score_threshold: float = 0.45,
    camera_residual_area_threshold: float = 0.08,
    flow_scale: float = 0.5,
    blur_ksize: int = 7,
    blur_sigma: float = 1.5,
    alpha: float = 0.6,
    beta: float = 0.4,
) -> MotionCompensationLayers:
    """
    Full compensation pipeline:
    1) affine from matched points, else median-flow translation
    2) residual_flow = original_flow - global_motion_flow
    3) suppress background jitter on residual while keeping compact local motion
    """
    from motion_analyzer.v1.pixel_motion import _normalize_map, compute_frame_diff, compute_optical_flow

    h, w = prev_gray.shape
    cfg = config or GlobalMotionConfig()
    jcfg = jitter_config or JitterSuppressConfig(motion_threshold=motion_threshold)

    original_diff = compute_frame_diff(prev_gray, curr_gray, blur_ksize, blur_sigma)
    original_flow = compute_optical_flow(prev_gray, curr_gray, scale=flow_scale)
    original_flow_mag = _flow_magnitude_map(original_flow, blur_ksize, blur_sigma)
    original_map = _normalize_map(alpha * original_diff + beta * original_flow_mag)

    M, inlier_ratio = estimate_global_affine(prev_gray, curr_gray, cfg)
    method = "affine"
    dx = dy = 0.0
    scale = 1.0
    rotation = 0.0

    if M is not None:
        global_flow = affine_flow_field(h, w, M)
        dx, dy, scale, rotation = _decompose_partial_affine(M)
    else:
        # Affine failed → median flow vector fallback (wind / mild shake).
        global_flow, dx, dy = _median_translation_flow(original_flow)
        M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)
        inlier_ratio = 0.0
        method = "median_flow"

    global_motion_map = _flow_magnitude_map(global_flow, blur_ksize, blur_sigma)
    residual_flow = (original_flow - global_flow).astype(np.float32)

    aligned_prev = cv2.warpAffine(prev_gray, M, (w, h), flags=cv2.INTER_LINEAR)
    diff = cv2.absdiff(aligned_prev, curr_gray).astype(np.float32)
    if blur_ksize > 1:
        diff = cv2.GaussianBlur(diff, (blur_ksize, blur_ksize), blur_sigma)
    residual_diff = _normalize_map(diff)
    residual_flow_mag = _flow_magnitude_map(residual_flow, blur_ksize, blur_sigma)
    residual_map = _normalize_map(alpha * residual_diff + beta * residual_flow_mag)

    suppressed_map, jitter_flag, jitter_score = suppress_background_jitter(
        original_map,
        residual_map,
        original_flow=original_flow,
        residual_flow=residual_flow,
        config=jcfg,
    )

    res_mean, res_area = _residual_stats(residual_map, motion_threshold)
    gscore = _global_motion_score(dx, dy, scale, rotation, inlier_ratio, h, w)
    camera_flag = gscore >= camera_score_threshold and res_area < camera_residual_area_threshold
    coherence = flow_direction_coherence(original_flow)

    metadata = GlobalMotionResult(
        dx=round(dx, 4),
        dy=round(dy, 4),
        scale=round(scale, 6),
        rotation=round(rotation, 6),
        global_motion_score=gscore,
        residual_motion_mean=res_mean,
        residual_motion_area_ratio=res_area,
        camera_motion_flag=camera_flag,
        affine_matrix=M,
        inlier_ratio=round(inlier_ratio, 6),
        global_flow_method=method,
        original_motion_mean=round(float(original_map.mean()), 6),
        global_motion_mean=round(float(global_motion_map.mean()), 6),
        flow_direction_coherence=round(coherence, 6),
        background_jitter_flag=jitter_flag,
        background_jitter_score=jitter_score,
        jitter_suppression_applied=jitter_flag,
    )

    return MotionCompensationLayers(
        original_map=original_map,
        global_motion_map=global_motion_map,
        residual_map=residual_map,
        suppressed_map=suppressed_map,
        original_flow=original_flow,
        global_flow=global_flow,
        residual_flow=residual_flow,
        metadata=metadata,
    )


def compensate_global_motion(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    *,
    config: GlobalMotionConfig | None = None,
    motion_threshold: float = 0.05,
    camera_score_threshold: float = 0.45,
    camera_residual_area_threshold: float = 0.08,
    flow_scale: float = 0.5,
) -> tuple[np.ndarray, np.ndarray | None, GlobalMotionResult]:
    """Backward-compatible API: returns final suppressed map + original flow."""
    layers = compute_motion_compensation_layers(
        prev_gray,
        curr_gray,
        config=config,
        motion_threshold=motion_threshold,
        camera_score_threshold=camera_score_threshold,
        camera_residual_area_threshold=camera_residual_area_threshold,
        flow_scale=flow_scale,
    )
    return layers.suppressed_map, layers.original_flow, layers.metadata
