"""Background jitter / camera-shake suppression on compensated residual motion."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

EPS = 1e-8


@dataclass
class JitterSuppressConfig:
    """Heuristics for wide coherent low-residual motion (wind / camera shake)."""

    motion_threshold: float = 0.05
    large_area_ratio: float = 0.22
    min_flow_coherence: float = 0.68
    moderate_mag_max: float = 0.18
    low_residual_mean: float = 0.10
    low_residual_area_ratio: float = 0.12
    local_peak_percentile: float = 90.0
    local_min_peak: float = 0.07
    local_max_component_area_ratio: float = 0.06
    suppress_gain: float = 0.02


def flow_direction_coherence(flow: np.ndarray) -> float:
    """Global mean resultant length of flow vectors in [0, 1]."""
    if flow is None or flow.size == 0:
        return 0.0
    fx = flow[..., 0].astype(np.float64).ravel()
    fy = flow[..., 1].astype(np.float64).ravel()
    mag = np.sqrt(fx * fx + fy * fy)
    active = mag > EPS
    if active.sum() < 16:
        return 0.0
    angles = np.arctan2(fy[active], fx[active])
    mean_cos = float(np.cos(angles).mean())
    mean_sin = float(np.sin(angles).mean())
    return float(np.sqrt(mean_cos * mean_cos + mean_sin * mean_sin))


def _area_ratio_above(map_arr: np.ndarray, threshold: float) -> float:
    return float((map_arr > threshold).sum()) / max(map_arr.size, 1)


def detect_background_jitter(
    original_map: np.ndarray,
    residual_map: np.ndarray,
    *,
    original_flow: np.ndarray | None = None,
    residual_flow: np.ndarray | None = None,
    config: JitterSuppressConfig | None = None,
) -> tuple[bool, float]:
    """
    Return (is_background_jitter, score).

    Jitter when motion is wide + coherent + moderate magnitude, but compensation
  leaves little residual (whole-frame wind/camera shake, not local objects).
    """
    cfg = config or JitterSuppressConfig()
    coverage = _area_ratio_above(original_map, cfg.motion_threshold)
    flow_for_coherence = original_flow if original_flow is not None else residual_flow
    coherence = flow_direction_coherence(flow_for_coherence)
    mag_mean = float(original_map.mean())
    res_mean = float(residual_map.mean())
    res_area = _area_ratio_above(residual_map, cfg.motion_threshold)

    wide = coverage >= cfg.large_area_ratio
    coherent = coherence >= cfg.min_flow_coherence
    moderate = mag_mean <= cfg.moderate_mag_max
    low_residual = res_mean <= cfg.low_residual_mean and res_area <= cfg.low_residual_area_ratio

    score = (
        0.30 * min(1.0, coverage / max(cfg.large_area_ratio, EPS))
        + 0.30 * coherence
        + 0.20 * (1.0 - min(1.0, mag_mean / max(cfg.moderate_mag_max, EPS)))
        + 0.20 * (1.0 - min(1.0, res_mean / max(cfg.low_residual_mean, EPS)))
    )

    is_jitter = bool(wide and coherent and moderate and low_residual)
    return is_jitter, round(score, 6)


def _keep_compact_local_motion(
    residual_map: np.ndarray,
    *,
    config: JitterSuppressConfig,
) -> np.ndarray:
    """Keep spatially localized compact residual blobs; zero background jitter."""
    cfg = config or JitterSuppressConfig()
    h, w = residual_map.shape
    max_area = int(h * w * cfg.local_max_component_area_ratio)
    peak = float(np.percentile(residual_map, cfg.local_peak_percentile))
    thresh = max(cfg.local_min_peak, peak)

    binary = (residual_map >= thresh).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    kept = np.zeros_like(residual_map, dtype=np.float32)
    for label_id in range(1, n_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area <= max_area:
            mask = labels == label_id
            kept[mask] = residual_map[mask]
    return kept


def suppress_background_jitter(
    original_map: np.ndarray,
    residual_map: np.ndarray,
    *,
    original_flow: np.ndarray | None = None,
    residual_flow: np.ndarray | None = None,
    config: JitterSuppressConfig | None = None,
) -> tuple[np.ndarray, bool, float]:
    """
    Suppress frame-wide jitter on residual motion while keeping compact local motion.

    Returns (suppressed_map, jitter_flag, jitter_score).
    """
    cfg = config or JitterSuppressConfig()
    is_jitter, score = detect_background_jitter(
        original_map,
        residual_map,
        original_flow=original_flow,
        residual_flow=residual_flow,
        config=cfg,
    )
    if not is_jitter:
        return residual_map.astype(np.float32), False, score

    local = _keep_compact_local_motion(residual_map, config=cfg)
    suppressed = (cfg.suppress_gain * residual_map + local).astype(np.float32)
    lo, hi = float(suppressed.min()), float(suppressed.max())
    if hi - lo > EPS:
        suppressed = ((suppressed - lo) / (hi - lo)).astype(np.float32)
    return suppressed, True, score
