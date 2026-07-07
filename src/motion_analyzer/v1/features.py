"""Frame-level motion feature extraction."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class FeatureConfig:
    motion_threshold: float = 0.05
    entropy_bins: int = 64
    small_component_max_area: int = 100
    new_motion_history_frames: int = 5
    new_motion_low_threshold: float = 0.02
    new_motion_high_threshold: float = 0.05
    scene_change_window: int = 10


@dataclass
class FeatureExtractor:
    """Stateful extractor maintaining history for temporal features."""

    config: FeatureConfig = field(default_factory=FeatureConfig)
    _motion_history: deque[np.ndarray] = field(init=False)
    _frame_diff_means: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        n = self.config.new_motion_history_frames
        w = self.config.scene_change_window
        self._motion_history = deque(maxlen=n)
        self._frame_diff_means = deque(maxlen=w)

    def reset(self) -> None:
        self._motion_history.clear()
        self._frame_diff_means.clear()

    def extract(
        self,
        motion_map: np.ndarray,
        frame_idx: int,
        timestamp_sec: float,
        frame_diff_mean: float,
    ) -> dict[str, float | int]:
        """Compute all frame-level features for one motion map."""
        cfg = self.config
        flat = motion_map.ravel()
        total_energy = float(flat.sum())
        n_pixels = flat.size

        motion_magnitude_mean = float(motion_map.mean())
        motion_magnitude_sum = total_energy

        mask = motion_map > cfg.motion_threshold
        motion_area_ratio = float(mask.sum()) / n_pixels

        motion_entropy = _spatial_entropy(motion_map, cfg.entropy_bins)
        motion_concentration = _top_fraction_energy(motion_map, fraction=0.10)
        small_motion_energy = _small_component_energy(
            motion_map, cfg.motion_threshold, cfg.small_component_max_area
        )
        new_motion_region_score = _new_motion_region_score(
            motion_map,
            list(self._motion_history),
            cfg.new_motion_low_threshold,
            cfg.new_motion_high_threshold,
        )
        scene_change_score = _scene_change_score(
            frame_diff_mean, list(self._frame_diff_means)
        )

        self._motion_history.append(motion_map.copy())
        self._frame_diff_means.append(frame_diff_mean)

        return {
            "frame_idx": frame_idx,
            "timestamp_sec": round(timestamp_sec, 4),
            "motion_magnitude_mean": round(motion_magnitude_mean, 6),
            "motion_magnitude_sum": round(motion_magnitude_sum, 4),
            "motion_area_ratio": round(motion_area_ratio, 6),
            "motion_entropy": round(motion_entropy, 6),
            "motion_concentration": round(motion_concentration, 6),
            "small_motion_energy": round(small_motion_energy, 6),
            "new_motion_region_score": round(new_motion_region_score, 6),
            "scene_change_score": round(scene_change_score, 6),
        }


def _spatial_entropy(motion_map: np.ndarray, n_bins: int) -> float:
    """Shannon entropy of the spatial motion distribution."""
    hist, _ = np.histogram(motion_map.ravel(), bins=n_bins, range=(0.0, 1.0), density=False)
    total = hist.sum()
    if total == 0:
        return 0.0
    probs = hist.astype(np.float64) / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _top_fraction_energy(motion_map: np.ndarray, fraction: float = 0.10) -> float:
    """Fraction of total energy contained in the top *fraction* of pixels."""
    flat = motion_map.ravel()
    total = flat.sum()
    if total <= 0:
        return 0.0
    k = max(1, int(np.ceil(fraction * flat.size)))
    top_k_sum = float(np.partition(flat, -k)[-k:].sum())
    return top_k_sum / total


def _small_component_energy(
    motion_map: np.ndarray,
    threshold: float,
    max_area: int,
) -> float:
    """Sum of motion energy in connected components smaller than *max_area*."""
    binary = (motion_map > threshold).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    small_energy = 0.0
    for label_id in range(1, n_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area <= max_area:
            component_mask = labels == label_id
            small_energy += float(motion_map[component_mask].sum())
    return small_energy


def _new_motion_region_score(
    current: np.ndarray,
    history: list[np.ndarray],
    low_threshold: float,
    high_threshold: float,
) -> float:
    """
    Measure motion appearing in regions that were previously quiet.

    Pixels where the historical mean motion was below *low_threshold* but
    current motion exceeds *high_threshold* contribute their current values.
    Score is normalized by total motion energy.
    """
    if not history:
        return 0.0

    hist_stack = np.stack(history, axis=0)
    hist_mean = hist_stack.mean(axis=0)
    new_region_mask = (hist_mean < low_threshold) & (current > high_threshold)
    new_energy = float(current[new_region_mask].sum())
    total = float(current.sum())
    if total <= 0:
        return 0.0
    return new_energy / total


def _scene_change_score(frame_diff_mean: float, history: list[float]) -> float:
    """
    Detect abrupt increases in overall frame difference.

    Returns the ratio of current frame-diff mean to the recent historical
    mean (excluding current). Values >> 1 indicate a scene cut.
    """
    if not history:
        return 0.0
    hist_mean = float(np.mean(history))
    if hist_mean < 1e-6:
        return float(frame_diff_mean)
    return max(0.0, frame_diff_mean / hist_mean - 1.0)


def extract_all_features(
    motion_maps: list[np.ndarray],
    frame_indices: list[int],
    timestamps: list[float],
    frame_diff_means: list[float],
    config: FeatureConfig | None = None,
    global_motion_rows: list[dict] | None = None,
) -> list[dict]:
    """Extract features for all motion maps in a video."""
    extractor = FeatureExtractor(config=config or FeatureConfig())
    features = []
    for i, (motion_map, fidx, ts, fd_mean) in enumerate(
        zip(motion_maps, frame_indices, timestamps, frame_diff_means)
    ):
        row = extractor.extract(motion_map, fidx, ts, fd_mean)
        if global_motion_rows and i < len(global_motion_rows):
            row.update(global_motion_rows[i])
        features.append(row)
    return features
