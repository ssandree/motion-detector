"""Motion tube feature computation and importance scoring."""

from __future__ import annotations

from typing import Any

import numpy as np

from motion_analyzer.v1.tube_builder import TubeSegment

EPS = 1e-8

TUBE_IMPORTANCE_WEIGHTS = {
    "mean_motion_density": 0.20,
    "max_blob_importance": 0.15,
    "temporal_persistence": 0.15,
    "spatial_novelty_mean": 0.15,
    "scale_score_mean": 0.15,
    "motion_peak": 0.10,
    "mean_speed": 0.10,
}


def _min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < EPS:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def compute_tube_features(
    tube: TubeSegment,
    *,
    total_sampled_frames: int,
    video_duration_sec: float,
    small_tube_area_threshold: float = 0.002,
) -> dict[str, Any]:
    """Compute scalar features for one motion tube."""
    obs = tube.observations
    densities = [o["motion_density"] for o in obs]
    importances = [o["blob_importance"] for o in obs]
    energies = [o["motion_energy"] for o in obs]
    area_ratios = [o["bbox_area_ratio"] for o in obs]
    novelties = [o["spatial_novelty_score"] for o in obs]
    scales = [o["scale_score"] for o in obs]

    start_frame = int(obs[0]["frame_idx"])
    end_frame = int(obs[-1]["frame_idx"])
    start_time = float(obs[0]["timestamp_sec"])
    end_time = float(obs[-1]["timestamp_sec"])
    duration = max(end_time - start_time, 1.0 / max(total_sampled_frames, 1))
    num_frames = len(obs)

    centers = [(o["center_x"], o["center_y"]) for o in obs]
    trajectory = 0.0
    for i in range(1, len(centers)):
        trajectory += float(
            np.hypot(centers[i][0] - centers[i - 1][0], centers[i][1] - centers[i - 1][1])
        )

    mean_speed = trajectory / max(duration, EPS)
    temporal_persistence = num_frames / max(total_sampled_frames, 1)
    motion_peak = float(max(densities))
    motion_variance = float(np.var(densities))
    mean_area = float(np.mean(area_ratios))

    return {
        "tube_id": tube.tube_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_time_sec": round(start_time, 4),
        "end_time_sec": round(end_time, 4),
        "duration_sec": round(duration, 4),
        "num_frames": num_frames,
        "mean_bbox_area_ratio": round(mean_area, 8),
        "mean_motion_energy": round(float(np.mean(energies)), 4),
        "mean_motion_density": round(float(np.mean(densities)), 6),
        "max_motion_density": round(float(max(densities)), 6),
        "mean_blob_importance": round(float(np.mean(importances)), 6),
        "max_blob_importance": round(float(max(importances)), 6),
        "temporal_persistence": round(temporal_persistence, 6),
        "motion_peak": round(motion_peak, 6),
        "motion_variance": round(motion_variance, 6),
        "trajectory_length": round(trajectory, 2),
        "mean_speed": round(mean_speed, 4),
        "spatial_novelty_mean": round(float(np.mean(novelties)), 6),
        "scale_score_mean": round(float(np.mean(scales)), 6),
        "small_tube_flag": mean_area < small_tube_area_threshold,
        "tube_importance": 0.0,
    }


def add_tube_importance(tubes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add tube_importance via video-wide min-max normalization."""
    if not tubes:
        return tubes

    norm_keys = list(TUBE_IMPORTANCE_WEIGHTS.keys())
    normalized = {
        key: _min_max_normalize([t[key] for t in tubes]) for key in norm_keys
    }

    for i, tube in enumerate(tubes):
        importance = sum(
            TUBE_IMPORTANCE_WEIGHTS[key] * normalized[key][i] for key in norm_keys
        )
        tube["tube_importance"] = round(importance, 6)

    return tubes


def tubes_to_feature_rows(
    tube_segments: list[TubeSegment],
    *,
    total_sampled_frames: int,
    video_duration_sec: float,
    small_tube_area_threshold: float = 0.002,
) -> list[dict[str, Any]]:
    """Convert tube segments to feature dicts with importance."""
    features = [
        compute_tube_features(
            tube,
            total_sampled_frames=total_sampled_frames,
            video_duration_sec=video_duration_sec,
            small_tube_area_threshold=small_tube_area_threshold,
        )
        for tube in tube_segments
    ]
    return add_tube_importance(features)
