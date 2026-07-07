"""Object motion tube feature computation and importance scoring."""

from __future__ import annotations

from typing import Any

import numpy as np

from motion_analyzer.v2.bbox_utils import min_max_normalize
from motion_analyzer.v2.object_tube_builder import ObjectTubeSegment

EPS = 1e-8

OBJECT_TUBE_IMPORTANCE_WEIGHTS = {
    "mean_object_motion_density": 0.20,
    "motion_persistence": 0.15,
    "mean_object_motion_coverage": 0.15,
    "mean_speed": 0.15,
    "scale_score_mean": 0.15,
    "novelty_score_mean": 0.10,
    "mean_detector_confidence": 0.10,
}


def compute_object_tube_features(
    tube: ObjectTubeSegment,
    *,
    total_sampled_frames: int,
) -> dict[str, Any]:
    obs = tube.observations
    densities = [float(o.get("object_motion_density", 0)) for o in obs]
    coverages = [float(o.get("object_motion_coverage", 0)) for o in obs]
    confs = [float(o.get("detector_confidence", 0)) for o in obs]
    scales = [float(o.get("scale_score", 0)) for o in obs]
    area_ratios = [float(o.get("bbox_area_ratio", 0)) for o in obs]

    start_frame = int(obs[0]["frame_idx"])
    end_frame = int(obs[-1]["frame_idx"])
    start_time = float(obs[0]["timestamp_sec"])
    end_time = float(obs[-1]["timestamp_sec"])
    duration = max(end_time - start_time, 1.0 / max(total_sampled_frames, 1))
    num_visible = len(obs)
    num_moving = sum(
        1 for o in obs if float(o.get("object_motion_density", 0)) >= 0.02
    )

    centers = [(float(o["center_x"]), float(o["center_y"])) for o in obs]
    trajectory = 0.0
    for i in range(1, len(centers)):
        trajectory += float(np.hypot(
            centers[i][0] - centers[i - 1][0],
            centers[i][1] - centers[i - 1][1],
        ))
    mean_speed = trajectory / max(duration, EPS)
    motion_persistence = num_moving / max(num_visible, 1)

    novelty_scores = [
        float(o.get("mean_associated_blob_importance", 0)) for o in obs
    ]

    return {
        "object_tube_id": tube.object_tube_id,
        "class_name": tube.class_name,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_time_sec": round(start_time, 4),
        "end_time_sec": round(end_time, 4),
        "duration_sec": round(duration, 4),
        "num_visible_frames": num_visible,
        "num_moving_frames": num_moving,
        "mean_detector_confidence": round(float(np.mean(confs)), 6),
        "mean_object_motion_density": round(float(np.mean(densities)), 6),
        "max_object_motion_density": round(float(max(densities)), 6),
        "mean_object_motion_coverage": round(float(np.mean(coverages)), 6),
        "motion_persistence": round(motion_persistence, 6),
        "trajectory_length": round(trajectory, 2),
        "mean_speed": round(mean_speed, 4),
        "mean_bbox_area_ratio": round(float(np.mean(area_ratios)), 8),
        "scale_score_mean": round(float(np.mean(scales)), 6),
        "novelty_score_mean": round(float(np.mean(novelty_scores)), 6),
        "object_tube_importance": 0.0,
        "is_stationary": num_moving == 0,
    }


def add_object_tube_importance(tubes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not tubes:
        return tubes
    keys = list(OBJECT_TUBE_IMPORTANCE_WEIGHTS.keys())
    norm = {k: min_max_normalize([t[k] for t in tubes]) for k in keys}
    for i, tube in enumerate(tubes):
        imp = sum(OBJECT_TUBE_IMPORTANCE_WEIGHTS[k] * norm[k][i] for k in keys)
        if tube.get("is_stationary"):
            imp *= 0.3
        tube["object_tube_importance"] = round(imp, 6)
    return tubes


def object_tubes_to_rows(
    segments: list[ObjectTubeSegment],
    *,
    total_sampled_frames: int,
) -> list[dict[str, Any]]:
    features = [
        compute_object_tube_features(t, total_sampled_frames=total_sampled_frames)
        for t in segments
    ]
    return add_object_tube_importance(features)
