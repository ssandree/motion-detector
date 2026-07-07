"""Video-level motion feature aggregation and motion-type classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

EPS = 1e-8

MotionType = Literal["background_motion", "event_motion"]

MOTION_TYPES: tuple[MotionType, ...] = ("background_motion", "event_motion")

# Legacy category names retained only for adaptive_input_plan compatibility.
LEGACY_CATEGORIES = [
    "Static / Sparse Local Motion",
    "Dense Global Motion",
    "Dense Local Motion",
    "Multi-motion / Competing Blobs",
    "Scene Change Dominant",
]


@dataclass
class MotionTypeConfig:
    """Thresholds for background_motion vs event_motion classification."""

    low_global_motion_level: float = 0.015
    low_motion_area_mean: float = 0.08
    camera_global_motion_score: float = 0.45
    low_residual_motion_area: float = 0.05
    camera_motion_frame_ratio: float = 0.5
    weak_tube_count: int = 3
    min_tube_importance_for_event: float = 0.2
    min_tube_frames_for_event: int = 2
    repetitive_translation_global_score: float = 0.4
    low_local_motion_importance: float = 0.3
    high_motion_concentration: float = 0.55
    multi_motion_event_score: float = 0.3
    min_event_tubes: int = 3


def compute_video_features(
    frame_df: pd.DataFrame,
    tubes: list[dict[str, Any]],
    blobs_df: pd.DataFrame,
    *,
    sampled_fps: float,
    scene_change_threshold: float = 1.0,
) -> dict[str, Any]:
    """Aggregate frame, blob, and tube statistics into video-level features."""
    duration_sec = float(frame_df["timestamp_sec"].max()) if len(frame_df) else 0.0
    num_sampled_frames = len(frame_df)

    global_motion_level = float(frame_df["motion_magnitude_mean"].mean())
    motion_area_mean = float(frame_df["motion_area_ratio"].mean())
    motion_area_variance = float(frame_df["motion_area_ratio"].var())

    burst_threshold = global_motion_level + frame_df["motion_magnitude_mean"].std()
    motion_burst_ratio = float(
        (frame_df["motion_magnitude_mean"] > burst_threshold).mean()
    )

    scene_change_count = int(
        (frame_df["scene_change_score"] > scene_change_threshold).sum()
    )

    num_motion_blobs = len(blobs_df)
    num_motion_tubes = len(tubes)

    if tubes:
        small_tube_ratio = sum(1 for t in tubes if t["small_tube_flag"]) / len(tubes)
        total_energy = sum(t["mean_motion_energy"] for t in tubes)
        total_importance = sum(t["tube_importance"] for t in tubes)
        top_energy = max(tubes, key=lambda t: t["mean_motion_energy"])
        top_importance = max(tubes, key=lambda t: t["tube_importance"])
        dominant_tube_energy_ratio = (
            top_energy["mean_motion_energy"] / max(total_energy, EPS)
        )
        dominant_tube_importance_ratio = (
            top_importance["tube_importance"] / max(total_importance, EPS)
        )
        small_importance = sum(
            t["tube_importance"] for t in tubes if t["small_tube_flag"]
        )
        local_motion_importance_ratio = small_importance / max(total_importance, EPS)
        max_tube_importance = float(max(t["tube_importance"] for t in tubes))
        strong_tube_count = sum(
            1
            for t in tubes
            if t["tube_importance"] >= 0.2 and t["num_frames"] >= 2
        )
    else:
        small_tube_ratio = 0.0
        dominant_tube_energy_ratio = 0.0
        dominant_tube_importance_ratio = 0.0
        local_motion_importance_ratio = 0.0
        max_tube_importance = 0.0
        strong_tube_count = 0

    multi_motion_score = _multi_motion_score(blobs_df)

    mean_global_motion_score = 0.0
    mean_residual_motion_area = 0.0
    camera_motion_frame_ratio = 0.0
    mean_motion_concentration = 0.0
    if not frame_df.empty and "global_motion_score" in frame_df.columns:
        mean_global_motion_score = float(frame_df["global_motion_score"].mean())
        mean_residual_motion_area = float(
            frame_df.get("residual_motion_area_ratio", pd.Series(dtype=float)).mean()
        )
        camera_motion_frame_ratio = float(
            frame_df.get("camera_motion_flag", pd.Series(dtype=bool)).astype(bool).mean()
        )
    if not frame_df.empty and "motion_concentration" in frame_df.columns:
        mean_motion_concentration = float(frame_df["motion_concentration"].mean())

    return {
        "duration_sec": round(duration_sec, 4),
        "sampled_fps": sampled_fps,
        "num_sampled_frames": num_sampled_frames,
        "global_motion_level": round(global_motion_level, 6),
        "motion_burst_ratio": round(motion_burst_ratio, 6),
        "motion_area_mean": round(motion_area_mean, 6),
        "motion_area_variance": round(motion_area_variance, 6),
        "num_motion_blobs": num_motion_blobs,
        "num_motion_tubes": num_motion_tubes,
        "small_tube_ratio": round(small_tube_ratio, 6),
        "dominant_tube_energy_ratio": round(dominant_tube_energy_ratio, 6),
        "dominant_tube_importance_ratio": round(dominant_tube_importance_ratio, 6),
        "multi_motion_score": round(multi_motion_score, 6),
        "scene_change_count": scene_change_count,
        "local_motion_importance_ratio": round(local_motion_importance_ratio, 6),
        "max_tube_importance": round(max_tube_importance, 6),
        "strong_tube_count": strong_tube_count,
        "mean_global_motion_score": round(mean_global_motion_score, 6),
        "mean_residual_motion_area": round(mean_residual_motion_area, 6),
        "camera_motion_frame_ratio": round(camera_motion_frame_ratio, 6),
        "mean_motion_concentration": round(mean_motion_concentration, 6),
    }


def _multi_motion_score(blobs_df: pd.DataFrame) -> float:
    """Average concurrent blob count normalized by frame count."""
    if blobs_df.empty:
        return 0.0
    counts = blobs_df.groupby("frame_idx").size()
    return float(counts.mean() / max(counts.max(), 1))


def classify_video_category(
    features: dict[str, Any],
    config: MotionTypeConfig | None = None,
) -> dict[str, Any]:
    """
    Rule-based motion type assignment (background_motion vs event_motion).

    Returns motion_type label, confidence score, and rule trace.
    """
    cfg = config or MotionTypeConfig()
    background_rules: list[tuple[float, str]] = []
    event_rules: list[tuple[float, str]] = []

    global_level = features["global_motion_level"]
    area_mean = features["motion_area_mean"]

    has_strong_event_tubes = (
        features["strong_tube_count"] >= cfg.min_event_tubes
        or features["max_tube_importance"] >= cfg.min_tube_importance_for_event + 0.15
    )

    if (
        global_level < cfg.low_global_motion_level
        and area_mean < cfg.low_motion_area_mean
        and not has_strong_event_tubes
    ):
        background_rules.append((0.75, "low overall motion magnitude and area"))

    camera_global_dominates = (
        features["mean_global_motion_score"] >= cfg.camera_global_motion_score
        and features["mean_residual_motion_area"] <= cfg.low_residual_motion_area
    )
    camera_frames_dominant = (
        features["camera_motion_frame_ratio"] >= cfg.camera_motion_frame_ratio
    )
    if (camera_global_dominates or camera_frames_dominant) and not has_strong_event_tubes:
        score = 0.65 + 0.2 * features["mean_global_motion_score"]
        reason = (
            "camera/global motion dominates with low residual motion"
            if camera_global_dominates
            else "majority of frames flagged as camera motion"
        )
        background_rules.append((min(score, 0.95), reason))

    weak_tubes = (
        features["num_motion_tubes"] < cfg.weak_tube_count
        or features["strong_tube_count"] == 0
        or features["max_tube_importance"] < cfg.min_tube_importance_for_event
    )
    if weak_tubes:
        background_rules.append((0.6, "valid motion tubes are weak or absent"))

    repetitive_translation = (
        features["mean_global_motion_score"] >= cfg.repetitive_translation_global_score
        and features["local_motion_importance_ratio"] <= cfg.low_local_motion_importance
        and area_mean >= cfg.low_motion_area_mean * 0.5
        and not has_strong_event_tubes
    )
    if repetitive_translation:
        background_rules.append(
            (0.7, "large consistent translation/global motion without local ROI need")
        )

    persistent_tubes = (
        features["strong_tube_count"] >= cfg.min_event_tubes
        or (
            features["num_motion_tubes"] >= cfg.min_event_tubes
            and features["max_tube_importance"] >= cfg.min_tube_importance_for_event
        )
    )
    if persistent_tubes:
        event_rules.append(
            (
                0.55 + features["max_tube_importance"] * 0.3,
                "persistent valid motion tubes",
            )
        )

    if features["mean_motion_concentration"] >= cfg.high_motion_concentration:
        event_rules.append(
            (
                0.5 + features["mean_motion_concentration"] * 0.3,
                "high motion concentration / local peaks",
            )
        )

    if (
        features["local_motion_importance_ratio"] > cfg.low_local_motion_importance
        and features["num_motion_tubes"] >= cfg.min_tube_frames_for_event
    ):
        event_rules.append(
            (
                0.5 + features["local_motion_importance_ratio"] * 0.25,
                "localized motion survives with meaningful tube importance",
            )
        )

    if (
        features["multi_motion_score"] > cfg.multi_motion_event_score
        and features["num_motion_tubes"] > cfg.weak_tube_count
    ):
        event_rules.append(
            (
                0.55 + features["multi_motion_score"] * 0.2,
                "multiple concurrent motion blobs/tubes",
            )
        )

    if not background_rules and not event_rules:
        event_rules.append((0.45, "default: motion present without clear background signal"))

    background_rules.sort(key=lambda r: r[0], reverse=True)
    event_rules.sort(key=lambda r: r[0], reverse=True)

    bg_score = background_rules[0][0] if background_rules else 0.0
    ev_score = event_rules[0][0] if event_rules else 0.0

    if background_rules and (not event_rules or bg_score >= ev_score):
        motion_type: MotionType = "background_motion"
        confidence, primary_reason = background_rules[0]
    else:
        motion_type = "event_motion"
        confidence, primary_reason = event_rules[0]

    all_candidates: list[dict[str, Any]] = []
    for score, reason in background_rules:
        all_candidates.append(
            {"motion_type": "background_motion", "score": round(score, 4), "reason": reason}
        )
    for score, reason in event_rules:
        all_candidates.append(
            {"motion_type": "event_motion", "score": round(score, 4), "reason": reason}
        )
    all_candidates.sort(key=lambda c: c["score"], reverse=True)

    return {
        "motion_type": motion_type,
        "confidence": round(min(confidence, 1.0), 4),
        "primary_reason": primary_reason,
        "all_candidates": all_candidates,
        "features_used": {
            "global_motion_level": features["global_motion_level"],
            "motion_area_mean": features["motion_area_mean"],
            "local_motion_importance_ratio": features["local_motion_importance_ratio"],
            "multi_motion_score": features["multi_motion_score"],
            "scene_change_count": features["scene_change_count"],
            "mean_global_motion_score": features["mean_global_motion_score"],
            "mean_residual_motion_area": features["mean_residual_motion_area"],
            "camera_motion_frame_ratio": features["camera_motion_frame_ratio"],
            "strong_tube_count": features["strong_tube_count"],
            "max_tube_importance": features["max_tube_importance"],
            "mean_motion_concentration": features["mean_motion_concentration"],
        },
    }


def legacy_category_for_input_plan(
    motion_type: MotionType,
    features: dict[str, Any],
) -> str:
    """
    Map motion_type to a legacy category label for adaptive_input_plan builders.

    Input-plan schema is unchanged in Task 1; this bridge keeps existing plans working.
    """
    if motion_type == "background_motion":
        if (
            features["global_motion_level"] > 0.03
            and features["motion_area_mean"] > 0.15
        ):
            return "Dense Global Motion"
        if features["scene_change_count"] >= 3:
            return "Scene Change Dominant"
        return "Static / Sparse Local Motion"

    if (
        features["multi_motion_score"] > 0.3
        and features["num_motion_tubes"] > 20
    ):
        return "Multi-motion / Competing Blobs"
    if (
        features["local_motion_importance_ratio"] > 0.5
        and features["num_motion_tubes"] > 5
    ):
        return "Dense Local Motion"
    if features["scene_change_count"] >= 3:
        return "Scene Change Dominant"
    return "Dense Local Motion"
