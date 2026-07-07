"""Motion evidence scoring for ByteTrack object tracks (motion-first)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from motion_analyzer.v2.bbox_utils import (
    bbox_iou,
    center_inside,
    expand_bbox,
    min_max_normalize,
    union_bbox,
)
from motion_analyzer.v2.object_detector import CLASS_WEIGHTS

logger = logging.getLogger(__name__)

OBSERVED_MOTION = "observed_motion"
WEAK_MOTION = "weak_motion"
DETECTOR_ONLY = "detector_only"

MOTION_MAP_THRESHOLD = 0.05
BLOB_IOU_THRESHOLD = 0.05
EXPAND_MARGIN = 0.30

# Frame-level observation thresholds
OBSERVED_MIN_DENSITY = 0.35
OBSERVED_MIN_COVERAGE = 0.04
WEAK_MEAN_RATIO = 2.5
WEAK_COVERAGE_MIN = 0.03
WEAK_PEAK_RATIO = 2.5
WEAK_INSIDE_MEAN_RATIO = 1.35
WEAK_PEAK_TO_MEAN_MIN = 1.45

# Camera / wind shake gating (frame-level)
SHAKE_GLOBAL_SCORE = 0.42
SHAKE_RESIDUAL_AREA = 0.18
SHAKE_DISCOUNT = 0.55

# Track-level moving conditions (general)
COND_A_OBSERVED_RATIO = 0.15
COND_A_MEAN_DENSITY = 0.55
COND_A_MEAN_COVERAGE = 0.035

COND_B_NORM_DISPLACEMENT = 0.06
COND_B_MEAN_SPEED = 15.0
COND_B_MAX_JITTER = 4.0

VEHICLE_MOVING_MIN_DISPLACEMENT = 0.06
VEHICLE_JITTER_DISPLACEMENT_CAP = 0.04

COND_C_MAX_DENSITY = 1.8
COND_C_MIN_OBSERVED_FRAMES = 2
COND_C_MIN_DURATION_SEC = 0.4

# Vehicle stationary
VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle", "bicycle"})
VEHICLE_STATIONARY_NORM_DISP = 0.02
VEHICLE_STATIONARY_OBSERVED_RATIO = 0.10
VEHICLE_STATIONARY_SPEED = 10.0
VEHICLE_MIN_VISIBLE_FRAMES = 8

# Person (more lenient)
PERSON_OBSERVED_RATIO = 0.08
PERSON_NORM_DISPLACEMENT = 0.015
PERSON_SMALL_SPEED = 6.0
PERSON_MIN_WEAK_FRAMES = 2

IMPORTANCE_WEIGHTS = {
    "mean_motion_density": 0.30,
    "observed_motion_ratio": 0.20,
    "mean_motion_coverage": 0.15,
    "normalized_track_displacement": 0.15,
    "mean_speed": 0.10,
    "class_weight": 0.10,
}


def _class_weight(class_name: str) -> float:
    return CLASS_WEIGHTS.get(class_name, 0.5)


def _region_motion_stats(
    motion_map: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    frame_mean: float,
) -> dict[str, float]:
    h, w = motion_map.shape[:2]
    x1c = max(0, min(x1, w - 1))
    y1c = max(0, min(y1, h - 1))
    x2c = max(x1c + 1, min(x2, w))
    y2c = max(y1c + 1, min(y2, h))
    region = motion_map[y1c:y2c, x1c:x2c]
    if region.size == 0:
        return {
            "motion_inside_mean": 0.0,
            "motion_inside_sum": 0.0,
            "motion_coverage": 0.0,
            "motion_density": 0.0,
            "motion_peak": 0.0,
            "peak_to_mean_ratio": 0.0,
        }
    mean_v = float(region.mean())
    sum_v = float(region.sum())
    peak_v = float(region.max())
    coverage = float((region > MOTION_MAP_THRESHOLD).mean())
    density = mean_v / max(frame_mean, 1e-6)
    peak_ratio = peak_v / max(mean_v, 1e-6)
    return {
        "motion_inside_mean": round(mean_v, 6),
        "motion_inside_sum": round(sum_v, 4),
        "motion_coverage": round(coverage, 6),
        "motion_density": round(density, 6),
        "motion_peak": round(peak_v, 6),
        "peak_to_mean_ratio": round(peak_ratio, 6),
    }


def _associate_blobs(track_box: dict, blobs: list[dict]) -> list[dict]:
    matched = []
    for blob in blobs:
        if bbox_iou(track_box, blob) >= BLOB_IOU_THRESHOLD or center_inside(blob, track_box):
            matched.append(blob)
    return matched


def _frame_shake_score(frame_row: pd.Series | dict) -> float:
    """0–1 score: high when global/camera motion likely dominates the frame."""
    if isinstance(frame_row, pd.Series):
        row = frame_row.to_dict()
    else:
        row = frame_row
    gm = float(row.get("global_motion_score", 0.0))
    res = float(row.get("residual_motion_area_ratio", 1.0))
    cam = bool(row.get("camera_motion_flag", False))
    if cam:
        return min(1.0, 0.55 + gm * 0.45)
    if gm >= SHAKE_GLOBAL_SCORE and res <= SHAKE_RESIDUAL_AREA:
        return min(1.0, gm)
    return 0.0


def _is_widespread_camera_shake(
    inside_stats: dict[str, float],
    expanded_stats: dict[str, float],
    shake_score: float,
    associated_blobs: list[dict],
) -> bool:
    """Reject diffuse neighborhood motion typical of wind / tripod shake."""
    if shake_score < 0.35:
        return False
    if associated_blobs:
        strong_blob = any(
            float(b.get("blob_importance", 0.0)) >= 0.22
            and float(b.get("flow_direction_coherence", 0.0)) >= 0.35
            for b in associated_blobs
        )
        if strong_blob:
            return False
    if expanded_stats["motion_coverage"] > 0.14 and expanded_stats["peak_to_mean_ratio"] < 1.75:
        return True
    if inside_stats["motion_coverage"] < 0.04 and expanded_stats["motion_coverage"] > 0.12:
        return True
    return False


def _is_static_background_pattern(
    inside_stats: dict[str, float],
    expanded_stats: dict[str, float],
) -> bool:
    """Neighborhood motion high but bbox interior flat → parked object / background."""
    in_mean = inside_stats["motion_inside_mean"]
    exp_mean = expanded_stats["motion_inside_mean"]
    if exp_mean <= 0:
        return False
    return in_mean < exp_mean * 0.55 and inside_stats["motion_coverage"] < 0.025


def _is_uniform_flicker(stats: dict[str, float]) -> bool:
    """Low peak-to-mean → diffuse flicker, not object motion."""
    return stats["peak_to_mean_ratio"] < WEAK_PEAK_TO_MEAN_MIN and stats["motion_coverage"] < 0.06


def _classify_observation(
    inside_stats: dict[str, float],
    expanded_stats: dict[str, float],
    associated_blobs: list[dict],
    frame_mean: float,
    *,
    shake_score: float = 0.0,
) -> str:
    if _is_widespread_camera_shake(inside_stats, expanded_stats, shake_score, associated_blobs):
        return DETECTOR_ONLY

    if associated_blobs:
        if shake_score > 0.4:
            blob_motion = any(
                float(b.get("blob_importance", 0.0)) >= 0.2
                and float(b.get("flow_direction_coherence", 0.0)) >= 0.4
                for b in associated_blobs
            )
            if blob_motion:
                return OBSERVED_MOTION
        if (
            inside_stats["motion_density"] >= OBSERVED_MIN_DENSITY
            or inside_stats["motion_coverage"] >= OBSERVED_MIN_COVERAGE
        ):
            return OBSERVED_MOTION

    region = expanded_stats
    region_mean = region["motion_inside_mean"]
    if _is_static_background_pattern(inside_stats, expanded_stats):
        return DETECTOR_ONLY
    if _is_uniform_flicker(region):
        return DETECTOR_ONLY

    shake_factor = 1.0 + shake_score * SHAKE_DISCOUNT
    weak_ok = (
        region_mean > frame_mean * WEAK_MEAN_RATIO * shake_factor
        and region["motion_coverage"] > WEAK_COVERAGE_MIN
        and region["motion_peak"] > frame_mean * WEAK_PEAK_RATIO * shake_factor
        and inside_stats["motion_inside_mean"] > frame_mean * WEAK_INSIDE_MEAN_RATIO * shake_factor
        and region["peak_to_mean_ratio"] >= WEAK_PEAK_TO_MEAN_MIN
    )
    if weak_ok:
        return WEAK_MOTION

    if (
        inside_stats["motion_density"] > 0.5
        and inside_stats["motion_coverage"] > 0.05
        and inside_stats["motion_inside_mean"] > frame_mean * 1.8
    ):
        return WEAK_MOTION

    return DETECTOR_ONLY


def _compute_displacement_metrics(
    centers: list[tuple[float, float]],
    frame_w: int,
    frame_h: int,
    duration_sec: float,
) -> dict[str, float]:
    if len(centers) < 2:
        return {
            "track_displacement_px": 0.0,
            "normalized_track_displacement": 0.0,
            "trajectory_length": 0.0,
            "mean_speed": 0.0,
            "median_speed": 0.0,
            "bbox_jitter_score": 0.0,
        }

    steps = [
        float(np.hypot(centers[i][0] - centers[i - 1][0], centers[i][1] - centers[i - 1][1]))
        for i in range(1, len(centers))
    ]
    trajectory = float(sum(steps))
    net_disp = float(np.hypot(centers[-1][0] - centers[0][0], centers[-1][1] - centers[0][1]))
    diag = float(np.hypot(frame_w, frame_h))
    dur = max(duration_sec, 0.2)
    jitter = trajectory / max(net_disp, 1.0)

    return {
        "track_displacement_px": round(net_disp, 2),
        "normalized_track_displacement": round(net_disp / max(diag, 1.0), 6),
        "trajectory_length": round(trajectory, 2),
        "mean_speed": round(trajectory / dur, 4),
        "median_speed": round(float(np.median(steps)) * 5.0, 4) if steps else 0.0,
        "bbox_jitter_score": round(jitter, 4),
    }


def _is_stationary_vehicle(feat: dict[str, Any]) -> bool:
    if feat["class_name"] not in VEHICLE_CLASSES:
        return False

    low_disp = feat["normalized_track_displacement"] < VEHICLE_STATIONARY_NORM_DISP
    low_speed = feat["mean_speed"] < VEHICLE_STATIONARY_SPEED
    long_visible = feat["num_visible_frames"] >= VEHICLE_MIN_VISIBLE_FRAMES

    if long_visible and feat["normalized_track_displacement"] < 0.015 and low_speed:
        return True

    if (
        feat["bbox_jitter_score"] > 5.0
        and feat["normalized_track_displacement"] < VEHICLE_JITTER_DISPLACEMENT_CAP
    ):
        return True

    return long_visible and low_disp and feat["observed_motion_ratio"] < VEHICLE_STATIONARY_OBSERVED_RATIO and low_speed


def _vehicle_moving_criteria(feat: dict[str, Any]) -> bool:
    if feat["num_visible_frames"] < 2:
        return False
    if feat["normalized_track_displacement"] < VEHICLE_MOVING_MIN_DISPLACEMENT:
        if not (
            feat["max_motion_density"] >= COND_C_MAX_DENSITY * 1.5
            and feat["num_observed_motion_frames"] >= COND_C_MIN_OBSERVED_FRAMES
            and feat["duration_sec"] >= COND_C_MIN_DURATION_SEC
            and feat["normalized_track_displacement"] >= 0.04
        ):
            return False
    return _condition_a(feat) or _condition_b(feat) or _condition_c(feat)


def _condition_a(feat: dict[str, Any]) -> bool:
    return (
        feat["observed_motion_ratio"] >= COND_A_OBSERVED_RATIO
        and feat["mean_motion_density"] >= COND_A_MEAN_DENSITY
        and feat["mean_motion_coverage"] >= COND_A_MEAN_COVERAGE
    )


def _condition_b(feat: dict[str, Any]) -> bool:
    return (
        feat["normalized_track_displacement"] >= COND_B_NORM_DISPLACEMENT
        and feat["mean_speed"] >= COND_B_MEAN_SPEED
        and feat["bbox_jitter_score"] <= COND_B_MAX_JITTER
    )


def _condition_c(feat: dict[str, Any]) -> bool:
    return (
        feat["max_motion_density"] >= COND_C_MAX_DENSITY
        and feat["num_observed_motion_frames"] >= COND_C_MIN_OBSERVED_FRAMES
        and feat["duration_sec"] >= COND_C_MIN_DURATION_SEC
    )


def _person_moving_criteria(feat: dict[str, Any]) -> bool:
    if feat.get("num_observed_motion_frames", 0) >= 2:
        return True
    if (
        feat.get("associated_motion_blob_count", 0) >= 2
        and feat.get("mean_associated_blob_importance", 0.0) >= 0.18
    ):
        return True
    if feat["observed_motion_ratio"] >= PERSON_OBSERVED_RATIO:
        return True
    if feat["normalized_track_displacement"] >= PERSON_NORM_DISPLACEMENT:
        return True
    if (
        feat["num_weak_motion_frames"] >= PERSON_MIN_WEAK_FRAMES
        and feat["mean_speed"] > PERSON_SMALL_SPEED
        and feat["num_observed_motion_frames"] >= 1
    ):
        return True
    return False


def _classify_moving_track(feat: dict[str, Any]) -> tuple[bool, bool, str]:
    """
    Returns (is_moving_track, is_stationary_track, track_state).
    weak_motion alone never confirms moving for vehicles.
    """
    if _is_stationary_vehicle(feat):
        return False, True, "stationary_vehicle"

    class_name = feat["class_name"]
    if class_name == "person":
        if not _person_moving_criteria(feat):
            return False, True, "stationary_person"
        if _condition_a(feat) or _condition_b(feat) or _condition_c(feat):
            return True, False, "moving_person"
        if feat["observed_motion_ratio"] >= PERSON_OBSERVED_RATIO:
            return True, False, "moving_person"
        if feat["normalized_track_displacement"] >= PERSON_NORM_DISPLACEMENT:
            return True, False, "moving_person"
        if (
            feat["num_weak_motion_frames"] >= PERSON_MIN_WEAK_FRAMES
            and feat["mean_speed"] > PERSON_SMALL_SPEED
            and feat["num_observed_motion_frames"] >= 1
        ):
            return True, False, "moving_person"
        return False, True, "stationary_person"

    if class_name in VEHICLE_CLASSES:
        if not _vehicle_moving_criteria(feat):
            return False, True, "stationary_vehicle" if _is_stationary_vehicle(feat) else "weak_only_stationary"
        return True, False, "moving_vehicle"

    if _condition_a(feat) or _condition_b(feat) or _condition_c(feat):
        return True, False, "moving_object"

    if feat["num_observed_motion_frames"] == 0:
        return False, True, "weak_only_stationary"

    return False, True, "stationary_object"


def enrich_tracks_with_motion(
    tracks_df: pd.DataFrame,
    motion_maps: np.ndarray,
    frame_df: pd.DataFrame,
    blobs_df: pd.DataFrame,
    *,
    frame_w: int,
    frame_h: int,
) -> tuple[pd.DataFrame, dict[int, list[dict]]]:
    """Add per-frame motion evidence; return enriched df and track obs map."""
    if tracks_df.empty:
        return tracks_df, {}

    frame_to_map_idx = {
        int(row["frame_idx"]): i for i, (_, row) in enumerate(frame_df.iterrows())
    }
    frame_means = (
        frame_df["motion_magnitude_mean"].to_dict()
        if "motion_magnitude_mean" in frame_df
        else {}
    )
    frame_rows = {
        int(row["frame_idx"]): row for _, row in frame_df.iterrows()
    }

    blobs_by_frame: dict[int, list[dict]] = {}
    for fi in blobs_df["frame_idx"].unique():
        blobs_by_frame[int(fi)] = blobs_df[blobs_df["frame_idx"] == fi].to_dict("records")

    enriched_rows: list[dict] = []
    track_obs_map: dict[int, list[dict]] = {}

    for _, tr in tracks_df.iterrows():
        rec = tr.to_dict()
        fi = int(rec["frame_idx"])
        map_idx = frame_to_map_idx.get(fi)
        if map_idx is None or map_idx >= len(motion_maps):
            rec["observation_type"] = DETECTOR_ONLY
            enriched_rows.append(rec)
            continue

        motion_map = motion_maps[map_idx]
        frame_mean = float(frame_means.get(fi, float(motion_map.mean())))
        shake_score = _frame_shake_score(frame_rows.get(fi, {}))

        x1, y1, x2, y2 = int(rec["x1"]), int(rec["y1"]), int(rec["x2"]), int(rec["y2"])
        inside = _region_motion_stats(motion_map, x1, y1, x2, y2, frame_mean)
        ex1, ey1, ex2, ey2 = expand_bbox(x1, y1, x2, y2, EXPAND_MARGIN, frame_w, frame_h)
        expanded = _region_motion_stats(motion_map, ex1, ey1, ex2, ey2, frame_mean)

        blobs = blobs_by_frame.get(fi, [])
        associated = _associate_blobs(rec, blobs)
        obs_type = _classify_observation(
            inside, expanded, associated, frame_mean, shake_score=shake_score
        )

        blob_ids = ",".join(str(int(b.get("blob_id_in_frame", 0))) for b in associated)
        blob_imp = (
            float(np.mean([b.get("blob_importance", 0) for b in associated]))
            if associated
            else 0.0
        )

        rec.update({
            "observation_type": obs_type,
            "motion_inside_mean": inside["motion_inside_mean"],
            "motion_inside_sum": inside["motion_inside_sum"],
            "motion_coverage": inside["motion_coverage"],
            "motion_density": inside["motion_density"],
            "motion_peak": inside["motion_peak"],
            "expanded_motion_mean": expanded["motion_inside_mean"],
            "expanded_motion_coverage": expanded["motion_coverage"],
            "associated_blob_ids": blob_ids,
            "associated_blob_count": len(associated),
            "mean_associated_blob_importance": round(blob_imp, 6),
        })
        enriched_rows.append(rec)

        tid = int(rec["track_id"])
        if tid >= 0:
            obs = dict(rec)
            if associated:
                ubox = union_bbox(associated)
                obs["associated_motion_bbox"] = [
                    int(ubox["x1"]), int(ubox["y1"]), int(ubox["x2"]), int(ubox["y2"]),
                ]
            track_obs_map.setdefault(tid, []).append(obs)

    return pd.DataFrame(enriched_rows), track_obs_map


def aggregate_track_features(
    enriched_tracks: pd.DataFrame,
    track_obs_map: dict[int, list[dict]],
    *,
    frame_w: int,
    frame_h: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Build per-track features and split into moving / stationary.

    Returns (all_track_features, moving_tracks, stationary_tracks).
    """
    if enriched_tracks.empty:
        return [], [], []

    valid = enriched_tracks[enriched_tracks["track_id"] >= 0]
    all_features: list[dict[str, Any]] = []

    for track_id, group in valid.groupby("track_id"):
        group = group.sort_values("frame_idx")
        obs_list = track_obs_map.get(int(track_id), [])
        class_name = str(group["class_name"].mode().iloc[0])

        num_visible = len(group)
        num_observed = sum(1 for o in obs_list if o.get("observation_type") == OBSERVED_MOTION)
        num_weak = sum(1 for o in obs_list if o.get("observation_type") == WEAK_MOTION)
        num_detector_only = sum(1 for o in obs_list if o.get("observation_type") == DETECTOR_ONLY)

        observed_ratio = num_observed / max(num_visible, 1)
        weak_ratio = num_weak / max(num_visible, 1)

        inside_means = [float(o.get("motion_inside_mean", 0)) for o in obs_list]
        coverages = [float(o.get("motion_coverage", 0)) for o in obs_list]
        densities = [float(o.get("motion_density", 0)) for o in obs_list]
        blob_counts = [int(o.get("associated_blob_count", 0)) for o in obs_list]
        blob_imps = [
            float(o.get("mean_associated_blob_importance", 0))
            for o in obs_list
            if o.get("associated_blob_count", 0) > 0
        ]

        centers = [(float(r["center_x"]), float(r["center_y"])) for _, r in group.iterrows()]
        start_time = float(group.iloc[0]["timestamp_sec"])
        end_time = float(group.iloc[-1]["timestamp_sec"])
        duration = max(end_time - start_time, 0.2)
        disp = _compute_displacement_metrics(centers, frame_w, frame_h, duration)

        feat: dict[str, Any] = {
            "track_id": int(track_id),
            "class_name": class_name,
            "start_frame": int(group.iloc[0]["frame_idx"]),
            "end_frame": int(group.iloc[-1]["frame_idx"]),
            "start_time_sec": round(start_time, 4),
            "end_time_sec": round(end_time, 4),
            "duration_sec": round(duration, 4),
            "num_visible_frames": num_visible,
            "num_observed_motion_frames": num_observed,
            "num_weak_motion_frames": num_weak,
            "num_detector_only_frames": num_detector_only,
            "observed_motion_ratio": round(observed_ratio, 6),
            "weak_motion_ratio": round(weak_ratio, 6),
            "mean_motion_inside_bbox": round(float(np.mean(inside_means)), 6) if inside_means else 0.0,
            "max_motion_inside_bbox": round(float(max(inside_means)), 6) if inside_means else 0.0,
            "mean_motion_coverage": round(float(np.mean(coverages)), 6) if coverages else 0.0,
            "mean_motion_density": round(float(np.mean(densities)), 6) if densities else 0.0,
            "max_motion_density": round(float(max(densities)), 6) if densities else 0.0,
            "associated_motion_blob_count": int(sum(blob_counts)),
            "mean_associated_blob_importance": round(float(np.mean(blob_imps)) if blob_imps else 0.0, 6),
            "track_displacement_px": disp["track_displacement_px"],
            "normalized_track_displacement": disp["normalized_track_displacement"],
            "trajectory_length": disp["trajectory_length"],
            "mean_speed": disp["mean_speed"],
            "median_speed": disp["median_speed"],
            "bbox_jitter_score": disp["bbox_jitter_score"],
            "class_weight": _class_weight(class_name),
            "track_motion_importance": 0.0,
            "is_stationary_track": False,
            "is_moving_track": False,
            "track_state": "unknown",
        }

        is_moving, is_stationary, state = _classify_moving_track(feat)
        feat["is_moving_track"] = is_moving
        feat["is_stationary_track"] = is_stationary
        feat["track_state"] = state
        all_features.append(feat)

    all_features = _add_track_importance(all_features)
    moving = [t for t in all_features if t["is_moving_track"]]
    stationary = [t for t in all_features if t["is_stationary_track"]]
    logger.info(
        "Track classification: %d total, %d moving, %d stationary",
        len(all_features), len(moving), len(stationary),
    )
    return all_features, moving, stationary


def _add_track_importance(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not tracks:
        return tracks

    moving = [t for t in tracks if t.get("is_moving_track")]
    if not moving:
        for t in tracks:
            t["track_motion_importance"] = 0.0
        return tracks

    norm_keys = list(IMPORTANCE_WEIGHTS.keys())
    norm = {k: min_max_normalize([t.get(k, 0) for t in moving]) for k in norm_keys}

    for i, t in enumerate(tracks):
        if not t.get("is_moving_track"):
            t["track_motion_importance"] = 0.0
            continue
        mi = moving.index(t)
        imp = sum(IMPORTANCE_WEIGHTS[k] * norm[k][mi] for k in norm_keys)
        t["track_motion_importance"] = round(max(0.0, imp), 6)
    return tracks


def export_moving_object_tracks(
    enriched_tracks: pd.DataFrame,
    moving_tracks: list[dict[str, Any]],
) -> pd.DataFrame:
    """Frame-level rows for confirmed moving tracks only."""
    if enriched_tracks.empty or not moving_tracks:
        return pd.DataFrame()
    moving_ids = {t["track_id"] for t in moving_tracks}
    return enriched_tracks[enriched_tracks["track_id"].isin(moving_ids)].copy()


def export_stationary_object_tracks(
    all_features: list[dict[str, Any]],
) -> pd.DataFrame:
    """Track-level rows for stationary / rejected tracks."""
    stationary = [t for t in all_features if t.get("is_stationary_track")]
    return pd.DataFrame(stationary) if stationary else pd.DataFrame()
