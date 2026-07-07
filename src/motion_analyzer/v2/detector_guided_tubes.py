"""Detector-guided motion tubes: motion-first tubes enriched with detector context."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from motion_analyzer.v1.tube_builder import TubeSegment
from motion_analyzer.v2.bbox_utils import (
    expand_bbox,
    min_max_normalize,
    union_bbox,
)
from motion_analyzer.v2.blob_detector_context import (
    PERSON_CLASS,
    STATIC_CLASSES,
    VEHICLE_CLASSES,
)
from motion_analyzer.v2.gap_filling_tube_tracker import (
    DETECTOR_SUPPORTED,
    OBSERVED_MOTION,
    GapFillingTubeConfig,
    build_gap_filling_tubes,
    compute_track_continuity_metrics,
)

PERSON_MARGIN = 0.25
VEHICLE_MARGIN = 0.20
NO_ASSOC_MARGIN = 0.50

IMPORTANCE_WEIGHTS = {
    "mean_motion_density": 0.25,
    "motion_persistence": 0.18,
    "track_continuity": 0.12,
    "spatial_novelty_mean": 0.15,
    "scale_score_mean": 0.12,
    "mean_speed": 0.08,
    "mean_detector_context_score": 0.10,
}
STATIC_TEXTURE_PENALTY = 0.15


def _parse_classes(s: str) -> list[str]:
    if not s or (isinstance(s, float) and np.isnan(s)):
        return []
    return [c.strip() for c in str(s).split(",") if c.strip()]


def _obs_static_texture_score(obs: dict) -> float:
    if not obs.get("motion_observed", True):
        return 0.0
    tex = float(obs.get("texture_flicker_score", 0.0))
    edge = float(obs.get("edge_density", 0.0))
    disp = float(obs.get("centroid_displacement", 0.0))
    static_overlap = any(c in STATIC_CLASSES for c in _parse_classes(obs.get("associated_classes", "")))
    score = tex * 0.5 + (edge if disp < 3.0 else 0.0) * 0.3
    if static_overlap:
        score += 0.3
    return min(1.0, score)


def _best_detector_box_for_obs(obs: dict, detections_by_frame: dict[int, list[dict]]) -> dict | None:
    if obs.get("context_detector_bbox"):
        bb = obs["context_detector_bbox"]
        return {"x1": bb[0], "y1": bb[1], "x2": bb[2], "y2": bb[3]}

    frame_idx = int(obs["frame_idx"])
    dets = detections_by_frame.get(frame_idx, [])
    if not dets:
        return None

    classes = _parse_classes(obs.get("associated_classes", ""))
    if not classes:
        return None

    det_ids = obs.get("associated_det_ids", "")
    id_set = {int(x) for x in str(det_ids).split(",") if x.strip().isdigit()}

    candidates = [d for d in dets if int(d["det_id_in_frame"]) in id_set]
    if not candidates:
        candidates = [d for d in dets if d["class_name"] in classes]
    if not candidates:
        return None

    priority = {PERSON_CLASS: 3, **{c: 2 for c in VEHICLE_CLASSES}}
    candidates.sort(
        key=lambda d: (priority.get(d["class_name"], 1), float(d.get("confidence", 0))),
        reverse=True,
    )
    return candidates[0]


def _primary_bbox(obs: dict) -> dict[str, float]:
    if obs.get("motion_observed") and obs.get("motion_density", 0) > 0:
        return {"x1": obs["x1"], "y1": obs["y1"], "x2": obs["x2"], "y2": obs["y2"]}
    if obs.get("predicted_bbox_coords"):
        pc = obs["predicted_bbox_coords"]
        return {"x1": pc[0], "y1": pc[1], "x2": pc[2], "y2": pc[3]}
    return {"x1": obs["x1"], "y1": obs["y1"], "x2": obs["x2"], "y2": obs["y2"]}


def compute_roi_crop_bbox(
    obs: dict,
    *,
    detections_by_frame: dict[int, list[dict]],
    frame_w: int,
    frame_h: int,
) -> tuple[list[int] | None, list[int] | None, list[int], list[int] | None]:
    """
  Returns motion_bbox, predicted_bbox, crop_bbox, context_detector_bbox.
    """
    obs_type = obs.get("observation_type", OBSERVED_MOTION)
    primary = _primary_bbox(obs)
    det = _best_detector_box_for_obs(obs, detections_by_frame)
    classes = _parse_classes(obs.get("associated_classes", ""))

    motion_bbox: list[int] | None = None
    predicted_bbox: list[int] | None = None
    context_bbox: list[int] | None = None

    if obs_type == OBSERVED_MOTION:
        motion_bbox = [int(primary["x1"]), int(primary["y1"]), int(primary["x2"]), int(primary["y2"])]
    elif obs.get("predicted_bbox_coords"):
        predicted_bbox = [int(x) for x in obs["predicted_bbox_coords"]]

    if det is not None:
        context_bbox = [int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])]
    elif obs.get("context_detector_bbox"):
        context_bbox = list(obs["context_detector_bbox"])

    if obs_type == OBSERVED_MOTION:
        crop_base = dict(primary)
        if PERSON_CLASS in classes and det and det["class_name"] == PERSON_CLASS:
            crop_base = union_bbox([primary, det])
            margin = PERSON_MARGIN
        elif any(c in VEHICLE_CLASSES for c in classes) and det and det["class_name"] in VEHICLE_CLASSES:
            crop_base = union_bbox([primary, det])
            margin = VEHICLE_MARGIN
        else:
            margin = NO_ASSOC_MARGIN
    elif obs_type == DETECTOR_SUPPORTED:
        crop_base = union_bbox([primary, det]) if det else primary
        margin = PERSON_MARGIN if PERSON_CLASS in classes else VEHICLE_MARGIN
    else:
        crop_base = primary
        margin = NO_ASSOC_MARGIN

    crop = expand_bbox(
        crop_base["x1"], crop_base["y1"], crop_base["x2"], crop_base["y2"],
        margin, frame_w, frame_h,
    )
    return motion_bbox, predicted_bbox, crop, context_bbox


def _compute_tube_scalar_features(
    seg: TubeSegment,
    frame_to_seq: dict[int, int],
    *,
    duration_sec: float,
) -> dict[str, Any]:
    obs = seg.observations
    observed = [o for o in obs if o.get("motion_observed")]

    densities = [float(o.get("motion_density", 0)) for o in observed] or [0.0]
    novelties = [float(o.get("spatial_novelty_score", 0)) for o in observed] or [0.0]
    scales = [float(o.get("scale_score", 0)) for o in observed] or [0.0]
    area_ratios = [float(o.get("bbox_area_ratio", 0)) for o in observed] or [0.0]

    start_frame = int(obs[0]["frame_idx"])
    end_frame = int(obs[-1]["frame_idx"])
    start_time = float(obs[0]["timestamp_sec"])
    end_time = float(obs[-1]["timestamp_sec"])
    duration = max(end_time - start_time, 0.2)

    centers = [(float(o["center_x"]), float(o["center_y"])) for o in observed] or [
        (float(obs[0]["center_x"]), float(obs[0]["center_y"]))
    ]
    trajectory = 0.0
    for i in range(1, len(centers)):
        trajectory += float(np.hypot(centers[i][0] - centers[i - 1][0], centers[i][1] - centers[i - 1][1]))
    mean_speed = trajectory / max(duration, 1e-6)

    continuity = compute_track_continuity_metrics(obs, frame_to_seq)

    ctx_scores = [float(o.get("detector_context_score", 0)) for o in observed]
    static_tex = [_obs_static_texture_score(o) for o in obs]

    all_classes: set[str] = set()
    for o in obs:
        all_classes.update(_parse_classes(o.get("associated_classes", "")))

    return {
        "tube_id": seg.tube_id,
        "motion_tube_id": seg.tube_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_time_sec": round(start_time, 4),
        "end_time_sec": round(end_time, 4),
        "duration_sec": round(duration, 4),
        "num_frames": len(obs),
        "mean_bbox_area_ratio": round(float(np.mean(area_ratios)), 8),
        "mean_motion_density": round(float(np.mean(densities)), 6),
        "max_motion_density": round(float(max(densities)), 6),
        "spatial_novelty_mean": round(float(np.mean(novelties)), 6),
        "scale_score_mean": round(float(np.mean(scales)), 6),
        "mean_speed": round(mean_speed, 4),
        "trajectory_length": round(trajectory, 2),
        "mean_detector_context_score": round(float(np.mean(ctx_scores)) if ctx_scores else 0.0, 4),
        "max_detector_context_score": round(float(max(ctx_scores)) if ctx_scores else 0.0, 4),
        "mean_static_texture_score": round(float(np.mean(static_tex)), 4),
        "associated_classes": ",".join(sorted(all_classes)) if all_classes else "",
        "num_moving_frames": sum(1 for o in observed if float(o.get("motion_density", 0)) >= 0.02),
        "has_detector_association": any(ctx_scores),
        **continuity,
        "detector_guided_motion_importance": 0.0,
    }


def build_detector_guided_motion_tubes(
    blobs_ctx_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    detections_df: pd.DataFrame,
    *,
    frame_w: int,
    frame_h: int,
    duration_sec: float,
    gap_config: GapFillingTubeConfig | None = None,
) -> tuple[list[dict[str, Any]], list[TubeSegment], dict[int, list[dict]]]:
    """Build gap-filled motion tubes and score with motion-first importance."""
    cfg = gap_config or GapFillingTubeConfig()

    dets_by_frame: dict[int, list[dict]] = {}
    if detections_df is not None and not detections_df.empty:
        for fi in detections_df["frame_idx"].unique():
            dets_by_frame[int(fi)] = (
                detections_df[detections_df["frame_idx"] == fi].to_dict("records")
            )

    segments = build_gap_filling_tubes(
        blobs_ctx_df,
        frame_df,
        dets_by_frame,
        frame_w=frame_w,
        frame_h=frame_h,
        config=cfg,
    )

    frame_to_seq = {
        int(fi): i for i, fi in enumerate(frame_df["frame_idx"].tolist())
    }

    tube_rows: list[dict[str, Any]] = []
    obs_map: dict[int, list[dict]] = {}

    for seg in segments:
        observed = [o for o in seg.observations if o.get("motion_observed")]
        if not observed:
            continue
        if not any(float(o.get("motion_density", 0)) >= 0.02 for o in observed):
            continue

        row = _compute_tube_scalar_features(seg, frame_to_seq, duration_sec=duration_sec)
        tube_rows.append(row)
        obs_map[seg.tube_id] = seg.observations

    tube_rows = _add_detector_guided_importance(tube_rows)
    return tube_rows, segments, obs_map


def _add_detector_guided_importance(tubes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not tubes:
        return tubes

    pos_keys = list(IMPORTANCE_WEIGHTS.keys())
    norm_pos = {k: min_max_normalize([t[k] for t in tubes]) for k in pos_keys}
    norm_static = min_max_normalize([t["mean_static_texture_score"] for t in tubes])

    for i, tube in enumerate(tubes):
        imp = sum(IMPORTANCE_WEIGHTS[k] * norm_pos[k][i] for k in pos_keys)
        imp -= STATIC_TEXTURE_PENALTY * norm_static[i]
        if tube["num_observed_motion_frames"] == 0:
            imp *= 0.05
        elif tube["motion_persistence"] < 0.15:
            imp *= 0.5
        tube["detector_guided_motion_importance"] = round(max(0.0, imp), 6)

    return tubes


def enrich_tube_observations_with_crop(
    observations: list[dict],
    detections_by_frame: dict[int, list[dict]],
    *,
    frame_w: int,
    frame_h: int,
) -> list[dict]:
    """Add motion_bbox, predicted_bbox, context_detector_bbox, crop_bbox."""
    enriched = []
    for obs in observations:
        motion_bb, pred_bb, crop_bb, ctx_bb = compute_roi_crop_bbox(
            obs,
            detections_by_frame=detections_by_frame,
            frame_w=frame_w,
            frame_h=frame_h,
        )
        enriched.append({
            **obs,
            "motion_bbox": motion_bb,
            "predicted_bbox": pred_bb,
            "context_detector_bbox": ctx_bb,
            "crop_bbox": crop_bb,
        })
    return enriched


def tubes_context_summary(
    tube_rows: list[dict[str, Any]],
    motion_tubes_df,
) -> list[dict[str, Any]]:
    """Merge v1 motion_tubes.csv fields with detector context aggregates."""
    ctx_by_id = {t["motion_tube_id"]: t for t in tube_rows}
    rows = []
    for _, row in motion_tubes_df.iterrows():
        tid = int(row["tube_id"])
        rec = row.to_dict()
        ctx = ctx_by_id.get(tid, {})
        for key in (
            "mean_detector_context_score", "associated_classes",
            "detector_guided_motion_importance", "motion_persistence",
            "track_continuity", "num_observed_motion_frames",
            "num_detector_supported_frames", "num_interpolated_frames",
        ):
            if key in ctx:
                rec[key] = ctx[key]
        rows.append(rec)
    return rows
