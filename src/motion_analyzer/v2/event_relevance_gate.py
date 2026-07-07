"""Event Relevance Gate — motion pattern classification and ROI relevance scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from motion_analyzer.v2.bbox_utils import bbox_iou, center_distance_norm

EPS = 1e-8

PERSON_CLASSES = frozenset({"person"})
VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle", "bicycle"})
PROXIMITY_DIST = 0.14
INTERACTION_DIST = 0.12
DOOR_TRUNK_MARGIN = 0.30
ENTRANCE_BOTTOM_RATIO = 0.22

MOTION_PATTERNS = (
    "background_jitter",
    "background_flow",
    "transit_motion",
    "ambiguous_motion",
    "event_motion",
)


@dataclass
class EventRelevanceGateConfig:
    """Thresholds for transit detection, event detection, and relevance scoring."""

    allow_ambiguous_roi: bool = False
    min_event_relevance_score: float = 0.10

    transit_net_displacement_min: float = 0.22
    transit_path_linearity_min: float = 0.65
    transit_direction_consistency_min: float = 0.68
    transit_stop_ratio_max: float = 0.18
    transit_dwell_near_object_max: float = 0.22
    transit_interaction_ratio_max: float = 0.12
    transit_semantic_anchor_max: float = 0.18

    stop_speed_threshold: float = 0.012
    dwell_move_threshold: float = 0.008
    stop_go_speed_cv_min: float = 0.45

    jitter_group_area: float = 0.40
    jitter_penalty_value: float = 0.35
    background_flow_coherence: float = 0.72
    background_flow_area_ratio: float = 0.06
    small_event_area_ratio: float = 0.012


def _bbox_dict(b: list[int]) -> dict[str, int]:
    return {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]}


def _obs_near_bbox(
    obs: dict[str, Any],
    probe: dict[str, int],
    frame_w: int,
    frame_h: int,
    dist: float,
) -> bool:
    tb = {"x1": obs["x1"], "y1": obs["y1"], "x2": obs["x2"], "y2": obs["y2"]}
    if bbox_iou(probe, tb) >= 0.02:
        return True
    return center_distance_norm(probe, tb, frame_w, frame_h) <= dist


def _norm_center(bbox: list[int], frame_w: int, frame_h: int) -> tuple[float, float]:
    cx = (bbox[0] + bbox[2]) * 0.5 / max(frame_w, 1)
    cy = (bbox[1] + bbox[3]) * 0.5 / max(frame_h, 1)
    return cx, cy


def _member_frame_centers(
    members: list[dict[str, Any]],
    frame_w: int,
    frame_h: int,
) -> list[tuple[int, tuple[float, float]]]:
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for m in members:
        by_frame.setdefault(int(m["frame_idx"]), []).append(m)
    out: list[tuple[int, tuple[float, float]]] = []
    for fi in sorted(by_frame.keys()):
        boxes = by_frame[fi]
        x1 = min(b["x1"] for b in boxes)
        y1 = min(b["y1"] for b in boxes)
        x2 = max(b["x2"] for b in boxes)
        y2 = max(b["y2"] for b in boxes)
        out.append((fi, _norm_center([x1, y1, x2, y2], frame_w, frame_h)))
    return out


def _compute_kinematics(
    bbox_sequence: list[dict[str, Any]],
    *,
    frame_w: int,
    frame_h: int,
    sampled_fps: float,
    cfg: EventRelevanceGateConfig,
    members: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    member_traj = _member_frame_centers(members, frame_w, frame_h) if members else []
    if len(member_traj) >= 2:
        centers = [c for _, c in member_traj]
    elif len(bbox_sequence) >= 2:
        centers = [_norm_center(e["bbox"], frame_w, frame_h) for e in bbox_sequence]
    else:
        return {
            "net_displacement_ratio": 0.0,
            "path_linearity": 0.0,
            "direction_consistency": 0.0,
            "speed_mean": 0.0,
            "speed_cv": 0.0,
            "stop_ratio": 0.0,
            "dwell_ratio": 0.0,
        }

    speeds: list[float] = []
    dirs: list[tuple[float, float]] = []

    for i in range(1, len(centers)):
        dx = centers[i][0] - centers[i - 1][0]
        dy = centers[i][1] - centers[i - 1][1]
        seg = float(np.hypot(dx, dy))
        speeds.append(seg * sampled_fps)
        if seg > EPS:
            dirs.append((dx / seg, dy / seg))

    path_len = float(sum(
        np.hypot(centers[i][0] - centers[i - 1][0], centers[i][1] - centers[i - 1][1])
        for i in range(1, len(centers))
    ))
    net_disp = float(np.hypot(centers[-1][0] - centers[0][0], centers[-1][1] - centers[0][1]))
    net_displacement_ratio = net_disp / max(np.sqrt(2.0), EPS)
    path_linearity = net_disp / max(path_len, EPS)

    direction_consistency = 0.0
    if len(dirs) >= 2:
        dots = [dirs[i][0] * dirs[i - 1][0] + dirs[i][1] * dirs[i - 1][1] for i in range(1, len(dirs))]
        direction_consistency = float(np.clip(np.mean(dots), 0.0, 1.0))
    elif len(dirs) == 1:
        direction_consistency = 1.0

    speed_mean = float(np.mean(speeds)) if speeds else 0.0
    speed_cv = float(np.std(speeds) / max(speed_mean, EPS)) if speeds else 0.0
    stop_ratio = float(np.mean([s < cfg.stop_speed_threshold for s in speeds])) if speeds else 0.0

    dwell_count = 0
    for i in range(1, len(centers)):
        if np.hypot(centers[i][0] - centers[i - 1][0], centers[i][1] - centers[i - 1][1]) < cfg.dwell_move_threshold:
            dwell_count += 1
    dwell_ratio = dwell_count / max(len(centers) - 1, 1)

    return {
        "net_displacement_ratio": round(net_displacement_ratio, 4),
        "path_linearity": round(path_linearity, 4),
        "direction_consistency": round(direction_consistency, 4),
        "speed_mean": round(speed_mean, 4),
        "speed_cv": round(speed_cv, 4),
        "stop_ratio": round(stop_ratio, 4),
        "dwell_ratio": round(dwell_ratio, 4),
    }


def _person_near_vehicle_door_trunk(
    person_obs: dict[str, Any],
    vehicle_obs: dict[str, Any],
) -> bool:
    """Heuristic: person near vehicle door/trunk zone (lateral or rear band)."""
    vx1, vy1, vx2, vy2 = (
        float(vehicle_obs["x1"]), float(vehicle_obs["y1"]),
        float(vehicle_obs["x2"]), float(vehicle_obs["y2"]),
    )
    px = float(person_obs.get("center_x", (person_obs["x1"] + person_obs["x2"]) / 2))
    py = float(person_obs.get("center_y", (person_obs["y1"] + person_obs["y2"]) / 2))
    vw, vh = max(vx2 - vx1, 1.0), max(vy2 - vy1, 1.0)
    rel_x = (px - vx1) / vw
    rel_y = (py - vy1) / vh
    lateral = rel_x <= DOOR_TRUNK_MARGIN or rel_x >= (1.0 - DOOR_TRUNK_MARGIN)
    rear = rel_y >= (1.0 - DOOR_TRUNK_MARGIN)
    return lateral or rear


def _person_near_entrance(
    person_obs: dict[str, Any],
    *,
    frame_h: int,
) -> bool:
    """Proxy for building entrance: person in lower frame band with localized bbox."""
    py = float(person_obs.get("center_y", (person_obs["y1"] + person_obs["y2"]) / 2))
    return py / max(frame_h, 1) >= (1.0 - ENTRANCE_BOTTOM_RATIO)


def _compute_semantic_context(
    bbox_sequence: list[dict[str, Any]],
    aux_track_obs: dict[int, list[dict]],
    *,
    frame_w: int,
    frame_h: int,
    mean_area: float,
    sampled_fps: float,
    members: list[dict[str, Any]] | None = None,
    cfg: EventRelevanceGateConfig,
) -> dict[str, float]:
    n_seq = max(len(bbox_sequence), 1)
    person_frames = 0
    vehicle_frames = 0
    interaction_frames = 0
    dwell_near_object = 0
    person_vehicle_dwell = 0
    door_trunk_frames = 0
    entrance_frames = 0
    object_proximity_frames = 0
    slow_near_object_frames = 0

    member_traj = _member_frame_centers(members, frame_w, frame_h) if members else []
    member_center_by_frame = {fi: c for fi, c in member_traj}

    centers_prev: tuple[float, float] | None = None
    for entry in bbox_sequence:
        fi = int(entry["frame_idx"])
        probe = _bbox_dict(entry["bbox"])
        cx, cy = member_center_by_frame.get(fi, _norm_center(entry["bbox"], frame_w, frame_h))
        is_dwell = False
        seg_speed = 0.0
        if centers_prev is not None:
            seg_speed = float(np.hypot(cx - centers_prev[0], cy - centers_prev[1])) * sampled_fps
            is_dwell = seg_speed < cfg.stop_speed_threshold
        centers_prev = (cx, cy)

        persons: list[dict] = []
        vehicles: list[dict] = []
        for obs_list in aux_track_obs.values():
            for obs in obs_list:
                if int(obs["frame_idx"]) != fi:
                    continue
                cls = str(obs.get("class_name", ""))
                if not _obs_near_bbox(obs, probe, frame_w, frame_h, PROXIMITY_DIST):
                    continue
                if cls in PERSON_CLASSES:
                    persons.append(obs)
                if cls in VEHICLE_CLASSES:
                    vehicles.append(obs)

        has_person = bool(persons)
        has_vehicle = bool(vehicles)
        if has_person:
            person_frames += 1
        if has_vehicle:
            vehicle_frames += 1
        if has_person and has_vehicle:
            interaction_frames += 1
        if has_person or has_vehicle:
            object_proximity_frames += 1
            if is_dwell or seg_speed < cfg.stop_speed_threshold * 1.5:
                dwell_near_object += 1
                slow_near_object_frames += 1
            if has_person and has_vehicle and is_dwell:
                person_vehicle_dwell += 1

        for p in persons:
            for v in vehicles:
                if _person_near_vehicle_door_trunk(p, v):
                    door_trunk_frames += 1
                    break
            if _person_near_entrance(p, frame_h=frame_h) and mean_area <= cfg.small_event_area_ratio * 4:
                entrance_frames += 1

    interaction_ratio = interaction_frames / n_seq
    object_proximity_ratio = object_proximity_frames / n_seq
    dwell_near_object_ratio = dwell_near_object / n_seq
    person_vehicle_dwell_ratio = person_vehicle_dwell / n_seq
    door_trunk_ratio = door_trunk_frames / n_seq
    entrance_ratio = entrance_frames / n_seq

    small_localized = mean_area <= cfg.small_event_area_ratio
    semantic_parts = [
        0.45 * min(1.0, person_vehicle_dwell_ratio * 3.0),
        0.35 * min(1.0, door_trunk_ratio * 4.0),
        0.30 * min(1.0, entrance_ratio * 4.0),
        0.25 * min(1.0, interaction_ratio * 2.5) if small_localized else 0.0,
        0.20 * min(1.0, object_proximity_ratio) if small_localized else 0.0,
    ]
    semantic_anchor_score = round(min(1.0, max(semantic_parts)), 4)

    return {
        "interaction_ratio": round(interaction_ratio, 4),
        "object_proximity_ratio": round(object_proximity_ratio, 4),
        "dwell_near_object_ratio": round(dwell_near_object_ratio, 4),
        "person_vehicle_dwell_ratio": round(person_vehicle_dwell_ratio, 4),
        "door_trunk_ratio": round(door_trunk_ratio, 4),
        "entrance_ratio": round(entrance_ratio, 4),
        "semantic_anchor_score": semantic_anchor_score,
        "person_near_frames": person_frames,
        "vehicle_near_frames": vehicle_frames,
        "interaction_frames": interaction_frames,
    }


def _is_transit_motion(features: dict[str, float], cfg: EventRelevanceGateConfig) -> bool:
    return (
        features["net_displacement_ratio"] >= cfg.transit_net_displacement_min
        and features["path_linearity"] >= cfg.transit_path_linearity_min
        and features["direction_consistency"] >= cfg.transit_direction_consistency_min
        and features["stop_ratio"] <= cfg.transit_stop_ratio_max
        and features["dwell_near_object_ratio"] <= cfg.transit_dwell_near_object_max
        and features["interaction_ratio"] <= cfg.transit_interaction_ratio_max
        and features["semantic_anchor_score"] <= cfg.transit_semantic_anchor_max
    )


def _transit_penalty(features: dict[str, float], cfg: EventRelevanceGateConfig) -> float:
    signals = [
        features["net_displacement_ratio"] >= cfg.transit_net_displacement_min * 0.85,
        features["path_linearity"] >= cfg.transit_path_linearity_min * 0.9,
        features["direction_consistency"] >= cfg.transit_direction_consistency_min * 0.9,
        features["stop_ratio"] <= cfg.transit_stop_ratio_max * 1.5,
        features["dwell_near_object_ratio"] <= cfg.transit_dwell_near_object_max * 1.3,
        features["interaction_ratio"] <= cfg.transit_interaction_ratio_max * 1.5,
        features["semantic_anchor_score"] <= cfg.transit_semantic_anchor_max * 1.5,
    ]
    return round(min(1.0, 0.15 * sum(signals)), 4)


def _event_motion_signals(
    features: dict[str, float],
    semantic: dict[str, float],
    *,
    mean_area: float,
    persistence_sec: float,
    motion_raw: float,
    n_seq: int,
    interaction_score: float = 0.0,
    cfg: EventRelevanceGateConfig,
) -> list[str]:
    reasons: list[str] = []
    small_localized = mean_area <= cfg.small_event_area_ratio

    if semantic["person_vehicle_dwell_ratio"] >= 0.12:
        reasons.append("person_vehicle_dwell")
    elif semantic["interaction_ratio"] >= 0.12 and (
        features["dwell_ratio"] >= 0.18 or features["stop_ratio"] >= 0.15
    ):
        reasons.append("person_vehicle_proximity_dwell")

    person_ratio = semantic["person_near_frames"] / n_seq
    vehicle_ratio = semantic["vehicle_near_frames"] / n_seq
    if (
        person_ratio >= 0.12
        and vehicle_ratio >= 0.12
        and (features["dwell_ratio"] >= 0.12 or features["stop_ratio"] >= 0.12)
    ):
        reasons.append("person_vehicle_temporal_dwell")

    if person_ratio >= 0.18 and small_localized and persistence_sec >= 0.35:
        reasons.append("localized_person_activity")

    if (
        small_localized
        and vehicle_ratio >= 0.20
        and (features["stop_ratio"] >= 0.15 or features["dwell_ratio"] >= 0.15)
        and persistence_sec >= 0.4
        and motion_raw >= 0.10
        and features["net_displacement_ratio"] < 0.35
    ):
        reasons.append("localized_vehicle_dwell_motion")

    if interaction_score >= 0.30 and semantic["object_proximity_ratio"] >= 0.20:
        if (
            features["net_displacement_ratio"] < 0.35
            and semantic["interaction_ratio"] < 0.12
            and semantic["person_vehicle_dwell_ratio"] < 0.10
        ):
            reasons.append("strong_interaction_score")

    if semantic["door_trunk_ratio"] >= 0.10 and small_localized:
        reasons.append("person_vehicle_door_trunk")

    if semantic["entrance_ratio"] >= 0.10 and small_localized:
        reasons.append("person_entrance_localized")

    if (
        small_localized
        and semantic["object_proximity_ratio"] >= 0.22
        and persistence_sec >= 0.35
        and motion_raw >= 0.15
        and (
            person_ratio >= 0.10
            or semantic["door_trunk_ratio"] >= 0.08
            or semantic["entrance_ratio"] >= 0.08
            or semantic["interaction_ratio"] >= 0.08
        )
    ):
        reasons.append("small_localized_near_object")

    stop_go = (
        features["stop_ratio"] >= 0.18
        and features["speed_cv"] >= cfg.stop_go_speed_cv_min
        and semantic["object_proximity_ratio"] >= 0.20
        and (
            semantic["person_vehicle_dwell_ratio"] >= 0.08
            or semantic["interaction_ratio"] >= 0.10
            or semantic["door_trunk_ratio"] >= 0.08
        )
    )
    if stop_go:
        reasons.append("stop_and_go_near_object")

    if (
        small_localized
        and semantic["semantic_anchor_score"] >= 0.22
        and semantic["object_proximity_ratio"] >= 0.18
    ):
        reasons.append("localized_object_context")

    if (
        small_localized
        and persistence_sec >= 0.8
        and motion_raw >= 0.18
        and features["net_displacement_ratio"] <= 0.30
        and (features["stop_ratio"] >= 0.20 or features["dwell_ratio"] >= 0.15)
    ):
        reasons.append("localized_small_blob_dwell")

    if (
        small_localized
        and persistence_sec >= 0.35
        and motion_raw >= 0.25
        and semantic["object_proximity_ratio"] >= 0.08
        and features["net_displacement_ratio"] <= 0.35
    ):
        reasons.append("localized_near_object_motion")

    return reasons


def _classify_motion_pattern(
    features: dict[str, float],
    semantic: dict[str, float],
    *,
    mean_area: float,
    mean_coh: float,
    persistence_sec: float,
    motion_raw: float,
    jitter_ratio: float,
    is_background_flow_region: bool,
    background_flow_penalty: float,
    n_seq: int,
    interaction_score: float = 0.0,
    cfg: EventRelevanceGateConfig,
) -> tuple[str, str]:
    if (
        jitter_ratio >= 0.55
        or (mean_area >= cfg.jitter_group_area and mean_coh >= 0.68)
        or (jitter_ratio >= 0.35 and mean_coh >= 0.72 and mean_area >= 0.25)
    ):
        return "background_jitter", "high_jitter_or_shake"

    if is_background_flow_region or (
        mean_coh >= cfg.background_flow_coherence
        and mean_area >= cfg.background_flow_area_ratio
        and semantic["interaction_ratio"] < 0.12
        and background_flow_penalty >= 0.35
    ):
        return "background_flow", "coherent_background_flow"

    if _is_transit_motion({**features, **semantic}, cfg):
        return "transit_motion", "linear_transit_kinematics"

    if (
        features["net_displacement_ratio"] >= 0.40
        and features["path_linearity"] >= 0.65
        and semantic["interaction_ratio"] <= 0.05
        and semantic["person_vehicle_dwell_ratio"] <= 0.05
        and semantic["semantic_anchor_score"] <= 0.25
    ):
        return "transit_motion", "linear_pass_through"

    event_signals = _event_motion_signals(
        features, semantic,
        mean_area=mean_area,
        persistence_sec=persistence_sec,
        motion_raw=motion_raw,
        n_seq=max(n_seq, 1),
        interaction_score=interaction_score,
        cfg=cfg,
    )
    if event_signals:
        return "event_motion", ";".join(event_signals)

    small_localized = mean_area <= cfg.small_event_area_ratio

    if (
        small_localized
        and persistence_sec >= 0.5
        and motion_raw >= 0.15
        and (
            semantic["semantic_anchor_score"] >= 0.08
            or semantic["object_proximity_ratio"] >= 0.06
            or features["stop_ratio"] >= 0.20
        )
    ):
        return "ambiguous_motion", "partial_event_signals"

    if semantic["semantic_anchor_score"] >= 0.12 or semantic["interaction_ratio"] >= 0.06:
        return "ambiguous_motion", "partial_event_signals"

    return "ambiguous_motion", "unclear_event_relevance"


def _compute_relevance_components(
    features: dict[str, float],
    semantic: dict[str, float],
    *,
    mean_area: float,
    persistence_sec: float,
    motion_raw: float,
    background_flow_penalty: float,
    jitter_penalty: float,
    cfg: EventRelevanceGateConfig,
) -> dict[str, float]:
    small_localized = mean_area <= cfg.small_event_area_ratio

    interaction_dwell_score = round(min(1.0, (
        0.45 * semantic["person_vehicle_dwell_ratio"] * 3.5
        + 0.30 * semantic["dwell_near_object_ratio"] * 2.5
        + 0.25 * semantic["interaction_ratio"] * 2.8
    )), 4)

    stop_or_slowdown_score = 0.0
    if semantic["object_proximity_ratio"] >= 0.15:
        stop_or_slowdown_score = round(min(1.0, (
            0.55 * features["stop_ratio"] * 2.0
            + 0.25 * features["dwell_ratio"]
            + 0.20 * (1.0 - min(1.0, features["speed_mean"] * 8.0))
        )), 4)

    localized_persistence_score = round(min(1.0, (
        (0.6 if small_localized else 0.25)
        * min(1.0, persistence_sec / 1.8)
        * min(1.0, motion_raw * 2.0)
    )), 4)

    small_critical_motion_score = 0.0
    if small_localized and semantic["object_proximity_ratio"] >= 0.12:
        small_critical_motion_score = round(min(1.0, (
            0.5 * min(1.0, motion_raw * 2.2)
            + 0.3 * min(1.0, persistence_sec / 1.2)
            + 0.2 * semantic["semantic_anchor_score"]
        )), 4)
    elif small_localized and persistence_sec >= 0.8 and motion_raw >= 0.18:
        small_critical_motion_score = round(min(1.0, (
            0.55 * min(1.0, motion_raw * 2.0)
            + 0.45 * min(1.0, persistence_sec / 2.0)
        )), 4)

    transit_penalty = _transit_penalty({**features, **semantic}, cfg)
    if _is_transit_motion({**features, **semantic}, cfg):
        transit_penalty = round(min(1.0, transit_penalty + 0.35), 4)
    if features["stop_ratio"] >= 0.35 or features["dwell_ratio"] >= 0.35:
        transit_penalty = round(min(transit_penalty, 0.12), 4)

    event_relevance_score = round(max(0.0, (
        interaction_dwell_score
        + semantic["semantic_anchor_score"]
        + stop_or_slowdown_score
        + localized_persistence_score
        + small_critical_motion_score
        - transit_penalty
        - background_flow_penalty
        - jitter_penalty
    )), 6)

    return {
        "interaction_dwell_score": interaction_dwell_score,
        "stop_or_slowdown_score": stop_or_slowdown_score,
        "localized_persistence_score": localized_persistence_score,
        "small_critical_motion_score": small_critical_motion_score,
        "transit_penalty": transit_penalty,
        "event_relevance_score": event_relevance_score,
    }


def apply_event_relevance_gate(
    *,
    members: list[dict[str, Any]],
    bbox_sequence: list[dict[str, Any]],
    aux_track_obs: dict[int, list[dict]],
    frame_df: Any,
    frame_w: int,
    frame_h: int,
    sampled_fps: float,
    base_scores: dict[str, Any],
    gate_cfg: EventRelevanceGateConfig | None = None,
) -> dict[str, Any]:
    """
    Classify motion_pattern and compute event_relevance_score for one blob group.

    Returns all candidate-level features required for debug and selection.
    """
    cfg = gate_cfg or EventRelevanceGateConfig()
    active_frames = {int(m["frame_idx"]) for m in members}
    mean_area = float(base_scores.get("mean_bbox_area_ratio", np.mean([m["bbox_area_ratio"] for m in members])))
    mean_coh = float(base_scores.get("mean_flow_coherence", 0.0))
    motion_raw = float(np.mean([m["blob_importance"] for m in members]))
    blob_persistence_sec = len(active_frames) / max(sampled_fps, EPS)
    if bbox_sequence:
        timeline_persistence = float(
            bbox_sequence[-1]["timestamp_sec"] - bbox_sequence[0]["timestamp_sec"]
        )
        persistence_sec = max(blob_persistence_sec, timeline_persistence)
    else:
        persistence_sec = blob_persistence_sec

    kinematics = _compute_kinematics(
        bbox_sequence,
        frame_w=frame_w,
        frame_h=frame_h,
        sampled_fps=sampled_fps,
        cfg=cfg,
        members=members,
    )
    semantic = _compute_semantic_context(
        bbox_sequence, aux_track_obs,
        frame_w=frame_w, frame_h=frame_h, mean_area=mean_area,
        sampled_fps=sampled_fps, members=members, cfg=cfg,
    )

    jitter_ratio = 0.0
    if frame_df is not None and not frame_df.empty and "background_jitter_flag" in frame_df.columns:
        sub = frame_df[frame_df["frame_idx"].isin(active_frames)]
        if not sub.empty:
            jitter_ratio = float(sub["background_jitter_flag"].astype(bool).mean())

    background_flow_penalty = float(base_scores.get("background_flow_penalty", 0.0))
    is_background_flow_region = bool(base_scores.get("is_background_flow_region", False))
    jitter_penalty = cfg.jitter_penalty_value if (
        mean_area >= cfg.jitter_group_area and mean_coh >= 0.70
    ) else round(0.25 * jitter_ratio, 4)

    motion_pattern, pattern_reason = _classify_motion_pattern(
        kinematics, semantic,
        mean_area=mean_area,
        mean_coh=mean_coh,
        persistence_sec=persistence_sec,
        motion_raw=motion_raw,
        jitter_ratio=jitter_ratio,
        is_background_flow_region=is_background_flow_region,
        background_flow_penalty=background_flow_penalty,
        n_seq=len(bbox_sequence),
        interaction_score=float(base_scores.get("interaction_score", 0.0)),
        cfg=cfg,
    )

    relevance = _compute_relevance_components(
        kinematics, semantic,
        mean_area=mean_area,
        persistence_sec=persistence_sec,
        motion_raw=motion_raw,
        background_flow_penalty=background_flow_penalty,
        jitter_penalty=jitter_penalty,
        cfg=cfg,
    )

    rejection_reason = ""
    if motion_pattern == "background_jitter":
        rejection_reason = "background_jitter"
    elif motion_pattern == "background_flow":
        rejection_reason = "background_flow"
    elif motion_pattern == "transit_motion":
        rejection_reason = "transit_motion"
    elif motion_pattern == "ambiguous_motion":
        if cfg.allow_ambiguous_roi:
            if (
                mean_area <= cfg.small_event_area_ratio
                and persistence_sec >= 0.5
                and motion_raw >= 0.12
            ):
                relevance["event_relevance_score"] = max(
                    relevance["event_relevance_score"],
                    cfg.min_event_relevance_score + 0.02,
                )
            elif (
                mean_area <= cfg.small_event_area_ratio * 4
                and persistence_sec >= 0.35
                and motion_raw >= 0.10
            ):
                relevance["event_relevance_score"] = max(
                    relevance["event_relevance_score"],
                    cfg.min_event_relevance_score,
                )
            rejection_reason = ""
        else:
            rejection_reason = "ambiguous_motion"
    elif motion_pattern == "event_motion" and relevance["event_relevance_score"] < cfg.min_event_relevance_score:
        rejection_reason = "low_event_relevance_score"

    return {
        **kinematics,
        **semantic,
        **relevance,
        "jitter_penalty": round(jitter_penalty, 4),
        "background_flow_penalty": background_flow_penalty,
        "motion_pattern": motion_pattern,
        "pattern_reason": pattern_reason,
        "rejection_reason": rejection_reason,
        "jitter_ratio": round(jitter_ratio, 4),
    }


def summarize_gate_for_plan(
    candidates: list[dict[str, Any]],
    selected_tracks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate gate stats for adaptive_input_plan_debug.json."""
    by_pattern: dict[str, int] = {p: 0 for p in MOTION_PATTERNS}
    for c in candidates:
        pat = str(c.get("motion_pattern", "ambiguous_motion"))
        by_pattern[pat] = by_pattern.get(pat, 0) + 1

    rejected_transit = sum(1 for c in candidates if c.get("motion_pattern") == "transit_motion")
    rejected_background = sum(
        1 for c in candidates
        if c.get("motion_pattern") in ("background_jitter", "background_flow")
    )

    no_roi_reason = ""
    if not selected_tracks:
        event_count = by_pattern.get("event_motion", 0)
        if event_count == 0:
            no_roi_reason = "no_event_motion_candidates"
        else:
            no_roi_reason = "event_motion_candidates_filtered_by_nms_or_relevance"

    return {
        "num_candidates_by_pattern": by_pattern,
        "selected_roi_count": len(selected_tracks),
        "rejected_transit_count": rejected_transit,
        "rejected_background_count": rejected_background,
        "no_roi_reason": no_roi_reason,
    }
