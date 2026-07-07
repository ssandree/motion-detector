"""Event-group ROI from motion blobs (primary). ByteTrack is auxiliary for box expansion only."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from motion_analyzer.v1.input_plan_builder import MAX_ROI_TRACKS
from motion_analyzer.v2.bbox_utils import bbox_iou, center_distance_norm, expand_bbox, union_bbox
from motion_analyzer.v2.event_relevance_gate import (
    EventRelevanceGateConfig,
    apply_event_relevance_gate,
    summarize_gate_for_plan,
)
from motion_analyzer.v2.event_roi_selection import (
    RoiSelectionConfig,
    compute_blob_persistence_stats,
    select_coverage_aware_rois,
)
from motion_analyzer.v2.flow_group_merge import group_frame_blobs_by_flow

EPS = 1e-8

PERSON_CLASSES = frozenset({"person"})
VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle", "bicycle"})
INTERACTION_DIST = 0.12
PROXIMITY_DIST = 0.14


@dataclass
class EventRoiConfig:
    temporal_window_sec: float = 0.8
    spatial_merge_dist: float = 0.10
    flow_coherence_delta: float = 0.35
    context_margin: float = 0.28
    small_event_area_ratio: float = 0.012
    small_event_margin: float = 0.38
    object_absorb_dist: float = 0.09
    object_absorb_iou: float = 0.02
    jitter_frame_large_area: float = 0.30
    jitter_group_area: float = 0.40
    max_roi_tracks: int = MAX_ROI_TRACKS
    smooth_alpha: float = 0.35
    scene_change_penalty_weight: float = 0.15
    min_roi_duration_sec: float = 2.0
    min_roi_frames: int = 4
    context_frames: int = 4
    nms_spatial_iou: float = 0.35
    nms_temporal_iou: float = 0.40
    min_event_score: float = 0.08
    min_event_relevance_score: float = 0.10
    allow_ambiguous_roi: bool = True
    aux_timeline_extend: bool = True
    aux_timeline_extend_dist: float = 0.14
    aux_trailing_extend_sec: float = 28.0
    max_active_gap_frames: int = 2
    tube_smooth_alpha: float = 0.22
    background_flow_area_ratio: float = 0.06
    background_flow_coherence: float = 0.72
    background_flow_min_duration_sec: float = 3.0

    def gate_config(self) -> EventRelevanceGateConfig:
        return EventRelevanceGateConfig(
            allow_ambiguous_roi=self.allow_ambiguous_roi,
            min_event_relevance_score=self.min_event_relevance_score,
            jitter_group_area=self.jitter_group_area,
            background_flow_coherence=self.background_flow_coherence,
            background_flow_area_ratio=self.background_flow_area_ratio,
            small_event_area_ratio=self.small_event_area_ratio,
        )

    def selection_config(self) -> RoiSelectionConfig:
        return RoiSelectionConfig()


@dataclass
class EventRoiBuildResult:
    candidates: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    selected_tracks: list[dict[str, Any]] = field(default_factory=list)
    selection_rejected: list[dict[str, Any]] = field(default_factory=list)
    debug_rejected_high_rank: list[dict[str, Any]] = field(default_factory=list)
    background_flow_regions: list[dict[str, Any]] = field(default_factory=list)
    debug_report: dict[str, Any] = field(default_factory=dict)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _blob_dict(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "frame_idx": int(row["frame_idx"]),
        "timestamp_sec": float(row["timestamp_sec"]),
        "x1": int(row["x1"]),
        "y1": int(row["y1"]),
        "x2": int(row["x2"]),
        "y2": int(row["y2"]),
        "blob_importance": float(row.get("blob_importance", 0.0)),
        "flow_direction_coherence": float(row.get("flow_direction_coherence", 0.0)),
        "bbox_area_ratio": float(row.get("bbox_area_ratio", 0.0)),
        "blob_id_in_frame": int(row.get("blob_id_in_frame", -1)),
    }


def _sampled_timeline(frame_df: pd.DataFrame) -> list[tuple[int, float]]:
    return [
        (int(r["frame_idx"]), float(r["timestamp_sec"]))
        for _, r in frame_df.sort_values("frame_idx").iterrows()
    ]


def filter_jitter_blobs(
    blobs_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    *,
    config: EventRoiConfig | None = None,
) -> pd.DataFrame:
    if blobs_df.empty:
        return blobs_df
    cfg = config or EventRoiConfig()
    jitter_cols = {
        int(r["frame_idx"]): bool(r.get("background_jitter_flag", False))
        for _, r in frame_df.iterrows()
    }
    keep_rows: list[dict[str, Any]] = []
    for row in blobs_df.to_dict("records"):
        b = _blob_dict(row)
        if b["bbox_area_ratio"] >= 0.55:
            continue
        if jitter_cols.get(b["frame_idx"], False):
            if b["bbox_area_ratio"] >= cfg.jitter_frame_large_area and b["flow_direction_coherence"] >= 0.65:
                continue
            if b["bbox_area_ratio"] >= 0.55:
                continue
        keep_rows.append(row)
    return pd.DataFrame(keep_rows) if keep_rows else blobs_df.iloc[0:0]


def _flow_compatible(a: dict[str, Any], b: dict[str, Any], cfg: EventRoiConfig) -> bool:
    ca = float(a.get("flow_direction_coherence", 0.0))
    cb = float(b.get("flow_direction_coherence", 0.0))
    if ca < 0.45 and cb < 0.45:
        return True
    return abs(ca - cb) <= cfg.flow_coherence_delta


def _link_spatiotemporal_groups(
    blobs: list[dict[str, Any]],
    *,
    frame_w: int,
    frame_h: int,
    sampled_fps: float,
    config: EventRoiConfig,
    frame_to_pos: dict[int, int] | None = None,
) -> list[list[dict[str, Any]]]:
    if not blobs:
        return []
    cfg = config
    n = len(blobs)
    uf = _UnionFind(n)
    max_sample_gap = max(1, int(round(cfg.temporal_window_sec * sampled_fps)))

    if frame_to_pos is None:
        frame_to_pos = {int(b["frame_idx"]): i for i, b in enumerate(
            sorted({int(b["frame_idx"]) for b in blobs})
        )}

    by_frame: dict[int, list[int]] = defaultdict(list)
    for i, b in enumerate(blobs):
        by_frame[int(b["frame_idx"])].append(i)

    for fi in sorted(by_frame.keys()):
        idxs = by_frame[fi]
        frame_blobs = [blobs[i] for i in idxs]
        _, groups = group_frame_blobs_by_flow(frame_blobs, frame_w=frame_w, frame_h=frame_h)
        id_to_local = {int(b.get("blob_id_in_frame", j)): j for j, b in enumerate(frame_blobs)}
        for grp in groups:
            member_local = []
            for bid in grp.get("member_blob_ids", []):
                if bid in id_to_local:
                    member_local.append(idxs[id_to_local[bid]])
            for a in range(1, len(member_local)):
                uf.union(member_local[0], member_local[a])

    for a in range(n):
        for b in range(a + 1, n):
            ba, bb = blobs[a], blobs[b]
            pa = frame_to_pos.get(int(ba["frame_idx"]))
            pb = frame_to_pos.get(int(bb["frame_idx"]))
            if pa is None or pb is None:
                continue
            if abs(pa - pb) > max_sample_gap:
                continue
            if center_distance_norm(ba, bb, frame_w, frame_h) > cfg.spatial_merge_dist:
                continue
            if not _flow_compatible(ba, bb, cfg):
                continue
            uf.union(a, b)

    comp: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for i, blob in enumerate(blobs):
        comp[uf.find(i)].append(blob)
    return list(comp.values())


def _group_union_bbox(members: list[dict[str, Any]]) -> list[int]:
    boxes = [{"x1": m["x1"], "y1": m["y1"], "x2": m["x2"], "y2": m["y2"]} for m in members]
    u = union_bbox(boxes)
    return [int(u["x1"]), int(u["y1"]), int(u["x2"]), int(u["y2"])]


def _frame_union_bbox(members: list[dict[str, Any]]) -> list[int]:
    return _group_union_bbox(members)


def _absorb_auxiliary_detections(
    bbox: list[int],
    frame_idx: int,
    aux_track_obs: dict[int, list[dict]],
    *,
    frame_w: int,
    frame_h: int,
    config: EventRoiConfig,
) -> tuple[list[int], list[int]]:
    """ByteTrack boxes: expansion only. Returns (expanded_bbox, auxiliary_track_ids)."""
    cfg = config
    boxes = [{"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]}]
    probe = boxes[0]
    aux_ids: set[int] = set()
    for tid, obs_list in aux_track_obs.items():
        for obs in obs_list:
            if int(obs["frame_idx"]) != frame_idx:
                continue
            tb = {"x1": obs["x1"], "y1": obs["y1"], "x2": obs["x2"], "y2": obs["y2"]}
            if bbox_iou(probe, tb) >= cfg.object_absorb_iou:
                boxes.append(tb)
                aux_ids.add(int(tid))
            elif center_distance_norm(probe, tb, frame_w, frame_h) <= cfg.object_absorb_dist:
                boxes.append(tb)
                aux_ids.add(int(tid))
    u = union_bbox(boxes)
    return [int(u["x1"]), int(u["y1"]), int(u["x2"]), int(u["y2"])], sorted(aux_ids)


def _expand_bbox_context(
    bbox: list[int],
    *,
    area_ratio: float,
    frame_w: int,
    frame_h: int,
    config: EventRoiConfig,
) -> list[int]:
    margin = config.small_event_margin if area_ratio <= config.small_event_area_ratio else config.context_margin
    return expand_bbox(bbox[0], bbox[1], bbox[2], bbox[3], margin, frame_w, frame_h)


def _collect_aux_motion_frames(
    group_union: list[int],
    aux_track_obs: dict[int, list[dict]],
    *,
    frame_w: int,
    frame_h: int,
    mean_area: float,
    config: EventRoiConfig,
    vehicle_only: bool = False,
) -> set[int]:
    """Frames where auxiliary tracks overlap the group's spatial envelope."""
    if not aux_track_obs:
        return set()
    probe = expand_bbox_context(
        group_union, area_ratio=mean_area, frame_w=frame_w, frame_h=frame_h, config=config,
    )
    probe_d = _bbox_dict(probe)
    dist = max(config.object_absorb_dist, config.aux_timeline_extend_dist)
    frames: set[int] = set()
    for obs_list in aux_track_obs.values():
        for obs in obs_list:
            cls = str(obs.get("class_name", ""))
            if vehicle_only and cls not in VEHICLE_CLASSES:
                continue
            if not vehicle_only and cls not in PERSON_CLASSES and cls not in VEHICLE_CLASSES:
                continue
            if _obs_near_bbox(obs, probe_d, frame_w, frame_h, dist):
                frames.add(int(obs["frame_idx"]))
    return frames


def _extend_active_frames_with_aux_tracks(
    active_frames: set[int],
    members: list[dict[str, Any]],
    aux_track_obs: dict[int, list[dict]],
    *,
    frame_w: int,
    frame_h: int,
    mean_area: float,
    sampled_fps: float,
    config: EventRoiConfig,
) -> set[int]:
    """Extend blob-active frames with auxiliary tracks trailing the blob span."""
    if not config.aux_timeline_extend or not aux_track_obs or not active_frames:
        return active_frames
    group_union = _group_union_bbox(members)
    traffic_lane = mean_area > config.small_event_area_ratio * 4
    vehicle_frames = _collect_aux_motion_frames(
        group_union, aux_track_obs,
        frame_w=frame_w, frame_h=frame_h, mean_area=mean_area,
        config=config, vehicle_only=True,
    )
    if traffic_lane:
        aux_frames = vehicle_frames
    else:
        aux_frames = _collect_aux_motion_frames(
            group_union, aux_track_obs,
            frame_w=frame_w, frame_h=frame_h, mean_area=mean_area,
            config=config, vehicle_only=False,
        )

    blob_min = min(active_frames)
    blob_max = max(active_frames)
    leading_margin = config.context_frames * 2
    trailing_margin = max(1, int(config.aux_trailing_extend_sec * sampled_fps))
    allowed: set[int] = set()
    for fi in aux_frames:
        if fi < blob_min - leading_margin:
            continue
        if fi > blob_max + trailing_margin:
            continue
        allowed.add(fi)
    return active_frames | allowed


def _timeline_indices_for_group(
    active_frames: set[int],
    timeline: list[tuple[int, float]],
    cfg: EventRoiConfig,
    sampled_fps: float,
) -> list[int]:
    """Expand blob-active frames with context; enforce min_roi_frames / min duration."""
    if not active_frames or not timeline:
        return []

    frame_to_pos = {fi: i for i, (fi, _) in enumerate(timeline)}
    positions = sorted(frame_to_pos[fi] for fi in active_frames if fi in frame_to_pos)
    if not positions:
        return []

    positions = _bridge_active_positions(positions, cfg.max_active_gap_frames)

    start_pos = max(0, positions[0] - cfg.context_frames)
    end_pos = min(len(timeline) - 1, positions[-1] + cfg.context_frames)

    while (end_pos - start_pos + 1) < cfg.min_roi_frames:
        grew = False
        if start_pos > 0:
            start_pos -= 1
            grew = True
        if end_pos < len(timeline) - 1:
            end_pos += 1
            grew = True
        if not grew:
            break

    indices = list(range(start_pos, end_pos + 1))
    duration_sec = (timeline[end_pos][1] - timeline[start_pos][1])
    if duration_sec < cfg.min_roi_duration_sec:
        while duration_sec < cfg.min_roi_duration_sec and (start_pos > 0 or end_pos < len(timeline) - 1):
            if start_pos > 0:
                start_pos -= 1
            if end_pos < len(timeline) - 1:
                end_pos += 1
            duration_sec = timeline[end_pos][1] - timeline[start_pos][1]
        indices = list(range(start_pos, end_pos + 1))

    return indices


def _build_bbox_sequence(
    members: list[dict[str, Any]],
    timeline: list[tuple[int, float]],
    timeline_positions: list[int],
    aux_track_obs: dict[int, list[dict]],
    *,
    frame_w: int,
    frame_h: int,
    config: EventRoiConfig,
) -> tuple[list[dict[str, Any]], list[int]]:
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for m in members:
        by_frame[int(m["frame_idx"])].append(m)

    group_union = _group_union_bbox(members)
    mean_area = float(np.mean([m["bbox_area_ratio"] for m in members]))
    group_spatial = expand_bbox_context(
        group_union, area_ratio=mean_area, frame_w=frame_w, frame_h=frame_h, config=config,
    )

    seq: list[dict[str, Any]] = []
    all_aux: set[int] = set()
    prev_bb: list[int] | None = None

    for pos in timeline_positions:
        fi, ts = timeline[pos]
        if fi in by_frame:
            frame_union = _frame_union_bbox(by_frame[fi])
            base_boxes = [
                {"x1": group_union[0], "y1": group_union[1], "x2": group_union[2], "y2": group_union[3]},
                {"x1": frame_union[0], "y1": frame_union[1], "x2": frame_union[2], "y2": frame_union[3]},
            ]
            u = union_bbox(base_boxes)
            base = [int(u["x1"]), int(u["y1"]), int(u["x2"]), int(u["y2"])]
        else:
            base = list(group_union)

        expanded = _expand_bbox_context(
            base, area_ratio=mean_area, frame_w=frame_w, frame_h=frame_h, config=config,
        )
        absorbed, aux_ids = _absorb_auxiliary_detections(
            expanded, fi, aux_track_obs, frame_w=frame_w, frame_h=frame_h, config=config,
        )
        all_aux.update(aux_ids)

        if prev_bb is not None:
            alpha = config.smooth_alpha
            absorbed = [int(round(alpha * absorbed[i] + (1 - alpha) * prev_bb[i])) for i in range(4)]

        seq.append({"frame_idx": fi, "timestamp_sec": round(ts, 4), "bbox": absorbed})
        prev_bb = absorbed

    seq = _smooth_bbox_tube(seq, alpha=config.tube_smooth_alpha)
    return seq, sorted(all_aux)


def expand_bbox_context(
    bbox: list[int],
    *,
    area_ratio: float,
    frame_w: int,
    frame_h: int,
    config: EventRoiConfig,
) -> list[int]:
    return _expand_bbox_context(bbox, area_ratio=area_ratio, frame_w=frame_w, frame_h=frame_h, config=config)


def _scene_penalty(frame_indices: set[int], frame_df: pd.DataFrame) -> float:
    if not frame_indices or frame_df.empty or "scene_change_score" not in frame_df.columns:
        return 0.0
    sub = frame_df[frame_df["frame_idx"].isin(frame_indices)]
    return float(sub["scene_change_score"].mean()) if not sub.empty else 0.0


def _bridge_active_positions(positions: list[int], max_gap: int) -> list[int]:
    """Fill short gaps between blob-active sample positions."""
    if len(positions) < 2 or max_gap <= 0:
        return positions
    bridged: set[int] = set(positions)
    for i in range(len(positions) - 1):
        gap = positions[i + 1] - positions[i]
        if 1 < gap <= max_gap + 1:
            for p in range(positions[i] + 1, positions[i + 1]):
                bridged.add(p)
    return sorted(bridged)


def _smooth_bbox_tube(seq: list[dict[str, Any]], *, alpha: float) -> list[dict[str, Any]]:
    """Temporal smoothing so ROI tube stays visible through weak-motion frames."""
    if len(seq) < 2:
        return seq
    out: list[dict[str, Any]] = [dict(seq[0])]
    for entry in seq[1:]:
        prev = out[-1]["bbox"]
        curr = entry["bbox"]
        smoothed = [int(round(alpha * c + (1.0 - alpha) * p)) for p, c in zip(prev, curr)]
        out.append({**entry, "bbox": smoothed})
    return out


def _bbox_dict(b: list[int]) -> dict[str, int]:
    return {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]}


def _obs_near_bbox(obs: dict[str, Any], probe: dict[str, int], frame_w: int, frame_h: int, dist: float) -> bool:
    tb = {"x1": obs["x1"], "y1": obs["y1"], "x2": obs["x2"], "y2": obs["y2"]}
    if bbox_iou(probe, tb) >= 0.02:
        return True
    return center_distance_norm(probe, tb, frame_w, frame_h) <= dist


def _score_group_components(
    members: list[dict[str, Any]],
    bbox_sequence: list[dict[str, Any]],
    aux_track_obs: dict[int, list[dict]],
    *,
    frame_df: pd.DataFrame,
    frame_w: int,
    frame_h: int,
    sampled_fps: float,
    config: EventRoiConfig,
) -> dict[str, Any]:
    """Decomposed scores for event ROI ranking and debug."""
    cfg = config
    motion_raw = float(np.mean([m["blob_importance"] for m in members]))
    active_frames = {int(m["frame_idx"]) for m in members}
    persistence_sec = len(active_frames) / max(sampled_fps, EPS)
    mean_area = float(np.mean([m["bbox_area_ratio"] for m in members]))
    mean_coh = float(np.mean([m["flow_direction_coherence"] for m in members]))
    track_duration = len(bbox_sequence) / max(sampled_fps, EPS)

    small_persistent = mean_area <= cfg.small_event_area_ratio and persistence_sec >= 0.4
    motion_score = round(
        min(1.0, (
            0.45 * min(1.0, motion_raw * 2.0)
            + 0.35 * min(1.0, persistence_sec / 1.5)
            + 0.20 * (1.0 if small_persistent else 0.35 * min(1.0, persistence_sec / 2.0))
        )),
        4,
    )

    person_frames = 0
    vehicle_frames = 0
    interaction_frames = 0
    n_seq = max(len(bbox_sequence), 1)
    for entry in bbox_sequence:
        fi = int(entry["frame_idx"])
        probe = _bbox_dict(entry["bbox"])
        has_person = False
        has_vehicle = False
        for obs_list in aux_track_obs.values():
            for obs in obs_list:
                if int(obs["frame_idx"]) != fi:
                    continue
                cls = str(obs.get("class_name", ""))
                if not _obs_near_bbox(obs, probe, frame_w, frame_h, PROXIMITY_DIST):
                    continue
                if cls in PERSON_CLASSES:
                    has_person = True
                if cls in VEHICLE_CLASSES:
                    has_vehicle = True
        if has_person:
            person_frames += 1
        if has_vehicle:
            vehicle_frames += 1
        if has_person and has_vehicle:
            interaction_frames += 1

    object_proximity_score = round(
        min(1.0, (person_frames + vehicle_frames) / n_seq * 1.2),
        4,
    )
    interaction_score = round(
        min(1.0, interaction_frames / n_seq * 2.5 + (0.2 if small_persistent and interaction_frames else 0.0)),
        4,
    )
    if person_frames > 0 and vehicle_frames > 0 and mean_area <= 0.05:
        interaction_score = round(min(1.0, interaction_score + 0.15), 4)

    union = _group_union_bbox(members)
    union_area_ratio = ((union[2] - union[0]) * (union[3] - union[1])) / max(frame_w * frame_h, 1)

    jitter_ratio = 0.0
    if not frame_df.empty and "background_jitter_flag" in frame_df.columns:
        sub = frame_df[frame_df["frame_idx"].isin(active_frames)]
        if not sub.empty:
            jitter_ratio = float(sub["background_jitter_flag"].astype(bool).mean())

    global_motion_pen = 0.0
    if not frame_df.empty and "global_motion_score" in frame_df.columns:
        sub = frame_df[frame_df["frame_idx"].isin(active_frames)]
        if not sub.empty:
            global_motion_pen = float(sub["global_motion_score"].mean())

    is_coherent_flow = (
        mean_coh >= cfg.background_flow_coherence
        and mean_area >= cfg.background_flow_area_ratio
        and track_duration >= cfg.background_flow_min_duration_sec
    )
    vehicle_only = vehicle_frames > 0 and person_frames == 0
    small_localized = mean_area <= cfg.small_event_area_ratio

    if vehicle_frames > 0 and small_localized:
        object_proximity_score = round(
            min(1.0, object_proximity_score + 0.35 * (vehicle_frames / n_seq)),
            4,
        )
        interaction_score = round(
            min(1.0, interaction_score + 0.45 * (vehicle_frames / n_seq)),
            4,
        )

    background_flow_penalty = round(min(1.0, (
        (0.35 if is_coherent_flow and not small_localized else 0.0)
        + (0.25 if is_coherent_flow and vehicle_only and not small_localized else 0.0)
        + (0.20 if mean_coh >= 0.85 and mean_area >= 0.04 and not small_localized else 0.0)
        + (0.25 * jitter_ratio)
        + (0.15 * min(1.0, global_motion_pen))
        + (0.15 if union_area_ratio >= 0.18 and interaction_frames == 0 and not small_localized else 0.0)
    )), 4)

    scene_pen = _scene_penalty(active_frames, frame_df)
    jitter_penalty = 0.35 if mean_area >= cfg.jitter_group_area and mean_coh >= 0.70 else 0.0

    event_score = round(max(0.0, (
        0.18 * motion_score
        + 0.22 * object_proximity_score
        + 0.42 * interaction_score
        + 0.12 * (1.0 if small_localized and (person_frames or vehicle_frames) else 0.0)
        - 0.55 * background_flow_penalty
        - cfg.scene_change_penalty_weight * min(1.0, scene_pen)
        - jitter_penalty
    )), 6)

    is_background_flow_region = bool(
        is_coherent_flow
        and not small_localized
        and interaction_score < 0.15
        and background_flow_penalty >= 0.40
        and (vehicle_only or union_area_ratio >= 0.12)
    )

    return {
        "motion_score": motion_score,
        "object_proximity_score": object_proximity_score,
        "interaction_score": interaction_score,
        "background_flow_penalty": background_flow_penalty,
        "event_score": event_score,
        "is_background_flow_region": is_background_flow_region,
        "person_near_frames": person_frames,
        "vehicle_near_frames": vehicle_frames,
        "interaction_frames": interaction_frames,
        "mean_bbox_area_ratio": round(mean_area, 4),
        "mean_flow_coherence": round(mean_coh, 4),
        "union_area_ratio": round(union_area_ratio, 4),
    }


def _score_blob_group(
    members: list[dict[str, Any]],
    bbox_sequence: list[dict[str, Any]],
    aux_track_obs: dict[int, list[dict]],
    *,
    frame_df: pd.DataFrame,
    frame_w: int,
    frame_h: int,
    sampled_fps: float,
    config: EventRoiConfig,
) -> dict[str, Any]:
    return _score_group_components(
        members,
        bbox_sequence,
        aux_track_obs,
        frame_df=frame_df,
        frame_w=frame_w,
        frame_h=frame_h,
        sampled_fps=sampled_fps,
        config=config,
    )


def _bbox_iou_list(a: list[int], b: list[int]) -> float:
    probe_a = {"x1": a[0], "y1": a[1], "x2": a[2], "y2": a[3]}
    probe_b = {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]}
    return float(bbox_iou(probe_a, probe_b))


def _temporal_iou(seq_a: list[dict], seq_b: list[dict]) -> float:
    fa = {int(e["frame_idx"]) for e in seq_a}
    fb = {int(e["frame_idx"]) for e in seq_b}
    if not fa or not fb:
        return 0.0
    return len(fa & fb) / len(fa | fb)


def _mean_spatial_iou(seq_a: list[dict], seq_b: list[dict]) -> float:
    fb_by_frame = {int(e["frame_idx"]): e["bbox"] for e in seq_b}
    ious: list[float] = []
    for e in seq_a:
        fi = int(e["frame_idx"])
        if fi in fb_by_frame:
            ious.append(_bbox_iou_list(e["bbox"], fb_by_frame[fi]))
    if not ious:
        mid_a = seq_a[len(seq_a) // 2]["bbox"]
        mid_b = seq_b[len(seq_b) // 2]["bbox"]
        return _bbox_iou_list(mid_a, mid_b)
    return float(np.mean(ious))


def build_event_roi_candidates(
    blobs_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    aux_track_obs: dict[int, list[dict]] | None = None,
    *,
    frame_w: int,
    frame_h: int,
    sampled_fps: float,
    config: EventRoiConfig | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Build ROI candidates from motion blob groups only.
    aux_track_obs: optional ByteTrack observations for bbox expansion (not ROI identity).
    Returns (accepted_candidates, rejected_records, background_flow_regions).
    """
    cfg = config or EventRoiConfig()
    aux_track_obs = aux_track_obs or {}
    rejected: list[dict[str, Any]] = []
    timeline = _sampled_timeline(frame_df)
    if not timeline:
        return [], rejected, []

    filtered = filter_jitter_blobs(blobs_df, frame_df, config=cfg)
    if filtered.empty:
        return [], rejected, []

    blobs = [_blob_dict(r) for r in filtered.to_dict("records")]
    frame_to_pos = {fi: i for i, (fi, _) in enumerate(timeline)}
    groups = _link_spatiotemporal_groups(
        blobs,
        frame_w=frame_w,
        frame_h=frame_h,
        sampled_fps=sampled_fps,
        config=cfg,
        frame_to_pos=frame_to_pos,
    )

    candidates: list[dict[str, Any]] = []
    background_flow_regions: list[dict[str, Any]] = []
    for group_id, members in enumerate(groups, start=1):
        base_reject = {
            "group_id": group_id,
            "primary_source": "blob_group",
            "num_member_blobs": len(members),
        }
        if not members:
            rejected.append({**base_reject, "reason": "empty_group"})
            continue

        if any(m["bbox_area_ratio"] >= 0.55 for m in members):
            rejected.append({**base_reject, "reason": "full_frame_blob_member"})
            continue

        mean_area = float(np.mean([m["bbox_area_ratio"] for m in members]))
        mean_coh = float(np.mean([m["flow_direction_coherence"] for m in members]))

        active_frames = {int(m["frame_idx"]) for m in members}
        active_frames = _extend_active_frames_with_aux_tracks(
            active_frames,
            members,
            aux_track_obs,
            frame_w=frame_w,
            frame_h=frame_h,
            mean_area=mean_area,
            sampled_fps=sampled_fps,
            config=cfg,
        )
        timeline_positions = _timeline_indices_for_group(
            active_frames, timeline, cfg, sampled_fps,
        )
        if len(timeline_positions) < cfg.min_roi_frames:
            rejected.append({
                **base_reject,
                "reason": "duration_too_short",
                "bbox_sequence_len": len(timeline_positions),
                "min_required_frames": cfg.min_roi_frames,
            })
            continue

        duration_sec = timeline[timeline_positions[-1]][1] - timeline[timeline_positions[0]][1]
        if duration_sec < cfg.min_roi_duration_sec:
            rejected.append({
                **base_reject,
                "reason": "duration_sec_below_min",
                "duration_sec": round(duration_sec, 4),
                "min_required_sec": cfg.min_roi_duration_sec,
            })
            continue

        bbox_sequence, aux_ids = _build_bbox_sequence(
            members,
            timeline,
            timeline_positions,
            aux_track_obs,
            frame_w=frame_w,
            frame_h=frame_h,
            config=cfg,
        )

        if len(bbox_sequence) < cfg.min_roi_frames:
            rejected.append({
                **base_reject,
                "reason": "bbox_sequence_too_short",
                "bbox_sequence_len": len(bbox_sequence),
            })
            continue

        scores = _score_blob_group(
            members,
            bbox_sequence,
            aux_track_obs,
            frame_df=frame_df,
            frame_w=frame_w,
            frame_h=frame_h,
            sampled_fps=sampled_fps,
            config=cfg,
        )
        gate = apply_event_relevance_gate(
            members=members,
            bbox_sequence=bbox_sequence,
            aux_track_obs=aux_track_obs,
            frame_df=frame_df,
            frame_w=frame_w,
            frame_h=frame_h,
            sampled_fps=sampled_fps,
            base_scores=scores,
            gate_cfg=cfg.gate_config(),
        )
        blob_stats = compute_blob_persistence_stats(members)
        event_relevance_score = gate["event_relevance_score"]
        motion_pattern = gate["motion_pattern"]
        mean_coh = scores["mean_flow_coherence"]
        mean_area = scores["mean_bbox_area_ratio"]

        if motion_pattern == "background_flow" or scores["is_background_flow_region"]:
            background_flow_regions.append({
                "group_id": group_id,
                "start_sec": bbox_sequence[0]["timestamp_sec"],
                "end_sec": bbox_sequence[-1]["timestamp_sec"],
                "duration_sec": round(bbox_sequence[-1]["timestamp_sec"] - bbox_sequence[0]["timestamp_sec"], 4),
                "event_relevance_score": event_relevance_score,
                "motion_pattern": motion_pattern,
                "motion_score": scores["motion_score"],
                "object_proximity_score": scores["object_proximity_score"],
                "interaction_score": scores["interaction_score"],
                "background_flow_penalty": scores["background_flow_penalty"],
                "mean_bbox_area_ratio": mean_area,
                "mean_flow_coherence": mean_coh,
                "status": "background_flow_region",
            })

        start_fi = int(bbox_sequence[0]["frame_idx"])
        end_fi = int(bbox_sequence[-1]["frame_idx"])
        candidates.append({
            "candidate_id": group_id,
            "group_id": group_id,
            "primary_source": "blob_group",
            "auxiliary_track_ids": aux_ids,
            "event_score": scores["event_score"],
            "event_relevance_score": event_relevance_score,
            "motion_score": scores["motion_score"],
            "object_proximity_score": scores["object_proximity_score"],
            "interaction_score": scores["interaction_score"],
            "background_flow_penalty": scores["background_flow_penalty"],
            "is_background_flow_region": scores["is_background_flow_region"],
            "num_member_blobs": len(members),
            "num_blob_active_frames": blob_stats["num_blob_active_frames"],
            "mean_blob_importance": blob_stats["mean_blob_importance"],
            "max_blob_importance": blob_stats["max_blob_importance"],
            "high_importance_frame_ratio": blob_stats["high_importance_frame_ratio"],
            "person_near_frames": gate.get("person_near_frames", 0),
            "vehicle_near_frames": gate.get("vehicle_near_frames", 0),
            "bbox_sequence_len": len(bbox_sequence),
            "duration_sec": round(bbox_sequence[-1]["timestamp_sec"] - bbox_sequence[0]["timestamp_sec"], 4),
            "start_frame_idx": start_fi,
            "end_frame_idx": end_fi,
            "start_sec": bbox_sequence[0]["timestamp_sec"],
            "end_sec": bbox_sequence[-1]["timestamp_sec"],
            "bbox_sequence": bbox_sequence,
            "mean_flow_coherence": mean_coh,
            "mean_bbox_area_ratio": mean_area,
            "motion_pattern": motion_pattern,
            "pattern_reason": gate.get("pattern_reason", ""),
            "rejection_reason": gate.get("rejection_reason", ""),
            "net_displacement_ratio": gate["net_displacement_ratio"],
            "path_linearity": gate["path_linearity"],
            "direction_consistency": gate["direction_consistency"],
            "speed_mean": gate["speed_mean"],
            "speed_cv": gate["speed_cv"],
            "stop_ratio": gate["stop_ratio"],
            "dwell_ratio": gate["dwell_ratio"],
            "object_proximity_ratio": gate["object_proximity_ratio"],
            "dwell_near_object_ratio": gate["dwell_near_object_ratio"],
            "person_vehicle_dwell_ratio": gate["person_vehicle_dwell_ratio"],
            "semantic_anchor_score": gate["semantic_anchor_score"],
            "transit_penalty": gate["transit_penalty"],
            "jitter_penalty": gate["jitter_penalty"],
            "interaction_dwell_score": gate["interaction_dwell_score"],
            "stop_or_slowdown_score": gate["stop_or_slowdown_score"],
            "localized_persistence_score": gate["localized_persistence_score"],
            "small_critical_motion_score": gate["small_critical_motion_score"],
            "interaction_ratio": gate["interaction_ratio"],
            "door_trunk_ratio": gate.get("door_trunk_ratio", 0.0),
            "entrance_ratio": gate.get("entrance_ratio", 0.0),
            "selection_status": "candidate",
        })

    candidates.sort(key=lambda c: c["event_relevance_score"], reverse=True)
    return candidates, rejected, background_flow_regions


def select_event_roi_tracks(
    candidates: list[dict[str, Any]],
    *,
    max_rois: int = MAX_ROI_TRACKS,
    config: EventRoiConfig | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Coverage-aware event_motion ROI selection. Returns (tracks, rejected, debug_rejected)."""
    cfg = config or EventRoiConfig()
    gate_cfg = cfg.gate_config()
    eligible_patterns = {"event_motion"}
    if gate_cfg.allow_ambiguous_roi:
        eligible_patterns.add("ambiguous_motion")

    selected, selection_rejected, debug_rejected = select_coverage_aware_rois(
        candidates,
        max_rois=max_rois,
        eligible_patterns=eligible_patterns,
        selection_cfg=cfg.selection_config(),
        nms_temporal_iou=cfg.nms_temporal_iou,
        nms_spatial_iou=cfg.nms_spatial_iou,
        min_roi_frames=cfg.min_roi_frames,
    )

    tracks: list[dict[str, Any]] = []
    for roi_id, cand in enumerate(selected, start=1):
        tracks.append({
            "roi_id": roi_id,
            "source": "blob_group",
            "primary_source": "blob_group",
            "group_id": cand["group_id"],
            "auxiliary_track_ids": cand.get("auxiliary_track_ids", []),
            "source_track_ids": [],
            "source_tube_ids": [],
            "event_score": cand.get("event_score", 0.0),
            "event_relevance_score": cand.get("event_relevance_score", 0.0),
            "revised_event_score": cand.get("revised_event_score", 0.0),
            "motion_score": cand.get("motion_score", 0.0),
            "object_proximity_score": cand.get("object_proximity_score", 0.0),
            "interaction_score": cand.get("interaction_score", 0.0),
            "background_flow_penalty": cand.get("background_flow_penalty", 0.0),
            "motion_pattern": cand.get("motion_pattern", "event_motion"),
            "duration_sec": cand.get("duration_sec", 0.0),
            "bbox_sequence_len": len(cand.get("bbox_sequence", [])),
            "bbox_sequence": cand.get("bbox_sequence", []),
            "start_sec": cand.get("start_sec", 0.0),
            "end_sec": cand.get("end_sec", 0.0),
            "start_frame_idx": cand.get("start_frame_idx"),
            "end_frame_idx": cand.get("end_frame_idx"),
            "selection_status": "selected",
            "selection_reason": cand.get("selection_reason", ""),
            "replacement_reason": cand.get("replacement_reason", ""),
        })
    return tracks, selection_rejected, debug_rejected


def build_event_roi_debug_report(
    *,
    candidates: list[dict[str, Any]],
    build_rejected: list[dict[str, Any]],
    selected_tracks: list[dict[str, Any]],
    selection_rejected: list[dict[str, Any]],
    background_flow_regions: list[dict[str, Any]],
    motion_type: str,
    sampled_fps: float,
    config: EventRoiConfig | None = None,
    debug_rejected_high_rank: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cfg = config or EventRoiConfig()

    def _candidate_summary(c: dict[str, Any], *, status: str) -> dict[str, Any]:
        return {
            "group_id": c.get("group_id"),
            "start_sec": c.get("start_sec"),
            "end_sec": c.get("end_sec"),
            "duration_sec": c.get("duration_sec"),
            "original_score": c.get("original_score", c.get("event_relevance_score")),
            "revised_event_score": c.get("revised_event_score"),
            "event_relevance_score": c.get("event_relevance_score"),
            "event_score": c.get("event_score"),
            "selected_rank_before": c.get("selected_rank_before"),
            "selected_rank_after": c.get("selected_rank_after"),
            "rejection_reason": c.get("rejection_reason"),
            "replacement_reason": c.get("replacement_reason"),
            "selection_reason": c.get("selection_reason"),
            "duration_score": c.get("duration_score"),
            "persistence_score": c.get("persistence_score"),
            "high_importance_blob_score": c.get("high_importance_blob_score"),
            "vehicle_only_penalty": c.get("vehicle_only_penalty"),
            "short_event_penalty": c.get("short_event_penalty"),
            "motion_pattern": c.get("motion_pattern"),
            "selection_status": c.get("selection_status", status),
        }

    gate_summary = summarize_gate_for_plan(candidates, selected_tracks)

    return {
        "motion_type": motion_type,
        "sampled_fps": sampled_fps,
        "rules": {
            "primary_source": "blob_group",
            "auxiliary": "bytetrack_boxes_for_expansion_only",
            "event_relevance_gate": True,
            "min_roi_duration_sec": cfg.min_roi_duration_sec,
            "min_roi_frames": cfg.min_roi_frames,
            "context_frames": cfg.context_frames,
            "min_event_relevance_score": cfg.min_event_relevance_score,
            "allow_ambiguous_roi": cfg.allow_ambiguous_roi,
            "final_roi_patterns": ["event_motion"] + (["ambiguous_motion"] if cfg.allow_ambiguous_roi else []),
            "no_forward_fill_outside_track": True,
            "scoring_prefers": [
                "revised_event_score",
                "duration_score",
                "persistence_score",
                "high_importance_blob_score",
                "human_scale_motion_score",
                "long_persistent_priority",
            ],
            "scoring_downweights": [
                "vehicle_only_penalty",
                "short_event_penalty",
                "interaction_without_person_penalty",
            ],
        },
        "summary": {
            "num_candidates": len(candidates),
            "num_build_rejected": len(build_rejected),
            "num_selected": len(selected_tracks),
            "num_selection_rejected": len(selection_rejected),
            "num_background_flow_regions": len(background_flow_regions),
            **gate_summary,
        },
        "selected_rois": [
            {
                "roi_id": t["roi_id"],
                "group_id": t["group_id"],
                "start_sec": t.get("start_sec"),
                "end_sec": t.get("end_sec"),
                "duration_sec": t.get("duration_sec"),
                "event_relevance_score": t.get("event_relevance_score"),
                "revised_event_score": t.get("revised_event_score"),
                "event_score": t.get("event_score"),
                "motion_score": t.get("motion_score"),
                "object_proximity_score": t.get("object_proximity_score"),
                "interaction_score": t.get("interaction_score"),
                "background_flow_penalty": t.get("background_flow_penalty"),
                "motion_pattern": t.get("motion_pattern"),
                "selection_status": "selected",
                "reason": t.get("selection_reason", "coverage_aware_selection"),
            }
            for t in selected_tracks
        ],
        "candidates": [_candidate_summary(c, status="candidate") for c in candidates],
        "background_flow_regions": background_flow_regions,
        "build_rejected": build_rejected,
        "selection_rejected": selection_rejected,
        "debug_rejected_high_rank": [
            _candidate_summary(c, status="debug_rejected")
            for c in (debug_rejected_high_rank or [])
        ],
        "gate_summary": gate_summary,
    }


_CANDIDATE_CSV_COLUMNS = [
    "group_id", "motion_pattern", "selection_status",
    "rejection_reason", "replacement_reason", "selection_reason", "pattern_reason",
    "start_sec", "end_sec", "duration_sec",
    "original_score", "revised_event_score", "event_relevance_score", "event_score",
    "selected_rank_before", "selected_rank_after",
    "duration_score", "persistence_score", "high_importance_blob_score",
    "spatial_compactness_score", "human_scale_motion_score",
    "vehicle_only_penalty", "short_event_penalty", "interaction_without_person_penalty",
    "motion_score", "object_proximity_score", "interaction_score",
    "interaction_dwell_score", "semantic_anchor_score",
    "stop_or_slowdown_score", "localized_persistence_score", "small_critical_motion_score",
    "transit_penalty", "background_flow_penalty", "jitter_penalty",
    "net_displacement_ratio", "path_linearity", "direction_consistency",
    "speed_mean", "speed_cv", "stop_ratio", "dwell_ratio",
    "object_proximity_ratio", "dwell_near_object_ratio",
    "person_vehicle_dwell_ratio", "interaction_ratio",
    "person_near_frames", "vehicle_near_frames",
    "mean_blob_importance", "max_blob_importance", "high_importance_frame_ratio",
    "door_trunk_ratio", "entrance_ratio",
    "mean_bbox_area_ratio", "mean_flow_coherence",
    "num_member_blobs", "num_blob_active_frames", "bbox_sequence_len",
]


def save_event_roi_candidates_csv(candidates: list[dict[str, Any]], path: Path) -> Path:
    """Write flat candidate table (no bbox_sequence) for inspection."""
    rows = []
    for c in candidates:
        rows.append({col: c.get(col) for col in _CANDIDATE_CSV_COLUMNS})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def load_auxiliary_track_obs(video_dir: Path) -> dict[int, list[dict]]:
    """Load object_tracks.csv for ROI bbox expansion only (not ROI identity)."""
    from motion_analyzer.v1.io import data_dir

    ddir = data_dir(video_dir)
    tracks_csv = ddir / "object_tracks.csv"
    if not tracks_csv.is_file():
        return {}
    obs_map: dict[int, list[dict]] = defaultdict(list)
    import pandas as pd

    for row in pd.read_csv(tracks_csv).to_dict("records"):
        tid = int(row["track_id"])
        obs_map[tid].append(row)
    return dict(obs_map)


def build_event_roi_pipeline(
    blobs_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    aux_track_obs: dict[int, list[dict]] | None = None,
    *,
    frame_w: int,
    frame_h: int,
    sampled_fps: float,
    motion_type: str = "event_motion",
    config: EventRoiConfig | None = None,
) -> EventRoiBuildResult:
    cfg = config or EventRoiConfig()
    if motion_type != "event_motion":
        return EventRoiBuildResult(
            debug_report={"motion_type": motion_type, "skipped": "background_motion"},
        )

    candidates, build_rejected, background_flow_regions = build_event_roi_candidates(
        blobs_df,
        frame_df,
        aux_track_obs,
        frame_w=frame_w,
        frame_h=frame_h,
        sampled_fps=sampled_fps,
        config=cfg,
    )
    selected, selection_rejected, debug_rejected = select_event_roi_tracks(
        candidates, max_rois=cfg.max_roi_tracks, config=cfg,
    )
    report = build_event_roi_debug_report(
        candidates=candidates,
        build_rejected=build_rejected,
        selected_tracks=selected,
        selection_rejected=selection_rejected,
        debug_rejected_high_rank=debug_rejected,
        background_flow_regions=background_flow_regions,
        motion_type=motion_type,
        sampled_fps=sampled_fps,
        config=cfg,
    )
    return EventRoiBuildResult(
        candidates=candidates,
        rejected=build_rejected,
        selected_tracks=selected,
        selection_rejected=selection_rejected,
        debug_rejected_high_rank=debug_rejected,
        background_flow_regions=background_flow_regions,
        debug_report=report,
    )


# Backward-compatible alias
def build_event_roi_pipeline_legacy(
    blobs_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    track_obs_map: dict[int, list[dict]],
    **kwargs: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result = build_event_roi_pipeline(blobs_df, frame_df, track_obs_map, **kwargs)
    return result.candidates, result.selected_tracks
