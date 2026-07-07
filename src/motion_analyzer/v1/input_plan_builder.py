"""Adaptive VLM input plan construction from motion tubes."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

SPARSE_GLOBAL_FPS = 0.25
MAX_ROI_TRACKS = 2
MERGE_CENTER_DISTANCE = 0.18
LARGE_ROI_AREA_RATIO = 0.12

MotionType = Literal["background_motion", "event_motion"]
RoiPattern = Literal["none", "single_roi", "multi_blob_merge", "large_small_blobs"]


def _sparse_global_frames(
    frame_df: pd.DataFrame,
    *,
    sparse_fps: float = SPARSE_GLOBAL_FPS,
    native_fps: float = 5.0,
) -> list[dict[str, Any]]:
    """Select sparse global frames at *sparse_fps* from sampled frame indices."""
    if frame_df.empty:
        return []

    indices = frame_df["frame_idx"].tolist()
    timestamps = frame_df["timestamp_sec"].tolist()
    step = max(1, int(round(native_fps / sparse_fps)))

    entries = []
    for i in range(0, len(indices), step):
        entries.append(
            {
                "frame_idx": int(indices[i]),
                "timestamp_sec": float(timestamps[i]),
                "type": "sparse_global",
                "crop_bbox": None,
                "tube_id": None,
                "importance": 0.0,
            }
        )
    return entries


def _high_motion_windows(
    frame_df: pd.DataFrame,
    *,
    top_fraction: float = 0.15,
    window_half: int = 3,
) -> list[dict[str, Any]]:
    """Dense full-frame windows around high-motion periods (v2 legacy plan helper)."""
    if frame_df.empty:
        return []

    threshold = frame_df["motion_magnitude_mean"].quantile(1.0 - top_fraction)
    high_frames = frame_df[frame_df["motion_magnitude_mean"] >= threshold]
    indices = frame_df["frame_idx"].tolist()
    idx_to_ts = dict(zip(frame_df["frame_idx"], frame_df["timestamp_sec"]))

    entries = []
    seen: set[int] = set()
    for _, row in high_frames.iterrows():
        center_idx = int(row["frame_idx"])
        try:
            pos = indices.index(center_idx)
        except ValueError:
            continue
        for offset in range(-window_half, window_half + 1):
            p = pos + offset
            if 0 <= p < len(indices):
                fidx = int(indices[p])
                if fidx in seen:
                    continue
                seen.add(fidx)
                entries.append(
                    {
                        "frame_idx": fidx,
                        "timestamp_sec": float(idx_to_ts[fidx]),
                        "type": "dense_full_frame",
                        "crop_bbox": None,
                        "tube_id": None,
                        "importance": float(row["motion_magnitude_mean"]),
                    }
                )
    return entries


def _scene_boundary_frames(
    frame_df: pd.DataFrame,
    *,
    scene_threshold: float = 1.0,
) -> list[dict[str, Any]]:
    """Frames at scene change boundaries (v2 legacy plan helper)."""
    if frame_df.empty:
        return []

    indices = frame_df["frame_idx"].tolist()
    idx_to_ts = dict(zip(frame_df["frame_idx"], frame_df["timestamp_sec"]))
    entries = []

    for _, row in frame_df.iterrows():
        if row["scene_change_score"] <= scene_threshold:
            continue
        fidx = int(row["frame_idx"])
        try:
            pos = indices.index(fidx)
        except ValueError:
            continue
        for offset, role in [(-1, "scene_before"), (0, "scene_boundary"), (1, "scene_after")]:
            p = pos + offset
            if 0 <= p < len(indices):
                bfidx = int(indices[p])
                entries.append(
                    {
                        "frame_idx": bfidx,
                        "timestamp_sec": float(idx_to_ts[bfidx]),
                        "type": role,
                        "crop_bbox": None,
                        "tube_id": None,
                        "importance": float(row["scene_change_score"]),
                    }
                )
    return entries


def _obs_bbox(obs: dict) -> list[int]:
    return [int(obs["x1"]), int(obs["y1"]), int(obs["x2"]), int(obs["y2"])]


def _tube_observations(tube: dict) -> list[dict]:
    return tube.get("_observations", [])


def _attach_observations(
    tubes: list[dict], tube_segments_obs: dict[int, list[dict]]
) -> list[dict]:
    enriched = []
    for t in tubes:
        t_copy = dict(t)
        t_copy["_observations"] = tube_segments_obs.get(t["tube_id"], [])
        enriched.append(t_copy)
    return enriched


def _tube_center_norm(tube: dict[str, Any]) -> tuple[float, float]:
    """Normalized (0-1) center from tube observations or metadata."""
    obs = _tube_observations(tube)
    if obs:
        cx = float(np.mean([o["center_x"] for o in obs]))
        cy = float(np.mean([o["center_y"] for o in obs]))
        return cx, cy
    return 0.5, 0.5


def _center_distance_norm(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax, ay = _tube_center_norm(a)
    bx, by = _tube_center_norm(b)
    return float(np.hypot(ax - bx, ay - by))


def _group_mean_center(tubes: list[dict[str, Any]]) -> tuple[float, float]:
    if not tubes:
        return 0.5, 0.5
    centers = [_tube_center_norm(t) for t in tubes]
    return float(np.mean([c[0] for c in centers])), float(np.mean([c[1] for c in centers]))


def _group_center_distance(group: list[dict[str, Any]], tube: dict[str, Any]) -> float:
    gx, gy = _group_mean_center(group)
    tx, ty = _tube_center_norm(tube)
    return float(np.hypot(gx - tx, gy - ty))


def _merge_tubes_into_roi_groups(
    tubes: list[dict[str, Any]],
    *,
    max_groups: int = MAX_ROI_TRACKS,
) -> list[list[dict[str, Any]]]:
    """Cluster tubes by importance ranking into at most *max_groups* spatial groups."""
    sorted_tubes = sorted(tubes, key=lambda t: t["tube_importance"], reverse=True)
    if not sorted_tubes:
        return []
    if len(sorted_tubes) <= max_groups:
        return [[t] for t in sorted_tubes]

    groups: list[list[dict[str, Any]]] = [[sorted_tubes[0]], [sorted_tubes[1]]]
    for tube in sorted_tubes[2:]:
        dists = [_group_center_distance(g, tube) for g in groups]
        nearest = int(np.argmin(dists))
        if dists[nearest] <= MERGE_CENTER_DISTANCE:
            groups[nearest].append(tube)
        else:
            smallest_idx = min(range(len(groups)), key=lambda i: len(groups[i]))
            groups[smallest_idx].append(tube)

    groups.sort(
        key=lambda g: max(t["tube_importance"] for t in g),
        reverse=True,
    )
    return groups[:max_groups]


def _roi_sampling_fps(mean_speed: float) -> float:
    """Map ROI internal motion speed to sampling fps."""
    if mean_speed < 15.0:
        return 0.5
    if mean_speed < 40.0:
        return 1.0
    return 2.0


def _merged_bbox_sequence(tubes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect observation bboxes across tubes, one bbox per frame (highest density)."""
    by_frame: dict[int, dict[str, Any]] = {}
    for tube in tubes:
        for obs in _tube_observations(tube):
            fidx = int(obs["frame_idx"])
            density = float(obs.get("motion_density", 0.0))
            if fidx not in by_frame or density > by_frame[fidx]["motion_density"]:
                by_frame[fidx] = {
                    "frame_idx": fidx,
                    "timestamp_sec": float(obs["timestamp_sec"]),
                    "bbox": _obs_bbox(obs),
                    "motion_density": density,
                }
    return [
        {
            "frame_idx": v["frame_idx"],
            "timestamp_sec": v["timestamp_sec"],
            "bbox": v["bbox"],
        }
        for _, v in sorted(by_frame.items())
    ]


def _mean_area_ratio(tubes: list[dict[str, Any]]) -> float:
    values = [float(t.get("mean_bbox_area_ratio", 0.0)) for t in tubes]
    return float(np.mean(values)) if values else 0.0


def _build_roi_input(
    tubes: list[dict[str, Any]],
    *,
    roi_id: int,
) -> dict[str, Any]:
    """Merge a tube group into one motion ROI track."""
    start_sec = min(float(t["start_time_sec"]) for t in tubes)
    end_sec = max(float(t["end_time_sec"]) for t in tubes)
    speeds = [float(t.get("mean_speed", 0.0)) for t in tubes]
    importances = [float(t["tube_importance"]) for t in tubes]
    total_imp = sum(importances) or 1.0
    mean_speed = sum(s * w for s, w in zip(speeds, importances)) / total_imp

    source = "motion_tube" if len(tubes) == 1 else "motion_tube"
    if len(tubes) > 1:
        source = "motion_tube"

    return {
        "type": "motion_roi",
        "roi_id": roi_id,
        "source": source,
        "source_tube_ids": [int(t["tube_id"]) for t in tubes],
        "merged_tube_count": len(tubes),
        "start_sec": round(start_sec, 4),
        "end_sec": round(end_sec, 4),
        "bbox_sequence": _merged_bbox_sequence(tubes),
        "mean_motion_speed": round(mean_speed, 4),
        "sampling_fps": _roi_sampling_fps(mean_speed),
        "mean_bbox_area_ratio": round(_mean_area_ratio(tubes), 8),
        "combined_importance": round(max(importances), 6),
    }


def _infer_roi_pattern(
    roi_inputs: list[dict[str, Any]],
) -> RoiPattern:
    if not roi_inputs:
        return "none"
    if len(roi_inputs) == 1:
        roi = roi_inputs[0]
        if roi.get("merged_tube_count", 1) > 1:
            return "multi_blob_merge"
        return "single_roi"

    areas = [float(r.get("mean_bbox_area_ratio", 0.0)) for r in roi_inputs]
    has_large = any(a >= LARGE_ROI_AREA_RATIO for a in areas)
    has_small = any(a < LARGE_ROI_AREA_RATIO * 0.5 for a in areas)
    if has_large and has_small:
        return "large_small_blobs"
    if any(r.get("merged_tube_count", 1) > 1 for r in roi_inputs):
        return "multi_blob_merge"
    return "single_roi"


def build_adaptive_input_plan_v1(
    motion_type: str,
    frame_df: pd.DataFrame,
    tubes: list[dict[str, Any]],
    tube_segments_obs: dict[int, list[dict]],
    *,
    video_id: str = "",
    sampled_fps: float = 5.0,
    max_roi_tracks: int = MAX_ROI_TRACKS,
    num_raw_blobs: int | None = None,
) -> dict[str, Any]:
    """
    Build motion-type-driven adaptive input plan for VLM construction.

    background_motion → sparse global only
    event_motion      → sparse global + up to 2 motion ROI tracks
    """
    enriched = _attach_observations(tubes, tube_segments_obs)
    sparse_frames = _sparse_global_frames(frame_df, native_fps=sampled_fps)

    global_inputs: list[dict[str, Any]] = [
        {
            "type": "sparse_global",
            "fps": SPARSE_GLOBAL_FPS,
            "always_include": True,
            "frame_indices": [f["frame_idx"] for f in sparse_frames],
            "timestamps_sec": [f["timestamp_sec"] for f in sparse_frames],
        }
    ]

    roi_inputs: list[dict[str, Any]] = []
    if motion_type == "event_motion":
        candidate_tubes = [
            t for t in enriched if _tube_observations(t) and t["tube_importance"] > 0
        ]
        groups = _merge_tubes_into_roi_groups(
            candidate_tubes, max_groups=max_roi_tracks
        )
        for idx, group in enumerate(groups, start=1):
            roi_inputs.append(_build_roi_input(group, roi_id=idx))

    roi_pattern = _infer_roi_pattern(roi_inputs)

    return {
        "video_id": video_id,
        "motion_type": motion_type,
        "global_inputs": global_inputs,
        "roi_inputs": roi_inputs,
        "event_anchors": [],
        "diagnostics": {
            "roi_pattern": roi_pattern,
            "num_raw_blobs": num_raw_blobs,
            "num_raw_tubes": len(tubes),
            "num_selected_rois": len(roi_inputs),
            "sparse_global_frame_count": len(sparse_frames),
            "motion_summary": {
                "max_tube_importance": round(
                    max((t["tube_importance"] for t in enriched), default=0.0), 6
                ),
                "num_candidate_tubes": sum(
                    1 for t in enriched if _tube_observations(t)
                ),
            },
        },
    }
