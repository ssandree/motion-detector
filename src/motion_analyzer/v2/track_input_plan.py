"""Adaptive input plan from moving object tracks + motion fallback."""

from __future__ import annotations

from typing import Any

import pandas as pd

from motion_analyzer.v1.input_plan_builder import (
    MAX_ROI_TRACKS,
    SPARSE_GLOBAL_FPS,
    _high_motion_windows,
    _scene_boundary_frames,
    _sparse_global_frames,
)
from motion_analyzer.v2.bbox_utils import expand_bbox, union_bbox
from motion_analyzer.v2.track_motion_features import DETECTOR_ONLY, OBSERVED_MOTION, WEAK_MOTION

PERSON_MARGIN = 0.30
VEHICLE_MARGIN = 0.20
FALLBACK_MARGIN = 0.50
VEHICLE_CLASSES = {"car", "bicycle", "motorcycle", "bus", "truck"}


def _crop_for_observed(obs: dict, frame_w: int, frame_h: int) -> list[int]:
    track_box = {"x1": obs["x1"], "y1": obs["y1"], "x2": obs["x2"], "y2": obs["y2"]}
    cls = obs.get("class_name", "")
    margin = VEHICLE_MARGIN if cls in VEHICLE_CLASSES else PERSON_MARGIN
    if obs.get("associated_motion_bbox"):
        bb = obs["associated_motion_bbox"]
        motion_box = {"x1": bb[0], "y1": bb[1], "x2": bb[2], "y2": bb[3]}
        merged = union_bbox([track_box, motion_box])
        return expand_bbox(merged["x1"], merged["y1"], merged["x2"], merged["y2"], margin, frame_w, frame_h)
    return expand_bbox(track_box["x1"], track_box["y1"], track_box["x2"], track_box["y2"], margin, frame_w, frame_h)


def _crop_for_weak(obs: dict, frame_w: int, frame_h: int) -> list[int]:
    cls = obs.get("class_name", "")
    margin = VEHICLE_MARGIN if cls in VEHICLE_CLASSES else PERSON_MARGIN
    return expand_bbox(obs["x1"], obs["y1"], obs["x2"], obs["y2"], margin, frame_w, frame_h)


def _track_roi_entries(
    track: dict[str, Any],
    observations: list[dict],
    *,
    frame_w: int,
    frame_h: int,
    dense: bool = False,
) -> list[dict[str, Any]]:
    moving_obs = [o for o in observations if o.get("observation_type") == OBSERVED_MOTION]
    if not moving_obs:
        moving_obs = [
            o for o in observations
            if o.get("observation_type") == WEAK_MOTION and track.get("class_name") == "person"
        ]
    if not moving_obs:
        return []

    if dense:
        selected = moving_obs
    else:
        peak = max(moving_obs, key=lambda o: o.get("motion_density", 0))
        selected = [moving_obs[0], peak, moving_obs[-1]]
        seen: set[int] = set()
        unique = []
        for o in selected:
            fi = int(o["frame_idx"])
            if fi not in seen:
                seen.add(fi)
                unique.append(o)
        selected = unique

    entries = []
    for obs in selected:
        obs_type = obs.get("observation_type", WEAK_MOTION)
        if obs_type == OBSERVED_MOTION:
            crop = _crop_for_observed(obs, frame_w, frame_h)
        else:
            crop = _crop_for_weak(obs, frame_w, frame_h)

        motion_bb = obs.get("associated_motion_bbox")
        entries.append({
            "frame_idx": int(obs["frame_idx"]),
            "timestamp_sec": float(obs["timestamp_sec"]),
            "type": "roi_crop_dense" if dense else "roi_crop",
            "source": "moving_object_track",
            "observation_type": obs_type,
            "motion_observed": obs_type == OBSERVED_MOTION,
            "track_id": track["track_id"],
            "class_name": track["class_name"],
            "motion_bbox": motion_bb,
            "track_bbox": [int(obs["x1"]), int(obs["y1"]), int(obs["x2"]), int(obs["y2"])],
            "crop_bbox": crop,
            "importance": track["track_motion_importance"],
        })
    return entries


def _fallback_entries(
    tube: dict[str, Any],
    observations: list[dict],
    *,
    frame_w: int,
    frame_h: int,
) -> list[dict[str, Any]]:
    if not observations:
        return []
    peak = max(observations, key=lambda o: o.get("motion_density", 0))
    seen: set[int] = set()
    selected = []
    for o in [observations[0], peak, observations[-1]]:
        fi = int(o["frame_idx"])
        if fi not in seen:
            seen.add(fi)
            selected.append(o)

    entries = []
    for obs in selected:
        crop = expand_bbox(obs["x1"], obs["y1"], obs["x2"], obs["y2"], FALLBACK_MARGIN, frame_w, frame_h)
        entries.append({
            "frame_idx": int(obs["frame_idx"]),
            "timestamp_sec": float(obs["timestamp_sec"]),
            "type": "roi_crop",
            "source": "motion_fallback_tube",
            "observation_type": "observed_motion",
            "motion_observed": True,
            "motion_tube_id": tube["motion_tube_id"],
            "motion_bbox": [int(obs["x1"]), int(obs["y1"]), int(obs["x2"]), int(obs["y2"])],
            "crop_bbox": crop,
            "importance": tube.get("tube_importance", 0),
        })
    return entries


def build_track_roi_inputs(
    entries: list[dict[str, Any]],
    *,
    max_rois: int = MAX_ROI_TRACKS,
) -> list[dict[str, Any]]:
    """
    Canonical roi_inputs for v2 ByteTrack plans.

    Root cause: legacy frame parsing ranked all crop_bbox by importance, so
    motion_fallback_tubes (high tube_importance) could displace moving_object_track
    ROIs (e.g. motorcycle track 197) in overlay selection.
    Tracks are always preferred; fallback tubes fill remaining slots only.
    """
    by_key: dict[str, list[dict[str, Any]]] = {}
    meta: dict[str, dict[str, Any]] = {}

    for entry in entries:
        bbox = entry.get("crop_bbox")
        if not bbox:
            continue

        if entry.get("track_id") is not None:
            key = f"track:{int(entry['track_id'])}"
            meta[key] = {
                "priority": 0,
                "importance": float(entry.get("importance", 0.0)),
                "source": "track",
                "source_track_ids": [int(entry["track_id"])],
                "source_tube_ids": [],
            }
        elif entry.get("motion_tube_id") is not None:
            key = f"mtube:{int(entry['motion_tube_id'])}"
            meta[key] = {
                "priority": 1,
                "importance": float(entry.get("importance", 0.0)),
                "source": "mtube",
                "source_track_ids": [],
                "source_tube_ids": [int(entry["motion_tube_id"])],
            }
        else:
            continue

        by_key.setdefault(key, []).append(
            {
                "frame_idx": int(entry["frame_idx"]),
                "timestamp_sec": float(entry["timestamp_sec"]),
                "bbox": [int(v) for v in bbox],
            }
        )

    if not by_key:
        return []

    ranked = sorted(
        by_key.keys(),
        key=lambda k: (meta[k]["priority"], -meta[k]["importance"]),
    )[:max_rois]

    roi_inputs: list[dict[str, Any]] = []
    for roi_id, key in enumerate(ranked, start=1):
        seq = sorted(by_key[key], key=lambda e: e["frame_idx"])
        m = meta[key]
        roi_inputs.append(
            {
                "roi_id": roi_id,
                "source": m["source"],
                "source_track_ids": m["source_track_ids"],
                "source_tube_ids": m["source_tube_ids"],
                "merged_tube_count": len(seq),
                "bbox_sequence": seq,
                "start_sec": seq[0]["timestamp_sec"],
                "end_sec": seq[-1]["timestamp_sec"],
            }
        )
    return roi_inputs


def _dedupe(entries: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out = []
    for e in entries:
        key = (e.get("frame_idx"), e.get("source"), e.get("track_id"), e.get("motion_tube_id"), tuple(e.get("crop_bbox") or []))
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    out.sort(key=lambda x: (x.get("frame_idx", 0), x.get("source", "")))
    return out


def build_track_adaptive_input_plan(
    category: str,
    frame_df: pd.DataFrame,
    moving_tracks: list[dict[str, Any]],
    track_obs_map: dict[int, list[dict]],
    fallback_tubes: list[dict[str, Any]],
    fallback_obs_map: dict[int, list[dict]],
    *,
    sampled_fps: float = 5.0,
    top_k_tracks: int = 5,
    top_k_fallback: int = 3,
    frame_w: int = 1280,
    frame_h: int = 720,
) -> dict[str, Any]:
    sparse = _sparse_global_frames(frame_df, native_fps=sampled_fps)
    for e in sparse:
        e["source"] = "sparse_global"

    entries: list[dict] = list(sparse)
    strategy = [
        "ByteTrack ID continuity for object tracks",
        "motion map primary evidence; detector_only tracks excluded from ROI",
        "motion_fallback_tube for detector misses",
    ]

    sorted_tracks = sorted(moving_tracks, key=lambda t: t["track_motion_importance"], reverse=True)[:top_k_tracks]
    for track in sorted_tracks:
        obs = track_obs_map.get(track["track_id"], [])
        entries.extend(_track_roi_entries(track, obs, frame_w=frame_w, frame_h=frame_h, dense=False))

    if category in ("Dense Local Motion", "Multi-motion / Competing Blobs") and sorted_tracks:
        track = sorted_tracks[0]
        obs = track_obs_map.get(track["track_id"], [])
        entries.extend(_track_roi_entries(track, obs, frame_w=frame_w, frame_h=frame_h, dense=True))

    for tube in sorted(fallback_tubes, key=lambda t: t["tube_importance"], reverse=True)[:top_k_fallback]:
        obs = fallback_obs_map.get(tube["motion_tube_id"], [])
        entries.extend(_fallback_entries(tube, obs, frame_w=frame_w, frame_h=frame_h))

    if category == "Dense Global Motion":
        for e in _high_motion_windows(frame_df):
            e["source"] = "dense_global"
            entries.append(e)

    if category == "Scene Change Dominant":
        entries.extend(_scene_boundary_frames(frame_df))
        entries.extend(_high_motion_windows(frame_df, top_fraction=0.1, window_half=2))

    entries = _dedupe(entries)
    by_source: dict[str, int] = {}
    for e in entries:
        src = e.get("source", "?")
        by_source[src] = by_source.get(src, 0) + 1

    roi_inputs: list[dict[str, Any]] = []

    return {
        "version": "v2-bytetrack-motion-first",
        "category": category,
        "strategy": strategy,
        "sparse_global_fps": SPARSE_GLOBAL_FPS,
        "estimated_total_frame_count": len(entries),
        "estimated_crop_count": sum(1 for e in entries if e.get("crop_bbox")),
        "entries_by_source": by_source,
        "top_moving_tracks": [
            {
                "track_id": t["track_id"],
                "class_name": t["class_name"],
                "track_motion_importance": t["track_motion_importance"],
                "observed_motion_ratio": t.get("observed_motion_ratio", 0),
                "num_observed_motion_frames": t.get("num_observed_motion_frames", 0),
                "normalized_track_displacement": t.get("normalized_track_displacement", 0),
                "track_state": t.get("track_state", ""),
            }
            for t in sorted_tracks
        ],
        "fallback_tubes": [{"motion_tube_id": t["motion_tube_id"]} for t in fallback_tubes[:top_k_fallback]],
        "roi_inputs": roi_inputs,
        "frames": entries,
    }
