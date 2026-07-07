"""Motion fallback tubes vs moving object tracks."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from motion_analyzer.v1.tube_builder import TubeMatchConfig, build_tubes
from motion_analyzer.v1.tube_features import tubes_to_feature_rows
from motion_analyzer.v2.bbox_utils import bbox_iou

logger = logging.getLogger(__name__)


def _tube_bbox(obs_list: list[dict]) -> dict:
    return {
        "x1": min(o["x1"] for o in obs_list),
        "y1": min(o["y1"] for o in obs_list),
        "x2": max(o["x2"] for o in obs_list),
        "y2": max(o["y2"] for o in obs_list),
    }


def _overlaps_moving_track(
    tube_obs: list[dict],
    track_obs_map: dict[int, list[dict]],
    iou_threshold: float = 0.30,
) -> bool:
    if not tube_obs or not track_obs_map:
        return False
    for observations in track_obs_map.values():
        moving_obs = [o for o in observations if o.get("observation_type") != "detector_only"]
        if not moving_obs:
            continue
        for tobs in moving_obs:
            for o in tube_obs:
                if int(o["frame_idx"]) == int(tobs["frame_idx"]) and bbox_iou(o, tobs) >= iou_threshold:
                    return True
        if bbox_iou(_tube_bbox(tube_obs), _tube_bbox(moving_obs)) >= iou_threshold:
            return True
    return False


def select_motion_fallback_tubes_from_tracks(
    blobs_df: pd.DataFrame,
    moving_tracks: list[dict[str, Any]],
    track_obs_map: dict[int, list[dict]],
    *,
    frame_w: int,
    frame_h: int,
    total_sampled_frames: int,
    duration_sec: float,
    top_fraction: float = 0.30,
    min_duration_sec: float = 0.4,
    min_motion_density: float = 0.03,
    min_tube_importance_quantile: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[int, list[dict]]]:
    """Build motion tubes from blobs; keep high-importance tubes not absorbed by tracks."""
    moving_track_ids = {t["track_id"] for t in moving_tracks}
    moving_obs_map = {tid: track_obs_map[tid] for tid in moving_track_ids if tid in track_obs_map}

    segments = build_tubes(
        blobs_df,
        frame_w=frame_w,
        frame_h=frame_h,
        config=TubeMatchConfig(max_gap=6, match_threshold=0.25),
        min_tube_length=2,
    )
    tube_rows = tubes_to_feature_rows(
        segments,
        total_sampled_frames=total_sampled_frames,
        video_duration_sec=duration_sec,
    )
    obs_map = {s.tube_id: s.observations for s in segments}

    if not tube_rows:
        return [], obs_map

    imp_values = sorted([t["tube_importance"] for t in tube_rows], reverse=True)
    cutoff = imp_values[max(0, int(len(imp_values) * min_tube_importance_quantile) - 1)]

    fallback: list[dict[str, Any]] = []
    for tube in tube_rows:
        tid = tube["tube_id"]
        obs = obs_map.get(tid, [])
        if tube["duration_sec"] < min_duration_sec:
            continue
        if tube["mean_motion_density"] < min_motion_density:
            continue
        if tube["tube_importance"] < cutoff:
            continue
        if _overlaps_moving_track(obs, moving_obs_map):
            continue

        row = {k: v for k, v in tube.items() if not k.startswith("_")}
        row["motion_tube_id"] = tid
        row["source"] = "motion_fallback_tube"
        row["fallback_reason"] = "detector_miss_not_overlapping_track"
        fallback.append(row)

    fallback.sort(key=lambda t: t["tube_importance"], reverse=True)
    top_n = max(1, int(len(tube_rows) * top_fraction))
    fallback = fallback[:top_n]
    logger.info("Selected %d motion fallback tubes (track-based)", len(fallback))
    return fallback, obs_map
