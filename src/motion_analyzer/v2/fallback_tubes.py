"""Motion-only fallback tubes for detector misses (motion-first pipeline)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from motion_analyzer.v2.bbox_utils import bbox_iou

logger = logging.getLogger(__name__)


def _tube_bbox_from_obs(obs: list[dict]) -> dict[str, float]:
    x1 = min(o["x1"] for o in obs)
    y1 = min(o["y1"] for o in obs)
    x2 = max(o["x2"] for o in obs)
    y2 = max(o["y2"] for o in obs)
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _spatial_overlap_tubes(
    tube_a_obs: list[dict],
    tube_b_obs: list[dict],
    iou_threshold: float = 0.35,
) -> bool:
    box_a = _tube_bbox_from_obs(tube_a_obs)
    frames_b = {int(o["frame_idx"]): o for o in tube_b_obs}
    for oa in tube_a_obs:
        fi = int(oa["frame_idx"])
        if fi in frames_b and bbox_iou(oa, frames_b[fi]) >= iou_threshold:
            return True
    return bbox_iou(box_a, _tube_bbox_from_obs(tube_b_obs)) >= iou_threshold


def select_motion_fallback_tubes(
    guided_tubes: list[dict[str, Any]],
    guided_obs_map: dict[int, list[dict]],
    *,
    top_guided_fraction: float = 0.35,
    min_duration_sec: float = 0.4,
    min_motion_density: float = 0.03,
    max_detector_context: float = 0.35,
    texture_threshold: float = 0.60,
    global_motion_threshold: float = 0.58,
    min_spatial_novelty: float = 0.12,
) -> list[dict[str, Any]]:
    """
    Select fallback tubes: strong motion with weak/no detector context.

    High-importance detector-guided tubes are excluded from fallback.
    """
    if not guided_tubes:
        return []

    sorted_by_imp = sorted(
        guided_tubes,
        key=lambda t: t["detector_guided_motion_importance"],
        reverse=True,
    )
    cutoff_idx = max(0, int(len(sorted_by_imp) * top_guided_fraction))
    top_ids = {t["motion_tube_id"] for t in sorted_by_imp[:cutoff_idx]}

    fallback: list[dict[str, Any]] = []
    for tube in guided_tubes:
        tid = tube["motion_tube_id"]
        if tid in top_ids:
            continue
        if tube.get("mean_detector_context_score", 0) > max_detector_context:
            continue
        if tube["duration_sec"] < min_duration_sec:
            continue
        if tube["mean_motion_density"] < min_motion_density:
            continue
        if tube.get("spatial_novelty_mean", 0) < min_spatial_novelty:
            continue

        obs = guided_obs_map.get(tid, [])
        if not obs:
            continue

        tex = float(sum(o.get("texture_flicker_score", 0) for o in obs) / len(obs))
        gm = float(sum(o.get("global_motion_score", 0) for o in obs) / len(obs))
        if tex > texture_threshold or gm > global_motion_threshold:
            continue

        absorbed = False
        for top_tube in sorted_by_imp[:cutoff_idx]:
            top_obs = guided_obs_map.get(top_tube["motion_tube_id"], [])
            if top_obs and _spatial_overlap_tubes(obs, top_obs):
                absorbed = True
                break
        if absorbed:
            continue

        row = {k: v for k, v in tube.items() if not k.startswith("_")}
        row["source"] = "motion_fallback_tube"
        row["fallback_reason"] = "detector_miss_strong_motion"
        fallback.append(row)

    fallback.sort(key=lambda t: t["detector_guided_motion_importance"], reverse=True)
    logger.info("Selected %d motion fallback tubes", len(fallback))
    return fallback
