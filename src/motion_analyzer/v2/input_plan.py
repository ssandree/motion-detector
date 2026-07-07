"""Adaptive VLM input plan — motion-first with detector-guided context (v2)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from motion_analyzer.v1.input_plan_builder import (
    SPARSE_GLOBAL_FPS,
    _high_motion_windows,
    _scene_boundary_frames,
    _sparse_global_frames,
)
from motion_analyzer.v2.detector_guided_tubes import enrich_tube_observations_with_crop
from motion_analyzer.v2.gap_filling_tube_tracker import OBSERVED_MOTION


def _parse_classes(s: str) -> list[str]:
    if not s:
        return []
    return [c.strip() for c in str(s).split(",") if c.strip()]


def _observation_entry(
    obs: dict,
    tube: dict[str, Any],
    *,
    dense: bool,
) -> dict[str, Any]:
    return {
        "frame_idx": int(obs["frame_idx"]),
        "timestamp_sec": float(obs["timestamp_sec"]),
        "type": "roi_crop_dense" if dense else "roi_crop",
        "source": "detector_guided_motion_tube",
        "observation_type": obs.get("observation_type", OBSERVED_MOTION),
        "motion_observed": bool(obs.get("motion_observed", False)),
        "detector_supported": bool(obs.get("detector_supported", False)),
        "interpolated": bool(obs.get("interpolated", False)),
        "predicted_bbox": bool(obs.get("predicted_bbox", False)),
        "motion_tube_id": tube["motion_tube_id"],
        "associated_classes": _parse_classes(tube.get("associated_classes", "")),
        "motion_bbox": obs.get("motion_bbox"),
        "predicted_bbox": obs.get("predicted_bbox"),
        "context_detector_bbox": obs.get("context_detector_bbox") or None,
        "crop_bbox": obs.get("crop_bbox"),
        "importance": tube["detector_guided_motion_importance"],
    }


def _select_dense_observations(enriched: list[dict]) -> list[dict]:
    """Prefer observed_motion frames; include gap-fill frames to maintain continuity."""
    observed = [o for o in enriched if o.get("observation_type") == OBSERVED_MOTION]
    gap_fill = [o for o in enriched if o.get("observation_type") != OBSERVED_MOTION]

    if not observed:
        return enriched

    selected = list(observed)
    obs_frames = {int(o["frame_idx"]) for o in observed}

    for gf in gap_fill:
        fi = int(gf["frame_idx"])
        near = any(abs(fi - of) <= 6 for of in obs_frames)
        if near and gf.get("observation_type") == "detector_supported":
            selected.append(gf)

    selected.sort(key=lambda o: int(o["frame_idx"]))
    return selected


def _guided_tube_roi_entries(
    tube: dict[str, Any],
    observations: list[dict],
    detections_by_frame: dict[int, list[dict]],
    *,
    frame_w: int,
    frame_h: int,
    dense: bool = False,
) -> list[dict[str, Any]]:
    if not observations:
        return []

    if tube.get("num_observed_motion_frames", tube.get("num_moving_frames", 0)) == 0:
        return []

    enriched = enrich_tube_observations_with_crop(
        observations,
        detections_by_frame,
        frame_w=frame_w,
        frame_h=frame_h,
    )

    if dense:
        selected = _select_dense_observations(enriched)
    else:
        observed = [o for o in enriched if o.get("observation_type") == OBSERVED_MOTION]
        pool = observed if observed else enriched
        peak = max(pool, key=lambda o: o.get("motion_density", 0))
        selected = [pool[0], peak, pool[-1]]
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
        if obs.get("observation_type") == OBSERVED_MOTION and float(obs.get("motion_density", 0)) < 0.01:
            continue
        entries.append(_observation_entry(obs, tube, dense=dense))
    return entries


def _fallback_roi_entries(
    tube: dict[str, Any],
    observations: list[dict],
    detections_by_frame: dict[int, list[dict]],
    *,
    frame_w: int,
    frame_h: int,
) -> list[dict[str, Any]]:
    if not observations:
        return []

    enriched = enrich_tube_observations_with_crop(
        observations,
        detections_by_frame,
        frame_w=frame_w,
        frame_h=frame_h,
    )
    observed = [o for o in enriched if o.get("observation_type") == OBSERVED_MOTION] or enriched
    peak = max(observed, key=lambda o: o.get("motion_density", 0))
    seen: set[int] = set()
    selected = []
    for o in [observed[0], peak, observed[-1]]:
        fi = int(o["frame_idx"])
        if fi not in seen:
            seen.add(fi)
            selected.append(o)

    entries = []
    for obs in selected:
        entries.append({
            "frame_idx": int(obs["frame_idx"]),
            "timestamp_sec": float(obs["timestamp_sec"]),
            "type": "roi_crop",
            "source": "motion_fallback_tube",
            "observation_type": obs.get("observation_type", OBSERVED_MOTION),
            "motion_observed": bool(obs.get("motion_observed", True)),
            "motion_tube_id": tube["motion_tube_id"],
            "associated_classes": _parse_classes(tube.get("associated_classes", "")),
            "motion_bbox": obs.get("motion_bbox"),
            "predicted_bbox": obs.get("predicted_bbox"),
            "context_detector_bbox": None,
            "crop_bbox": obs.get("crop_bbox"),
            "importance": tube.get("detector_guided_motion_importance", 0),
        })
    return entries


def _dedupe_v2_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple] = set()
    result = []
    for e in entries:
        key = (
            e.get("frame_idx"),
            e.get("type"),
            e.get("source"),
            e.get("motion_tube_id"),
            e.get("observation_type"),
            tuple(e.get("crop_bbox") or []),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(e)
    result.sort(key=lambda x: (x.get("frame_idx", 0), x.get("type", "")))
    return result


def build_adaptive_input_plan_v2(
    category: str,
    frame_df: pd.DataFrame,
    guided_tubes: list[dict[str, Any]],
    guided_obs_map: dict[int, list[dict]],
    fallback_tubes: list[dict[str, Any]],
    fallback_obs_map: dict[int, list[dict]],
    detections_by_frame: dict[int, list[dict]],
    *,
    sampled_fps: float = 5.0,
    top_k_guided: int = 5,
    top_k_fallback: int = 3,
    frame_w: int = 1280,
    frame_h: int = 720,
) -> dict[str, Any]:
    """Motion-first adaptive plan; ROI crops always originate from motion tubes."""
    sparse = _sparse_global_frames(frame_df, native_fps=sampled_fps)
    for e in sparse:
        e["source"] = "sparse_global"

    entries: list[dict[str, Any]] = list(sparse)
    strategy_notes = [
        "motion-first ROI from detector_guided_motion_tubes with gap filling",
        "observed_motion frames prioritized; detector_supported fills continuity",
        "detector box never used as sole ROI",
    ]

    moving_tubes = [t for t in guided_tubes if t.get("num_observed_motion_frames", 0) > 0]
    sorted_guided = sorted(
        moving_tubes,
        key=lambda t: t["detector_guided_motion_importance"],
        reverse=True,
    )
    top_guided = sorted_guided[:top_k_guided]

    for tube in top_guided:
        obs = guided_obs_map.get(tube["motion_tube_id"], [])
        entries.extend(_guided_tube_roi_entries(
            tube, obs, detections_by_frame,
            frame_w=frame_w, frame_h=frame_h, dense=False,
        ))

    if category in ("Dense Local Motion", "Multi-motion / Competing Blobs") and top_guided:
        tube = top_guided[0]
        obs = guided_obs_map.get(tube["motion_tube_id"], [])
        entries.extend(_guided_tube_roi_entries(
            tube, obs, detections_by_frame,
            frame_w=frame_w, frame_h=frame_h, dense=True,
        ))

    sorted_fallback = sorted(
        fallback_tubes,
        key=lambda t: t.get("detector_guided_motion_importance", 0),
        reverse=True,
    )[:top_k_fallback]
    for tube in sorted_fallback:
        obs = fallback_obs_map.get(tube["motion_tube_id"], [])
        entries.extend(_fallback_roi_entries(
            tube, obs, detections_by_frame,
            frame_w=frame_w, frame_h=frame_h,
        ))

    if category == "Dense Global Motion":
        dense_global = _high_motion_windows(frame_df)
        for e in dense_global:
            e["source"] = "dense_global"
        entries.extend(dense_global)

    if category == "Scene Change Dominant":
        entries.extend(_scene_boundary_frames(frame_df))
        entries.extend(_high_motion_windows(frame_df, top_fraction=0.1, window_half=2))

    entries = _dedupe_v2_entries(entries)

    global_count = sum(1 for e in entries if e.get("crop_bbox") is None)
    crop_count = sum(1 for e in entries if e.get("crop_bbox") is not None)
    by_source: dict[str, int] = {}
    by_obs_type: dict[str, int] = {}
    for e in entries:
        src = e.get("source", "unknown")
        by_source[src] = by_source.get(src, 0) + 1
        ot = e.get("observation_type", "n/a")
        if e.get("crop_bbox"):
            by_obs_type[ot] = by_obs_type.get(ot, 0) + 1

    guided_summaries = [
        {
            "motion_tube_id": t["motion_tube_id"],
            "associated_classes": t.get("associated_classes", ""),
            "detector_guided_motion_importance": t["detector_guided_motion_importance"],
            "mean_motion_density": t["mean_motion_density"],
            "motion_persistence": t.get("motion_persistence", 0),
            "track_continuity": t.get("track_continuity", 0),
            "num_observed_motion_frames": t.get("num_observed_motion_frames", 0),
        }
        for t in top_guided
    ]

    clean_entries = [{k: v for k, v in e.items() if not k.startswith("_")} for e in entries]

    return {
        "version": "v2-motion-first-gap-fill",
        "category": category,
        "strategy": strategy_notes,
        "sparse_global_fps": SPARSE_GLOBAL_FPS,
        "estimated_global_frame_count": global_count,
        "estimated_crop_count": crop_count,
        "estimated_total_frame_count": len(clean_entries),
        "entries_by_source": by_source,
        "entries_by_observation_type": by_obs_type,
        "top_detector_guided_tubes": guided_summaries,
        "fallback_tubes": [
            {"motion_tube_id": t["motion_tube_id"], "source": "motion_fallback_tube"}
            for t in sorted_fallback
        ],
        "frames": clean_entries,
    }
