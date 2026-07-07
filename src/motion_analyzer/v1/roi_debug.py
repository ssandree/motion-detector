"""Debug timeline for ROI track / overlay consistency."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from motion_analyzer.v1.input_plan_visualize import roi_bbox_at_frame, roi_inputs_from_plan


def _slim_blob(blob: dict[str, Any]) -> dict[str, Any]:
    return {
        "blob_id_in_frame": int(blob.get("blob_id_in_frame", -1)),
        "bbox": [int(blob["x1"]), int(blob["y1"]), int(blob["x2"]), int(blob["y2"])],
        "blob_importance": round(float(blob.get("blob_importance", 0.0)), 4),
        "flow_direction_coherence": round(float(blob.get("flow_direction_coherence", 0.0)), 4),
    }


def build_debug_roi_timeline(
    plan: dict[str, Any],
    frame_df: pd.DataFrame,
    blobs_df: pd.DataFrame,
    *,
    flow_group_map: dict[int, dict[str, Any]] | None = None,
    roi_source: str = "adaptive_input_plan.json",
    event_roi_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Per-sampled-frame debug record for overlay / ROI consistency checks."""
    roi_tracks = roi_inputs_from_plan(plan)
    score_by_group: dict[int, dict[str, Any]] = {}
    if event_roi_debug:
        for block in (
            event_roi_debug.get("candidates", [])
            + event_roi_debug.get("selected_rois", [])
            + event_roi_debug.get("background_flow_regions", [])
        ):
            gid = block.get("group_id")
            if gid is not None:
                score_by_group[int(gid)] = block
    blobs_by_frame: dict[int, list[dict[str, Any]]] = {}
    if not blobs_df.empty:
        for fi in blobs_df["frame_idx"].unique():
            blobs_by_frame[int(fi)] = blobs_df[blobs_df["frame_idx"] == fi].to_dict("records")

    timeline: list[dict[str, Any]] = []
    for _, row in frame_df.iterrows():
        fidx = int(row["frame_idx"])
        ts = float(row["timestamp_sec"])
        raw = blobs_by_frame.get(fidx, [])
        fg = (flow_group_map or {}).get(fidx, {})
        selected = []
        for roi in roi_tracks:
            bbox = roi_bbox_at_frame(roi, fidx)
            if bbox:
                gid = roi.get("group_id")
                entry = {"roi_id": int(roi["roi_id"]), "bbox": bbox, "group_id": gid}
                if gid is not None and int(gid) in score_by_group:
                    sc = score_by_group[int(gid)]
                    entry["event_score"] = sc.get("event_score")
                    entry["interaction_score"] = sc.get("interaction_score")
                selected.append(entry)
        timeline.append(
            {
                "frame_idx": fidx,
                "timestamp_sec": round(ts, 4),
                "raw_blobs": [_slim_blob(b) for b in raw],
                "grouped_blobs": fg.get("groups", []),
                "selected_roi_tracks": selected,
                "has_final_roi": bool(selected),
            }
        )

    return {
        "roi_source": roi_source,
        "motion_type": plan.get("motion_type"),
        "num_roi_tracks": len(roi_tracks),
        "background_flow_region_count": len(event_roi_debug.get("background_flow_regions", []))
        if event_roi_debug
        else 0,
        "selected_roi_tracks_summary": [
            {
                "roi_id": r.get("roi_id"),
                "group_id": r.get("group_id"),
                "source": r.get("source"),
                "start_sec": r.get("start_sec"),
                "end_sec": r.get("end_sec"),
                "num_keyframes": len(r.get("bbox_sequence", [])),
                "event_score": r.get("event_score"),
                "interaction_score": r.get("interaction_score"),
                "motion_score": r.get("motion_score"),
                "object_proximity_score": r.get("object_proximity_score"),
                "background_flow_penalty": r.get("background_flow_penalty"),
            }
            for r in roi_tracks
        ],
        "timeline": timeline,
    }


def save_debug_roi_timeline(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
