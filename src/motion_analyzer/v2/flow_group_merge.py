"""Group spatially co-located blobs with consistent flow (wind / foliage patterns)."""

from __future__ import annotations

from typing import Any

import numpy as np

MIN_COHERENCE = 0.40
MERGE_CENTER_DIST = 0.12
MAX_BLOB_AREA_RATIO = 0.20


def _blob_center_norm(blob: dict[str, Any], frame_w: int, frame_h: int) -> tuple[float, float]:
    cx = (float(blob["x1"]) + float(blob["x2"])) / 2.0 / max(frame_w, 1)
    cy = (float(blob["y1"]) + float(blob["y2"])) / 2.0 / max(frame_h, 1)
    return cx, cy


def group_frame_blobs_by_flow(
    blobs: list[dict[str, Any]],
    *,
    frame_w: int,
    frame_h: int,
    min_coherence: float = MIN_COHERENCE,
    merge_center_dist: float = MERGE_CENTER_DIST,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Cluster nearby blobs with similar flow coherence (uniform directional texture).

    Returns (ungrouped_individual_blobs, merged_groups).
    """
    if not blobs:
        return [], []

    candidates = [
        b for b in blobs
        if float(b.get("bbox_area_ratio", 0.0)) <= MAX_BLOB_AREA_RATIO
        and float(b.get("flow_direction_coherence", 0.0)) >= min_coherence
        and float(b.get("blob_importance", 0.0)) >= 0.08
    ]
    if len(candidates) < 2:
        return list(blobs), []

    used: set[int] = set()
    groups: list[dict[str, Any]] = []
    group_id = 0
    ranked = sorted(candidates, key=lambda b: b.get("blob_importance", 0), reverse=True)

    for seed in ranked:
        sid = int(seed.get("blob_id_in_frame", id(seed)))
        if sid in used:
            continue
        seed_cx, seed_cy = _blob_center_norm(seed, frame_w, frame_h)
        seed_coh = float(seed.get("flow_direction_coherence", 0.0))
        members = [seed]
        used.add(sid)

        for other in ranked:
            oid = int(other.get("blob_id_in_frame", id(other)))
            if oid in used:
                continue
            ocx, ocy = _blob_center_norm(other, frame_w, frame_h)
            dist = float(np.hypot(ocx - seed_cx, ocy - seed_cy))
            if dist > merge_center_dist:
                continue
            o_coh = float(other.get("flow_direction_coherence", 0.0))
            if abs(o_coh - seed_coh) > 0.35:
                continue
            members.append(other)
            used.add(oid)

        if len(members) < 2:
            continue

        group_id += 1
        x1 = min(int(m["x1"]) for m in members)
        y1 = min(int(m["y1"]) for m in members)
        x2 = max(int(m["x2"]) for m in members)
        y2 = max(int(m["y2"]) for m in members)
        groups.append({
            "group_id": group_id,
            "member_count": len(members),
            "mean_coherence": round(
                float(np.mean([float(m.get("flow_direction_coherence", 0.0)) for m in members])), 4
            ),
            "bbox": [x1, y1, x2, y2],
            "member_blob_ids": [int(m.get("blob_id_in_frame", -1)) for m in members],
        })

    grouped_ids = used
    individuals = [
        b for b in blobs
        if int(b.get("blob_id_in_frame", -1)) not in grouped_ids
    ]
    return individuals, groups


def build_per_frame_flow_groups(
    blobs_df,
    *,
    frame_w: int,
    frame_h: int,
) -> dict[int, dict[str, Any]]:
    """Build per-frame individual blobs and merged flow groups."""
    out: dict[int, dict[str, Any]] = {}
    if blobs_df is None or blobs_df.empty:
        return out
    for fi in blobs_df["frame_idx"].unique():
        frame_blobs = blobs_df[blobs_df["frame_idx"] == fi].to_dict("records")
        ind, groups = group_frame_blobs_by_flow(
            frame_blobs, frame_w=frame_w, frame_h=frame_h
        )
        out[int(fi)] = {"individual": ind, "groups": groups}
    return out
