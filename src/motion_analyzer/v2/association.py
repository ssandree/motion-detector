"""Motion blob ↔ object detection association and region features."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from motion_analyzer.v2.bbox_utils import (
    bbox_area,
    bbox_from_dict,
    bbox_iou,
    center_distance_norm,
    center_inside,
    min_max_normalize,
    union_bbox,
)
from motion_analyzer.v2.object_detector import CLASS_WEIGHTS

logger = logging.getLogger(__name__)

EPS = 1e-8


@dataclass
class AssociationConfig:
    iou_weight: float = 0.40
    center_inside_weight: float = 0.25
    center_dist_weight: float = 0.20
    proximity_weight: float = 0.15
    match_threshold: float = 0.30
    person_proximity_margin: float = 0.15


def _proximity_score(blob: dict, det: dict, frame_w: int, frame_h: int) -> float:
    """Score when blob bbox is near det box (expanded)."""
    dx1, dy1, dx2, dy2 = bbox_from_dict(det)
    w = dx2 - dx1
    h = dy2 - dy1
    margin = 0.15 * max(w, h)
    bx1, by1, bx2, by2 = bbox_from_dict(blob)
    ex1, ey1, ex2, ey2 = dx1 - margin, dy1 - margin, dx2 + margin, dy2 + margin
    ix1 = max(bx1, ex1)
    iy1 = max(by1, ey1)
    ix2 = min(bx2, ex2)
    iy2 = min(by2, ey2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    blob_a = bbox_area(bx1, by1, bx2, by2)
    return min(1.0, inter / max(blob_a, EPS))


def association_score(
    blob: dict,
    det: dict,
    frame_w: int,
    frame_h: int,
    config: AssociationConfig,
) -> float:
    iou = bbox_iou(blob, det)
    inside = 1.0 if center_inside(blob, det) else 0.0
    dist = center_distance_norm(blob, det, frame_w, frame_h)
    dist_sim = 1.0 - min(1.0, dist / 0.20)
    prox = _proximity_score(blob, det, frame_w, frame_h)
    return (
        config.iou_weight * iou
        + config.center_inside_weight * inside
        + config.center_dist_weight * dist_sim
        + config.proximity_weight * prox
    )


def _class_weight(class_name: str) -> float:
    return CLASS_WEIGHTS.get(class_name, 0.5)


OBJECT_MOTION_SCORE_WEIGHTS = {
    "object_motion_density": 0.25,
    "object_motion_coverage": 0.20,
    "max_associated_blob_importance": 0.15,
    "detector_confidence": 0.15,
    "scale_score": 0.15,
    "class_weight": 0.10,
}


def _compute_region_features(
    det: dict,
    blobs: list[dict],
    frame_w: int,
    frame_h: int,
    region_id: int,
) -> dict[str, Any]:
    det_area = bbox_area(det["x1"], det["y1"], det["x2"], det["y2"])
    if not blobs:
        return {
            "region_id_in_frame": region_id,
            "video_name": det.get("video_name", ""),
            "sampled_idx": det["sampled_idx"],
            "frame_idx": det["frame_idx"],
            "timestamp_sec": det["timestamp_sec"],
            "det_id_in_frame": det["det_id_in_frame"],
            "class_name": det["class_name"],
            "class_id": det["class_id"],
            "x1": det["x1"], "y1": det["y1"], "x2": det["x2"], "y2": det["y2"],
            "center_x": det["center_x"], "center_y": det["center_y"],
            "bbox_area_ratio": det["bbox_area_ratio"],
            "object_motion_energy": 0.0,
            "object_motion_density": 0.0,
            "object_motion_coverage": 0.0,
            "num_associated_motion_blobs": 0,
            "max_associated_blob_importance": 0.0,
            "mean_associated_blob_importance": 0.0,
            "associated_blob_ids": "",
            "motion_inside_ratio": 0.0,
            "detector_confidence": det["confidence"],
            "class_weight": _class_weight(det["class_name"]),
            "scale_score": 1.0 / float(np.sqrt(det["bbox_area_ratio"] + EPS)),
            "object_motion_score": 0.0,
            "is_target_class": det["is_target_class"],
        }

    energies = [float(b.get("motion_energy", 0)) for b in blobs]
    densities = [float(b.get("motion_density", 0)) for b in blobs]
    importances = [float(b.get("blob_importance", 0)) for b in blobs]
    blob_areas = [
        bbox_area(b["x1"], b["y1"], b["x2"], b["y2"]) for b in blobs
    ]
    motion_inside = sum(blob_areas)
    coverage = motion_inside / max(det_area, EPS)

    ubox = union_bbox(blobs)
    blob_ids = ",".join(str(int(b.get("blob_id_in_frame", i))) for i, b in enumerate(blobs))

    return {
        "region_id_in_frame": region_id,
        "video_name": det.get("video_name", blobs[0].get("video_name", "")),
        "sampled_idx": det["sampled_idx"],
        "frame_idx": det["frame_idx"],
        "timestamp_sec": det["timestamp_sec"],
        "det_id_in_frame": det["det_id_in_frame"],
        "class_name": det["class_name"],
        "class_id": det["class_id"],
        "x1": int(det["x1"]), "y1": int(det["y1"]),
        "x2": int(det["x2"]), "y2": int(det["y2"]),
        "center_x": det["center_x"], "center_y": det["center_y"],
        "bbox_area_ratio": det["bbox_area_ratio"],
        "motion_union_x1": int(ubox["x1"]),
        "motion_union_y1": int(ubox["y1"]),
        "motion_union_x2": int(ubox["x2"]),
        "motion_union_y2": int(ubox["y2"]),
        "object_motion_energy": round(sum(energies), 4),
        "object_motion_density": round(float(np.mean(densities)), 6),
        "object_motion_coverage": round(float(coverage), 6),
        "num_associated_motion_blobs": len(blobs),
        "max_associated_blob_importance": round(max(importances), 6),
        "mean_associated_blob_importance": round(float(np.mean(importances)), 6),
        "associated_blob_ids": blob_ids,
        "motion_inside_ratio": round(float(coverage), 6),
        "detector_confidence": det["confidence"],
        "class_weight": _class_weight(det["class_name"]),
        "scale_score": round(1.0 / float(np.sqrt(det["bbox_area_ratio"] + EPS)), 6),
        "object_motion_score": 0.0,
        "is_target_class": det["is_target_class"],
    }


def add_object_motion_scores(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not regions:
        return regions
    keys = list(OBJECT_MOTION_SCORE_WEIGHTS.keys())
    norm = {k: min_max_normalize([r.get(k, 0.0) for r in regions]) for k in keys}
    for i, r in enumerate(regions):
        score = sum(OBJECT_MOTION_SCORE_WEIGHTS[k] * norm[k][i] for k in keys)
        r["object_motion_score"] = round(score, 6)
    return regions


def associate_motion_objects(
    blobs_df: pd.DataFrame,
    detections_df: pd.DataFrame,
    *,
    frame_w: int,
    frame_h: int,
    config: AssociationConfig | None = None,
    target_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Associate blobs with detections per frame.

    Returns:
        object_motion_regions_df: per-detection merged regions (target classes)
        blob_association_df: per-blob association metadata
    """
    cfg = config or AssociationConfig()
    regions: list[dict[str, Any]] = []
    blob_assoc_rows: list[dict[str, Any]] = []

    if detections_df.empty:
        logger.warning("No detections — all blobs remain unassociated")
        for _, blob in blobs_df.iterrows():
            blob_assoc_rows.append({
                **blob.to_dict(),
                "associated_det_id": -1,
                "associated_class": "",
                "association_score": 0.0,
                "motion_only_candidate": True,
            })
        return pd.DataFrame(), pd.DataFrame(blob_assoc_rows)

    dets = detections_df.copy()
    if target_only:
        dets = dets[dets["is_target_class"]]

    for frame_idx in sorted(blobs_df["frame_idx"].unique()):
        frame_blobs = blobs_df[blobs_df["frame_idx"] == frame_idx].to_dict("records")
        frame_dets = dets[dets["frame_idx"] == frame_idx].to_dict("records")

        assigned_blobs: set[int] = set()
        region_id = 0

        for det in frame_dets:
            matched: list[tuple[dict, float]] = []
            for blob in frame_blobs:
                bid = int(blob.get("blob_id_in_frame", id(blob)))
                score = association_score(blob, det, frame_w, frame_h, cfg)
                threshold = cfg.match_threshold
                if det["class_name"] == "person":
                    threshold *= 0.85
                if score >= threshold:
                    matched.append((blob, score))

            matched.sort(key=lambda x: x[1], reverse=True)
            matched_blobs = [b for b, _ in matched]

            if matched_blobs:
                region = _compute_region_features(det, matched_blobs, frame_w, frame_h, region_id)
                regions.append(region)
                region_id += 1
                for blob, score in matched:
                    bid = int(blob.get("blob_id_in_frame", 0))
                    assigned_blobs.add(bid)
                    blob_assoc_rows.append({
                        **blob,
                        "associated_det_id": det["det_id_in_frame"],
                        "associated_class": det["class_name"],
                        "association_score": round(score, 4),
                        "motion_only_candidate": False,
                        "region_id_in_frame": region["region_id_in_frame"],
                    })

        for blob in frame_blobs:
            bid = int(blob.get("blob_id_in_frame", 0))
            if bid not in assigned_blobs:
                blob_assoc_rows.append({
                    **blob,
                    "associated_det_id": -1,
                    "associated_class": "",
                    "association_score": 0.0,
                    "motion_only_candidate": True,
                    "region_id_in_frame": -1,
                })

    regions = add_object_motion_scores(regions)
    return pd.DataFrame(regions), pd.DataFrame(blob_assoc_rows)
