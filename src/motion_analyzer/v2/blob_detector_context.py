"""Attach detector context features to motion blobs (motion-first, detector as auxiliary)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from motion_analyzer.v2.bbox_utils import (
    bbox_from_dict,
    bbox_iou,
    center_distance_norm,
    center_inside,
    union_bbox,
)

logger = logging.getLogger(__name__)

PERSON_CLASS = "person"
VEHICLE_CLASSES = {"car", "bicycle", "motorcycle", "bus", "truck"}
PERSON_VEHICLE_CLASSES = {PERSON_CLASS} | VEHICLE_CLASSES
BAG_CLASSES = {"backpack", "handbag", "suitcase"}
STATIC_CLASSES = {"chair", "bench", "umbrella", "potted plant", "dining table"}
STRONG_MOTION_DENSITY = 0.08


@dataclass
class BlobDetectorContextConfig:
    iou_threshold: float = 0.05
    distance_ratio: float = 0.05
    person_vehicle_distance_ratio: float = 0.07
    person_vehicle_iou_threshold: float = 0.03


def _is_associated(
    blob: dict,
    det: dict,
    frame_w: int,
    frame_h: int,
    cfg: BlobDetectorContextConfig,
) -> bool:
    iou = bbox_iou(blob, det)
    cls = det.get("class_name", "")
    iou_thr = cfg.person_vehicle_iou_threshold if cls in PERSON_VEHICLE_CLASSES else cfg.iou_threshold
    if iou > iou_thr:
        return True
    if center_inside(blob, det):
        return True
    dist_norm = center_distance_norm(blob, det, frame_w, frame_h)
    dist_thr = (
        cfg.person_vehicle_distance_ratio
        if cls in PERSON_VEHICLE_CLASSES
        else cfg.distance_ratio
    )
    return dist_norm < dist_thr


def _detector_context_score(
    associated_classes: list[str],
    motion_density: float,
) -> float:
    if not associated_classes:
        if motion_density >= STRONG_MOTION_DENSITY:
            return 0.3
        return 0.0

    has_static_only = all(c in STATIC_CLASSES for c in associated_classes)
    has_target = any(c in PERSON_VEHICLE_CLASSES | BAG_CLASSES for c in associated_classes)

    if has_static_only and not has_target:
        return -0.5

    score = 0.0
    if PERSON_CLASS in associated_classes:
        score = max(score, 1.0)
    if any(c in VEHICLE_CLASSES for c in associated_classes):
        score = max(score, 0.8)
    if any(c in {"bicycle", "motorcycle"} for c in associated_classes):
        score = max(score, 0.8)
    if any(c in BAG_CLASSES for c in associated_classes):
        score = max(score, 0.6)

    if score == 0.0 and motion_density >= STRONG_MOTION_DENSITY:
        return 0.3
    return score


def annotate_blobs_with_detector_context(
    blobs_df: pd.DataFrame,
    detections_df: pd.DataFrame,
    *,
    frame_w: int,
    frame_h: int,
    config: BlobDetectorContextConfig | None = None,
) -> pd.DataFrame:
    """
    Add detector context columns to each motion blob without replacing blob geometry.
    """
    cfg = config or BlobDetectorContextConfig()
    rows: list[dict[str, Any]] = []

    dets_by_frame: dict[int, list[dict]] = {}
    if not detections_df.empty:
        for fi in detections_df["frame_idx"].unique():
            dets_by_frame[int(fi)] = (
                detections_df[detections_df["frame_idx"] == fi].to_dict("records")
            )

    for _, blob in blobs_df.iterrows():
        rec = blob.to_dict()
        frame_idx = int(rec["frame_idx"])
        frame_dets = dets_by_frame.get(frame_idx, [])

        matched: list[tuple[dict, float, float]] = []
        for det in frame_dets:
            if not _is_associated(rec, det, frame_w, frame_h, cfg):
                continue
            iou = bbox_iou(rec, det)
            dist = center_distance_norm(rec, det, frame_w, frame_h)
            matched.append((det, iou, dist))

        if matched:
            det_ids = [str(int(d["det_id_in_frame"])) for d, _, _ in matched]
            classes = list({d["class_name"] for d, _, _ in matched})
            max_iou = max(iou for _, iou, _ in matched)
            nearest = min(matched, key=lambda x: x[2])
            nearest_class = nearest[0]["class_name"]
            nearest_dist = nearest[2]
            inside_pv = any(
                center_inside(rec, d) and d["class_name"] in PERSON_VEHICLE_CLASSES
                for d, _, _ in matched
            )
        else:
            det_ids = []
            classes = []
            max_iou = 0.0
            nearest_class = ""
            nearest_dist = 1.0
            inside_pv = False

        motion_density = float(rec.get("motion_density", 0.0))
        ctx_score = _detector_context_score(classes, motion_density)

        rec["associated_det_ids"] = ",".join(det_ids)
        rec["associated_classes"] = ",".join(sorted(classes))
        rec["max_iou_with_detector"] = round(max_iou, 6)
        rec["nearest_detector_class"] = nearest_class
        rec["nearest_detector_distance"] = round(nearest_dist, 6)
        rec["inside_person_or_vehicle_box"] = inside_pv
        rec["detector_context_score"] = round(ctx_score, 4)
        rows.append(rec)

    return pd.DataFrame(rows)


def merge_person_body_blobs(
    blobs_ctx_df: pd.DataFrame,
    detections_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge motion blobs inside the same person detector box into merged_motion_regions.

    Merged bbox is union of motion blobs (not detector box).
    """
    if blobs_ctx_df.empty or detections_df.empty:
        return pd.DataFrame()

    person_dets = detections_df[detections_df["class_name"] == PERSON_CLASS]
    if person_dets.empty:
        return pd.DataFrame()

    merged_rows: list[dict[str, Any]] = []
    region_id = 0

    for frame_idx in sorted(blobs_ctx_df["frame_idx"].unique()):
        frame_blobs = blobs_ctx_df[blobs_ctx_df["frame_idx"] == frame_idx].to_dict("records")
        frame_persons = person_dets[person_dets["frame_idx"] == frame_idx].to_dict("records")

        assigned: set[int] = set()
        for pdet in frame_persons:
            group: list[dict] = []
            for blob in frame_blobs:
                bid = int(blob["blob_id_in_frame"])
                if bid in assigned:
                    continue
                if center_inside(blob, pdet) or bbox_iou(blob, pdet) > 0.03:
                    group.append(blob)
                    assigned.add(bid)

            if len(group) < 2:
                continue

            ubox = union_bbox(group)
            classes = set()
            for b in group:
                if b.get("associated_classes"):
                    classes.update(b["associated_classes"].split(","))

            merged_rows.append({
                "merged_region_id": region_id,
                "frame_idx": int(frame_idx),
                "timestamp_sec": float(group[0]["timestamp_sec"]),
                "person_det_id": int(pdet["det_id_in_frame"]),
                "num_merged_blobs": len(group),
                "merged_blob_ids": ",".join(str(int(b["blob_id_in_frame"])) for b in group),
                "x1": int(ubox["x1"]),
                "y1": int(ubox["y1"]),
                "x2": int(ubox["x2"]),
                "y2": int(ubox["y2"]),
                "center_x": round(ubox["center_x"], 2),
                "center_y": round(ubox["center_y"], 2),
                "person_det_x1": int(pdet["x1"]),
                "person_det_y1": int(pdet["y1"]),
                "person_det_x2": int(pdet["x2"]),
                "person_det_y2": int(pdet["y2"]),
                "mean_motion_density": round(float(np.mean([b["motion_density"] for b in group])), 6),
                "mean_blob_importance": round(float(np.mean([b["blob_importance"] for b in group])), 6),
                "associated_classes": ",".join(sorted(c for c in classes if c)),
            })
            region_id += 1

    return pd.DataFrame(merged_rows)
