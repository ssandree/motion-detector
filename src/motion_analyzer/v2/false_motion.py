"""False motion suppression using detector classes and blob motion cues."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from motion_analyzer.v2.bbox_utils import bbox_iou

logger = logging.getLogger(__name__)

DEFAULT_TEXTURE_FLICKER_HIGH = 0.55
DEFAULT_GLOBAL_MOTION_HIGH = 0.45
DEFAULT_EDGE_DENSITY_HIGH = 0.12
DEFAULT_CENTROID_MOVEMENT_LOW = 3.0
DEFAULT_STATIC_OVERLAP_IOU = 0.25
DEFAULT_TREE_FLICKER = 0.50


def _blob_get(blob: dict, key: str, default: float = 0.0) -> float:
    val = blob.get(key, default)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        if default != 0.0:
            return default
        logger.debug("Missing blob feature %s, using 0", key)
        return 0.0
    return float(val)


def _static_overlap_score(blob: dict, static_dets: list[dict]) -> float:
    if not static_dets:
        return 0.0
    return max(bbox_iou(blob, d) for d in static_dets)


def _tree_like_penalty(blob: dict) -> float:
    """Heuristic for tree-like static texture (no COCO tree class)."""
    edge = _blob_get(blob, "edge_density")
    disp = _blob_get(blob, "centroid_displacement")
    flicker = _blob_get(blob, "texture_flicker_score")
    penalty = 0.0
    if edge > DEFAULT_EDGE_DENSITY_HIGH and disp < DEFAULT_CENTROID_MOVEMENT_LOW:
        penalty += 0.35
    if flicker > DEFAULT_TREE_FLICKER:
        penalty += 0.25
    if _blob_get(blob, "temporal_position_stability", 0) > 0.7 and disp < DEFAULT_CENTROID_MOVEMENT_LOW:
        penalty += 0.20
    return min(1.0, penalty)


def compute_false_motion_penalty(
    blob: dict,
    static_dets: list[dict],
) -> tuple[float, list[str]]:
    """
    Return (penalty 0-1, list of reason tags).
    Does not delete blobs — marks penalty for downstream filtering.
    """
    reasons: list[str] = []
    penalty = 0.0

    tex = _blob_get(blob, "texture_flicker_score")
    gm = _blob_get(blob, "global_motion_score")
    edge = _blob_get(blob, "edge_density")
    disp = _blob_get(blob, "centroid_displacement")

    if tex > DEFAULT_TEXTURE_FLICKER_HIGH:
        penalty += 0.25
        reasons.append("high_texture_flicker")
    if gm > DEFAULT_GLOBAL_MOTION_HIGH:
        penalty += 0.20
        reasons.append("high_global_motion")
    if edge > DEFAULT_EDGE_DENSITY_HIGH and disp < DEFAULT_CENTROID_MOVEMENT_LOW:
        penalty += 0.20
        reasons.append("high_edge_low_movement")

    static_iou = _static_overlap_score(blob, static_dets)
    if static_iou >= DEFAULT_STATIC_OVERLAP_IOU:
        penalty += 0.35 * static_iou
        reasons.append("static_class_overlap")

    tree_p = _tree_like_penalty(blob)
    if tree_p > 0:
        penalty += tree_p
        reasons.append("tree_like_texture")

    return min(1.0, penalty), reasons


def apply_false_motion_suppression(
    blob_assoc_df: pd.DataFrame,
    detections_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add false_motion_penalty and is_suppressed columns to blob association data.

    Suppressed blobs are kept as motion_only_candidate but penalized for ROI selection.
    """
    if blob_assoc_df.empty:
        return blob_assoc_df

    df = blob_assoc_df.copy()
    penalties: list[float] = []
    reasons_col: list[str] = []
    suppressed: list[bool] = []

    static_by_frame: dict[int, list[dict]] = {}
    if not detections_df.empty:
        static_dets = detections_df[detections_df["is_static_suppression_class"]]
        for frame_idx in static_dets["frame_idx"].unique():
            static_by_frame[int(frame_idx)] = (
                static_dets[static_dets["frame_idx"] == frame_idx].to_dict("records")
            )

    for _, row in df.iterrows():
        blob = row.to_dict()
        frame_idx = int(blob["frame_idx"])
        static_dets = static_by_frame.get(frame_idx, [])
        penalty, reasons = compute_false_motion_penalty(blob, static_dets)
        penalties.append(round(penalty, 4))
        reasons_col.append(";".join(reasons) if reasons else "")
        suppressed.append(penalty >= 0.55)

    df["false_motion_penalty"] = penalties
    df["false_motion_reasons"] = reasons_col
    df["is_suppressed"] = suppressed

    if "blob_importance" in df.columns:
        df["adjusted_blob_importance"] = (
            df["blob_importance"] * (1.0 - df["false_motion_penalty"])
        ).round(6)
    else:
        df["adjusted_blob_importance"] = 0.0

    return df
