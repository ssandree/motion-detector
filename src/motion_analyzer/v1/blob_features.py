"""Blob-level motion feature computation and importance scoring."""

from __future__ import annotations

from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from motion_analyzer.v1.blob_extractor import RawBlob

EPS = 1e-8

IMPORTANCE_POSITIVE = {
    "motion_density": 0.22,
    "spatial_novelty_score": 0.18,
    "scale_score": 0.15,
    "local_peak_score": 0.15,
    "object_motion_score": 0.15,
    "flow_direction_coherence": 0.10,
}

IMPORTANCE_NEGATIVE = {
    "texture_flicker_score": 0.20,
    "global_motion_score": 0.15,
}


@dataclass
class BlobFeatureConfig:
    small_blob_area_ratio_threshold: float = 0.003
    spatial_novelty_history_frames: int = 5
    spatial_novelty_radius_ratio: float = 0.05
    scale_eps: float = 1e-6
    use_texture_suppression: bool = True
    texture_flicker_penalty: float = 0.20
    global_motion_penalty: float = 0.15


@dataclass
class BlobHistory:
    """Recent blob centers for spatial novelty scoring."""

    max_frames: int = 5
    _frames: deque[list[tuple[float, float]]] = field(init=False)

    def __post_init__(self) -> None:
        self._frames = deque(maxlen=self.max_frames)

    def push(self, centers: list[tuple[float, float]]) -> None:
        self._frames.append(centers)

    def centers(self) -> list[tuple[float, float]]:
        flat: list[tuple[float, float]] = []
        for frame_centers in self._frames:
            flat.extend(frame_centers)
        return flat


def _min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < EPS:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _spatial_novelty(
    center_x: float,
    center_y: float,
    history_centers: list[tuple[float, float]],
    frame_w: int,
    frame_h: int,
    radius_ratio: float,
) -> float:
    if not history_centers:
        return 1.0

    diag = float(np.hypot(frame_w, frame_h))
    radius = radius_ratio * diag
    min_dist = min(
        float(np.hypot(center_x - hx, center_y - hy))
        for hx, hy in history_centers
    )
    if min_dist >= radius:
        return 1.0
    return min_dist / max(radius, EPS)


def _local_peak_score(motion_density: float, frame_mean_motion: float) -> float:
    return motion_density / (frame_mean_motion + EPS)


def _scale_score(bbox_area_ratio: float, eps: float) -> float:
    return 1.0 / float(np.sqrt(bbox_area_ratio + eps))


def _edge_density(curr_gray: np.ndarray | None, x1: int, y1: int, x2: int, y2: int) -> float:
    if curr_gray is None:
        return 0.0
    roi = curr_gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    edges = cv2.Canny(roi, 50, 150)
    return float(edges.mean()) / 255.0


def compute_texture_flicker_score(
    edge_density: float,
    centroid_displacement: float,
    temporal_position_stability: float,
    motion_density: float,
) -> float:
    """High when edges flicker in a fixed location with little displacement."""
    low_disp = 1.0 - min(1.0, centroid_displacement / 0.025)
    motion_factor = min(1.0, motion_density * 12.0)
    score = edge_density * low_disp * temporal_position_stability * motion_factor
    return round(min(1.0, score), 6)


def compute_object_motion_score(
    centroid_displacement: float,
    flow_direction_coherence: float,
    texture_flicker_score: float,
) -> float:
    disp = min(1.0, centroid_displacement / 0.06)
    coh = flow_direction_coherence
    anti_tex = 1.0 - texture_flicker_score
    score = 0.45 * disp + 0.35 * coh + 0.20 * anti_tex
    return round(min(1.0, max(0.0, score)), 6)


def compute_blob_features(
    raw: RawBlob,
    motion_map: np.ndarray,
    *,
    video_name: str,
    frame_idx: int,
    timestamp_sec: float,
    frame_mean_motion: float,
    history: BlobHistory,
    config: BlobFeatureConfig | None = None,
    curr_gray: np.ndarray | None = None,
    global_motion_score: float = 0.0,
    camera_motion_flag: bool = False,
    merge_stage: str = "raw",
) -> dict[str, Any]:
    """Compute scalar features for a single blob."""
    cfg = config or BlobFeatureConfig()
    h, w = motion_map.shape
    frame_area = h * w

    x1, y1, x2, y2 = raw.x1, raw.y1, raw.x2, raw.y2
    roi = motion_map[y1:y2, x1:x2]
    masked = roi[raw.label_mask]

    motion_energy = float(masked.sum())
    mean_motion = float(masked.mean()) if masked.size > 0 else 0.0
    max_motion = float(masked.max()) if masked.size > 0 else 0.0

    bbox_w = x2 - x1
    bbox_h = y2 - y1
    bbox_area = bbox_w * bbox_h
    bbox_area_ratio = bbox_area / frame_area if frame_area > 0 else 0.0
    motion_density = motion_energy / max(bbox_area, 1)

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    aspect_ratio = bbox_w / max(bbox_h, 1)

    scale = _scale_score(bbox_area_ratio, cfg.scale_eps)
    local_peak = _local_peak_score(motion_density, frame_mean_motion)
    spatial_novelty = _spatial_novelty(
        center_x,
        center_y,
        history.centers(),
        w,
        h,
        cfg.spatial_novelty_radius_ratio,
    )
    small_blob_flag = bbox_area_ratio < cfg.small_blob_area_ratio_threshold
    edge_d = _edge_density(curr_gray, x1, y1, x2, y2)

    return {
        "video_name": video_name,
        "frame_idx": frame_idx,
        "timestamp_sec": round(timestamp_sec, 4),
        "blob_id_in_frame": raw.blob_id,
        "merge_stage": merge_stage,
        "source_raw_blob_ids": ",".join(str(i) for i in raw.source_blob_ids),
        "grid_row": raw.grid_row,
        "grid_col": raw.grid_col,
        "num_grid_cells": raw.num_grid_cells,
        "mean_flow_magnitude": raw.mean_flow_magnitude,
        "flow_direction_coherence": raw.flow_direction_coherence,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "bbox_width": bbox_w,
        "bbox_height": bbox_h,
        "bbox_area_ratio": round(bbox_area_ratio, 8),
        "motion_energy": round(motion_energy, 4),
        "mean_motion": round(mean_motion, 6),
        "max_motion": round(max_motion, 6),
        "motion_density": round(motion_density, 6),
        "center_x": round(center_x, 2),
        "center_y": round(center_y, 2),
        "aspect_ratio": round(aspect_ratio, 4),
        "scale_score": round(scale, 6),
        "local_peak_score": round(local_peak, 6),
        "spatial_novelty_score": round(spatial_novelty, 6),
        "small_blob_flag": small_blob_flag,
        "edge_density": round(edge_d, 6),
        "centroid_displacement": 0.0,
        "temporal_position_stability": 0.0,
        "texture_flicker_score": 0.0,
        "object_motion_score": 0.0,
        "global_motion_score": round(global_motion_score, 6),
        "camera_motion_flag": camera_motion_flag,
    }


def add_importance_and_ranks(
    blobs: list[dict[str, Any]],
    config: BlobFeatureConfig | None = None,
) -> list[dict[str, Any]]:
    """Add blob_importance and per-frame ranks."""
    if not blobs:
        return blobs

    cfg = config or BlobFeatureConfig()
    pos_keys = list(IMPORTANCE_POSITIVE.keys())
    neg_keys = list(IMPORTANCE_NEGATIVE.keys())

    normalized: dict[str, list[float]] = {}
    for key in pos_keys + neg_keys:
        normalized[key] = _min_max_normalize([float(b.get(key, 0.0)) for b in blobs])

    tex_penalty = cfg.texture_flicker_penalty if cfg.use_texture_suppression else 0.0
    gm_penalty = cfg.global_motion_penalty if cfg.use_texture_suppression else 0.0

    for i, blob in enumerate(blobs):
        importance = sum(
            IMPORTANCE_POSITIVE[key] * normalized[key][i] for key in pos_keys
        )
        if cfg.use_texture_suppression:
            importance -= tex_penalty * normalized["texture_flicker_score"][i]
            importance -= gm_penalty * normalized["global_motion_score"][i]
        blob["blob_importance"] = round(max(0.0, importance), 6)

    by_energy = sorted(range(len(blobs)), key=lambda i: blobs[i]["motion_energy"], reverse=True)
    by_importance = sorted(
        range(len(blobs)), key=lambda i: blobs[i]["blob_importance"], reverse=True
    )

    energy_rank = [0] * len(blobs)
    importance_rank = [0] * len(blobs)
    for rank, idx in enumerate(by_energy, start=1):
        energy_rank[idx] = rank
    for rank, idx in enumerate(by_importance, start=1):
        importance_rank[idx] = rank

    for i, blob in enumerate(blobs):
        blob["rank_by_motion_energy"] = energy_rank[i]
        blob["rank_by_importance"] = importance_rank[i]

    return blobs


def _bbox_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter / max(area_a + area_b - inter, 1)


def enrich_temporal_blob_features(
    all_blobs: list[dict[str, Any]],
    *,
    frame_h: int,
    frame_w: int,
    config: BlobFeatureConfig | None = None,
) -> list[dict[str, Any]]:
    """Add centroid displacement, stability, texture/object scores across frames."""
    cfg = config or BlobFeatureConfig()
    if not all_blobs:
        return all_blobs

    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for blob in all_blobs:
        by_frame[int(blob["frame_idx"])].append(blob)

    frame_order = sorted(by_frame.keys())

    for fidx in frame_order:
        for blob in by_frame[fidx]:
            best_iou = 0.0
            best_prev: dict[str, Any] | None = None
            for pfidx in reversed(frame_order):
                if pfidx >= fidx:
                    continue
                if fidx - pfidx > 3:
                    break
                for prev in by_frame[pfidx]:
                    iou = _bbox_iou(blob, prev)
                    if iou > best_iou:
                        best_iou = iou
                        best_prev = prev
                if best_iou > 0.2:
                    break

            disp = 0.0
            stability = 0.0
            if best_prev is not None and best_iou > 0.05:
                disp = float(
                    np.hypot(
                        blob["center_x"] - best_prev["center_x"],
                        blob["center_y"] - best_prev["center_y"],
                    )
                ) / max(frame_h, 1)
                stability = max(0.0, 1.0 - disp * 8.0)

            tex = (
                compute_texture_flicker_score(
                    blob.get("edge_density", 0.0),
                    disp,
                    stability,
                    blob.get("motion_density", 0.0),
                )
                if cfg.use_texture_suppression
                else 0.0
            )
            obj = compute_object_motion_score(
                disp,
                blob.get("flow_direction_coherence", 0.0),
                tex,
            )
            blob["centroid_displacement"] = round(disp, 6)
            blob["temporal_position_stability"] = round(stability, 6)
            blob["texture_flicker_score"] = tex
            blob["object_motion_score"] = obj

    return all_blobs


def process_frame_blobs(
    motion_map: np.ndarray,
    raw_blobs: list[RawBlob],
    *,
    video_name: str,
    frame_idx: int,
    timestamp_sec: float,
    history: BlobHistory,
    config: BlobFeatureConfig | None = None,
    curr_gray: np.ndarray | None = None,
    global_motion_score: float = 0.0,
    camera_motion_flag: bool = False,
    merge_stage: str = "raw",
    compute_ranks: bool = True,
) -> list[dict[str, Any]]:
    """Extract features for all blobs in one frame."""
    frame_mean = float(motion_map.mean())
    features = [
        compute_blob_features(
            raw,
            motion_map,
            video_name=video_name,
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            frame_mean_motion=frame_mean,
            history=history,
            config=config,
            curr_gray=curr_gray,
            global_motion_score=global_motion_score,
            camera_motion_flag=camera_motion_flag,
            merge_stage=merge_stage,
        )
        for raw in raw_blobs
    ]
    if compute_ranks:
        features = add_importance_and_ranks(features, config)

    if merge_stage != "raw":
        centers = [(f["center_x"], f["center_y"]) for f in features]
        history.push(centers)

    return features


def filter_persistent_blobs(
    all_blobs: list[dict[str, Any]],
    *,
    frame_w: int,
    frame_h: int,
    min_frames: int = 2,
    config: BlobFeatureConfig | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Keep blob observations belonging to tracks of >= *min_frames*."""
    import pandas as pd

    from motion_analyzer.v1.tube_builder import TubeMatchConfig, build_tubes

    if not all_blobs or min_frames <= 1:
        return all_blobs, 0

    before = len(all_blobs)
    df = pd.DataFrame(all_blobs)
    match_cfg = TubeMatchConfig(top_blobs_per_frame=None)
    tubes = build_tubes(
        df,
        frame_w=frame_w,
        frame_h=frame_h,
        config=match_cfg,
        min_tube_length=min_frames,
    )

    by_frame: dict[int, list[dict[str, Any]]] = {}
    for tube in tubes:
        for obs in tube.observations:
            obs = dict(obs)
            obs.pop("_seq_idx", None)
            obs["persistence_track_id"] = tube.tube_id
            obs["persistence_frames"] = len(tube.observations)
            by_frame.setdefault(int(obs["frame_idx"]), []).append(obs)

    filtered: list[dict[str, Any]] = []
    for frame_idx in sorted(by_frame):
        frame_blobs = add_importance_and_ranks(by_frame[frame_idx], config)
        for i, blob in enumerate(frame_blobs):
            blob["blob_id_in_frame"] = i
        filtered.extend(frame_blobs)

    return filtered, before - len(filtered)


def select_top_k_blobs(
    all_blobs: list[dict[str, Any]],
    top_k: int,
    *,
    rank_key: str = "rank_by_importance",
) -> list[dict[str, Any]]:
    """Select top-k blobs per frame using precomputed rank."""
    if not all_blobs:
        return []

    frames: dict[int, list[dict[str, Any]]] = {}
    for blob in all_blobs:
        frames.setdefault(blob["frame_idx"], []).append(blob)

    top: list[dict[str, Any]] = []
    for frame_idx in sorted(frames):
        frame_blobs = sorted(frames[frame_idx], key=lambda b: b[rank_key])
        top.extend(frame_blobs[:top_k])
    return top
