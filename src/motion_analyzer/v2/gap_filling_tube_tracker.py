"""Gap-filling motion tube tracker with detector-supported interpolation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from motion_analyzer.v1.tube_builder import TubeSegment, _prepare_blobs
from motion_analyzer.v2.bbox_utils import (
    bbox_iou,
    center_distance_norm,
    size_similarity,
    union_bbox,
)
from motion_analyzer.v2.blob_detector_context import PERSON_CLASS, VEHICLE_CLASSES

logger = logging.getLogger(__name__)

EPS = 1e-8

OBSERVED_MOTION = "observed_motion"
DETECTOR_SUPPORTED = "detector_supported"
INTERPOLATED = "interpolated"

CLASS_COMPAT: dict[str, set[str]] = {
    PERSON_CLASS: {PERSON_CLASS},
    "car": VEHICLE_CLASSES | {PERSON_CLASS},
    "truck": VEHICLE_CLASSES,
    "bus": VEHICLE_CLASSES,
    "bicycle": VEHICLE_CLASSES | {PERSON_CLASS},
    "motorcycle": VEHICLE_CLASSES | {PERSON_CLASS},
}


@dataclass
class GapFillingTubeConfig:
    tube_max_gap: int = 6
    gap_match_decay: float = 0.85
    max_gap_seconds: float = 1.2
    sampled_fps: float = 5.0
    base_match_threshold: float = 0.25
    top_blobs_per_frame: int | None = 50
    min_observed_frames: int = 2

    @property
    def effective_max_gap(self) -> int:
        from_seconds = max(1, int(round(self.max_gap_seconds * self.sampled_fps)))
        return min(self.tube_max_gap, from_seconds)


def _parse_classes(s: str) -> list[str]:
    if not s or (isinstance(s, float) and np.isnan(s)):
        return []
    return [c.strip() for c in str(s).split(",") if c.strip()]


def _classes_compatible(tube_classes: set[str], det_class: str) -> bool:
    if not tube_classes:
        return det_class in ({PERSON_CLASS} | VEHICLE_CLASSES)
    for tc in tube_classes:
        compat = CLASS_COMPAT.get(tc, {tc})
        if det_class in compat:
            return True
    return False


def _bbox_dict(x1: float, y1: float, x2: float, y2: float) -> dict[str, float]:
    return {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "center_x": (x1 + x2) / 2,
        "center_y": (y1 + y2) / 2,
        "bbox_width": x2 - x1,
        "bbox_height": y2 - y1,
    }


def _clamp_bbox(x1: float, y1: float, x2: float, y2: float, fw: int, fh: int) -> dict[str, float]:
    x1 = max(0.0, min(x1, fw - 1))
    y1 = max(0.0, min(y1, fh - 1))
    x2 = max(x1 + 1, min(x2, fw))
    y2 = max(y1 + 1, min(y2, fh))
    return _bbox_dict(x1, y1, x2, y2)


def _tag_observation(
    obs: dict,
    *,
    observation_type: str,
    motion_observed: bool,
    detector_supported: bool,
    interpolated: bool,
    predicted_bbox: bool,
) -> dict:
    out = dict(obs)
    out["observation_type"] = observation_type
    out["motion_observed"] = motion_observed
    out["detector_supported"] = detector_supported
    out["interpolated"] = interpolated
    out["predicted_bbox"] = predicted_bbox
    return out


def _observed_from_blob(blob: dict) -> dict:
    return _tag_observation(
        blob,
        observation_type=OBSERVED_MOTION,
        motion_observed=True,
        detector_supported=False,
        interpolated=False,
        predicted_bbox=False,
    )


def _estimate_velocity(observations: list[dict], frame_to_seq: dict[int, int]) -> tuple[float, float]:
    observed = [o for o in observations if o.get("motion_observed")]
    if len(observed) < 2:
        return 0.0, 0.0

    recent = observed[-3:]
    vxs, vys = [], []
    for i in range(1, len(recent)):
        prev, curr = recent[i - 1], recent[i]
        ds = frame_to_seq.get(int(curr["frame_idx"]), 0) - frame_to_seq.get(int(prev["frame_idx"]), 0)
        if ds <= 0:
            continue
        vxs.append((curr["center_x"] - prev["center_x"]) / ds)
        vys.append((curr["center_y"] - prev["center_y"]) / ds)
    if not vxs:
        return 0.0, 0.0
    return float(np.mean(vxs)), float(np.mean(vys))


def _avg_bbox_size(observations: list[dict]) -> tuple[float, float]:
    observed = [o for o in observations if o.get("motion_observed")] or observations
    recent = observed[-3:]
    ws = [float(o.get("bbox_width", o["x2"] - o["x1"])) for o in recent]
    hs = [float(o.get("bbox_height", o["y2"] - o["y1"])) for o in recent]
    return float(np.mean(ws)), float(np.mean(hs))


def predict_bbox(
    tube: TubeSegment,
    gap: int,
    frame_to_seq: dict[int, int],
    frame_w: int,
    frame_h: int,
) -> dict[str, float]:
    last = tube.observations[-1]
    vx, vy = _estimate_velocity(tube.observations, frame_to_seq)
    w, h = _avg_bbox_size(tube.observations)
    cx = float(last["center_x"]) + vx * gap
    cy = float(last["center_y"]) + vy * gap
    return _clamp_bbox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, frame_w, frame_h)


def _detector_context_similarity(tube_classes: set[str], candidate: dict) -> float:
    det_classes = _parse_classes(candidate.get("associated_classes", ""))
    if not det_classes and "class_name" in candidate:
        det_classes = [candidate["class_name"]]
    if not tube_classes and not det_classes:
        return 0.5
    if tube_classes & set(det_classes):
        return 1.0
    if any(_classes_compatible(tube_classes, c) for c in det_classes):
        return 0.7
    return 0.0


def _motion_direction_similarity(tube: TubeSegment, candidate: dict, frame_to_seq: dict[int, int]) -> float:
    vx, vy = _estimate_velocity(tube.observations, frame_to_seq)
    if abs(vx) < EPS and abs(vy) < EPS:
        return 0.5
    last = tube.observations[-1]
    dx = candidate["center_x"] - last["center_x"]
    dy = candidate["center_y"] - last["center_y"]
    norm_c = float(np.hypot(dx, dy))
    norm_v = float(np.hypot(vx, vy))
    if norm_c < EPS or norm_v < EPS:
        return 0.5
    cos_sim = (dx * vx + dy * vy) / (norm_c * norm_v)
    return max(0.0, min(1.0, (cos_sim + 1) / 2))


def gap_match_score(
    tube: TubeSegment,
    candidate: dict,
    *,
    gap: int,
    frame_w: int,
    frame_h: int,
    frame_to_seq: dict[int, int],
    config: GapFillingTubeConfig,
) -> float:
    predicted = predict_bbox(tube, gap, frame_to_seq, frame_w, frame_h)
    tube_classes = set()
    for o in tube.observations:
        tube_classes.update(_parse_classes(o.get("associated_classes", "")))

    iou = bbox_iou(predicted, candidate)
    center_sim = 1.0 - min(1.0, center_distance_norm(predicted, candidate, frame_w, frame_h) / 0.18)
    size_sim = size_similarity(predicted, candidate)
    det_sim = _detector_context_similarity(tube_classes, candidate)
    dir_sim = _motion_direction_similarity(tube, candidate, frame_to_seq)

    raw = (
        0.35 * iou
        + 0.30 * center_sim
        + 0.15 * size_sim
        + 0.10 * det_sim
        + 0.10 * dir_sim
    )
    return raw * (config.gap_match_decay ** max(0, gap - 1))


def _match_threshold(gap: int, config: GapFillingTubeConfig) -> float:
    return min(0.85, config.base_match_threshold / (config.gap_match_decay ** max(0, gap - 1)))


def _tube_classes(tube: TubeSegment) -> set[str]:
    classes: set[str] = set()
    for o in tube.observations:
        classes.update(_parse_classes(o.get("associated_classes", "")))
    return classes


def _find_detector_support(
    predicted: dict[str, float],
    dets: list[dict],
    tube_classes: set[str],
    frame_w: int,
    frame_h: int,
) -> dict | None:
    best: dict | None = None
    best_score = 0.0
    for det in dets:
        cls = det.get("class_name", "")
        if cls not in ({PERSON_CLASS} | VEHICLE_CLASSES) and not _classes_compatible(tube_classes, cls):
            continue
        if tube_classes and not _classes_compatible(tube_classes, cls):
            continue
        iou = bbox_iou(predicted, det)
        dist = center_distance_norm(predicted, det, frame_w, frame_h)
        score = iou + (1.0 - min(1.0, dist / 0.12)) * 0.5
        if cls in ({PERSON_CLASS} | VEHICLE_CLASSES):
            score += 0.15
        if score > best_score and (iou > 0.05 or dist < 0.08):
            best_score = score
            best = det
    return best


def _synthetic_observation(
    frame_idx: int,
    timestamp_sec: float,
    bbox: dict[str, float],
    *,
    observation_type: str,
    detector_supported: bool,
    interpolated: bool,
    predicted_bbox: bool,
    det: dict | None = None,
    tube_classes: set[str] | None = None,
) -> dict:
    classes = sorted(tube_classes or set())
    if det:
        classes = sorted(set(classes) | {det["class_name"]})
    pred_coords = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
    obs = {
        "frame_idx": frame_idx,
        "timestamp_sec": timestamp_sec,
        "x1": int(bbox["x1"]),
        "y1": int(bbox["y1"]),
        "x2": int(bbox["x2"]),
        "y2": int(bbox["y2"]),
        "center_x": bbox["center_x"],
        "center_y": bbox["center_y"],
        "bbox_width": bbox["bbox_width"],
        "bbox_height": bbox["bbox_height"],
        "motion_density": 0.0,
        "motion_energy": 0.0,
        "blob_importance": 0.0,
        "associated_classes": ",".join(classes),
        "detector_context_score": float(det.get("confidence", 0)) if det else 0.0,
        "predicted_bbox_coords": pred_coords,
    }
    if det:
        obs["context_det_id"] = int(det["det_id_in_frame"])
        obs["context_detector_bbox"] = [int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])]
    return _tag_observation(
        obs,
        observation_type=observation_type,
        motion_observed=False,
        detector_supported=detector_supported,
        interpolated=interpolated,
        predicted_bbox=predicted_bbox,
    )


def compute_track_continuity_metrics(observations: list[dict], frame_to_seq: dict[int, int]) -> dict[str, Any]:
    if not observations:
        return {
            "num_observed_motion_frames": 0,
            "num_detector_supported_frames": 0,
            "num_interpolated_frames": 0,
            "track_continuity": 0.0,
            "motion_persistence": 0.0,
            "gap_count": 0,
            "max_gap_length": 0,
            "mean_gap_length": 0.0,
            "total_tube_frames": 0,
        }

    n_obs = sum(1 for o in observations if o.get("observation_type") == OBSERVED_MOTION)
    n_det = sum(1 for o in observations if o.get("observation_type") == DETECTOR_SUPPORTED)
    n_interp = sum(1 for o in observations if o.get("observation_type") == INTERPOLATED)
    total = len(observations)

    start_seq = frame_to_seq.get(int(observations[0]["frame_idx"]), 0)
    end_seq = frame_to_seq.get(int(observations[-1]["frame_idx"]), start_seq)
    expected = max(1, end_seq - start_seq + 1)

    gaps: list[int] = []
    for i in range(1, len(observations)):
        s0 = frame_to_seq.get(int(observations[i - 1]["frame_idx"]), 0)
        s1 = frame_to_seq.get(int(observations[i]["frame_idx"]), 0)
        g = s1 - s0 - 1
        if g > 0:
            gaps.append(g)

    return {
        "num_observed_motion_frames": n_obs,
        "num_detector_supported_frames": n_det,
        "num_interpolated_frames": n_interp,
        "total_tube_frames": total,
        "track_continuity": round(total / expected, 6),
        "motion_persistence": round(n_obs / max(total, 1), 6),
        "gap_count": len(gaps),
        "max_gap_length": max(gaps) if gaps else 0,
        "mean_gap_length": round(float(np.mean(gaps)), 4) if gaps else 0.0,
    }


def _last_observed_seq(tube: TubeSegment) -> int:
    for obs in reversed(tube.observations):
        if obs.get("motion_observed"):
            return int(obs["_seq_idx"])
    return int(tube.observations[0]["_seq_idx"])


def _fill_gap_frames(
    tube: TubeSegment,
    frame_indices: list[int],
    timestamps: dict[int, float],
    frame_to_seq: dict[int, int],
    current_seq: int,
    detections_by_frame: dict[int, list[dict]],
    frame_w: int,
    frame_h: int,
    max_gap: int,
) -> None:
    """Insert synthetic observations for intermediate missing frames (within max_gap of last observed)."""
    last_obs_seq = _last_observed_seq(tube)
    tube_cls = _tube_classes(tube)

    for seq in range(tube.last_seq + 1, current_seq):
        if seq - last_obs_seq > max_gap:
            break
        fi = frame_indices[seq]
        ts = timestamps[fi]
        step_gap = seq - last_obs_seq
        predicted = predict_bbox(tube, step_gap, frame_to_seq, frame_w, frame_h)
        dets = detections_by_frame.get(fi, [])
        det = _find_detector_support(predicted, dets, tube_cls, frame_w, frame_h)

        if det:
            merged = union_bbox([predicted, det])
            syn = _synthetic_observation(
                fi, ts, merged,
                observation_type=DETECTOR_SUPPORTED,
                detector_supported=True,
                interpolated=False,
                predicted_bbox=True,
                det=det,
                tube_classes=tube_cls,
            )
        else:
            syn = _synthetic_observation(
                fi, ts, predicted,
                observation_type=INTERPOLATED,
                detector_supported=False,
                interpolated=True,
                predicted_bbox=True,
                tube_classes=tube_cls,
            )
        syn["_seq_idx"] = seq
        tube.observations.append(syn)


def build_gap_filling_tubes(
    blobs_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    detections_by_frame: dict[int, list[dict]],
    *,
    frame_w: int,
    frame_h: int,
    config: GapFillingTubeConfig | None = None,
) -> list[TubeSegment]:
    """Build motion tubes with gap filling via velocity prediction and detector support."""
    cfg = config or GapFillingTubeConfig()
    max_gap = cfg.effective_max_gap

    from motion_analyzer.v1.tube_builder import TubeMatchConfig

    prep_cfg = TubeMatchConfig(top_blobs_per_frame=cfg.top_blobs_per_frame)
    df = _prepare_blobs(blobs_df, prep_cfg)

    frame_indices = [int(x) for x in frame_df["frame_idx"].tolist()]
    timestamps = {int(r["frame_idx"]): float(r["timestamp_sec"]) for _, r in frame_df.iterrows()}
    frame_to_seq = {fi: i for i, fi in enumerate(frame_indices)}

    blobs_by_frame: dict[int, list[dict]] = {}
    for fi in frame_indices:
        rows = df[df["frame_idx"] == fi]
        blobs_by_frame[fi] = [dict(r) for r in rows.to_dict("records")]

    active: list[TubeSegment] = []
    finished: list[TubeSegment] = []
    next_id = 0

    for seq_idx, frame_idx in enumerate(frame_indices):
        ts = timestamps[frame_idx]
        frame_blobs = blobs_by_frame.get(frame_idx, [])
        assigned_blobs: set[int] = set()
        assigned_tubes: set[int] = set()

        still_active: list[TubeSegment] = []
        for tube in active:
            gap_since_obs = seq_idx - _last_observed_seq(tube)
            if gap_since_obs > max_gap + 1:
                finished.append(tube)
            else:
                still_active.append(tube)
        active = still_active

        sorted_blobs = sorted(frame_blobs, key=lambda b: b.get("blob_importance", 0), reverse=True)

        for blob in sorted_blobs:
            bid = int(blob.get("blob_id_in_frame", 0))
            best_tube: TubeSegment | None = None
            best_score = -1.0
            best_gap = 1

            for tube in active:
                if tube.tube_id in assigned_tubes:
                    continue
                gap = seq_idx - tube.last_seq
                if gap < 1 or gap > max_gap + 1:
                    continue
                score = gap_match_score(
                    tube, blob, gap=gap,
                    frame_w=frame_w, frame_h=frame_h,
                    frame_to_seq=frame_to_seq, config=cfg,
                )
                threshold = _match_threshold(gap, cfg)
                if score > threshold and score > best_score:
                    best_score = score
                    best_tube = tube
                    best_gap = gap

            if best_tube is not None:
                if best_gap > 1:
                    _fill_gap_frames(
                        best_tube, frame_indices, timestamps, frame_to_seq,
                        seq_idx, detections_by_frame, frame_w, frame_h, max_gap,
                    )
                obs = _observed_from_blob(blob)
                obs["_seq_idx"] = seq_idx
                best_tube.observations.append(obs)
                assigned_tubes.add(best_tube.tube_id)
                assigned_blobs.add(bid)
            else:
                tube = TubeSegment(tube_id=next_id, observations=[])
                obs = _observed_from_blob(blob)
                obs["_seq_idx"] = seq_idx
                tube.observations.append(obs)
                next_id += 1
                active.append(tube)
                assigned_blobs.add(bid)

        for tube in active:
            if tube.tube_id in assigned_tubes:
                continue
            gap_since_obs = seq_idx - _last_observed_seq(tube)
            if gap_since_obs < 1 or gap_since_obs > max_gap:
                continue
            gap_from_last = seq_idx - tube.last_seq
            if gap_from_last > 1:
                _fill_gap_frames(
                    tube, frame_indices, timestamps, frame_to_seq,
                    seq_idx, detections_by_frame, frame_w, frame_h, max_gap,
                )
            predicted = predict_bbox(tube, gap_since_obs, frame_to_seq, frame_w, frame_h)
            dets = detections_by_frame.get(frame_idx, [])
            tube_cls = _tube_classes(tube)
            det = _find_detector_support(predicted, dets, tube_cls, frame_w, frame_h)
            if det:
                merged = union_bbox([predicted, det])
                syn = _synthetic_observation(
                    frame_idx, ts, merged,
                    observation_type=DETECTOR_SUPPORTED,
                    detector_supported=True,
                    interpolated=False,
                    predicted_bbox=True,
                    det=det,
                    tube_classes=tube_cls,
                )
            else:
                syn = _synthetic_observation(
                    frame_idx, ts, predicted,
                    observation_type=INTERPOLATED,
                    detector_supported=False,
                    interpolated=True,
                    predicted_bbox=True,
                    tube_classes=tube_cls,
                )
            syn["_seq_idx"] = seq_idx
            tube.observations.append(syn)

    finished.extend(active)

    result: list[TubeSegment] = []
    for tube in finished:
        for obs in tube.observations:
            obs.pop("_seq_idx", None)
        n_observed = sum(1 for o in tube.observations if o.get("motion_observed"))
        if n_observed >= cfg.min_observed_frames:
            result.append(tube)

    logger.info(
        "Gap-filling tubes: %d (max_gap=%d, decay=%.2f)",
        len(result), max_gap, cfg.gap_match_decay,
    )
    return result
