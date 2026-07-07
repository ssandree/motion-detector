"""Temporal object-level motion tube tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from motion_analyzer.v2.bbox_utils import bbox_iou, center_distance_norm, size_similarity

EPS = 1e-8


@dataclass
class ObjectTubeMatchConfig:
    max_gap: int = 2
    iou_weight: float = 0.40
    center_weight: float = 0.30
    size_weight: float = 0.20
    class_weight: float = 0.10
    match_threshold: float = 0.30
    min_moving_frames: int = 1


@dataclass
class ObjectTubeSegment:
    object_tube_id: int
    class_name: str
    observations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def last_frame(self) -> int:
        return int(self.observations[-1]["frame_idx"])

    @property
    def last_seq(self) -> int:
        return int(self.observations[-1]["_seq_idx"])

    @property
    def last_obs(self) -> dict[str, Any]:
        return self.observations[-1]


def _match_score(
    prev: dict,
    curr: dict,
    frame_w: int,
    frame_h: int,
    config: ObjectTubeMatchConfig,
) -> float:
    if prev["class_name"] != curr["class_name"]:
        return 0.0
    iou = bbox_iou(prev, curr)
    center_sim = 1.0 - min(1.0, center_distance_norm(prev, curr, frame_w, frame_h) / 0.18)
    size_sim = size_similarity(prev, curr)
    return (
        config.iou_weight * iou
        + config.center_weight * center_sim
        + config.size_weight * size_sim
        + config.class_weight
    )


def build_object_tubes(
    regions_df: pd.DataFrame,
    *,
    frame_w: int,
    frame_h: int,
    config: ObjectTubeMatchConfig | None = None,
    min_tube_length: int = 1,
    moving_density_threshold: float = 0.02,
) -> list[ObjectTubeSegment]:
    """
    Link motion-associated target object regions across sampled frames.

    Only regions with is_target_class=True and object_motion_score > 0 are tracked.
    """
    cfg = config or ObjectTubeMatchConfig()
    if regions_df.empty:
        return []

    df = regions_df[regions_df["is_target_class"]].copy()
    if df.empty:
        return []

    frames = sorted(df["frame_idx"].unique())
    frame_to_seq = {int(f): i for i, f in enumerate(frames)}
    by_frame: dict[int, list[dict]] = {}
    for frame_idx in frames:
        rows = df[df["frame_idx"] == frame_idx].to_dict("records")
        seq = frame_to_seq[int(frame_idx)]
        for rec in rows:
            rec["_seq_idx"] = seq
            rec["_is_moving"] = float(rec.get("object_motion_density", 0)) >= moving_density_threshold
        by_frame[int(frame_idx)] = rows

    active: list[ObjectTubeSegment] = []
    finished: list[ObjectTubeSegment] = []
    next_id = 0

    for frame_idx in frames:
        seq_idx = frame_to_seq[int(frame_idx)]
        frame_regions = by_frame[int(frame_idx)]

        still_active: list[ObjectTubeSegment] = []
        for tube in active:
            if seq_idx - tube.last_seq <= cfg.max_gap + 1:
                still_active.append(tube)
            else:
                finished.append(tube)
        active = still_active

        assigned: set[int] = set()
        sorted_regions = sorted(
            frame_regions,
            key=lambda r: r.get("object_motion_score", 0),
            reverse=True,
        )

        for region in sorted_regions:
            best_tube: ObjectTubeSegment | None = None
            best_score = cfg.match_threshold

            for tube in active:
                if tube.object_tube_id in assigned:
                    continue
                if tube.class_name != region["class_name"]:
                    continue
                seq_gap = seq_idx - tube.last_seq
                if seq_gap < 1 or seq_gap > cfg.max_gap + 1:
                    continue
                score = _match_score(tube.last_obs, region, frame_w, frame_h, cfg)
                if score > best_score:
                    best_score = score
                    best_tube = tube

            if best_tube is not None:
                best_tube.observations.append(region)
                assigned.add(best_tube.object_tube_id)
            else:
                tube = ObjectTubeSegment(
                    object_tube_id=next_id,
                    class_name=region["class_name"],
                    observations=[region],
                )
                next_id += 1
                active.append(tube)

    finished.extend(active)

    if min_tube_length > 1:
        finished = [t for t in finished if len(t.observations) >= min_tube_length]

    for tube in finished:
        for obs in tube.observations:
            obs.pop("_seq_idx", None)

    return finished
