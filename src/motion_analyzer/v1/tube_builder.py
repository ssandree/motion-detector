"""Temporal motion tube construction from frame-level blobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-8


@dataclass
class TubeMatchConfig:
    max_gap: int = 2
    iou_weight: float = 0.45
    center_weight: float = 0.35
    size_weight: float = 0.20
    match_threshold: float = 0.25
    top_blobs_per_frame: int | None = 50


@dataclass
class TubeSegment:
    """A temporally linked sequence of blob observations."""

    tube_id: int
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


def _bbox_iou(a: dict, b: dict) -> float:
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / max(union, EPS)


def _center_distance_norm(a: dict, b: dict, frame_w: int, frame_h: int) -> float:
    dist = float(np.hypot(a["center_x"] - b["center_x"], a["center_y"] - b["center_y"]))
    diag = float(np.hypot(frame_w, frame_h))
    return dist / max(diag, EPS)


def _size_similarity(a: dict, b: dict) -> float:
    area_a = max(a["bbox_width"] * a["bbox_height"], 1)
    area_b = max(b["bbox_width"] * b["bbox_height"], 1)
    ratio = min(area_a, area_b) / max(area_a, area_b)
    return float(ratio)


def match_score(
    prev: dict,
    curr: dict,
    frame_w: int,
    frame_h: int,
    config: TubeMatchConfig,
) -> float:
    """Combined IoU + center proximity + size similarity."""
    # Same grid centroid and same merged size → likely same region.
    if (
        "grid_row" in prev
        and "grid_col" in prev
        and prev.get("grid_row") == curr.get("grid_row")
        and prev.get("grid_col") == curr.get("grid_col")
        and prev.get("num_grid_cells") == curr.get("num_grid_cells")
    ):
        return 1.0

    iou = _bbox_iou(prev, curr)
    center_sim = 1.0 - min(1.0, _center_distance_norm(prev, curr, frame_w, frame_h) / 0.15)
    size_sim = _size_similarity(prev, curr)
    return (
        config.iou_weight * iou
        + config.center_weight * center_sim
        + config.size_weight * size_sim
    )


def _prepare_blobs(df: pd.DataFrame, config: TubeMatchConfig) -> pd.DataFrame:
    """Optionally keep top-N blobs per frame by importance."""
    if config.top_blobs_per_frame is None:
        return df.sort_values(["frame_idx", "blob_id_in_frame"]).reset_index(drop=True)

    ranked = (
        df.sort_values(["frame_idx", "blob_importance"], ascending=[True, False])
        .groupby("frame_idx", group_keys=False)
        .head(config.top_blobs_per_frame)
    )
    return ranked.sort_values(["frame_idx", "blob_id_in_frame"]).reset_index(drop=True)


def build_tubes(
    blobs_df: pd.DataFrame,
    *,
    frame_w: int = 1280,
    frame_h: int = 720,
    config: TubeMatchConfig | None = None,
    min_tube_length: int = 2,
) -> list[TubeSegment]:
    """
    Greedy temporal linking of blobs into motion tubes.

    Gap is measured in sampled-frame sequence steps (not raw frame indices),
    so max_gap=2 allows up to 2 missing sampled frames between observations.
    """
    cfg = config or TubeMatchConfig()
    df = _prepare_blobs(blobs_df, cfg)

    frames = sorted(df["frame_idx"].unique())
    frame_to_seq = {int(f): i for i, f in enumerate(frames)}
    blobs_by_frame: dict[int, list[dict]] = {}
    for frame_idx in frames:
        frame_rows = df[df["frame_idx"] == frame_idx]
        records = frame_rows.to_dict("records")
        seq = frame_to_seq[int(frame_idx)]
        for rec in records:
            rec["_seq_idx"] = seq
        blobs_by_frame[int(frame_idx)] = records

    active: list[TubeSegment] = []
    finished: list[TubeSegment] = []
    next_id = 0

    for frame_idx in frames:
        seq_idx = frame_to_seq[int(frame_idx)]
        frame_blobs = blobs_by_frame[int(frame_idx)]

        still_active: list[TubeSegment] = []
        for tube in active:
            if seq_idx - tube.last_seq <= cfg.max_gap + 1:
                still_active.append(tube)
            else:
                finished.append(tube)
        active = still_active

        assigned_tubes: set[int] = set()
        sorted_blobs = sorted(
            frame_blobs, key=lambda b: b["blob_importance"], reverse=True
        )

        for blob in sorted_blobs:
            best_tube: TubeSegment | None = None
            best_score = cfg.match_threshold

            for tube in active:
                if tube.tube_id in assigned_tubes:
                    continue
                seq_gap = seq_idx - tube.last_seq
                if seq_gap < 1 or seq_gap > cfg.max_gap + 1:
                    continue
                score = match_score(tube.last_obs, blob, frame_w, frame_h, cfg)
                if score > best_score:
                    best_score = score
                    best_tube = tube

            if best_tube is not None:
                best_tube.observations.append(blob)
                assigned_tubes.add(best_tube.tube_id)
            else:
                tube = TubeSegment(tube_id=next_id, observations=[blob])
                next_id += 1
                active.append(tube)

    finished.extend(active)

    if min_tube_length > 1:
        finished = [t for t in finished if len(t.observations) >= min_tube_length]

    # Remove internal sequence index before downstream use
    for tube in finished:
        for obs in tube.observations:
            obs.pop("_seq_idx", None)

    return finished
