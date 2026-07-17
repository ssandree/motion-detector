"""Temporal grouping of window ROI candidates → compact novelty ROIs.

Fixes vs prior version:
  - Link by seed proximity / Jaccard (not grown-ROI center)
  - After frequency filter, split into 8-CC components (no disconnected union)
  - Soft-cap persistence; use peak novelty; hard area/blocksize limits
  - Diversity-aware final Top-K (spatially separated)
  - Fixed grid-aligned bbox over each group's active span
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from motion_analyzer.temporal.frame_window_fusion import (
    FusedFrameROI,
    FusedFrameResult,
    FusedROISource,
    cells_to_grid_bbox,
)
from motion_analyzer.temporal.sliding_window_regions import (
    MotionRegion,
    SlidingWindowConfig,
    WindowResult,
)

DEFAULT_MAX_WINDOW_CANDIDATES = 10
DEFAULT_MIN_WINDOW_PERSISTENCE = 2
DEFAULT_BLOCK_FREQUENCY = 0.3
DEFAULT_MAX_ROIS = 2
DEFAULT_SEED_DIST = 0.0  # exact seed only (neighbors via Jaccard)
DEFAULT_JACCARD_LINK = 0.5
DEFAULT_MAX_REP_BLOCKS = 4
DEFAULT_MAX_AREA_RATIO = 0.25
DEFAULT_PERSIST_CAP = 5
DEFAULT_MIN_CENTER_SEP = 2.0
DEFAULT_N_TIME_BINS = 3
DEFAULT_SCORE_ACTIVE_FRAC = 0.35  # trim span to strong members


@dataclass
class TemporalGroup:
    group_id: int
    members: list[MotionRegion]
    window_indices: list[int]
    rep_cells: list[tuple[int, int]]
    rep_bbox: list[float]
    mean_novelty: float
    peak_novelty: float
    window_persistence: int
    compactness: float
    area_ratio: float
    group_score: float
    valid_start_frame: int
    valid_end_frame: int
    block_frequencies: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": int(self.group_id),
            "window_indices": [int(v) for v in self.window_indices],
            "window_persistence": int(self.window_persistence),
            "n_members": int(len(self.members)),
            "rep_cells": [[int(a), int(b)] for a, b in self.rep_cells],
            "n_rep_blocks": int(len(self.rep_cells)),
            "rep_bbox": [float(v) for v in self.rep_bbox],
            "mean_novelty": round(float(self.mean_novelty), 6),
            "peak_novelty": round(float(self.peak_novelty), 6),
            "compactness": round(float(self.compactness), 6),
            "area_ratio": round(float(self.area_ratio), 6),
            "group_score": round(float(self.group_score), 6),
            "valid_start_frame": int(self.valid_start_frame),
            "valid_end_frame": int(self.valid_end_frame),
            "block_frequencies": {
                k: round(float(v), 4) for k, v in self.block_frequencies.items()
            },
            "members": [
                {
                    "window_index": int(m.window_index),
                    "region_id": int(m.region_id),
                    "window_start_frame": int(m.window_start_frame),
                    "window_end_frame": int(m.window_end_frame),
                    "score": round(float(_region_score(m)), 6),
                    "seed_cell": list(m.seed_cell) if m.seed_cell else None,
                    "cells": [[int(a), int(b)] for a, b in m.cells],
                }
                for m in self.members
            ],
        }


class _UF:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _region_score(r: MotionRegion) -> float:
    return float(
        r.component_score
        if r.component_score
        else (r.block_score if r.block_score else r.final_score)
    )


def _cell_set(r: MotionRegion) -> set[tuple[int, int]]:
    return {(int(a), int(b)) for a, b in r.cells}


def _seed(r: MotionRegion) -> tuple[int, int] | None:
    if r.seed_cell is not None:
        return (int(r.seed_cell[0]), int(r.seed_cell[1]))
    cells = list(_cell_set(r))
    if not cells:
        return None
    return cells[0]


def _jaccard(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return float(inter) / float(max(uni, 1))


def _seed_chebyshev(a: MotionRegion, b: MotionRegion) -> float:
    sa, sb = _seed(a), _seed(b)
    if sa is None or sb is None:
        return 1e9
    return float(max(abs(sa[0] - sb[0]), abs(sa[1] - sb[1])))


def _same_group_link(
    a: MotionRegion,
    b: MotionRegion,
    *,
    seed_dist: float,
    jaccard_link: float,
) -> bool:
    """Adjacent windows only; seed near OR cell Jaccard high."""
    if abs(int(a.window_index) - int(b.window_index)) != 1:
        return False
    if _seed_chebyshev(a, b) <= float(seed_dist):
        return True
    return _jaccard(_cell_set(a), _cell_set(b)) >= float(jaccard_link)


def _bbox_block_area(cells: Sequence[tuple[int, int]]) -> int:
    if not cells:
        return 0
    ys = [int(r) for r, _ in cells]
    xs = [int(c) for _, c in cells]
    return int((max(ys) - min(ys) + 1) * (max(xs) - min(xs) + 1))


def _grid_center(cells: Sequence[tuple[int, int]]) -> tuple[float, float]:
    if not cells:
        return (0.0, 0.0)
    ys = [float(r) for r, _ in cells]
    xs = [float(c) for _, c in cells]
    return (sum(ys) / len(ys), sum(xs) / len(xs))


def _split_8cc(cells: Sequence[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    if not cells:
        return []
    ys = [int(r) for r, _ in cells]
    xs = [int(c) for _, c in cells]
    r0, r1 = min(ys), max(ys)
    c0, c1 = min(xs), max(xs)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    mask = np.zeros((h, w), dtype=np.uint8)
    for r, c in cells:
        mask[int(r) - r0, int(c) - c0] = 1
    n, labels = cv2.connectedComponents(mask, connectivity=8)
    comps: list[list[tuple[int, int]]] = []
    for lab in range(1, n):
        yy, xx = np.where(labels == lab)
        comps.append([(int(y + r0), int(x + c0)) for y, x in zip(yy.tolist(), xx.tolist())])
    return comps


def _build_frequency_map(
    members: Sequence[MotionRegion],
) -> tuple[dict[tuple[int, int], float], list[int], dict[int, set[tuple[int, int]]]]:
    by_window: dict[int, set[tuple[int, int]]] = {}
    for m in members:
        wi = int(m.window_index)
        by_window.setdefault(wi, set()).update(_cell_set(m))
    window_ids = sorted(by_window.keys())
    n_w = max(len(window_ids), 1)
    all_blocks: set[tuple[int, int]] = set()
    for cells in by_window.values():
        all_blocks |= cells
    freqs: dict[tuple[int, int], float] = {}
    for cell in all_blocks:
        count = sum(1 for wi in window_ids if cell in by_window[wi])
        freqs[cell] = float(count) / float(n_w)
    return freqs, window_ids, by_window


def _score_group(
    *,
    peak: float,
    mean: float,
    persistence: int,
    compactness: float,
    area_ratio: float,
    persist_cap: int,
    max_area_ratio: float,
    baseline_mean: float = 0.0,
) -> float:
    if area_ratio > float(max_area_ratio) or compactness <= 0:
        return 0.0
    persist_term = math.log1p(float(min(int(persistence), int(persist_cap))))
    novelty = 0.75 * float(peak) + 0.25 * float(mean)
    # Quiet locations get a boost; chronically busy roads are down-weighted.
    # Stronger quiet boost so sparse top/doorboard motion can beat busy roads.
    quiet = max(0.12, 1.0 - float(baseline_mean)) ** 2.0
    area_gate = 1.0 if area_ratio <= 0.5 * float(max_area_ratio) else max(
        0.2, 1.0 - (area_ratio / float(max_area_ratio))
    )
    return float(novelty * persist_term * compactness * area_gate * quiet)


def _members_for_cells(
    members: Sequence[MotionRegion],
    cells: Sequence[tuple[int, int]],
) -> list[MotionRegion]:
    cell_set = {(int(r), int(c)) for r, c in cells}
    out: list[MotionRegion] = []
    for m in members:
        if _cell_set(m) & cell_set:
            out.append(m)
        elif m.seed_cell is not None and tuple(m.seed_cell) in cell_set:
            out.append(m)
    return out if out else list(members)


def build_temporal_groups(
    windows: Sequence[WindowResult],
    *,
    cfg: SlidingWindowConfig,
    min_window_persistence: int = DEFAULT_MIN_WINDOW_PERSISTENCE,
    block_frequency: float = DEFAULT_BLOCK_FREQUENCY,
    max_rois: int = DEFAULT_MAX_ROIS,
    seed_dist: float = DEFAULT_SEED_DIST,
    jaccard_link: float = DEFAULT_JACCARD_LINK,
    max_rep_blocks: int = DEFAULT_MAX_REP_BLOCKS,
    max_area_ratio: float = DEFAULT_MAX_AREA_RATIO,
    persist_cap: int = DEFAULT_PERSIST_CAP,
    min_center_sep: float = DEFAULT_MIN_CENTER_SEP,
    n_time_bins: int = DEFAULT_N_TIME_BINS,
    score_active_frac: float = DEFAULT_SCORE_ACTIVE_FRAC,
) -> tuple[list[TemporalGroup], dict[str, Any]]:
    candidates: list[MotionRegion] = []
    for w in windows:
        for reg in w.regions:
            if reg.cells:
                candidates.append(reg)

    n = len(candidates)
    uf = _UF(n)
    n_links = 0
    for i in range(n):
        for j in range(i + 1, n):
            if _same_group_link(
                candidates[i],
                candidates[j],
                seed_dist=seed_dist,
                jaccard_link=jaccard_link,
            ):
                uf.union(i, j)
                n_links += 1

    buckets: dict[int, list[int]] = {}
    for i in range(n):
        buckets.setdefault(uf.find(i), []).append(i)

    frame_area = float(max(1, int(cfg.frame_w) * int(cfg.frame_h)))
    raw_groups: list[TemporalGroup] = []
    n_dropped_persistence = 0
    n_dropped_area = 0
    n_cc_splits = 0

    for members_idx in buckets.values():
        members = [candidates[i] for i in members_idx]
        freqs, window_ids, _ = _build_frequency_map(members)
        persistence = len(window_ids)
        if persistence < int(min_window_persistence):
            n_dropped_persistence += 1
            continue

        kept = [cell for cell, f in freqs.items() if f >= float(block_frequency)]
        if not kept:
            n_dropped_persistence += 1
            continue

        comps = _split_8cc(kept)
        if len(comps) > 1:
            n_cc_splits += len(comps) - 1

        for comp in comps:
            # If too many blocks, keep highest-frequency ones (cap size).
            if len(comp) > int(max_rep_blocks):
                comp = sorted(
                    comp,
                    key=lambda rc: (-freqs.get(rc, 0.0), rc[0], rc[1]),
                )[: int(max_rep_blocks)]
                # Re-split after truncation for compactness
                sub = _split_8cc(comp)
                comp = max(sub, key=len) if sub else comp

            sub_members = _members_for_cells(members, comp)
            sub_wins = sorted({int(m.window_index) for m in sub_members})
            if len(sub_wins) < int(min_window_persistence):
                # Fallback: use parent windows that touch these cells via freq map
                touch_wins = []
                for wi in window_ids:
                    # approximate: any member of this window intersects comp
                    if any(
                        int(m.window_index) == wi and (_cell_set(m) & set(comp))
                        for m in members
                    ):
                        touch_wins.append(wi)
                sub_wins = sorted(set(touch_wins)) or sub_wins
            persist = len(sub_wins)
            if persist < int(min_window_persistence):
                continue

            scores = [_region_score(m) for m in sub_members] or [
                _region_score(m) for m in members
            ]
            peak = float(max(scores)) if scores else 0.0
            mean = float(sum(scores) / max(len(scores), 1))
            bb_area = max(_bbox_block_area(comp), 1)
            compactness = float(len(comp)) / float(bb_area)
            bbox = cells_to_grid_bbox(comp, cfg)
            area = max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            area_ratio = float(area / frame_area)
            if area_ratio > float(max_area_ratio) or len(comp) > int(max_rep_blocks):
                n_dropped_area += 1
                continue

            touch_members = [
                m for m in members if _cell_set(m) & set(comp)
            ] or sub_members
            # Trim to strong members so weak long tails don't extend the ROI forever.
            strong = [
                m for m in touch_members
                if _region_score(m) >= float(score_active_frac) * peak
            ] or touch_members
            baseline_mean = _mean_baseline_for_cells(windows, strong, comp)
            gscore = _score_group(
                peak=peak,
                mean=mean,
                persistence=persist,
                compactness=compactness,
                area_ratio=area_ratio,
                persist_cap=persist_cap,
                max_area_ratio=max_area_ratio,
                baseline_mean=baseline_mean,
            )
            if gscore <= 0:
                n_dropped_area += 1
                continue

            valid_start = min(int(m.window_start_frame) for m in strong)
            valid_end = max(int(m.window_end_frame) for m in strong)
            freq_str = {f"{r},{c}": float(freqs.get((r, c), 0.0)) for r, c in comp}
            raw_groups.append(
                TemporalGroup(
                    group_id=0,
                    members=strong,
                    window_indices=sorted({int(m.window_index) for m in strong}),
                    rep_cells=sorted(comp),
                    rep_bbox=bbox,
                    mean_novelty=mean,
                    peak_novelty=peak,
                    window_persistence=int(len({int(m.window_index) for m in strong})),
                    compactness=float(compactness),
                    area_ratio=float(area_ratio),
                    group_score=float(gscore),
                    valid_start_frame=int(valid_start),
                    valid_end_frame=int(valid_end),
                    block_frequencies=freq_str,
                )
            )

    ranked = sorted(
        raw_groups,
        key=lambda g: (-g.group_score, -g.peak_novelty, -g.compactness),
    )
    n_frames_est = 0
    if windows:
        n_frames_est = max(int(w.window_end) for w in windows) + 1
    n_rows = 1
    n_cols = 1
    for w in windows:
        if w.block_score is not None:
            n_rows = int(w.block_score.shape[0])
            n_cols = int(w.block_score.shape[1])
            break
    selected = _time_bin_diverse_select(
        ranked,
        n_frames=n_frames_est,
        max_rois=int(max_rois),
        min_center_sep=float(min_center_sep),
        n_time_bins=int(n_time_bins),
        n_rows=n_rows,
        n_cols=n_cols,
    )
    for i, g in enumerate(selected, start=1):
        g.group_id = i

    stats = {
        "method": "temporal_group_compact_novelty_roi",
        "n_window_candidates": int(n),
        "n_temporal_links": int(n_links),
        "n_raw_connected_components": int(len(buckets)),
        "n_cc_splits": int(n_cc_splits),
        "n_groups_before_topk": int(len(raw_groups)),
        "n_dropped_persistence": int(n_dropped_persistence),
        "n_dropped_area": int(n_dropped_area),
        "n_final_rois": int(len(selected)),
        "min_window_persistence": int(min_window_persistence),
        "block_frequency_threshold": float(block_frequency),
        "max_rois": int(max_rois),
        "max_rep_blocks": int(max_rep_blocks),
        "max_area_ratio": float(max_area_ratio),
        "persist_cap": int(persist_cap),
        "seed_dist": float(seed_dist),
        "jaccard_link": float(jaccard_link),
        "n_time_bins": int(n_time_bins),
        "cross_window_matching": "adjacent_exact_seed_or_jaccard",
        "interpolation": "none_fixed_representative_bbox",
        "stabilization": "fixed_grid_bbox_over_strong_span",
        "selected_groups": [g.to_dict() for g in selected],
        "top_candidate_scores": [
            {
                "peak_novelty": round(float(g.peak_novelty), 4),
                "mean_novelty": round(float(g.mean_novelty), 4),
                "persist": int(g.window_persistence),
                "compactness": round(float(g.compactness), 4),
                "area_ratio": round(float(g.area_ratio), 4),
                "group_score": round(float(g.group_score), 4),
                "n_rep": int(len(g.rep_cells)),
                "span": [int(g.valid_start_frame), int(g.valid_end_frame)],
                "cells": [[int(a), int(b)] for a, b in g.rep_cells],
            }
            for g in ranked[:15]
        ],
    }
    return selected, stats


def _mean_baseline_for_cells(
    windows: Sequence[WindowResult],
    members: Sequence[MotionRegion],
    cells: Sequence[tuple[int, int]],
) -> float:
    win_map = {int(w.window_index): w for w in windows}
    vals: list[float] = []
    for m in members:
        w = win_map.get(int(m.window_index))
        if w is None or w.baseline_activity is None:
            continue
        ba = w.baseline_activity
        for r, c in cells:
            if 0 <= r < ba.shape[0] and 0 <= c < ba.shape[1]:
                vals.append(float(ba[r, c]))
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _overlap_frac(g: TemporalGroup, t0: int, t1: int) -> float:
    o0 = max(int(g.valid_start_frame), int(t0))
    o1 = min(int(g.valid_end_frame), int(t1))
    if o1 < o0:
        return 0.0
    glen = max(1, int(g.valid_end_frame) - int(g.valid_start_frame) + 1)
    return float(o1 - o0 + 1) / float(glen)


def _temporal_overlap_frac(a: TemporalGroup, b: TemporalGroup) -> float:
    """Overlap length / min(span); 0 if disjoint in time."""
    o0 = max(int(a.valid_start_frame), int(b.valid_start_frame))
    o1 = min(int(a.valid_end_frame), int(b.valid_end_frame))
    if o1 < o0:
        return 0.0
    len_a = max(1, int(a.valid_end_frame) - int(a.valid_start_frame) + 1)
    len_b = max(1, int(b.valid_end_frame) - int(b.valid_start_frame) + 1)
    return float(o1 - o0 + 1) / float(min(len_a, len_b))


def _diversity_select(
    ranked: Sequence[TemporalGroup],
    *,
    max_rois: int,
    min_center_sep: float,
) -> list[TemporalGroup]:
    """Spatially diversify only among temporally overlapping ROIs.

    Same place / different times (e.g. boarding then departure) must both survive.
    """
    selected: list[TemporalGroup] = []
    for g in ranked:
        if len(selected) >= int(max_rois):
            break
        cy, cx = _grid_center(g.rep_cells)
        ok = True
        for s in selected:
            # Disjoint in time → allow co-located events.
            if _temporal_overlap_frac(g, s) < 0.15:
                continue
            sy, sx = _grid_center(s.rep_cells)
            dist = max(abs(cy - sy), abs(cx - sx))
            if dist < float(min_center_sep):
                ok = False
                break
            inter_x = max(
                0.0,
                min(g.rep_bbox[2], s.rep_bbox[2]) - max(g.rep_bbox[0], s.rep_bbox[0]),
            )
            inter_y = max(
                0.0,
                min(g.rep_bbox[3], s.rep_bbox[3]) - max(g.rep_bbox[1], s.rep_bbox[1]),
            )
            inter = inter_x * inter_y
            area_g = max(
                1e-6, (g.rep_bbox[2] - g.rep_bbox[0]) * (g.rep_bbox[3] - g.rep_bbox[1])
            )
            area_s = max(
                1e-6, (s.rep_bbox[2] - s.rep_bbox[0]) * (s.rep_bbox[3] - s.rep_bbox[1])
            )
            if inter / min(area_g, area_s) >= 0.3:
                ok = False
                break
        if ok:
            selected.append(g)
    return selected


def _row_band(g: TemporalGroup, n_rows: int) -> str:
    cy, _ = _grid_center(g.rep_cells)
    if n_rows <= 1:
        return "mid"
    if cy < n_rows / 3.0:
        return "top"
    if cy < 2.0 * n_rows / 3.0:
        return "mid"
    return "bottom"


def _time_bin_diverse_select(
    ranked: Sequence[TemporalGroup],
    *,
    n_frames: int,
    max_rois: int,
    min_center_sep: float,
    n_time_bins: int,
    n_rows: int = 1,
    n_cols: int = 1,
) -> list[TemporalGroup]:
    """Pick late motion + first-half novelty onset (prefer top-band weak events).

    Busy mid/road blobs often outscore quiet boarding; when a top-band onset with
    non-trivial peak novelty exists in the first half, prefer it over mid-road.
    Edge-column top noise is down-weighted vs interior top novelty.
    """
    if not ranked:
        return []
    n_frames = max(int(n_frames), 1)
    n_cols = max(int(n_cols), 1)
    _ = n_time_bins  # API compatibility

    def _mid_t(g: TemporalGroup) -> float:
        return 0.5 * (float(g.valid_start_frame) + float(g.valid_end_frame))

    def _is_interior(g: TemporalGroup) -> bool:
        _, cx = _grid_center(g.rep_cells)
        cx_frac = float(cx) / float(max(n_cols - 1, 1))
        return 0.18 <= cx_frac <= 0.88

    def _onset_key(g: TemporalGroup) -> tuple[float, float, float]:
        band = _row_band(g, n_rows)
        band_w = 2.2 if band == "top" else (1.0 if band == "mid" else 0.35)
        interior = 1.35 if _is_interior(g) else 0.25
        return (
            float(g.peak_novelty) * band_w * interior,
            float(g.group_score),
            float(g.peak_novelty),
        )

    # Late: second-half non-bottom; prefer top-band when scores are close (car depart).
    late_cands = [
        g
        for g in ranked
        if _mid_t(g) >= 0.52 * n_frames and _row_band(g, n_rows) != "bottom"
    ]

    def _late_key(g: TemporalGroup) -> tuple[float, float]:
        band = _row_band(g, n_rows)
        band_w = 1.45 if band == "top" else (1.0 if band == "mid" else 0.4)
        interior = 1.2 if _is_interior(g) else 0.7
        return (float(g.group_score) * band_w * interior, float(g.peak_novelty))

    late = max(late_cands, key=_late_key) if late_cands else None

    # First half: prefer interior top-band onset if available.
    first_half = [
        g
        for g in ranked
        if _mid_t(g) < 0.55 * n_frames
        and _row_band(g, n_rows) != "bottom"
        and float(g.peak_novelty) >= 0.04
        and float(g.group_score) > 0.0
    ]
    first_top = [
        g
        for g in first_half
        if _row_band(g, n_rows) == "top" and float(g.peak_novelty) >= 0.06
    ]
    first_top_int = [g for g in first_top if _is_interior(g)]
    if first_top_int:
        onset = max(first_top_int, key=_onset_key)
    elif first_top:
        onset = max(first_top, key=_onset_key)
    elif first_half:
        onset = max(first_half, key=_onset_key)
    else:
        onset = None

    preferred: list[TemporalGroup] = []
    if late is not None and onset is not None:
        preferred = [late, onset]
    elif late is not None:
        preferred = [late]
    elif onset is not None:
        preferred = [onset]

    seen: set[int] = set()
    pool: list[TemporalGroup] = []
    for g in preferred + list(ranked):
        key = id(g)
        if key in seen:
            continue
        seen.add(key)
        pool.append(g)
    return _diversity_select(pool, max_rois=max_rois, min_center_sep=min_center_sep)


def temporal_groups_to_frames(
    groups: Sequence[TemporalGroup],
    *,
    n_frames: int,
    frame_idx_map: dict[int, int] | None = None,
) -> list[FusedFrameResult]:
    n_frames = int(n_frames)
    results: list[FusedFrameResult] = []
    for fi in range(n_frames):
        rois: list[FusedFrameROI] = []
        for g in groups:
            if not (g.valid_start_frame <= fi <= g.valid_end_frame):
                continue
            sources = [
                FusedROISource(
                    window_index=int(m.window_index),
                    region_id=int(m.region_id),
                    score=float(_region_score(m)),
                )
                for m in g.members
            ]
            rois.append(
                FusedFrameROI(
                    region_id=int(g.group_id),
                    cells=list(g.rep_cells),
                    bbox=list(g.rep_bbox),
                    score=float(g.group_score),
                    mean_score=float(g.peak_novelty),
                    sources=sources,
                )
            )
        rois.sort(key=lambda r: r.region_id)
        results.append(
            FusedFrameResult(
                frame_index=fi,
                frame_idx=(frame_idx_map or {}).get(fi),
                rois=rois,
            )
        )
    return results


def build_temporal_group_rois(
    windows: Sequence[WindowResult],
    *,
    n_frames: int,
    cfg: SlidingWindowConfig,
    frame_idx_map: dict[int, int] | None = None,
    min_window_persistence: int = DEFAULT_MIN_WINDOW_PERSISTENCE,
    block_frequency: float = DEFAULT_BLOCK_FREQUENCY,
    max_rois: int = DEFAULT_MAX_ROIS,
    seed_dist: float = DEFAULT_SEED_DIST,
    jaccard_link: float = DEFAULT_JACCARD_LINK,
    max_rep_blocks: int = DEFAULT_MAX_REP_BLOCKS,
    max_area_ratio: float = DEFAULT_MAX_AREA_RATIO,
    persist_cap: int = DEFAULT_PERSIST_CAP,
    min_center_sep: float = DEFAULT_MIN_CENTER_SEP,
    n_time_bins: int = DEFAULT_N_TIME_BINS,
    score_active_frac: float = DEFAULT_SCORE_ACTIVE_FRAC,
) -> tuple[list[FusedFrameResult], list[TemporalGroup], dict[str, Any]]:
    groups, stats = build_temporal_groups(
        windows,
        cfg=cfg,
        min_window_persistence=min_window_persistence,
        block_frequency=block_frequency,
        max_rois=max_rois,
        seed_dist=seed_dist,
        jaccard_link=jaccard_link,
        max_rep_blocks=max_rep_blocks,
        max_area_ratio=max_area_ratio,
        persist_cap=persist_cap,
        min_center_sep=min_center_sep,
        n_time_bins=n_time_bins,
        score_active_frac=score_active_frac,
    )
    frames = temporal_groups_to_frames(
        groups, n_frames=n_frames, frame_idx_map=frame_idx_map
    )
    roi_counts = [len(fr.rois) for fr in frames]
    stats.update(
        {
            "n_frames": int(n_frames),
            "mean_rois_per_frame": float(np.mean(roi_counts)) if roi_counts else 0.0,
            "max_rois_per_frame": int(max(roi_counts)) if roi_counts else 0,
            "frames_with_roi": int(sum(1 for c in roi_counts if c > 0)),
            "grid_aligned": True,
            "block_px": int(cfg.block_px),
        }
    )
    return frames, groups, stats


def save_temporal_groups_json(
    path: Path | str,
    groups: Sequence[TemporalGroup],
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    payload = {"meta": meta or {}, "groups": [g.to_dict() for g in groups]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


__all__ = [
    "TemporalGroup",
    "build_temporal_groups",
    "build_temporal_group_rois",
    "temporal_groups_to_frames",
    "save_temporal_groups_json",
    "DEFAULT_MAX_WINDOW_CANDIDATES",
    "DEFAULT_MIN_WINDOW_PERSISTENCE",
    "DEFAULT_BLOCK_FREQUENCY",
    "DEFAULT_MAX_ROIS",
]
