"""Sliding temporal window → Motion Regions via causal baseline novelty.

Per window starting at frame t:
  causal baseline from frames [0, t) only (no future leak)
  current/previous short-segment mean pooled RMS
  magnitude_novelty + onset_score → confidence-weighted block_score
  up to 2 seed ROIs (1-hop neighbors; no 8-CC; max 4 blocks)
  second seed excludes first ROI + Chebyshev distance <= 1
  overlapping pair merge if IoS>=0.3 and union area_ratio<=0.35

Farneback / 16×16 / 12×12 RMS are never recomputed; caches are reused.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from motion_analyzer.block_motion.sub_block_aggregation import cell_pixel_bounds
from motion_analyzer.temporal.temporal_linking import (
    FrameBlob,
    load_blobs_from_components_csv,
)

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SIZE = 10
DEFAULT_STRIDE = 5
DEFAULT_MIN_ACTIVE_FRAMES = 2
DEFAULT_CONNECTIVITY = 8
DEFAULT_BASE_BLOCK_PX = 16
DEFAULT_AGG_FACTOR = 12
DEFAULT_BLOCK_PX = DEFAULT_BASE_BLOCK_PX * DEFAULT_AGG_FACTOR  # 192
DEFAULT_TOP_K_REGIONS = 10  # max_window_candidates (final Top-2 after temporal groups)
DEFAULT_MAX_WINDOW_CANDIDATES = 10
DEFAULT_LARGE_AREA_RATIO = 0.60
DEFAULT_NORM_EPS = 1e-6
DEFAULT_MAG_NOVELTY_WEIGHT = 0.6
DEFAULT_ONSET_WEIGHT = 0.4
DEFAULT_WARMUP_FRAMES = 20
DEFAULT_ACTIVITY_FLOOR = 0.05
DEFAULT_NEIGHBOR_SCORE_RATIO = 0.5
DEFAULT_MAX_BLOCKS_PER_ROI = 2
DEFAULT_ROBUST_P_LOW = 5.0
DEFAULT_ROBUST_P_HIGH = 95.0
DEFAULT_ROI_MERGE_IOS = 0.3
DEFAULT_ROI_MERGE_MAX_AREA_RATIO = 0.35
DEFAULT_HIGH_BASELINE = 0.5

_EIGHT_NEIGHBORS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


@dataclass
class SlidingWindowConfig:
    window_size: int = DEFAULT_WINDOW_SIZE
    stride: int = DEFAULT_STRIDE
    min_active_frames: int = DEFAULT_MIN_ACTIVE_FRAMES
    connectivity: int = DEFAULT_CONNECTIVITY
    base_block_px: int = DEFAULT_BASE_BLOCK_PX
    agg_factor: int = DEFAULT_AGG_FACTOR
    base_rows: int = 0
    base_cols: int = 0
    frame_w: int = 1280
    frame_h: int = 720
    top_k_regions: int = DEFAULT_TOP_K_REGIONS
    large_area_ratio: float = DEFAULT_LARGE_AREA_RATIO
    norm_eps: float = DEFAULT_NORM_EPS
    mag_novelty_weight: float = DEFAULT_MAG_NOVELTY_WEIGHT
    onset_weight: float = DEFAULT_ONSET_WEIGHT
    warmup_frames: int = DEFAULT_WARMUP_FRAMES
    activity_floor: float = DEFAULT_ACTIVITY_FLOOR
    neighbor_score_ratio: float = DEFAULT_NEIGHBOR_SCORE_RATIO
    max_blocks_per_roi: int = DEFAULT_MAX_BLOCKS_PER_ROI
    robust_p_low: float = DEFAULT_ROBUST_P_LOW
    robust_p_high: float = DEFAULT_ROBUST_P_HIGH
    roi_merge_ios: float = DEFAULT_ROI_MERGE_IOS
    roi_merge_max_area_ratio: float = DEFAULT_ROI_MERGE_MAX_AREA_RATIO
    activity_weight: float = DEFAULT_MAG_NOVELTY_WEIGHT
    magnitude_weight: float = DEFAULT_ONSET_WEIGHT
    bg_baseline_activity: float = DEFAULT_HIGH_BASELINE

    @property
    def block_px(self) -> int:
        return int(self.base_block_px) * int(self.agg_factor)


@dataclass
class SpatialBaseline:
    """Snapshot of causal baseline (typically last window)."""

    baseline_activity: np.ndarray
    baseline_mean_mag: np.ndarray
    n_frames: int
    grid_rows: int
    grid_cols: int
    causal: bool = True
    note: str = "causal: past frames only; mean_mag averages ALL past frames"

    def summary_dict(self) -> dict[str, Any]:
        act = self.baseline_activity.astype(np.float64).ravel()
        mag = self.baseline_mean_mag.astype(np.float64).ravel()
        high = act >= float(DEFAULT_HIGH_BASELINE)
        return {
            "n_frames": int(self.n_frames),
            "grid_shape": [int(self.grid_rows), int(self.grid_cols)],
            "n_cells": int(act.size),
            "causal": bool(self.causal),
            "note": self.note,
            "baseline_activity_distribution": {
                "mean": float(act.mean()) if act.size else 0.0,
                "std": float(act.std()) if act.size else 0.0,
                "min": float(act.min()) if act.size else 0.0,
                "p25": float(np.percentile(act, 25)) if act.size else 0.0,
                "p50": float(np.percentile(act, 50)) if act.size else 0.0,
                "p75": float(np.percentile(act, 75)) if act.size else 0.0,
                "p90": float(np.percentile(act, 90)) if act.size else 0.0,
                "p95": float(np.percentile(act, 95)) if act.size else 0.0,
                "max": float(act.max()) if act.size else 0.0,
                "frac_ge_0.5": float(high.mean()) if act.size else 0.0,
                "n_ge_0.5": int(high.sum()),
                "histogram_edges": [0.0, 0.1, 0.25, 0.5, 0.75, 1.01],
                "histogram_counts": [
                    int(c)
                    for c in np.histogram(act, bins=[0.0, 0.1, 0.25, 0.5, 0.75, 1.01])[0]
                ],
            },
            "baseline_mean_mag_distribution": {
                "mean": float(mag.mean()) if mag.size else 0.0,
                "std": float(mag.std()) if mag.size else 0.0,
                "min": float(mag.min()) if mag.size else 0.0,
                "p50": float(np.percentile(mag, 50)) if mag.size else 0.0,
                "p90": float(np.percentile(mag, 90)) if mag.size else 0.0,
                "max": float(mag.max()) if mag.size else 0.0,
            },
        }


@dataclass
class MotionRegion:
    window_index: int
    window_start_frame: int
    window_end_frame: int
    region_id: int
    bbox: list[float]
    active_block_count: int
    mean_active_frames: float
    max_active_frames: int
    cells: list[tuple[int, int]] = field(default_factory=list)
    component_score: float = 0.0
    is_global_motion: bool = False
    area_ratio: float = 0.0
    activity_novelty: float = 0.0
    magnitude_novelty: float = 0.0
    onset_score: float = 0.0
    confidence: float = 1.0
    block_score: float = 0.0
    seed_cell: tuple[int, int] | None = None
    neighbor_cells: list[tuple[int, int]] = field(default_factory=list)
    region_strength: float = 0.0
    persistence: float = 0.0
    bbox_block_area: int = 0
    compactness: float = 0.0
    size_penalty: float = 1.0
    area_penalty: float = 1.0
    roi_score: float = 0.0
    final_score: float = 0.0
    old_score: float = 0.0
    cc_label: int = 0

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

    @property
    def window_start(self) -> int:
        return self.window_start_frame

    @property
    def window_end(self) -> int:
        return self.window_end_frame

    @property
    def union_bbox(self) -> list[float]:
        return self.bbox


@dataclass
class WindowResult:
    window_index: int
    window_start: int
    window_end: int
    regions: list[MotionRegion]
    active_count: np.ndarray
    persistent_mask: np.ndarray
    n_persistent_blocks: int
    global_motion_regions: list[MotionRegion] = field(default_factory=list)
    score_comparison: dict[str, Any] = field(default_factory=dict)
    window_activity: np.ndarray | None = None
    window_mean_mag: np.ndarray | None = None
    previous_mean_mag: np.ndarray | None = None
    baseline_activity: np.ndarray | None = None
    baseline_mean_mag: np.ndarray | None = None
    activity_novelty: np.ndarray | None = None
    magnitude_novelty: np.ndarray | None = None
    onset_score: np.ndarray | None = None
    activity_novelty_norm: np.ndarray | None = None
    magnitude_novelty_norm: np.ndarray | None = None
    onset_score_norm: np.ndarray | None = None
    block_score: np.ndarray | None = None
    confidence: float = 1.0
    past_frame_count: int = 0
    background_motion_zone: np.ndarray | None = None
    seed_mask: np.ndarray | None = None
    neighbor_mask: np.ndarray | None = None
    roi_mask: np.ndarray | None = None
    window_stats: dict[str, Any] = field(default_factory=dict)
    block_features: list[dict[str, Any]] = field(default_factory=list)


def load_active_mask_stack(agg_dir: Path | str) -> np.ndarray:
    path = Path(agg_dir) / "active_mask.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing active_mask.npy: {path}")
    mask = np.load(path)
    if mask.ndim != 3:
        raise ValueError(f"active_mask must be TxHxW, got {mask.shape}")
    return (mask > 0).astype(np.uint8)


def load_pooled_rms_stack(agg_dir: Path | str) -> np.ndarray:
    path = Path(agg_dir) / "aggregated_rms_mag.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing aggregated_rms_mag.npy: {path}")
    rms = np.load(path)
    if rms.ndim != 3:
        raise ValueError(f"aggregated_rms_mag must be TxHxW, got {rms.shape}")
    return rms.astype(np.float64)


def load_frame_thresholds(agg_dir: Path | str) -> np.ndarray:
    path = Path(agg_dir) / "aggregation_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing aggregation_metadata.json: {path}")
    meta = json.loads(path.read_text(encoding="utf-8"))
    frames = meta.get("frames") or []
    if not frames:
        raise ValueError(f"No frames[] with threshold in {path}")
    return np.asarray([float(fr["threshold"]) for fr in frames], dtype=np.float64)


def compute_spatial_baseline(
    active_stack: np.ndarray,
    pooled_rms: np.ndarray,
    *,
    n_frames: int | None = None,
) -> SpatialBaseline:
    """Legacy full-video snapshot (not used for scoring)."""
    t = int(n_frames if n_frames is not None else active_stack.shape[0])
    active = active_stack[:t].astype(bool)
    rms = pooled_rms[:t].astype(np.float64)
    return SpatialBaseline(
        baseline_activity=active.mean(axis=0).astype(np.float64),
        baseline_mean_mag=rms.mean(axis=0).astype(np.float64),
        n_frames=t,
        grid_rows=int(active.shape[1]),
        grid_cols=int(active.shape[2]),
        causal=False,
        note="legacy full-video mean (not used for scoring)",
    )


def save_spatial_baseline(
    output_dir: Path | str,
    baseline: SpatialBaseline,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    act_path = output_dir / "spatial_baseline_activity.npy"
    mag_path = output_dir / "spatial_baseline_mean_mag.npy"
    sum_path = output_dir / "spatial_baseline_summary.json"
    np.save(act_path, baseline.baseline_activity.astype(np.float32))
    np.save(mag_path, baseline.baseline_mean_mag.astype(np.float32))
    sum_path.write_text(json.dumps(baseline.summary_dict(), indent=2), encoding="utf-8")
    return {
        "spatial_baseline_activity_npy": act_path,
        "spatial_baseline_mean_mag_npy": mag_path,
        "spatial_baseline_summary_json": sum_path,
    }


def active_mask_from_components(
    blobs: Sequence[FrameBlob],
    *,
    n_frames: int,
    grid_rows: int,
    grid_cols: int,
    components_csv: Path | str | None = None,
) -> np.ndarray:
    mask = np.zeros((n_frames, grid_rows, grid_cols), dtype=np.uint8)
    if components_csv is not None and Path(components_csv).is_file():
        with Path(components_csv).open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                fi = int(row["frame_index"])
                r0, r1 = int(row["grid_r0"]), int(row["grid_r1"])
                c0, c1 = int(row["grid_c0"]), int(row["grid_c1"])
                mask[fi, r0 : r1 + 1, c0 : c1 + 1] = 1
        return mask
    raise ValueError("components.csv with grid_r0..grid_c1 required for fallback mask")


def iter_window_starts(n_frames: int, window_size: int, stride: int) -> list[int]:
    if n_frames <= 0:
        return []
    return [s for s in range(0, n_frames, int(stride)) if s < n_frames]


def _component_bbox(
    ys: np.ndarray,
    xs: np.ndarray,
    cfg: SlidingWindowConfig,
) -> list[float]:
    r0, r1 = int(ys.min()), int(ys.max())
    c0, c1 = int(xs.min()), int(xs.max())
    x1, y1, _, _ = cell_pixel_bounds(
        r0, c0,
        factor=cfg.agg_factor, base_rows=cfg.base_rows, base_cols=cfg.base_cols,
        frame_h=cfg.frame_h, frame_w=cfg.frame_w, base_px=cfg.base_block_px,
    )
    _, _, x2, y2 = cell_pixel_bounds(
        r1, c1,
        factor=cfg.agg_factor, base_rows=cfg.base_rows, base_cols=cfg.base_cols,
        frame_h=cfg.frame_h, frame_w=cfg.frame_w, base_px=cfg.base_block_px,
    )
    return [float(x1), float(y1), float(x2), float(y2)]


def _bbox_area(bbox: Sequence[float]) -> float:
    return max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))


def _bbox_intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection_over_smaller(a: Sequence[float], b: Sequence[float]) -> float:
    inter = _bbox_intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    smaller = min(_bbox_area(a), _bbox_area(b))
    if smaller <= 0.0:
        return 0.0
    return float(inter / smaller)


def _union_bbox(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [
        float(min(a[0], b[0])),
        float(min(a[1], b[1])),
        float(max(a[2], b[2])),
        float(max(a[3], b[3])),
    ]


def _robust_normalize(
    arr: np.ndarray,
    *,
    lo: float,
    hi: float,
    eps: float,
) -> np.ndarray:
    denom = max(float(hi) - float(lo), float(eps))
    return np.clip((arr.astype(np.float64) - float(lo)) / denom, 0.0, 1.0)


def _percentile_bounds(
    values: np.ndarray,
    *,
    p_low: float,
    p_high: float,
) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 1.0
    pos = values[values > 0]
    sample = pos if pos.size >= 8 else values
    lo = float(np.percentile(sample, float(p_low)))
    hi = float(np.percentile(sample, float(p_high)))
    if hi <= lo:
        hi = float(sample.max()) if sample.size else 1.0
        lo = float(sample.min()) if sample.size else 0.0
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _causal_baseline_at(
    active_stack: np.ndarray,
    pooled_rms: np.ndarray,
    t: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Baseline from frames [0, t). mean_mag averages ALL past frames."""
    gh, gw = int(active_stack.shape[1]), int(active_stack.shape[2])
    past = int(max(0, t))
    if past <= 0:
        z = np.zeros((gh, gw), dtype=np.float64)
        return z, z.copy(), 0
    act = active_stack[:past].astype(bool)
    rms = pooled_rms[:past].astype(np.float64)
    return act.mean(axis=0).astype(np.float64), rms.mean(axis=0).astype(np.float64), past


def _segment_mean_mag(
    pooled_rms: np.ndarray,
    start: int,
    end_inclusive: int,
) -> np.ndarray:
    gh, gw = int(pooled_rms.shape[1]), int(pooled_rms.shape[2])
    if end_inclusive < start or start >= int(pooled_rms.shape[0]):
        return np.zeros((gh, gw), dtype=np.float64)
    s = max(0, int(start))
    e = min(int(pooled_rms.shape[0]) - 1, int(end_inclusive))
    if e < s:
        return np.zeros((gh, gw), dtype=np.float64)
    return pooled_rms[s : e + 1].astype(np.float64).mean(axis=0)


def _grow_seed_roi(
    seed_r: int,
    seed_c: int,
    block_score: np.ndarray,
    forbidden: set[tuple[int, int]],
    assigned: set[tuple[int, int]],
    *,
    neighbor_ratio: float,
    max_blocks: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    gh, gw = block_score.shape
    seed_score = float(block_score[seed_r, seed_c])
    cells: list[tuple[int, int]] = [(seed_r, seed_c)]
    assigned.add((seed_r, seed_c))
    cand: list[tuple[float, int, int]] = []
    for dr, dc in _EIGHT_NEIGHBORS:
        nr, nc = seed_r + dr, seed_c + dc
        if nr < 0 or nc < 0 or nr >= gh or nc >= gw:
            continue
        if (nr, nc) in assigned or (nr, nc) in forbidden:
            continue
        nscore = float(block_score[nr, nc])
        if nscore >= float(neighbor_ratio) * seed_score:
            cand.append((-nscore, nr, nc))
    cand.sort()
    neighbors: list[tuple[int, int]] = []
    for _, nr, nc in cand:
        if len(cells) >= int(max_blocks):
            break
        cells.append((nr, nc))
        neighbors.append((nr, nc))
        assigned.add((nr, nc))
    return cells, neighbors


def _expand_exclusion(
    cells: Sequence[tuple[int, int]],
    *,
    gh: int,
    gw: int,
    radius: int = 1,
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < gh and 0 <= nc < gw:
                    out.add((nr, nc))
    return out


def _select_seed_rois(
    block_score: np.ndarray,
    *,
    top_k: int,
    neighbor_ratio: float,
    max_blocks: int,
) -> list[tuple[tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]]]:
    valid = block_score > 0.0
    if not np.any(valid):
        return []
    ys, xs = np.where(valid)
    order = np.argsort(-block_score[ys, xs], kind="stable")
    assigned: set[tuple[int, int]] = set()
    forbidden: set[tuple[int, int]] = set()
    rois: list[tuple[tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]]] = []
    for idx in order.tolist():
        if len(rois) >= int(top_k):
            break
        r, c = int(ys[idx]), int(xs[idx])
        if (r, c) in assigned or (r, c) in forbidden:
            continue
        cells, neighbors = _grow_seed_roi(
            r, c, block_score, forbidden, assigned,
            neighbor_ratio=neighbor_ratio,
            max_blocks=max_blocks,
        )
        rois.append(((r, c), cells, neighbors))
        # Keep scoring candidates for later temporal grouping.
        # No soft neighborhood exclusion (that was for early Top-2).
    return rois


def _maybe_merge_rois(
    regions: list[MotionRegion],
    *,
    cfg: SlidingWindowConfig,
) -> list[MotionRegion]:
    if len(regions) != 2:
        return regions
    a, b = regions[0], regions[1]
    ios = _intersection_over_smaller(a.bbox, b.bbox)
    if ios < float(cfg.roi_merge_ios):
        return regions
    ub = _union_bbox(a.bbox, b.bbox)
    frame_area = float(max(1, int(cfg.frame_w) * int(cfg.frame_h)))
    area_ratio = float(_bbox_area(ub) / frame_area)
    if area_ratio > float(cfg.roi_merge_max_area_ratio):
        return regions
    keep = a if a.component_score >= b.component_score else b
    cells = sorted(set(list(a.cells) + list(b.cells)))
    neighbors = sorted(set(list(a.neighbor_cells) + list(b.neighbor_cells)))
    ys = np.asarray([r for r, _ in cells], dtype=np.int32)
    xs = np.asarray([c for _, c in cells], dtype=np.int32)
    r0, r1 = int(ys.min()), int(ys.max())
    c0, c1 = int(xs.min()), int(xs.max())
    return [
        MotionRegion(
            window_index=keep.window_index,
            window_start_frame=keep.window_start_frame,
            window_end_frame=keep.window_end_frame,
            region_id=1,
            bbox=ub,
            active_block_count=int(len(cells)),
            mean_active_frames=0.5 * (a.mean_active_frames + b.mean_active_frames),
            max_active_frames=max(a.max_active_frames, b.max_active_frames),
            cells=cells,
            component_score=max(a.component_score, b.component_score),
            is_global_motion=False,
            area_ratio=area_ratio,
            activity_novelty=0.5 * (a.activity_novelty + b.activity_novelty),
            magnitude_novelty=0.5 * (a.magnitude_novelty + b.magnitude_novelty),
            onset_score=0.5 * (a.onset_score + b.onset_score),
            confidence=keep.confidence,
            block_score=0.5 * (a.block_score + b.block_score),
            seed_cell=keep.seed_cell,
            neighbor_cells=neighbors,
            bbox_block_area=int((r1 - r0 + 1) * (c1 - c0 + 1)),
            compactness=float(len(cells)) / float(max((r1 - r0 + 1) * (c1 - c0 + 1), 1)),
            final_score=max(a.final_score, b.final_score),
            old_score=a.old_score + b.old_score,
            cc_label=1,
        )
    ]


def _region_from_cells(
    *,
    window_index: int,
    window_start: int,
    window_end: int,
    region_id: int,
    cells: list[tuple[int, int]],
    seed: tuple[int, int],
    neighbors: list[tuple[int, int]],
    active_count: np.ndarray,
    mag_nov: np.ndarray,
    onset: np.ndarray,
    block_score: np.ndarray,
    confidence: float,
    cfg: SlidingWindowConfig,
) -> MotionRegion:
    ys = np.asarray([r for r, _ in cells], dtype=np.int32)
    xs = np.asarray([c for _, c in cells], dtype=np.int32)
    counts = active_count[ys, xs]
    bbox = _component_bbox(ys, xs, cfg)
    frame_area = float(max(1, int(cfg.frame_w) * int(cfg.frame_h)))
    mean_mag_nov = float(mag_nov[ys, xs].mean()) if len(cells) else 0.0
    mean_onset = float(onset[ys, xs].mean()) if len(cells) else 0.0
    seed_score = float(block_score[seed[0], seed[1]])
    mean_score = float(block_score[ys, xs].mean()) if len(cells) else 0.0
    r0, r1 = int(ys.min()), int(ys.max())
    c0, c1 = int(xs.min()), int(xs.max())
    bbox_block_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
    return MotionRegion(
        window_index=window_index,
        window_start_frame=window_start,
        window_end_frame=window_end,
        region_id=region_id,
        bbox=bbox,
        active_block_count=int(len(cells)),
        mean_active_frames=float(counts.mean()) if counts.size else 0.0,
        max_active_frames=int(counts.max()) if counts.size else 0,
        cells=list(cells),
        component_score=seed_score,
        is_global_motion=False,
        area_ratio=float(_bbox_area(bbox) / frame_area),
        activity_novelty=mean_onset,
        magnitude_novelty=mean_mag_nov,
        onset_score=mean_onset,
        confidence=float(confidence),
        block_score=mean_score,
        seed_cell=(int(seed[0]), int(seed[1])),
        neighbor_cells=[(int(r), int(c)) for r, c in neighbors],
        region_strength=mean_mag_nov,
        persistence=mean_onset,
        bbox_block_area=bbox_block_area,
        compactness=float(len(cells)) / float(max(bbox_block_area, 1)),
        roi_score=mean_score,
        final_score=seed_score,
        old_score=float(counts.sum()) if counts.size else 0.0,
        cc_label=int(region_id),
    )


def _region_feature_dict(r: MotionRegion) -> dict[str, Any]:
    return {
        "region_id": int(r.region_id),
        "window_index": int(r.window_index),
        "window_start_frame": int(r.window_start_frame),
        "window_end_frame": int(r.window_end_frame),
        "bbox": [float(v) for v in r.bbox],
        "active_block_count": int(r.active_block_count),
        "magnitude_novelty": round(float(r.magnitude_novelty), 6),
        "onset_score": round(float(r.onset_score), 6),
        "confidence": round(float(r.confidence), 6),
        "block_score": round(float(r.block_score), 6),
        "component_score": round(float(r.component_score), 6),
        "final_score": round(float(r.final_score), 6),
        "seed_cell": list(r.seed_cell) if r.seed_cell is not None else None,
        "neighbor_cells": [list(c) for c in r.neighbor_cells],
        "cells": [list(c) for c in r.cells],
        "area_ratio": round(float(r.area_ratio), 6),
        "is_global_motion": bool(r.is_global_motion),
    }


def _block_feature_rows(
    *,
    window_index: int,
    window_start: int,
    window_end: int,
    past_count: int,
    confidence: float,
    phase: str,
    ba: np.ndarray,
    bm: np.ndarray,
    cur: np.ndarray,
    prev: np.ndarray,
    mag_nov: np.ndarray,
    onset: np.ndarray,
    block_score: np.ndarray,
    seed_mask: np.ndarray,
    roi_mask: np.ndarray,
    top_n: int = 8,
) -> list[dict[str, Any]]:
    gh, gw = block_score.shape
    selected: set[tuple[int, int]] = set()
    ys_roi, xs_roi = np.where(roi_mask.astype(bool))
    for r, c in zip(ys_roi.tolist(), xs_roi.tolist()):
        selected.add((int(r), int(c)))
    flat = block_score.ravel()
    if flat.size:
        for idx in np.argsort(-flat)[: max(int(top_n), 1)].tolist():
            selected.add((int(idx // gw), int(idx % gw)))
    rows: list[dict[str, Any]] = []
    for r, c in sorted(selected):
        rows.append(
            {
                "window_index": int(window_index),
                "window_start_frame": int(window_start),
                "window_end_frame": int(window_end),
                "past_frame_count": int(past_count),
                "confidence": round(float(confidence), 6),
                "phase": phase,
                "row": int(r),
                "col": int(c),
                "baseline_activity": round(float(ba[r, c]), 6),
                "baseline_mean_mag": round(float(bm[r, c]), 6),
                "current_mean_mag": round(float(cur[r, c]), 6),
                "previous_mean_mag": round(float(prev[r, c]), 6),
                "magnitude_novelty": round(float(mag_nov[r, c]), 6),
                "onset_score": round(float(onset[r, c]), 6),
                "block_score": round(float(block_score[r, c]), 6),
                "is_seed": int(bool(seed_mask[r, c])),
                "is_roi": int(bool(roi_mask[r, c])),
            }
        )
    return rows


def _collect_causal_features(
    active_stack: np.ndarray,
    pooled_rms: np.ndarray,
    cfg: SlidingWindowConfig,
    n_frames: int,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    starts = iter_window_starts(n_frames, cfg.window_size, cfg.stride)
    packs: list[dict[str, Any]] = []
    mag_vals: list[np.ndarray] = []
    onset_vals: list[np.ndarray] = []
    warmup = int(cfg.warmup_frames)

    for wi, start in enumerate(starts):
        end = min(start + int(cfg.window_size) - 1, n_frames - 1)
        seg_len = int(end - start + 1)
        ba, bm, past = _causal_baseline_at(active_stack, pooled_rms, start)
        cur = _segment_mean_mag(pooled_rms, start, end)
        prev_start = start - seg_len
        if start <= 0:
            prev = np.zeros_like(cur)
        elif prev_start < 0:
            prev = _segment_mean_mag(pooled_rms, 0, start - 1)
        else:
            prev = _segment_mean_mag(pooled_rms, prev_start, start - 1)

        mag_nov = np.maximum(cur - bm, 0.0).astype(np.float64)
        onset = np.maximum(cur - prev, 0.0).astype(np.float64)
        confidence = float(min(past / float(max(warmup, 1)), 1.0))
        phase = "warmup" if past < warmup else "after_warmup"

        win_mask = active_stack[start : end + 1]
        active_count = win_mask.astype(np.int32).sum(axis=0).astype(np.int32)
        win_act = active_count.astype(np.float64) / float(max(seg_len, 1))

        packs.append(
            {
                "window_index": wi,
                "window_start": start,
                "window_end": end,
                "past_frame_count": past,
                "confidence": confidence,
                "phase": phase,
                "baseline_activity": ba,
                "baseline_mean_mag": bm,
                "current_mean_mag": cur,
                "previous_mean_mag": prev,
                "magnitude_novelty": mag_nov,
                "onset_score": onset,
                "active_count": active_count,
                "window_activity": win_act,
            }
        )
        mag_vals.append(mag_nov.ravel())
        onset_vals.append(onset.ravel())

    mag_all = np.concatenate(mag_vals) if mag_vals else np.zeros(0, dtype=np.float64)
    onset_all = (
        np.concatenate(onset_vals) if onset_vals else np.zeros(0, dtype=np.float64)
    )
    return packs, mag_all, onset_all


def process_window_novelty(
    pack: dict[str, Any],
    *,
    cfg: SlidingWindowConfig,
    mag_lo: float,
    mag_hi: float,
    onset_lo: float,
    onset_hi: float,
) -> WindowResult:
    wi = int(pack["window_index"])
    start = int(pack["window_start"])
    end = int(pack["window_end"])
    past = int(pack["past_frame_count"])
    confidence = float(pack["confidence"])
    phase = str(pack["phase"])
    ba = pack["baseline_activity"]
    bm = pack["baseline_mean_mag"]
    cur = pack["current_mean_mag"]
    prev = pack["previous_mean_mag"]
    mag_nov = pack["magnitude_novelty"]
    onset = pack["onset_score"]
    active_count = pack["active_count"]
    win_act = pack["window_activity"]

    mag_norm = _robust_normalize(mag_nov, lo=mag_lo, hi=mag_hi, eps=float(cfg.norm_eps))
    onset_norm = _robust_normalize(
        onset, lo=onset_lo, hi=onset_hi, eps=float(cfg.norm_eps)
    )
    activity_gate = np.maximum(float(cfg.activity_floor), 1.0 - ba) ** 2
    block_score = (
        confidence
        * (
            float(cfg.mag_novelty_weight) * mag_norm
            + float(cfg.onset_weight) * onset_norm
        )
        * activity_gate
    ).astype(np.float64)

    high_bl = ba >= float(cfg.bg_baseline_activity)
    bg_zone = high_bl & (block_score < 0.05)

    persistent = (active_count >= int(cfg.min_active_frames)).astype(np.uint8)
    selected = _select_seed_rois(
        block_score,
        top_k=int(cfg.top_k_regions),
        neighbor_ratio=float(cfg.neighbor_score_ratio),
        max_blocks=int(cfg.max_blocks_per_roi),
    )
    gh, gw = block_score.shape
    seed_mask = np.zeros((gh, gw), dtype=np.uint8)
    neighbor_mask = np.zeros((gh, gw), dtype=np.uint8)
    roi_mask = np.zeros((gh, gw), dtype=np.uint8)
    regions: list[MotionRegion] = []
    for i, (seed, cells, neighbors) in enumerate(selected, start=1):
        seed_mask[seed] = 1
        for r, c in neighbors:
            neighbor_mask[r, c] = 1
        for r, c in cells:
            roi_mask[r, c] = 1
        regions.append(
            _region_from_cells(
                window_index=wi,
                window_start=start,
                window_end=end,
                region_id=i,
                cells=cells,
                seed=seed,
                neighbors=neighbors,
                active_count=active_count,
                mag_nov=mag_nov,
                onset=onset,
                block_score=block_score,
                confidence=confidence,
                cfg=cfg,
            )
        )
    # No window-level Top-2 / IoS merge — temporal grouping selects max 2 later.

    block_features = _block_feature_rows(
        window_index=wi,
        window_start=start,
        window_end=end,
        past_count=past,
        confidence=confidence,
        phase=phase,
        ba=ba,
        bm=bm,
        cur=cur,
        prev=prev,
        mag_nov=mag_nov,
        onset=onset,
        block_score=block_score,
        seed_mask=seed_mask,
        roi_mask=roi_mask,
    )

    window_stats = {
        "window_index": wi,
        "window_start_frame": start,
        "window_end_frame": end,
        "past_frame_count": past,
        "confidence": round(float(confidence), 6),
        "phase": phase,
        "n_background_motion_zone_blocks": int(bg_zone.sum()),
        "frac_background_motion_zone_blocks": float(bg_zone.mean()),
        "n_high_baseline_blocks": int(high_bl.sum()),
        "n_positive_score_blocks": int((block_score > 0).sum()),
        "n_final_rois": int(len(regions)),
        "rois": [_region_feature_dict(r) for r in regions],
        "block_features": block_features,
    }

    return WindowResult(
        window_index=wi,
        window_start=start,
        window_end=end,
        regions=regions,
        active_count=active_count,
        persistent_mask=persistent,
        n_persistent_blocks=int(persistent.sum()),
        global_motion_regions=[],
        score_comparison={},
        window_activity=win_act,
        window_mean_mag=cur,
        previous_mean_mag=prev,
        baseline_activity=ba,
        baseline_mean_mag=bm,
        activity_novelty=onset,
        magnitude_novelty=mag_nov,
        onset_score=onset,
        activity_novelty_norm=onset_norm,
        magnitude_novelty_norm=mag_norm,
        onset_score_norm=onset_norm,
        block_score=block_score,
        confidence=confidence,
        past_frame_count=past,
        background_motion_zone=bg_zone.astype(np.uint8),
        seed_mask=seed_mask,
        neighbor_mask=neighbor_mask,
        roi_mask=roi_mask,
        window_stats=window_stats,
        block_features=block_features,
    )


def build_motion_regions(
    active_stack: np.ndarray,
    cfg: SlidingWindowConfig | None = None,
    *,
    n_frames: int | None = None,
    pooled_rms: np.ndarray | None = None,
    frame_thresholds: np.ndarray | None = None,
    baseline: SpatialBaseline | None = None,
) -> tuple[list[WindowResult], dict[str, Any], SpatialBaseline]:
    del frame_thresholds
    cfg = cfg or SlidingWindowConfig()
    if n_frames is None:
        n_frames = int(active_stack.shape[0])
    n_frames = int(n_frames)
    if pooled_rms is None:
        raise ValueError("pooled_rms is required for novelty scoring")
    if int(pooled_rms.shape[0]) < n_frames:
        raise ValueError(
            f"rms length mismatch: rms={pooled_rms.shape[0]} n_frames={n_frames}"
        )

    packs, mag_all, onset_all = _collect_causal_features(
        active_stack, pooled_rms, cfg, n_frames
    )
    mag_lo, mag_hi = _percentile_bounds(
        mag_all, p_low=cfg.robust_p_low, p_high=cfg.robust_p_high
    )
    onset_lo, onset_hi = _percentile_bounds(
        onset_all, p_low=cfg.robust_p_low, p_high=cfg.robust_p_high
    )

    windows: list[WindowResult] = []
    all_regions: list[MotionRegion] = []
    window_stats_list: list[dict[str, Any]] = []
    all_block_features: list[dict[str, Any]] = []
    roi_counts: list[int] = []
    confidences: list[float] = []

    for pack in packs:
        result = process_window_novelty(
            pack,
            cfg=cfg,
            mag_lo=mag_lo,
            mag_hi=mag_hi,
            onset_lo=onset_lo,
            onset_hi=onset_hi,
        )
        windows.append(result)
        all_regions.extend(result.regions)
        window_stats_list.append(result.window_stats)
        all_block_features.extend(result.block_features)
        roi_counts.append(int(result.window_stats["n_final_rois"]))
        confidences.append(float(result.confidence))

    if windows and windows[-1].baseline_activity is not None:
        snap = SpatialBaseline(
            baseline_activity=windows[-1].baseline_activity.copy(),
            baseline_mean_mag=windows[-1].baseline_mean_mag.copy(),
            n_frames=int(windows[-1].past_frame_count),
            grid_rows=int(windows[-1].baseline_activity.shape[0]),
            grid_cols=int(windows[-1].baseline_activity.shape[1]),
            causal=True,
            note="causal baseline of last window (frames 0..t-1; mean_mag=all past)",
        )
    elif baseline is not None:
        snap = baseline
    else:
        snap = compute_spatial_baseline(active_stack, pooled_rms, n_frames=n_frames)

    frame_area = float(cfg.frame_w * cfg.frame_h)
    areas = [r.area for r in all_regions]
    area_ratios = [a / frame_area for a in areas] if frame_area > 0 else []
    bl_sum = snap.summary_dict()
    warmup_rows = [r for r in all_block_features if r.get("phase") == "warmup"]
    after = [r for r in all_block_features if r.get("phase") == "after_warmup"]
    if after:
        by_win: dict[int, float] = {}
        for r in after:
            wi = int(r["window_index"])
            by_win[wi] = max(by_win.get(wi, 0.0), float(r["onset_score"]))
        top_onset_wins = {
            wi for wi, _ in sorted(by_win.items(), key=lambda kv: -kv[1])[:8]
        }
        onset_trace = [r for r in after if int(r["window_index"]) in top_onset_wins]
    else:
        onset_trace = []

    stats = {
        "num_windows": int(len(windows)),
        "mean_regions_per_window": float(np.mean(roi_counts)) if roi_counts else 0.0,
        "mean_global_motion_per_window": 0.0,
        "windows_with_global_motion": 0,
        "total_regions": int(len(all_regions)),
        "total_global_motion_regions": 0,
        "mean_bbox_area_ratio": float(np.mean(area_ratios)) if area_ratios else 0.0,
        "mean_union_bbox_area": float(np.mean(areas)) if areas else 0.0,
        "top_k_regions": int(cfg.top_k_regions),
        "scoring": (
            "causal baseline; block_score=confidence*"
            "(0.6*norm(mag_novelty)+0.4*norm(onset))*max(0.2,1-baseline_activity)"
        ),
        "warmup_frames": int(cfg.warmup_frames),
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "n_warmup_windows": int(
            sum(1 for w in windows if w.past_frame_count < cfg.warmup_frames)
        ),
        "novelty_normalization": {
            "magnitude_novelty_p_low": float(mag_lo),
            "magnitude_novelty_p_high": float(mag_hi),
            "onset_score_p_low": float(onset_lo),
            "onset_score_p_high": float(onset_hi),
            "robust_p_low": float(cfg.robust_p_low),
            "robust_p_high": float(cfg.robust_p_high),
        },
        "baseline_activity_distribution": bl_sum["baseline_activity_distribution"],
        "baseline_mean_mag_distribution": bl_sum["baseline_mean_mag_distribution"],
        "rois_per_window": {
            "mean": float(np.mean(roi_counts)) if roi_counts else 0.0,
            "max": int(max(roi_counts)) if roi_counts else 0,
            "histogram": {
                str(int(k)): int(v)
                for k, v in zip(*np.unique(roi_counts or [0], return_counts=True))
            },
        },
        "roi_features": [_region_feature_dict(r) for r in all_regions],
        "block_features": all_block_features,
        "per_window_stats": [
            {k: v for k, v in ws.items() if k != "block_features"}
            for ws in window_stats_list
        ],
        "score_trace_warmup_rows": warmup_rows,
        "score_trace_onset_rows": onset_trace,
        "n_frames": n_frames,
        "grid_shape": [int(active_stack.shape[1]), int(active_stack.shape[2])],
        "method": "causal_baseline_onset_seed_roi",
        "config": asdict(cfg),
    }
    return windows, stats, snap


def save_score_trace_csv(
    output_dir: Path | str,
    stats: dict[str, Any],
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "score_trace_warmup_and_onset.csv"
    rows = list(stats.get("score_trace_warmup_rows") or []) + list(
        stats.get("score_trace_onset_rows") or []
    )
    if not rows:
        rows = list(stats.get("block_features") or [])
    fieldnames = [
        "window_index", "window_start_frame", "window_end_frame",
        "past_frame_count", "confidence", "phase",
        "row", "col",
        "baseline_activity", "baseline_mean_mag",
        "current_mean_mag", "previous_mean_mag",
        "magnitude_novelty", "onset_score", "block_score",
        "is_seed", "is_roi",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _region_row(r: MotionRegion) -> dict[str, Any]:
    x1, y1, x2, y2 = r.bbox
    return {
        "window_index": r.window_index,
        "window_start_frame": r.window_start_frame,
        "window_end_frame": r.window_end_frame,
        "region_id": r.region_id,
        "bbox": f"[{x1:g},{y1:g},{x2:g},{y2:g}]",
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "active_block_count": r.active_block_count,
        "mean_active_frames": round(r.mean_active_frames, 4),
        "max_active_frames": r.max_active_frames,
        "magnitude_novelty": round(float(r.magnitude_novelty), 6),
        "onset_score": round(float(r.onset_score), 6),
        "confidence": round(float(r.confidence), 6),
        "block_score": round(float(r.block_score), 6),
        "component_score": round(float(r.component_score), 6),
        "final_score": round(float(r.final_score), 6),
        "seed_r": int(r.seed_cell[0]) if r.seed_cell else "",
        "seed_c": int(r.seed_cell[1]) if r.seed_cell else "",
        "n_neighbor_blocks": int(len(r.neighbor_cells)),
        "area_ratio": round(float(r.area_ratio), 6),
        "is_global_motion": int(bool(r.is_global_motion)),
    }


def _region_json(r: MotionRegion) -> dict[str, Any]:
    d = _region_feature_dict(r)
    d["mean_active_frames"] = r.mean_active_frames
    d["max_active_frames"] = r.max_active_frames
    return d


def regions_to_rows(windows: Sequence[WindowResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for w in windows:
        for r in w.regions:
            rows.append(_region_row(r))
        for r in w.global_motion_regions:
            rows.append(_region_row(r))
    return rows


def windows_to_json(windows: Sequence[WindowResult]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for w in windows:
        out.append(
            {
                "window_index": w.window_index,
                "window_start_frame": w.window_start,
                "window_end_frame": w.window_end,
                "past_frame_count": w.past_frame_count,
                "confidence": w.confidence,
                "n_persistent_blocks": w.n_persistent_blocks,
                "has_global_motion": bool(w.global_motion_regions),
                "window_stats": {
                    k: v for k, v in w.window_stats.items() if k != "block_features"
                },
                "block_features": w.block_features,
                "regions": [_region_json(r) for r in w.regions],
                "global_motion_regions": [
                    _region_json(r) for r in w.global_motion_regions
                ],
            }
        )
    return out


def frame_has_global_motion(
    windows: Sequence[WindowResult],
    frame_index: int,
) -> bool:
    fi = int(frame_index)
    for w in windows:
        if w.window_start <= fi <= w.window_end and w.global_motion_regions:
            return True
    return False


def select_covering_window(
    windows: Sequence[WindowResult],
    frame_index: int,
) -> WindowResult | None:
    fi = int(frame_index)
    covering = [w for w in windows if w.window_start <= fi <= w.window_end]
    if not covering:
        return None
    return min(
        covering,
        key=lambda w: abs(0.5 * (w.window_start + w.window_end) - fi),
    )


def save_motion_region_outputs(
    output_dir: Path | str,
    windows: Sequence[WindowResult],
    summary: dict[str, Any],
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "motion_regions.csv"
    json_path = output_dir / "motion_regions.json"
    summary_path = output_dir / "sliding_window_summary.json"
    rows = regions_to_rows(windows)
    fieldnames = [
        "window_index", "window_start_frame", "window_end_frame", "region_id",
        "bbox", "x1", "y1", "x2", "y2",
        "active_block_count", "mean_active_frames", "max_active_frames",
        "magnitude_novelty", "onset_score", "confidence", "block_score",
        "component_score", "final_score",
        "seed_r", "seed_c", "n_neighbor_blocks",
        "area_ratio", "is_global_motion",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    json_path.write_text(json.dumps(windows_to_json(windows), indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    trace_path = save_score_trace_csv(output_dir, summary)
    return {
        "motion_regions_csv": csv_path,
        "motion_regions_json": json_path,
        "sliding_window_summary": summary_path,
        "score_trace_warmup_and_onset_csv": trace_path,
    }


def paint_block_mask(
    frame_bgr: np.ndarray,
    mask_hw: np.ndarray,
    *,
    cfg: SlidingWindowConfig,
    color_bgr: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    out = frame_bgr.copy()
    if mask_hw is None or mask_hw.size == 0 or not np.any(mask_hw):
        return out
    overlay = out.copy()
    ys, xs = np.where(mask_hw.astype(bool))
    for r, c in zip(ys.tolist(), xs.tolist()):
        x1, y1, x2, y2 = cell_pixel_bounds(
            int(r), int(c),
            factor=cfg.agg_factor,
            base_rows=cfg.base_rows,
            base_cols=cfg.base_cols,
            frame_h=cfg.frame_h,
            frame_w=cfg.frame_w,
            base_px=cfg.base_block_px,
        )
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)
    return cv2.addWeighted(overlay, float(alpha), out, 1.0 - float(alpha), 0.0)


def render_novelty_debug_frame(
    frame_bgr: np.ndarray,
    *,
    window: WindowResult | None,
    baseline: SpatialBaseline | None,
    regions: Sequence[MotionRegion],
    frame_blobs: Sequence[FrameBlob],
    cfg: SlidingWindowConfig,
    frame_index: int,
) -> np.ndarray:
    out = frame_bgr.copy()
    if window is not None and window.baseline_activity is not None:
        high_bl = window.baseline_activity >= float(cfg.bg_baseline_activity)
        out = paint_block_mask(
            out, high_bl.astype(np.uint8), cfg=cfg,
            color_bgr=(80, 80, 80), alpha=0.22,
        )
    elif baseline is not None:
        high_bl = baseline.baseline_activity >= float(cfg.bg_baseline_activity)
        out = paint_block_mask(
            out, high_bl.astype(np.uint8), cfg=cfg,
            color_bgr=(80, 80, 80), alpha=0.22,
        )
    if window is not None:
        if window.background_motion_zone is not None:
            out = paint_block_mask(
                out, window.background_motion_zone, cfg=cfg,
                color_bgr=(160, 160, 40), alpha=0.30,
            )
        if window.neighbor_mask is not None:
            out = paint_block_mask(
                out, window.neighbor_mask, cfg=cfg,
                color_bgr=(255, 180, 0), alpha=0.40,
            )
        if window.seed_mask is not None:
            out = paint_block_mask(
                out, window.seed_mask, cfg=cfg,
                color_bgr=(0, 220, 0), alpha=0.45,
            )
        bg_n = (
            int(np.sum(window.background_motion_zone))
            if window.background_motion_zone is not None
            else 0
        )
        win_label = (
            f"win={window.window_index} [{window.window_start},{window.window_end}] "
            f"conf={window.confidence:.2f} rois={len(window.regions)} bg={bg_n}"
        )
    else:
        win_label = "win=none"

    yellow = (0, 255, 255)
    if frame_blobs:
        for b in frame_blobs:
            cv2.rectangle(
                out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), yellow, 1
            )

    red = (0, 0, 255)
    for r in regions:
        x1, y1, x2, y2 = [int(v) for v in r.bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), red, 3)
        label = f"R{r.region_id} s={r.block_score:.2f}"
        cv2.putText(
            out, label, (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2, cv2.LINE_AA,
        )

    hud = (
        f"fi={frame_index}  {win_label}  "
        f"gray=high-baseline  green=seed  cyan=nbr  red=ROI"
    )
    cv2.putText(
        out, hud, (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return out


def render_sliding_window_frame(
    frame_bgr: np.ndarray,
    *,
    frame_active_mask: np.ndarray | None,
    frame_blobs: Sequence[FrameBlob],
    regions: Sequence[MotionRegion],
    persistent_mask: np.ndarray,
    window_index: int,
    window_start: int,
    window_end: int,
    cfg: SlidingWindowConfig,
    background_motion_zone: np.ndarray | None = None,
    seed_mask: np.ndarray | None = None,
    neighbor_mask: np.ndarray | None = None,
    baseline_high_mask: np.ndarray | None = None,
) -> np.ndarray:
    out = frame_bgr.copy()
    if baseline_high_mask is not None:
        out = paint_block_mask(
            out, baseline_high_mask, cfg=cfg, color_bgr=(80, 80, 80), alpha=0.22
        )
    if background_motion_zone is not None:
        out = paint_block_mask(
            out, background_motion_zone, cfg=cfg, color_bgr=(160, 160, 40), alpha=0.30
        )
    elif persistent_mask is not None:
        out = paint_block_mask(
            out, persistent_mask, cfg=cfg, color_bgr=(0, 140, 255), alpha=0.28
        )
    if neighbor_mask is not None:
        out = paint_block_mask(
            out, neighbor_mask, cfg=cfg, color_bgr=(255, 180, 0), alpha=0.40
        )
    if seed_mask is not None:
        out = paint_block_mask(
            out, seed_mask, cfg=cfg, color_bgr=(0, 220, 0), alpha=0.45
        )

    yellow = (0, 255, 255)
    if frame_blobs:
        for b in frame_blobs:
            cv2.rectangle(
                out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), yellow, 1
            )
    elif frame_active_mask is not None and np.any(frame_active_mask):
        ys, xs = np.where(frame_active_mask.astype(bool))
        for r, c in zip(ys.tolist(), xs.tolist()):
            x1, y1, x2, y2 = cell_pixel_bounds(
                int(r), int(c),
                factor=cfg.agg_factor,
                base_rows=cfg.base_rows,
                base_cols=cfg.base_cols,
                frame_h=cfg.frame_h,
                frame_w=cfg.frame_w,
                base_px=cfg.base_block_px,
            )
            cv2.rectangle(out, (x1, y1), (x2, y2), yellow, 1)

    red = (0, 0, 255)
    for r in regions:
        x1, y1, x2, y2 = [int(v) for v in r.bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), red, 3)
        label = f"R{r.region_id} n={r.active_block_count}"
        cv2.putText(
            out, label, (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, red, 2, cv2.LINE_AA,
        )
    hud = (
        f"win={window_index}  frames=[{window_start},{window_end}]  "
        f"regions={len(regions)}"
    )
    cv2.putText(
        out, hud, (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return out


__all__ = [
    "SlidingWindowConfig",
    "SpatialBaseline",
    "MotionRegion",
    "WindowResult",
    "build_motion_regions",
    "compute_spatial_baseline",
    "save_spatial_baseline",
    "save_score_trace_csv",
    "load_active_mask_stack",
    "load_pooled_rms_stack",
    "load_frame_thresholds",
    "load_blobs_from_components_csv",
    "save_motion_region_outputs",
    "render_sliding_window_frame",
    "render_novelty_debug_frame",
    "select_covering_window",
    "frame_has_global_motion",
    "FrameBlob",
    "DEFAULT_TOP_K_REGIONS",
    "DEFAULT_LARGE_AREA_RATIO",
    "DEFAULT_WARMUP_FRAMES",
]
