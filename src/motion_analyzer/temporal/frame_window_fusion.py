"""Frame-level fusion of overlapping sliding-window ROI candidates.

No Hungarian matching / temporal segments / center interpolation.

For each sampled frame index fi:
  1) Collect ROI candidates from all windows covering fi
  2) Union their 192×192 block cell sets
  3) Merge ROIs whose blocks overlap or are 8-neighbor adjacent
  4) Grid-aligned union bbox; keep Top-2 by max score
  5) If final pair has IoS >= 0.5, merge to one (do not refill)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from motion_analyzer.block_motion.sub_block_aggregation import cell_pixel_bounds
from motion_analyzer.temporal.sliding_window_regions import (
    MotionRegion,
    SlidingWindowConfig,
    WindowResult,
)

DEFAULT_MAX_FRAME_ROIS = 2
DEFAULT_FRAME_IOS_MERGE = 0.5


@dataclass
class FusedROISource:
    window_index: int
    region_id: int
    score: float


@dataclass
class FusedFrameROI:
    region_id: int
    cells: list[tuple[int, int]]
    bbox: list[float]
    score: float  # max of sources
    mean_score: float
    sources: list[FusedROISource] = field(default_factory=list)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


@dataclass
class FusedFrameResult:
    frame_index: int
    frame_idx: int | None
    rois: list[FusedFrameROI] = field(default_factory=list)


def _roi_score(r: MotionRegion) -> float:
    return float(
        r.component_score
        if r.component_score
        else (r.block_score if r.block_score else r.final_score)
    )


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


def cells_touch_or_overlap(
    a: Sequence[tuple[int, int]],
    b: Sequence[tuple[int, int]],
) -> bool:
    """True if cell sets intersect or any pair is 8-neighbor adjacent."""
    sa = {(int(r), int(c)) for r, c in a}
    sb = {(int(r), int(c)) for r, c in b}
    if not sa or not sb:
        return False
    if sa & sb:
        return True
    for r, c in sa:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if (r + dr, c + dc) in sb:
                    return True
    return False


def cells_to_grid_bbox(
    cells: Sequence[tuple[int, int]],
    cfg: SlidingWindowConfig,
) -> list[float]:
    """Union bbox snapped to 192×192 grid cell pixel boundaries."""
    if not cells:
        return [0.0, 0.0, 0.0, 0.0]
    ys = [int(r) for r, _ in cells]
    xs = [int(c) for _, c in cells]
    r0, r1 = min(ys), max(ys)
    c0, c1 = min(xs), max(xs)
    x1, y1, _, _ = cell_pixel_bounds(
        r0, c0,
        factor=cfg.agg_factor,
        base_rows=cfg.base_rows,
        base_cols=cfg.base_cols,
        frame_h=cfg.frame_h,
        frame_w=cfg.frame_w,
        base_px=cfg.base_block_px,
    )
    _, _, x2, y2 = cell_pixel_bounds(
        r1, c1,
        factor=cfg.agg_factor,
        base_rows=cfg.base_rows,
        base_cols=cfg.base_cols,
        frame_h=cfg.frame_h,
        frame_w=cfg.frame_w,
        base_px=cfg.base_block_px,
    )
    return [float(x1), float(y1), float(x2), float(y2)]


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


def _merge_candidate_rois(
    candidates: Sequence[tuple[MotionRegion, float]],
    *,
    cfg: SlidingWindowConfig,
) -> list[FusedFrameROI]:
    """Merge touching ROI candidates into groups (only on selected candidates)."""
    n = len(candidates)
    if n == 0:
        return []
    uf = _UF(n)
    cell_sets = [[(int(r), int(c)) for r, c in reg.cells] for reg, _ in candidates]
    for i in range(n):
        for j in range(i + 1, n):
            if cells_touch_or_overlap(cell_sets[i], cell_sets[j]):
                uf.union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)

    fused: list[FusedFrameROI] = []
    for members in groups.values():
        cells: set[tuple[int, int]] = set()
        sources: list[FusedROISource] = []
        scores: list[float] = []
        for i in members:
            reg, sc = candidates[i]
            cells.update(cell_sets[i])
            scores.append(float(sc))
            sources.append(
                FusedROISource(
                    window_index=int(reg.window_index),
                    region_id=int(reg.region_id),
                    score=float(sc),
                )
            )
        cell_list = sorted(cells)
        fused.append(
            FusedFrameROI(
                region_id=0,
                cells=cell_list,
                bbox=cells_to_grid_bbox(cell_list, cfg),
                score=float(max(scores)) if scores else 0.0,
                mean_score=float(sum(scores) / len(scores)) if scores else 0.0,
                sources=sources,
            )
        )
    return fused


def _finalize_top_rois(
    groups: list[FusedFrameROI],
    *,
    max_rois: int,
    ios_merge: float,
    cfg: SlidingWindowConfig,
) -> list[FusedFrameROI]:
    ranked = sorted(groups, key=lambda g: (-g.score, -len(g.cells), g.bbox[0], g.bbox[1]))
    top = ranked[: max(0, int(max_rois))]
    if len(top) == 2 and _intersection_over_smaller(top[0].bbox, top[1].bbox) >= float(ios_merge):
        cells = sorted(set(top[0].cells) | set(top[1].cells))
        sources = list(top[0].sources) + list(top[1].sources)
        scores = [s.score for s in sources]
        top = [
            FusedFrameROI(
                region_id=1,
                cells=cells,
                bbox=cells_to_grid_bbox(cells, cfg),
                score=float(max(scores)) if scores else 0.0,
                mean_score=float(sum(scores) / len(scores)) if scores else 0.0,
                sources=sources,
            )
        ]
    for i, g in enumerate(top, start=1):
        g.region_id = i
    return top


def fuse_windows_to_frames(
    windows: Sequence[WindowResult],
    *,
    n_frames: int,
    cfg: SlidingWindowConfig,
    frame_idx_map: dict[int, int] | None = None,
    max_frame_rois: int = DEFAULT_MAX_FRAME_ROIS,
    ios_merge_threshold: float = DEFAULT_FRAME_IOS_MERGE,
) -> tuple[list[FusedFrameResult], dict[str, Any]]:
    """Build fused grid-aligned ROIs for every sampled frame index."""
    n_frames = int(n_frames)
    results: list[FusedFrameResult] = []
    n_groups_before = 0
    n_groups_after_topk = 0
    n_ios_merges = 0
    roi_counts: list[int] = []

    for fi in range(n_frames):
        candidates: list[tuple[MotionRegion, float]] = []
        for w in windows:
            if not (w.window_start <= fi <= w.window_end):
                continue
            for reg in w.regions:
                if not reg.cells:
                    continue
                candidates.append((reg, _roi_score(reg)))

        groups = _merge_candidate_rois(candidates, cfg=cfg)
        n_groups_before += len(groups)
        before_n = len(groups)
        final = _finalize_top_rois(
            groups,
            max_rois=int(max_frame_rois),
            ios_merge=float(ios_merge_threshold),
            cfg=cfg,
        )
        if before_n >= 2 and len(final) == 1:
            # Approximate: top-2 existed and merged
            ranked = sorted(groups, key=lambda g: -g.score)
            if len(ranked) >= 2 and _intersection_over_smaller(
                ranked[0].bbox, ranked[1].bbox
            ) >= float(ios_merge_threshold):
                n_ios_merges += 1
        n_groups_after_topk += len(final)
        roi_counts.append(len(final))
        results.append(
            FusedFrameResult(
                frame_index=fi,
                frame_idx=(frame_idx_map or {}).get(fi),
                rois=final,
            )
        )

    stats = {
        "method": "frame_window_block_union",
        "n_frames": n_frames,
        "max_frame_rois": int(max_frame_rois),
        "frame_ios_merge_threshold": float(ios_merge_threshold),
        "mean_rois_per_frame": float(np.mean(roi_counts)) if roi_counts else 0.0,
        "max_rois_per_frame": int(max(roi_counts)) if roi_counts else 0,
        "frames_with_roi": int(sum(1 for c in roi_counts if c > 0)),
        "n_groups_before_topk_total": int(n_groups_before),
        "n_rois_after_topk_total": int(n_groups_after_topk),
        "n_final_ios_merges": int(n_ios_merges),
        "cross_window_matching": "none",
        "interpolation": "none",
        "grid_aligned": True,
        "block_px": int(cfg.block_px),
    }
    return results, stats


def fused_roi_to_dict(r: FusedFrameROI) -> dict[str, Any]:
    return {
        "region_id": int(r.region_id),
        "bbox": [float(v) for v in r.bbox],
        "cells": [[int(a), int(b)] for a, b in r.cells],
        "n_blocks": int(len(r.cells)),
        "score": round(float(r.score), 6),
        "mean_score": round(float(r.mean_score), 6),
        "sources": [
            {
                "window_index": int(s.window_index),
                "region_id": int(s.region_id),
                "score": round(float(s.score), 6),
            }
            for s in r.sources
        ],
    }


def save_fused_frame_roi_json(
    path: Path | str,
    frames: Sequence[FusedFrameResult],
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    payload = {
        "meta": meta or {},
        "frames": [
            {
                "frame_index": int(fr.frame_index),
                "frame_idx": fr.frame_idx,
                "n_rois": int(len(fr.rois)),
                "rois": [fused_roi_to_dict(r) for r in fr.rois],
            }
            for fr in frames
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def build_sample_lookup(
    frame_pairs: Sequence[tuple[int, int]],
) -> tuple[list[int], list[int]]:
    """Return (sample_frame_indices, original_frame_idxs) sorted by original idx."""
    pairs = sorted(((int(fi), int(fidx)) for fi, fidx in frame_pairs), key=lambda x: x[1])
    if not pairs:
        return [], []
    return [p[0] for p in pairs], [p[1] for p in pairs]


def map_original_to_sample(
    original_frame_idx: int,
    sample_fis: Sequence[int],
    original_idxs: Sequence[int],
) -> int | None:
    """Map original video frame to nearest preceding sampled frame_index."""
    if not sample_fis:
        return None
    v = int(original_frame_idx)
    # first sample at or before v; else first sample
    lo, hi = 0, len(original_idxs) - 1
    if v < int(original_idxs[0]):
        return int(sample_fis[0])
    if v >= int(original_idxs[-1]):
        return int(sample_fis[-1])
    ans = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if int(original_idxs[mid]) <= v:
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return int(sample_fis[ans])


def render_fused_roi_frame(
    frame_bgr: np.ndarray,
    *,
    rois: Sequence[FusedFrameROI],
    original_frame_idx: int,
) -> np.ndarray:
    """Draw only final frame-level fused ROIs (no window/segment labels)."""
    out = frame_bgr.copy()
    red = (0, 0, 255)
    for r in rois:
        x1, y1, x2, y2 = [int(v) for v in r.bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), red, 3)
        label = f"R{r.region_id}"
        cv2.putText(
            out, label, (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, red, 2, cv2.LINE_AA,
        )
    hud = f"frame={original_frame_idx}  rois={len(rois)}"
    cv2.putText(
        out, hud, (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return out


def write_full_video_fused_overlay(
    *,
    video_path: Path,
    fused_by_sample: dict[int, FusedFrameResult],
    frame_pairs: Sequence[tuple[int, int]],
    output_mp4: Path,
) -> dict[str, Any]:
    """Write overlay with same frame count / fps / duration as source video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok0, probe = cap.read()
    if not ok0 or probe is None:
        cap.release()
        raise RuntimeError(f"Failed to read first frame: {video_path}")
    frame_h, frame_w = probe.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    sample_fis, original_idxs = build_sample_lookup(frame_pairs)
    writer = cv2.VideoWriter(
        str(output_mp4),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(fps, 0.1),
        (frame_w, frame_h),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter: {output_mp4}")

    n_written = 0
    for vidx in range(n_total):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[0] != frame_h or frame.shape[1] != frame_w:
            frame = cv2.resize(frame, (frame_w, frame_h), interpolation=cv2.INTER_AREA)
        si = map_original_to_sample(vidx, sample_fis, original_idxs)
        rois: list[FusedFrameROI] = []
        if si is not None and si in fused_by_sample:
            rois = fused_by_sample[si].rois
        vis = render_fused_roi_frame(frame, rois=rois, original_frame_idx=vidx)
        writer.write(vis)
        n_written += 1

    cap.release()
    writer.release()
    return {
        "overlay_mp4": str(output_mp4),
        "overlay_frames": int(n_written),
        "overlay_fps": float(fps),
        "frame_width": int(frame_w),
        "frame_height": int(frame_h),
        "source_frame_count": int(n_total),
        "overlay_matches_source_length": bool(n_written == n_total),
    }


__all__ = [
    "FusedFrameROI",
    "FusedFrameResult",
    "FusedROISource",
    "fuse_windows_to_frames",
    "save_fused_frame_roi_json",
    "write_full_video_fused_overlay",
    "render_fused_roi_frame",
    "DEFAULT_MAX_FRAME_ROIS",
    "DEFAULT_FRAME_IOS_MERGE",
]
