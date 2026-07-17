"""Linear temporal ROI from sliding-window Motion Regions.

Per window: representative Motion Regions (fixed for the window span).
Within-window IoS>=0.5 merge → cross-window association → temporal segments.

Stabilization:
  - segment canonical size = max width / max height over member windows
  - overlap: lerp centers only; never interpolate width/height
  - no forward-fill outside covering windows
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from motion_analyzer.temporal.sliding_window_regions import (
    MotionRegion,
    WindowResult,
    frame_has_global_motion,
)
from motion_analyzer.temporal.temporal_linking import bbox_iou


@dataclass
class FrameROI:
    roi_id: int
    bbox: list[float]
    source_windows: list[int]
    region_ids: list[int]
    interpolated: bool
    alpha: float
    window_index: int  # primary / HUD source window


@dataclass
class FrameROIResult:
    frame_index: int
    frame_idx: int | None
    rois: list[FrameROI] = field(default_factory=list)
    global_motion: bool = False


def _clone_region(
    r: MotionRegion,
    *,
    window_index: int | None = None,
    window_start: int | None = None,
    window_end: int | None = None,
    region_id: int | None = None,
    bbox: Sequence[float] | None = None,
    active_block_count: int | None = None,
    mean_active_frames: float | None = None,
    max_active_frames: int | None = None,
    cells: Sequence[tuple[int, int]] | None = None,
    component_score: float | None = None,
    is_global_motion: bool | None = None,
    area_ratio: float | None = None,
) -> MotionRegion:
    score = float(r.component_score if component_score is None else component_score)
    return MotionRegion(
        window_index=int(r.window_index if window_index is None else window_index),
        window_start_frame=int(
            r.window_start_frame if window_start is None else window_start
        ),
        window_end_frame=int(r.window_end_frame if window_end is None else window_end),
        region_id=int(r.region_id if region_id is None else region_id),
        bbox=list(r.bbox if bbox is None else bbox),
        active_block_count=int(
            r.active_block_count if active_block_count is None else active_block_count
        ),
        mean_active_frames=float(
            r.mean_active_frames if mean_active_frames is None else mean_active_frames
        ),
        max_active_frames=int(
            r.max_active_frames if max_active_frames is None else max_active_frames
        ),
        cells=list(r.cells if cells is None else cells),
        component_score=score,
        is_global_motion=bool(
            r.is_global_motion if is_global_motion is None else is_global_motion
        ),
        area_ratio=float(r.area_ratio if area_ratio is None else area_ratio),
        activity_novelty=float(r.activity_novelty),
        magnitude_novelty=float(r.magnitude_novelty),
        onset_score=float(r.onset_score),
        confidence=float(r.confidence),
        block_score=float(r.block_score if component_score is None else component_score),
        seed_cell=None if r.seed_cell is None else (int(r.seed_cell[0]), int(r.seed_cell[1])),
        neighbor_cells=[(int(a), int(b)) for a, b in r.neighbor_cells],
        region_strength=float(r.region_strength),
        persistence=float(r.persistence),
        bbox_block_area=int(r.bbox_block_area),
        compactness=float(r.compactness),
        size_penalty=float(r.size_penalty),
        area_penalty=float(r.area_penalty),
        roi_score=float(r.roi_score),
        final_score=float(r.final_score if component_score is None else component_score),
        old_score=float(r.old_score),
        cc_label=int(r.cc_label),
    )


def merge_motion_regions_by_ios(
    regions: Sequence[MotionRegion],
    *,
    ios_threshold: float = 0.5,
    frame_w: int | None = None,
    frame_h: int | None = None,
) -> tuple[list[MotionRegion], int]:
    """Within-window post-process: IoS >= threshold → union bbox merge.

    Returns (merged_regions, n_merge_ops).
    """
    if len(regions) <= 1:
        return list(regions), 0

    items = list(regions)
    n_merges = 0
    changed = True
    while changed and len(items) >= 2:
        changed = False
        best_pair: tuple[int, int, float] | None = None
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                ios = intersection_over_smaller(items[i].bbox, items[j].bbox)
                if ios >= float(ios_threshold):
                    if best_pair is None or ios > best_pair[2]:
                        best_pair = (i, j, ios)
        if best_pair is None:
            break
        i, j, _ = best_pair
        a, b = items[i], items[j]
        keep = a if a.region_id <= b.region_id else b
        cells = sorted(set(list(a.cells) + list(b.cells)))
        bbox = _union_bbox(a.bbox, b.bbox)
        area = _bbox_area(bbox)
        frame_area = float(max(1, int(frame_w or 0) * int(frame_h or 0)))
        area_ratio = float(area / frame_area) if (frame_w and frame_h) else 0.0
        score = max(float(a.component_score), float(b.component_score))
        merged = MotionRegion(
            window_index=keep.window_index,
            window_start_frame=keep.window_start_frame,
            window_end_frame=keep.window_end_frame,
            region_id=keep.region_id,
            bbox=bbox,
            active_block_count=len(cells) if cells else (
                int(a.active_block_count) + int(b.active_block_count)
            ),
            mean_active_frames=float(
                (a.mean_active_frames + b.mean_active_frames) / 2.0
            ),
            max_active_frames=max(int(a.max_active_frames), int(b.max_active_frames)),
            cells=cells,
            component_score=score,
            is_global_motion=False,
            area_ratio=area_ratio,
            activity_novelty=0.5 * (float(a.activity_novelty) + float(b.activity_novelty)),
            magnitude_novelty=0.5 * (
                float(a.magnitude_novelty) + float(b.magnitude_novelty)
            ),
            onset_score=0.5 * (float(a.onset_score) + float(b.onset_score)),
            confidence=max(float(a.confidence), float(b.confidence)),
            block_score=0.5 * (float(a.block_score) + float(b.block_score)),
            seed_cell=a.seed_cell if a.component_score >= b.component_score else b.seed_cell,
            neighbor_cells=sorted(
                set(list(a.neighbor_cells) + list(b.neighbor_cells))
            ),
            final_score=score,
        )
        items = [r for k, r in enumerate(items) if k != i and k != j]
        items.append(merged)
        changed = True
        n_merges += 1

    for i, r in enumerate(sorted(items, key=lambda x: x.region_id), start=1):
        r.region_id = i
    return sorted(items, key=lambda x: x.region_id), n_merges


def apply_ios_merge_within_windows(
    windows: Sequence[WindowResult],
    *,
    ios_threshold: float = 0.5,
    frame_w: int | None = None,
    frame_h: int | None = None,
    large_area_ratio: float | None = None,
) -> tuple[list[WindowResult], dict[str, Any]]:
    """Post-process: merge overlapping ROIs inside each window via IoS threshold.

    If large_area_ratio is set, merged/local ROIs that grow to >= threshold are
    moved to global_motion (not kept as ROI).
    """
    out: list[WindowResult] = []
    n_before = 0
    n_after = 0
    n_merges = 0
    windows_merged = 0
    n_promoted_global = 0
    frame_area = float(max(1, int(frame_w or 0) * int(frame_h or 0)))
    thr = float(large_area_ratio) if large_area_ratio is not None else None

    for w in windows:
        n_before += len(w.regions)
        merged, merges = merge_motion_regions_by_ios(
            w.regions,
            ios_threshold=ios_threshold,
            frame_w=frame_w,
            frame_h=frame_h,
        )
        n_merges += merges
        if merges > 0:
            windows_merged += 1

        local: list[MotionRegion] = []
        grown_global: list[MotionRegion] = []
        for i, r in enumerate(merged, start=1):
            rr = _clone_region(
                r,
                window_index=w.window_index,
                window_start=w.window_start,
                window_end=w.window_end,
                region_id=i,
            )
            if thr is not None and frame_w and frame_h:
                ar = float(rr.area / frame_area)
                rr.area_ratio = ar
                if ar >= thr:
                    rr.is_global_motion = True
                    grown_global.append(rr)
                    n_promoted_global += 1
                    continue
            rr.is_global_motion = False
            local.append(rr)

        for i, r in enumerate(local, start=1):
            r.region_id = i
        for i, r in enumerate(grown_global, start=1):
            r.region_id = i

        n_after += len(local)
        out.append(
            WindowResult(
                window_index=w.window_index,
                window_start=w.window_start,
                window_end=w.window_end,
                regions=local,
                active_count=w.active_count,
                persistent_mask=w.persistent_mask,
                n_persistent_blocks=w.n_persistent_blocks,
                global_motion_regions=list(w.global_motion_regions) + grown_global,
                score_comparison=w.score_comparison,
                window_activity=w.window_activity,
                window_mean_mag=w.window_mean_mag,
                previous_mean_mag=w.previous_mean_mag,
                baseline_activity=w.baseline_activity,
                baseline_mean_mag=w.baseline_mean_mag,
                activity_novelty=w.activity_novelty,
                magnitude_novelty=w.magnitude_novelty,
                onset_score=w.onset_score,
                activity_novelty_norm=w.activity_novelty_norm,
                magnitude_novelty_norm=w.magnitude_novelty_norm,
                onset_score_norm=w.onset_score_norm,
                block_score=w.block_score,
                confidence=w.confidence,
                past_frame_count=w.past_frame_count,
                background_motion_zone=w.background_motion_zone,
                seed_mask=w.seed_mask,
                neighbor_mask=w.neighbor_mask,
                roi_mask=w.roi_mask,
                window_stats=w.window_stats,
                block_features=w.block_features,
            )
        )
    stats = {
        "ios_merge_threshold": float(ios_threshold),
        "ios_merge_ops": int(n_merges),
        "ios_merge_windows_affected": int(windows_merged),
        "regions_before_ios_merge": int(n_before),
        "regions_after_ios_merge": int(n_after),
        "ios_merge_promoted_to_global": int(n_promoted_global),
        "large_area_ratio_threshold": thr,
    }
    return out, stats


def lerp_bbox(a: Sequence[float], b: Sequence[float], alpha: float) -> list[float]:
    t = float(np.clip(alpha, 0.0, 1.0))
    return [float(a[i]) * (1.0 - t) + float(b[i]) * t for i in range(4)]


def _bbox_area(bbox: Sequence[float]) -> float:
    return max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))


def _bbox_intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_over_smaller(a: Sequence[float], b: Sequence[float]) -> float:
    """inter(a,b) / min(area(a), area(b))."""
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


def bbox_to_cwh(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2), w, h


def cwh_to_bbox(
    cx: float,
    cy: float,
    w: float,
    h: float,
    *,
    frame_w: int | None = None,
    frame_h: int | None = None,
) -> list[float]:
    """Build bbox from center+size. Shift to stay in-frame without changing size."""
    w = float(w)
    h = float(h)
    x1 = float(cx) - 0.5 * w
    y1 = float(cy) - 0.5 * h
    if frame_w is not None and w <= float(frame_w):
        if x1 < 0.0:
            x1 = 0.0
        if x1 + w > float(frame_w):
            x1 = float(frame_w) - w
    if frame_h is not None and h <= float(frame_h):
        if y1 < 0.0:
            y1 = 0.0
        if y1 + h > float(frame_h):
            y1 = float(frame_h) - h
    x2 = x1 + w
    y2 = y1 + h
    # If canonical size exceeds frame (rare), fall back to clipped extents.
    if frame_w is not None and w > float(frame_w):
        x1, x2 = 0.0, float(frame_w)
    if frame_h is not None and h > float(frame_h):
        y1, y2 = 0.0, float(frame_h)
    return [float(x1), float(y1), float(x2), float(y2)]


def lerp_center_fixed_size(
    bbox_a: Sequence[float],
    bbox_b: Sequence[float],
    alpha: float,
    *,
    width: float,
    height: float,
    frame_w: int | None = None,
    frame_h: int | None = None,
) -> list[float]:
    """Lerp centers only; width/height stay at the canonical segment size."""
    t = float(np.clip(alpha, 0.0, 1.0))
    cxa, cya, _, _ = bbox_to_cwh(bbox_a)
    cxb, cyb, _, _ = bbox_to_cwh(bbox_b)
    cx = cxa * (1.0 - t) + cxb * t
    cy = cya * (1.0 - t) + cyb * t
    return cwh_to_bbox(cx, cy, width, height, frame_w=frame_w, frame_h=frame_h)


def merge_frame_rois_by_ios(
    rois: Sequence[FrameROI],
    *,
    ios_threshold: float = 0.5,
) -> tuple[list[FrameROI], int]:
    """Frame-level post-process: merge overlapping/nested ROIs by IoS.

    If the smaller box is mostly inside the larger (IoS >= threshold), keep the
    larger bbox only (containment). Otherwise union the boxes.
    Prefer keeping the larger ROI's id.
    """
    if len(rois) <= 1:
        return list(rois), 0
    items = [
        FrameROI(
            roi_id=r.roi_id,
            bbox=[float(v) for v in r.bbox],
            source_windows=list(r.source_windows),
            region_ids=list(r.region_ids),
            interpolated=bool(r.interpolated),
            alpha=float(r.alpha),
            window_index=int(r.window_index),
        )
        for r in rois
    ]
    n_merges = 0
    changed = True
    while changed and len(items) >= 2:
        changed = False
        best_pair: tuple[int, int, float] | None = None
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                ios = intersection_over_smaller(items[i].bbox, items[j].bbox)
                if ios >= float(ios_threshold):
                    if best_pair is None or ios > best_pair[2]:
                        best_pair = (i, j, ios)
        if best_pair is None:
            break
        i, j, ios_val = best_pair
        a, b = items[i], items[j]
        area_a, area_b = _bbox_area(a.bbox), _bbox_area(b.bbox)
        if area_a >= area_b:
            larger, smaller = a, b
        else:
            larger, smaller = b, a
        # Nested small-inside-large: keep larger only. Similar sizes: union.
        if max(area_a, area_b) >= 1.5 * min(area_a, area_b):
            bbox = list(larger.bbox)
        else:
            bbox = _union_bbox(a.bbox, b.bbox)
        merged = FrameROI(
            roi_id=int(larger.roi_id),
            bbox=bbox,
            source_windows=sorted(set(a.source_windows) | set(b.source_windows)),
            region_ids=sorted(set(a.region_ids) | set(b.region_ids)),
            interpolated=bool(a.interpolated or b.interpolated),
            alpha=float(max(a.alpha, b.alpha)),
            window_index=int(larger.window_index),
        )
        items = [r for k, r in enumerate(items) if k != i and k != j]
        items.append(merged)
        changed = True
        n_merges += 1
    items.sort(key=lambda r: r.roi_id)
    return items, n_merges


def apply_ios_merge_to_frame_rois(
    frame_rois: Sequence[FrameROIResult],
    *,
    ios_threshold: float = 0.5,
    large_area_ratio: float | None = None,
    frame_w: int | None = None,
    frame_h: int | None = None,
) -> tuple[list[FrameROIResult], dict[str, Any]]:
    """Merge spatially overlapping ROIs on each frame (post temporal linking).

    After merge, drop ROIs whose area_ratio >= large_area_ratio (if set) and
    mark the frame as global_motion instead of drawing that ROI.
    """
    out: list[FrameROIResult] = []
    total_ops = 0
    frames_affected = 0
    rois_before = 0
    rois_after = 0
    dropped_large = 0
    frame_area = float(max(1, int(frame_w or 0) * int(frame_h or 0)))
    thr = float(large_area_ratio) if large_area_ratio is not None else None

    for fr in frame_rois:
        rois_before += len(fr.rois)
        merged, ops = merge_frame_rois_by_ios(fr.rois, ios_threshold=ios_threshold)
        total_ops += ops
        if ops > 0:
            frames_affected += 1

        kept: list[FrameROI] = []
        gm = bool(fr.global_motion)
        for r in merged:
            if thr is not None and frame_w and frame_h:
                ar = float(_bbox_area(r.bbox) / frame_area)
                if ar >= thr:
                    dropped_large += 1
                    gm = True
                    continue
            kept.append(r)
        rois_after += len(kept)
        out.append(
            FrameROIResult(
                frame_index=fr.frame_index,
                frame_idx=fr.frame_idx,
                rois=kept,
                global_motion=gm,
            )
        )
    stats = {
        "frame_ios_merge_threshold": float(ios_threshold),
        "frame_ios_merge_ops": int(total_ops),
        "frame_ios_merge_frames_affected": int(frames_affected),
        "frame_rois_before_ios_merge": int(rois_before),
        "frame_rois_after_ios_merge": int(rois_after),
        "frame_rois_dropped_large": int(dropped_large),
        "large_area_ratio_threshold": thr,
    }
    return out, stats


def match_regions_by_iou(
    prev: Sequence[MotionRegion],
    nxt: Sequence[MotionRegion],
    *,
    min_iou: float = 0.0,
) -> tuple[list[tuple[MotionRegion, MotionRegion, float]], list[MotionRegion], list[MotionRegion]]:
    """Legacy greedy one-to-one IoU matching (used for before/after comparison)."""
    pairs: list[tuple[float, int, int]] = []
    for i, a in enumerate(prev):
        for j, b in enumerate(nxt):
            iou = bbox_iou(a.bbox, b.bbox)
            if iou > float(min_iou):
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    used_i: set[int] = set()
    used_j: set[int] = set()
    matched: list[tuple[MotionRegion, MotionRegion, float]] = []
    for iou, i, j in pairs:
        if i in used_i or j in used_j:
            continue
        matched.append((prev[i], nxt[j], float(iou)))
        used_i.add(i)
        used_j.add(j)
    unmatched_prev = [r for i, r in enumerate(prev) if i not in used_i]
    unmatched_next = [r for j, r in enumerate(nxt) if j not in used_j]
    return matched, unmatched_prev, unmatched_next


def _bbox_center(bbox: Sequence[float]) -> tuple[float, float]:
    return float(bbox[0] + bbox[2]) * 0.5, float(bbox[1] + bbox[3]) * 0.5


def center_distance_px(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    return float(np.hypot(ax - bx, ay - by))


@dataclass
class RegionMatch:
    region_a: MotionRegion
    region_b: MotionRegion
    iou: float
    ios: float
    center_dist: float
    window_gap: int
    primary_reason: str  # "ios" | "iou" | "center"


def match_regions_extended(
    prev: Sequence[MotionRegion],
    nxt: Sequence[MotionRegion],
    *,
    window_gap: int,
    min_iou: float = 0.10,
    min_ios: float = 0.50,
    max_center_dist: float = 192.0,
) -> list[RegionMatch]:
    """Hungarian one-to-one match. Eligible if IoU/IoS/center condition holds.

    Cost minimizes a combination that prefers higher IoS, higher IoU, closer
    center, then smaller window gap. Ineligible pairs get a large cost and are
    rejected after assignment.
    """
    from scipy.optimize import linear_sum_assignment

    if not prev or not nxt:
        return []

    n, m = len(prev), len(nxt)
    inf = 1.0e9
    cost = np.full((n, m), inf, dtype=np.float64)
    meta: list[list[tuple[float, float, float, str] | None]] = [
        [None] * m for _ in range(n)
    ]
    thr = float(max_center_dist) if float(max_center_dist) > 0 else 192.0

    for i, a in enumerate(prev):
        for j, b in enumerate(nxt):
            iou = float(bbox_iou(a.bbox, b.bbox))
            ios = float(intersection_over_smaller(a.bbox, b.bbox))
            dist = float(center_distance_px(a.bbox, b.bbox))
            ok_iou = iou >= float(min_iou)
            ok_ios = ios >= float(min_ios)
            ok_center = dist <= thr
            if not (ok_iou or ok_ios or ok_center):
                continue
            if ok_ios:
                reason = "ios"
            elif ok_iou:
                reason = "iou"
            else:
                reason = "center"
            # Minimize: prefer high IoS/IoU, low distance, low gap.
            cost[i, j] = (
                -1000.0 * ios
                - 100.0 * iou
                + (dist / thr)
                + 0.01 * float(window_gap)
            )
            meta[i][j] = (iou, ios, dist, reason)

    row_ind, col_ind = linear_sum_assignment(cost)
    matched: list[RegionMatch] = []
    for i, j in zip(row_ind.tolist(), col_ind.tolist()):
        if cost[i, j] >= inf * 0.5 or meta[i][j] is None:
            continue
        iou, ios, dist, reason = meta[i][j]  # type: ignore[misc]
        matched.append(
            RegionMatch(
                region_a=prev[i],
                region_b=nxt[j],
                iou=float(iou),
                ios=float(ios),
                center_dist=float(dist),
                window_gap=int(window_gap),
                primary_reason=str(reason),
            )
        )
    return matched


def _windows_covering(windows: Sequence[WindowResult], frame_index: int) -> list[WindowResult]:
    return [w for w in windows if w.window_start <= frame_index <= w.window_end]


@dataclass
class SegmentMember:
    window_index: int
    region_id: int
    bbox: list[float]
    window_start: int
    window_end: int


@dataclass
class ROISegment:
    segment_id: int
    members: list[SegmentMember]
    canonical_width: float
    canonical_height: float

    @property
    def duration_frames(self) -> int:
        if not self.members:
            return 0
        return int(max(m.window_end for m in self.members) - min(m.window_start for m in self.members) + 1)


def _segment_groups_from_unions(
    key_to_member: dict[tuple[int, int], SegmentMember],
    parent: dict[tuple[int, int], tuple[int, int]],
    find_fn,
) -> list[ROISegment]:
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for key in parent:
        root = find_fn(key)
        groups.setdefault(root, []).append(key)

    segments: list[ROISegment] = []
    roots = sorted(groups.keys(), key=lambda k: min(groups[k]))
    for sid, root in enumerate(roots, start=1):
        members = [key_to_member[k] for k in groups[root]]
        members.sort(key=lambda m: (m.window_start, m.window_index, m.region_id))
        widths = [bbox_to_cwh(m.bbox)[2] for m in members]
        heights = [bbox_to_cwh(m.bbox)[3] for m in members]
        segments.append(
            ROISegment(
                segment_id=sid,
                members=members,
                canonical_width=float(max(widths)) if widths else 0.0,
                canonical_height=float(max(heights)) if heights else 0.0,
            )
        )
    return segments


def _init_union_members(
    windows: Sequence[WindowResult],
) -> tuple[
    dict[tuple[int, int], tuple[int, int]],
    dict[tuple[int, int], SegmentMember],
]:
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    key_to_member: dict[tuple[int, int], SegmentMember] = {}
    for w in windows:
        for r in w.regions:
            key = (w.window_index, r.region_id)
            parent[key] = key
            key_to_member[key] = SegmentMember(
                window_index=w.window_index,
                region_id=r.region_id,
                bbox=[float(v) for v in r.bbox],
                window_start=w.window_start,
                window_end=w.window_end,
            )
    return parent, key_to_member


def _make_find(parent: dict[tuple[int, int], tuple[int, int]]):
    def _find(x: tuple[int, int]) -> tuple[int, int]:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: tuple[int, int], b: tuple[int, int]) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    return _find, _union


def _build_roi_segments_legacy_iou(
    windows: Sequence[WindowResult],
) -> list[ROISegment]:
    """Previous association: adjacent overlapping windows, IoU>0 greedy only."""
    windows = sorted(windows, key=lambda w: (w.window_start, w.window_index))
    parent, key_to_member = _init_union_members(windows)
    if not parent:
        return []
    find, union = _make_find(parent)

    for a, b in zip(windows[:-1], windows[1:]):
        if b.window_start > a.window_end:
            continue
        matched, _, _ = match_regions_by_iou(a.regions, b.regions, min_iou=0.0)
        for ra, rb, _iou in matched:
            union((a.window_index, ra.region_id), (b.window_index, rb.region_id))
    return _segment_groups_from_unions(key_to_member, parent, find)


def build_roi_segments(
    windows: Sequence[WindowResult],
    *,
    max_window_gap: int = 1,
    min_iou: float = 0.10,
    min_ios: float = 0.50,
    max_center_dist: float = 192.0,
) -> tuple[list[ROISegment], dict[str, Any]]:
    """Cross-window segment linking with IoU / IoS / center-distance rules.

    Only pairs windows with index difference in [1, max_window_gap+1]
    (gap=0 adjacent … gap=max_window_gap one skipped window).
    Does NOT run a global all-ROI connected component: edges exist only for
    those window pairs, each with Hungarian one-to-one matching.
    """
    windows = sorted(windows, key=lambda w: (w.window_start, w.window_index))
    parent, key_to_member = _init_union_members(windows)
    empty_stats = {
        "num_segments_before": 0,
        "num_segments_after": 0,
        "mean_segment_duration_frames": 0.0,
        "median_segment_duration_frames": 0.0,
        "max_segment_duration_frames": 0,
        "gap_recovered_links": 0,
        "match_by_ios": 0,
        "match_by_iou": 0,
        "match_by_center": 0,
        "match_total": 0,
        "max_window_gap": int(max_window_gap),
        "min_iou": float(min_iou),
        "min_ios": float(min_ios),
        "max_center_dist": float(max_center_dist),
    }
    if not parent:
        return [], empty_stats

    find, union = _make_find(parent)
    win_by_pos = list(windows)

    match_by_ios = 0
    match_by_iou = 0
    match_by_center = 0
    gap_recovered = 0
    match_total = 0

    # Process smaller gaps first. Edges only between window pairs with
    # index diff in [1, max_window_gap+1]; no global all-ROI CC.
    for gap in range(0, int(max_window_gap) + 1):
        delta = gap + 1
        for pos, wa in enumerate(win_by_pos):
            pos_b = pos + delta
            if pos_b >= len(win_by_pos):
                continue
            wb = win_by_pos[pos_b]
            # Only sequential window_index chain (no scrambling)
            if int(wb.window_index) - int(wa.window_index) != delta:
                continue
            matches = match_regions_extended(
                wa.regions,
                wb.regions,
                window_gap=gap,
                min_iou=min_iou,
                min_ios=min_ios,
                max_center_dist=max_center_dist,
            )
            for m in matches:
                union(
                    (wa.window_index, m.region_a.region_id),
                    (wb.window_index, m.region_b.region_id),
                )
                match_total += 1
                if m.primary_reason == "ios":
                    match_by_ios += 1
                elif m.primary_reason == "iou":
                    match_by_iou += 1
                else:
                    match_by_center += 1
                if gap >= 1:
                    gap_recovered += 1

    segments = _segment_groups_from_unions(key_to_member, parent, find)
    legacy = _build_roi_segments_legacy_iou(windows)
    durations = [s.duration_frames for s in segments]
    stats = {
        "num_segments_before": int(len(legacy)),
        "num_segments_after": int(len(segments)),
        "mean_segment_duration_frames": (
            float(np.mean(durations)) if durations else 0.0
        ),
        "median_segment_duration_frames": (
            float(np.median(durations)) if durations else 0.0
        ),
        "max_segment_duration_frames": int(max(durations) if durations else 0),
        "gap_recovered_links": int(gap_recovered),
        "match_by_ios": int(match_by_ios),
        "match_by_iou": int(match_by_iou),
        "match_by_center": int(match_by_center),
        "match_total": int(match_total),
        "max_window_gap": int(max_window_gap),
        "min_iou": float(min_iou),
        "min_ios": float(min_ios),
        "max_center_dist": float(max_center_dist),
    }
    return segments, stats


# Back-compat alias used by older call sites
def _build_roi_segments(windows: Sequence[WindowResult]) -> list[ROISegment]:
    segs, _ = build_roi_segments(windows)
    return segs


def build_linear_temporal_rois(
    windows: Sequence[WindowResult],
    *,
    n_frames: int,
    frame_idx_map: dict[int, int] | None = None,
    frame_w: int | None = None,
    frame_h: int | None = None,
    max_window_gap: int = 1,
    min_iou: float = 0.10,
    min_ios: float = 0.50,
    max_center_dist: float = 192.0,
) -> tuple[list[FrameROIResult], dict[str, Any]]:
    """Per-frame ROIs with center-only lerp and segment-canonical fixed size.

    Local ROIs are always emitted even when global_motion flag is True on the
    same frame (global is a separate flag only).
    """
    windows = sorted(windows, key=lambda w: (w.window_start, w.window_index))
    win_by_idx = {w.window_index: w for w in windows}
    segments, linking_stats = build_roi_segments(
        windows,
        max_window_gap=max_window_gap,
        min_iou=min_iou,
        min_ios=min_ios,
        max_center_dist=max_center_dist,
    )

    results: list[FrameROIResult] = []
    for fi in range(int(n_frames)):
        rois: list[FrameROI] = []
        covering = {w.window_index for w in _windows_covering(windows, fi)}

        for seg in segments:
            active = [m for m in seg.members if m.window_index in covering]
            if not active:
                continue

            active.sort(key=lambda m: (m.window_start, m.window_index))
            cw, ch = seg.canonical_width, seg.canonical_height

            if len(active) == 1:
                m = active[0]
                cx, cy, _, _ = bbox_to_cwh(m.bbox)
                bbox = cwh_to_bbox(cx, cy, cw, ch, frame_w=frame_w, frame_h=frame_h)
                rois.append(
                    FrameROI(
                        roi_id=seg.segment_id,
                        bbox=bbox,
                        source_windows=[m.window_index],
                        region_ids=[m.region_id],
                        interpolated=False,
                        alpha=0.0,
                        window_index=m.window_index,
                    )
                )
            else:
                ma, mb = active[0], active[-1]
                wa = win_by_idx[ma.window_index]
                wb = win_by_idx[mb.window_index]
                overlap_start = wb.window_start
                overlap_end = wa.window_end
                denom = max(int(overlap_end - overlap_start), 1)
                alpha = float(np.clip((fi - overlap_start) / float(denom), 0.0, 1.0))
                bbox = lerp_center_fixed_size(
                    ma.bbox,
                    mb.bbox,
                    alpha,
                    width=cw,
                    height=ch,
                    frame_w=frame_w,
                    frame_h=frame_h,
                )
                rois.append(
                    FrameROI(
                        roi_id=seg.segment_id,
                        bbox=bbox,
                        source_windows=[ma.window_index, mb.window_index],
                        region_ids=[ma.region_id, mb.region_id],
                        interpolated=True,
                        alpha=alpha,
                        window_index=ma.window_index if alpha < 0.5 else mb.window_index,
                    )
                )

        # global_motion is independent of local ROIs (never drops them).
        results.append(
            FrameROIResult(
                frame_index=fi,
                frame_idx=(frame_idx_map or {}).get(fi),
                rois=rois,
                global_motion=frame_has_global_motion(windows, fi),
            )
        )

    gm_frames = sum(1 for fr in results if fr.global_motion)
    linking_stats = {
        **linking_stats,
        "global_motion_frames": int(gm_frames),
        "global_frame_ratio": float(gm_frames / max(int(n_frames), 1)),
        "n_frames": int(n_frames),
    }
    return results, linking_stats


def save_linear_temporal_roi_json(
    path: Path | str,
    frame_rois: Sequence[FrameROIResult],
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    payload: dict[str, Any] = {
        "meta": meta or {},
        "frames": [
            {
                "frame_index": fr.frame_index,
                "frame_idx": fr.frame_idx,
                "global_motion": bool(fr.global_motion),
                "rois": [
                    {
                        "roi_id": r.roi_id,
                        "bbox": [float(v) for v in r.bbox],
                        "source_windows": list(r.source_windows),
                        "region_ids": list(r.region_ids),
                        "interpolated": bool(r.interpolated),
                        "alpha": float(r.alpha),
                        "window_index": int(r.window_index),
                    }
                    for r in fr.rois
                ],
            }
            for fr in frame_rois
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def render_linear_roi_frame(
    frame_bgr: np.ndarray,
    *,
    frame_result: FrameROIResult,
) -> np.ndarray:
    from motion_analyzer.temporal.temporal_linking import color_for_track

    out = frame_bgr.copy()
    h, w = out.shape[:2]

    # Large scene motion: label only — no ROI crop box.
    if frame_result.global_motion:
        gm_color = (180, 180, 180)
        cv2.rectangle(out, (8, 8), (w - 9, h - 9), gm_color, 2)
        cv2.putText(
            out,
            "GLOBAL_MOTION",
            (16, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            gm_color,
            2,
            cv2.LINE_AA,
        )

    for r in frame_result.rois:
        color = color_for_track(r.roi_id)
        x1, y1, x2, y2 = [int(round(v)) for v in r.bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        src = ",".join(str(w) for w in r.source_windows)
        tag = "lerp" if r.interpolated else "fixed"
        label = f"R{r.roi_id} W[{src}] {tag}"
        cv2.putText(
            out,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    sources = sorted({w for r in frame_result.rois for w in r.source_windows}) or ["-"]
    any_lerp = any(r.interpolated for r in frame_result.rois)
    hud = (
        f"fi={frame_result.frame_index} frame_idx={frame_result.frame_idx}  "
        f"rois={len(frame_result.rois)}  srcW={sources}  "
        f"interp={'Y' if any_lerp else 'N'}  "
        f"gm={'Y' if frame_result.global_motion else 'N'}"
    )
    cv2.putText(
        out,
        hud,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


@dataclass
class ROISpan:
    """Lifespan of one roi_id for timeline packing (like a track bar)."""

    roi_id: int
    start_frame_index: int
    end_frame_index: int


def build_roi_spans(frame_rois: Sequence[FrameROIResult]) -> list[ROISpan]:
    """Union lifespan per roi_id: first→last frame where the id appears."""
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for fr in frame_rois:
        for r in fr.rois:
            rid = int(r.roi_id)
            fi = int(fr.frame_index)
            if rid not in first:
                first[rid] = fi
                last[rid] = fi
            else:
                first[rid] = min(first[rid], fi)
                last[rid] = max(last[rid], fi)
    return [
        ROISpan(roi_id=rid, start_frame_index=first[rid], end_frame_index=last[rid])
        for rid in sorted(first.keys())
    ]


def assign_roi_timeline_lanes(spans: Sequence[ROISpan]) -> list[tuple[ROISpan, int]]:
    ordered = sorted(spans, key=lambda s: (s.start_frame_index, s.end_frame_index, s.roi_id))
    lane_ends: list[int] = []
    assigned: list[tuple[ROISpan, int]] = []
    for sp in ordered:
        start, end = int(sp.start_frame_index), int(sp.end_frame_index)
        lane = None
        for i, last_end in enumerate(lane_ends):
            if start > last_end:
                lane = i
                lane_ends[i] = end
                break
        if lane is None:
            lane = len(lane_ends)
            lane_ends.append(end)
        assigned.append((sp, lane))
    return assigned


def render_roi_timeline_panel(
    spans: Sequence[ROISpan],
    *,
    current_frame_index: int,
    width: int,
    t_min: int,
    t_max: int,
    panel_height: int | None = None,
    lane_height: int = 22,
    pad_x: int = 8,
    pad_y: int = 10,
    title_h: int = 22,
) -> np.ndarray:
    """DAW-like ROI timeline under the video (matches temporal_linking style)."""
    from motion_analyzer.temporal.temporal_linking import color_for_track

    assignments = assign_roi_timeline_lanes(spans)
    n_lanes = max((lane for _, lane in assignments), default=-1) + 1
    n_lanes = max(n_lanes, 1)
    if panel_height is None:
        panel_height = title_h + pad_y * 2 + n_lanes * lane_height + 4
    panel_height = max(int(panel_height), title_h + pad_y * 2 + lane_height)

    panel = np.full((panel_height, int(width), 3), 28, dtype=np.uint8)
    content_top = title_h + pad_y
    content_bot = panel_height - pad_y
    usable_h = max(content_bot - content_top, lane_height)
    row_h = usable_h / float(n_lanes)

    cv2.putText(
        panel,
        "track timeline  (x=frame_index)",
        (pad_x, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    span = max(int(t_max) - int(t_min), 1)
    inner_w = max(int(width) - 2 * pad_x, 1)

    def x_of(fi: int) -> int:
        return int(pad_x + (float(fi - t_min) / float(span)) * inner_w)

    for sp, lane in assignments:
        color = color_for_track(sp.roi_id)
        x1 = x_of(sp.start_frame_index)
        x2 = max(x_of(sp.end_frame_index + 1), x1 + 2)
        y1 = int(content_top + lane * row_h + 2)
        y2 = max(int(content_top + (lane + 1) * row_h - 2), y1 + 4)
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(panel, (x1, y1), (x2, y2), (20, 20, 20), 1)
        if x2 - x1 >= 28:
            cv2.putText(
                panel,
                f"R{sp.roi_id}",
                (x1 + 3, y1 + max(12, (y2 - y1) // 2 + 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (15, 15, 15),
                1,
                cv2.LINE_AA,
            )

    px = x_of(int(current_frame_index))
    cv2.line(panel, (px, title_h), (px, panel_height - 2), (240, 240, 240), 1)
    return panel


def compose_frame_with_roi_timeline(
    video_bgr: np.ndarray,
    spans: Sequence[ROISpan],
    *,
    current_frame_index: int,
    t_min: int,
    t_max: int,
    timeline_height: int | None = None,
) -> np.ndarray:
    h, w = video_bgr.shape[:2]
    timeline = render_roi_timeline_panel(
        spans,
        current_frame_index=current_frame_index,
        width=w,
        t_min=t_min,
        t_max=t_max,
        panel_height=timeline_height,
    )
    return np.vstack([video_bgr, timeline])
