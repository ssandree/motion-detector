"""Approach B+: causal block_score → compact but wider novelty ROIs + context.

Shared with A: sliding-window causal novelty `block_score`.
B+ changes:
  - Seed on per-cell temporal peaks (simple)
  - Spatially grow to coherent neighbors (wider than 1×192)
  - Longer temporal spans
  - Attach role/reason for overlays (why this ROI exists)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import cv2
import numpy as np

from motion_analyzer.temporal.frame_window_fusion import (
    FusedFrameROI,
    FusedFrameResult,
    FusedROISource,
    build_sample_lookup,
    cells_to_grid_bbox,
    map_original_to_sample,
)
from motion_analyzer.temporal.sliding_window_regions import (
    SlidingWindowConfig,
    WindowResult,
)

DEFAULT_MAX_ROIS = 2
DEFAULT_MIN_PEAK = 0.05
DEFAULT_SPAN_FRAC = 0.15
DEFAULT_MIN_PERSIST = 2
DEFAULT_TOP_ONSET_PEAK = 0.05
DEFAULT_NEIGHBOR_RATIO = 0.28
DEFAULT_MAX_CELLS = 6
DEFAULT_SPAN_PAD = 2
SpatialMode = Literal["seed", "hop1", "core3"]

FPS_HINT = 30.0
SAMPLE_FIRST = 6
SAMPLE_STEP = 6


def sample_to_sec(s: int) -> float:
    return (SAMPLE_FIRST + SAMPLE_STEP * int(s)) / FPS_HINT


@dataclass
class BlockPeakEvent:
    event_id: int
    cell: tuple[int, int]  # seed
    cells: list[tuple[int, int]]
    peak_score: float
    mean_score: float
    baseline_at_peak: float
    quiet: float
    persist: int
    event_score: float
    valid_start_frame: int
    valid_end_frame: int
    peak_window_index: int
    rep_bbox: list[float]
    window_indices: list[int] = field(default_factory=list)
    role: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": int(self.event_id),
            "group_id": int(self.event_id),
            "seed_cell": [int(self.cell[0]), int(self.cell[1])],
            "rep_cells": [[int(a), int(b)] for a, b in self.cells],
            "n_rep_blocks": int(len(self.cells)),
            "rep_bbox": [float(v) for v in self.rep_bbox],
            "peak_novelty": round(float(self.peak_score), 6),
            "mean_novelty": round(float(self.mean_score), 6),
            "baseline_at_peak": round(float(self.baseline_at_peak), 6),
            "quiet": round(float(self.quiet), 6),
            "window_persistence": int(self.persist),
            "group_score": round(float(self.event_score), 6),
            "valid_start_frame": int(self.valid_start_frame),
            "valid_end_frame": int(self.valid_end_frame),
            "valid_start_sec": round(sample_to_sec(self.valid_start_frame), 2),
            "valid_end_sec": round(sample_to_sec(self.valid_end_frame), 2),
            "peak_window_index": int(self.peak_window_index),
            "window_indices": [int(v) for v in self.window_indices],
            "role": self.role,
            "reason": self.reason,
            "compactness": 1.0,
            "area_ratio": 0.0,
        }


def _row_band(row: float, n_rows: int) -> str:
    if n_rows <= 1:
        return "mid"
    if row < n_rows / 3.0:
        return "top"
    if row < 2.0 * n_rows / 3.0:
        return "mid"
    return "bottom"


def _is_interior(col: float, n_cols: int) -> bool:
    if n_cols <= 1:
        return True
    frac = float(col) / float(max(n_cols - 1, 1))
    return 0.15 <= frac <= 0.88


def _mid_t(ev: BlockPeakEvent) -> float:
    return 0.5 * (float(ev.valid_start_frame) + float(ev.valid_end_frame))


def _stack_scores(
    windows: Sequence[WindowResult],
) -> tuple[np.ndarray, np.ndarray, list[WindowResult]]:
    usable = [w for w in windows if w.block_score is not None]
    if not usable:
        raise ValueError("No windows with block_score for Approach B")
    scores = np.stack([w.block_score for w in usable], axis=0).astype(np.float64)
    if usable[0].baseline_activity is None:
        bas = np.zeros_like(scores)
    else:
        bas = np.stack([w.baseline_activity for w in usable], axis=0).astype(np.float64)
    return scores, bas, usable


def _temporal_peaks_1d(series: np.ndarray, min_peak: float) -> list[int]:
    n = int(series.shape[0])
    peaks: list[int] = []
    for i in range(n):
        v = float(series[i])
        if v < float(min_peak):
            continue
        left = float(series[i - 1]) if i > 0 else -1.0
        right = float(series[i + 1]) if i + 1 < n else -1.0
        if v >= left and v >= right:
            if i > 0 and v == left:
                continue
            peaks.append(i)
    return peaks


def _expand_span(
    series: np.ndarray,
    peak_i: int,
    *,
    frac: float,
    min_persist: int,
    pad: int = 0,
) -> tuple[int, int]:
    n = int(series.shape[0])
    peak = float(series[peak_i])
    thr = float(frac) * peak
    lo = hi = int(peak_i)
    while lo > 0 and float(series[lo - 1]) >= thr:
        lo -= 1
    while hi + 1 < n and float(series[hi + 1]) >= thr:
        hi += 1
    if (hi - lo + 1) < int(min_persist):
        lo = max(0, int(peak_i) - (int(min_persist) // 2))
        hi = min(n - 1, lo + int(min_persist) - 1)
        lo = max(0, hi - int(min_persist) + 1)
    lo = max(0, lo - int(pad))
    hi = min(n - 1, hi + int(pad))
    return lo, hi


def _grow_cells(
    *,
    seed: tuple[int, int],
    peak_map: np.ndarray,
    mode: SpatialMode,
    neighbor_ratio: float,
    max_cells: int,
) -> list[tuple[int, int]]:
    """Grow a compact ROI around seed using the peak-time score map."""
    n_rows, n_cols = peak_map.shape
    sr, sc = int(seed[0]), int(seed[1])
    seed_val = float(peak_map[sr, sc])
    if mode == "seed":
        return [(sr, sc)]

    if mode == "core3":
        cells = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = sr + dr, sc + dc
                if 0 <= rr < n_rows and 0 <= cc < n_cols:
                    cells.append((rr, cc))
        return cells[: int(max_cells)]

    # hop1: seed + neighbors with score >= ratio * seed
    thr = float(neighbor_ratio) * max(seed_val, 1e-6)
    cells = {(sr, sc)}
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = sr + dr, sc + dc
            if 0 <= rr < n_rows and 0 <= cc < n_cols and float(peak_map[rr, cc]) >= thr:
                cells.add((rr, cc))
    ranked = sorted(
        cells,
        key=lambda rc: (
            0 if rc == (sr, sc) else 1,
            -float(peak_map[rc[0], rc[1]]),
            rc[0],
            rc[1],
        ),
    )
    return ranked[: int(max_cells)]


def _span_iou(a: BlockPeakEvent, b: BlockPeakEvent) -> float:
    o0 = max(a.valid_start_frame, b.valid_start_frame)
    o1 = min(a.valid_end_frame, b.valid_end_frame)
    if o1 < o0:
        return 0.0
    inter = o1 - o0 + 1
    ua = a.valid_end_frame - a.valid_start_frame + 1
    ub = b.valid_end_frame - b.valid_start_frame + 1
    return float(inter) / float(max(ua + ub - inter, 1))


def _nms_same_seed(
    events: Sequence[BlockPeakEvent], *, iou_thr: float
) -> list[BlockPeakEvent]:
    by_cell: dict[tuple[int, int], list[BlockPeakEvent]] = {}
    for e in events:
        by_cell.setdefault(e.cell, []).append(e)
    kept: list[BlockPeakEvent] = []
    for cell_events in by_cell.values():
        cell_events = sorted(cell_events, key=lambda e: -e.event_score)
        chosen: list[BlockPeakEvent] = []
        for e in cell_events:
            if any(_span_iou(e, c) >= float(iou_thr) for c in chosen):
                continue
            chosen.append(e)
        kept.extend(chosen)
    return kept


def extract_block_peak_events(
    windows: Sequence[WindowResult],
    *,
    cfg: SlidingWindowConfig,
    min_peak: float = DEFAULT_MIN_PEAK,
    span_frac: float = DEFAULT_SPAN_FRAC,
    min_persist: int = DEFAULT_MIN_PERSIST,
    spatial_mode: SpatialMode = "hop1",
    neighbor_ratio: float = DEFAULT_NEIGHBOR_RATIO,
    max_cells: int = DEFAULT_MAX_CELLS,
    span_pad: int = DEFAULT_SPAN_PAD,
) -> list[BlockPeakEvent]:
    scores, bas, usable = _stack_scores(windows)
    n_rows, n_cols = scores.shape[1], scores.shape[2]
    events: list[BlockPeakEvent] = []

    for r in range(n_rows):
        for c in range(n_cols):
            series = scores[:, r, c]
            peaks = _temporal_peaks_1d(series, min_peak=min_peak)
            for pi in peaks:
                lo, hi = _expand_span(
                    series,
                    pi,
                    frac=span_frac,
                    min_persist=min_persist,
                    pad=span_pad,
                )
                seg = series[lo : hi + 1]
                peak = float(series[pi])
                mean = float(seg.mean()) if seg.size else peak
                ba = float(bas[pi, r, c])
                quiet = max(0.12, 1.0 - ba) ** 2.0
                persist = int(hi - lo + 1)
                score = float(peak * quiet * math.log1p(float(persist)))
                if score <= 0:
                    continue
                peak_map = scores[lo : hi + 1].mean(axis=0)
                cells = _grow_cells(
                    seed=(r, c),
                    peak_map=peak_map,
                    mode=spatial_mode,
                    neighbor_ratio=neighbor_ratio,
                    max_cells=max_cells,
                )
                bbox = cells_to_grid_bbox(cells, cfg)
                win_idx = [int(usable[i].window_index) for i in range(lo, hi + 1)]
                events.append(
                    BlockPeakEvent(
                        event_id=0,
                        cell=(int(r), int(c)),
                        cells=list(cells),
                        peak_score=peak,
                        mean_score=mean,
                        baseline_at_peak=ba,
                        quiet=quiet,
                        persist=persist,
                        event_score=score,
                        valid_start_frame=int(usable[lo].window_start),
                        valid_end_frame=int(usable[hi].window_end),
                        peak_window_index=int(usable[pi].window_index),
                        rep_bbox=[float(x) for x in bbox],
                        window_indices=win_idx,
                    )
                )

    return _nms_same_seed(events, iou_thr=0.5)


def select_top_events(
    events: Sequence[BlockPeakEvent],
    *,
    n_frames: int,
    n_rows: int,
    n_cols: int,
    max_rois: int = DEFAULT_MAX_ROIS,
    top_onset_peak: float = DEFAULT_TOP_ONSET_PEAK,
) -> list[BlockPeakEvent]:
    if not events:
        return []
    n_frames = max(int(n_frames), 1)
    n_rows = max(int(n_rows), 1)
    n_cols = max(int(n_cols), 1)
    diag = float(max(n_rows, n_cols, 1))

    def band(e: BlockPeakEvent) -> str:
        return _row_band(float(e.cell[0]), n_rows)

    def cheb(a: BlockPeakEvent, b: BlockPeakEvent) -> float:
        return float(max(abs(a.cell[0] - b.cell[0]), abs(a.cell[1] - b.cell[1])))

    non_bottom = [e for e in events if band(e) != "bottom"]
    pool = non_bottom if len(non_bottom) >= 2 else list(events)

    def g1_key(e: BlockPeakEvent) -> float:
        top_w = 1.35 if band(e) == "top" else 1.0
        late_w = 1.45 if _mid_t(e) >= 0.48 * n_frames else 0.85
        interior_w = 1.15 if _is_interior(float(e.cell[1]), n_cols) else 0.7
        return float(e.event_score) * top_w * late_w * interior_w

    g1 = max(pool, key=g1_key)
    g1.role = "late_primary" if _mid_t(g1) >= 0.48 * n_frames else "primary_novelty"
    g1.reason = (
        f"{g1.role}: peak={g1.peak_score:.2f} quiet={g1.quiet:.2f} "
        f"seed={g1.cell} n_blocks={len(g1.cells)} "
        f"t={sample_to_sec(g1.valid_start_frame):.1f}-{sample_to_sec(g1.valid_end_frame):.1f}s"
    )

    def g2_key(e: BlockPeakEvent) -> float:
        if e is g1:
            return -1.0
        cell_w = 0.2 if e.cell == g1.cell else 1.0
        sep = cheb(e, g1) / diag
        tsep = abs(_mid_t(e) - _mid_t(g1)) / float(n_frames)
        band_w = 1.4 if band(e) == "top" else 1.0
        onset_w = 1.0
        if _mid_t(g1) >= 0.48 * n_frames and _mid_t(e) < 0.55 * n_frames:
            if (
                band(e) == "top"
                and _is_interior(float(e.cell[1]), n_cols)
                and e.peak_score >= float(top_onset_peak)
            ):
                onset_w = 8.0
            elif band(e) != "bottom":
                onset_w = 1.1
        if _mid_t(g1) < 0.50 * n_frames and _mid_t(e) >= 0.45 * n_frames:
            onset_w = max(onset_w, 1.5)
        return (
            float(e.event_score)
            * band_w
            * onset_w
            * cell_w
            * (1.0 + 2.0 * sep)
            * (1.0 + 1.0 * tsep)
        )

    g2 = None
    if _mid_t(g1) >= 0.60 * n_frames:
        first_top = [
            e
            for e in pool
            if e is not g1
            and e.cell != g1.cell
            and _mid_t(e) < 0.55 * n_frames
            and band(e) == "top"
            and _is_interior(float(e.cell[1]), n_cols)
            and e.peak_score >= float(top_onset_peak)
        ]
        mid_top = [
            e
            for e in pool
            if e is not g1
            and e.cell != g1.cell
            and 0.40 * n_frames <= _mid_t(e) <= 0.70 * n_frames
            and band(e) == "top"
            and _is_interior(float(e.cell[1]), n_cols)
            and e.peak_score >= float(top_onset_peak)
            and cheb(e, g1) >= 2.0
        ]
        best_first = (
            max(first_top, key=lambda e: (e.peak_score, e.event_score))
            if first_top
            else None
        )
        best_mid = (
            max(mid_top, key=lambda e: (e.peak_score, e.event_score))
            if mid_top
            else None
        )
        if best_mid is not None and (
            best_first is None or best_mid.peak_score >= best_first.peak_score + 0.05
        ):
            g2 = best_mid
            g2.role = "mid_top_onset"
            g2.reason = (
                f"mid_top_onset (partner of late primary): peak={g2.peak_score:.2f} "
                f"seed={g2.cell} n_blocks={len(g2.cells)} "
                f"t={sample_to_sec(g2.valid_start_frame):.1f}-{sample_to_sec(g2.valid_end_frame):.1f}s"
            )
        elif best_first is not None:
            g2 = best_first
            g2.role = "early_top_onset"
            g2.reason = (
                f"early_top_onset (quiet boarding-like): peak={g2.peak_score:.2f} "
                f"seed={g2.cell} n_blocks={len(g2.cells)} "
                f"t={sample_to_sec(g2.valid_start_frame):.1f}-{sample_to_sec(g2.valid_end_frame):.1f}s"
            )

    candidates = [
        e
        for e in pool
        if e is not g1 and not (e.cell == g1.cell and _span_iou(e, g1) > 0.15)
    ]
    if g2 is None and candidates:
        g2 = max(candidates, key=g2_key)
        g2.role = "spatial_temporal_partner"
        g2.reason = (
            f"partner (space/time diversity vs {g1.cell}): peak={g2.peak_score:.2f} "
            f"seed={g2.cell} n_blocks={len(g2.cells)} "
            f"t={sample_to_sec(g2.valid_start_frame):.1f}-{sample_to_sec(g2.valid_end_frame):.1f}s"
        )

    selected: list[BlockPeakEvent] = [g1]
    if g2 is not None:
        selected.append(g2)
    selected = selected[: int(max_rois)]
    for i, e in enumerate(selected, start=1):
        e.event_id = i
    return selected


def events_to_fused_frames(
    events: Sequence[BlockPeakEvent],
    *,
    n_frames: int,
    frame_idx_map: dict[int, int] | None = None,
) -> list[FusedFrameResult]:
    n_frames = int(n_frames)
    out: list[FusedFrameResult] = []
    for fi in range(n_frames):
        rois: list[FusedFrameROI] = []
        for e in events:
            if not (e.valid_start_frame <= fi <= e.valid_end_frame):
                continue
            rois.append(
                FusedFrameROI(
                    region_id=int(e.event_id),
                    cells=list(e.cells),
                    bbox=list(e.rep_bbox),
                    score=float(e.event_score),
                    mean_score=float(e.peak_score),
                    sources=[
                        FusedROISource(
                            window_index=int(wi),
                            region_id=int(e.event_id),
                            score=float(e.peak_score),
                        )
                        for wi in e.window_indices[:8]
                    ],
                )
            )
        fidx = int(frame_idx_map.get(fi, fi)) if frame_idx_map else int(fi)
        out.append(FusedFrameResult(frame_index=fi, frame_idx=fidx, rois=rois))
    return out


def build_block_peak_rois(
    windows: Sequence[WindowResult],
    *,
    n_frames: int,
    cfg: SlidingWindowConfig,
    frame_idx_map: dict[int, int] | None = None,
    max_rois: int = DEFAULT_MAX_ROIS,
    min_peak: float = DEFAULT_MIN_PEAK,
    span_frac: float = DEFAULT_SPAN_FRAC,
    min_persist: int = DEFAULT_MIN_PERSIST,
    top_onset_peak: float = DEFAULT_TOP_ONSET_PEAK,
    spatial_mode: SpatialMode = "hop1",
    neighbor_ratio: float = DEFAULT_NEIGHBOR_RATIO,
    max_cells: int = DEFAULT_MAX_CELLS,
    span_pad: int = DEFAULT_SPAN_PAD,
) -> tuple[list[FusedFrameResult], list[BlockPeakEvent], dict[str, Any]]:
    events = extract_block_peak_events(
        windows,
        cfg=cfg,
        min_peak=min_peak,
        span_frac=span_frac,
        min_persist=min_persist,
        spatial_mode=spatial_mode,
        neighbor_ratio=neighbor_ratio,
        max_cells=max_cells,
        span_pad=span_pad,
    )
    n_rows = n_cols = 1
    for w in windows:
        if w.block_score is not None:
            n_rows = int(w.block_score.shape[0])
            n_cols = int(w.block_score.shape[1])
            break
    n_frames_est = int(n_frames)
    if windows:
        n_frames_est = max(n_frames_est, max(int(w.window_end) for w in windows) + 1)

    selected = select_top_events(
        events,
        n_frames=n_frames_est,
        n_rows=n_rows,
        n_cols=n_cols,
        max_rois=max_rois,
        top_onset_peak=top_onset_peak,
    )
    fused = events_to_fused_frames(
        selected, n_frames=n_frames, frame_idx_map=frame_idx_map
    )
    stats = {
        "method": "block_peak_roi_B_plus",
        "spatial_mode": spatial_mode,
        "neighbor_ratio": float(neighbor_ratio),
        "max_cells": int(max_cells),
        "span_frac": float(span_frac),
        "span_pad": int(span_pad),
        "n_raw_events": int(len(events)),
        "n_final_rois": int(len(selected)),
        "selected_events": [e.to_dict() for e in selected],
    }
    return fused, selected, stats


def save_block_peak_events_json(
    path: Path,
    events: Sequence[BlockPeakEvent],
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    meta = dict(meta or {})
    groups = []
    for e in events:
        g = e.to_dict()
        bb = g["rep_bbox"]
        area = max(0.0, (bb[2] - bb[0]) * (bb[3] - bb[1]))
        if "frame_area" in meta and float(meta["frame_area"]) > 0:
            g["area_ratio"] = round(area / float(meta["frame_area"]), 6)
        groups.append(g)
    path.write_text(
        json.dumps({"meta": meta, "groups": groups, "events": groups}, indent=2),
        encoding="utf-8",
    )
    return path


def render_context_roi_frame(
    frame_bgr: np.ndarray,
    *,
    events: Sequence[BlockPeakEvent],
    sample_index: int | None,
    original_frame_idx: int,
    variant_name: str = "B",
    cfg: SlidingWindowConfig | None = None,
) -> np.ndarray:
    """Draw wider ROIs + readable why-labels / legend."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    colors = [(0, 0, 255), (0, 200, 255), (80, 220, 80), (255, 180, 0)]
    active: list[BlockPeakEvent] = []
    if sample_index is not None:
        for e in events:
            if e.valid_start_frame <= sample_index <= e.valid_end_frame:
                active.append(e)

    for i, e in enumerate(active):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = [int(v) for v in e.rep_bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        if cfg is not None:
            sx1, sy1, sx2, sy2 = [int(v) for v in cells_to_grid_bbox([e.cell], cfg)]
            cv2.rectangle(out, (sx1, sy1), (sx2, sy2), color, 1)
            cx = (sx1 + sx2) // 2
            cy = (sy1 + sy2) // 2
        else:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
        cv2.drawMarker(
            out, (cx, cy), color,
            markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2,
        )
        tag = f"R{e.event_id}:{e.role or 'roi'}"
        cv2.putText(
            out, tag, (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )
        line2 = f"peak={e.peak_score:.2f} blocks={len(e.cells)} seed={e.cell}"
        cv2.putText(
            out, line2, (x1, min(h - 8, y2 + 22)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
        )

    hud = f"{variant_name}  frame={original_frame_idx}  active={len(active)}"
    cv2.putText(
        out, hud, (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )

    y0 = h - 18 - 22 * max(len(events), 1)
    cv2.rectangle(out, (0, y0 - 8), (w, h), (0, 0, 0), -1)
    cv2.putText(
        out, "WHY (selected Top-2):", (12, y0 + 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA,
    )
    for i, e in enumerate(events):
        on = (
            sample_index is not None
            and e.valid_start_frame <= sample_index <= e.valid_end_frame
        )
        col = colors[i % len(colors)] if on else (160, 160, 160)
        msg = (
            f"R{e.event_id} [{sample_to_sec(e.valid_start_frame):.1f}-"
            f"{sample_to_sec(e.valid_end_frame):.1f}s] {e.reason or e.role}"
        )
        cv2.putText(
            out, msg[:110], (12, y0 + 34 + 20 * i),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA,
        )
    return out


def write_context_overlay_video(
    *,
    video_path: Path,
    events: Sequence[BlockPeakEvent],
    frame_pairs: Sequence[tuple[int, int]],
    output_mp4: Path,
    variant_name: str = "B",
    cfg: SlidingWindowConfig | None = None,
) -> dict[str, Any]:
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
    n_written = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        orig_idx = n_written
        sample_fi = map_original_to_sample(orig_idx, sample_fis, original_idxs)
        out = render_context_roi_frame(
            frame,
            events=events,
            sample_index=sample_fi,
            original_frame_idx=orig_idx,
            variant_name=variant_name,
            cfg=cfg,
        )
        writer.write(out)
        n_written += 1
        if n_total > 0 and n_written >= n_total:
            break
    cap.release()
    writer.release()
    return {
        "overlay_mp4": str(output_mp4),
        "overlay_frames": int(n_written),
        "overlay_fps": float(fps),
        "variant": variant_name,
    }
