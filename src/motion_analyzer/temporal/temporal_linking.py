"""Temporal linking of frame-level motion blobs (max_gap=1).

Reuses existing frame-level 8-CC components (e.g. factor=12 / 192×192) without
recomputing Farneback, MAD threshold, or connected components.

Pipeline:
  frame-level blobs → association (spatial + optional motion assist)
  → greedy one-to-one matching → track lifecycle → min_track_length filter
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-8
DEFAULT_AGG_FACTOR = 12
DEFAULT_PIXEL_SIZE = 192
DEFAULT_MAX_GAP = 1
DEFAULT_MIN_TRACK_LENGTH = 2
DEFAULT_CENTER_DISTANCE_THRESHOLD = 1.5
DEFAULT_MOTION_DIFFERENCE_THRESHOLD = 0.85
DEFAULT_EXPAND_PROP = 0.5
DEFAULT_TRAIL_LENGTH = 12


@dataclass
class TemporalLinkingConfig:
    """Association / lifecycle / filter knobs (all overridable from CLI)."""

    max_gap: int = DEFAULT_MAX_GAP
    min_track_length: int = DEFAULT_MIN_TRACK_LENGTH
    center_distance_threshold: float = DEFAULT_CENTER_DISTANCE_THRESHOLD
    motion_difference_threshold: float = DEFAULT_MOTION_DIFFERENCE_THRESHOLD
    expand_margin_px: float = float(DEFAULT_PIXEL_SIZE)
    expand_prop: float = DEFAULT_EXPAND_PROP
    block_px: int = DEFAULT_PIXEL_SIZE
    # Higher matching score is better (greedy descending).
    w_iou: float = 1.0
    w_center: float = 1.0
    w_gap: float = 0.35
    w_motion: float = 0.15
    trail_length: int = DEFAULT_TRAIL_LENGTH

    @property
    def max_frame_gap(self) -> int:
        """Allowed frame_index delta: 1 (adjacent) or 2 (one miss)."""
        return int(self.max_gap) + 1


@dataclass
class FrameBlob:
    """One frame-level component reused from components.csv / equivalent."""

    frame_index: int
    frame_idx: int
    blob_id: int
    bbox: list[float]  # [x1, y1, x2, y2] pixel coords
    center: tuple[float, float]
    motion_score: float
    timestamp_sec: float = 0.0
    active_block_count: int | None = None
    peak_score: float | None = None
    factor: int | None = None
    pixel_size: int | None = None
    video_name: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def x1(self) -> float:
        return float(self.bbox[0])

    @property
    def y1(self) -> float:
        return float(self.bbox[1])

    @property
    def x2(self) -> float:
        return float(self.bbox[2])

    @property
    def y2(self) -> float:
        return float(self.bbox[3])

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def diagonal(self) -> float:
        return float(math.hypot(self.width, self.height))


@dataclass
class TrackObservation:
    blob: FrameBlob
    frame_gap_from_previous: int | None  # None for first detection


@dataclass
class Trajectory:
    track_id: int
    observations: list[TrackObservation] = field(default_factory=list)
    terminated: bool = False

    @property
    def last_blob(self) -> FrameBlob:
        return self.observations[-1].blob

    @property
    def last_frame_index(self) -> int:
        return int(self.last_blob.frame_index)

    @property
    def detected_frame_count(self) -> int:
        return len(self.observations)

    @property
    def start_frame_index(self) -> int:
        return int(self.observations[0].blob.frame_index)

    @property
    def end_frame_index(self) -> int:
        return int(self.observations[-1].blob.frame_index)

    @property
    def start_frame_idx(self) -> int:
        return int(self.observations[0].blob.frame_idx)

    @property
    def end_frame_idx(self) -> int:
        return int(self.observations[-1].blob.frame_idx)

    def gap1_link_count(self) -> int:
        return sum(
            1
            for obs in self.observations
            if obs.frame_gap_from_previous is not None and obs.frame_gap_from_previous == 2
        )


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= EPS:
        return 0.0
    return float(inter / denom)


def expand_bbox(
    bbox: Sequence[float],
    *,
    frame_w: int,
    frame_h: int,
    margin_px: float,
    expand_prop: float,
) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    mx = max(float(margin_px), float(expand_prop) * w)
    my = max(float(margin_px), float(expand_prop) * h)
    return [
        max(0.0, x1 - mx),
        max(0.0, y1 - my),
        min(float(frame_w), x2 + mx),
        min(float(frame_h), y2 + my),
    ]


def bboxes_overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def normalized_center_distance(prev: FrameBlob, cur: FrameBlob, block_px: float) -> float:
    dx = float(cur.center[0] - prev.center[0])
    dy = float(cur.center[1] - prev.center[1])
    dist = math.hypot(dx, dy)
    denom = max(prev.diagonal, cur.diagonal, float(block_px), EPS)
    return float(dist / denom)


def normalized_motion_difference(prev_score: float, cur_score: float) -> float:
    a = float(prev_score)
    b = float(cur_score)
    return float(abs(b - a) / (max(a, b) + EPS))


def spatial_match(
    prev: FrameBlob,
    cur: FrameBlob,
    cfg: TemporalLinkingConfig,
    *,
    frame_w: int,
    frame_h: int,
) -> tuple[bool, dict[str, Any]]:
    iou = bbox_iou(prev.bbox, cur.bbox)
    exp_prev = expand_bbox(
        prev.bbox,
        frame_w=frame_w,
        frame_h=frame_h,
        margin_px=cfg.expand_margin_px,
        expand_prop=cfg.expand_prop,
    )
    exp_cur = expand_bbox(
        cur.bbox,
        frame_w=frame_w,
        frame_h=frame_h,
        margin_px=cfg.expand_margin_px,
        expand_prop=cfg.expand_prop,
    )
    expanded_overlap = bboxes_overlap(exp_prev, exp_cur)
    ncd = normalized_center_distance(prev, cur, cfg.block_px)
    center_ok = ncd <= float(cfg.center_distance_threshold)
    ok = bool(iou > 0.0 or expanded_overlap or center_ok)
    info = {
        "iou": iou,
        "expanded_overlap": expanded_overlap,
        "normalized_center_distance": ncd,
        "center_ok": center_ok,
    }
    return ok, info


def association_allowed(
    prev: FrameBlob,
    cur: FrameBlob,
    cfg: TemporalLinkingConfig,
    *,
    frame_w: int,
    frame_h: int,
) -> tuple[bool, dict[str, Any]]:
    frame_gap = int(cur.frame_index) - int(prev.frame_index)
    if frame_gap < 1 or frame_gap > cfg.max_frame_gap:
        return False, {"frame_gap": frame_gap, "reason": "frame_gap"}

    ok_spatial, spatial_info = spatial_match(
        prev, cur, cfg, frame_w=frame_w, frame_h=frame_h
    )
    mot_diff = normalized_motion_difference(prev.motion_score, cur.motion_score)
    # Motion is assistive only: never the sole reject reason.
    info = {
        "frame_gap": frame_gap,
        "normalized_motion_difference": mot_diff,
        **spatial_info,
    }
    if not ok_spatial:
        info["reason"] = "spatial"
        return False, info
    info["reason"] = "ok"
    info["motion_similar"] = mot_diff <= float(cfg.motion_difference_threshold)
    return True, info


def matching_score(
    prev: FrameBlob,
    cur: FrameBlob,
    info: dict[str, Any],
    cfg: TemporalLinkingConfig,
) -> float:
    iou = float(info.get("iou", 0.0))
    ncd = float(info.get("normalized_center_distance", 1e9))
    frame_gap = int(info.get("frame_gap", cfg.max_frame_gap))
    mot_diff = float(info.get("normalized_motion_difference", 1.0))

    center_term = 1.0 - min(ncd / max(float(cfg.center_distance_threshold), EPS), 1.0)
    # Prefer gap=1 over gap=2.
    gap_term = 1.0 - (frame_gap - 1) / max(float(cfg.max_gap), 1.0)
    motion_term = 1.0 - min(mot_diff, 1.0)

    return float(
        cfg.w_iou * iou
        + cfg.w_center * center_term
        + cfg.w_gap * gap_term
        + cfg.w_motion * motion_term
    )


def _blob_from_component_row(row: dict[str, Any]) -> FrameBlob:
    x1 = float(row["x1"])
    y1 = float(row["y1"])
    x2 = float(row["x2"])
    y2 = float(row["y2"])
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    active = row.get("active_block_count")
    peak = row.get("peak_score")
    factor = row.get("factor")
    pixel_size = row.get("pixel_size")
    return FrameBlob(
        frame_index=int(row["frame_index"]),
        frame_idx=int(row["frame_idx"]),
        blob_id=int(row["blob_id"]),
        bbox=[x1, y1, x2, y2],
        center=(cx, cy),
        motion_score=float(row.get("mean_score", row.get("motion_score", 0.0))),
        timestamp_sec=float(row.get("timestamp_sec", 0.0)),
        active_block_count=int(active) if active not in (None, "") else None,
        peak_score=float(peak) if peak not in (None, "") else None,
        factor=int(factor) if factor not in (None, "") else None,
        pixel_size=int(pixel_size) if pixel_size not in (None, "") else None,
        video_name=str(row.get("video_name")) if row.get("video_name") else None,
        extras={
            k: row[k]
            for k in ("threshold", "median", "mad")
            if k in row and row[k] not in (None, "")
        },
    )


def load_blobs_from_components_csv(
    path: Path | str,
    *,
    factor: int = DEFAULT_AGG_FACTOR,
    pixel_size: int | None = DEFAULT_PIXEL_SIZE,
) -> list[FrameBlob]:
    """Load frame-level blobs from a block_aggregation_experiment components.csv."""
    path = Path(path)
    blobs: list[FrameBlob] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if factor is not None and int(row["factor"]) != int(factor):
                continue
            if pixel_size is not None and int(row["pixel_size"]) != int(pixel_size):
                continue
            blobs.append(_blob_from_component_row(row))
    blobs.sort(key=lambda b: (b.frame_index, b.blob_id))
    return blobs


def group_blobs_by_frame(blobs: Sequence[FrameBlob]) -> dict[int, list[FrameBlob]]:
    by_frame: dict[int, list[FrameBlob]] = {}
    for b in blobs:
        by_frame.setdefault(int(b.frame_index), []).append(b)
    for fi in by_frame:
        by_frame[fi].sort(key=lambda x: x.blob_id)
    return by_frame


def link_blobs(
    blobs: Sequence[FrameBlob],
    cfg: TemporalLinkingConfig | None = None,
    *,
    frame_w: int,
    frame_h: int,
) -> tuple[list[Trajectory], dict[str, Any]]:
    """Link all input blobs then filter by min_track_length (no pre-delete)."""
    cfg = cfg or TemporalLinkingConfig()
    by_frame = group_blobs_by_frame(blobs)
    frame_indices = sorted(by_frame.keys())

    active: list[Trajectory] = []
    finished: list[Trajectory] = []
    next_track_id = 1
    n_gap1_links = 0
    n_new_starts = 0

    def _terminate_stale(current_fi: int) -> None:
        still: list[Trajectory] = []
        for tr in active:
            # last detection + max_frame_gap already passed → close
            if current_fi - tr.last_frame_index > cfg.max_frame_gap:
                tr.terminated = True
                finished.append(tr)
            else:
                still.append(tr)
        active[:] = still

    for fi in frame_indices:
        _terminate_stale(fi)
        frame_blobs = by_frame[fi]

        candidates: list[tuple[float, int, int, dict[str, Any]]] = []
        for ti, tr in enumerate(active):
            prev = tr.last_blob
            for bi, blob in enumerate(frame_blobs):
                ok, info = association_allowed(
                    prev, blob, cfg, frame_w=frame_w, frame_h=frame_h
                )
                if not ok:
                    continue
                score = matching_score(prev, blob, info, cfg)
                candidates.append((score, ti, bi, info))

        candidates.sort(key=lambda x: x[0], reverse=True)
        used_tracks: set[int] = set()
        used_blobs: set[int] = set()

        for score, ti, bi, info in candidates:
            if ti in used_tracks or bi in used_blobs:
                continue
            tr = active[ti]
            blob = frame_blobs[bi]
            gap = int(info["frame_gap"])
            tr.observations.append(
                TrackObservation(blob=blob, frame_gap_from_previous=gap)
            )
            if gap == 2:
                n_gap1_links += 1
            used_tracks.add(ti)
            used_blobs.add(bi)

        for bi, blob in enumerate(frame_blobs):
            if bi in used_blobs:
                continue
            tr = Trajectory(track_id=next_track_id)
            next_track_id += 1
            tr.observations.append(
                TrackObservation(blob=blob, frame_gap_from_previous=None)
            )
            active.append(tr)
            n_new_starts += 1

    for tr in active:
        tr.terminated = True
        finished.append(tr)
    active.clear()

    n_before = len(finished)
    one_frame = [tr for tr in finished if tr.detected_frame_count < cfg.min_track_length]
    kept = [tr for tr in finished if tr.detected_frame_count >= cfg.min_track_length]
    # Stable track_id renumber for kept tracks (optional readability).
    for new_id, tr in enumerate(kept, start=1):
        tr.track_id = new_id

    lengths = [tr.detected_frame_count for tr in kept]
    gap1_kept = int(sum(tr.gap1_link_count() for tr in kept))
    stats = {
        "input_blob_count": int(len(blobs)),
        "trajectories_before_filter": int(n_before),
        "one_frame_trajectory_count": int(len(one_frame)),
        "removed_one_frame_trajectory_count": int(len(one_frame)),
        "final_trajectory_count": int(len(kept)),
        "track_length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "track_length_median": float(np.median(lengths)) if lengths else 0.0,
        "track_length_max": int(max(lengths)) if lengths else 0,
        "gap1_link_count": gap1_kept,
        "gap1_link_count_before_filter": int(n_gap1_links),
        "new_start_blob_count": int(n_new_starts),
        "config": asdict(cfg),
    }
    return kept, stats


def tracks_to_rows(tracks: Sequence[Trajectory]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tr in tracks:
        det_count = tr.detected_frame_count
        start_f = tr.start_frame_idx
        end_f = tr.end_frame_idx
        for obs in tr.observations:
            b = obs.blob
            rows.append(
                {
                    "track_id": tr.track_id,
                    "frame_index": b.frame_index,
                    "frame_idx": b.frame_idx,
                    "blob_id": b.blob_id,
                    "bbox": f"[{b.x1:g},{b.y1:g},{b.x2:g},{b.y2:g}]",
                    "x1": b.x1,
                    "y1": b.y1,
                    "x2": b.x2,
                    "y2": b.y2,
                    "center_x": b.center[0],
                    "center_y": b.center[1],
                    "motion_score": b.motion_score,
                    "frame_gap_from_previous": (
                        ""
                        if obs.frame_gap_from_previous is None
                        else obs.frame_gap_from_previous
                    ),
                    "detected_frame_count": det_count,
                    "start_frame": start_f,
                    "end_frame": end_f,
                    "active_block_count": (
                        "" if b.active_block_count is None else b.active_block_count
                    ),
                    "timestamp_sec": b.timestamp_sec,
                    "is_gap1_link": int(
                        obs.frame_gap_from_previous is not None
                        and obs.frame_gap_from_previous == 2
                    ),
                }
            )
    return rows


def tracks_to_json(tracks: Sequence[Trajectory]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tr in tracks:
        dets = []
        for obs in tr.observations:
            b = obs.blob
            dets.append(
                {
                    "frame_index": b.frame_index,
                    "frame_idx": b.frame_idx,
                    "blob_id": b.blob_id,
                    "bbox": [b.x1, b.y1, b.x2, b.y2],
                    "center": [b.center[0], b.center[1]],
                    "motion_score": b.motion_score,
                    "active_block_count": b.active_block_count,
                    "timestamp_sec": b.timestamp_sec,
                    "frame_gap_from_previous": obs.frame_gap_from_previous,
                    "is_gap1_link": bool(
                        obs.frame_gap_from_previous is not None
                        and obs.frame_gap_from_previous == 2
                    ),
                }
            )
        out.append(
            {
                "track_id": tr.track_id,
                "detected_frame_count": tr.detected_frame_count,
                "start_frame": tr.start_frame_idx,
                "end_frame": tr.end_frame_idx,
                "start_frame_index": tr.start_frame_index,
                "end_frame_index": tr.end_frame_index,
                "gap1_link_count": tr.gap1_link_count(),
                "detections": dets,
            }
        )
    return out


def save_temporal_linking_outputs(
    output_dir: Path | str,
    tracks: Sequence[Trajectory],
    summary: dict[str, Any],
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "temporal_tracks.csv"
    json_path = output_dir / "temporal_tracks.json"
    summary_path = output_dir / "temporal_linking_summary.json"

    rows = tracks_to_rows(tracks)
    fieldnames = [
        "track_id",
        "frame_index",
        "frame_idx",
        "blob_id",
        "bbox",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
        "motion_score",
        "frame_gap_from_previous",
        "detected_frame_count",
        "start_frame",
        "end_frame",
        "active_block_count",
        "timestamp_sec",
        "is_gap1_link",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    json_path.write_text(
        json.dumps(tracks_to_json(tracks), indent=2),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "temporal_tracks_csv": csv_path,
        "temporal_tracks_json": json_path,
        "temporal_linking_summary": summary_path,
    }


def infer_sample_fps_from_blobs(
    blobs: Sequence[FrameBlob], fallback: float = 5.0
) -> float:
    by_fi: dict[int, float] = {}
    for b in blobs:
        by_fi.setdefault(int(b.frame_index), float(b.timestamp_sec))
    items = sorted(by_fi.items())
    if len(items) < 2:
        return float(fallback)
    dts = [items[i + 1][1] - items[i][1] for i in range(len(items) - 1)]
    dts = [d for d in dts if d > 1e-6]
    if not dts:
        return float(fallback)
    med = float(np.median(dts))
    return float(1.0 / med) if med > 1e-6 else float(fallback)


def color_for_track(track_id: int) -> tuple[int, int, int]:
    # Stable BGR palette (avoid near-duplicates for nearby ids).
    palette = [
        (0, 165, 255),
        (0, 255, 0),
        (255, 128, 0),
        (255, 0, 255),
        (0, 255, 255),
        (80, 80, 255),
        (255, 255, 0),
        (180, 105, 255),
    ]
    return palette[(int(track_id) - 1) % len(palette)]


def build_overlay_index(
    tracks: Sequence[Trajectory],
) -> dict[int, list[tuple[Trajectory, TrackObservation, list[tuple[float, float]]]]]:
    """Map frame_index → list of (track, obs, recent_centers including current)."""
    index: dict[int, list[tuple[Trajectory, TrackObservation, list[tuple[float, float]]]]] = {}
    for tr in tracks:
        centers: list[tuple[float, float]] = []
        for obs in tr.observations:
            centers.append(obs.blob.center)
            index.setdefault(obs.blob.frame_index, []).append(
                (tr, obs, list(centers))
            )
    return index


def render_temporal_overlay_frame(
    frame_bgr: np.ndarray,
    entries: Iterable[tuple[Trajectory, TrackObservation, list[tuple[float, float]]]],
    *,
    trail_length: int = DEFAULT_TRAIL_LENGTH,
) -> np.ndarray:
    import cv2  # local import so pure linking stays usable without cv2 at import time

    out = frame_bgr.copy()
    for tr, obs, centers_so_far in entries:
        color = color_for_track(tr.track_id)
        b = obs.blob
        x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
        thickness = 3 if obs.frame_gap_from_previous == 2 else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        cx, cy = int(round(b.center[0])), int(round(b.center[1]))
        cv2.circle(out, (cx, cy), 4, color, -1)

        trail = centers_so_far[-max(int(trail_length), 1) :]
        if len(trail) >= 2:
            pts = np.array(
                [[int(round(p[0])), int(round(p[1]))] for p in trail],
                dtype=np.int32,
            )
            cv2.polylines(out, [pts], False, color, 2, lineType=cv2.LINE_AA)

        gap_tag = " gap1" if obs.frame_gap_from_previous == 2 else ""
        label = f"T{tr.track_id}{gap_tag}"
        cv2.putText(
            out,
            label,
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def assign_timeline_lanes(
    tracks: Sequence[Trajectory],
) -> list[tuple[Trajectory, int]]:
    """Greedy pack tracks into lanes so overlapping spans avoid the same row."""
    ordered = sorted(
        tracks,
        key=lambda tr: (tr.start_frame_index, tr.end_frame_index, tr.track_id),
    )
    lane_ends: list[int] = []  # last exclusive end frame_index per lane
    assigned: list[tuple[Trajectory, int]] = []
    for tr in ordered:
        start = int(tr.start_frame_index)
        end = int(tr.end_frame_index)
        lane = None
        for i, last_end in enumerate(lane_ends):
            if start > last_end:
                lane = i
                lane_ends[i] = end
                break
        if lane is None:
            lane = len(lane_ends)
            lane_ends.append(end)
        assigned.append((tr, lane))
    return assigned


def render_track_timeline_panel(
    tracks: Sequence[Trajectory],
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
    """Draw a DAW-like timeline strip: colored bars = track lifespan.

    X = frame_index. Lanes are packed (multiple tracks per row when non-overlapping).
    Bar color matches ``color_for_track`` used on the video bbox.
    """
    import cv2

    assignments = assign_timeline_lanes(tracks)
    n_lanes = max((lane for _, lane in assignments), default=-1) + 1
    n_lanes = max(n_lanes, 1)
    if panel_height is None:
        panel_height = title_h + pad_y * 2 + n_lanes * lane_height + 4
    panel_height = max(int(panel_height), title_h + pad_y * 2 + lane_height)

    panel = np.full((panel_height, int(width), 3), 28, dtype=np.uint8)
    # Subtle lane separators
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

    for tr, lane in assignments:
        color = color_for_track(tr.track_id)
        x1 = x_of(tr.start_frame_index)
        x2 = x_of(tr.end_frame_index + 1)  # inclusive end → half-open bar
        x2 = max(x2, x1 + 2)
        y1 = int(content_top + lane * row_h + 2)
        y2 = int(content_top + (lane + 1) * row_h - 2)
        y2 = max(y2, y1 + 4)
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, -1)
        # Darker edge for readability
        cv2.rectangle(panel, (x1, y1), (x2, y2), (20, 20, 20), 1)
        if x2 - x1 >= 28:
            cv2.putText(
                panel,
                f"T{tr.track_id}",
                (x1 + 3, y1 + max(12, (y2 - y1) // 2 + 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (15, 15, 15),
                1,
                cv2.LINE_AA,
            )

    # Playhead
    px = x_of(int(current_frame_index))
    cv2.line(panel, (px, title_h), (px, panel_height - 2), (240, 240, 240), 1)
    return panel


def compose_frame_with_timeline(
    video_bgr: np.ndarray,
    tracks: Sequence[Trajectory],
    *,
    current_frame_index: int,
    t_min: int,
    t_max: int,
    timeline_height: int | None = None,
) -> np.ndarray:
    """Stack video frame above a matching-color track timeline panel."""
    h, w = video_bgr.shape[:2]
    timeline = render_track_timeline_panel(
        tracks,
        current_frame_index=current_frame_index,
        width=w,
        t_min=t_min,
        t_max=t_max,
        panel_height=timeline_height,
    )
    return np.vstack([video_bgr, timeline])
