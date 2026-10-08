"""Hysteresis block-events + 24-neighbor / proximity ROI tubes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from motion_analyzer.config import (
    ROI_MAX_GAP,
    ROI_MERGE_SPATIAL_DIST,
    ROI_MERGE_TEMPORAL_GAP,
    ROI_MIN_BLOCK_EVENT,
    ROI_MIN_BLOCK_PX,
    ROI_MIN_TUBE_CELLS,
    ROI_MIN_TUBE_DURATION,
    ROI_NEIGH_RADIUS,
    ROI_SUPPRESS_CONTAINED,
    ROI_TAU_HIGH,
    ROI_TAU_LOW,
    UNIT_CELL_PX,
)


def chebyshev_offsets(radius: int) -> tuple[tuple[int, int], ...]:
    """All (dy, dx) with 1 ≤ max(|dy|,|dx|) ≤ radius (24-cc when radius=2)."""
    r = int(radius)
    if r < 1:
        raise ValueError("radius must be ≥ 1")
    offs: list[tuple[int, int]] = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue
            if max(abs(dy), abs(dx)) <= r:
                offs.append((dy, dx))
    return tuple(offs)


@dataclass(frozen=True)
class BlockEvent:
    """Per-block temporal event on the unit grid."""

    y: int
    x: int
    t0: int  # inclusive
    t1: int  # inclusive
    event_id: int

    @property
    def duration(self) -> int:
        return int(self.t1 - self.t0 + 1)

    def active_at(self, t: int) -> bool:
        return int(self.t0) <= int(t) <= int(self.t1)

    def overlaps_time(self, other: "BlockEvent") -> bool:
        return not (self.t1 < other.t0 or other.t1 < self.t0)

    def time_gap(self, other: "BlockEvent") -> int:
        """0 if intervals overlap/touch; else frames between them."""
        if self.overlaps_time(other) or self.t1 + 1 == other.t0 or other.t1 + 1 == self.t0:
            return 0
        if self.t1 < other.t0:
            return int(other.t0 - self.t1 - 1)
        return int(self.t0 - other.t1 - 1)

    def spatial_chebyshev(self, other: "BlockEvent") -> int:
        return int(max(abs(self.y - other.y), abs(self.x - other.x)))


@dataclass
class RoiTube:
    tube_id: int
    members: list[BlockEvent] = field(default_factory=list)
    source_tube_id: int | None = None
    place_class: str | None = None
    expanded_bbox: tuple[int, int, int, int] | None = None
    added_cells: list[tuple[int, int]] = field(default_factory=list)

    @property
    def t0(self) -> int:
        return min(m.t0 for m in self.members)

    @property
    def t1(self) -> int:
        return max(m.t1 for m in self.members)

    @property
    def duration(self) -> int:
        return int(self.t1 - self.t0 + 1)

    @property
    def num_cells(self) -> int:
        return len({(m.y, m.x) for m in self.members})

    def cells_at(self, t: int) -> list[tuple[int, int]]:
        return [(m.y, m.x) for m in self.members if m.active_at(t)]

    def bbox_at(self, t: int) -> tuple[int, int, int, int] | None:
        cells = self.cells_at(t)
        if not cells:
            return None
        ys = [c[0] for c in cells]
        xs = [c[1] for c in cells]
        return int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1

    def spatial_bbox(self) -> tuple[int, int, int, int]:
        ys = [m.y for m in self.members]
        xs = [m.x for m in self.members]
        return int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1

    def display_bbox(self) -> tuple[int, int, int, int]:
        if self.expanded_bbox is not None:
            return tuple(int(v) for v in self.expanded_bbox)
        return self.spatial_bbox()

    def time_gap(self, other: "RoiTube") -> int:
        if not (self.t1 < other.t0 or other.t1 < self.t0):
            return 0
        if self.t1 < other.t0:
            return int(other.t0 - self.t1 - 1)
        return int(self.t0 - other.t1 - 1)

    def spatial_chebyshev(self, other: "RoiTube") -> int:
        """Min Chebyshev distance between any member cells of the two tubes."""
        best = 10**9
        cells_a = {(m.y, m.x) for m in self.members}
        cells_b = {(m.y, m.x) for m in other.members}
        for ya, xa in cells_a:
            for yb, xb in cells_b:
                best = min(best, max(abs(ya - yb), abs(xa - xb)))
                if best == 0:
                    return 0
        return int(best)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def add(self, item: int) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: int) -> int:
        self.add(item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def extract_block_events(
    seq: np.ndarray,
    *,
    y: int,
    x: int,
    tau_high: float,
    tau_low: float,
    max_gap: int,
    start_event_id: int,
) -> list[BlockEvent]:
    """Hysteresis + max_gap hold on a 1-D magnitude sequence."""
    values = np.asarray(seq, dtype=np.float32)
    events: list[BlockEvent] = []
    active = False
    start = 0
    last_support = -10**9
    next_id = int(start_event_id)

    for t, raw in enumerate(values.tolist()):
        v = float(raw)
        finite = np.isfinite(v)
        if not active:
            if finite and v >= float(tau_high):
                active = True
                start = t
                last_support = t
            continue

        if finite and v >= float(tau_low):
            last_support = t

        if t - last_support > int(max_gap):
            events.append(
                BlockEvent(y=y, x=x, t0=start, t1=last_support, event_id=next_id)
            )
            next_id += 1
            active = False
            if finite and v >= float(tau_high):
                active = True
                start = t
                last_support = t

    if active:
        events.append(
            BlockEvent(y=y, x=x, t0=start, t1=last_support, event_id=next_id)
        )
    return events


def build_block_events(
    mag: np.ndarray,
    *,
    tau_high: float = ROI_TAU_HIGH,
    tau_low: float = ROI_TAU_LOW,
    max_gap: int = ROI_MAX_GAP,
    min_block_event: int = ROI_MIN_BLOCK_EVENT,
) -> list[BlockEvent]:
    if mag.ndim != 3:
        raise ValueError(f"expected (T,H,W), got {mag.shape}")
    _, height, width = mag.shape
    events: list[BlockEvent] = []
    next_id = 1
    for y in range(height):
        for x in range(width):
            cell_events = extract_block_events(
                mag[:, y, x],
                y=y,
                x=x,
                tau_high=tau_high,
                tau_low=tau_low,
                max_gap=max_gap,
                start_event_id=next_id,
            )
            for ev in cell_events:
                if ev.duration >= int(min_block_event):
                    events.append(ev)
                next_id = max(next_id, ev.event_id + 1)
    return events


def event_to_dict(event: BlockEvent, mag: np.ndarray | None = None) -> dict:
    """Serialize one per-cell temporal event (no 3D clustering)."""
    per_frame: list[dict] = []
    for t in range(int(event.t0), int(event.t1) + 1):
        value = 0.0
        if mag is not None and 0 <= t < mag.shape[0]:
            raw = float(mag[t, int(event.y), int(event.x)])
            if np.isfinite(raw):
                value = raw
        per_frame.append(
            {
                "frame_index": int(t),
                "cells": [[int(event.y), int(event.x)]],
                "magnitudes": [value],
            }
        )
    return {
        "event_id": int(event.event_id),
        "start_frame": int(event.t0),
        "end_frame": int(event.t1),
        "duration": int(event.duration),
        "y": int(event.y),
        "x": int(event.x),
        "per_frame": per_frame,
    }


def _same_region(
    a: BlockEvent,
    b: BlockEvent,
    region_id_grid: np.ndarray | None,
) -> bool:
    if region_id_grid is None:
        return True
    grid = np.asarray(region_id_grid)
    gh, gw = grid.shape

    def _at(y: int, x: int) -> int:
        if 0 <= int(y) < gh and 0 <= int(x) < gw:
            return int(grid[int(y), int(x)])
        return -(int(y) * 10_000 + int(x) + 1)

    return _at(a.y, a.x) == _at(b.y, b.x)


def cluster_block_events(
    events: list[BlockEvent],
    *,
    spatial_dist: int = ROI_MERGE_SPATIAL_DIST,
    temporal_gap: int = ROI_MERGE_TEMPORAL_GAP,
    min_tube_cells: int = ROI_MIN_TUBE_CELLS,
    min_tube_duration: int = ROI_MIN_TUBE_DURATION,
    region_id_grid: np.ndarray | None = None,
) -> list[RoiTube]:
    """Cluster per-block event tubes into ROI tubes.

    Two events are linked if both hold:
      [spatial]  Chebyshev distance ≤ ``spatial_dist`` (same-frame when they overlap)
      [temporal] frame gap between intervals ≤ ``temporal_gap``
    When ``region_id_grid`` is set, they must also share the same cell region id.
    """
    if not events:
        return []

    uf = _UnionFind()
    for ev in events:
        uf.add(ev.event_id)

    dist = int(spatial_dist)
    gap_lim = int(temporal_gap)
    region = None if region_id_grid is None else np.asarray(region_id_grid)
    for i, a in enumerate(events):
        for b in events[i + 1 :]:
            if a.spatial_chebyshev(b) > dist or a.time_gap(b) > gap_lim:
                continue
            if not _same_region(a, b, region):
                continue
            uf.union(a.event_id, b.event_id)

    groups: dict[int, list[BlockEvent]] = {}
    for ev in events:
        groups.setdefault(uf.find(ev.event_id), []).append(ev)

    tubes: list[RoiTube] = []
    next_id = 1
    for members in groups.values():
        tube = RoiTube(
            tube_id=next_id,
            members=sorted(members, key=lambda m: (m.t0, m.y, m.x)),
        )
        if tube.num_cells >= int(min_tube_cells) and tube.duration >= int(
            min_tube_duration
        ):
            tubes.append(tube)
            next_id += 1
    return _finalize_tubes(tubes)


def merge_block_events(
    events: list[BlockEvent],
    *,
    neigh_radius: int = ROI_NEIGH_RADIUS,
    min_tube_cells: int = ROI_MIN_TUBE_CELLS,
    min_tube_duration: int = ROI_MIN_TUBE_DURATION,
) -> list[RoiTube]:
    """Merge block-events when within Chebyshev radius and time intervals overlap."""
    if not events:
        return []

    offsets = chebyshev_offsets(int(neigh_radius))
    uf = _UnionFind()
    for ev in events:
        uf.add(ev.event_id)

    by_cell: dict[tuple[int, int], list[BlockEvent]] = {}
    for ev in events:
        by_cell.setdefault((ev.y, ev.x), []).append(ev)

    for ev in events:
        for dy, dx in offsets:
            neigh = by_cell.get((ev.y + dy, ev.x + dx))
            if not neigh:
                continue
            for other in neigh:
                if other.event_id <= ev.event_id:
                    continue
                if ev.overlaps_time(other):
                    uf.union(ev.event_id, other.event_id)

    groups: dict[int, list[BlockEvent]] = {}
    for ev in events:
        groups.setdefault(uf.find(ev.event_id), []).append(ev)

    tubes: list[RoiTube] = []
    next_id = 1
    for members in groups.values():
        tube = RoiTube(
            tube_id=next_id,
            members=sorted(members, key=lambda m: (m.t0, m.y, m.x)),
        )
        # Hard spatial floor: ≤1 unit block (64×64) cannot form an ROI.
        # Duration is an additional requirement (AND), not an escape hatch.
        if tube.num_cells >= int(min_tube_cells) and tube.duration >= int(
            min_tube_duration
        ):
            tubes.append(tube)
            next_id += 1
    return _finalize_tubes(tubes)


def proximity_merge_tubes(
    tubes: list[RoiTube],
    *,
    spatial_dist: int = ROI_MERGE_SPATIAL_DIST,
    temporal_gap: int = ROI_MERGE_TEMPORAL_GAP,
) -> list[RoiTube]:
    """Merge tubes that are spatially and temporally close (iterative)."""
    if len(tubes) <= 1:
        return _finalize_tubes(tubes)

    current = list(tubes)
    changed = True
    while changed:
        changed = False
        uf = _UnionFind()
        for tube in current:
            uf.add(tube.tube_id)
        for i, a in enumerate(current):
            for b in current[i + 1 :]:
                if a.spatial_chebyshev(b) <= int(spatial_dist) and a.time_gap(
                    b
                ) <= int(temporal_gap):
                    uf.union(a.tube_id, b.tube_id)
                    changed = True
        if not changed:
            break
        groups: dict[int, list[RoiTube]] = {}
        for tube in current:
            groups.setdefault(uf.find(tube.tube_id), []).append(tube)
        merged: list[RoiTube] = []
        for index, members in enumerate(groups.values(), start=1):
            all_events: list[BlockEvent] = []
            for tube in members:
                all_events.extend(tube.members)
            merged.append(
                RoiTube(
                    tube_id=index,
                    members=sorted(all_events, key=lambda m: (m.t0, m.y, m.x)),
                )
            )
        if len(merged) == len(current):
            break
        current = merged
    return _finalize_tubes(current)


def _finalize_tubes(tubes: list[RoiTube]) -> list[RoiTube]:
    tubes = sorted(tubes, key=lambda t: (t.t0, -t.num_cells, t.tube_id))
    for index, tube in enumerate(tubes, start=1):
        tube.tube_id = index
    return tubes


def finalize_tubes(tubes: list[RoiTube]) -> list[RoiTube]:
    return _finalize_tubes(tubes)


def tube_from_events(members: list[BlockEvent], *, tube_id: int) -> RoiTube:
    return RoiTube(
        tube_id=int(tube_id),
        members=sorted(members, key=lambda m: (m.t0, m.y, m.x, m.event_id)),
    )


def tube_area_cost(tube: RoiTube) -> float:
    """Lifetime AABB area × duration (Stage3 crop representation)."""
    x0, y0, x1, y1 = tube.spatial_bbox()
    area = max(0, int(x1) - int(x0)) * max(0, int(y1) - int(y0))
    return float(area * int(tube.duration))


def frame_indices_from_meta(meta: dict, n_frames: int) -> list[int]:
    align_start = int(meta.get("align_start_sampled_index") or 0)
    curr = meta.get("sampled_index_curr")
    if curr is not None:
        return [int(i) for i in curr.tolist()]
    return list(range(align_start, align_start + int(n_frames)))


def tube_fully_contained(inner: RoiTube, outer: RoiTube) -> bool:
    """True if inner's time span and spatial bbox are strictly inside outer."""
    if inner.tube_id == outer.tube_id:
        return False
    if not (outer.t0 <= inner.t0 and inner.t1 <= outer.t1):
        return False
    ix0, iy0, ix1, iy1 = inner.spatial_bbox()
    ox0, oy0, ox1, oy1 = outer.spatial_bbox()
    if not (ox0 <= ix0 and oy0 <= iy0 and ix1 <= ox1 and iy1 <= oy1):
        return False
    # Require proper containment (not identical extent).
    return (inner.t0, inner.t1, ix0, iy0, ix1, iy1) != (
        outer.t0,
        outer.t1,
        ox0,
        oy0,
        ox1,
        oy1,
    )


def cell_clipped_pixel_wh(
    y: int,
    x: int,
    *,
    cell_px: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int]:
    """Pixel size of one unit grid cell after clipping to the video frame."""
    px0 = int(x) * int(cell_px)
    py0 = int(y) * int(cell_px)
    px1 = min(int(frame_width), px0 + int(cell_px))
    py1 = min(int(frame_height), py0 + int(cell_px))
    return max(0, px1 - px0), max(0, py1 - py0)


def event_covers_full_block(
    event: BlockEvent,
    *,
    cell_px: int,
    frame_width: int,
    frame_height: int,
) -> bool:
    """True if this block event sits on a full cell_px×cell_px region in-frame."""
    width, height = cell_clipped_pixel_wh(
        int(event.y),
        int(event.x),
        cell_px=int(cell_px),
        frame_width=int(frame_width),
        frame_height=int(frame_height),
    )
    return width >= int(cell_px) and height >= int(cell_px)


def filter_partial_block_events(
    events: list[BlockEvent],
    *,
    cell_px: int,
    frame_width: int,
    frame_height: int,
    neighbor_chebyshev: int = 1,
) -> tuple[list[BlockEvent], list[int]]:
    """Drop orphan partial edge/corner cells; keep partials next to full motion.

    A partial (< cell_px×cell_px in-frame) event is kept when some full-block
    event lies within ``neighbor_chebyshev`` and overlaps in time. Orphan
    partials with no such neighbor are dropped so they never form standalone
    ROIs, while edge strips adjacent to real motion stay attached.
    """
    if not events or int(frame_width) <= 0 or int(frame_height) <= 0:
        return list(events), []

    full: list[BlockEvent] = []
    partial: list[BlockEvent] = []
    for event in events:
        if event_covers_full_block(
            event,
            cell_px=int(cell_px),
            frame_width=int(frame_width),
            frame_height=int(frame_height),
        ):
            full.append(event)
        else:
            partial.append(event)

    if not partial:
        return list(events), []

    neigh = int(neighbor_chebyshev)
    kept_partial: list[BlockEvent] = []
    dropped: list[int] = []
    for event in partial:
        ey, ex = int(event.y), int(event.x)
        attached = False
        for other in full:
            if max(abs(ey - int(other.y)), abs(ex - int(other.x))) > neigh:
                continue
            if event.overlaps_time(other) or event.time_gap(other) <= 0:
                attached = True
                break
        if attached:
            kept_partial.append(event)
        else:
            dropped.append(int(event.event_id))

    kept = full + kept_partial
    kept.sort(key=lambda e: (int(e.t0), int(e.y), int(e.x), int(e.event_id)))
    return kept, sorted(dropped)


def clipped_pixel_wh(
    tube: RoiTube,
    *,
    cell_px: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int]:
    """Pixel size of the tube bbox after clipping to the video frame."""
    x0, y0, x1, y1 = tube.spatial_bbox()
    px0 = int(x0) * int(cell_px)
    py0 = int(y0) * int(cell_px)
    px1 = min(int(frame_width), int(x1) * int(cell_px))
    py1 = min(int(frame_height), int(y1) * int(cell_px))
    return max(0, px1 - px0), max(0, py1 - py0)


def tube_covers_full_block(
    tube: RoiTube,
    *,
    cell_px: int,
    frame_width: int,
    frame_height: int,
) -> bool:
    """True if the clipped bbox can contain at least one full unit block."""
    width, height = clipped_pixel_wh(
        tube,
        cell_px=cell_px,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    return width >= int(cell_px) and height >= int(cell_px)


def drop_sub_block_tubes(
    tubes: list[RoiTube],
    *,
    cell_px: int,
    frame_width: int,
    frame_height: int,
) -> tuple[list[RoiTube], list[int]]:
    """Remove tubes whose clipped pixel bbox is smaller than one unit block."""
    if not tubes or int(frame_width) <= 0 or int(frame_height) <= 0:
        return _finalize_tubes(tubes), []
    removed: list[int] = []
    kept: list[RoiTube] = []
    for tube in tubes:
        if tube_covers_full_block(
            tube,
            cell_px=cell_px,
            frame_width=frame_width,
            frame_height=frame_height,
        ):
            kept.append(tube)
        else:
            removed.append(tube.tube_id)
    return _finalize_tubes(kept), sorted(removed)


def suppress_contained_tubes(tubes: list[RoiTube]) -> tuple[list[RoiTube], list[int]]:
    """Remove tubes whose spatio-temporal AABB is fully inside another tube."""
    if len(tubes) <= 1:
        return _finalize_tubes(tubes), []
    remove: set[int] = set()
    for a in tubes:
        for b in tubes:
            if tube_fully_contained(a, b):
                remove.add(a.tube_id)
                break
    kept = [t for t in tubes if t.tube_id not in remove]
    return _finalize_tubes(kept), sorted(remove)


def build_roi_tubes(
    mag: np.ndarray,
    *,
    tau_high: float = ROI_TAU_HIGH,
    tau_low: float = ROI_TAU_LOW,
    max_gap: int = ROI_MAX_GAP,
    min_block_event: int = ROI_MIN_BLOCK_EVENT,
    neigh_radius: int = ROI_NEIGH_RADIUS,
    merge_spatial_dist: int = ROI_MERGE_SPATIAL_DIST,
    merge_temporal_gap: int = ROI_MERGE_TEMPORAL_GAP,
    min_tube_cells: int = ROI_MIN_TUBE_CELLS,
    min_tube_duration: int = ROI_MIN_TUBE_DURATION,
    suppress_contained: bool = ROI_SUPPRESS_CONTAINED,
    min_block_px: bool = ROI_MIN_BLOCK_PX,
    cell_px: int = UNIT_CELL_PX,
    frame_width: int = 0,
    frame_height: int = 0,
    region_id_grid: np.ndarray | None = None,
) -> tuple[list[BlockEvent], list[RoiTube], list[int], list[int]]:
    events = build_block_events(
        mag,
        tau_high=tau_high,
        tau_low=tau_low,
        max_gap=max_gap,
        min_block_event=min_block_event,
    )
    tubes = cluster_block_events(
        events,
        spatial_dist=int(merge_spatial_dist),
        temporal_gap=int(merge_temporal_gap),
        min_tube_cells=min_tube_cells,
        min_tube_duration=min_tube_duration,
        region_id_grid=region_id_grid,
    )
    # neigh_radius is kept for CLI/output tags; clustering uses merge_spatial_dist.
    _ = neigh_radius
    dropped_sub: list[int] = []
    contained: list[int] = []
    if min_block_px:
        tubes, dropped_sub = drop_sub_block_tubes(
            tubes,
            cell_px=int(cell_px),
            frame_width=int(frame_width),
            frame_height=int(frame_height),
        )
    if suppress_contained:
        tubes, contained = suppress_contained_tubes(tubes)
    return events, tubes, dropped_sub, contained


def tubes_to_frame_overlays(
    tubes: Iterable[RoiTube],
    *,
    num_frames: int,
) -> dict[int, list[tuple[int, tuple[int, int, int, int], list[tuple[int, int]]]]]:
    """frame → list of (tube_id, fixed_spatial_bbox, active_cells)."""
    by_frame: dict[
        int, list[tuple[int, tuple[int, int, int, int], list[tuple[int, int]]]]
    ] = {t: [] for t in range(int(num_frames))}
    for tube in tubes:
        fixed = tube.display_bbox()
        for t in range(tube.t0, tube.t1 + 1):
            cells = tube.cells_at(t)
            # Show fixed tube bbox for the whole lifetime; cells may be empty in gaps.
            by_frame[t].append((tube.tube_id, fixed, cells))
    return by_frame


def tube_to_dict(tube: RoiTube) -> dict:
    fixed = list(tube.spatial_bbox())
    return {
        "tube_id": tube.tube_id,
        "t0": tube.t0,
        "t1": tube.t1,
        "duration": tube.duration,
        "num_cells": tube.num_cells,
        "num_block_events": len(tube.members),
        "spatial_bbox_grid": fixed,
        "members": [
            {
                "event_id": m.event_id,
                "y": m.y,
                "x": m.x,
                "t0": m.t0,
                "t1": m.t1,
                "duration": m.duration,
            }
            for m in tube.members
        ],
        "per_frame": [
            {
                "frame_index": t,
                "bbox_grid": fixed,
                "cells": [[y, x] for y, x in tube.cells_at(t)],
            }
            for t in range(tube.t0, tube.t1 + 1)
        ],
    }


def _tube_st_aabb(tube: RoiTube) -> tuple[int, int, int, int, int, int]:
    x0, y0, x1, y1 = tube.spatial_bbox()
    return (
        int(x0),
        int(y0),
        int(x1),
        int(y1),
        int(tube.t0),
        int(tube.t1) + 1,
    )


def _tube_st_iou(a: RoiTube, b: RoiTube) -> float:
    ax0, ay0, ax1, ay1, at0, at1 = _tube_st_aabb(a)
    bx0, by0, bx1, by1, bt0, bt1 = _tube_st_aabb(b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    it0, it1 = max(at0, bt0), min(at1, bt1)
    if ix1 <= ix0 or iy1 <= iy0 or it1 <= it0:
        return 0.0
    inter = float((ix1 - ix0) * (iy1 - iy0) * (it1 - it0))
    va = float(max(0, ax1 - ax0) * max(0, ay1 - ay0) * max(0, at1 - at0))
    vb = float(max(0, bx1 - bx0) * max(0, by1 - by0) * max(0, bt1 - bt0))
    union = va + vb - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def suppress_high_iou_tubes(
    tubes: list[RoiTube],
    *,
    iou_tau: float = 0.4,
) -> tuple[list[RoiTube], dict[str, Any]]:
    tau = float(iou_tau)
    ordered = sorted(
        tubes,
        key=lambda t: (
            float(tube_area_cost(t)),
            int(t.num_cells),
            int(t.duration),
            -int(t.tube_id),
        ),
        reverse=True,
    )
    kept: list[RoiTube] = []
    suppressed: list[dict[str, Any]] = []
    for cand in ordered:
        hit = None
        best_iou = 0.0
        for k in kept:
            iou = _tube_st_iou(cand, k)
            if iou >= tau and iou >= best_iou:
                hit = k
                best_iou = iou
        if hit is None:
            kept.append(cand)
        else:
            suppressed.append(
                {
                    "dropped_tube_id": int(cand.tube_id),
                    "kept_tube_id": int(hit.tube_id),
                    "iou": round(float(best_iou), 6),
                    "dropped_cost": float(tube_area_cost(cand)),
                    "kept_cost": float(tube_area_cost(hit)),
                    "dropped_n_cells": int(cand.num_cells),
                    "kept_n_cells": int(hit.num_cells),
                }
            )
    kept = finalize_tubes(kept)
    return kept, {
        "iou_tau": tau,
        "num_before": len(tubes),
        "num_after": len(kept),
        "num_suppressed": len(suppressed),
        "suppressed_pairs": suppressed,
    }


def apply_stage3_roi_filters(
    tubes: list[RoiTube],
    *,
    params: dict[str, Any],
    cell_px: int,
    frame_width: int,
    frame_height: int,
    iou_nms_tau: float | None = None,
) -> tuple[list[RoiTube], dict[str, Any]]:
    min_cells = int(params.get("min_tube_cells", ROI_MIN_TUBE_CELLS))
    min_dur = int(params.get("min_tube_duration", ROI_MIN_TUBE_DURATION))
    before = len(tubes)
    dropped_small = [
        int(t.tube_id)
        for t in tubes
        if t.num_cells < min_cells or t.duration < min_dur
    ]
    kept = [
        t
        for t in tubes
        if t.num_cells >= min_cells and t.duration >= min_dur
    ]
    dropped_sub: list[int] = []
    contained: list[int] = []
    if bool(params.get("min_block_px", ROI_MIN_BLOCK_PX)):
        kept, dropped_sub = drop_sub_block_tubes(
            kept,
            cell_px=int(cell_px),
            frame_width=int(frame_width),
            frame_height=int(frame_height),
        )
    if bool(params.get("suppress_contained", ROI_SUPPRESS_CONTAINED)):
        kept, contained = suppress_contained_tubes(kept)
    iou_log: dict[str, Any] | None = None
    if iou_nms_tau is not None and float(iou_nms_tau) > 0:
        kept, iou_log = suppress_high_iou_tubes(kept, iou_tau=float(iou_nms_tau))
    kept = finalize_tubes(kept)
    out: dict[str, Any] = {
        "num_before_filter": before,
        "num_after_filter": len(kept),
        "dropped_small_tube_ids": dropped_small,
        "dropped_sub_block_tube_ids": dropped_sub,
        "suppressed_contained_tube_ids": contained,
    }
    if iou_log is not None:
        out["iou_nms"] = iou_log
    return kept, out
