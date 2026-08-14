"""Hysteresis block-events + 24-neighbor / proximity ROI tubes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from motion_analyzer.config import (
    ROI_MAX_GAP,
    ROI_MERGE_SPATIAL_DIST,
    ROI_MERGE_TEMPORAL_GAP,
    ROI_MIN_BLOCK_EVENT,
    ROI_MIN_TUBE_CELLS,
    ROI_MIN_TUBE_DURATION,
    ROI_NEIGH_RADIUS,
    ROI_SUPPRESS_CONTAINED,
    ROI_TAU_HIGH,
    ROI_TAU_LOW,
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


@dataclass
class RoiTube:
    tube_id: int
    members: list[BlockEvent] = field(default_factory=list)

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
) -> tuple[list[BlockEvent], list[RoiTube], list[int]]:
    events = build_block_events(
        mag,
        tau_high=tau_high,
        tau_low=tau_low,
        max_gap=max_gap,
        min_block_event=min_block_event,
    )
    tubes = merge_block_events(
        events,
        neigh_radius=neigh_radius,
        min_tube_cells=min_tube_cells,
        min_tube_duration=min_tube_duration,
    )
    tubes = proximity_merge_tubes(
        tubes,
        spatial_dist=merge_spatial_dist,
        temporal_gap=merge_temporal_gap,
    )
    # Re-apply size filter after proximity merge (merged tubes may still be tiny).
    tubes = [
        t
        for t in tubes
        if t.num_cells >= int(min_tube_cells)
        and t.duration >= int(min_tube_duration)
    ]
    tubes = _finalize_tubes(tubes)
    removed: list[int] = []
    if suppress_contained:
        tubes, removed = suppress_contained_tubes(tubes)
    return events, tubes, removed


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
        fixed = tube.spatial_bbox()
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
