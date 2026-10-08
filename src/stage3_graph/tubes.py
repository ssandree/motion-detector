"""ROI tubes from graph clusters — fixed lifetime AABB for VLM crops."""

from __future__ import annotations

from dataclasses import dataclass, field

from stage3_graph.nodes import GraphNode


@dataclass
class GraphRoiTube:
    cluster_id: int
    members: list[GraphNode] = field(default_factory=list)

    @property
    def t0(self) -> int:
        return min(m.t0 for m in self.members)

    @property
    def t1(self) -> int:
        return max(m.t1 for m in self.members)

    @property
    def n_events(self) -> int:
        return len(self.members)

    @property
    def n_cells(self) -> int:
        return len({(m.y, m.x) for m in self.members})

    def cells_at(self, t: int) -> list[tuple[int, int]]:
        return [(m.y, m.x) for m in self.members if m.active_at(t)]

    def spatial_extent(self) -> tuple[int, int, int, int]:
        """Lifetime union bbox of all member blocks [x0,y0,x1,y1)."""
        ys = [m.y for m in self.members]
        xs = [m.x for m in self.members]
        return int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1

    def bbox_at(self, t: int) -> tuple[int, int, int, int] | None:
        """Fixed lifetime AABB while the tube is alive (stable VLM ROI)."""
        if not (int(self.t0) <= int(t) <= int(self.t1)):
            return None
        return self.spatial_extent()

    def clipped_pixel_wh(
        self,
        *,
        cell_px: int,
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int]:
        x0, y0, x1, y1 = self.spatial_extent()
        px0 = int(x0) * int(cell_px)
        py0 = int(y0) * int(cell_px)
        px1 = min(int(frame_width), int(x1) * int(cell_px))
        py1 = min(int(frame_height), int(y1) * int(cell_px))
        return max(0, px1 - px0), max(0, py1 - py0)

    def covers_min_block(
        self,
        *,
        cell_px: int,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        w, h = self.clipped_pixel_wh(
            cell_px=cell_px,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        return w >= int(cell_px) and h >= int(cell_px)


def build_roi_tubes(
    nodes: list[GraphNode],
    labels: list[int],
) -> list[GraphRoiTube]:
    if len(nodes) != len(labels):
        raise ValueError("nodes/labels length mismatch")
    groups: dict[int, list[GraphNode]] = {}
    for node, lab in zip(nodes, labels):
        groups.setdefault(int(lab), []).append(node)
    tubes: list[GraphRoiTube] = []
    for cid in sorted(groups):
        members = sorted(groups[cid], key=lambda m: (m.t0, m.y, m.x, m.event_id))
        tubes.append(GraphRoiTube(cluster_id=int(cid), members=members))
    return _renumber_tubes(tubes)


def drop_sub_block_tubes(
    tubes: list[GraphRoiTube],
    *,
    cell_px: int,
    frame_width: int,
    frame_height: int,
) -> tuple[list[GraphRoiTube], list[int]]:
    """Drop tubes whose clipped lifetime AABB is smaller than one unit block."""
    if not tubes or int(frame_width) <= 0 or int(frame_height) <= 0:
        return _renumber_tubes(tubes), []
    kept: list[GraphRoiTube] = []
    removed: list[int] = []
    for tube in tubes:
        if tube.covers_min_block(
            cell_px=int(cell_px),
            frame_width=int(frame_width),
            frame_height=int(frame_height),
        ):
            kept.append(tube)
        else:
            removed.append(int(tube.cluster_id))
    return _renumber_tubes(kept), removed


def _renumber_tubes(tubes: list[GraphRoiTube]) -> list[GraphRoiTube]:
    ordered = sorted(tubes, key=lambda t: (t.t0, -t.n_events, t.cluster_id))
    for index, tube in enumerate(ordered, start=1):
        tube.cluster_id = index
    return ordered


def tube_summary(tube: GraphRoiTube) -> dict:
    x0, y0, x1, y1 = tube.spatial_extent()
    return {
        "cluster_id": int(tube.cluster_id),
        "n_events": int(tube.n_events),
        "n_cells": int(tube.n_cells),
        "temporal_range": [int(tube.t0), int(tube.t1)],
        "spatial_extent_grid": [int(x0), int(y0), int(x1), int(y1)],
        "roi_mode": "lifetime_aabb",
    }
