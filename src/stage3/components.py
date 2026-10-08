"""Event components and local pair features for Stage3 Union-Find."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from stage3.hysteresis_tube import BlockEvent
from stage3_graph.config import (
    FLOW_EPS,
    SIGMA_T,
    SIGMA_XY,
    SPATIAL_CHEBYSHEV_MAX,
    TAU_SPATIAL,
    TEMPORAL_GAP_MAX,
    W_DIR,
    W_T,
    W_XY,
)
from stage3_graph.edges import (
    GraphEdge,
    chebyshev_xy,
    cosine_similarity,
    spatial_similarity,
    temporal_gap_frames,
    temporal_overlap_over_min_duration,
    temporal_similarity_gap,
)
from stage3_graph.nodes import GraphNode

USE_TEMPORAL_BRIDGE = True
R_DIR_STRONG = 0.4


def node_to_event(node: GraphNode) -> BlockEvent:
    return BlockEvent(
        y=int(node.y),
        x=int(node.x),
        t0=int(node.t0),
        t1=int(node.t1),
        event_id=int(node.event_id),
    )


def mean_motion(nodes: list[GraphNode]) -> tuple[float, float, float]:
    """Unweighted mean event flow + directional coherence of that mean."""
    if not nodes:
        return 0.0, 0.0, 0.0
    u = float(np.mean([float(n.u) for n in nodes]))
    v = float(np.mean([float(n.v) for n in nodes]))
    speeds = [float(np.hypot(float(n.u), float(n.v))) for n in nodes]
    mean_speed = float(np.mean(speeds)) if speeds else 0.0
    coher = float(np.hypot(u, v) / (mean_speed + float(FLOW_EPS)))
    return u, v, float(np.clip(coher, 0.0, 1.0))


@dataclass
class GraphComponent:
    """One Union-Find / partition group of block events."""

    component_id: int
    nodes: list[GraphNode] = field(default_factory=list)

    @property
    def event_ids(self) -> list[int]:
        return [int(n.event_id) for n in self.nodes]

    @property
    def n_events(self) -> int:
        return len(self.nodes)

    @property
    def t0(self) -> int:
        return min(int(n.t0) for n in self.nodes)

    @property
    def t1(self) -> int:
        return max(int(n.t1) for n in self.nodes)

    def events(self) -> list[BlockEvent]:
        return [node_to_event(n) for n in self.nodes]

    def mean_motion(self) -> tuple[float, float, float]:
        return mean_motion(self.nodes)


def event_pair_features(
    a: GraphNode,
    b: GraphNode,
    *,
    kind: str,
    sigma_xy: float,
    sigma_t: float,
    w_xy: float,
    w_t: float,
    w_dir: float,
    tau_graph: float,
    r_dir_strong: float,
) -> dict:
    """c_event = w_xy*S_xy + w_t*S_t - w_dir*R_dir - tau_graph."""
    d_xy = chebyshev_xy(a, b)
    gap = temporal_gap_frames(a, b)
    support = temporal_overlap_over_min_duration(a, b)
    s_xy = float(spatial_similarity(float(d_xy), sigma_xy=float(sigma_xy)))
    s_t = float(support) if support > 0.0 else float(
        temporal_similarity_gap(int(gap), sigma_t=float(sigma_t))
    )
    cos = cosine_similarity(a.u, a.v, b.u, b.v)
    s_v = float(0.5 * (1.0 + cos))
    r_dir = float(min(float(a.coherence), float(b.coherence)) * (1.0 - s_v))
    c_event = (
        float(w_xy) * s_xy
        + float(w_t) * s_t
        - float(w_dir) * r_dir
        - float(tau_graph)
    )
    i, j = (a.node_id, b.node_id) if a.node_id < b.node_id else (b.node_id, a.node_id)
    strong = bool(c_event < 0.0 and r_dir >= float(r_dir_strong))
    return {
        "i": int(i),
        "j": int(j),
        "event_i": int(a.event_id),
        "event_j": int(b.event_id),
        "kind": kind,
        "d_xy": int(d_xy),
        "temporal_gap": int(gap),
        "s_xy": float(s_xy),
        "s_support": float(support),
        "s_t": float(s_t),
        "s_v": float(s_v),
        "c_i": float(a.coherence),
        "c_j": float(b.coherence),
        "r_dir": float(r_dir),
        "c_event": float(c_event),
        "strong_repulsive": strong,
        "u_i": float(a.u),
        "v_i": float(a.v),
        "u_j": float(b.u),
        "v_j": float(b.v),
    }


def build_local_event_edges(
    nodes: list[GraphNode],
    *,
    chebyshev: int = SPATIAL_CHEBYSHEV_MAX,
    temporal_gap: int = TEMPORAL_GAP_MAX,
    sigma_xy: float = SIGMA_XY,
    sigma_t: float = SIGMA_T,
    w_xy: float = W_XY,
    w_t: float = W_T,
    w_dir: float = W_DIR,
    tau_graph: float = TAU_SPATIAL,
    r_dir_strong: float = R_DIR_STRONG,
    use_temporal_bridge: bool = USE_TEMPORAL_BRIDGE,
) -> tuple[list[GraphEdge], list[dict]]:
    """Local candidates only: overlap+Chebyshev, optional gap bridges."""
    rows: list[dict] = []
    edges: list[GraphEdge] = []
    n = len(nodes)
    cheb = int(chebyshev)
    gap_max = int(temporal_gap)
    for ia in range(n):
        a = nodes[ia]
        for ib in range(ia + 1, n):
            b = nodes[ib]
            d_xy = chebyshev_xy(a, b)
            if d_xy > cheb:
                continue
            support = temporal_overlap_over_min_duration(a, b)
            gap = temporal_gap_frames(a, b)
            kind = None
            if support > 0.0:
                kind = "spatial"
            elif use_temporal_bridge and 1 <= gap <= gap_max:
                kind = "temporal_bridge"
            if kind is None:
                continue
            row = event_pair_features(
                a,
                b,
                kind=kind,
                sigma_xy=sigma_xy,
                sigma_t=sigma_t,
                w_xy=w_xy,
                w_t=w_t,
                w_dir=w_dir,
                tau_graph=tau_graph,
                r_dir_strong=r_dir_strong,
            )
            rows.append(row)
            edges.append(
                GraphEdge(
                    i=int(row["i"]),
                    j=int(row["j"]),
                    kind=kind,
                    d_xy=float(row["d_xy"]),
                    temporal_iou=float(row["s_support"]),
                    temporal_gap=int(row["temporal_gap"]),
                    s_xy=float(row["s_xy"]),
                    s_t=float(row["s_t"]),
                    s_v=float(row["s_v"]),
                    s=float(w_xy) * float(row["s_xy"]) + float(w_t) * float(row["s_t"]),
                    cost=float(row["c_event"]),
                    r_dir=float(row["r_dir"]),
                )
            )
    return edges, rows


__all__ = [
    "GraphComponent",
    "build_local_event_edges",
    "event_pair_features",
    "mean_motion",
    "node_to_event",
]
