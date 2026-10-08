"""Candidate spatial / temporal edges and signed costs (no affinity)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stage3_graph.config import (
    EDGE_TAU,
    FLOW_EPS,
    SIGMA_T,
    SIGMA_XY,
    SPATIAL_CHEBYSHEV_MAX,
    TAU_SPATIAL,
    TEMPORAL_GAP_MAX,
    TEMPORAL_PRED_RADIUS,
    W_DIR,
    W_T,
    W_XY,
)
from stage3_graph.nodes import GraphNode


@dataclass(frozen=True)
class GraphEdge:
    i: int
    j: int
    kind: str  # "spatial" | "temporal"
    d_xy: float
    temporal_iou: float  # spatial: overlap/min(dur); temporal: 0
    temporal_gap: int
    s_xy: float
    s_t: float
    s_v: float
    s: float
    cost: float
    r_dir: float = 0.0  # spatial only: min(C_i,C_j)*(1-S_v)


def interval_intersection(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(int(a1), int(b1)) - max(int(a0), int(b0)) + 1)


def interval_union(a0: int, a1: int, b0: int, b1: int) -> int:
    inter = interval_intersection(a0, a1, b0, b1)
    len_a = int(a1) - int(a0) + 1
    len_b = int(b1) - int(b0) + 1
    return int(len_a + len_b - inter)


def temporal_iou(a: GraphNode, b: GraphNode) -> float:
    """IoU of time intervals (used only as overlap indicator / legacy)."""
    inter = interval_intersection(a.t0, a.t1, b.t0, b.t1)
    union = interval_union(a.t0, a.t1, b.t0, b.t1)
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def temporal_overlap_over_min_duration(a: GraphNode, b: GraphNode) -> float:
    """S_t base: overlap_frames / min(duration_i, duration_j) ∈ [0,1]."""
    inter = interval_intersection(a.t0, a.t1, b.t0, b.t1)
    if inter <= 0:
        return 0.0
    dur_a = int(a.t1) - int(a.t0) + 1
    dur_b = int(b.t1) - int(b.t0) + 1
    denom = min(dur_a, dur_b)
    if denom <= 0:
        return 0.0
    return float(inter) / float(denom)


def temporal_gap_frames(a: GraphNode, b: GraphNode) -> int:
    """0 if overlap/touch; else exclusive frames between intervals."""
    if interval_intersection(a.t0, a.t1, b.t0, b.t1) > 0:
        return 0
    if a.t1 + 1 == b.t0 or b.t1 + 1 == a.t0:
        return 0
    if a.t1 < b.t0:
        return int(b.t0 - a.t1 - 1)
    return int(a.t0 - b.t1 - 1)


def chebyshev_xy(a: GraphNode, b: GraphNode) -> int:
    return int(max(abs(a.x - b.x), abs(a.y - b.y)))


def euclidean_xy(a: GraphNode, b: GraphNode) -> float:
    return float(np.hypot(float(a.x - b.x), float(a.y - b.y)))


def cosine_similarity(u0: float, v0: float, u1: float, v1: float) -> float:
    n0 = float(np.hypot(u0, v0))
    n1 = float(np.hypot(u1, v1))
    if n0 < FLOW_EPS or n1 < FLOW_EPS:
        return 0.0  # → S_v = 0.5 after (1+cos)/2
    return float((u0 * u1 + v0 * v1) / (n0 * n1))


def motion_direction_similarity(a: GraphNode, b: GraphNode) -> float:
    cos = cosine_similarity(a.u, a.v, b.u, b.v)
    return float(0.5 * (1.0 + cos))


def spatial_similarity(d_xy: float, *, sigma_xy: float) -> float:
    s = max(float(sigma_xy), 1e-6)
    return float(np.exp(-(float(d_xy) ** 2) / (2.0 * s * s)))


def temporal_similarity_gap(gap: int, *, sigma_t: float) -> float:
    s = max(float(sigma_t), 1e-6)
    return float(np.exp(-float(gap) / s))


def direction_repulsion(a: GraphNode, b: GraphNode, *, s_v: float) -> float:
    """R_dir = min(C_i, C_j) * (1 - S_v); strong only when both are coherent."""
    return float(min(float(a.coherence), float(b.coherence)) * (1.0 - float(s_v)))


def temporal_edge_features(
    a: GraphNode,
    b: GraphNode,
    *,
    d_xy: float,
    gap: int,
    sigma_xy: float,
    sigma_t: float,
    tau: float,
) -> GraphEdge:
    """Temporal score: S=(S_xy+S_t+S_v)/3, c=S-tau; S_t from gap decay."""
    s_xy = spatial_similarity(d_xy, sigma_xy=sigma_xy)
    s_t = temporal_similarity_gap(gap, sigma_t=sigma_t)
    s_v = motion_direction_similarity(a, b)
    s = float((s_xy + s_t + s_v) / 3.0)
    cost = float(s - float(tau))
    i, j = (a.node_id, b.node_id) if a.node_id < b.node_id else (b.node_id, a.node_id)
    return GraphEdge(
        i=i,
        j=j,
        kind="temporal",
        d_xy=float(d_xy),
        temporal_iou=0.0,
        temporal_gap=int(gap),
        s_xy=float(s_xy),
        s_t=float(s_t),
        s_v=float(s_v),
        s=s,
        cost=cost,
        r_dir=0.0,
    )


def spatial_edge_features(
    a: GraphNode,
    b: GraphNode,
    *,
    d_xy: float,
    overlap_ratio: float,
    sigma_xy: float,
    w_xy: float,
    w_t: float,
    w_dir: float,
    tau_spatial: float,
) -> GraphEdge:
    """c = w_xy*S_xy + w_t*S_t - w_dir*R_dir - tau_spatial.

    S_t = overlap / min(duration_i, duration_j).
    """
    s_xy = spatial_similarity(d_xy, sigma_xy=sigma_xy)
    s_t = float(np.clip(overlap_ratio, 0.0, 1.0))
    s_v = motion_direction_similarity(a, b)
    r_dir = direction_repulsion(a, b, s_v=s_v)
    s = float(float(w_xy) * s_xy + float(w_t) * s_t)
    cost = float(s - float(w_dir) * r_dir - float(tau_spatial))
    i, j = (a.node_id, b.node_id) if a.node_id < b.node_id else (b.node_id, a.node_id)
    return GraphEdge(
        i=i,
        j=j,
        kind="spatial",
        d_xy=float(d_xy),
        temporal_iou=float(s_t),
        temporal_gap=0,
        s_xy=float(s_xy),
        s_t=float(s_t),
        s_v=float(s_v),
        s=s,
        cost=cost,
        r_dir=float(r_dir),
    )


def build_candidate_edges(
    nodes: list[GraphNode],
    *,
    cell_px: float,
    spatial_chebyshev: int = SPATIAL_CHEBYSHEV_MAX,
    temporal_gap_max: int = TEMPORAL_GAP_MAX,
    pred_radius: float = TEMPORAL_PRED_RADIUS,  # unused; kept for call-site compat
    sigma_xy: float = SIGMA_XY,
    sigma_t: float = SIGMA_T,
    tau: float = EDGE_TAU,
    w_xy: float = W_XY,
    w_t: float = W_T,
    w_dir: float = W_DIR,
    tau_spatial: float = TAU_SPATIAL,
) -> tuple[list[GraphEdge], list[GraphEdge], list[GraphEdge]]:
    """Return (spatial, temporal, affinity). Affinity is always empty.

    Spatial: time overlap + Chebyshev ≤ R.
    Temporal: no overlap, gap ∈ [1, gap_max], Chebyshev ≤ R (no flow gate).
    """
    del cell_px, pred_radius  # flow prediction gate removed
    spatial: list[GraphEdge] = []
    temporal: list[GraphEdge] = []
    affinity: list[GraphEdge] = []
    n = len(nodes)
    cheb_max = int(spatial_chebyshev)
    gap_max = int(temporal_gap_max)

    for ia in range(n):
        a = nodes[ia]
        for ib in range(ia + 1, n):
            b = nodes[ib]
            d_cheb = chebyshev_xy(a, b)
            iou = temporal_iou(a, b)
            gap = temporal_gap_frames(a, b)
            d_xy = euclidean_xy(a, b)

            # A. Local spatial edge (overlap + neighbour)
            if iou > 0.0 and d_cheb <= cheb_max:
                spatial.append(
                    spatial_edge_features(
                        a,
                        b,
                        d_xy=d_xy,
                        overlap_ratio=temporal_overlap_over_min_duration(a, b),
                        sigma_xy=sigma_xy,
                        w_xy=w_xy,
                        w_t=w_t,
                        w_dir=w_dir,
                        tau_spatial=tau_spatial,
                    )
                )
                continue

            # B. Temporal bridge: short gap + local neighbour (no flow gate)
            if iou <= 0.0 and 1 <= gap <= gap_max and d_cheb <= cheb_max:
                temporal.append(
                    temporal_edge_features(
                        a,
                        b,
                        d_xy=d_xy,
                        gap=gap,
                        sigma_xy=sigma_xy,
                        sigma_t=sigma_t,
                        tau=tau,
                    )
                )

    return spatial, temporal, affinity
