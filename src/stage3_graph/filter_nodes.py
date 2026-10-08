"""Pre-graph filters for Stage3_graph nodes."""

from __future__ import annotations

from stage3_graph.edges import chebyshev_xy, temporal_iou
from stage3_graph.nodes import GraphNode


def _reindex(nodes: list[GraphNode]) -> list[GraphNode]:
    out: list[GraphNode] = []
    for i, n in enumerate(nodes):
        out.append(
            GraphNode(
                node_id=i,
                event_id=n.event_id,
                x=n.x,
                y=n.y,
                t0=n.t0,
                t1=n.t1,
                u=n.u,
                v=n.v,
                mean_mag=n.mean_mag,
                coherence=float(n.coherence),
            )
        )
    return out


def has_active_neighbor(
    node: GraphNode,
    others: list[GraphNode],
    *,
    neigh_chebyshev: int = 2,
) -> bool:
    """True if some other event overlaps in time within Chebyshev radius."""
    r = int(neigh_chebyshev)
    for other in others:
        if other is node:
            continue
        if chebyshev_xy(node, other) > r:
            continue
        if temporal_iou(node, other) > 0.0:
            return True
    return False


def filter_isolated_single_block_nodes(
    nodes: list[GraphNode],
    *,
    neigh_chebyshev: int = 2,
) -> tuple[list[GraphNode], list[GraphNode]]:
    """Drop single-block events with no temporally-overlapping neighbor.

    Each Stage3 event is one unit block (≈64×64). Isolated ones are noise for
    VLM ROIs and are excluded from the graph entirely. Events that sit next to
    another active event (Chebyshev ≤ ``neigh_chebyshev``, time overlap) stay.
    """
    if not nodes:
        return [], []
    kept_raw: list[GraphNode] = []
    dropped: list[GraphNode] = []
    for node in nodes:
        if has_active_neighbor(node, nodes, neigh_chebyshev=neigh_chebyshev):
            kept_raw.append(node)
        else:
            dropped.append(node)
    return _reindex(kept_raw), dropped
