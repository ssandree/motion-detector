"""Light conservative split: local attractive Union-Find (no coarse group, no KL).

Same signed cost as exp_v2c Graph-cut:
  c = w_xy*S_xy + w_t*S_t - w_dir*R_dir - tau_graph
Then: cost > 0 → union, cost ≤ 0 → leave split.

Edge candidates use a spatial grid hash (Chebyshev window) instead of
an all-pairs scan.
"""
from __future__ import annotations

from collections import defaultdict

from stage3.components import GraphComponent, event_pair_features
from stage3_graph.config import SIGMA_T, SIGMA_XY, W_T, W_XY
from stage3_graph.edges import temporal_gap_frames, temporal_overlap_over_min_duration
from stage3_graph.nodes import GraphNode

CONSERVATIVE_TAU_GRAPH = 0.25
CONSERVATIVE_W_DIR = 1.0
CONSERVATIVE_CHEBYSHEV = 2
CONSERVATIVE_TEMPORAL_GAP = 10


def iter_hashed_local_pairs(
    nodes: list[GraphNode],
    *,
    chebyshev: int,
    temporal_gap: int,
    use_temporal_bridge: bool = True,
) -> list[tuple[int, int, str]]:
    """Index pairs with Chebyshev ≤ chebyshev and spatial/bridge temporal support."""
    cheb = int(chebyshev)
    gap_max = int(temporal_gap)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, node in enumerate(nodes):
        buckets[(int(node.x), int(node.y))].append(i)

    pairs: list[tuple[int, int, str]] = []
    for ia, a in enumerate(nodes):
        ax, ay = int(a.x), int(a.y)
        for dx in range(-cheb, cheb + 1):
            for dy in range(-cheb, cheb + 1):
                for ib in buckets.get((ax + dx, ay + dy), ()):
                    if ib <= ia:
                        continue
                    b = nodes[ib]
                    support = temporal_overlap_over_min_duration(a, b)
                    gap = temporal_gap_frames(a, b)
                    kind = None
                    if support > 0.0:
                        kind = "spatial"
                    elif use_temporal_bridge and 1 <= gap <= gap_max:
                        kind = "temporal_bridge"
                    if kind is None:
                        continue
                    pairs.append((ia, ib, kind))
    return pairs


def split_attractive_unionfind(
    nodes: list[GraphNode],
    *,
    chebyshev: int = CONSERVATIVE_CHEBYSHEV,
    temporal_gap: int = CONSERVATIVE_TEMPORAL_GAP,
    sigma_xy: float = SIGMA_XY,
    sigma_t: float = SIGMA_T,
    w_xy: float = W_XY,
    w_t: float = W_T,
    w_dir: float = CONSERVATIVE_W_DIR,
    tau_graph: float = CONSERVATIVE_TAU_GRAPH,
    use_temporal_bridge: bool = True,
) -> tuple[list[GraphComponent], dict]:
    """Partition nodes by union-find on attractive (cost>0) local edges."""
    n = len(nodes)
    if n == 0:
        return [], {
            "num_nodes": 0,
            "num_edges": 0,
            "num_attractive": 0,
            "num_repulsive": 0,
            "num_comps": 0,
        }

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pairs = iter_hashed_local_pairs(
        nodes,
        chebyshev=chebyshev,
        temporal_gap=temporal_gap,
        use_temporal_bridge=use_temporal_bridge,
    )
    n_attr = 0
    n_rep = 0
    for ia, ib, kind in pairs:
        row = event_pair_features(
            nodes[ia],
            nodes[ib],
            kind=kind,
            sigma_xy=sigma_xy,
            sigma_t=sigma_t,
            w_xy=w_xy,
            w_t=w_t,
            w_dir=w_dir,
            tau_graph=tau_graph,
            r_dir_strong=0.5,
        )
        if float(row["c_event"]) > 0.0:
            union(ia, ib)
            n_attr += 1
        else:
            n_rep += 1

    groups: dict[int, list[GraphNode]] = {}
    for i, node in enumerate(nodes):
        groups.setdefault(find(i), []).append(node)

    comps: list[GraphComponent] = []
    for cid, members in enumerate(
        sorted(groups.values(), key=lambda ns: min(n.t0 for n in ns)),
        start=1,
    ):
        ordered = sorted(members, key=lambda n: (n.t0, n.y, n.x, n.event_id))
        comps.append(GraphComponent(component_id=cid, nodes=ordered))

    log = {
        "num_nodes": n,
        "num_edges": len(pairs),
        "num_attractive": n_attr,
        "num_repulsive": n_rep,
        "num_comps": len(comps),
        "params": {
            "chebyshev": int(chebyshev),
            "temporal_gap": int(temporal_gap),
            "tau_graph": float(tau_graph),
            "w_dir": float(w_dir),
            "w_xy": float(w_xy),
            "w_t": float(w_t),
        },
    }
    return comps, log
