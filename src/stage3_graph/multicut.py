"""Signed multicut / correlation clustering via GAEC + Kernighan–Lin.

Environment note: the PyPI ``nifty`` wheel is a stub without ``nifty.graph``.
This module implements the standard CV approximation used when a native
multicut backend is unavailable:

  1) GAEC — greedy additive edge contraction on signed costs
  2) KL   — node-move local search on the same global objective

Objective (Minimum Cost Multicut form):
  minimize  sum_{(i,j)} c_ij * [cluster_i != cluster_j]
with c_ij = S_ij - tau  (attractive if >0, repulsive if <0).

This is *not* threshold-then-CC and *not* pairwise greedy merge without
additive cost updates.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass

from stage3_graph.config import KL_MAX_PASSES
from stage3_graph.edges import GraphEdge


@dataclass
class PartitionResult:
    labels: list[int]  # node_id → cluster_id (1-based contiguous)
    n_clusters: int
    objective: float
    solver: str


class _UF:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra


def _edge_map(edges: list[GraphEdge]) -> dict[tuple[int, int], float]:
    costs: dict[tuple[int, int], float] = {}
    for e in edges:
        key = (int(e.i), int(e.j)) if e.i < e.j else (int(e.j), int(e.i))
        prev = costs.get(key)
        if prev is None or float(e.cost) > float(prev):
            costs[key] = float(e.cost)
    return costs


def gaec_partition(n_nodes: int, edge_costs: dict[tuple[int, int], float]) -> list[int]:
    """Greedy Additive Edge Contraction on signed edge costs."""
    n = int(n_nodes)
    uf = _UF(n)
    neighbors: list[dict[int, float]] = [dict() for _ in range(n)]
    for (i, j), c in edge_costs.items():
        ii, jj = int(i), int(j)
        neighbors[ii][jj] = float(c)
        neighbors[jj][ii] = float(c)

    heap: list[tuple[float, int, int]] = [
        (-float(c), int(i), int(j)) for (i, j), c in edge_costs.items()
    ]
    heapq.heapify(heap)

    while heap:
        neg_c, a, b = heapq.heappop(heap)
        cost = -float(neg_c)
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue
        cur = neighbors[ra].get(rb)
        if cur is None:
            continue
        if abs(float(cur) - cost) > 1e-8:
            heapq.heappush(heap, (-float(cur), min(ra, rb), max(ra, rb)))
            continue
        if float(cur) <= 0.0:
            break

        root = uf.union(ra, rb)
        other = rb if root == ra else ra

        for nb, weight in list(neighbors[other].items()):
            nb_root = uf.find(nb)
            neighbors[nb_root].pop(other, None)
            if nb_root == root:
                continue
            new_w = float(neighbors[root].get(nb_root, 0.0)) + float(weight)
            neighbors[root][nb_root] = new_w
            neighbors[nb_root][root] = new_w
            heapq.heappush(heap, (-new_w, min(root, nb_root), max(root, nb_root)))

        neighbors[other].clear()
        neighbors[root].pop(other, None)

    return [uf.find(i) for i in range(n)]


def partition_objective(
    labels: list[int], edge_costs: dict[tuple[int, int], float]
) -> float:
    total = 0.0
    for (i, j), c in edge_costs.items():
        if labels[i] != labels[j]:
            total += float(c)
    return float(total)


def _relabel_contiguous(raw: list[int]) -> list[int]:
    mapping: dict[int, int] = {}
    out: list[int] = []
    next_id = 1
    for lab in raw:
        if lab not in mapping:
            mapping[lab] = next_id
            next_id += 1
        out.append(mapping[lab])
    return out


def kl_refine(
    labels: list[int],
    edge_costs: dict[tuple[int, int], float],
    *,
    max_passes: int = KL_MAX_PASSES,
) -> list[int]:
    """Kernighan–Lin style node moves minimizing signed multicut objective.

    Candidate destinations for v are neighboring clusters (via residual edges)
    plus a fresh singleton cluster — keeps local search tractable.
    """
    n = len(labels)
    labels = list(labels)
    nbrs: list[dict[int, float]] = [dict() for _ in range(n)]
    for (i, j), c in edge_costs.items():
        nbrs[i][j] = float(c)
        nbrs[j][i] = float(c)

    sizes: dict[int, int] = defaultdict(int)
    for lab in labels:
        sizes[lab] += 1
    next_id = max(sizes) + 1 if sizes else 1

    def move_delta(v: int, dest: int) -> float:
        src = labels[v]
        if dest == src:
            return 0.0
        delta = 0.0
        for u, c in nbrs[v].items():
            lu = labels[u]
            before_cut = 1.0 if lu != src else 0.0
            after_cut = 1.0 if lu != dest else 0.0
            delta += float(c) * (after_cut - before_cut)
        return float(delta)

    for _ in range(int(max_passes)):
        improved = False
        while True:
            best_v = -1
            best_dest = -1
            best_delta = 0.0
            for v in range(n):
                src = labels[v]
                dest_set = {labels[u] for u in nbrs[v] if labels[u] != src}
                dest_set.add(next_id)  # open a new cluster
                for dest in dest_set:
                    if dest == next_id and sizes.get(src, 0) <= 1:
                        continue
                    d = move_delta(v, dest)
                    if d < best_delta - 1e-12:
                        best_delta = d
                        best_v = v
                        best_dest = dest
            if best_v < 0:
                break
            src = labels[best_v]
            labels[best_v] = best_dest
            sizes[src] -= 1
            if sizes[src] <= 0:
                sizes.pop(src, None)
            sizes[best_dest] = sizes.get(best_dest, 0) + 1
            if best_dest == next_id:
                next_id += 1
            improved = True
        if not improved:
            break
    return labels

def solve_multicut(
    n_nodes: int,
    edges: list[GraphEdge],
    *,
    kl_passes: int = KL_MAX_PASSES,
) -> PartitionResult:
    """Partition nodes with GAEC + KL on signed edge costs."""
    n = int(n_nodes)
    costs = _edge_map(edges)
    if n <= 0:
        return PartitionResult(labels=[], n_clusters=0, objective=0.0, solver="gaec_kl")
    if not costs:
        labels = list(range(1, n + 1))
        return PartitionResult(
            labels=labels, n_clusters=n, objective=0.0, solver="gaec_kl"
        )

    raw = gaec_partition(n, costs)
    refined = kl_refine(raw, costs, max_passes=kl_passes)
    labels = _relabel_contiguous(refined)
    obj = partition_objective(labels, costs)
    return PartitionResult(
        labels=labels,
        n_clusters=len(set(labels)),
        objective=obj,
        solver="gaec_kl",
    )


def cluster_members(labels: list[int]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for node_id, lab in enumerate(labels):
        groups[int(lab)].append(int(node_id))
    return dict(sorted(groups.items()))
