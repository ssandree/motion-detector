from __future__ import annotations

import unittest

from stage3.components import build_local_event_edges
from stage3_graph.nodes import GraphNode
from stage3.v2.light_split import iter_hashed_local_pairs, split_attractive_unionfind


def _node(
    nid: int,
    x: int,
    y: int,
    t0: int,
    t1: int,
    u: float = 1.0,
    v: float = 0.0,
    coherence: float = 1.0,
) -> GraphNode:
    return GraphNode(
        node_id=nid,
        event_id=nid + 1,
        x=x,
        y=y,
        t0=t0,
        t1=t1,
        u=u,
        v=v,
        mean_mag=1.0,
        coherence=coherence,
    )


class LightSplitTests(unittest.TestCase):
    def test_hashed_pairs_match_bruteforce_edges(self):
        nodes = [
            _node(0, 5, 5, 0, 10),
            _node(1, 6, 5, 5, 15),
            _node(2, 5, 5, 8, 12),
            _node(3, 12, 5, 0, 10),  # far: no edge
        ]
        hashed = {
            (i, j, k)
            for i, j, k in iter_hashed_local_pairs(
                nodes, chebyshev=2, temporal_gap=10, use_temporal_bridge=True
            )
        }
        edges, _rows = build_local_event_edges(
            nodes, chebyshev=2, temporal_gap=10, use_temporal_bridge=True
        )
        brute = set()
        id_to_idx = {n.node_id: i for i, n in enumerate(nodes)}
        for e in edges:
            ia, ib = id_to_idx[int(e.i)], id_to_idx[int(e.j)]
            if ia > ib:
                ia, ib = ib, ia
            brute.add((ia, ib, e.kind))
        self.assertEqual(hashed, brute)

    def test_same_direction_neighbours_merge(self):
        nodes = [
            _node(0, 5, 5, 0, 20, u=2.0, v=0.0),
            _node(1, 6, 5, 0, 20, u=2.0, v=0.0),
        ]
        comps, log = split_attractive_unionfind(nodes)
        self.assertEqual(len(comps), 1)
        self.assertGreater(log["num_attractive"], 0)

    def test_opposite_coherent_motion_stays_split(self):
        nodes = [
            _node(0, 5, 5, 0, 20, u=2.0, v=0.0, coherence=1.0),
            _node(1, 6, 5, 0, 20, u=-2.0, v=0.0, coherence=1.0),
        ]
        comps, log = split_attractive_unionfind(nodes)
        self.assertEqual(len(comps), 2)
        self.assertGreater(log["num_repulsive"], 0)


if __name__ == "__main__":
    unittest.main()
