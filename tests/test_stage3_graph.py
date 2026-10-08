from __future__ import annotations

import unittest

import numpy as np

from stage3_graph.edges import (
    GraphEdge,
    build_candidate_edges,
    direction_repulsion,
    spatial_edge_features,
    temporal_gap_frames,
    temporal_iou,
    temporal_overlap_over_min_duration,
)
from stage3_graph.multicut import gaec_partition, partition_objective, solve_multicut
from stage3_graph.nodes import GraphNode, _mean_flow_mag_coherence
from stage3_graph.tubes import build_roi_tubes


def _node(
    nid: int,
    x: int,
    y: int,
    t0: int,
    t1: int,
    u: float = 0.0,
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


class MulticutTests(unittest.TestCase):
    def test_gaec_contracts_attractive_triangle(self):
        costs = {(0, 1): 0.4, (1, 2): 0.4, (0, 2): 0.3}
        labels = gaec_partition(3, costs)
        self.assertEqual(len(set(labels)), 1)

    def test_gaec_keeps_repulsive_cut(self):
        costs = {(0, 1): -0.4}
        labels = gaec_partition(2, costs)
        self.assertEqual(len(set(labels)), 2)
        self.assertLess(
            partition_objective(labels, costs),
            partition_objective([0, 0], costs),
        )

    def test_solve_multicut_on_edges(self):
        edges = [
            GraphEdge(0, 1, "spatial", 1, 0.5, 0, 0.8, 0.5, 0.9, 0.73, 0.23),
            GraphEdge(1, 2, "spatial", 1, 0.5, 0, 0.8, 0.5, 0.9, 0.73, 0.23),
            GraphEdge(0, 2, "spatial", 1, 0.2, 0, 0.2, 0.2, 0.1, 0.17, -0.33),
        ]
        part = solve_multicut(3, edges, kl_passes=5)
        self.assertEqual(part.n_clusters, len(set(part.labels)))
        self.assertEqual(len(part.labels), 3)


class EdgeTests(unittest.TestCase):
    def test_spatial_edge_when_overlap_and_near(self):
        nodes = [
            _node(0, 5, 5, 0, 10, u=1.0, v=0.0),
            _node(1, 6, 5, 5, 15, u=1.0, v=0.0),
            _node(2, 15, 15, 0, 10, u=0.0, v=1.0),
        ]
        spatial, temporal, affinity = build_candidate_edges(
            nodes, cell_px=64.0, tau=0.5, pred_radius=2.0
        )
        pairs = {(e.i, e.j) for e in spatial}
        self.assertIn((0, 1), pairs)
        self.assertNotIn((0, 2), pairs)
        self.assertEqual(affinity, [])
        self.assertEqual(temporal_iou(nodes[0], nodes[1]), 6 / 16)
        # S_t = overlap / min(dur): overlap=6, min(11,11)=11
        self.assertAlmostEqual(
            temporal_overlap_over_min_duration(nodes[0], nodes[1]), 6 / 11, places=5
        )
        self.assertAlmostEqual(spatial[0].s_t, 6 / 11, places=5)

    def test_no_affinity_for_longrange_similar_motion(self):
        # Chebyshev=4 would have been affinity before; now no edge.
        nodes = [
            _node(0, 5, 5, 0, 20, u=2.0, v=0.0),
            _node(1, 9, 5, 0, 20, u=2.0, v=0.0),
        ]
        spatial, temporal, affinity = build_candidate_edges(nodes, cell_px=64.0)
        self.assertEqual(spatial, [])
        self.assertEqual(temporal, [])
        self.assertEqual(affinity, [])

    def test_spatial_repulsion_only_when_both_coherent(self):
        a = _node(0, 5, 5, 0, 20, u=2.0, v=0.0, coherence=1.0)
        b = _node(1, 6, 5, 0, 20, u=-2.0, v=0.0, coherence=1.0)
        e = spatial_edge_features(
            a,
            b,
            d_xy=1.0,
            overlap_ratio=1.0,
            sigma_xy=1.5,
            w_xy=0.5,
            w_t=0.5,
            w_dir=0.5,
            tau_spatial=0.5,
        )
        self.assertAlmostEqual(e.s_v, 0.0, places=5)
        self.assertAlmostEqual(e.r_dir, 1.0, places=5)
        b_weak = _node(1, 6, 5, 0, 20, u=-2.0, v=0.0, coherence=0.0)
        e_weak = spatial_edge_features(
            a,
            b_weak,
            d_xy=1.0,
            overlap_ratio=1.0,
            sigma_xy=1.5,
            w_xy=0.5,
            w_t=0.5,
            w_dir=0.5,
            tau_spatial=0.5,
        )
        self.assertAlmostEqual(e_weak.r_dir, 0.0, places=5)
        self.assertGreater(e_weak.cost, e.cost)

    def test_direction_repulsion_formula(self):
        a = _node(0, 0, 0, 0, 1, u=1.0, v=0.0, coherence=0.8)
        b = _node(1, 1, 0, 0, 1, u=-1.0, v=0.0, coherence=0.5)
        self.assertAlmostEqual(direction_repulsion(a, b, s_v=0.0), 0.5, places=5)

    def test_temporal_gap(self):
        a = _node(0, 0, 0, 0, 5)
        b = _node(1, 0, 0, 8, 12)
        self.assertEqual(temporal_gap_frames(a, b), 2)

    def test_temporal_edge_local_neighbour_no_flow_gate(self):
        # Same block, gap=2 → temporal (no flow prediction required).
        nodes = [
            _node(0, 5, 5, 0, 5, u=0.0, v=0.0),
            _node(1, 5, 5, 8, 12, u=0.0, v=0.0),
        ]
        spatial, temporal, affinity = build_candidate_edges(nodes, cell_px=64.0)
        self.assertEqual(spatial, [])
        self.assertEqual(len(temporal), 1)
        self.assertEqual(temporal[0].kind, "temporal")
        self.assertEqual(affinity, [])

    def test_temporal_edge_requires_local_neighbour(self):
        # gap=2 but Chebyshev=4 → no temporal bridge.
        nodes = [
            _node(0, 5, 5, 0, 5, u=10.0, v=0.0),
            _node(1, 9, 5, 8, 12, u=10.0, v=0.0),
        ]
        _sp, temporal, affinity = build_candidate_edges(nodes, cell_px=64.0)
        self.assertEqual(temporal, [])
        self.assertEqual(affinity, [])

class CoherenceTests(unittest.TestCase):
    def test_coherent_motion_near_one(self):
        flow = np.zeros((5, 1, 1, 2), dtype=np.float32)
        flow[:, 0, 0, 0] = 2.0
        u, v, _mag, c = _mean_flow_mag_coherence(flow, None, y=0, x=0, t0=0, t1=4)
        self.assertAlmostEqual(u, 2.0, places=5)
        self.assertAlmostEqual(c, 1.0, places=5)

    def test_oscillating_motion_near_zero(self):
        flow = np.zeros((4, 1, 1, 2), dtype=np.float32)
        flow[0, 0, 0, 0] = 2.0
        flow[1, 0, 0, 0] = -2.0
        flow[2, 0, 0, 0] = 2.0
        flow[3, 0, 0, 0] = -2.0
        _u, _v, _mag, c = _mean_flow_mag_coherence(flow, None, y=0, x=0, t0=0, t1=3)
        self.assertLess(c, 0.05)


class TubeTests(unittest.TestCase):
    def test_bbox_is_lifetime_aabb(self):
        nodes = [
            _node(0, 1, 2, 0, 5),
            _node(1, 4, 2, 3, 8),
        ]
        tubes = build_roi_tubes(nodes, [1, 1])
        self.assertEqual(len(tubes), 1)
        self.assertEqual(tubes[0].bbox_at(1), (1, 2, 5, 3))
        self.assertEqual(tubes[0].bbox_at(4), (1, 2, 5, 3))
        self.assertEqual(tubes[0].bbox_at(7), (1, 2, 5, 3))
        self.assertIsNone(tubes[0].bbox_at(9))


class FilterTests(unittest.TestCase):
    def test_drops_isolated_singleton(self):
        from stage3_graph.filter_nodes import filter_isolated_single_block_nodes

        nodes = [
            _node(0, 1, 1, 0, 10, u=1.0),
            _node(1, 2, 1, 0, 10, u=1.0),
            _node(2, 15, 15, 0, 10, u=1.0),
        ]
        kept, dropped = filter_isolated_single_block_nodes(nodes, neigh_chebyshev=2)
        self.assertEqual(len(kept), 2)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0].x, 15)
        self.assertEqual(kept[0].coherence, 1.0)


class HeatmapOverlayTests(unittest.TestCase):
    def test_heatmap_tints_high_magnitude_cells(self):
        from stage3_graph.viz import draw_graph_roi_frame

        frame = np.full((128, 128, 3), 40, dtype=np.uint8)
        mag = np.zeros((2, 2), dtype=np.float32)
        mag[0, 0] = 3.5
        out = draw_graph_roi_frame(frame, [], t=0, cell_px=64, mag_frame=mag)
        self.assertFalse(np.array_equal(out[:64, :64], frame[:64, :64]))
        self.assertTrue(np.array_equal(out[64:, 64:], frame[64:, 64:]))


if __name__ == "__main__":
    unittest.main()
