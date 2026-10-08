from __future__ import annotations

import unittest

import numpy as np

from stage2_exp.occupancy_surface import (
    expand_unit_occupancy,
    occupancy_from_magnitude,
    voxel_surface_quads,
)


class OccupancySurfaceTests(unittest.TestCase):
    def test_single_voxel_has_six_faces(self):
        occ = np.zeros((3, 3, 3), dtype=bool)
        occ[1, 1, 1] = True
        quads, normals = voxel_surface_quads(occ)
        self.assertEqual(len(quads), 6)
        self.assertEqual(len(normals), 6)

    def test_empty_has_no_faces(self):
        quads, normals = voxel_surface_quads(np.zeros((2, 2, 2), dtype=bool))
        self.assertEqual(len(quads), 0)
        self.assertEqual(len(normals), 0)

    def test_min_voxels_drops_speckle(self):
        mag = np.zeros((4, 4, 4), dtype=np.float32)
        mag[0, 0, 0] = 1.0
        mag[2:, 2:, 2:] = 1.0
        occ = occupancy_from_magnitude(mag, threshold=0.5, sigma=0.0, min_voxels=4)
        self.assertFalse(bool(occ[0, 0, 0]))
        self.assertTrue(bool(occ[2:, 2:, 2:].all()))

    def test_expand_unit_occupancy_repeats_4x4(self):
        occ = np.zeros((2, 2, 3), dtype=bool)
        occ[0, 0, 0] = True
        occ[1, 1, 2] = True
        painted = expand_unit_occupancy(occ, block=4)
        self.assertEqual(painted.shape, (2, 8, 12))
        self.assertTrue(bool(painted[0, 0:4, 0:4].all()))
        self.assertFalse(bool(painted[0, 0, 4]))
        self.assertTrue(bool(painted[1, 4:8, 8:12].all()))
        self.assertEqual(int(painted.sum()), 2 * 16)
        cropped = expand_unit_occupancy(occ, block=4, fine_hw=(7, 11))
        self.assertEqual(cropped.shape, (2, 7, 11))
        self.assertTrue(bool(cropped[0, 0:4, 0:4].all()))
        self.assertFalse(bool(cropped[0, 0, 4]))


if __name__ == "__main__":
    unittest.main()
