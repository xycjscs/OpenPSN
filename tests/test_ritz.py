"""Regression tests for the transport eigenpair and sweep accounting."""
import sys
from pathlib import Path
from unittest import TestCase, main

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from psn2d.solver import PSN2D


class RitzTests(TestCase):
    def test_one_reflective_cell(self):
        psn = PSN2D(np.array([[0]]), 1.0, [[1.0]], [[[0.6]]], [[0.5]],
                    M=2, threads=1)
        k, phi, q = psn.keff(ritz=True, verbose=False, outer_tol=1e-10)
        self.assertAlmostEqual(k, 0.5 / (1.0 - 0.6), places=10)
        self.assertAlmostEqual(float(phi[0, 0]), k, places=10)
        self.assertEqual(q.shape, (1, 1, 5))
        self.assertEqual(psn.sweeps_executed, 2)
        self.assertEqual(psn.sweeps_executed,
                         psn.outer_iterations + psn.extra_sweeps)
        self.assertLessEqual(psn.source_residual, 5e-10)

    def test_nonuniform_rectangles_with_vacuum(self):
        def make():
            return PSN2D(np.array([[0, 1], [0, 0]]), 1.0, [[1.0, 0.9]],
                         [[[0.6, 0.5]]], [[0.5, 0.32]], M=2, threads=1,
                         widths=(np.array([[1.0, 0.8], [1.0, 0.8]]),
                                 np.array([[1.0, 1.0], [1.2, 1.2]])),
                         boundary=("reflect", "reflect", "vacuum", "vacuum"))

        reference, _, _ = make().keff(verbose=False, outer_tol=1e-11)
        psn = make()
        k, phi, q = psn.keff(ritz=True, verbose=False, outer_tol=1e-10)
        self.assertAlmostEqual(k, reference, places=8)
        self.assertTrue(np.all(phi > 0.0))
        replay_phi, replay_moments = psn._sweep_all_serial(q)
        matidx = psn.mat[psn.j_idx, psn.i_idx]
        updated = psn._source_update(replay_moments, k, matidx)
        np.testing.assert_array_equal(replay_phi, phi)
        self.assertLessEqual(np.linalg.norm(updated - q) / np.linalg.norm(q),
                             5e-10)

    def test_sweep_limit_includes_verification(self):
        psn = PSN2D(np.array([[0]]), 1.0, [[1.0]], [[[0.6]]], [[0.5]],
                    M=2, threads=1)
        with self.assertRaisesRegex(ValueError, "at least two sweeps"):
            psn.keff(ritz=True, verbose=False, max_outer=1)
        with self.assertRaisesRegex(ValueError, "standalone"):
            psn.keff(ritz=True, extrapolate=True, verbose=False)


if __name__ == "__main__":
    main()
