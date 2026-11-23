"""Exercise the packed backend against the original scalar transport path."""
from pathlib import Path
from unittest import TestCase, skipUnless

import numpy as np

from psn2d import memopt
from psn2d.solver import PSN2D


@skipUnless(Path(memopt._RUST_DIR_DEFAULT, "libpsn_chol.so").is_file(),
            "build rust/build.sh to exercise the packed backend")
class PackedCholTests(TestCase):
    def test_rectangular_sweep_and_ritz(self):
        def make():
            width = np.array([[1.0, 0.8], [1.0, 0.8]])
            return PSN2D(np.array([[0, 1], [1, 0]]), 1.0, [[1.0, 0.9]],
                         [[[0.6, 0.5]]], [[0.5, 0.32]], M=4, threads=1,
                         widths=(width, width.T),
                         boundary=("reflect", "reflect", "vacuum", "vacuum"))

        plain, packed = make(), make()
        report = memopt.install_optimized(packed, backend="chol-packed-rs")
        self.assertEqual(report["backend"], "chol-packed-rs")
        self.assertTrue(report["pool_gb_measured"])
        source = np.random.default_rng(17).normal(size=(1, 4, 5))
        for rhs in (source, np.zeros_like(source)):
            actual = packed._sweep_all_serial(rhs)
            expected = plain._sweep_all_serial(rhs)
            for observed, reference in zip(actual, expected):
                np.testing.assert_allclose(observed, reference, atol=1e-12, rtol=1e-11)
        k_packed, _, _ = packed.keff(ritz=True, verbose=False)
        k_plain, _, _ = plain.keff(ritz=True, verbose=False)
        self.assertAlmostEqual(k_packed, k_plain, places=11)
        self.assertLessEqual(packed.source_residual, 5e-10)
