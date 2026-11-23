"""Operator and eigenpair regressions for the sector diffusion closure."""
import sys
from pathlib import Path
from unittest import TestCase, main

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from psn2d.__main__ import run_case
from psn2d.model import load_spec
from psn2d.sector_diffusion import SectorDiffusion, _directions


class SectorDiffusionTests(TestCase):
    def test_reflective_homogeneous_eigenpair(self):
        psn = SectorDiffusion(np.zeros((2, 2), dtype=int), 0.3,
                              [[1.0]], [[[0.6]]], [[0.5]],
                              M=4, I=4, generic=True, threads=2)
        k, phi, source = psn.keff(outer_tol=1e-10, verbose=False)
        self.assertAlmostEqual(k, 1.25, places=10)
        np.testing.assert_allclose(phi, k, rtol=0, atol=1e-10)
        np.testing.assert_allclose(source[:, :, 0], phi, rtol=0, atol=1e-10)
        np.testing.assert_allclose(source[:, :, 1:], 0, rtol=0, atol=1e-10)
        self.assertLessEqual(psn.source_residual, 5e-10)
        self.assertEqual(psn.sweeps_executed, psn.outer_iterations + psn.extra_sweeps)

    def test_vacuum_slab_matches_analytic_angular_diffusion(self):
        thickness = 1.6
        cells = 32
        psn = SectorDiffusion(np.zeros((cells, 1), dtype=int), thickness / cells,
                              [[1.0]], [[[0.6]]], [[0.5]], M=4, I=4,
                              generic=True, threads=1, azimuth_degree=1,
                              boundary=("reflect", "vacuum", "reflect", "vacuum"))
        psn._prefactor()
        source = np.zeros((1, cells, 9))
        source[:, :, 0] = 1
        flux, _ = psn._sweep_all(source)
        expected = sum(2 * weight * (1 + y / thickness * np.expm1(-thickness / y))
                       for _, y, weight in psn.angles)
        self.assertAlmostEqual(float(flux.mean()), expected, places=6)

    def test_transpose_fold_matches_full_quadrature(self):
        # M=6 and A=0 include a point on the diagonal of the quadrant.
        mat = np.array([[0, 1], [1, 0]])
        args = (mat, 0.5, [[1.0, 1.2]], [[[0.6, 0.5]]], [[0.5, 0.0]])
        kwargs = dict(M=6, I=4, generic=True, threads=2, azimuth_degree=0,
                      boundary=("reflect", "reflect", "vacuum", "vacuum"))
        folded, full = (SectorDiffusion(*args, **kwargs) for _ in range(2))
        self.assertTrue(folded.transpose_symmetry)
        full.transpose_symmetry = False
        z, w = np.polynomial.legendre.leggauss(2)
        full.angles = list(_directions(6, 0, np.sqrt(1 - ((1 + z) / 2) ** 2),
                                       w / 2, transpose_symmetry=False))
        rng = np.random.default_rng(5)
        source = folded._transpose_average(rng.normal(size=(1, 4, 9)))
        folded._prefactor()
        full._prefactor()
        np.testing.assert_allclose(folded._sweep_all(source)[1],
                                   full._sweep_all(source)[1], atol=3e-14, rtol=1e-12)


    def test_strong_checkerboard_cancellation_points(self):
        root = Path(__file__).resolve().parents[1]
        spec = load_spec(root / "examples/psn_repro/checkerboard_1g_strong.yaml")
        errors = []
        for s, m in ((2, 4), (4, 8), (4, 12), (8, 12)):
            case = dict(name="strong", closure="sector_diffusion",
                        model="generic", M=m, I=30, A=3,
                        subdivide=s, kref=0.51673)
            result = run_case(spec, case, verbose=False, threads=2)
            self.assertTrue(result["converged"])
            errors.append(result["pcm"])
        self.assertTrue(all(error < 0 for error in errors))
        self.assertGreater(abs(errors[0]), abs(errors[1]))
        self.assertGreater(abs(errors[2]), abs(errors[3]))


if __name__ == "__main__":
    main()
