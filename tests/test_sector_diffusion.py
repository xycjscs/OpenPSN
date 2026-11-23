"""Operator and eigenpair regressions for the sector diffusion closure."""
import sys
from pathlib import Path
from unittest import TestCase, main

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from psn2d.__main__ import run_case
from psn2d.model import load_spec
from psn2d.sector_diffusion import SectorAverage, SectorDiffusion, _directions, _sector_averages


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


class SectorAverageTests(TestCase):
    def test_finite_sector_tensor_and_boundary_average(self):
        # The first azimuth contains a finite interval of positive directions.
        sectors, bands = 6, 3
        cells = list(_sector_averages(sectors, bands, transpose_symmetry=False))
        self.assertEqual(len(cells), bands * (sectors // 2))
        self.assertAlmostEqual(sum(cell[-1] for cell in cells), 0.5)
        xx, xy, yy, vacuum_x, vacuum_y, weight = cells[0]
        azimuth_width = np.pi / sectors
        polar_start = np.cos(np.pi / (2 * bands))
        polar_width = 1 - polar_start
        p1 = quad(lambda u: np.sqrt(1 - u * u), polar_start, 1)[0] / polar_width
        p2 = quad(lambda u: 1 - u * u, polar_start, 1)[0] / polar_width
        expected = [quad(f, 0, azimuth_width)[0] / azimuth_width * p for f, p in (
            (lambda phi: np.cos(phi)**2, p2),
            (lambda phi: np.cos(phi) * np.sin(phi), p2),
            (lambda phi: np.sin(phi)**2, p2),
            (np.cos, p1),
            (np.sin, p1),
        )]
        np.testing.assert_allclose([xx, xy, yy, vacuum_x, vacuum_y], expected, rtol=1e-11)
        self.assertGreater(xx * yy - xy * xy, 0)
        self.assertAlmostEqual(weight, polar_width / sectors)

    def test_nested_refinement_on_strong_checkerboard(self):
        root = Path(__file__).resolve().parents[1]
        spec = load_spec(root / "examples/psn_repro/checkerboard_1g_strong.yaml")
        results = {}
        for s, m in ((2, 4), (2, 8), (4, 4), (4, 8)):
            case = dict(name="strong", closure="sector_average", model="generic",
                        M=m, I=30, subdivide=s, kref=0.51673)
            result = run_case(spec, case, verbose=False, threads=2)
            self.assertTrue(result["converged"])
            self.assertGreaterEqual(result["trial_sweeps"], 1)
            self.assertEqual(result["K"], 1)
            self.assertEqual(result["phase_systems"], 15 * (m // 4))
            self.assertLess(result["pcm"], 0)
            results[s, m] = result["keff"]
        self.assertGreater(results[2, 8], results[2, 4])
        self.assertGreater(results[4, 4], results[2, 4])
        self.assertGreater(results[4, 8], max(results[2, 8], results[4, 4]))

    def test_reflective_homogeneous_eigenpair(self):
        psn = SectorAverage(np.zeros((2, 2), dtype=int), 0.3,
                            [[1.0]], [[[0.6]]], [[0.5]],
                            M=6, I=4, threads=2)
        k, flux, _ = psn.keff(outer_tol=1e-10, verbose=False)
        self.assertAlmostEqual(k, 1.25, places=10)
        np.testing.assert_allclose(flux, k, rtol=0, atol=1e-10)

    def test_vacuum_slab_uses_averaged_tensor_and_leakage(self):
        thickness, cells = 1.6, 32
        psn = SectorAverage(np.zeros((cells, 1), dtype=int), thickness / cells,
                            [[1.0]], [[[0.6]]], [[0.5]], M=4, I=4, threads=1,
                            boundary=("reflect", "vacuum", "reflect", "vacuum"))
        psn._prefactor()
        source = np.zeros((1, cells, 9))
        source[:, :, 0] = 1
        flux, _ = psn._sweep_all(source)
        expected = 0.0
        for _, _, yy, _, vacuum_y, weight in psn.angles:
            diffusion_length = np.sqrt(yy)
            z = thickness / (2 * diffusion_length)
            amplitude = vacuum_y / (diffusion_length * np.sinh(z) + vacuum_y * np.cosh(z))
            expected += 2 * weight * (1 - amplitude * np.sinh(z) / z)
        self.assertAlmostEqual(float(flux.mean()), expected, places=6)


if __name__ == "__main__":
    main()
