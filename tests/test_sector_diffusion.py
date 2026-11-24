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

    def test_linear_azimuth_tensor_retains_cross_mode_coupling(self):
        sectors, bands = 4, 3
        tensors, _ = next(_sector_averages(sectors, bands, False, 1))
        mean, mixed, slope = tensors[0, 0], tensors[0, 1], tensors[1, 1]
        p2 = mean[0] + mean[2]
        step = np.pi / sectors

        def shape(phi):
            return np.sqrt(3) * (2 * phi / step - 1)

        expected_mixed = p2 * quad(lambda phi: shape(phi) * np.cos(phi)**2, 0, step)[0] / step
        expected_slope = p2 * quad(lambda phi: shape(phi)**2 * np.cos(phi)**2, 0, step)[0] / step
        p1 = mean[4] * step / (1 - np.cos(step))
        expected_vacuum = p1 * quad(lambda phi: shape(phi) * np.sin(phi), 0, step)[0] / step
        self.assertAlmostEqual(mixed[0], expected_mixed, places=13)
        self.assertAlmostEqual(slope[0], expected_slope, places=13)
        self.assertAlmostEqual(mixed[4], expected_vacuum, places=13)
        self.assertGreater(abs(mixed[0]), 1e-4)

    def test_linear_azimuth_transpose_fold_matches_full_cells(self):
        mat = np.array([[0, 1], [1, 0]])
        args = (mat, 0.5, [[1.0, 1.2]], [[[0.6, 0.5]]], [[0.5, 0.0]])
        options = dict(M=6, I=4, threads=1, azimuth_degree=1,
                       boundary=("reflect", "reflect", "vacuum", "vacuum"))
        folded, full = (SectorAverage(*args, **options) for _ in range(2))
        self.assertTrue(folded.transpose_symmetry)
        full.transpose_symmetry = False
        full.angles = list(_sector_averages(6, 2, False, 1))
        rng = np.random.default_rng(7)
        source = folded._transpose_average(rng.normal(size=(1, 4, 9)))
        for solver in (folded, full):
            solver._prefactor()
        np.testing.assert_allclose(folded._sweep_all(source)[1],
                                   full._sweep_all(source)[1], atol=3e-14, rtol=1e-12)

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

    def test_azimuth_enrichment_improves_strong_checkerboard_at_equal_cell_count(self):
        root = Path(__file__).resolve().parents[1]
        spec = load_spec(root / "examples/psn_repro/checkerboard_1g_strong.yaml")
        case = dict(name="strong", closure="sector_average", model="generic",
                    M=4, I=30, subdivide=4, kref=0.51673)
        mean, linear, quadratic = (
            run_case(spec, dict(case, A=degree), verbose=False, threads=1)
            for degree in (0, 1, 2)
        )
        self.assertTrue(all(row["converged"] for row in (mean, linear, quadratic)))
        self.assertEqual(mean["phase_systems"], linear["phase_systems"])
        self.assertEqual(mean["phase_systems"], quadratic["phase_systems"])
        self.assertEqual(quadratic["K"], 3)
        self.assertLess(mean["keff"], linear["keff"])
        self.assertLess(linear["keff"], quadratic["keff"])
        self.assertLess(quadratic["pcm"], 0)

    def test_bilinear_slope_is_a_conforming_restriction(self):
        mat = np.array([[0, 1]])
        args = (mat, 0.4, [[1.0, 1.2]], [[[0.6, 0.5]]], [[0.5, 0.0]])
        options = dict(M=4, I=4, threads=1, azimuth_degree=1,
                       boundary=("reflect", "vacuum", "vacuum", "reflect"))
        quadratic = SectorAverage(*args, **options)
        bilinear = SectorAverage(*args, **options, slope_space="bilinear")
        mesh = bilinear.mesh
        prolongation = np.zeros((mesh.size, mesh.bilinear_size))
        points = (0.0, 0.5, 1.0)
        shape = np.array([[(1-x) * (1-y), x * (1-y), (1-x) * y, x * y]
                          for y in points for x in points])
        for cells in (mesh.cells, mesh.mirror[mesh.cells]):
            for cell in cells:
                corners = mesh.bilinear_id[cell[[0, 2, 6, 8]]]
                prolongation[np.ix_(cell, corners)] = shape
        projection = np.zeros((2 * mesh.size, mesh.size + mesh.bilinear_size))
        projection[:mesh.size, :mesh.size] = np.eye(mesh.size)
        projection[mesh.size:, mesh.size:] = prolongation
        st = quadratic.St[0, mat.ravel()]
        full = quadratic._angular_matrix(st, quadratic.angles[0]).toarray()
        reduced = bilinear._angular_matrix(st, bilinear.angles[0]).toarray()
        np.testing.assert_allclose(reduced, projection.T @ full @ projection,
                                   atol=5e-16, rtol=1e-13)

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
