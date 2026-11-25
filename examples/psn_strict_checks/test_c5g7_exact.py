import json
from fractions import Fraction as F
from pathlib import Path
from unittest import TestCase

from c5g7_exact import dot, group_weight_bound, read_materials, reflecting_eigenpair


HERE = Path(__file__).resolve().parent


class ExactC5G7Tests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chi, cls.materials = read_materials(HERE / "c5g7_materials_pure.yaml")

    def test_exact_group_balance_and_fission_eigen_equation(self):
        for material in self.materials:
            with self.subTest(material=material.name):
                self.assertEqual(len(material.total), 7)
                self.assertTrue(all(t > sum(row) for t, row in zip(material.total, material.scattering)))
                k, spectrum = reflecting_eigenpair(material, self.chi)
                self.assertTrue(all(phi >= 0 for phi in spectrum))
                self.assertEqual(tuple(dot(row, spectrum) for row in material.loss_matrix()), self.chi)
                if any(material.production):
                    self.assertGreater(k, 0)
                    for lhs, chi in zip((dot(row, spectrum) for row in material.loss_matrix()), self.chi):
                        self.assertEqual(k*lhs, chi*dot(material.production, spectrum))
                else:
                    self.assertEqual(k, 0)

    def test_chi_scaling_is_exact_in_each_fissile_material(self):
        scale = sum(self.chi)
        self.assertEqual(scale, F(100000917761, 100000000000))
        normalized = tuple(value / scale for value in self.chi)
        for material in self.materials:
            if any(material.production):
                k, flux = reflecting_eigenpair(material, self.chi)
                k_norm, flux_norm = reflecting_eigenpair(material, normalized)
                self.assertEqual(k, scale * k_norm)
                self.assertEqual(flux, tuple(scale * value for value in flux_norm))

    def test_group_balance_certificate_covers_all_materials_and_groups(self):
        data = json.loads((HERE / "group_weight_certificate.json").read_text())
        weights = tuple(map(F, data["weights"]))
        self.assertEqual(dot(self.chi, weights), 1)
        certified_bound = group_weight_bound(self.materials, self.chi, weights)
        self.assertEqual(certified_bound, F(data["bound"]))
        self.assertLess(certified_bound, F(947, 500))
        for material in self.materials:
            self.assertLessEqual(reflecting_eigenpair(material, self.chi)[0], certified_bound)
