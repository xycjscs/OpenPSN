from fractions import Fraction as F
from unittest import TestCase

from exact_slab import Layer, add, directional_traces, scale, slab_response


class ExactSlabTests(TestCase):
    def test_two_materials_have_exact_interface_moments_and_global_balance(self):
        scalar, currents, absorption, source = slab_response([Layer(F(1), F(1)), Layer(F(1), F(2))])
        self.assertEqual(scalar[1], {F(0): F(3, 2), F(1): F(-3, 2)})
        self.assertEqual(currents[1], {F(0): F(-1, 4), F(1): F(1, 2)})
        leakage = add(scale(currents[0], F(-1)), currents[-1])
        self.assertEqual(leakage, {F(0): F(3, 4), F(2): F(-3, 2)})
        self.assertEqual(add(*absorption), {F(0): F(9, 4), F(2): F(3, 2)})
        self.assertEqual(add(*absorption, leakage), add(*source))

    def test_arbitrary_rational_layer_inputs_have_exact_cell_balance(self):
        layers = [Layer(F(3, 7), F(7, 3)), Layer(F(5, 4), F(0)), Layer(F(2, 3), F(4, 9))]
        _, currents, absorption, source = slab_response(layers)
        for i in range(len(layers)):
            balance = add(absorption[i], currents[i + 1], scale(currents[i], F(-1)))
            self.assertEqual(balance, source[i])

    def test_inserting_a_virtual_interface_preserves_both_directional_responses(self):
        a, b = Layer(F(11, 7), F(2, 3)), Layer(F(5, 11), F(4, 3))
        first = Layer(F(5, 7), a.equilibrium)
        second = Layer(a.depth - first.depth, a.equilibrium)
        old_right, old_left = directional_traces([a, b])
        new_right, new_left = directional_traces([first, second, b])
        self.assertEqual(old_right, [new_right[i] for i in (0, 2, 3)])
        self.assertEqual(old_left, [new_left[i] for i in (0, 2, 3)])
        old_result = slab_response([a, b])
        new_result = slab_response([first, second, b])
        self.assertEqual(add(*old_result[2]), add(*new_result[2]))

    def test_angular_band_split_is_exact_and_balanced_in_each_band(self):
        layers = [Layer(F(3, 7), F(7, 3)), Layer(F(5, 4), F(0)), Layer(F(2, 3), F(4, 9))]
        edges = [F(0), F(2, 7), F(4, 5), F(1)]
        parts = [slab_response(layers, band) for band in zip(edges[:-1], edges[1:])]
        for _, currents, absorption, source in parts:
            for i in range(len(layers)):
                balance = add(absorption[i], currents[i + 1], scale(currents[i], F(-1)))
                self.assertEqual(balance, source[i])
        for total, *split in zip(slab_response(layers), *parts):
            self.assertEqual(total, [add(*values) for values in zip(*split)])
