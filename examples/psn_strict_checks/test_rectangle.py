from fractions import Fraction as F
from unittest import TestCase

from sympy import atan, cancel, diff, symbols

from strict_rectangle import Material, Rectangle


class ExactObliqueRayTests(TestCase):
    def setUp(self):
        self.mesh = Rectangle(x_edges=(F(0), F(1), F(2)), y_edges=(F(0), F(1), F(2)),
                              rows=((Material(F(2), F(1)), Material(F(4), F(0))),
                                    (Material(F(3), F(2)), Material(F(1), F(3)))))
        self.start = (F(0), F(1, 5))
        self.direction = (F(3, 5), F(4, 5))

    def test_oblique_ray_crosses_two_material_interfaces_exactly(self):
        end, hit, path = self.mesh.chord(self.start, self.direction)
        self.assertEqual((end, hit), ((F(27, 20), F(2)), (1,)))
        self.assertEqual([(p.depth, p.equilibrium) for p in path],
                         [(F(2), F(1)), (F(2), F(2)), (F(7, 12), F(3))])
        _, _, out = self.mesh.transmit(self.start, self.direction, {})
        self.assertEqual(out, {F(0): F(3), F(7, 12): F(-1), F(31, 12): F(-1), F(55, 12): F(-1)})

    def test_virtual_cuts_preserve_oblique_response(self):
        x = (F(0), F(1, 3), F(1), F(7, 5), F(2))
        y = (F(0), F(1, 2), F(1), F(4, 3), F(2))
        rows = tuple(tuple(self.mesh.material_at(((left + right)/2, (bottom + top)/2))
                           for left, right in zip(x[:-1], x[1:]))
                     for bottom, top in zip(y[:-1], y[1:]))
        new_mesh = Rectangle(x, y, rows)
        _, _, path = new_mesh.chord(self.start, self.direction)
        self.assertGreater(len(path), 3)
        incoming = {F(0): F(1, 7), F(3, 8): F(2, 5)}
        self.assertEqual(new_mesh.transmit(self.start, self.direction, incoming),
                         self.mesh.transmit(self.start, self.direction, incoming))

    def test_specular_top_reflection_passes_the_entire_outgoing_trace(self):
        end, hit, top_out = self.mesh.transmit(self.start, self.direction, {})
        self.assertEqual(hit, (1,))
        reflected = self.direction[0], -self.direction[1]
        right, hit, response = self.mesh.transmit(end, reflected, top_out)
        self.assertEqual((right, hit), ((F(2), F(17, 15)), (0,)))
        self.assertEqual(response, {F(0): F(3), F(5, 3): F(-1), F(11, 3): F(-1), F(17, 3): F(-1)})

    def test_internal_corner_is_a_single_event(self):
        _, hit, path = self.mesh.chord((F(0), F(0)), (F(1, 2), F(1, 2)))
        self.assertEqual(hit, (0, 1))
        self.assertEqual([(p.depth, p.equilibrium) for p in path], [(F(4), F(1)), (F(2), F(3))])


class ExactContinuousAngleRectangleTests(TestCase):
    def setUp(self):
        self.t, self.p, self.q, self.cx, self.sx, self.cy, self.sy = symbols("t p q cx sx cy sy")
        t, p, q, cx, sx, cy, sy = self.t, self.p, self.q, self.cx, self.sx, self.cy, self.sy
        on_plus = (t*(cx*cy-sx*sy) + (p+q)*(sx*cy+cx*sy)) / (t*t+(p+q)**2)
        on_minus = (t*(cx*cy+sx*sy) + (p-q)*(sx*cy-cx*sy)) / (t*t+(p-q)**2)
        self.psi = (on_plus + on_minus) / 2

    def stream(self, expression):
        p, q, cx, sx, cy, sy = self.p, self.q, self.cx, self.sx, self.cy, self.sy
        return p*(-sx*diff(expression, cx) + cx*diff(expression, sx)) + q*(-sy*diff(expression, cy) + cy*diff(expression, sy))

    def test_transport_and_even_parity_hold_symbolically_for_continuous_direction(self):
        t, p, q = self.t, self.p, self.q
        q_space = self.cx * self.cy
        backward = self.psi.subs({p: -p, q: -q}, simultaneous=True)
        u = (self.psi + backward)/2
        w = (self.psi - backward)/2
        self.assertEqual(cancel(self.stream(self.psi) + t*self.psi - q_space), 0)
        self.assertEqual(cancel(w + self.stream(u)/t), 0)
        self.assertEqual(cancel(-self.stream(self.stream(u))/t + t*u - q_space), 0)

    def test_all_specular_faces_match_symbolically(self):
        for normal, cosine, sine in ((self.p, self.cx, self.sx), (self.q, self.cy, self.sy)):
            for face_cosine in (-1, 1):
                face = self.psi.subs({cosine: face_cosine, sine: 0})
                self.assertEqual(cancel(face - face.subs(normal, -normal)), 0)

    def test_angular_integral_and_isotropic_source_closure(self):
        t, length, mu, scatter, source = symbols("t length mu scatter source", positive=True)
        primitive = atan(length*mu/t) / length
        self.assertEqual(cancel(diff(primitive, mu) - t/(t*t+length*length*mu*mu)), 0)
        angular_response = primitive.subs(mu, 1) - primitive.subs(mu, 0)
        flux = source * angular_response / (1 - scatter * angular_response)
        self.assertEqual(cancel(flux - (source + scatter * flux) * angular_response), 0)
