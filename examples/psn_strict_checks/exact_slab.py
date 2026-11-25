"""Exact characteristic and moment responses for layered, purely absorbing slabs.

The angular flux is a finite sum of exp(-depth / mu) with rational optical
depths and coefficients. An integrated expression uses the same dict with
E_n(depth) instead, and depth zero stores a rational constant. No floating
point, quadrature, or angular truncation is used.
"""

from dataclasses import dataclass
from fractions import Fraction


Expression = dict[Fraction, Fraction]
ZERO = Fraction(0)
ONE = Fraction(1)
FULL_BAND = (ZERO, ONE)


@dataclass(frozen=True)
class Layer:
    depth: Fraction
    equilibrium: Fraction

    def __post_init__(self):
        if not isinstance(self.depth, Fraction) or not isinstance(self.equilibrium, Fraction):
            raise TypeError("optical depth and equilibrium source must be fractions")
        if self.depth <= 0:
            raise ValueError("a layer must have positive optical depth")


def add(*expressions: Expression) -> Expression:
    result = {}
    for expression in expressions:
        for depth, coefficient in expression.items():
            result[depth] = result.get(depth, ZERO) + coefficient
    return {depth: coefficient for depth, coefficient in result.items() if coefficient}


def scale(expression: Expression, factor: Fraction) -> Expression:
    return {depth: factor * coefficient for depth, coefficient in expression.items() if factor * coefficient}


def attenuate(expression: Expression, depth: Fraction) -> Expression:
    return {optical_depth + depth: coefficient for optical_depth, coefficient in expression.items()}


def advance(incoming: Expression, layer: Layer) -> Expression:
    """Apply the exact response in either direction of travel."""
    equilibrium = layer.equilibrium
    return add({ZERO: equilibrium}, attenuate(add(incoming, {ZERO: -equilibrium}), layer.depth))


def integrate(expression: Expression, order: int, band=FULL_BAND) -> Expression:
    """Integrate mu**(order-2) times the angular flux over a rational band."""
    lower, upper = band
    if not isinstance(lower, Fraction) or not isinstance(upper, Fraction):
        raise TypeError("angular band endpoints must be fractions")
    if not ZERO <= lower < upper <= ONE:
        raise ValueError("angular band must have positive width inside [0, 1]")

    def primitive(endpoint):
        if not endpoint:
            return {}
        factor = endpoint ** (order - 1)
        return {depth / endpoint: factor * (coefficient if depth else coefficient / (order - 1))
                for depth, coefficient in expression.items()}

    return add(primitive(upper), scale(primitive(lower), -ONE))


def directional_traces(layers: list[Layer]) -> tuple[list[Expression], list[Expression]]:
    """Return the right- and left-traveling traces at every face, with vacuum ends."""
    incoming_left, incoming_right = [{}], [{}]
    for layer in layers:
        incoming_left.append(advance(incoming_left[-1], layer))
    for layer in reversed(layers):
        incoming_right.append(advance(incoming_right[-1], layer))
    return incoming_left, incoming_right[::-1]


def slab_response(layers: list[Layer], band=FULL_BAND):
    """Return a band contribution to scalar flux, current, absorption and source."""
    rightward, leftward = directional_traces(layers)
    half = Fraction(1, 2)
    scalar = [scale(integrate(add(right, left), 2, band), half)
              for right, left in zip(rightward, leftward)]
    current = [scale(integrate(add(right, scale(left, -ONE)), 3, band), half)
               for right, left in zip(rightward, leftward)]
    absorption, source = [], []
    for i, layer in enumerate(layers):
        source.append(scale({ZERO: layer.depth}, layer.equilibrium * (band[1] - band[0])))
        deficit = add(rightward[i], leftward[i + 1], {ZERO: -2 * layer.equilibrium})
        # Integrate the directional solution along the node first. Its
        # nonconstant part integrates to mu * (1 - exp(-depth / mu)).
        integral = add(deficit, scale(attenuate(deficit, layer.depth), -ONE))
        absorption.append(add(source[-1], scale(integrate(integral, 3, band), half)))
    return scalar, current, absorption, source


def format_expression(expression: Expression, order: int) -> str:
    terms = []
    for depth, coefficient in sorted(expression.items()):
        magnitude = abs(coefficient)
        value = (f"{magnitude}*E_{order}({depth})" if depth else str(magnitude))
        terms.append((coefficient < 0, value))
    if not terms:
        return "0"
    negative, first = terms[0]
    return ("-" if negative else "") + first + "".join(
        (" - " if negative else " + ") + value for negative, value in terms[1:])


if __name__ == "__main__":
    slab = [Layer(Fraction(1), Fraction(1)), Layer(Fraction(1), Fraction(2))]
    scalar, current, absorption, source = slab_response(slab)
    leak = add(scale(current[0], Fraction(-1)), current[-1])
    print("Interface scalar flux =", format_expression(scalar[1], 2))
    print("Interface net current =", format_expression(current[1], 3))
    print("Total leakage =", format_expression(leak, 3))
    print("Total absorption =", format_expression(add(*absorption), 3))
    print("Absorption + leakage - source =",
          format_expression(add(*absorption, leak, scale(add(*source), Fraction(-1))), 3))
