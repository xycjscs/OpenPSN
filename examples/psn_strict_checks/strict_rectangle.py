"""Exact oblique characteristic response on a rational Cartesian mesh."""

from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction

from exact_slab import Expression, Layer, advance


Point = tuple[Fraction, Fraction]
ZERO = Fraction(0)


@dataclass(frozen=True)
class Material:
    total: Fraction
    source_over_total: Fraction

    def __post_init__(self):
        if not isinstance(self.total, Fraction) or not isinstance(self.source_over_total, Fraction):
            raise TypeError("material parameters must be rational")
        if self.total <= 0:
            raise ValueError("total cross section must be positive")


@dataclass(frozen=True)
class Rectangle:
    x_edges: tuple[Fraction, ...]
    y_edges: tuple[Fraction, ...]
    rows: tuple[tuple[Material, ...], ...]

    def __post_init__(self):
        for edges in (self.x_edges, self.y_edges):
            if any(not isinstance(edge, Fraction) for edge in edges):
                raise TypeError("rectangle edges must be rational")
            if len(edges) < 2 or any(b <= a for a, b in zip(edges, edges[1:])):
                raise ValueError("rectangle edges must increase")
        if len(self.rows) != len(self.y_edges) - 1 or any(len(row) != len(self.x_edges) - 1
                                                          for row in self.rows):
            raise ValueError("material array does not match the rectangle")

    def material_at(self, point: Point) -> Material:
        x = bisect_right(self.x_edges, point[0]) - 1
        y = bisect_right(self.y_edges, point[1]) - 1
        return self.rows[y][x]

    def chord(self, start: Point, direction: Point) -> tuple[Point, tuple[int, ...], list[Layer]]:
        """Clip the next rectangle chord, measuring distance along unit Ω."""
        if any(not isinstance(value, Fraction) for value in (*start, *direction)):
            raise TypeError("ray origin and direction must be rational")
        if not direction[0] and not direction[1]:
            raise ValueError("direction needs a nonzero in-plane projection")
        if sum(d*d for d in direction) > 1:
            raise ValueError("in-plane direction has norm greater than one")
        edges = self.x_edges, self.y_edges
        for axis_edges, coordinate, component in zip(edges, start, direction):
            if not axis_edges[0] <= coordinate <= axis_edges[-1]:
                raise ValueError("ray starts outside the rectangle")
            if not component and coordinate in (axis_edges[0], axis_edges[-1]):
                raise ValueError("ray lies along a physical boundary")
        exits = {
            axis: (axis_edges[-1 if component > 0 else 0] - coordinate) / component
            for axis, (axis_edges, coordinate, component) in enumerate(zip(edges, start, direction))
            if component
        }
        end = min(exits.values())
        if end <= 0:
            raise ValueError("starting direction must enter the rectangle")
        cuts = {ZERO, end}
        for axis_edges, coordinate, component in zip(edges, start, direction):
            if component:
                for edge in axis_edges[1:-1]:
                    t = (edge - coordinate) / component
                    if ZERO < t < end:
                        cuts.add(t)
        ordered = sorted(cuts)

        def point_at(distance):
            return start[0] + direction[0]*distance, start[1] + direction[1]*distance

        path = []
        for a, b in zip(ordered[:-1], ordered[1:]):
            material = self.material_at(point_at((a + b) / 2))
            path.append(Layer(material.total*(b - a), material.source_over_total))
        return point_at(end), tuple(axis for axis, hit in exits.items() if hit == end), path

    def transmit(self, start: Point, direction: Point, incoming: Expression):
        """Compose all exact subinterval responses as sums of exp(-optical_depth)."""
        end, hit, path = self.chord(start, direction)
        for layer in path:
            incoming = advance(incoming, layer)
        return end, hit, incoming
