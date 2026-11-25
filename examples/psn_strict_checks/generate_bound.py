"""Find and independently check a rational group-weight certificate."""

import argparse
import json
from fractions import Fraction
from pathlib import Path

from sympy import Rational
from sympy.solvers.simplex import linprog

from c5g7_exact import group_weight_bound, read_materials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("materials", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    chi, materials = read_materials(args.materials)
    constraints = []
    for material in materials:
        for g, production in enumerate(material.production):
            row = [s - (material.total[g] if g == h else 0)
                   for h, s in enumerate(material.scattering[g])]
            constraints.append(list(map(Rational, [*row, production])))
    objective = [0] * len(chi) + [-1]
    optimum, result = linprog(objective, constraints, [0] * len(constraints),
                             [list(map(Rational, [*chi, 0]))], [1])
    weights = tuple(Fraction(str(w)) for w in result[:-1])
    bound = group_weight_bound(materials, chi, weights)
    if bound != 1 / Fraction(str(-optimum)):
        raise ValueError("simplex result failed the exact certificate check")
    args.output.write_text(json.dumps(dict(weights=[str(w) for w in weights], bound=str(bound)), indent=2) + "\n")


if __name__ == "__main__":
    main()
