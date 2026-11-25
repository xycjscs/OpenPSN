"""Rational C5G7 material tests using the input decimal values verbatim."""

import argparse
import json
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Material:
    name: str
    total: tuple[Fraction, ...]
    scattering: tuple[tuple[Fraction, ...], ...]  # [from][to]
    production: tuple[Fraction, ...]

    def loss_matrix(self):
        return [[self.total[g] * (g == h) - self.scattering[h][g]
                 for h in range(len(self.total))] for g in range(len(self.total))]


def read_materials(path):
    # BaseLoader keeps the original decimal spellings out of binary floats.
    with path.open() as stream:
        spectrum = yaml.load(stream, Loader=yaml.BaseLoader)
    chi = tuple(map(Fraction, spectrum["fission_spectrum"]))
    materials = [Material(m["name"], tuple(map(Fraction, m["total"])),
                          tuple(tuple(map(Fraction, row)) for row in m["scatter"]),
                          tuple(map(Fraction, m["nu_fission"]))) for m in spectrum["materials"]]
    return chi, materials


def dot(left, right):
    return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))


def solve(matrix, rhs):
    """Small exact linear solve with partial pivoting in rational arithmetic."""
    rows = [list(row) + [b] for row, b in zip(matrix, rhs, strict=True)]
    for i in range(len(rows)):
        pivot = max(range(i, len(rows)), key=lambda j: abs(rows[j][i]))
        rows[i], rows[pivot] = rows[pivot], rows[i]
        diagonal = rows[i][i]
        rows[i] = [a / diagonal for a in rows[i]]
        for j in range(len(rows)):
            if j != i and rows[j][i]:
                factor = rows[j][i]
                rows[j] = [a - factor*b for a, b in zip(rows[j], rows[i], strict=True)]
    return tuple(row[-1] for row in rows)


def reflecting_eigenpair(material, chi):
    loss = material.loss_matrix()
    flux = solve(loss, chi)
    k = dot(material.production, flux)
    return k, flux


def group_weight_bound(materials, chi, weights):
    """Check a positive group-balance certificate and return its exact bound."""
    if any(w <= 0 for w in weights):
        raise ValueError("group weights must be positive")
    source_weight = dot(chi, weights)
    ratios = []
    for material in materials:
        for g, production in enumerate(material.production):
            removal = weights[g] * material.total[g] - dot(material.scattering[g], weights)
            if removal < 0 or (production and removal == 0):
                raise ValueError(f"invalid bound for {material.name}, group {g+1}")
            if production:
                ratios.append(source_weight * production / removal)
    return max(ratios)


def pretty(value):
    """Decimal presentation of an already calculated rational value."""
    with localcontext() as ctx:
        ctx.prec = 25
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("materials", type=Path)
    parser.add_argument("--certificate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    chi, materials = read_materials(args.materials)
    output = dict(spectrum_sum=str(sum(chi)), exact_reflecting_eigenpairs={})
    for material in materials:
        if not any(material.production):
            continue
        k, flux = reflecting_eigenpair(material, chi)
        output["exact_reflecting_eigenpairs"][material.name] = dict(
            k=str(k), k_display=pretty(k), k_with_normalized_chi_display=pretty(k / sum(chi)),
            flux_for_unit_fission_source=[str(v) for v in flux],
        )
    weights = (Fraction(1),) * len(chi)
    if args.certificate:
        weights = tuple(map(Fraction, json.loads(args.certificate.read_text())["weights"]))
    bound = group_weight_bound(materials, chi, weights)
    output["heterogeneous_upper_bound_raw_chi"] = dict(
        weights=[str(w) for w in weights], bound=str(bound), bound_display=pretty(bound),
    )
    rendered = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
