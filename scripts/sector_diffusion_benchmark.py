"""Run the PSN-4 square or rectangular benchmarks from the existing inputs.

Run from the repository root, for example:
    python -m scripts.sector_diffusion_benchmark strong --sm 2,4 4,8 8,12
    python -m scripts.sector_diffusion_benchmark rect --opt chol-rs --sm 1,4 1,8
"""
import argparse
import json
from pathlib import Path

import numpy as np

from psn2d.__main__ import run_case
from psn2d.model import expand_cases, graded_axis, load_spec
from scripts.study3_bilinear import CASES as ASSEMBLIES, EXAMPLES as C5G7_EXAMPLES


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
INPUTS = {
    "strong": EXAMPLES / "psn_repro/checkerboard_1g_strong.yaml",
    "weak": EXAMPLES / "psn_repro/checkerboard_1g.yaml",
    "rect": EXAMPLES / "c5g7/study2_rectangular/c5g7_rect_quarter_core.yaml",
}


def spatial_and_angular(value):
    try:
        s, m = map(int, value.split(","))
        if s > 0 and m >= 2 and m % 2 == 0:
            return s, m
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("expected S,M with S > 0 and even M >= 2")


def _remesh_core(geometry, step):
    """Refine the core while keeping the material interfaces at their coordinates."""
    material = np.asarray(geometry["mat_grid"])
    hx = np.asarray(geometry["widths_x"])
    width = hx[0]
    if (not np.array_equal(material, material.T)
            or not np.array_equal(hx, np.broadcast_to(width, material.shape))
            or not np.array_equal(hx.T, geometry["widths_y"])):
        raise ValueError("expected the symmetric rectangular quarter-core grid")

    # Coalesce only faces with identical materials along the entire axis.
    interfaces = np.flatnonzero(np.any(material[:, 1:] != material[:, :-1], axis=0))
    cuts = np.r_[0, interfaces + 1, len(width)]
    sizes = []
    for first, last in zip(cuts[:-1], cuts[1:]):
        span = sum(width[first:last])
        if first != cuts[-2] or np.unique(material[:, first:last]).size != 1:
            count = max(1, int(np.ceil(span / step)))
            sizes.append(np.full(count, span / count))
            continue

        # Space the outer water reflector more coarsely as the distance
        # from its interface with the core increases.
        distances = [0.0]
        while span - distances[-1] > step:
            distance = distances[-1]
            width_grown = step * np.exp(0.22 * max(0.0, distance - 2.0))
            size = max(step, min(width_grown, 0.75))
            distances.append(min(distance + size, span))
        steps = np.diff([*distances, span])
        if len(steps) > 1 and steps[-1] < step / 2:
            steps[-2] += steps[-1]
            steps = steps[:-1]
        sizes.append(steps)

    counts = [len(sizes_in_region) for sizes_in_region in sizes]
    widths = np.concatenate(sizes)
    np.testing.assert_allclose(np.add.reduceat(widths, np.r_[0, np.cumsum(counts)[:-1]]),
                               np.add.reduceat(width, cuts[:-1]), rtol=0, atol=1e-12)
    reduced = material[np.ix_(cuts[:-1], cuts[:-1])]
    remeshed = np.repeat(np.repeat(reduced, counts, axis=0), counts, axis=1)
    widths_x = np.broadcast_to(widths, remeshed.shape)
    geometry.update(mat_grid=remeshed, widths_x=widths_x, widths_y=widths_x.T)


def _redistribute_core(geometry, reflector_cells):
    """Refine the widest in-core intervals at a fixed Cartesian node count."""
    material = np.asarray(geometry["mat_grid"])
    hx = np.asarray(geometry["widths_x"])
    x = hx[0]
    if (not np.array_equal(material, material.T)
            or not np.array_equal(hx, np.broadcast_to(x, material.shape))
            or not np.array_equal(hx.T, geometry["widths_y"])):
        raise ValueError("expected the symmetric rectangular quarter-core grid")
    jumps = np.flatnonzero(np.any(material[:, :-1] != material[:, 1:], axis=0)) + 1
    cuts = np.r_[0, jumps, x.size]
    counts = np.diff(cuts)
    spans = np.add.reduceat(x, cuts[:-1])
    if (len(counts) < 2 or np.unique(material[:, cuts[-2]:]).size != 1
            or not 2 <= reflector_cells < counts[-1]):
        raise ValueError("reflector count must be at least two and smaller than the uniform outer reflector")
    steps = [x[first:last].copy() for first, last in zip(cuts[:-1], cuts[1:])]
    released = counts[-1] - reflector_cells
    counts[-1] = reflector_cells
    for _ in range(released):
        index = np.argmax([interval.max() for interval in steps[:-1]])
        counts[index] += 1
        steps[index] = np.full(counts[index], spans[index] / counts[index])
    # Both the core interface and the vacuum face can have thin layers.
    steps[-1] = graded_axis(spans[-1:], np.array([True, True]), reflector_cells, 0.75)
    grid = material[np.ix_(cuts[:-1], cuts[:-1])].repeat(counts, axis=0).repeat(counts, axis=1)
    widths_x = np.broadcast_to(np.concatenate(steps), grid.shape)
    geometry.update(mat_grid=grid, widths_x=widths_x, widths_y=widths_x.T)


def _redistribute_pin_strips(geometry, subdivisions):
    """Allocate each pin's fixed x/y budget among its three original intervals."""
    material = np.asarray(geometry["mat_grid"])

    def axis(widths):
        widths = widths.reshape(-1, 3)
        if not np.allclose(widths[:, 0], widths[:, 2], rtol=0, atol=1e-12):
            raise ValueError("pin redistribution requires paired edge strip widths")
        # Keep the outer strips paired and minimize the largest subcell.
        edge_counts = np.arange(1, (3 * subdivisions + 1) // 2)
        center_counts = 3 * subdivisions - 2 * edge_counts
        max_width = np.maximum(widths[:, :1] / edge_counts, widths[:, 1:2] / center_counts)
        best = np.argmin(max_width, axis=1)
        edge = edge_counts[best]
        counts = np.column_stack((edge, center_counts[best], edge)).ravel()
        return counts, np.repeat(widths.ravel() / counts, counts)

    cx, x = axis(np.asarray(geometry["widths_x"])[0])
    cy, y = axis(np.asarray(geometry["widths_y"])[:, 0])
    grid = material.repeat(cy, axis=0).repeat(cx, axis=1)
    return dict(geometry, mat_grid=grid, widths_x=np.broadcast_to(x, grid.shape),
                widths_y=np.broadcast_to(y[:, None], grid.shape))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=(*INPUTS, *ASSEMBLIES))
    parser.add_argument("--sm", nargs="+", type=spatial_and_angular, required=True)
    parser.add_argument("--closure", default="sector_diffusion",
                        choices=["sector_average", "sector_diffusion", "legacy", "strict_phase_space"])
    parser.add_argument("--opt", default="off",
                        choices=["off", "chol-rs", "chol-packed-rs", "sector-cholmod"])

    remesh = parser.add_mutually_exclusive_group()
    remesh.add_argument("--core-step", type=float, metavar="CM",
                        help="remesh the rectangular core with this target cell width")
    remesh.add_argument("--core-reflector-cells", type=int, metavar="N",
                        help="return outer reflector cells to the widest core strips")
    remesh.add_argument("--balance-pin", action="store_true",
                        help="redistribute subcells among the three intervals of each rectangular pin")
    remesh.add_argument("--interface-grading", type=float,
                        help="grade each subdivided interval toward material interfaces and vacuum faces")
    parser.add_argument("--polar-order", type=int)
    parser.add_argument("--azimuth-degree", type=int)
    parser.add_argument("--polar-degree", type=int, choices=(0, 1))
    parser.add_argument("--azimuth-grid", choices=("uniform", "axis"))
    parser.add_argument("--slope-space", choices=("quadratic", "bilinear", "quadratic-linear"))
    parser.add_argument("--threads", type=int)
    parser.add_argument("--mem-limit-gb", type=float)
    args = parser.parse_args()

    rect = args.model == "rect"
    assembly = args.model in ASSEMBLIES
    if assembly:
        path, reference = ASSEMBLIES[args.model]
        spec = load_spec(C5G7_EXAMPLES / path)
    else:
        spec = load_spec(INPUTS[args.model])
        reference = 1.187217554586068 if rect else expand_cases(spec)[0]["kref"]
    spec["solver"]["keff_tol"] = 1e-9
    if args.balance_pin and args.model not in ("rect_uo2", "rect_mox"):
        parser.error("--balance-pin requires rect_uo2 or rect_mox")

    if args.core_step is not None:
        if not rect or not np.isfinite(args.core_step) or args.core_step <= 0.0:
            parser.error("--core-step requires the rect input and a positive finite width")
        _remesh_core(spec["geometry"], args.core_step)
    if args.core_reflector_cells is not None:
        if not rect:
            parser.error("--core-reflector-cells requires the rect input")
        try:
            _redistribute_core(spec["geometry"], args.core_reflector_cells)
        except ValueError as exc:
            parser.error(str(exc))
    i = (8 if rect or assembly else 30) if args.polar_order is None else args.polar_order
    a = args.azimuth_degree
    if a is None:
        a = 0 if args.closure == "sector_average" else (1 if rect or assembly else 3)
    for s, m in args.sm:
        case = dict(name=f"{args.model}_S{s}_M{m}", model="generic",
                    I=i, A=a, M=m, subdivide=s, closure=args.closure,
                    kref=reference)
        if args.interface_grading is not None:
            case["interface_grading"] = args.interface_grading
        if args.slope_space is not None:
            case["slope_space"] = args.slope_space
        if args.polar_degree is not None or args.azimuth_grid is not None:
            if args.closure != "sector_average":
                parser.error("--polar-degree and --azimuth-grid require --closure sector_average")
            if args.polar_degree is not None:
                case["polar_degree"] = args.polar_degree
            if args.azimuth_grid is not None:
                case["azimuth_grid"] = args.azimuth_grid
        point = spec
        if args.balance_pin:
            point = dict(spec, geometry=_redistribute_pin_strips(spec["geometry"], s))
            case["subdivide"] = 1
        result = run_case(point, case, verbose=False, opt=args.opt, ritz=True,
                          threads=args.threads, mem_limit_gb=args.mem_limit_gb)
        if args.balance_pin:
            result.update(subdivide=s, balance_pin=True)
        if args.core_step is not None:
            result["core_step_cm"] = args.core_step
        if args.core_reflector_cells is not None:
            result["core_reflector_cells"] = args.core_reflector_cells
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
