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
from psn2d.model import expand_cases, load_spec


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=INPUTS)
    parser.add_argument("--sm", nargs="+", type=spatial_and_angular, required=True)
    parser.add_argument("--closure", default="sector_diffusion",
                        choices=["sector_average", "sector_diffusion", "legacy", "strict_phase_space"])
    parser.add_argument("--opt", default="off",
                        choices=["off", "chol-rs", "chol-packed-rs", "sector-cholmod"])

    parser.add_argument("--core-step", type=float, metavar="CM",
                        help="remesh the rectangular core with this target cell width")
    parser.add_argument("--polar-order", type=int)
    parser.add_argument("--azimuth-degree", type=int)
    parser.add_argument("--polar-degree", type=int, choices=(0, 1))
    parser.add_argument("--azimuth-grid", choices=("uniform", "axis"))
    parser.add_argument("--slope-space", choices=("quadratic", "bilinear", "quadratic-linear"))
    parser.add_argument("--threads", type=int)
    parser.add_argument("--mem-limit-gb", type=float)
    args = parser.parse_args()

    spec = load_spec(INPUTS[args.model])
    spec["solver"]["keff_tol"] = 1e-9
    rect = args.model == "rect"

    if args.core_step is not None:
        if not rect or not np.isfinite(args.core_step) or args.core_step <= 0.0:
            parser.error("--core-step requires the rect input and a positive finite width")
        _remesh_core(spec["geometry"], args.core_step)
    reference = 1.187217554586068 if rect else expand_cases(spec)[0]["kref"]
    i = (8 if rect else 30) if args.polar_order is None else args.polar_order
    a = args.azimuth_degree
    if a is None:
        a = 0 if args.closure == "sector_average" else (1 if rect else 3)
    for s, m in args.sm:
        case = dict(name=f"{args.model}_S{s}_M{m}", model="generic",
                    I=i, A=a, M=m, subdivide=s, closure=args.closure,
                    kref=reference)
        if args.slope_space is not None:
            case["slope_space"] = args.slope_space
        if args.polar_degree is not None or args.azimuth_grid is not None:
            if args.closure != "sector_average":
                parser.error("--polar-degree and --azimuth-grid require --closure sector_average")
            if args.polar_degree is not None:
                case["polar_degree"] = args.polar_degree
            if args.azimuth_grid is not None:
                case["azimuth_grid"] = args.azimuth_grid
        result = run_case(spec, case, verbose=False, opt=args.opt, ritz=True,
                          threads=args.threads, mem_limit_gb=args.mem_limit_gb)
        if args.core_step is not None:
            result["core_step_cm"] = args.core_step
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
