"""Checkpointed S/M sweep for the bilinear finite-sector C5G7 Study III."""
import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander

from psn2d.__main__ import build_solver
from psn2d.model import load_spec


EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "c5g7"
M_VALUES = (2, 4, 8, 12, 16, 24)
CASES = {
    "square_uo2": ("study1_homogenised/c5g7_uo2_assembly_bww_om.yaml", 1.3336285),
    "square_mox": ("study1_homogenised/c5g7_mox_assembly_bww.yaml", 1.1853993),
    "rect_uo2": ("study2_rectangular/c5g7_rect_uo2_assembly.yaml", 1.3347260),
    "rect_mox": ("study2_rectangular/c5g7_rect_mox_assembly.yaml", 1.1851053),
}


def overlap(old, new):
    points, weights = leggauss(3)
    scale = np.sqrt(2 * np.arange(3) + 1)
    transfer = np.zeros((new * 3, old * 3))
    for i in range(old):
        for j in range(new):
            lo, hi = max(i / old, j / new), min((i + 1) / old, (j + 1) / new)
            if lo < hi:
                x = (hi + lo + (hi - lo) * points) / 2
                a = legvander(2 * (old * x - i) - 1, 2) * scale
                b = legvander(2 * (new * x - j) - 1, 2) * scale
                transfer[3*j:3*j+3, 3*i:3*i+3] = b.T @ (weights[:, None] * a) * (hi-lo)*new/2
    return transfer


def project_source(q, old, new, base_shape):
    if old == new:
        return q
    ng, by, bx = q.shape[0], *base_shape
    # Each base cell has a separate tensor projection on the two axes.
    blocks = (q.reshape(ng, by, old, bx, old, 3, 3)
              .transpose(0, 1, 3, 2, 5, 4, 6)
              .reshape(ng, by, bx, old * 3, old * 3))
    transfer = overlap(old, new)
    mapped = transfer @ blocks @ transfer.T
    return (mapped.reshape(ng, by, bx, new, 3, new, 3)
            .transpose(0, 1, 3, 2, 5, 4, 6)
            .reshape(ng, by * bx * new * new, 9))


def point(directory, s, m):
    return directory / f"s{s}_m{m}.npz"


def run(args, *, backend_label=None):
    relpath, reference = CASES[args.case]
    spec = load_spec(EXAMPLES / relpath)
    # The published checkpoints are the sector_average ones; any other closure
    # keeps its own directory so a rerun cannot overwrite a stored point.
    directory = args.output / args.case if args.closure == "sector_average" \
        else args.output / f"{args.case}_{args.closure}"
    directory.mkdir(parents=True, exist_ok=True)
    geometry = spec["geometry"]
    rect = geometry.get("rect", False)
    base_shape = np.asarray(geometry["mat_grid" if rect else "grid"]).shape
    for s in args.s:
        for m in args.m:
            path = point(directory, s, m)
            original = None
            if path.exists():
                if not args.refresh_map:
                    continue
                with np.load(path, allow_pickle=False) as saved:
                    if "pin_source" in saved:
                        continue
                    original = json.loads(saved["info"].item())
            start = perf_counter()
            case = dict(name=args.case, model="generic", closure=args.closure, I=8,
                        M=m, A=args.azimuth_degree, subdivide=s, kref=reference)
            if args.closure == "sector_average":
                case["slope_space"] = "bilinear"
            psn = build_solver(spec, case, threads=args.threads, mem_limit_gb=args.memory,
                               opt=args.backend)
            near = [(abs(s - r) + abs(np.log2(m / a)), r, a)
                    for r in range(1, 7) for a in M_VALUES if point(directory, r, a).exists()]
            initial = None
            if near:
                _, old_s, old_m = min(near)
                with np.load(point(directory, old_s, old_m), allow_pickle=False) as saved:
                    initial = project_source(saved["source"], old_s, s, base_shape)
            print(f"{args.case} S={s} M={m} starting", flush=True)
            k, flux, source = psn.keff(outer_tol=args.tol, verbose=True,
                                       initial_source=initial)
            area = psn.mesh.area
            block = s * (3 if rect else 1)
            pin_area = area.reshape(17, block, 17, block).sum(axis=(1, 3))
            weighted_flux = flux * area
            pin_flux = weighted_flux.reshape(psn.ng, 17, block, 17, block).sum(axis=(2, 4)) / pin_area
            reaction = np.sum(weighted_flux * psn.nuSf[:, psn.mat.ravel()], axis=0)
            pin_source = reaction.reshape(17, block, 17, block).sum(axis=(1, 3)) / pin_area
            row = dict(case=args.case, S=s, M=m, I=8, A=args.azimuth_degree, keff=k,
                       reference=reference, closure=args.closure,
                       pcm=(k-reference)*1e5, seconds=perf_counter()-start,
                       factor_seconds=psn.factor_seconds, sweeps=psn.sweeps_executed,
                       source_residual=psn.source_residual, keff_residual=psn.keff_residual,
                       backend=backend_label or args.backend, threads=args.threads,
                       tolerance=args.tol, factor_bytes=psn.factor_bytes)
            if hasattr(psn, "packed_rtol"):
                row.update(pcg_rtol=psn.packed_rtol, pcg_maxiter=psn.packed_maxiter)
            if original is not None:
                if abs(k - original["keff"]) > 1e-7:
                    raise RuntimeError("pin-map rerun did not reproduce the stored eigenvalue")
                row = dict(original, field_seconds=row["seconds"],
                           field_source_residual=row["source_residual"])
            stream = NamedTemporaryFile(dir=directory, prefix=path.stem + "-", suffix=".partial",
                                        delete=False)
            temporary = Path(stream.name)
            try:
                with stream:
                    np.savez(stream, source=source, pin_flux=pin_flux, pin_source=pin_source,
                             info=json.dumps(row))
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            print(json.dumps(row), flush=True)


def main(*, backend_label=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--s", nargs="+", type=int, choices=range(1, 7), default=list(range(1, 7)))
    parser.add_argument("--m", nargs="+", type=int, choices=M_VALUES, default=list(M_VALUES))
    parser.add_argument("--threads", type=int, default=60)
    parser.add_argument("--memory", type=float, default=32)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--refresh-map", action="store_true",
                        help="recompute missing pin fields while preserving the original timing")
    parser.add_argument("--backend", default="sector-cholmod",
                        choices=("off", "sector-cholmod", "chol-packed-rs"))
    parser.add_argument("--closure", default="sector_average",
                        choices=("sector_average", "sector_diffusion"),
                        help="finite-sector closure; sector_average is the published one and keeps "
                             "the stored checkpoint directory, any other closure checkpoints "
                             "and records under its own name")
    parser.add_argument("--azimuth-degree", type=int, default=1,
                        help="A: local azimuthal modes per sector (sector_average) or extra Gauss "
                             "directions per sector minus one (sector_diffusion)")
    run(parser.parse_args(), backend_label=backend_label)


if __name__ == "__main__":
    main()
