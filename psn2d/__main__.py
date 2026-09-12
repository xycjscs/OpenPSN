# -*- coding: utf-8 -*-
"""PSN2D — diffusion-based phase-space nodal method, 2D multi-group.

Chao, Li & Chen, ANE 240 (2027) 112707 — independent Python implementation.
Run a YAML problem:

    python -m psn2d run examples/bwr_bundle_2g.yaml
    python -m psn2d run examples/c5g7_2d.yaml --case M16
    python -m psn2d run examples/checkerboard_1g.yaml --quick
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from . import model
from .solver import PSN2D


def build_solver(spec, case, threads=None, mem_limit_gb=None):
    geom = spec["geometry"]
    # runtime inputs: CLI > YAML solver.threads / solver.mem_limit_gb > default
    sol = spec.get("solver", {})
    threads = threads or sol.get("threads")
    mem_limit_gb = mem_limit_gb or sol.get("mem_limit_gb")
    if bool(geom.get("rect", False)):
        # rectangular nodes: per-node widths.  An optional per-case
        # `subdivide` S splits every node into S x S sub-nodes (widths / S,
        # material unchanged) — same kron recipe as the square path.  S=1 is
        # the identity (arrays pass through untouched -> bit-identical).
        mat_map0 = np.array(geom["mat_grid"], dtype=int)
        hx0 = np.asarray(geom["widths_x"], float)
        hy0 = np.asarray(geom["widths_y"], float)
        # scalar widths broadcast to the grid shape
        if hx0.ndim == 0:
            hx0 = np.full(mat_map0.shape, float(hx0))
            hy0 = np.full(mat_map0.shape, float(hy0))
        S = int(case.get("subdivide", 1))
        if S != 1:
            f = np.full((S, S), 1.0 / S)
            mat_map = np.kron(mat_map0, np.ones((S, S), int))
            widths = (np.kron(hx0, f), np.kron(hy0, f))
        else:
            mat_map, widths = mat_map0, (hx0, hy0)
        h = float(widths[0].ravel()[0])     # representative (unused by rect)
    else:
        grid = np.array(geom["grid"], dtype=int)
        S = int(case["subdivide"])
        mat_map = np.kron(grid, np.ones((S, S), int))
        h = float(geom["unit_size"]) / S
        widths = None
    St, Sgg, nuSf, chi = model.arrays(spec)
    bnd = spec["boundaries"]
    boundary = (bnd["left"], bnd["bottom"], bnd["right"], bnd["top"])
    generic = case["model"] == "generic"
    if generic:
        return PSN2D(mat_map, h, St, Sgg, nuSf, chi=chi,
                     boundary=boundary, generic=True, I=int(case["I"]),
                     M=int(case["M"]), widths=widths, threads=threads,
                     mem_limit_gb=mem_limit_gb)
    return PSN2D(mat_map, h, St, Sgg, nuSf, chi=chi,
                 boundary=boundary, generic=False, M=int(case["M"]),
                 widths=widths, threads=threads,
                 mem_limit_gb=mem_limit_gb)


def run_case(spec, case, verbose=True, opt="off", threads=None,
             mem_limit_gb=None):
    t0 = time.time()
    psn = build_solver(spec, case, threads=threads,
                       mem_limit_gb=mem_limit_gb)
    if opt != "off":
        from . import memopt
        report = memopt.install_optimized(psn, backend=opt)
    else:
        psn._memopt_backend = "plain"
        report = None
    sol = spec["solver"]
    k, phi, qnode = psn.keff(max_outer=int(sol["max_outer"]),
                             outer_tol=float(sol["keff_tol"]),
                             verbose=verbose and -1)
    dt = time.time() - t0
    out = {
        "name": case["name"],
        "model": case["model"],
        "I": int(case["I"]),
        "M": int(case["M"]),
        "subdivide": int(case["subdivide"]),
        "keff": float(k),
        "time_s": round(dt, 2),
        "nodes": psn.nodes,
    }
    if report is not None:
        out["memopt"] = report
    kref = case.get("kref")
    if kref is not None:
        out["kref"] = float(kref)
        out["pcm"] = float((k - kref) * 1e5)   # paper convention: absolute
    return out


def cmd_run(args):
    spec = model.load_spec(args.input)
    cases = model.expand_cases(spec)
    if args.case:
        cases = [c for c in cases if c["name"] == args.case]
        if not cases:
            print(f"case {args.case!r} not found; available: "
                  f"{[c['name'] for c in model.expand_cases(spec)]}")
            return 2
    results = []
    for case in cases:
        r = run_case(spec, case, verbose=not args.quiet, opt=args.opt,
                     threads=args.threads, mem_limit_gb=args.mem_limit_gb)
        results.append(r)
        if args.quiet:
            line = f"[{r['name']}] {r['model']} I={r['I']} M={r['M']} " \
                   f"S={r['subdivide']}: keff={r['keff']:.6f} ({r['time_s']} s)"
            if "pcm" in r:
                line += f"  ref={r['kref']:.5f}  Δ={r['pcm']:+.1f} pcm"
            if "memopt" in r:
                line += f"  [{r['memopt']['backend']}]"
            print(line, flush=True)
    if args.json:
        print(json.dumps(results, indent=1, ensure_ascii=False))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="psn2d")
    sub = p.add_subparsers(dest="cmd")
    pr = sub.add_parser("run", help="run a YAML problem")
    pr.add_argument("input")
    pr.add_argument("--case", help="run only the named case")
    pr.add_argument("--json", action="store_true", help="print results as JSON")
    pr.add_argument("--quiet", action="store_true", help="no per-iteration output")
    pr.add_argument("--opt", default="off",
                    choices=["off", "auto", "chol", "mmd", "lu"],
                    help="memory-optimized factorization backend "
                         "(off = plain path, default)")
    pr.add_argument("--threads", type=int, default=None,
                    help="total parallel-unit budget (default = half the "
                         "system core count; factor phase <= N threads, "
                         "sweep pool <= N workers x 1 thread)")
    pr.add_argument("--mem-limit-gb", type=float, default=None,
                    help="factor-pool memory cap in GB (default 32; "
                         "fail-loud if the resident (i,g) factor pool "
                         "exceeds it)")
    pr.set_defaults(func=cmd_run)
    args = p.parse_args(argv)
    if not getattr(args, "func", None):
        p.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
