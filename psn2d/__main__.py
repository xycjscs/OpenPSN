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


def build_solver(spec, case):
    grid = np.array(spec["geometry"]["grid"], dtype=int)
    S = int(case["subdivide"])
    mat_map = np.kron(grid, np.ones((S, S), int))
    h = float(spec["geometry"]["unit_size"]) / S
    St, Sgg, nuSf, chi = model.arrays(spec)
    bnd = spec["boundaries"]
    boundary = (bnd["left"], bnd["bottom"], bnd["right"], bnd["top"])
    generic = case["model"] == "generic"
    if generic:
        return PSN2D(mat_map, h, St, Sgg, nuSf, chi=chi,
                     boundary=boundary, generic=True, I=int(case["I"]),
                     M=int(case["M"]))
    return PSN2D(mat_map, h, St, Sgg, nuSf, chi=chi,
                 boundary=boundary, generic=False, M=int(case["M"]))


def run_case(spec, case, verbose=True):
    t0 = time.time()
    psn = build_solver(spec, case)
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
        r = run_case(spec, case, verbose=not args.quiet)
        results.append(r)
        if args.quiet:
            line = f"[{r['name']}] {r['model']} I={r['I']} M={r['M']} " \
                   f"S={r['subdivide']}: keff={r['keff']:.6f} ({r['time_s']} s)"
            if "pcm" in r:
                line += f"  ref={r['kref']:.5f}  Δ={r['pcm']:+.1f} pcm"
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
    pr.set_defaults(func=cmd_run)
    args = p.parse_args(argv)
    if not getattr(args, "func", None):
        p.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
