# -*- coding: utf-8 -*-
"""PSN2D — diffusion-based phase-space nodal method, 2D multi-group.

Chao, Li & Chen, ANE 240 (2027) 112707 — independent Python implementation.
Run a YAML problem:

    python -m psn2d run examples/psn_repro/bwr_bundle_2g.yaml
    python -m psn2d run examples/c5g7/study1_homogenised/c5g7_2d_quarter_core.yaml --case M16
    python -m psn2d run examples/psn_repro/checkerboard_1g.yaml --quick
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
    # (falsy guard: 0 / None both mean "not set"; 0 is not a valid budget)
    sol = spec.get("solver", {})
    threads = threads or sol.get("threads")
    mem_limit_gb = mem_limit_gb or sol.get("mem_limit_gb") or None
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
    strict = case.get("closure", "legacy") == "strict_phase_space"
    if generic:
        return PSN2D(mat_map, h, St, Sgg, nuSf, chi=chi,
                     boundary=boundary, generic=True, I=int(case["I"]),
                     M=int(case["M"]), widths=widths, threads=threads,
                     mem_limit_gb=mem_limit_gb, strict=strict)
    return PSN2D(mat_map, h, St, Sgg, nuSf, chi=chi,
                 boundary=boundary, generic=False, M=int(case["M"]),
                 widths=widths, threads=threads,
                 mem_limit_gb=mem_limit_gb, strict=strict)


def run_case(spec, case, verbose=True, opt="off", threads=None,
             mem_limit_gb=None):
    t0 = time.time()
    closure = case.get("closure", "legacy")
    _ALLOWED = ("legacy", "directional_matrix", "strict_phase_space")
    if closure not in _ALLOWED:
        raise ValueError(
            f"case closure {closure!r} not in {_ALLOWED}; short forms "
            "like 'strict' silently run the wrong path and are refused")
    A = int(case.get("A", 0))
    if closure == "strict_phase_space":
        # strict PSN runs on the plain power-iteration path; the node
        # kernel is swapped inside PSN2D (strict=True), faces unchanged.
        closure_run = "legacy"
    else:
        closure_run = closure
    if closure_run != "legacy":
        bnd = spec["boundaries"]
        if any(v not in ("reflect", "vacuum") for v in bnd.values()):
            raise ValueError(
                "high-order closure is validated for reflect and vacuum "
                "boundaries only; use closure=legacy for other types")
        if opt != "off":
            raise ValueError(
                "high-order closure requires --opt off (plain sparse LU); "
                "the scalar Cholesky/Schur/tile backends are not "
                "applicable for A > 0")
    psn = build_solver(spec, case, threads=threads,
                       mem_limit_gb=mem_limit_gb)
    if closure_run != "legacy":
        from . import highorder
        branch = ("directional" if closure == "directional_matrix"
                  else "original")
        sol = spec["solver"]
        inner = str(sol.get("inner", "lu"))
        if inner == "gmres":
            from .highsweep import keff_highorder_gmres
            _gp = sol.get("gmres_pairs", None)
            if _gp is not None:
                _gp = str(_gp) == "on"
            k, phi, qim, hinfo = keff_highorder_gmres(
                psn, A=A, branch=branch,
                max_outer=int(sol["max_outer"]),
                outer_tol=float(sol["keff_tol"]),
                rtol=float(sol.get("gmres_rtol", 1e-11)),
                maxiter=int(sol.get("gmres_maxiter", 800)),
                nworkers=sol.get("gmres_workers"), verbose=verbose and -1,
                pairs=_gp)
        elif inner == "lu":
            k, phi, qim, hinfo = highorder.keff_highorder(
                psn, A=A, branch=branch,
                max_outer=int(sol["max_outer"]),
                outer_tol=float(sol["keff_tol"]), verbose=verbose and -1)
        else:
            raise ValueError(f"solver.inner must be lu or gmres "
                             f"(got {inner!r})")
        dt = time.time() - t0
        out = {
            "name": case["name"],
            "model": case["model"],
            "I": int(case["I"]),
            "M": int(case["M"]),
            "A": A,
            "K": A + 1,
            "closure": closure,
            "subdivide": int(case["subdivide"]),
            "keff": float(k),
            "time_s": round(dt, 2),
            "nodes": psn.nodes,
            "converged": hinfo["converged"],
            "outer_iterations": hinfo["outer_iterations"],
            "keff_residual": hinfo["keff_residual"],
            "source_residual": hinfo["source_residual"],
            "max_local_kernel_residual": hinfo.get(
                "max_local_kernel_residual"),
            "max_face_system_residual": hinfo.get(
                "max_face_system_residual"),
            "face_unknowns_per_system": psn.ncol * (A + 1),
        }
        if inner == "gmres":
            out["inner"] = "gmres"
            out["gmres_matvecs_total"] = hinfo["gmres_matvecs_total"]
            out["gmres_nworkers"] = hinfo["nworkers"]
        if not hinfo["converged"]:
            out["warning"] = hinfo["warning"]
    else:
        if opt != "off":
            from . import memopt
            report = memopt.install_optimized(psn, backend=opt,
                                              mem_limit_gb=mem_limit_gb)
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
            "closure": ("strict_phase_space"
                        if closure == "strict_phase_space" else "legacy"),
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
            closure = r.get("closure", "legacy")
            extra = (f" A={r['A']} K={r['K']} [{closure}]"
                     if closure != "legacy" else "")
            line = (f"[{r['name']}] {r['model']} I={r['I']} M={r['M']} "
                    f"S={r['subdivide']}{extra}: "
                    f"keff={r['keff']:.6f} ({r['time_s']} s)")
            if "pcm" in r:
                line += f"  ref={r['kref']:.5f}  Δ={r['pcm']:+.1f} pcm"
            if "memopt" in r:
                line += f"  [{r['memopt']['backend']}]"
            if not r.get("converged", True):
                line += "  NOT_CONVERGED"
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
                    choices=["off", "auto", "chol", "mmd", "lu", "tile",
                             "angschr"],
                    help="memory-optimized factorization backend "
                         "(off = plain path, default; auto picks "
                         "shared-chol -> angschr -> tile from "
                         "--mem-limit-gb)")
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
