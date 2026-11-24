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


def build_solver(spec, case, threads=None, mem_limit_gb=None, opt="off"):
    geom = spec["geometry"]
    # runtime inputs: CLI > YAML solver.threads / solver.mem_limit_gb > default
    # (falsy guard: 0 / None both mean "not set"; 0 is not a valid budget)
    sol = spec.get("solver", {})
    threads = threads or sol.get("threads")
    mem_limit_gb = mem_limit_gb or sol.get("mem_limit_gb") or None
    bnd = spec["boundaries"]
    S = int(case.get("subdivide", geom.get("subdivide", 1)))
    grading = model.interface_grading(case.get("interface_grading", geom.get("interface_grading", 0.0)))
    if bool(geom.get("rect", False)):
        # rectangular nodes: per-node widths.  An optional per-case
        # `subdivide` S splits every node into S x S sub-nodes (widths / S,
        # material unchanged) — same kron recipe as the square path.  S=1 is
        # the identity (arrays pass through untouched -> bit-identical).
        mat_map0 = np.array(geom["mat_grid"], dtype=int)
        hx0 = np.asarray(geom["widths_x"], float)
        hy0 = np.asarray(geom["widths_y"], float)
        # scalar widths broadcast to the grid shape
        hx0 = np.broadcast_to(hx0, mat_map0.shape)
        hy0 = np.broadcast_to(hy0, mat_map0.shape)
        if grading and S > 1:
            mat_map = mat_map0.repeat(S, axis=0).repeat(S, axis=1)
            widths = model.graded_widths(mat_map0, hx0[0], hy0[:, 0], bnd, S, grading)
        elif S != 1:
            f = np.full((S, S), 1.0 / S)
            mat_map = np.kron(mat_map0, np.ones((S, S), int))
            widths = (np.kron(hx0, f), np.kron(hy0, f))
        else:
            mat_map, widths = mat_map0, (hx0, hy0)
        h = float(widths[0].ravel()[0])     # representative (unused by rect)
    else:
        grid = np.array(geom["grid"], dtype=int)
        mat_map = np.kron(grid, np.ones((S, S), int))
        unit_size = float(geom["unit_size"])
        h = unit_size / S
        widths = None
        if grading and S > 1:
            widths = model.graded_widths(grid, np.full(grid.shape[1], unit_size),
                                        np.full(grid.shape[0], unit_size), bnd, S, grading)
    St, Sgg, nuSf, chi = model.arrays(spec)
    boundary = (bnd["left"], bnd["bottom"], bnd["right"], bnd["top"])
    generic = case["model"] == "generic"
    closure = case.get("closure", "legacy")
    strict = closure == "strict_phase_space"
    cls, options = PSN2D, {"strict": strict}
    if closure in ("sector_diffusion", "sector_average"):
        from .sector_diffusion import SectorAverage, SectorDiffusion
        averaged = closure == "sector_average"
        cls = SectorAverage if averaged else SectorDiffusion
        options = {"azimuth_degree": case.get("A", 0 if averaged else 3), "backend": opt}
        if averaged:
            options["slope_space"] = case.get("slope_space", "quadratic")
            options["polar_degree"] = case.get("polar_degree", 0)
            options["reflection_reduction"] = case.get("reflection_reduction", False)
            options["azimuth_grid"] = case.get("azimuth_grid", "uniform")
            options["node_response"] = case.get("node_response", False)
    return cls(mat_map, h, St, Sgg, nuSf, chi=chi, boundary=boundary,
               generic=generic, I=int(case.get("I", 30)), M=int(case["M"]),
               widths=widths, threads=threads, mem_limit_gb=mem_limit_gb,
               **options)


def run_case(spec, case, verbose=True, opt="off", threads=None,
             mem_limit_gb=None, extrapolate=False, anderson=0, cmfd=0,
             cmfd_merge=1,
             cmfd_omega=1.0, cmfd_groups=0, cmfd_denom="prev",
             cmfd_current=False, ritz=False):
    t0 = time.time()
    closure = case.get("closure", "legacy")
    grading = case.get("interface_grading", spec["geometry"].get("interface_grading", 0.0))
    if opt == "sector-cholmod" and closure not in ("sector_diffusion", "sector_average"):
        raise ValueError("--opt sector-cholmod requires sector_diffusion or sector_average")
    if closure in ("sector_diffusion", "sector_average"):
        if cmfd or extrapolate or anderson:
            raise ValueError("sector diffusion uses Ritz and is exclusive with "
                             "--cmfd, --extrapolate and --anderson")
        psn = build_solver(spec, case, threads, mem_limit_gb, opt=opt)
        sol = spec["solver"]
        k, _, _ = psn.keff(max_outer=int(sol["max_outer"]),
                           outer_tol=float(sol["keff_tol"]), verbose=verbose)
        degree = case.get("A", 0 if closure == "sector_average" else 3)
        out = dict(name=case["name"], model=case["model"], I=int(case.get("I", 30)),
                   polar_nodes=psn.I, phase_systems=psn.ng * len(psn.angles),
                   M=psn.M, A=degree, K=psn.angular_modes if closure == "sector_average" else degree + 1,
                   closure=closure,
                   subdivide=int(case["subdivide"]), nodes=psn.nodes,
                   keff=float(k), time_s=round(time.time() - t0, 2),
                   factor_time_s=round(psn.factor_seconds, 2),
                   factor_bytes=psn.factor_bytes, backend=opt,
                   sweeps_executed=psn.sweeps_executed, converged=True,
                   outer_iterations=psn.outer_iterations,
                   trial_sweeps=psn.extra_sweeps, accel="ritz",
                   keff_residual=psn.keff_residual,
                   source_residual=psn.source_residual)
        if grading:
            out["interface_grading"] = grading
        if closure == "sector_average":
            if psn.angular_modes > 1:
                out["slope_space"] = psn.slope_space
                out["dofs_per_system"] = int(psn.phase_size)
            if psn.polar_degree:
                out["polar_degree"] = psn.polar_degree
            if psn.azimuth_grid != "uniform":
                out["azimuth_grid"] = psn.azimuth_grid
            if psn._phase_reduction is not None:
                out["reflection_reduction"] = True
                out["dofs_per_system"] = psn._phase_reduction.size
            if psn.node_response is not None:
                out.update(node_response=True, trace_dofs_per_system=psn.node_response.trace.size,
                           interior_dofs_per_system=psn.node_response.interior.size)
        if case.get("kref") is not None:
            out.update(kref=float(case["kref"]), pcm=(k - float(case["kref"])) * 1e5)
        return out
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
    if ritz and (closure_run != "legacy" or cmfd or extrapolate or anderson):
        raise ValueError("--ritz requires the scalar sweep and is exclusive "
                         "with --cmfd, --extrapolate and --anderson")
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
        acc_obj = None
        if cmfd:
            from .cmfd import Cmfd
            acc_obj = Cmfd(psn, block=int(cmfd), merge=int(cmfd_merge),
                           omega=float(cmfd_omega), groups=int(cmfd_groups),
                           denom=str(cmfd_denom), current=bool(cmfd_current),
                           verbose=bool(verbose and -1))
            psn._cmfd = acc_obj
        sol = spec["solver"]
        k, phi, qnode = psn.keff(max_outer=int(sol["max_outer"]),
                                 outer_tol=float(sol["keff_tol"]),
                                 verbose=verbose and -1,
                                 extrapolate=bool(extrapolate),
                                 anderson=int(anderson),
                                 ritz=ritz)
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
            "outer_iterations": int(getattr(psn, "outer_iterations", 0)),
            "trial_sweeps": int(getattr(psn, "extra_sweeps", 0)),
            "accel": "+".join([x for x in
                               (f"cmfd{int(cmfd)}" if cmfd else "",
                                "dombey" if extrapolate else "",
                                "ritz" if ritz else "",
                                f"anderson{int(anderson)}" if anderson else
                                "") if x])
                     or "none",
        }
        if int(anderson):
            out["anderson"] = getattr(psn, "aa_stats", {})
        if ritz:
            out.update(converged=True, sweeps_executed=psn.sweeps_executed,
                       keff_residual=psn.keff_residual,
                       source_residual=psn.source_residual)
            if os.environ.get("OPENPSN_TRACE"):
                out["ritz_sweep_seconds"] = psn.ritz_sweep_times
        if os.environ.get("OPENPSN_TRACE"):
            # (it, dkeff, r_src, mode, keff) per outer iteration -- the tail
            # policy in solver.keff is tuned from this, not from cold runs
            out["trace"] = [list(r) for r in getattr(psn, "trace", [])]
        if acc_obj is not None:
            out["cmfd"] = {"block": int(cmfd), "merge": int(cmfd_merge),
                           "omega": float(cmfd_omega), "nG": acc_obj.nG,
                           "denom": acc_obj.denom,
                           "current": bool(acc_obj.current), "nC": acc_obj.nC,
                           "coarse_iters": acc_obj.ncoarse_used,
                           "applied": acc_obj.n_applied,
                           "skipped": acc_obj.n_skip,
                           "skip_reason": acc_obj.skip,
                           "dev_last": acc_obj.dev_last,
                           "dev_max": acc_obj.dev_max,
                           "k_coarse": acc_obj.k_coarse,
                           "clipped": acc_obj.n_clip}
        if acc_obj is not None and acc_obj.n_applied == 0:
            # a no-op accelerator is indistinguishable from a working one in
            # the keff column, so say so out loud
            print(f"WARNING: --cmfd {cmfd} applied no coarse step "
                  f"({acc_obj.n_skip} skips, last reason: {acc_obj.skip})",
                  file=sys.stderr)
        if report is not None:
            out["memopt"] = report
    kref = case.get("kref")
    if kref is not None:
        out["kref"] = float(kref)
        out["pcm"] = float((k - kref) * 1e5)   # paper convention: absolute
    if grading:
        out["interface_grading"] = grading
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
                     threads=args.threads, mem_limit_gb=args.mem_limit_gb,
                     extrapolate=args.extrapolate, anderson=args.anderson,
                     cmfd=args.cmfd,
                     cmfd_merge=args.cmfd_merge,
                     cmfd_omega=args.cmfd_omega,
                     cmfd_groups=args.cmfd_groups,
                     cmfd_denom=args.cmfd_denom,
                     cmfd_current=args.cmfd_current, ritz=args.ritz)
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
            if r.get("outer_iterations"):
                line += (f"  {r['outer_iterations']} it"
                         f"+{r.get('trial_sweeps', 0)} trial"
                         f"[{r.get('accel', 'none')}]")
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
                    choices=["off", "auto", "chol", "chol-rs", "chol-packed-rs", "sector-cholmod",
                             "mmd", "lu", "tile", "angschr"],
                    help="memory-optimized factorization backend "
                         "(off = plain path, default; auto picks "
                         "shared-chol -> angschr -> tile from "
                         "--mem-limit-gb; chol-rs = shared Rust Cholesky; "
                         "chol-packed-rs = shared packed Cholesky with "
                         "FP64 PCG refinement; sector-cholmod = optional FP64 sector factorization)")
    pr.add_argument("--threads", type=int, default=None,
                    help="total parallel-unit budget (default = half the "
                         "system core count; factor phase <= N threads, "
                         "sweep pool <= N workers x 1 thread)")
    pr.add_argument("--mem-limit-gb", type=float, default=None,
                    help="factor-pool memory cap in GB (default 32; "
                         "fail-loud if the resident (i,g) factor pool "
                         "exceeds it)")
    pr.add_argument("--cmfd", type=int, default=0, metavar="BLOCK",
                    help="EXPERIMENTAL (default 0 = off): accelerate the "
                         "outer fission-source iteration with a coarse-mesh "
                         "finite-difference rebalance (BLOCK x BLOCK fine "
                         "nodes per coarse cell).  Not shippable on "
                         "heterogeneous lattices yet -- see psn2d/cmfd.py "
                         "for the measured C5G7 numbers; use --extrapolate")
    pr.add_argument("--cmfd-merge", type=int, default=1, metavar="MERGE",
                    help="with --cmfd: merge MERGE x MERGE non-fissile base "
                         "cells into one coarse cell, keeping one coarse cell "
                         "per fissionable (pin) cell -- pin-wise in fuel, "
                         "coarse in moderator/reflector")
    pr.add_argument("--cmfd-current", action="store_true",
                    help="EXPERIMENTAL, not convergent yet.  with --cmfd: "
                         "close the coarse mesh with one-sided "
                         "(discontinuity) face couplings routed so the "
                         "collapsed fine flux satisfies the coarse balance "
                         "exactly at the current k, instead of Fickian "
                         "D*(phiA-phiB).  A positive Fickian face term cannot "
                         "reproduce the fine leakage field on a pin mesh "
                         "(>=6% worst-cell residual, measured); this does "
                         "(2e-7 relative, mode/phiC 1.0000 at the converged "
                         "solution) but the transport-split coarse operator "
                         "is not an M-matrix mid-run, so the coupled "
                         "iteration drifts (fine denominator: 1.2468 at "
                         "1085 it) or diverges (prev denominator).  Default "
                         "off; --cmfd without this flag keeps the Fickian "
                         "operator and stays bit-exact")
    pr.add_argument("--cmfd-groups", type=int, default=0, metavar="G",
                    help="with --cmfd: condense the fine energy groups into G "
                         "contiguous coarse groups for the CMFD solve (0 = "
                         "keep one coarse group per fine group).  Reaction "
                         "rates are conserved by the condensation")
    pr.add_argument("--cmfd-denom", default="prev",
                    choices=["prev", "fine"],
                    help="with --cmfd: prolongation denominator.  prev = the "
                         "previous converged coarse eigenmode (exactly "
                         "unbiased, measured inert on C5G7); fine = the coarse "
                         "average of the fine flux from the sweep just "
                         "finished, the form OpenMOC uses")
    pr.add_argument("--cmfd-omega", type=float, default=1.0, metavar="OMEGA",
                    help="with --cmfd: under-relax the coarse multiplier, "
                         "R <- 1 + OMEGA*(R-1) (default 1 = trust the coarse "
                         "solve fully)")
    pr.add_argument("--anderson", type=int, default=0, metavar="M",
                    help="accelerate the outer fission-source iteration with "
                         "Anderson acceleration of depth M on the same source "
                         "map the power iteration uses (0 = off, default). "
                         "Costs no extra sweep per attempt, unlike "
                         "--extrapolate, and preserves the fixed point "
                         "exactly, so it carries no bias of its own.  With "
                         "--extrapolate also given it relays to Dombey once "
                         "dkeff < 1e-4 (Anderson for the shape, Dombey for the "
                         "tail), which inherits Dombey's sub-0.1 pcm bias")
    pr.add_argument("--extrapolate", action="store_true",
                    help="accelerate the outer fission-source iteration "
                         "with Dombey two-point source extrapolation "
                         "(~2.5x fewer sweeps on C5G7; the accepted keff "
                         "carries a sub-0.1 pcm bias, inside the 1 pcm "
                         "regression tolerance — see PSN2D.keff)")
    pr.add_argument("--ritz", action="store_true",
                    help="solve the outer eigenproblem with diffusion-"
                         "preconditioned Ritz projection and verify it with "
                         "a fresh transport sweep")
    pr.set_defaults(func=cmd_run)
    args = p.parse_args(argv)
    if not getattr(args, "func", None):
        p.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
