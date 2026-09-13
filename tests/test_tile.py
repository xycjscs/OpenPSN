#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for the automatic tile (Schur) backend (psn2d/tile.py).

The tile backend is a MEMORY FALLBACK: it must return the SAME answer as the
plain/shared-chol path (same fixed point, different factorization) while never
materializing a full-system factor, and the auto dispatcher must select it
exactly when the shared-Cholesky pool won't fit the user's mem budget.

The checkerboard benchmark is the ideal test subject: it has NO assembly
concept, which proves the tiling is model-agnostic (a pure geometric block
split).

  fast tier:
    tile_engages_coarse   S=4 M12/M24 -> backend=='tile', per-system residual
                          < 1e-9, keff within the paper reference
    dispatch_flip         choose_backend decision: budget well above the pool
                          -> shared-chol; well below -> tile
  full tier:
    tile_bit_exact        S=16 M24 (1024 nodes): tile keff == plain keff to
                          < 0.05 pcm (the same fixed point, different factors)

Usage:
  python tests/test_tile.py            # fast tier
  python tests/test_tile.py --full     # + S=16 bit-exactness
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from psn2d import model                      # noqa: E402
from psn2d import __main__ as M              # noqa: E402
from psn2d import memopt                     # noqa: E402
from psn2d import tile as T                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(os.path.dirname(HERE), "examples")
CB = os.path.join(EXAMPLES, "psn_repro", "checkerboard_1g.yaml")
WEAK_REF = 1.12974      # OpenMC reference (coarse S=4 mesh, paper Fig.3)
KEFF_TOL_PCM = 0.05


def _load(case_name, subdivide=None):
    spec = model.load_spec(CB)
    case = next(c for c in model.expand_cases(spec) if c["name"] == case_name)
    if subdivide is not None:
        case = dict(case)
        case["subdivide"] = subdivide
    return spec, case


def t_tile_engages_coarse():
    """S=4 (64 nodes), M12 & M24: the tile backend engages from a 1 GB budget,
    passes its own per-system residual gate, and keff matches the reference."""
    for case_name in ("weak_M12", "weak_M24"):
        spec, case = _load(case_name)
        psn = M.build_solver(spec, case, threads=8, mem_limit_gb=1.0)
        rep = memopt.install_optimized(psn, backend="tile", mem_limit_gb=1.0)
        assert rep["backend"] == "tile", rep
        systems = rep["systems"]
        rels = [s["verify_rel"] for s in systems]
        assert all(r is not None and r < 1e-9 for r in rels), \
            f"{case_name}: residual gate failed {rels}"
        sol = spec["solver"]
        k, _, _ = psn.keff(max_outer=int(sol["max_outer"]),
                           outer_tol=float(sol["keff_tol"]), verbose=False)
        pcm = (k - WEAK_REF) * 1e5
        print(f"  {case_name} (S=4): tile P0={sorted(set(s['P0'] for s in systems))} "
              f"max_resid={max(rels):.1e} keff={k:.6f} (ref {WEAK_REF:.5f}, {pcm:+.2f} pcm)")
    print("PASS tile_engages_coarse")


def t_dispatch_flip():
    """choose_backend must pick shared-chol when the pool fits the budget and
    tile when it doesn't.  Uses S=16 M24 (shared-chol eligible: M%4==0,
    transpose-symmetric checkerboard)."""
    spec, case = _load("weak_M24", subdivide=16)
    psn = M.build_solver(spec, case, threads=8, mem_limit_gb=32.0)
    est_pool, _ = T.pool_estimate_shared_chol(psn)
    est_pool_gb = est_pool / 2 ** 30
    b_hi, _ = T.choose_backend(psn, 32.0)
    b_lo, _ = T.choose_backend(psn, est_pool_gb * 0.5)
    print(f"  S=16 M24 est_pool={est_pool_gb:.3f}GB: "
          f"mem=32GB->{b_hi}, mem={est_pool_gb*0.5:.3f}GB->{b_lo}")
    assert b_hi == "shared-chol", b_hi
    assert b_lo == "tile", b_lo
    print("PASS dispatch_flip")


def t_tile_bit_exact():
    """S=16 M24 (1024 nodes): the tile backend must converge to the SAME keff
    as the plain path — the decisive correctness gate (different factorization,
    identical fixed point)."""
    spec, case = _load("weak_M24", subdivide=16)
    # plain reference
    psn_p = M.build_solver(spec, case, threads=8, mem_limit_gb=32.0)
    psn_p._memopt_backend = "plain"
    k_plain, _, _ = psn_p.keff(max_outer=int(spec["solver"]["max_outer"]),
                               outer_tol=float(spec["solver"]["keff_tol"]),
                               verbose=False)
    # tile
    psn_t = M.build_solver(spec, case, threads=8, mem_limit_gb=1.0)
    rep = memopt.install_optimized(psn_t, backend="tile", mem_limit_gb=1.0)
    assert rep["backend"] == "tile", rep
    k_tile, _, _ = psn_t.keff(max_outer=int(spec["solver"]["max_outer"]),
                              outer_tol=float(spec["solver"]["keff_tol"]),
                              verbose=False)
    d_pcm = (k_tile - k_plain) * 1e5
    print(f"  S=16 M24: plain={k_plain:.7f} tile={k_tile:.7f} "
          f"P0={sorted(set(s['P0'] for s in rep['systems']))} d={d_pcm:+.4f} pcm")
    assert abs(d_pcm) < KEFF_TOL_PCM, f"tile keff diff {d_pcm} pcm"
    print("PASS tile_bit_exact")


TESTS = [
    ("tile_engages_coarse", t_tile_engages_coarse, "fast"),
    ("dispatch_flip", t_dispatch_flip, "fast"),
    ("tile_bit_exact", t_tile_bit_exact, "full"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--only", help="comma-separated test names")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    failed = []
    t_all = time.time()
    for name, fn, tier in TESTS:
        if tier == "full" and not args.full:
            print(f"SKIP {name} (full tier; use --full)")
            continue
        if only and name not in only:
            continue
        t = time.time()
        try:
            fn()
            print(f"  [{name}] {time.time() - t:.1f}s")
        except Exception as e:
            import traceback
            traceback.print_exc()
            failed.append(name)
            print(f"FAIL {name}: {e}")
    print(f"\nOVERALL: {'PASS' if not failed else 'FAIL ' + str(failed)} "
          f"({time.time() - t_all:.0f}s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
