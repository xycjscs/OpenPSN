#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the optional memory-optimized factorization backends (memopt).

Safety contract:
  * the default solver path is untouched (tests run with --opt off implicitly
    by calling keff directly on an uninstalled solver);
  * compact assembly reproduces build_system ELEMENT-EXACTLY (FP64, no
    tolerance — the numbering and accumulation order must match);
  * every backend (lu / mmd / shared-chol) converges to the same keff as
    the plain path to source-iteration roundoff;
  * nonconforming geometries fail LOUD and ``auto`` falls back, never
    silently degrading.

Usage:
  python tests/test_memopt.py            # fast tier (seconds to ~1 min)
  python tests/test_memopt.py --full     # + quarter-core M12_S2 backends
  python tests/test_memopt.py --only compact_exact

Exit code 0 = all pass, 1 = at least one failure.
"""
import argparse
import gc
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402

from psn2d import model  # noqa: E402
from psn2d import __main__ as M  # noqa: E402
from psn2d import memopt  # noqa: E402
from psn2d.solver import PSN2D  # noqa: E402

EXAMPLES = os.path.join(os.path.dirname(HERE), "examples")
KEFF_TOL = 1e-9  # source-iteration roundoff between backends


def _spec(name):
    return model.load_spec(os.path.join(EXAMPLES, name))


def _case(spec, name):
    return next(c for c in model.expand_cases(spec) if c["name"] == name)


def _keff(spec, case_name, opt="off", quiet=True):
    """Run one case end to end; return (keff, report|None, wall_s, peak_GB)."""
    import resource
    t0 = time.time()
    psn = M.build_solver(_spec(spec), _case(_spec(spec), case_name))
    report = None
    if opt != "off":
        report = memopt.install_optimized(psn, backend=opt)
    sol = _spec(spec)["solver"]
    k, _, _ = psn.keff(max_outer=int(sol["max_outer"]),
                       outer_tol=float(sol["keff_tol"]),
                       verbose=0 if quiet else -1)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    dt = time.time() - t0
    return float(k), report, dt, peak


def _sys_exact(psn_a, psn_b, i, g):
    """Element-exact comparison of two (i, g) systems (plain vs compact)."""
    A1, rows1 = psn_a.build_system(i, g)
    A2, bt2 = psn_b._compact_build(i, g)
    A1, A2 = A1.tocsc().astype(np.float64), A2.tocsc().astype(np.float64)
    A1.sort_indices()
    A2.sort_indices()
    if A1.nnz != A2.nnz:
        raise AssertionError(f"system ({i},{g}): nnz {A1.nnz} != {A2.nnz}")
    d_ptr = int(np.max(np.abs(A1.indptr.astype(np.int64) - A2.indptr.astype(np.int64)))) \
        if A1.shape[0] else 0
    d_idx = int(np.max(np.abs(A1.indices.astype(np.int64) - A2.indices.astype(np.int64)))) \
        if A1.nnz else 0
    d_val = float(np.max(np.abs(A1.data - A2.data))) if A1.nnz else 0.0
    if d_ptr or d_idx or d_val:
        raise AssertionError(f"system ({i},{g}): indptr max diff {d_ptr}, "
                             f"indices max diff {d_idx}, data max diff {d_val}")
    # b-table: plain solver's own _bterms_for vs the compact one (int64 vs int32)
    psn_a._vectorize_setup()
    b1 = psn_a._bterms_for(i, g)
    if set(b1) != set(bt2):
        raise AssertionError(f"system ({i},{g}): b-term keys differ")
    for k in b1:
        for a, b2 in zip(b1[k], bt2[k]):
            if not np.array_equal(a.astype(np.int64), b2.astype(np.int64)):
                raise AssertionError(f"system ({i},{g}): b-term table {k} differs")
    return A1.nnz


# --------------------------------------------------------------------------- #
# fast tests
# --------------------------------------------------------------------------- #

def t_compact_exact():
    """compact assembly == build_system, element-exact, every (i, g) system."""
    total = 0
    for spec_name, case_name in [("c5g7_uo2_assembly.yaml", "M8_S2"),
                                 ("checkerboard_1g.yaml", "weak_M12"),
                                 ("c5g7_rect_uo2_assembly.yaml", "M8")]:
        spec = _spec(spec_name)
        case = _case(spec, case_name)
        a = M.build_solver(spec, case)
        b = M.build_solver(spec, case)
        memopt.install_compact_build(b)
        nsys = 0
        for i in range(a.I):
            for g in range(a.ng):
                total += _sys_exact(a, b, i, g)
                nsys += 1
        print(f"  {spec_name}::{case_name}: {nsys} systems element-exact "
              f"(rect={a.rect}, generic={a.generic}, M={a.M})")
    print(f"PASS compact_exact ({total} matrix elements across all systems, "
          f"max element diff 0.0)")


def t_keff_equiv_lu_mmd():
    """lu / mmd backends converge to the plain keff (small problems)."""
    for spec_name, case_name in [("checkerboard_1g.yaml", "weak_M12"),
                                 ("c5g7_uo2_assembly.yaml", "M8_S2"),
                                 ("c5g7_rect_uo2_assembly.yaml", "M8")]:
        k0, _, t0, _ = _keff(spec_name, case_name, "off")
        row = f"  {spec_name}::{case_name}: plain k={k0:.9f} ({t0:.1f}s)"
        for opt in ("lu", "mmd"):
            k, rep, t, _ = _keff(spec_name, case_name, opt)
            d = abs(k - k0)
            assert d < KEFF_TOL, f"{opt} keff diff {d}"
            row += f" | {opt} k={k:.9f} d={d:.1e} ({t:.1f}s)"
        print(row)
    print("PASS keff_equiv_lu_mmd")


def t_shared_chol_conforming():
    """shared-Cholesky on a conforming geometry (UO2 assembly, all-reflect,
    mat == mat.T, M % 4 == 0): keff must match plain to iteration roundoff,
    with the per-system verification records in place."""
    if memopt._find_eigen_chol() is None:
        print("SKIP shared_chol_conforming (libeigen_chol.so not built; "
              "run psn2d/memopt_cxx/build.py)")
        return
    spec_name, case_name = "c5g7_uo2_assembly.yaml", "M8_S2"
    k0, _, t0, p0 = _keff(spec_name, case_name, "off")
    k, rep, t, pk = _keff(spec_name, case_name, "chol")
    d = abs(k - k0)
    assert rep["backend"] == "shared-chol", rep
    assert d < KEFF_TOL, f"shared-chol keff diff {d}"
    # per-system verification records: symmetry + random-RHS residual gates
    psn = M.build_solver(_spec(spec_name), _case(_spec(spec_name), case_name))
    memopt.install_shared_chol(psn)
    for i in range(psn.I):
        for g in range(psn.ng):
            psn._lu(i, g)
    recs = psn._chol_records
    assert len(recs) == psn.I * psn.ng
    asym = max(r["scaled_asymmetry"] for r in recs)
    share = max(max(r["sharing_errors"]) for r in recs)
    assert asym <= 1e-12 and share <= 1e-12, (asym, share)
    print(f"  {spec_name}::{case_name}: plain k={k0:.9f} peak={p0:.2f}GB "
          f"({t0:.0f}s) | shared-chol k={k:.9f} d={d:.1e} peak={pk:.2f}GB "
          f"({t:.0f}s), asym<={asym:.1e} share<={share:.1e}, "
          f"records={len(recs)}")
    print("PASS shared_chol_conforming")


def _asym_psn():
    """3x3 grid, material pattern NOT transpose-symmetric, mixed boundaries."""
    mat = np.array([[0, 1, 0],
                    [0, 0, 1],
                    [0, 1, 0]], dtype=int)
    St = np.array([[1.0, 0.5]])           # (ng, nmat)
    Sgg = np.array([[[0.5, 0.25]]])       # (ng, ng, nmat)
    nuSf = np.array([[1.0, 1.0]])         # (ng, nmat)
    return PSN2D(mat, 1.0, St, Sgg, nuSf, chi=np.ones(1),
                 boundary=("reflect", "reflect", "vacuum", "vacuum"),
                 generic=False, M=8)


def t_fail_loud_and_fallback():
    """Nonconforming geometry: chol gate raises; auto falls back to
    compact-MMD; keff still matches the plain path exactly."""
    plain = _asym_psn()
    k0, _, t0, _ = _run_plain(plain)
    rep = memopt.install_optimized(plain, backend="auto")
    assert rep["backend"] != "shared-chol", rep
    assert "shared-chol" in rep["reason"], rep
    k, _, t, _ = _run_plain(plain)   # solver now on the compact-MMD backend
    d = abs(k - k0)
    assert d < KEFF_TOL, f"fallback keff diff {d}"
    print(f"  asymmetric 3x3: auto -> {rep['backend']} "
          f"(reason: {rep['reason'][:60]}...), d={d:.1e}, ({t:.1f}s)")
    # direct chol request must raise, not degrade
    p2 = _asym_psn()
    try:
        memopt.install_shared_chol(p2)
        raise AssertionError("shared-chol accepted a nonconforming geometry")
    except ValueError as e:
        print(f"  install_shared_chol fail-loud: ValueError({e})")
    print("PASS fail_loud_and_fallback")


def _run_plain(psn):
    import resource
    t0 = time.time()
    k, _, _ = psn.keff(max_outer=6000, outer_tol=1e-10, verbose=False)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    return float(k), None, time.time() - t0, peak


# --------------------------------------------------------------------------- #
# full tests (quarter core — exercises the fork pool COW path, ncol > 150k)
# --------------------------------------------------------------------------- #

def t_full_core_backends():
    spec_name, case_name = "c5g7_2d_quarter_core.yaml", "M12_S2"
    k0, rep0, t0, p0 = _keff(spec_name, case_name, "off")
    print(f"  {spec_name}::{case_name} plain: k={k0:.9f} peak={p0:.2f}GB ({t0:.0f}s)")
    row = ""
    for opt in ("mmd", "chol"):
        k, rep, t, pk = _keff(spec_name, case_name, opt)
        d = abs(k - k0)
        assert d < KEFF_TOL, f"{opt} keff diff {d}"
        row += f"\n    {opt}: k={k:.9f} d={d:.1e} peak={pk:.2f}GB ({t:.0f}s) " \
               f"[{rep['backend']}]"
        gc.collect()
    print(f"PASS full_core_backends{row}")


# --------------------------------------------------------------------------- #

TESTS = [
    ("compact_exact", t_compact_exact, "fast"),
    ("keff_equiv_lu_mmd", t_keff_equiv_lu_mmd, "fast"),
    ("shared_chol_conforming", t_shared_chol_conforming, "fast"),
    ("fail_loud_and_fallback", t_fail_loud_and_fallback, "fast"),
    ("full_core_backends", t_full_core_backends, "full"),
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
    print(f"\n{'ALL PASS' if not failed else f'FAILURES: {failed}'} "
          f"in {time.time() - t_all:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
