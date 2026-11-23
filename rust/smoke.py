"""Acceptance test 1: ctypes smoke test of libpsn_chol.so (small SPD systems).

Run: python3 rust/smoke.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix, random as sprand
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rustchol import RustChol, load_rust_chol  # noqa: E402

FAIL = []


def check(name, ok, extra=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {extra}", flush=True)
    if not ok:
        FAIL.append(name)


def reldev(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def spd(n, density, seed, extra=1.0):
    """Symmetric diagonally dominant SPD pattern with random positive weights."""
    rng = np.random.default_rng(seed)
    A = sprand(n, n, density=density, random_state=rng, dtype=float)
    A = csc_matrix(A + A.T)
    A.data = np.abs(A.data) + 0.1
    A = A - csc_matrix((np.diag(np.asarray(A.sum(axis=1)).ravel())))
    A = (-A) + csc_matrix((np.full(n, extra), (np.arange(n), np.arange(n))),
                          shape=(n, n))
    A.sum_duplicates()
    return csc_matrix(A)


def main():
    lib = load_rust_chol()
    print("loaded libpsn_chol.so:", lib._name)

    # --- explicit 3x3 SPD system ------------------------------------------------
    A = csc_matrix(np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]))
    b = np.array([1.0, 2.0, 3.0])
    for order in (0, 1):
        f = RustChol(A, ordering=order)
        x = f.solve(b)
        ref = np.linalg.solve(A.toarray(), b)
        check(f"3x3 SPD ordering={order}", reldev(x, ref) < 1e-14,
              f"reldev={reldev(x, ref):.2e} nnz={f.nnz}")
        check(f"3x3 residual ordering={order}",
              float(np.max(np.abs(A @ x - b))) < 1e-14,
              f"res={float(np.max(np.abs(A @ x - b))):.2e}")
        f.close()

    # --- random SPD, both orderings, dense reference ---------------------------
    for n, dens, seed in ((50, 0.08, 1), (400, 0.01, 2), (2000, 0.002, 3)):
        A = spd(n, dens, seed)
        ref = splu(A.toarray() if n < 100 else A)  # dense only for the tiny case
        rng = np.random.default_rng(seed + 100)
        B = rng.normal(size=(n, 4))
        for order in (0, 1):
            f = RustChol(A, ordering=order)
            err = 0.0
            for j in range(B.shape[1]):
                x = f.solve(B[:, j])
                err = max(err, reldev(A @ x, B[:, j]))
            # fused multi-RHS must agree with the single-RHS path
            xs = f.solve_k([B[:, j] for j in range(4)])
            errk = max(reldev(A @ xs[j], B[:, j]) for j in range(4))
            x0, x1 = f.solve2(B[:, 0], B[:, 1])
            err2 = max(reldev(A @ x0, B[:, 0]), reldev(A @ x1, B[:, 1]))
            d = f.diag()
            check(f"n={n} density={dens} ordering={order}", err < 1e-11,
                  f"max|Ax-b|/|b|={err:.2e} fused_k={errk:.2e} solve2={err2:.2e} "
                  f"nnzL={f.nnz} panels={d[1]} stored={d[3]}")
            d_ref = reldev(f.solve(B[:, 0]), np.asarray(ref.solve(B[:, 0])).ravel())
            check(f"n={n} ordering={order} match SuperLU", d_ref < 1e-10,
                  f"reldev={d_ref:.2e}")
            f.close()

    # --- fail loud: indefinite matrix ------------------------------------------
    Bad = csc_matrix(np.array([[1.0, 2.0], [2.0, 1.0]]))
    try:
        RustChol(Bad, ordering=1)
        check("indefinite matrix rejected", False, "no exception raised")
    except RuntimeError as e:
        check("indefinite matrix rejected", "status=2" in str(e), str(e)[:70])

    # --- fail loud: METIS missing must not silently use natural order ----------
    print("chol_last_error():", lib.chol_last_error())
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
