# -*- coding: utf-8 -*-
"""Numba kernel for the red-black matvec b-assembly (lever 3,
2026-09-14).

``_mc_b`` holds the GIL for ~87% of a matvec (fancy-index gather +
small einsums, measured 2026-09-14: 30.6 ms of a 36 ms pass).  This
module pre-packs a microclass's neighbour parts into flat contiguous
arrays and runs TWO prange loops (source term, then neighbour
contributions); the parallel regions release the GIL, so a worker can
run T numba threads on the b-assembly while its BLAS lu_solve stays
pinned to 1 thread.

Gather order matches _microclass exactly: gath[:, (f*nmodes + m)*K + s]
with f = local face, m = mode (nmodes = M in full mode, 2 in pair mode).
row1d ranges of distinct parts (distinct lf) are disjoint, so the
``+=`` in the neighbour loop has no cross-task write conflict.

Bit-exactness: summation order differs from the NumPy path, so
per-matvec agreement is ~1e-15, not bit-for-bit; end-to-end keff is
verified at the 1e-12 gate (same gate the pair split uses).
"""
import os
import numpy as np

try:
    from numba import njit, prange
    import numba as _numba
    HAVE_NUMBA = True
except Exception:                                  # pragma: no cover
    HAVE_NUMBA = False

# Runtime switch (lever-3 A/B): OPSN_NO_NUMBA=1 forces the NumPy path
# even when numba is installed.  Lets the benchmark isolate the angle
# (lever 1) and kernel (lever 3) contributions independently.
ENABLED = os.environ.get("OPSN_NO_NUMBA", "") != "1"


if HAVE_NUMBA:

    @njit(parallel=True, cache=True)
    def _mc_b_kernel(qn, nlist, SRC, b, u,
                     task_part, task_nloc, task_node,
                     gath_all, sgn_all,
                     C0_all, src5_all, row1d_all, sigma_all,
                     nmodes, K):
        Nn = nlist.shape[0]
        d = b.shape[1]
        for i in prange(Nn):
            ni = nlist[i]
            for c in range(d):
                acc = 0.0
                for q in range(5):
                    acc += qn[ni, q] * SRC[q, c]
                b[i, c] = acc
        ntask = task_part.shape[0]
        for t in prange(ntask):
            pi = task_part[t]
            i = task_nloc[t]
            qrow = task_node[t]
            sig = sigma_all[pi]
            for m in range(nmodes):
                rbase = m * K
                for a in range(K):
                    acc = 0.0
                    for f in range(4):
                        gb = (f * nmodes + m) * K
                        for s in range(K):
                            acc += (C0_all[pi, m, a, f, s]
                                    * sgn_all[t, m, f, s]
                                    * u[gath_all[t, gb + s]])
                    for q in range(5):
                        acc += sig * qn[qrow, q] * src5_all[pi, m, a, q]
                    b[i, row1d_all[pi, rbase + a]] += acc


def set_num_threads(n):
    """Runtime numba thread count (call in worker init, post-fork)."""
    if HAVE_NUMBA:
        try:
            _numba.set_num_threads(int(n))
        except Exception:
            import os
            os.environ["NUMBA_NUM_THREADS"] = str(int(n))


def pack_microclass(data):
    """Flat contiguous pack of one (pair) microclass's nb parts."""
    parts = []
    for lf in sorted(data["nb"]):
        for part in data["nb"][lf]:
            parts.append((lf, part))
    if not parts:
        return None
    nmodes = int(parts[0][1]["C0"].shape[0])
    K = int(parts[0][1]["C0"].shape[1])
    Nn = len(data["nodes"])
    d = data["A"].shape[0]
    ntask = sum(len(pt["idxs"]) for _, pt in parts)
    P = len(parts)
    pack = dict(
        nlist=np.ascontiguousarray(data["nodes"], dtype=np.int64),
        SRC=np.ascontiguousarray(data["SRC"].T),
        shape=(Nn, d),
        nmodes=nmodes, K=K,
        task_part=np.empty(ntask, np.int64),
        task_nloc=np.empty(ntask, np.int64),
        task_node=np.empty(ntask, np.int64),
        gath_all=np.ascontiguousarray(
            np.vstack([pt["gath"] for _, pt in parts])),
        sgn_all=np.ascontiguousarray(
            np.vstack([pt["sgn_nb"] for _, pt in parts])),
        C0_all=np.ascontiguousarray(
            np.stack([pt["C0"] for _, pt in parts])),
        src5_all=np.ascontiguousarray(
            np.stack([pt["src5"] for _, pt in parts])),
        row1d_all=np.ascontiguousarray(
            np.vstack([pt["rowidx"].ravel() for _, pt in parts])
            .astype(np.int64)),
        sigma_all=np.array([pt["sigma"] for _, pt in parts]),
    )
    t0 = 0
    for pi, (lf, pt) in enumerate(parts):
        Npi = len(pt["idxs"])
        pack["task_part"][t0:t0 + Npi] = pi
        pack["task_nloc"][t0:t0 + Npi] = pt["idxs"]
        pack["task_node"][t0:t0 + Npi] = pt["nodes_nb"]
        t0 += Npi
    return pack


def mc_b(pack, u, qn):
    """b = SRC q + neighbour contributions, numba prange."""
    b = np.empty(pack["shape"])
    _mc_b_kernel(qn, pack["nlist"], pack["SRC"], b, u,
                 pack["task_part"], pack["task_nloc"], pack["task_node"],
                 pack["gath_all"], pack["sgn_all"],
                 pack["C0_all"], pack["src5_all"], pack["row1d_all"],
                 pack["sigma_all"], pack["nmodes"], pack["K"])
    return b
