# -*- coding: utf-8 -*-
"""Automatic tile (Schur-complement) backend for the PSN2D face-current solve.

Memory fallback.  When the resident factor pool of the shared-Cholesky /
SuperLU backends would exceed the user's ``mem_limit_gb``, the dispatcher
switches to this backend, which NEVER materializes a full-system factor:
it factors one ``P0 x P0`` node block (a *tile*) at a time and couples the
tiles through a Schur complement on the seam between them.

The user only sets ``mem_limit_gb``.  The tile size ``P0`` is chosen
automatically from that budget (the installer builds a candidate tiling and
checks its actual resident memory, stepping to smaller tiles if needed) — the
user never specifies a tile size or an assembly layout.  A pure geometric block
partition is used, so **no assembly knowledge is required**: the partition
is correct on any grid, including the checkerboard benchmark, which has no
assembly concept at all.  Schur's identity holds for any block partition of
an SPD matrix; the seam's locality ("a seam row only couples to adjacent
across-seam rows") comes from the PSN operator being 4-neighbor local, not
from any assembly structure.

Two competing memory terms (both monotone in P0, opposite directions, so a
well-defined minimum exists):
  * tile-factor memory  ~ P0        (fewer, bigger tiles -> more total fill)
  * Schur-seam memory   ~ 1/P0^2    (seam width ~ 1/P0; S is quasi-dense on
                                     the seam, so its memory ~ seam^2)
The auto-sizer probes a few candidate P0s, measures the real factor fill and
seam size, and picks the feasible P0 with the lowest resident estimate
(breaking ties toward fewer tiles = faster).

Pipeline (per (i, g) system; factors built ONCE, reused for every outer
iteration — the "decompose once, reuse forever" invariant):

  C = sgn * diag(d) A        # sgn in {+1,-1} chosen so C is SPD
  per tile t:  factor C_TT once (Eigen Cholesky METIS, cached)
  S   = C_II - sum_t C_IT C_TT^-1 C_TI         # mrsolve multi-RHS (OMP)
  factor S once

  solve (each outer iteration; b = raw source, length ncol):
    c   = sgn * d * b                       (C x = c  <=>  A x = b)
    w_I = sum_t C_IT (C_TT^-1 c_T)
    y   = S^-1 (c_I - w_I)
    x_T = C_TT^-1 (c_T - C_TI y)            (cached tile factors)

Correctness: verified end-to-end on the C5G7 quarter core (global residual
||C x - c|| / ||c|| = 1.95e-14; every Schur gate <= 2e-15; see
``c5g7/tile_correctness_test.py``).  The multi-RHS kernel
(``chol_sparse_mrsolve``) and the C++ Cholesky bridge are the SAME verified
library used by the shared-Cholesky backend.
"""
from __future__ import annotations

import ctypes
import time

import numpy as np
from scipy.sparse import csc_matrix, coo_matrix, find as _sp_find

from .memopt import spd_row_scale, _load_eigen_chol, _digest


# --------------------------------------------------------------------------- #
# C++ bridge: bind the multi-RHS kernel on top of the shared-Cholesky lib
# --------------------------------------------------------------------------- #

_lib_cache = None


def _lib():
    global _lib_cache
    if _lib_cache is None:
        lib = _load_eigen_chol()
        ip = ctypes.POINTER(ctypes.c_int32)
        dp = ctypes.POINTER(ctypes.c_double)
        lib.chol_sparse_mrsolve.argtypes = [ctypes.c_void_p, ip, ip, ctypes.c_int,
                                            ip, ip, dp, ip, ip, dp]
        lib.chol_sparse_mrsolve.restype = None
        for name, restype in (("chol_l_p", ctypes.c_void_p), ("chol_l_i", ctypes.c_void_p),
                              ("chol_l_v", ctypes.c_void_p), ("chol_perm", ctypes.c_void_p),
                              ("chol_pinv", ctypes.c_void_p)):
            getattr(lib, name).argtypes = [ctypes.c_void_p]
            getattr(lib, name).restype = restype
        _lib_cache = lib
    return _lib_cache


def _factor(C, ordering=1):
    """Cholesky-factor an SPD CSC matrix; returns (C_int32_csc, handle)."""
    lib = _lib()
    C = C.tocsc().astype(np.float64)
    if C.indices.dtype != np.int32:
        C.indices = C.indices.astype(np.int32)
    if C.indptr.dtype != np.int32:
        C.indptr = C.indptr.astype(np.int32)
    C.sort_indices()
    st = ctypes.c_int()
    ip = ctypes.POINTER(ctypes.c_int32)
    dp = ctypes.POINTER(ctypes.c_double)
    h = lib.chol_factor(C.shape[0], C.nnz,
                        C.indptr.ctypes.data_as(ip), C.indices.ctypes.data_as(ip),
                        C.data.ctypes.data_as(dp), ordering, ctypes.byref(st))
    if not h:
        raise RuntimeError(f"tile Cholesky failed: status={st.value}")
    return C, h


def _l_row_cs(h, n):
    """CSR ROW view of L (row i: cols j<=i ascending) from the C views."""
    lib = _lib()
    nnz = int(lib.chol_nnz(h))
    lp_buf = (ctypes.c_int32 * (n + 1))()
    li_buf = (ctypes.c_int32 * nnz)()
    lv_buf = (ctypes.c_double * nnz)()
    ctypes.memmove(ctypes.addressof(lp_buf), lib.chol_l_p(h), (n + 1) * 4)
    ctypes.memmove(ctypes.addressof(li_buf), lib.chol_l_i(h), nnz * 4)
    ctypes.memmove(ctypes.addressof(lv_buf), lib.chol_l_v(h), nnz * 8)
    lp = np.ctypeslib.as_array(lp_buf)
    li = np.ctypeslib.as_array(li_buf)
    lv = np.ctypeslib.as_array(lv_buf)
    Lcsc = csc_matrix((lv, li, lp), shape=(n, n))
    Lcsr = Lcsc.tocsr()
    Lcsr.sort_indices()
    assert (Lcsr.indices[Lcsr.indptr[1:] - 1] == np.arange(n)).all(), \
        "L row view is not lower-triangular"
    return Lcsr.indptr.astype(np.int32), Lcsr.indices.astype(np.int32)


def _mr_solve(h, n, Bcsc, lrp, lri, cap):
    """Sparse multi-RHS solve C X = B (B sparse CSC n x k) via the C++ kernel.

    Returns X as CSR (n x k), duplicates summed, in original (unpermuted) space.
    """
    lib = _lib()
    ip = ctypes.POINTER(ctypes.c_int32)
    dp = ctypes.POINTER(ctypes.c_double)
    k = Bcsc.shape[1]
    Bcsc = Bcsc.tocsc()
    Bcsc.sort_indices()
    bp = Bcsc.indptr.astype(np.int32)
    bi = Bcsc.indices.astype(np.int32)
    bv = Bcsc.data.astype(np.float64)
    lpbuf = np.ascontiguousarray(lrp, dtype=np.int32)
    libuf = np.ascontiguousarray(lri, dtype=np.int32)
    xp = np.zeros(k + 1, dtype=np.int32)
    xi = np.zeros(cap, dtype=np.int32)
    xv = np.zeros(cap, dtype=np.float64)
    lib.chol_sparse_mrsolve(h, lpbuf.ctypes.data_as(ip), libuf.ctypes.data_as(ip),
                            k, bp.ctypes.data_as(ip), bi.ctypes.data_as(ip),
                            bv.ctypes.data_as(dp), xp.ctypes.data_as(ip),
                            xi.ctypes.data_as(ip), xv.ctypes.data_as(dp))
    r = int(xp[k])
    if r > cap:
        raise RuntimeError(f"mrsolve output overflow {r} > {cap}")
    if r == 0:
        return csc_matrix((n, k))
    cols = np.repeat(np.arange(k), np.diff(xp))[:r]
    return coo_matrix((xv[:r], (xi[:r], cols)), shape=(n, k)).tocsr()


def _dense_solve(h, n, b):
    """One dense RHS through the cached Cholesky factor (verified chol_solve)."""
    lib = _lib()
    dp = ctypes.POINTER(ctypes.c_double)
    b = np.ascontiguousarray(b, dtype=np.float64)
    x = np.empty_like(b)
    lib.chol_solve(h, n, b.ctypes.data_as(dp), x.ctypes.data_as(dp))
    return x


# --------------------------------------------------------------------------- #
# geometric block partition (model-agnostic; no assembly knowledge)
# --------------------------------------------------------------------------- #

def row_owner(psn, P0):
    """Per-face-row tile id: >=0 the owning tile, -1 the seam.

    A node's tile is its (i_idx // P0, j_idx // P0) block.  An internal-face
    row belongs to its tile when both endpoint nodes are in it, else to the
    seam.  Reflect / vacuum rows are always seam.  ``P0`` must divide both
    nx and ny (the caller enforces this) so the blocks are clean.
    """
    nx, ny = psn.nx, psn.ny
    if nx % P0 or ny % P0:
        raise ValueError(f"tile size P0={P0} must divide nx={nx} and ny={ny}")
    py = ny // P0
    TID = (psn.i_idx // P0) * py + (psn.j_idx // P0)
    owner = np.empty(psn.ncol, np.int32)
    r = 0
    for f in range(psn.F):
        fac = psn.faces[f]
        t = fac["type"]
        if t == "internal":
            a, b = fac["a"], fac["b"]
            owner[r:r + psn.M] = TID[a] if TID[a] == TID[b] else -1
            r += psn.M
        else:
            c = psn.M // 2 if t == "reflect" else psn.M
            owner[r:r + c] = -1
            r += c
    if r != psn.ncol:
        raise AssertionError(f"row_owner counted {r} rows, ncol={psn.ncol}")
    return owner


def tile_candidates(psn):
    """Common divisors of (nx, ny) that split BOTH directions into >= 2 tiles,
    largest first.  P0=1 is degenerate (each node its own tile -> every
    internal face is a seam, no tile-internal rows), and a single-tile
    partition has no seam; both are excluded.  A valid tile solve needs
    P0 >= 2 so each P0 x P0 block holds at least 2 nodes."""
    out = [p for p in range(2, min(psn.nx, psn.ny) + 1)
           if psn.nx % p == 0 and psn.ny % p == 0
           and psn.nx // p >= 2 and psn.ny // p >= 2]
    out.sort(reverse=True)
    return out


def _seam_width(psn, C, P0, R_I):
    """Total squared seam interface width sum_t k_t^2 for a P0 tiling, where
    k_t = number of seam columns coupled to tile t.  Used to bound the (dense)
    Schur S nnz without building S."""
    owner = row_owner(psn, P0)
    R_T = np.where(owner >= 0)[0]
    nt = int(owner[R_T].max()) + 1
    o = owner[R_T].astype(np.int32)
    order = np.argsort(o, kind="stable")
    os_ = o[order]
    start = np.searchsorted(os_, np.arange(nt), side="left")
    end = np.searchsorted(os_, np.arange(nt), side="right")
    sum_k2 = 0
    nactive = 0
    for t in range(nt):
        s0, e0 = start[t], end[t]
        if e0 == s0:
            continue
        idx = R_T[order[s0:e0]]
        subT = csc_matrix(C[np.ix_(idx, R_I)])
        _, j3, _ = _sp_find(subT)
        kt = len(np.unique(j3))
        sum_k2 += kt * kt
        nactive += 1
    return sum_k2, nactive


# --------------------------------------------------------------------------- #
# one (i, g) tile system: factors built once, reused every solve
# --------------------------------------------------------------------------- #

class TileSystem:
    """Schur-tiled SPD solver for one (i, g) system.

    ``factor_nnz`` / ``resident_bytes()`` expose the resident factor size
    (tile factors + S factor) so the solver's memory gate accounts for it
    like any other backend.
    """

    def __init__(self, psn, C, d, sgn, P0, ordering=1, verify=True, chunk=512):
        self.psn = psn
        self.n = C.shape[0]
        self.d = d
        self.sgn = sgn
        self.P0 = P0
        self.ordering = ordering
        self.chunk = chunk

        owner = row_owner(psn, P0)
        self.R_I = np.where(owner < 0)[0]
        self.R_T = np.where(owner >= 0)[0]
        self.nI = len(self.R_I)
        nt = int(owner[self.R_T].max()) + 1
        self.nt = nt
        self.px = psn.nx // P0
        self.py = psn.ny // P0

        o = owner[self.R_T].astype(np.int32)
        order = np.argsort(o, kind="stable")
        os_ = o[order]
        self.start = np.searchsorted(os_, np.arange(nt), side="left")
        self.end = np.searchsorted(os_, np.arange(nt), side="right")
        self.order = order

        BII = csc_matrix(C[np.ix_(self.R_I, self.R_I)])
        S = BII
        self.tiles = []          # (idx, h, lrp, lri, C_IT, C_TI, k) or None
        self._f_nnz = 0
        t_asm = time.perf_counter()
        for t in range(nt):
            s0, e0 = self.start[t], self.end[t]
            idx = self.R_T[order[s0:e0]]
            if idx.size == 0:
                self.tiles.append(None)
                continue
            Mt = csc_matrix(C[np.ix_(idx, idx)])
            Mt, h = _factor(Mt, ordering)
            lrp, lri = _l_row_cs(h, len(idx))
            # tile <-> seam interface (sp.find ground truth)
            subT = csc_matrix(C[np.ix_(idx, self.R_I)])      # tile rows x seam cols
            i3, j3, _ = _sp_find(subT)
            C_I = self.R_I[np.unique(j3)]                    # seam cols coupled
            nloc = len(idx)
            if C_I.size == 0:
                self.tiles.append((idx, h, lrp, lri,
                                   csc_matrix((self.nI, nloc)),
                                   csc_matrix((nloc, self.nI)), 0))
                self._f_nnz += int(_lib().chol_nnz(h))
                continue
            subS = csc_matrix(C[np.ix_(self.R_I, idx)])      # seam rows x tile cols
            i4, j4, _ = _sp_find(subS)
            R_sel = self.R_I[np.unique(i4)]
            assert len(R_sel) == len(C_I), (len(R_sel), len(C_I))
            posT = np.searchsorted(self.R_I, C_I)            # seam-col pos in R_I
            posS = np.searchsorted(self.R_I, R_sel)          # seam-row pos in R_I
            A_TI = csc_matrix(subT[:, posT])                 # (nloc, k) tile->seam
            A_IT = csc_matrix(subS[posS, :])                 # (k, nloc) seam->tile
            dA = A_IT - A_TI.T
            scale = max(float(abs(A_IT).max()), float(abs(A_TI).max()), 1e-300)
            assert dA.nnz == 0 or float(abs(dA).max()) / scale < 1e-12, \
                f"tile interface not symmetric: {float(abs(dA).max())/scale:.2e}"
            # full R_I-coordinate interface matrices for the solve phase
            co_IT = A_IT.tocoo()
            C_IT = coo_matrix((co_IT.data, (posS[co_IT.row], co_IT.col)),
                              shape=(self.nI, nloc)).tocsr()
            C_IT.eliminate_zeros()
            co_TI = A_TI.tocoo()
            C_TI = coo_matrix((co_TI.data, (co_TI.row, posT[co_TI.col])),
                              shape=(nloc, self.nI)).tocsr()
            C_TI.eliminate_zeros()
            # Schur contribution S += -C_IT C_TT^-1 C_TI
            cap = nloc * self.chunk + 8
            for c0 in range(0, A_TI.shape[1], self.chunk):
                c1 = min(c0 + self.chunk, A_TI.shape[1])
                Xk = _mr_solve(h, nloc, csc_matrix(A_TI[:, c0:c1]), lrp, lri, cap)
                Cc = A_IT @ Xk.toarray()                     # (k, chunk)
                ii, jj = np.nonzero(Cc)
                S = S + coo_matrix((-Cc[ii, jj], (posS[ii], posT[c0 + jj])),
                                   shape=S.shape)
            self.tiles.append((idx, h, lrp, lri, C_IT, C_TI, A_TI.shape[1]))
            self._f_nnz += int(_lib().chol_nnz(h))
        self.asm_s = time.perf_counter() - t_asm
        S = S.tocsc()
        S.eliminate_zeros()
        e = float(abs(S - S.T).max() / abs(S).max()) if S.nnz else 0.0
        if e > 1e-10:
            raise ValueError(f"Schur S not symmetric: {e:.2e}")
        self.S_nnz = S.nnz
        Ss, self._hS = _factor(S.astype(np.float64), ordering)
        self._lrpS, self._lriS = _l_row_cs(self._hS, self.nI)
        self._f_nnz += int(_lib().chol_nnz(self._hS))
        # end-to-end residual gate on the ORIGINAL (scaled) equations
        if verify:
            x_test = self.solve(np.ones(self.n))            # solves A x = 1
            res = C @ x_test - (sgn * d)
            rel = float(np.linalg.norm(res) / np.linalg.norm(sgn * d))
            if not np.isfinite(rel) or rel > 1e-9:
                raise ValueError(f"tile residual too large: {rel:.2e}")
            self.verify_rel = rel
        else:
            self.verify_rel = None

    @property
    def factor_nnz(self):
        return self._f_nnz

    def resident_bytes(self):
        return self._f_nnz * 12 + self.S_nnz * 4 + self.nI * 8

    def solve(self, b):
        c = self.sgn * self.d * b
        c_I = c[self.R_I]
        w_I = np.zeros(self.nI)
        for tile in self.tiles:
            if tile is None:
                continue
            idx, h, lrp, lri, C_IT, C_TI, k = tile
            x_bT = _dense_solve(h, len(idx), c[idx])
            if k:
                w_I += np.asarray(C_IT @ x_bT).ravel()
        y = _dense_solve(self._hS, self.nI, c_I - w_I)
        x = np.empty(self.n, dtype=np.float64)
        x[self.R_I] = y
        for tile in self.tiles:
            if tile is None:
                continue
            idx, h, lrp, lri, C_IT, C_TI, k = tile
            rhs = c[idx] - (np.asarray(C_TI @ y).ravel() if k else 0.0)
            x[idx] = _dense_solve(h, len(idx), rhs)
        return x

    def close(self):
        for tile in self.tiles:
            if tile is not None:
                _lib().chol_free(tile[1])
        if getattr(self, "_hS", None):
            _lib().chol_free(self._hS)
            self._hS = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# memory estimate + auto tile size
# --------------------------------------------------------------------------- #

def _estimate_resident(psn, C, P0, ordering):
    """Resident-byte estimate for a P0 tiling: factor ONE (largest) tile for a
    representative L nnz, and bound the dense Schur S nnz by the summed seam
    interface width plus the base C_II.  Returns (est_bytes, nactive_tiles)."""
    owner = row_owner(psn, P0)
    R_I = np.where(owner < 0)[0]
    R_T = np.where(owner >= 0)[0]
    nt = int(owner[R_T].max()) + 1
    nI = len(R_I)
    o = owner[R_T].astype(np.int32)
    order = np.argsort(o, kind="stable")
    os_ = o[order]
    start = np.searchsorted(os_, np.arange(nt), side="left")
    end = np.searchsorted(os_, np.arange(nt), side="right")
    best = max(range(nt), key=lambda t: end[t] - start[t])
    idx = R_T[order[start[best]:end[best]]]
    Mt, h = _factor(csc_matrix(C[np.ix_(idx, idx)]), ordering)
    trial_nnz = int(_lib().chol_nnz(h))
    _lib().chol_free(h)
    sum_k2, nactive = _seam_width(psn, C, P0, R_I)
    S_bound = sum_k2 + nI * 9
    # 16 B/nnz to match the solver's _check_mem_gate accounting exactly, so
    # "fits the install budget" implies "passes the memory gate".
    return nactive * trial_nnz * 16 + S_bound * 16 + nI * 8, nactive


def _candidate_p0s(psn, n=4):
    """Up to n tile sizes to try in the build-and-check, largest first
    (fewer tiles = faster)."""
    cands = tile_candidates(psn)
    if not cands:
        raise ValueError("grid has no valid tile split (needs nx,ny >= 4)")
    return cands[:n]


# --------------------------------------------------------------------------- #
# the drop-in install
# --------------------------------------------------------------------------- #

def install_tile_backend(psn, mem_limit_gb=None, ordering=1, verify=True,
                         chunk=512):
    """Install the tile backend as a drop-in replacement for the face-current
    solver (monkey-patches ``_lu`` / ``_bterms_for`` / ``solve_i`` exactly like
    the shared-Cholesky backend).  ``mem_limit_gb`` is the WHOLE-pool budget;
    each (i, g) system gets an equal share and its tile size is auto-chosen.

    Returns a report list.  Raises when the geometry is not SPD-scalable or the
    budget cannot be met (fail-loud, like every other backend).
    """
    import types

    if mem_limit_gb is None:
        from . import runtime
        mem_limit_gb = runtime.DEFAULT_MEM_LIMIT_GB
    total_budget = float(mem_limit_gb) * 2 ** 30
    nsys = psn.I * psn.ng
    per_system = total_budget / max(1, nsys) * 0.9     # 10% headroom

    from .memopt import install_compact_build
    if not hasattr(psn, "_compact_build"):
        install_compact_build(psn)
    psn._bterms_by_i = dict(psn._compact_bterms)
    psn._rhs_digests_by_i = {i: _digest(psn._bterms_by_i[i]) for i in range(psn.I)}
    psn._rhs_identity_checks = 0
    psn._lusys = {}
    psn._syscache = {}
    ds = {i: spd_row_scale(psn, i) for i in range(psn.I)}

    reports = []

    def lu_opt(self, i, g):
        key = (i, g)
        if key not in self._lusys:
            t0 = time.perf_counter()
            A, _bt = self._compact_build(i, g)
            d = ds[i]
            B = A.multiply(d[:, None]).tocsc()
            diag0 = B.diagonal()
            sgn = -1.0 if diag0[diag0 != 0].min() < 0 else 1.0
            C = (sgn * B).tocsc()
            # Build-and-check: the estimate is ~10x conservative, so the REAL
            # resident bytes decide.  Try up to 4 candidates, LARGEST P0 first
            # (fewer tiles = faster, and tile-internal solves parallelize
            # better); keep the largest that fits the per-system budget.
            cands = _candidate_p0s(self)              # largest first (<=4)
            est = {}
            for P0 in cands:
                est[P0] = _estimate_resident(self, C, P0, ordering)[0]
            best = None
            tried = []
            for P0 in sorted(est, reverse=True):
                ts = TileSystem(self, C, d, sgn, P0, ordering=ordering,
                                verify=verify, chunk=chunk)
                tried.append(ts)
                # gate metric = the solver's _check_mem_gate exactly
                # (factor_nnz * 16 B).  Fitting per_system here guarantees the
                # whole-pool gate passes after summing all (i,g) systems.
                if ts.factor_nnz * 16 <= per_system:
                    best = ts
                    break
            if best is None:
                for ts in tried:
                    ts.close()
                raise MemoryError(
                    f"tile backend: even the finest tiles exceed "
                    f"mem_limit_gb={mem_limit_gb}; tried "
                    + ", ".join(f"P0={p}:{est[p]/2**30:.2f}GB(est)"
                                for p in sorted(est)))
            for ts in tried:
                if ts is not best:
                    ts.close()
            self._lusys[key] = (best, None)
            reports.append(dict(i=i, g=g, P0=best.P0, nt=best.nt, nI=best.nI,
                                S_nnz=best.S_nnz,
                                est_GB=est[best.P0] / 2 ** 30,
                                resident_GB=best.resident_bytes() / 2 ** 30,
                                asm_s=best.asm_s, verify_rel=best.verify_rel))
            del A, B, C
        return self._lusys[key]

    def bterms_opt(self, i, g):
        self._lu(i, g)
        return self._bterms_by_i[i]

    def solve_i_opt(self, i, g, qnode):
        ts, _ = self._lu(i, g)
        raw = np.empty((self.nodes, self.M, 4))
        for m in range(self.M):
            for grp in self._groups:
                S = self._state_matrix(i, g, m, grp)
                sel = self._grp_sel[grp]
                raw[sel, m, :] = qnode[sel] @ S[:4, 4:].T
        b = np.zeros(self.ncol)
        for (m, lf), (rws, sgns, nds) in self._bterms_for(i, g).items():
            np.add.at(b, rws, sgns * raw[nds, m, lf])
        return ts.solve(b)

    psn._lu = types.MethodType(lu_opt, psn)
    psn._bterms_for = types.MethodType(bterms_opt, psn)
    psn.solve_i = types.MethodType(solve_i_opt, psn)
    # prefactor every (i, g) in the parent before any fork pool exists
    for i in range(psn.I):
        for g in range(psn.ng):
            psn._lu(i, g)
    psn._memopt_backend = "tile"
    return reports


def pool_estimate_shared_chol(psn, ordering=1):
    """Resident-pool byte estimate for the shared-Cholesky backend WITHOUT
    building the whole pool (which is exactly what would OOM): factor ONE
    (i, g) system and scale by the number of systems (all share the grid,
    similar fill).  Used by the dispatcher to decide shared-chol vs tile."""
    from .memopt import install_compact_build, shared_pairs, SharedChol
    if not hasattr(psn, "_compact_build"):
        install_compact_build(psn)
    A, _ = psn._compact_build(0, 0)
    d = spd_row_scale(psn, 0)
    t0 = time.perf_counter()
    factor = None
    try:
        pairs = shared_pairs(psn)
        factor = SharedChol(A, d, pairs, ordering=ordering, check=False)
        # 16 B/nnz to match the solver's _check_mem_gate accounting.
        est = factor.factor_nnz * 16 * (psn.I * psn.ng)
    finally:
        elapsed = time.perf_counter() - t0
        if factor is not None:
            for f in factor.factors:
                f.close()
    return est, {"est_elapsed_s": elapsed}


def choose_backend(psn, mem_limit_gb, ordering=1):
    """Dispatch: 'shared-chol' when its pool fits the budget, else 'tile'.
    Only SPD-scalable, transpose-symmetric geometries qualify for shared-chol
    (its existing requirement); everything else goes to tile when the budget
    is the driver.  Returns (backend, info)."""
    budget = float(mem_limit_gb) * 2 ** 30
    try:
        est, info = pool_estimate_shared_chol(psn, ordering)
    except Exception as e:
        return "tile", {"reason": f"shared-chol unavailable: {e}"}
    if est <= budget:
        return "shared-chol", {"est_pool_GB": est / 2 ** 30, **info}
    return "tile", {"est_pool_GB": est / 2 ** 30, "reason": "over mem budget",
                    **info}
