# -*- coding: utf-8 -*-
"""Memory-optimized factorization backends for the PSN2D face-current solve.

Optional, opt-in (``--opt`` CLI flag or :func:`install_optimized`).  The
default solver path is untouched and bit-identical.

Three layers, each strictly weaker than the last, all fail-loud:

1. compact assembly (:func:`install_compact_build`)
   Vectorized FP64 assembly that preserves the exact row/column numbering of
   ``build_system`` (validated element-wise against it) but skips the
   per-row Python dict-of-dicts.  Assembly is ~100x faster and the per-system
   matrix is built straight in CSC.  No geometric restriction; works for
   square and rectangular nodes, TY and generic polars.

2. compact b-term tables (part of every install below)
   int32 right-hand-side tables shared across energy groups per polar angle,
   built once at install time (group-invariant by construction; a per-polar
   digest is recorded for diagnostics).

3a. compact SuperLU (:func:`install_compact_lu`)
   COLAMD (default) or symmetric-pattern MMD ordering.  ~2-3x less resident
   memory than the plain path on C5G7-class problems.

3b. shared sparse Cholesky (:func:`install_shared_chol`)
   The big one (measured ~9x peak-RSS on C5G7 quarter core M16, 30-round keff
   unchanged to 1e-15).  For A x = b it solves the row-equivalent system
   B x = d*b with B = diag(d) * A:

     * ``spd_row_scale`` picks d so B is exactly symmetric (the PSN
       face-current operator is negative definite after this row scaling —
       for TY3 the negative-definite certificate is a closed-form 2x2 modal
       block; for other polar sets the SPD property is still verified
       numerically by the Cholesky factorization itself);
     * angular mirror pairs (m, M-1-m) reduce each polar to M/2 column
       blocks;
     * on a transpose-symmetric geometry (square or rect grid,
       mat == mat.T, x/y-matched boundary types, transpose-matched widths
       for rect nodes, half-plane M divisible by 4) the
       (x,y)-transpose of the angular pair carries each block onto another
       and all four blocks share ONE sparse Cholesky factor
       (``xy_transpose_permutation``);
     * the factor is a C++ Eigen SimplicialLLT (METIS ordering) reached
       through a small C ABI (see ``psn2d/memopt_cxx/``); the factor object
       is COW-shared read-only by the forked sweep pool.

   Nonconforming geometries raise ``ValueError`` at install time (checked
   to 1e-12 per system, plus a random-RHS residual check of the ORIGINAL
   unscaled equations on every cached system) — they are never silently
   solved approximately.  Without the compiled bridge, backends 3b is
   unavailable and ``auto`` falls back to 3a.

``install_optimized(psn, backend)`` dispatches:

    auto  shared-chol if it passes every gate, else compact MMD-LU,
          else plain solver untouched
    chol  shared-chol, hard error if a gate fails
    mmd   compact assembly + symmetric-pattern MMD SuperLU
    lu    compact assembly + COLAMD SuperLU
    off   (default) do nothing
"""
from __future__ import annotations

import ctypes
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix, coo_matrix
from scipy.sparse.linalg import splu

_HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# 1. compact assembly
# --------------------------------------------------------------------------- #

def install_compact_build(psn):
    """Vectorized assembly matching build_system's numbering exactly.

    Adds ``psn._compact_build(i, g) -> (A_csc, bterms)`` where ``bterms`` is
    the int32 (m, lf) -> (rows, sgns, nodes) table (already compact).  The
    R-matrix table is indexed by (material, shape) groups, so the square
    (nshape == 1) and rectangular paths share one code path.
    """
    import types
    from psn2d.node import node_alphas

    if not hasattr(psn, "_cidx"):
        psn._vectorize_setup()
    M = psn.M
    nd = psn.nodes
    cd = psn._cidx.transpose(2, 0, 1)     # (nd, M, 4)
    sg = psn._esgn.transpose(2, 0, 1)     # (nd, M, 4)
    if psn.rect:
        sh_of_node = psn.shape_of_node
        nshape = int(psn.shape_of_node.max()) + 1
    else:
        sh_of_node = np.zeros(nd, np.int32)
        nshape = 1
    nmat = int(psn.mat.max()) + 1

    groups = {t: [] for t in ("internal", "vacuum", "reflect")}
    offset = 0
    for f, face in enumerate(psn.faces):
        t = face["type"]
        n = face["a"]
        lf = psn._local_face(n, f)
        count = M // 2 if t == "reflect" else M
        groups[t].append((f, n, lf, offset))
        offset += count
    assert offset == psn.ncol

    topo = {}
    row_parts, col_parts = [], []
    for typ, records in groups.items():
        if not records:
            continue
        f, n, lf, start = np.asarray(records, dtype=np.int32).T
        mat = psn.mat[psn.j_idx[n], psn.i_idx[n]]
        sh = sh_of_node[n]
        dirs = (np.asarray([psn.face_pairs[int(k)] for k in f], dtype=np.int32)[:, :, 0]
                if typ == "reflect" else np.broadcast_to(np.arange(M), (len(f), M)))
        rows = start[:, None] + np.arange(dirs.shape[1])[None, :]
        t = dict(f=f, n=n, lf=lf, mat=mat, sh=sh, dirs=dirs, rows=rows,
                 signs=sg[n[:, None], dirs], cols=cd[n[:, None], dirs])
        def append_contributions(cols):
            row_parts.append(np.broadcast_to(rows[:, :, None], cols.shape).ravel().astype(np.int32))
            col_parts.append(np.asarray(cols, dtype=np.int32).ravel())
        append_contributions(t["cols"])
        if typ == "internal":
            nb = np.asarray([psn.faces[int(k)]["b"] for k in f], dtype=np.int32)
            lfb = np.asarray([psn._local_face(int(v), int(k)) for v, k in zip(nb, f)], dtype=np.int32)
            t.update(nb=nb, lfb=lfb,
                     matb=psn.mat[psn.j_idx[nb], psn.i_idx[nb]],
                     shb=sh_of_node[nb],
                     signsb=sg[nb[:, None], dirs], colsb=cd[nb[:, None], dirs])
            append_contributions(t["colsb"])
        elif typ == "reflect":
            db = np.asarray([psn.face_pairs[int(k)] for k in f], dtype=np.int32)[:, :, 1]
            t.update(dirsb=db, signsb=sg[n[:, None], db], colsb=cd[n[:, None], db])
            append_contributions(t["colsb"])
        else:
            # vacuum diagonal: same _col_of mapping as build_system
            t["diag_cols"] = np.asarray([[psn._col_of(int(k), m)[0] for m in range(M)]
                                         for k in f], dtype=np.int32)
            t["diag_values"] = np.asarray([[psn.node_fmap[int(v)][int(l)][1] *
                                            psn._col_of(int(k), m)[1] for m in range(M)]
                                           for k, v, l in zip(f, n, lf)])
            row_parts.append(rows.ravel().astype(np.int32))
            col_parts.append(t["diag_cols"].ravel())
        topo[typ] = t
    rr = np.concatenate(row_parts)
    cc = np.concatenate(col_parts)

    # per-polar b-terms + vacuum coefficients (group-invariant; the install
    # below verifies the digest on every (i, g) system)
    bt_cache, vacuum_coeff = {}, {}
    for i in range(psn.I):
        acc = defaultdict(lambda: ([], [], []))
        for typ, t in topo.items():
            n, lf, rows, dirs = (t[k] for k in ("n", "lf", "rows", "dirs"))
            if typ == "vacuum":
                alphas = np.asarray([psn._alpha_for(i, m) if psn.generic
                                     else node_alphas(psn.mu[i], psn.h,
                                                      psn._phi_m(m), psn.dphi)
                                     for m in range(M)])
                alpha = alphas[dirs, lf[:, None]]
                cos = np.cos(np.asarray([psn._phi_m(m) for m in range(M)])[None, :]
                             - np.asarray(psn.face_phi)[t["f"]][:, None])
                coef = 2. * np.where(cos >= 0., 1., -1.) * alpha
                vacuum_coeff[i] = coef
                terms = [(n, lf, dirs, coef)]
            elif typ == "internal":
                terms = [(n, lf, dirs, -np.ones_like(rows, dtype=float)),
                         (t["nb"], t["lfb"], dirs, np.ones_like(rows, dtype=float))]
            else:
                terms = [(n, lf, dirs, -np.ones_like(rows, dtype=float)),
                         (n, lf, t["dirsb"], np.ones_like(rows, dtype=float))]
            for nodes, lfaces, angles, coefs in terms:
                ns = np.broadcast_to(nodes[:, None], rows.shape)
                ls = np.broadcast_to(lfaces[:, None], rows.shape)
                for m in range(M):
                    for l in range(4):
                        mask = (angles == m) & (ls == l)
                        if np.any(mask):
                            ar = acc[(m, l)]
                            ar[0].append(rows[mask].astype(np.int32))
                            ar[1].append(coefs[mask])
                            ar[2].append(ns[mask].astype(np.int32))
        bt_cache[i] = {k: tuple(np.concatenate(parts) for parts in lists)
                       for k, lists in acc.items()}

    node_of_grp = psn._node_of_grp

    def compact_build(self, i, g):
        rt = np.zeros((nmat, nshape, M, 4, 4))
        for grp in self._groups:
            mat, sh = grp
            n0 = node_of_grp[grp]
            for m in range(M):
                rt[mat, sh, m] = self._R(i, g, m, mat, n0)
        val_parts = []
        for typ, t in topo.items():
            n, mat, sh, lf, dirs = (t[k] for k in ("n", "mat", "sh", "lf", "dirs"))
            va = rt[mat[:, None], sh[:, None], dirs, lf[:, None], :] * t["signs"]
            if typ == "internal":
                vb = -rt[t["matb"][:, None], t["shb"][:, None], dirs,
                         t["lfb"][:, None], :] * t["signsb"]
                val_parts.extend((va.ravel(), vb.ravel()))
            elif typ == "reflect":
                vb = -rt[mat[:, None], sh[:, None], t["dirsb"],
                         lf[:, None], :] * t["signsb"]
                val_parts.extend((va.ravel(), vb.ravel()))
            else:
                va *= -vacuum_coeff[i][:, :, None]
                val_parts.extend((va.ravel(), t["diag_values"].ravel()))
        A = coo_matrix((np.concatenate(val_parts), (rr, cc)),
                       shape=(self.ncol, self.ncol)).tocsc()
        return A, bt_cache[i]

    psn._compact_build = types.MethodType(compact_build, psn)
    psn._compact_bterms = bt_cache
    psn._compact_topology_nbytes = rr.nbytes + cc.nbytes


def compact_rows(rows_rhs, index_dtype=np.int32):
    """(m, lf) -> (rows, sgns, nodes) from build_system's rows_rhs (int32)."""
    acc = defaultdict(lambda: ([], [], []))
    for row, terms in rows_rhs:
        for n, m, lf, sgn in terms:
            a = acc[(m, lf)]
            a[0].append(row)
            a[1].append(sgn)
            a[2].append(n)
    return {k: (np.asarray(v[0], dtype=index_dtype),
                np.asarray(v[1], dtype=np.float64),
                np.asarray(v[2], dtype=index_dtype))
            for k, v in acc.items()}


def _digest(bt):
    """Stable byte digest of a (m, lf) -> (rows, sgns, nodes) table."""
    import hashlib
    h = hashlib.sha256()
    for k in sorted(bt):
        r, s, n = bt[k]
        h.update(f"{k[0]}:{k[1]}:".encode())
        h.update(r.tobytes())
        h.update(s.tobytes())
        h.update(n.tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# 3b. SPD row scaling + exact geometric sharing
# --------------------------------------------------------------------------- #

def spd_row_scale(psn, i):
    """Diagonal row scaling d with B = diag(d) A exactly symmetric.

    Local outward current is ``e`` times the global current, with e = -1 at
    left/bottom boundaries and +1 at right/top.  Reflective slaves already
    carry the further -1 elimination sign folded into A.  Valid for
    arbitrary configured faces (TY or generic polar, square or rect nodes);
    each row is weighted by its face length (x-faces: the node's y-width,
    y-faces: its x-width; square nodes: h on every face), which is what
    restores exact symmetry for rectangular nodes.
    """
    from psn2d.node import node_alphas

    d = np.empty(psn.ncol, dtype=np.float64)
    row = 0
    for f, fc in enumerate(psn.faces):
        n = fc["a"]
        lf = psn._local_face(n, f)
        j, ii = psn.j_idx[n], psn.i_idx[n]
        # face-length factor: x-faces carry the node's y-width, y-faces its
        # x-width (square nodes: both equal to h, the historical convention)
        ell = float(psn.hy[j, ii] if lf < 2 else psn.hx[j, ii]) if psn.rect else float(psn.h)
        if fc["type"] == "internal":
            d[row:row + psn.M] = -ell
            row += psn.M
            continue
        e = -1.0 if fc["side"] in (0, 1) else 1.0
        if fc["type"] == "reflect":
            count = len(psn.face_pairs[f])
            d[row:row + count] = -e * ell
            row += count
            continue
        if fc["type"] != "vacuum":
            raise ValueError(f"Unknown face type: {fc['type']}")
        for m in range(psn.M):
            alpha = psn._alpha_for(i, m)
            if alpha is None:
                alpha = node_alphas(psn.mu[i], psn.h,
                                    psn._phi_m(m), psn.dphi)
            if not np.isfinite(alpha[lf]) or alpha[lf] == 0:
                raise ValueError("Vacuum boundary alpha must be finite and nonzero")
            d[row] = e * ell / (2.0 * abs(alpha[lf]))
            row += 1
    if row != psn.ncol:
        raise AssertionError((row, psn.ncol))
    return d


def xy_transpose_permutation(psn):
    """Signed column map of the simultaneous x/y + angular transpose.

    Returns p, s with T[p[j], j] = s[j] where T is an orthogonal involution.
    On a transpose-symmetric material mesh with x/y-matched boundary types
    the exact SPD operator obeys T.T @ B @ T = B.  Rectangular nodes require
    transpose-matched widths (hx[j,i] == hy[i,j]) so the transposed node has
    the mirror aspect ratio.  Raises ValueError when the geometry does not
    conform (fail-loud, no silent approximation).
    """
    if not np.array_equal(psn.mat, psn.mat.T):
        raise ValueError("Exact reuse requires a transpose-symmetric "
                         "material grid")
    if psn.rect:
        if psn.nx != psn.ny:
            raise ValueError("Exact reuse requires a square node grid "
                             "(nx == ny)")
        if not np.allclose(psn.hx, psn.hy.T, rtol=0.0, atol=1e-12):
            raise ValueError("Exact reuse requires transpose-matched node "
                             "widths (hx[j,i] == hy[i,j])")
    elif psn.nx != psn.ny:
        raise ValueError("Exact reuse requires a square node grid")
    if psn.bnd[0] != psn.bnd[1] or psn.bnd[2] != psn.bnd[3]:
        raise ValueError("Exact reuse requires x/y-matched boundary types")
    if psn.M % 4 or psn.full_2pi:
        raise ValueError("Exact reuse supports half-plane M divisible by 4")
    p = np.empty(psn.ncol, dtype=np.int32)
    signs = np.empty(psn.ncol, dtype=np.float64)
    local_face_arrays = [psn.fxp, psn.fxm, psn.fyp, psn.fym]
    for f, fc in enumerate(psn.faces):
        n = fc["a"]
        nt = int(psn.i_idx[n] * psn.nx + psn.j_idx[n])
        lf = psn._local_face(n, f)
        ft = int(local_face_arrays[(lf + 2) % 4][nt])
        for m in range(psn.M):
            j = psn.colidx[f, m]
            if j < 0:
                continue
            mt = (psn.M // 2 - 1 - m) % psn.M
            dest, sign = psn._col_of(ft, mt)
            p[j] = dest
            signs[j] = sign
    if not np.array_equal(p[p], np.arange(psn.ncol)):
        raise AssertionError("Transpose permutation is not an involution")
    if not np.array_equal(signs * signs[p], np.ones(psn.ncol)):
        raise AssertionError("Signed transpose is not an involution")
    return p, signs


def mirror_pair_columns(psn):
    """M/2 index arrays partitioning columns into angular mirror pairs."""
    if psn.full_2pi:
        raise ValueError("Mirror-pair blocks are for the half-plane convention")
    cols = []
    for m in range(psn.M // 2):
        angles = [m, psn.M - 1 - m]
        indexes = psn.colidx[:, angles].ravel()
        cols.append(np.sort(indexes[indexes >= 0]).astype(np.int32))
    return cols


def shared_pairs(psn):
    """[(c, ct, s)] — (block columns, twin columns, twin row scale) per pair."""
    perm, sign = xy_transpose_permutation(psn)
    blocks = mirror_pair_columns(psn)
    if psn.M % 4:
        raise ValueError("Sharing requires M divisible by four")
    pairs = [(blocks[k], perm[blocks[k]], sign[blocks[k]]) for k in range(psn.M // 4)]
    allcols = np.concatenate([v for c, ct, s in pairs for v in (c, ct)])
    if len(allcols) != psn.ncol or not np.array_equal(np.sort(allcols), np.arange(psn.ncol)):
        raise ValueError("xy block maps do not partition columns")
    return pairs


# --------------------------------------------------------------------------- #
# C++ sparse Cholesky bridge (Eigen SimplicialLLT, AMD or METIS ordering)
# --------------------------------------------------------------------------- #

def _find_eigen_chol():
    cands = []
    env = os.environ.get("OPENPSN_EIGEN_CHOL")
    if env:
        cands.append(env)
    cands.append(str(_HERE / "memopt_cxx" / "libeigen_chol.so"))
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def _load_eigen_chol():
    path = _find_eigen_chol()
    if path is None:
        raise RuntimeError(
            "shared-Cholesky backend needs psn2d/memopt_cxx/libeigen_chol.so; "
            "build it with:  python psn2d/memopt_cxx/build.py "
            "[--eigen-include DIR --metis-include DIR --metis-lib DIR]\n"
            "(or point OPENPSN_EIGEN_CHOL at an existing .so)")
    lib = ctypes.CDLL(path)
    ip, dp = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)
    lib.chol_factor.argtypes = [ctypes.c_int, ctypes.c_int, ip, ip, dp,
                                ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.chol_factor.restype = ctypes.c_void_p
    lib.chol_solve.argtypes = [ctypes.c_void_p, ctypes.c_int, dp, dp]
    lib.chol_solve.restype = None
    lib.chol_nnz.argtypes = [ctypes.c_void_p]
    lib.chol_nnz.restype = ctypes.c_longlong
    lib.chol_free.argtypes = [ctypes.c_void_p]
    lib.chol_free.restype = None
    return lib


class EigenChol:
    """ctypes handle to a C++ SimplicialLLT factor of a CSC matrix.

    The factor is a C++ heap object; forked sweep workers share it COW
    read-only (chol_solve never writes the factor).
    """

    def __init__(self, B, ordering=1):
        lib = _load_eigen_chol()
        self._lib = lib
        self.h = None
        if B.dtype != np.float64 or B.indices.dtype != np.int32 \
                or B.indptr.dtype != np.int32:
            raise TypeError("C ABI requires FP64 values and int32 sparse indices")
        if B.shape[0] != B.shape[1]:
            raise ValueError("Cholesky matrix must be square")
        B = B.tocsc()
        B.sort_indices()
        status = ctypes.c_int()
        self.n = B.shape[0]
        self.h = lib.chol_factor(self.n, B.nnz,
                                 B.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                 B.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                 B.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                 ordering, ctypes.byref(status))
        if not self.h:
            raise RuntimeError(f"Cholesky factorization failed: status={status.value}")

    def solve(self, b):
        b = np.ascontiguousarray(b, dtype=np.float64)
        if b.shape != (self.n,):
            raise ValueError("Expected a one-dimensional RHS of matrix dimension")
        x = np.empty_like(b)
        self._lib.chol_solve(self.h, self.n,
                             b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                             x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return x

    @property
    def nnz(self):
        return self._lib.chol_nnz(self.h)

    def close(self):
        if self.h:
            self._lib.chol_free(self.h)
            self.h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class SharedChol:
    """Exact shared-Cholesky solve of the original (unscaled) system.

    Solves A x = b via B = diag(d) A with (M/4) independent blocks: for each
    mirror pair (c, ct, s) one factor serves both x[c] = L^-1 (d b)[c] and
    x[ct] = s * L^-1 (s d b)[ct]  (the twin block equals s^-1 diag(s) B_c
    diag(s) s^-1, verified numerically to <= 1e-12 at construction).
    """

    def __init__(self, A, d, pairs, ordering=1, check=True):
        self.d = d
        self.n = A.shape[0]
        self.pairs = pairs
        self.factors = []
        self.symmetry_errors = []
        B = A.multiply(d[:, None]).tocsc()
        if check:
            delta = B - B.T
            err = float(abs(delta).max()) / float(abs(B).max()) if delta.nnz else 0.
            if not np.isfinite(err) or err > 1e-12:
                raise ValueError(f"Row-scaled matrix not finite/symmetric: {err}")
            self.scaled_asymmetry = err
            del delta
        for c, ct, s in pairs:
            block = B[c, :][:, c].tocsc()
            if check:
                twin = B[ct, :][:, ct].multiply(s[:, None]).multiply(s[None, :]).tocsc()
                delta = block - twin
                err = float(abs(delta).max()) / float(abs(block).max()) if delta.nnz else 0.
                if not np.isfinite(err) or err > 1e-12:
                    raise ValueError(f"Geometry sharing not finite/exact: {err}")
                self.symmetry_errors.append(err)
                del twin, delta
            self.factors.append(EigenChol(block, ordering=ordering))
        self.factor_nnz = sum(f.nnz for f in self.factors)
        self.L_and_permutations_payload_bytes = sum(
            f.nnz * 12 + (len(c) + 1) * 4 + len(c) * 8
            for f, (c, ct, s) in zip(self.factors, self.pairs))

    def solve(self, b):
        rhs = self.d * b
        x = np.empty(self.n, dtype=np.float64)
        for f, (c, ct, s) in zip(self.factors, self.pairs):
            x[c] = f.solve(rhs[c])
            x[ct] = s * f.solve(s * rhs[ct])
        return x


# --------------------------------------------------------------------------- #
# installs (instance-level monkey patches, same contract as the solver)
# --------------------------------------------------------------------------- #

def install_shared_chol(psn, verify=True, ordering=1):
    """Compact assembly + SPD row scaling + xy factor sharing + C++ Cholesky.

    Raises ValueError/RuntimeError when the geometry or the compiled bridge
    is not available — callers (``auto``) catch and fall back.
    """
    import types
    import time

    install_compact_build(psn) if not hasattr(psn, "_compact_build") else None
    pairs = shared_pairs(psn)
    ds = {i: spd_row_scale(psn, i) for i in range(psn.I)}
    # Pre-fill the per-polar b-tables HERE (single-threaded): _prefactor
    # runs on a thread pool, so lu_opt must only READ shared state.
    psn._bterms_by_i = dict(psn._compact_bterms)
    psn._rhs_digests_by_i = {i: _digest(bt) for i, bt in psn._bterms_by_i.items()}
    psn._rhs_identity_checks = 0
    psn._chol_records = []
    # drop any factors built by the plain path on this instance — stale
    # entries would be silently reused instead of the new backend
    psn._lusys = {}
    psn._syscache = {}

    def lu_opt(self, i, g):
        key = (i, g)
        if key not in self._lusys:
            t = time.perf_counter()
            A, _bt = self._compact_build(i, g)
            del _bt
            factor = SharedChol(A, ds[i], pairs, ordering=ordering, check=verify)
            if verify:
                # random non-symmetric RHS of the ORIGINAL unscaled A:
                # the strongest available correctness gate per system
                rng = np.random.default_rng(1729 + 100 * i + g)
                test_rhs = rng.normal(size=A.shape[0])
                test_x = factor.solve(test_rhs)
                residual = A @ test_x - test_rhs
                rel_residual = float(np.linalg.norm(residual) / np.linalg.norm(test_rhs))
                max_residual = float(np.max(np.abs(residual)))
                if not np.isfinite(rel_residual) or rel_residual > 1e-11:
                    raise ValueError(f"Original equation residual too large: {rel_residual}")
                del test_rhs, test_x, residual
            self._lusys[key] = (factor, None)
            self._chol_records.append(dict(
                i=i, g=g, elapsed_s=time.perf_counter() - t,
                factor_nnz=factor.factor_nnz,
                scaled_asymmetry=getattr(factor, "scaled_asymmetry", None),
                sharing_errors=factor.symmetry_errors,
            ))
            del A
        return self._lusys[key]

    def bterms_opt(self, i, g):
        self._lu(i, g)
        return self._bterms_by_i[i]

    def solve_i_opt(self, i, g, qnode):
        factor, _ = self._lu(i, g)
        raw = np.empty((self.nodes, self.M, 4))
        for m in range(self.M):
            for grp in self._groups:
                S = self._state_matrix(i, g, m, grp)
                sel = self._grp_sel[grp]
                raw[sel, m, :] = qnode[sel] @ S[:4, 4:].T
        b = np.zeros(self.ncol)
        for (m, lf), (rws, sgns, nds) in self._bterms_for(i, g).items():
            np.add.at(b, rws, sgns * raw[nds, m, lf])
        return factor.solve(b)

    psn._lu = types.MethodType(lu_opt, psn)
    psn._bterms_for = types.MethodType(bterms_opt, psn)
    psn.solve_i = types.MethodType(solve_i_opt, psn)
    # Prefactor every (i, g) system HERE, in the parent, before the fork
    # pool can exist: the forked _sweep_worker reads psn._lusys[(i, g)]
    # DIRECTLY (bypassing _lu), so a missing key in a child would silently
    # factor in the child's COW copy.  With all entries present, no child
    # ever factors — the C++ factor is shared read-only (COW), and any
    # per-system verification failure surfaces in the parent before the
    # sweep starts.
    for i in range(psn.I):
        for g in range(psn.ng):
            psn._lu(i, g)
    psn._memopt_backend = "shared-chol"


def install_compact_lu(psn, permc_spec="COLAMD", verify_topology=True):
    """Compact assembly + int32 b-terms + SuperLU (COLAMD or MMD)."""
    import types

    install_compact_build(psn)
    # Pre-fill the per-polar b-tables HERE (single-threaded): _prefactor
    # runs on a thread pool, so lu_opt must only READ shared state.
    psn._bterms_by_i = dict(psn._compact_bterms)
    psn._rhs_digests_by_i = {i: _digest(bt) for i, bt in psn._bterms_by_i.items()}
    psn._rhs_identity_checks = 0
    # drop any factors built by the plain path on this instance — stale
    # entries would be silently reused instead of the new backend
    psn._lusys = {}
    psn._syscache = {}

    def lu_opt(self, i, g):
        key = (i, g)
        if key not in self._lusys:
            A, _bt = self._compact_build(i, g)
            del _bt
            if permc_spec == "MMD_AT_PLUS_A":
                lu = splu(csc_matrix(A), permc_spec="MMD_AT_PLUS_A",
                          diag_pivot_thresh=0.0, options={"SymmetricMode": True})
            else:
                lu = splu(csc_matrix(A), permc_spec=permc_spec)
            del A
            self._lusys[key] = (lu, None)
        return self._lusys[key]

    def bterms_opt(self, i, g):
        self._lu(i, g)
        return self._bterms_by_i[i]

    def solve_i_opt(self, i, g, qnode):
        lu, _ = self._lu(i, g)
        raw = np.empty((self.nodes, self.M, 4))
        for m in range(self.M):
            for grp in self._groups:
                S = self._state_matrix(i, g, m, grp)
                sel = self._grp_sel[grp]
                raw[sel, m, :] = qnode[sel] @ S[:4, 4:].T
        b = np.zeros(self.ncol)
        for (m, lf), (rws, sgns, nds) in self._bterms_for(i, g).items():
            np.add.at(b, rws, sgns * raw[nds, m, lf])
        return lu.solve(b)

    psn._lu = types.MethodType(lu_opt, psn)
    psn._bterms_for = types.MethodType(bterms_opt, psn)
    psn.solve_i = types.MethodType(solve_i_opt, psn)
    psn._memopt_backend = f"compact-{permc_spec.lower()}"


def install_optimized(psn, backend="auto", mem_limit_gb=None):
    """Install a memory-optimized backend.  Returns a report dict.

    auto (memory-aware): use shared-Cholesky when its factor pool fits
    ``mem_limit_gb``; otherwise switch to the **tile** backend, which never
    materializes a full-system factor (Schur tiles, auto-sized from the same
    budget).  If the tile backend is unavailable for the geometry/bridge it
    degrades to compact-mmd -> plain (fail-loud if none survive).

    tile: force the tile backend (auto tile size from ``mem_limit_gb``).
    chol/mmd/lu: the explicit legacy backends (unchanged).
    """
    if backend in (None, "off"):
        psn._memopt_backend = "plain"
        return {"backend": "plain", "reason": "opt off"}
    if backend in ("lu", "mmd", "chol", "auto", "tile"):
        pass
    else:
        raise ValueError(f"unknown backend {backend!r}")

    from . import tile as _tile
    if mem_limit_gb is None:
        from . import runtime
        mem_limit_gb = runtime.DEFAULT_MEM_LIMIT_GB

    def _try_tile():
        try:
            return _tile.install_tile_backend(psn, mem_limit_gb=mem_limit_gb)
        except Exception as e:
            psn._lusys = {}
            psn._syscache = {}
            return e

    if backend == "tile":
        rep = _try_tile()
        if isinstance(rep, Exception):
            raise RuntimeError(f"tile backend failed: {rep}")
        return {"backend": "tile", "systems": rep, "mem_limit_gb": mem_limit_gb}

    if backend == "lu":
        install_compact_lu(psn, permc_spec="COLAMD")
        return {"backend": psn._memopt_backend, "reason": "requested"}
    if backend == "mmd":
        install_compact_lu(psn, permc_spec="MMD_AT_PLUS_A")
        return {"backend": psn._memopt_backend, "reason": "requested"}

    errors = []
    fits = True
    est_pool_gb = None
    if backend in ("chol", "auto"):
        if backend == "auto":
            # memory-aware: estimate the shared-chol pool BEFORE building it
            # (building is exactly the step that would OOM).
            try:
                est, _ = _tile.pool_estimate_shared_chol(psn)
                est_pool_gb = est / 2 ** 30
                fits = est <= mem_limit_gb * 2 ** 30
            except Exception as e:
                fits = False
                errors.append(f"shared-chol estimate: {e}")
        if fits:
            try:
                install_shared_chol(psn)
                reason = "requested" if backend == "chol" \
                    else "pool fits mem budget"
                rep = {"backend": "shared-chol", "reason": reason,
                       "mem_limit_gb": mem_limit_gb}
                if est_pool_gb is not None:
                    rep["est_pool_gb"] = round(est_pool_gb, 3)
                return rep
            except (ValueError, RuntimeError, TypeError, AssertionError) as e:
                errors.append(f"shared-chol: {e}")
                psn._lusys = {}          # drop any half-built entries
        elif backend == "chol":
            # explicit chol but the pool won't fit: fail-loud
            raise RuntimeError(
                f"shared-Cholesky pool est. {est_pool_gb:.3f} GB exceeds "
                f"mem_limit_gb={mem_limit_gb}; use backend 'tile' or raise "
                f"the limit")
    # auto: pool over budget (or shared-chol unavailable) -> tile
    if backend == "auto":
        rep = _try_tile()
        if not isinstance(rep, Exception):
            why = ("pool over mem budget" if not fits
                   else "shared-chol unavailable")
            out = {"backend": "tile", "systems": rep,
                   "reason": f"auto -> tile ({why})",
                   "mem_limit_gb": mem_limit_gb}
            if est_pool_gb is not None:
                out["est_pool_gb"] = round(est_pool_gb, 3)
            return out
        errors.append(f"tile: {rep}")
    if backend == "chol":
        raise RuntimeError(f"shared-Cholesky unavailable: {errors}")
    try:
        install_compact_lu(psn, permc_spec="MMD_AT_PLUS_A")
        return {"backend": psn._memopt_backend, "reason": "; ".join(errors) or "mmd requested"}
    except Exception as e:                                    # pragma: no cover
        psn._lusys = {}
        psn._memopt_backend = "plain"
        return {"backend": "plain", "reason": "all backends failed: "
                + "; ".join(errors + [str(e)])}
