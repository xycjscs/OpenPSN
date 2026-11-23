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
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix, coo_matrix
from scipy.sparse.linalg import splu

_HERE = Path(__file__).resolve().parent

# dlopen'd C++ Cholesky bridge (see _load_eigen_chol); None until first use.
_CHOL_LIB = None


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


def _preload_metis(bridge_dir):
    """dlopen a METIS library the dynamic loader could not find by itself.

    The C++ bridge carries ``NEEDED libmetis.so`` — the *development* name,
    which distro packages ship only in ``libmetis-dev``; a runtime-only host
    has ``libmetis.so.5`` and the plain dlopen fails even though METIS is
    installed.  Search order: ``OPENPSN_METIS_LIB``, a library sitting next
    to the bridge (self-contained deployment), the ``metisx/lib`` directory
    beside the kernel deployment this checkout uses (``_RUST_DIR_DEFAULT``'s
    sibling -- without it ``--opt chol`` dies here on a host that has no
    ``libmetis-dev``, which is exactly how the Rust backend failed before its
    own loader was fixed), then the distro SONAME on the normal search path.
    Returns the path loaded, or None if nothing matched.
    """
    cands = []
    env = os.environ.get("OPENPSN_METIS_LIB")
    if env:
        cands.append(env)
    cands += sorted(str(p) for p in Path(bridge_dir).glob("libmetis.so*"))
    for d in (os.path.join(os.path.dirname(_RUST_DIR_DEFAULT), "metisx", "lib"),
              os.path.join(_RUST_DIR_DEFAULT, "..", "metisx", "lib")):
        cands += sorted(str(p) for p in Path(d).glob("libmetis.so*"))
    cands += ["libmetis.so.5", "libmetis.so.4"]
    for c in cands:
        try:
            ctypes.CDLL(c, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
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
    global _CHOL_LIB
    if _CHOL_LIB is not None:
        return _CHOL_LIB
    try:
        lib = ctypes.CDLL(path)
    except OSError as exc:
        if "metis" not in str(exc).lower():
            raise
        loaded = _preload_metis(str(Path(path).parent))
        if loaded is None:
            raise RuntimeError(
                f"{path} needs libmetis.so but none was found: install "
                f"libmetis-dev, drop a libmetis.so* next to the bridge, or "
                f"set OPENPSN_METIS_LIB=/path/to/libmetis.so.5 "
                f"(loader said: {exc})") from exc
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
    _CHOL_LIB = lib        # one dlopen + one argtype setup per process
    return lib


_RUST_DIR_ENV = "OPENPSN_RUST_CHOL"
_RUST_DIR_DEFAULT = str(Path(__file__).resolve().parents[1] / "rust")


def _load_rust_shared_chol():
    """Import the Rust Cholesky bridge (``rustchol.RustSharedChol``).

    The compiled kernel is a Rust cdylib (``libpsn_chol.so``) plus a thin
    ctypes layer, built in the repository's ``rust/`` directory by default.
    ``OPENPSN_RUST_CHOL`` can point to the directory holding both files or
    the shared object itself. It fails LOUD when either piece
    is missing: quietly falling back to the C++ bridge would let a run report
    the solve timings of a kernel it never used.

    Same contract as ``SharedChol`` -- ``solve(b)`` solves A x = b for the
    ORIGINAL unscaled A, ``factor_nnz``/``symmetry_errors``/
    ``scaled_asymmetry`` are published the same way -- so everything
    downstream (the per-system residual gate in ``lu_opt``, the pool
    accounting, the forked sweep workers reading the factor COW read-only)
    is unchanged.  Measured on M16_S2: the fused solve is 1.60x the Eigen
    one, the prefactor is 8% slower (its single-threaded rate is 0.53x, so
    this backend is for the pooled path, not a 1-worker factorisation).

    METIS is a requirement here too -- the Rust kernel asks for the same
    ordering library as the C++ bridge -- so the library is dlopened
    RTLD_GLOBAL before the import, which is what ``rustchol`` then finds
    through RTLD_DEFAULT.  Left to itself the kernel only searches
    ``$OPENPSN_METIS_LIB`` and the distro SONAMEs, so a self-contained
    deployment (METIS sitting next to the kernel, or in ``metisx/lib``
    beside it) would fail deep inside a prefactor worker; here it either
    resolves or fails loud at install time, which is the same rule as above.
    """
    import sys as _sys
    p = os.environ.get(_RUST_DIR_ENV) or _RUST_DIR_DEFAULT
    # Either spelling is accepted: a directory holding the bridge, or the shared
    # object itself.  ``rustchol`` reads the SAME variable as a library path, so
    # the documented "set it to the directory" used to die inside ctypes with
    # "cannot read file data: Is a directory" -- the override only ever worked
    # while it pointed at the default.
    d = p if os.path.isdir(p) else os.path.dirname(os.path.abspath(p))
    missing = [f for f in ("rustchol.py", "libpsn_chol.so")
               if not os.path.exists(os.path.join(d, f))]
    if missing:
        raise RuntimeError(
            f"Rust Cholesky bridge incomplete in {d!r}: missing "
            f"{', '.join(missing)} (set {_RUST_DIR_ENV})")
    if not os.environ.get("OPENPSN_METIS_LIB"):
        for cand in (d, os.path.join(os.path.dirname(d), "metisx", "lib")):
            if _preload_metis(cand) is not None:
                break
        else:
            raise RuntimeError(
                f"the Rust Cholesky backend needs METIS (AMG ordering) and "
                f"none was found next to {d!r}; install libmetis-dev or set "
                f"OPENPSN_METIS_LIB=/path/to/libmetis.so")
    if d not in _sys.path:
        _sys.path.insert(0, d)
    # ... and pin the variable to the shared object it resolved to.  `rustchol`
    # dlopens lazily, on the first factor, long after this function returned --
    # restoring the directory here would put the ctypes error back one level
    # down.  Both readers accept the file spelling, and forked sweep workers
    # inherit it, so one resolution is enough for the whole run.
    os.environ[_RUST_DIR_ENV] = os.path.join(d, "libpsn_chol.so")
    from rustchol import RustSharedChol         # noqa: E402
    return RustSharedChol


def install_fused_source_update(psn):
    """Fuse the B.3 source update into the Rust library (``psn_src_update``).

    ``PSN2D._source_update`` runs once per outer iteration and is pure data
    motion: NumPy builds the whole ``(ng, ng, nodes)`` coefficient array (5.8
    MiB on the quarter core) before reducing over the input group with a stride
    of ``nodes``, i.e. seven streams through 12 MiB.  In situ that is ~19 ms of
    the ~48 ms an iteration spends outside the sweep (``dev/runphase.py`` times
    it in place).  The kernel folds the coefficient build into the reduction and
    takes the two scatter tables node-major -- one contiguous ``ng x ng`` per
    node instead of a stride-``nodes`` walk -- which is affordable because the
    node -> material map is fixed for the whole run, so that pack is paid once.
    Each array is then touched exactly once and ``out`` is written once instead
    of fifty temporaries.

    The arithmetic per output element is deliberately unchanged -- ascending
    input-group accumulation from exact zero, no FMA -- because this loop feeds
    an accelerator whose sweep count is chaotic in the last ulp: check
    ``dev/srcident.py`` (byte-for-byte against the Python loop) and the three
    gate cases before believing any timing.  Returns False, keeping the NumPy
    path, when the loaded library predates the symbol; ``OPENPSN_SRC_RS=0`` is
    the A/B knob that keeps the NumPy path in a build that has it.
    """
    try:
        from rustchol import has_source_update, source_update   # noqa: E402
    except ImportError:
        return False
    if not has_source_update():
        return False
    tab = [None, None, None]                    # [matidx, base, nu]
    chi = np.ascontiguousarray(psn.chi, dtype=np.float64)
    shape, ng, nodes = (psn.ng, psn.nodes, 5), psn.ng, psn.nodes

    def _fused(qim, lam, matidx):
        if tab[0] is not matidx:                # node -> material map, fixed
            tab[:] = [matidx,
                      np.ascontiguousarray(psn.Sgg.transpose(2, 0, 1)[matidx]),
                      np.ascontiguousarray(psn.nuSf[:, matidx].T)]
        q = qim if (qim.dtype == np.float64 and qim.flags.c_contiguous) \
            else np.ascontiguousarray(qim, dtype=np.float64)
        out = np.empty(shape, dtype=np.float64)
        source_update(ng, nodes, tab[1], tab[2], chi, lam, q, out)
        return out

    psn._source_update = _fused
    psn._fused_src_update = True
    return True


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

_TAB_PSN = None   # the solver forked table workers see, see ``_fast_tab_one``


def _fast_tab_one(ig):
    """One ``(i, g)`` entry of ``fast_sweep_tables``.

    Module-level and reading a module global rather than taking ``psn`` as an
    argument, because a forked worker inherits the parent's vectorised solver
    by COW: pickling it into the task would serialise the whole mesh to get
    tables the size of a few hundred kilobytes.
    """
    psn = _TAB_PSN
    i, g = ig
    M = psn.M
    T, Ac, Bs, idx = [], [], [], []
    for grp in psn._groups:
        S = [psn._state_matrix(i, g, m, grp) for m in range(M)]
        idx.append(np.flatnonzero(psn._grp_sel[grp]))
        T.append(np.stack([x[:4, 4:] for x in S]))
        Ac.append(np.stack([x[4:9, :4] for x in S]))
        Bs.append(np.sum([x[4:9, 4:9] for x in S], axis=0))
    w = psn.W[i] / M
    return np.array(T), w * np.array(Ac), w * np.array(Bs), idx


# How many forked workers build the b-tables.  That build is I*ng independent
# one-time ``_state_matrix`` constructions -- 3.22 s of this solver's ~24 s of
# fixed cost on M16_S2, and pure cache replay afterwards (0.03 s for the same
# tables a second time), which makes it the one fixed-cost phase that forks
# cleanly: 0.58 s at 8 workers (5.5x), flat to 16 because 21 systems is all
# there is to hand out.  Nothing about the answer depends on the split (each
# system is built from its own inputs) and dev/btablewall.py compares the
# forked tables against the serial ones byte for byte.  0 = build in place.
TABLE_WORKERS = int(os.environ.get("OPENPSN_TABLE_WORKERS", "8"))


def fast_sweep_tables(psn, *, streamed=False):
    """Class-collapsed tables for the vectorised sweep core (see
    ``PSN2D._sweep_o``), built in the parent BEFORE the fork so no worker
    ever fills a cache in its COW copy.

    ``_state_matrix`` only depends on (i, g, m, class) where class is the
    (material, shape) group, so per class we stack the M azimuthal blocks
    into one array and let einsum do what used to be M x n_class masked
    matmuls per phase (``_grp_sel`` is a boolean mask, so every one of
    those rescanned all nodes).  Returns, per (i, g):

      T  (nc, M, 4, 5)  raw[n,m,:]     = q[n,:] @ T[c,m,:,:].T
      Ac (nc, M, 5, 4)  o[n,0:5]      = sum_m Ac[c,m] @ J4[n,m,:]
      Bs (nc, 5, 5)     o[n,0:5]     += (sum_m S[4:9,4:9])[c] @ q[n,:]
      idx list of intp node indices per class (the mask, precomputed)

    Ac and Bs carry the polar weight W[i]/M, exactly as the reference core
    multiplies it into o; folding it in here costs nothing per sweep (it is
    a property of the direction system, not of the source).

    The (nodes, M, 9) X and Y temporaries of the reference implementation
    are never formed: o is contracted straight out of J4 and q, which drops
    ~25 MB of traffic per system sweep on the C5G7 core.
    """
    if not hasattr(psn, "_Scache"):
        psn._vectorize_setup()
    if not streamed:
        psn._cidx_t = np.ascontiguousarray(psn._cidx.transpose(2, 0, 1))
        psn._esgn_t = np.ascontiguousarray(psn._esgn.transpose(2, 0, 1))
    keys = [(i, g) for i in range(psn.I) for g in range(psn.ng)]
    global _TAB_PSN
    prev, _TAB_PSN = _TAB_PSN, psn      # _fast_tab_one runs on THIS solver,
    try:                                # in-process or in a forked helper
        nw = min(TABLE_WORKERS, len(keys))
        if nw < 2:
            return {ig: _fast_tab_one(ig) for ig in keys}
        try:
            import multiprocessing as mp
            ctx = mp.get_context("fork")
            with ctx.Pool(nw) as pool:
                tabs = pool.map(_fast_tab_one, keys)
        except Exception as exc:        # a machine without fork, or a denied
            # clone: the serial build is the same arithmetic, only slower, so
            # this is not a correctness fallback -- but it costs ~2 s and the
            # run should say so.  stderr, because --quiet and --json own
            # stdout.
            sys.stderr.write(f"psn2d: b-table workers failed ({exc!r}), "
                             f"building in place\n")
            return {ig: _fast_tab_one(ig) for ig in keys}
        return dict(zip(keys, tabs))
    finally:
        _TAB_PSN = prev


def sweep_o_fast(self, i, g, qnode):
    """Vectorised ``PSN2D._sweep_o`` for the shared-Cholesky backend.

    Same math, same (i, g) contract, no (nodes, M, 9) temporaries and
    ~7 einsums instead of ~2 x M x n_class masked matmuls.  Falls back to
    the reference core for any system without prebuilt tables, so a
    partially installed table set can never produce a wrong answer.
    """
    tab = getattr(self, "_fast_tab", {}).get((i, g))
    if tab is None:
        return type(self)._sweep_o(self, i, g, qnode)
    T_all, A_all, Bs_all, idx = tab
    raw = np.empty((self.nodes, self.M, 4))
    for c, ii in enumerate(idx):
        raw[ii] = np.einsum("nj,mkj->nmk", qnode[ii], T_all[c],
                            optimize=True)
    # 64 unbuffered np.add.at calls, one per b-term: 12 ms of a 72-ms job on
    # M16_S2, the most expensive non-solve phase.  Fusing them into ONE
    # np.bincount over the terms concatenated in this same order IS bit-identical
    # (dev/bincount_equiv.py checks it on the real tables -- bincount walks its
    # input element by element, so every output slot sees the same sequence of
    # additions) and it is 2.1-2.8x faster IN ISOLATION.  It is not used: the
    # flattened gather index and its gathered values add 10.2 MiB per system
    # (213 MiB on M16_S2) and ~20 MiB of traffic per job, and a 6-leg
    # interleaved A/B of the sweep wall (dev/sweepms.py, fused on/off alternating)
    # could not separate the two builds at all -- 223/261/335 vs 248/307/329 ms,
    # overlapping ranges -- on a sweep whose 21-way concurrency is what its jobs
    # disagree about.  So it would trade 213 MiB of resident memory, in the one
    # resource this sweep is bound by, for a speed-up that cannot be measured.
    b = np.zeros(self.ncol)
    for (m, lf), (rws, sgns, nds) in self._bterms_for(i, g).items():
        np.add.at(b, rws, sgns * raw[nds, m, lf])
    lu, _ = self._lusys[(i, g)]
    u = lu.solve(b)
    J4 = self._esgn_t * u[self._cidx_t]
    o = np.empty((self.nodes, 5))
    for c, ii in enumerate(idx):
        o[ii] = (np.einsum("nmj,mij->ni", J4[ii], A_all[c], optimize=True)
                 + qnode[ii] @ Bs_all[c].T)
    return self.W[i] / self.M, o        # o is already weighted (see tables)


def sweep_o_streamed(self, i, g, qnode):
    """Assemble and reduce one azimuth at a time to bound sweep workspace."""
    tables, response, src, class_indices = self._fast_tab[(i, g)]
    b = np.zeros(self.ncol)
    raw = np.empty((self.nodes, 4))
    terms = self._bterms_for(i, g)
    for m in range(self.M):
        for c, idx in enumerate(class_indices):
            raw[idx] = qnode[idx] @ tables[c, m].T
        for face in range(4):
            term = terms.get((m, face))
            if term is not None:
                rows, signs, nodes = term
                np.add.at(b, rows, signs * raw[nodes, face])
    del raw
    factor, _ = self._lusys[(i, g)]
    u = factor.solve(b)
    del b
    out = np.empty((self.nodes, 5))
    for c, idx in enumerate(class_indices):
        out[idx] = qnode[idx] @ src[c].T
    for m in range(self.M):
        currents = self._esgn[m] * u[self._cidx[m]]
        for c, idx in enumerate(class_indices):
            out[idx] += currents[:, idx].T @ response[c, m].T
    return self.W[i] / self.M, out


def install_shared_chol(psn, verify=True, ordering=1, factor_cls=None, *,
                        packed_workers=None):
    """Compact assembly + SPD row scaling + xy factor sharing + C++ Cholesky.

    ``factor_cls`` selects the factor implementation
    (``SharedChol`` -> the C++/Eigen bridge, ``RustSharedChol`` from
    ``_load_rust_shared_chol()`` -> the Rust kernel, whose solve is 1.60x and
    whose factor_nnz is bit-equal to Eigen's).  Everything else -- assembly,
    row scaling, mirror-pair sharing, the residual gate -- is identical, so a
    backend switch cannot move keff by more than solve rounding.

    Raises ValueError/RuntimeError when the geometry or the compiled bridge
    is not available — callers (``auto``) catch and fall back.
    """
    import types
    import time

    cls = SharedChol if factor_cls is None else factor_cls

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
            factor = cls(A, ds[i], pairs, ordering=ordering, check=verify)
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
                pool_bytes=getattr(factor, "pool_bytes", factor.factor_nnz * 16),
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
    # vectorised node-response core for the sweep (parent-side tables, so
    # the fork pool inherits them read-only)
    streamed = packed_workers is not None
    psn._fast_tab = fast_sweep_tables(psn, streamed=streamed)
    psn._sweep_o = types.MethodType(
        sweep_o_streamed if streamed else sweep_o_fast, psn)
    # Prefactor every (i, g) system HERE, in the parent, before the fork
    # pool can exist: the forked _sweep_worker reads psn._lusys[(i, g)]
    # DIRECTLY (bypassing _lu), so a missing key in a child would silently
    # factor in the child's COW copy.  With all entries present, no child
    # ever factors — the C++ factor is shared read-only (COW), and any
    # per-system verification failure surfaces in the parent before the
    # sweep starts.
    #
    # lu_opt only reads (topology, row scales, pair blocks) and writes its
    # own ``_lusys[(i, g)]`` key, so the jobs are independent -- but how far
    # they scale with width depends on who else is on the box.  The original
    # Eigen-bridge numbers were taken while co-tenants were loading it (load
    # ~70): 1 worker 76 s, 6 workers 24 s, 21 workers 27 s, per-system median
    # 3.5 -> 5.0 -> 21.6 s -- a wide pool loses when the memory system is
    # already shared, which is why 6 was the shipped default for that kernel
    # and why it stays the default there.  ``OPENPSN_PREFACTOR_WORKERS``
    # overrides either choice.
    # ``PSN_METIS_SEED`` is fixed, so a worker's width changes no factor value
    # -- the legs below are bit-identical across widths, only the wall moves.
    # The Rust kernel takes one system per worker and parallelises nothing
    # inside one factorisation, so wider is better up to ~one worker per
    # system; the Eigen bridge already threads inside a factorisation and was
    # measured to REGRESS past 6 (1/6/21 workers = 76/24/27 s).  Rust widths on
    # M16_S2, phase-isolated in a fresh process each (dev/prefprobe.py), medians
    # of 2-3 samples: 6 workers 10.0 s, 12 workers 8.4 s, 21 workers 8.0 s, at
    # transient RSS 1.85 / 2.40 / 3.26 GiB -- the last 0.4 s is bought with
    # 0.9 GiB, so 12 is the point worth shipping.
    psn._prefactor_workers = (int(os.environ["OPENPSN_PREFACTOR_WORKERS"])
                              if os.environ.get("OPENPSN_PREFACTOR_WORKERS")
                              else min(psn.threads, 5 if streamed else
                                       (6 if factor_cls is None else 12)))
    if streamed:
        psn._prefactor_workers = min(psn._prefactor_workers,
                                     psn.threads // packed_workers)
    psn._prefactor()
    psn._memopt_backend = ("chol-packed-rs" if streamed else
                           "shared-chol" if factor_cls is None else
                           "shared-chol-rust")


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
    # vectorised node-response core for the sweep (parent-side tables, so
    # the fork pool inherits them read-only)
    psn._fast_tab = fast_sweep_tables(psn)
    psn._sweep_o = types.MethodType(sweep_o_fast, psn)
    psn._memopt_backend = f"compact-{permc_spec.lower()}"


def _pool_bytes_measured(psn):
    """Factor-pool bytes ``install_shared_chol`` ACTUALLY allocated, or None.

    ``_tile.pool_estimate_shared_chol`` produces the same figure by factorising
    one (i, g) system and scaling by the system count -- which costs a whole
    factorisation, measured at 4.7 s of the 29 s fixed cost of an M16_S2 run
    (16% of everything before the first sweep), and it is spent to print a
    diagnostic for a backend the caller already named. This sums the measured
    packed allocation sizes or the same 16 B/nnz accounting used by
    ``_check_mem_gate`` for unpacked factors. ``auto`` still estimates its
    pool size before building the factors.
    """
    rec = getattr(psn, "_chol_records", None)
    if not rec:
        return None
    return sum(r["pool_bytes"] for r in rec)


def install_optimized(psn, backend="auto", mem_limit_gb=None):
    """Install a memory-optimized backend.  Returns a report dict.

    auto (memory-aware): use shared-Cholesky when its factor pool fits
    ``mem_limit_gb``; otherwise the **angular Schur** backend when its pool
    estimate fits (exact two-level direction dissection, ~100x lighter than
    the tile seam for the same full-M system — the M192 workhorse);
    otherwise the **tile** backend (Schur tiles, auto-sized from the same
    budget).  If the tile backend is unavailable for the geometry/bridge it
    degrades to compact-mmd -> plain (fail-loud if none survive).

    angschr: force the angular Schur backend (fail-loud certificate when
    the pool exceeds the budget or the geometry is not direction-block-
    diagonal).
    tile: force the tile backend (auto tile size from ``mem_limit_gb``).
    chol/mmd/lu: the explicit legacy backends (unchanged).
    """
    if backend in (None, "off"):
        psn._memopt_backend = "plain"
        return {"backend": "plain", "reason": "opt off"}
    if backend in ("lu", "mmd", "chol", "chol-rs", "chol-packed-rs", "auto", "tile",
                   "angschr"):
        pass
    else:
        raise ValueError(f"unknown backend {backend!r}")

    from . import tile as _tile
    from . import angschr as _angschr
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

    def _try_angschur():
        try:
            return _angschr.install_angschur_backend(
                psn, mem_limit_gb=mem_limit_gb)
        except Exception as e:
            psn._lusys = {}
            psn._syscache = {}
            return e

    if backend == "tile":
        rep = _try_tile()
        if isinstance(rep, Exception):
            if isinstance(rep, _tile.TileInfeasible):
                raise rep            # deterministic: no tile size fits —
            # re-raise with the certificate, do not paper over it
            raise RuntimeError(f"tile backend failed: {rep}")
        return {"backend": "tile", "systems": rep, "mem_limit_gb": mem_limit_gb}

    if backend == "angschr":
        rep = _try_angschur()
        if isinstance(rep, Exception):
            if isinstance(rep, _angschr.AngSchurInfeasible):
                raise rep            # deterministic: pool over budget —
            # re-raise with the certificate
            raise RuntimeError(f"angschr backend failed: {rep}")
        return {"backend": "angschr", "systems": rep,
                "mem_limit_gb": mem_limit_gb}



    if backend in ("chol-rs", "chol-packed-rs"):
        # Explicit request: fail loud rather than degrade, exactly like
        # "chol".  The pool size is read off the factors that were just built
        # (see _pool_bytes_measured), so the memory contract is unchanged and
        # the trial factorisation that used to precede this -- 4.7 s of the
        # M16_S2 fixed cost -- is gone.
        from time import perf_counter

        start = perf_counter()
        cls = _load_rust_shared_chol()
        if backend == "chol-packed-rs":
            from functools import partial
            from .packedchol import PackedSharedChol, load_kernel
            from .solver import _pool_size

            processes = min(psn.threads, _pool_size(psn.I * psn.ng))
            workers = min(2, psn.threads // processes)
            cls = partial(PackedSharedChol, lib=load_kernel(), workers=workers)
            install_shared_chol(psn, factor_cls=cls, packed_workers=workers)
            trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
            if trim is not None:
                trim(0)
        else:
            install_shared_chol(psn, factor_cls=cls)
        if os.environ.get("OPENPSN_SRC_RS", "1") != "0":
            install_fused_source_update(psn)
        est = _pool_bytes_measured(psn)
        if est is not None and est > mem_limit_gb * 2 ** 30:
            raise RuntimeError(
                f"shared-Cholesky pool {est / 2 ** 30:.3f} GB exceeds "
                f"mem_limit_gb={mem_limit_gb}; use backend 'tile' or raise "
                f"the limit")
        rep = {"backend": psn._memopt_backend, "reason": "requested",
               "mem_limit_gb": mem_limit_gb}
        rep["fused_source_update"] = bool(getattr(psn, "_fused_src_update",
                                                  False))
        if backend == "chol-packed-rs":
            rep["setup_s"] = round(perf_counter() - start, 2)
        if est is not None:
            rep["est_pool_gb"] = round(est / 2 ** 30, 3)
            rep["pool_gb_measured"] = True
        return rep

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
                rep = {"backend": psn._memopt_backend, "reason": reason,
                       "mem_limit_gb": mem_limit_gb}
                # prefer what was built over what was predicted (``auto``
                # still needs the prediction to make its choice, so the
                # estimate is not removed there -- it just stops being the
                # reported number once the pool exists)
                measured = _pool_bytes_measured(psn)
                if measured is not None:
                    rep["est_pool_gb"] = round(measured / 2 ** 30, 3)
                    rep["pool_gb_measured"] = True
                elif est_pool_gb is not None:
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
    # auto: pool over budget (or shared-chol unavailable) -> angschr if its
    # estimated pool fits, then tile; tile's TileInfeasible remains the
    # final fail-loud certificate.
    if backend == "auto":
        angschr_est_gb = None
        try:
            a_est, a_info = _angschr.estimate_pool(psn)
            angschr_est_gb = a_est / 2**30
        except Exception as e:
            errors.append(f"angschr estimate: {e}")
        if angschr_est_gb is not None and angschr_est_gb <= mem_limit_gb:
            rep = _try_angschur()
            if not isinstance(rep, Exception):
                out = {"backend": "angschr", "systems": rep,
                       "reason": ("auto -> angschr (shared-chol pool over "
                                  "mem budget, angschr pool fits)"),
                       "mem_limit_gb": mem_limit_gb,
                       "angschr_est_pool_gb": round(angschr_est_gb, 3)}
                if est_pool_gb is not None:
                    out["shared_chol_est_pool_gb"] = round(est_pool_gb, 3)
                return out
            errors.append(f"angschr: {rep}")
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
        if isinstance(rep, _tile.TileInfeasible):
            # Deterministic infeasibility with a veto certificate: every
            # remaining backend (compact-mmd, plain) factors the FULL system
            # and is strictly more expensive than the tile pool.  Degrading
            # would turn a seconds-level certificate into an hours-level
            # OOM — fail loud instead.
            raise rep
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
