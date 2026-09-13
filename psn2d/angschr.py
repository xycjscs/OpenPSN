# -*- coding: utf-8 -*-
"""Angular (direction) Schur dissection — production backend.

Exact two-level factorization of the SPD-scaled face-current system,
partitioned into internal-face columns (I) and boundary mirror-master
columns (B):

  C_II  block-diagonal over angular directions (verified: cross-direction
        nnz = 0 for the C5G7 geometries; a direction couples only to
        itself in the interior)
  S = C_BB − Σ_m C_Bm C_mm⁻¹ C_mB    (M/2 independent small components)
  Solve:  x_B  = S⁻¹ (c_B − Σ_m C_Bm C_mm⁻¹ c_m)
          x_I[m] = C_mm⁻¹ (c_m − C_mB[m] x_B)

The math was validated to machine precision in the c5g7 probes before
productionization (M16 relres 2.6e-14, M96 relres 1.1e-13; per-direction
max 2.3e-11; S long-way identity 1e-14):

  _probe_angblock.py  cross-direction nnz of C_II = 0
  _probe_angv4.py     M16 full-chain residual
  _probe_angv5.py     M96 full-chain residual + localization
  _probe_angi.py      i-invariance (C_II i-dependent -> no factor
                      sharing across (i,g); S i-dependent too)

Memory (per (i,g), M-independent measured block fill 915,700 nnz):
  M16:  16 factors x 11 MB  = 0.17 GB
  M96:  96 factors x 11 MB  = 1.0 GB  + S comps ~0.1 GB
  M192: 192 factors x 11 MB = 2.1 GB  + S comps ~0.3 GB
  vs the tile seam for the SAME full M192 system: 185 GB (~100x).

The direction block is 36,720 x 36,720 (one direction's sub-matrix);
METIS Cholesky fill = 915,700 nnz = 11 MB, M-INDEPENDENT (verified at
M16/M96, identical to the nnz; M192 uses the same measured block).  S
has M/2 components, each of size = boundary-face count (544 for C5G7),
factored by SuperLU (COLAMD; scipy 1.10 has no METIS permc).  The solve
is exact (Schur identity, no iteration, no convergence risk).

Resident memory is accounted in TRUE bytes (factor_nnz reports
resident_bytes // 16 so the solver's pool gate sums actual resident
bytes instead of the generic 16 B/nnz conservative factor — which
would over-count this backend's thin factors by 33% and wrongly veto
M192 on a 62 GB machine).

Integration: drop-in monkey-patch (same contract as the tile and
shared-Cholesky backends: _lu / _bterms_for / solve_i, parent-side
prefactor, factor_nnz consumed by _check_mem_gate).

Thread budget: the mrsolve kernel parallelizes over RHS columns with
OpenMP, whose thread pool is sized from OMP_NUM_THREADS at FIRST
parallel region.  The install therefore sets the limit to
``threads // n_sform`` BEFORE the first factorization, so
n_sform Python workers x (threads // n_sform) OMP threads = threads
(the strict user budget, 2026-09-12).

Requires the compiled C++ Cholesky bridge (same dependency as tile).
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.sparse import csc_matrix, coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

from .memopt import spd_row_scale, _digest
from . import tile as T


# --------------------------------------------------------------------------- #
# partition helpers (verified logic from the probes)
# --------------------------------------------------------------------------- #

def _partition(psn):
    """(order, nB, coldir, ref, nI) with order = [internal | mirror-master].

    ref: (bidx, face, a, b) for each mirror row in f-ascending order,
    bidx = global row index of the mirror equation.
    """
    F, Mm = psn.F, psn.M
    mirror = set()
    for f in range(F):
        if psn.faces[f]['type'] == 'reflect':
            for (a, b) in psn.face_pairs[f]:
                mirror.add(int(psn.colidx[f, a]))
    isB = np.fromiter((c in mirror for c in range(psn.ncol)), bool, psn.ncol)
    nB = int(isB.sum())
    order = np.concatenate([np.where(~isB)[0], np.where(isB)[0]])
    nI = psn.ncol - nB

    coldir = np.full(psn.ncol, -1, np.int64)
    for f in range(F):
        for m in range(Mm):
            c, _ = psn._col_of(f, m)
            coldir[c] = m

    ref = []
    r = 0
    for f in range(F):
        if psn.faces[f]['type'] in ('internal', 'vacuum'):
            r += Mm
        else:
            for (a, b) in psn.face_pairs[f]:
                ref.append((r, f, a, b))
                r += 1
    assert r == psn.ncol, f"partition: {r} rows != ncol {psn.ncol}"
    return order, nB, coldir, ref, nI


# --------------------------------------------------------------------------- #
# one (i, g) system
# --------------------------------------------------------------------------- #

class AngSchurSystem:
    """Angular Schur two-level exact solver for one (i, g) system.

    Pre-computed state: per-direction Cholesky handle + pre-extracted
    C_Bm interface block; per-S-component SuperLU factor.  ``solve()``
    reads only this immutable state (C++ handles are const under
    chol_solve — COW-safe across forked sweep workers, same contract
    as the tile backend).
    """

    def __init__(self, psn, C, d, sgn, ordering=1, verify=True,
                 n_sform=8, tol_prune=1e-13):
        self.n = C.shape[0]
        self.d = d
        self.sgn = sgn
        self.M = psn.M

        order, nB, coldir, ref, nI = _partition(psn)
        self.order = order
        self.nB = nB
        self.nI = nI

        Cc = C[order, :][:, order].tocsc()
        A_II = Cc[:nI, :nI].tocsc()
        A_BI = Cc[nI:, :nI].tocsc()
        A_IB = Cc[:nI, nI:].tocsc()
        A_BB = Cc[nI:, nI:].tocsc()
        ci = order[:nI]
        cdI = coldir[ci]

        # ---- structural gate: C_II must be direction-block-diagonal ----
        A_IIr = A_II.tocsr()
        rows_of = np.repeat(np.arange(nI, dtype=np.int64),
                            np.diff(A_IIr.indptr))
        bad = int(np.sum(cdI[A_IIr.indices] != cdI[rows_of]))
        if bad != 0:
            raise ValueError(
                f"AngSchur: C_II not direction-block-diagonal "
                f"({bad} cross-direction nnz of {A_IIr.nnz}); this "
                f"geometry needs a different backend")

        # local B-space rows per direction
        rows = {m: [] for m in range(psn.M)}
        for (bidx, f, a, b) in ref:
            rows[a].append(bidx - nI)
            rows[b].append(bidx - nI)

        # ---- level 1: direction Cholesky factors ----
        t0 = time.perf_counter()
        self.dirs = []           # (cm, h, C_Bm_csc)
        self._rows = []          # local B-space row index per direction
        self.n_fnnz = 0
        for m in range(psn.M):
            cm = np.where(cdI == m)[0]
            Cmm = A_II[np.ix_(cm, cm)].tocsc()
            Cmm = (Cmm * 1.0).tocsc()      # float64, no dupes
            Cmm.sort_indices()
            _, h = T._factor(Cmm, ordering=ordering)
            fill = int(T._lib().chol_nnz(h))
            self.n_fnnz += fill
            rw = np.array(rows[m], np.int64)
            C_Bm = A_BI[np.ix_(rw, cm)].tocsc()   # solve-phase interface
            self.dirs.append((cm, h, C_Bm))
            self._rows.append(rw)
        self.fac_s = time.perf_counter() - t0
        self.n_cbm_nnz = sum(CB.nnz for (_, _, CB) in self.dirs)

        # ---- level 2: S formation, parallel over directions ----
        # l-row views are needed ONLY here (mrsolve); freed afterwards —
        # the solve phase uses dense chol_solve which needs no views.
        t1 = time.perf_counter()
        S_parts = [None] * psn.M

        def _form_one(m):
            cm, h, C_Bm = self.dirs[m]
            BmT = C_Bm.T.tocsc()                       # C_mB (n_cm x n_rw)
            lrp, lri = T._l_row_cs(h, len(cm))
            CH = 128
            Xd = np.zeros((len(cm), BmT.shape[1]))
            for c0 in range(0, BmT.shape[1], CH):
                cc = BmT[:, c0:c0 + CH].tocsc()
                cap = cc.shape[1] * 60000
                Xd[:, c0:c0 + CH] = T._mr_solve(
                    h, len(cm), cc, lrp, lri, cap).toarray()
            return C_Bm @ Xd

        if n_sform <= 1 or psn.M <= 2:
            for m in range(psn.M):
                S_parts[m] = _form_one(m)
        else:
            nw = min(n_sform, psn.M)
            with ThreadPoolExecutor(max_workers=nw) as pool:
                futs = {pool.submit(_form_one, m): m for m in range(psn.M)}
                for fut in futs:
                    S_parts[futs[fut]] = fut.result()

        # S = C_BB - sum_m C_Bm C_mm^-1 C_mB  (dense blocks per direction)
        data_l, row_l, col_l = [], [], []
        for m in range(psn.M):
            rw = np.array(rows[m], np.int64)
            Sm = S_parts[m]
            nz = Sm != 0
            ri, cj = np.nonzero(nz)
            data_l.append(-Sm[ri, cj])
            row_l.append(rw[ri])
            col_l.append(rw[cj])
        S = coo_matrix((np.concatenate(data_l),
                        (np.concatenate(row_l), np.concatenate(col_l))),
                       shape=(nB, nB))
        br, bc, bv = *A_BB.nonzero(), A_BB.data
        S = S + coo_matrix((bv, (br, bc)), shape=(nB, nB))
        S = S.tocsr()
        if tol_prune > 0:
            rn = np.repeat(np.arange(S.shape[0], dtype=np.int64),
                           np.diff(S.indptr))
            keep = np.abs(S.data) > tol_prune
            S = coo_matrix((S.data[keep], (rn[keep], S.indices[keep])),
                           shape=S.shape).tocsr()
        S.eliminate_zeros()
        self.sform_s = time.perf_counter() - t1

        e = float(abs(S - S.T).max() / abs(S).max()) if S.nnz else 0.0
        if e > 1e-10:
            raise ValueError(f"AngSchur: S not symmetric: {e:.2e}")
        self.S_nnz = S.nnz

        # ---- factor the S components (SuperLU COLAMD; scipy 1.10) ----
        t2 = time.perf_counter()
        ncomp, labels = connected_components(S.tocsr().astype(bool))
        self.s_lu = []
        for comp in np.sort(np.unique(labels)):
            idx = np.where(labels == comp)[0]
            Sc = S[np.ix_(idx, idx)].tocsc()
            lu = splu(Sc, permc_spec="COLAMD")
            self.s_lu.append((idx, lu))
            self.n_fnnz += lu.nnz
        self.sfac_s = time.perf_counter() - t2
        self.ncomp = int(ncomp)
        del S                      # splu owns the factors now

        self.A_IB = A_IB.tocsr()   # tiny (mirror rows x internal cols)
        del Cc, A_II, A_BI, A_IB, A_BB

        # ---- exactness gate: full-system residual ----
        if verify:
            rng = np.random.default_rng(1729)
            b = rng.standard_normal(self.n)
            x = self.solve(b)
            rel = float(np.linalg.norm(C @ x - sgn * d * b)
                        / np.linalg.norm(sgn * d * b))
            if not np.isfinite(rel) or rel > 1e-9:
                raise ValueError(f"AngSchur verify: relres={rel:.2e}")
            self.verify_rel = rel
        else:
            self.verify_rel = None

    @property
    def factor_nnz(self):
        """TRUE resident bytes / 16 — so the solver's pool gate
        (sum factor_nnz * 16) equals the actual resident factor pool.
        The generic 16 B/nnz would over-count thin Cholesky factors
        (12 B real) by 33% and wrongly veto M192 on a 62 GB machine."""
        return self.resident_bytes() // 16

    def resident_bytes(self):
        # direction factors (values+indices, 12 B/nnz, L views freed)
        # + Cholesky perm/pinv (2 x 4 B per unknown, per direction)
        # + S-component SuperLU factors (12 B/nnz)
        # + interface blocks C_Bm (solve phase) + A_IB + order arrays
        return (self.n_fnnz * 12 + self.nI * 8 + self.n_cbm_nnz * 12
                + self.A_IB.nnz * 12 + self.n * 8 + self.nB * 8)

    def solve(self, b):
        """Solve A x = b (original unscaled coordinates), exact."""
        Mm = self.M
        c = self.sgn * self.d * b
        cr = c[self.order]
        cI = cr[:self.nI]
        cB = cr[self.nI:]

        # phase 1: rhs_B = c_B - sum_m C_Bm C_mm^-1 c_m
        rhs_B = cB.copy()
        for m in range(Mm):
            cm, h, C_Bm = self.dirs[m]
            y = T._dense_solve(h, len(cm), cI[cm])
            # C_Bm is (n_rw x n_cm): rhs_B[rw] -= C_Bm @ y
            rhs_B[self._rows[m]] -= np.asarray(C_Bm @ y).ravel()

        # phase 2: x_B per S component
        xB = np.zeros(self.nB)
        for idx, lu in self.s_lu:
            xB[idx] = lu.solve(rhs_B[idx])

        # phase 3: x_I per direction
        Ax = self.A_IB @ xB
        xI = np.zeros(self.nI)
        for (cm, h, _cbm) in self.dirs:
            xI[cm] = T._dense_solve(h, len(cm), cI[cm] - Ax[cm])

        xr = np.empty(self.n)
        xr[:self.nI] = xI
        xr[self.nI:] = xB
        x = np.empty(self.n)
        x[self.order] = xr
        return x

    def close(self):
        for (_, h, _) in self.dirs:
            T._lib().chol_free(h)
        self.dirs = []
        self.s_lu = []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# infeasibility certificate
# --------------------------------------------------------------------------- #

class AngSchurInfeasible(MemoryError):
    """Deterministic infeasibility with a certificate in the message.
    Callers must NOT degrade to a heavier backend (a full-system factor
    is strictly more expensive than the angular pool)."""
    pass


# --------------------------------------------------------------------------- #
# install-time pool estimate (one direction block factor, M-independent)
# --------------------------------------------------------------------------- #

def estimate_pool(psn, ordering=1):
    """Resident-pool byte estimate WITHOUT building the pool: factor ONE
    direction block (fill is M-independent, verified) and scale.  True
    12 B/nnz accounting, same as resident_bytes().  Returns (est, info)."""
    from .memopt import install_compact_build
    if not hasattr(psn, "_compact_build"):
        install_compact_build(psn)
    A, _ = psn._compact_build(0, 0)
    d = spd_row_scale(psn, 0)
    B = A.multiply(d[:, None]).tocsc()
    sgn = -1.0 if B.diagonal()[B.diagonal() != 0].min() < 0 else 1.0
    C = (sgn * B).tocsc()
    del A, B

    order, nB, coldir, ref, nI = _partition(psn)
    Cc = C[order, :][:, order].tocsc()
    A_II = Cc[:nI, :nI].tocsc()
    cdI = coldir[order[:nI]]

    cm = np.where(cdI == 0)[0]
    Cmm = A_II[np.ix_(cm, cm)].tocsc()
    Cmm = (Cmm * 1.0).tocsc()
    Cmm.sort_indices()
    _, h = T._factor(Cmm, ordering=ordering)
    per_block = int(T._lib().chol_nnz(h))
    T._lib().chol_free(h)
    del Cmm

    # S components: M/2 of size = faces per pair, ~50% density (measured
    # 37.5% at M16, 58% at M96); splu fill ~ n^2 for near-dense.
    faces = nB // (psn.M // 2)
    s_nnz_est = (psn.M // 2) * faces * faces
    # interface blocks: sum_m C_Bm.nnz == A_BI.nnz exactly (each internal
    # column belongs to exactly one direction); A_IB = A_BI.T (symmetry)
    abii_nnz = int((Cc[nI:, :nI]).nnz)
    per_system = (psn.M * per_block * 12            # direction factors
                  + s_nnz_est * 12                  # S SuperLU factors
                  + 2 * abii_nnz * 12               # C_Bm + A_IB interfaces
                  + nB * 24 + nI * 8)               # order/index arrays
    nsys = psn.I * psn.ng
    est = nsys * per_system
    del Cc, A_II, C

    return est, {
        "per_block_fill": int(per_block),
        "per_block_MB": per_block * 12 / 2**20,
        "nB": int(nB), "nI": int(nI), "nsys": int(nsys),
        "faces_per_component": int(faces),
        "per_system_GB": per_system / 2**30,
        "pool_GB": est / 2**30,
    }


# --------------------------------------------------------------------------- #
# drop-in install (same contract as the tile backend)
# --------------------------------------------------------------------------- #

def install_angschur_backend(psn, mem_limit_gb=None, ordering=1,
                             verify=True, n_sform=8):
    """Install the angular Schur backend (monkey-patches _lu /
    _bterms_for / solve_i; prefactors every (i, g) in the parent before
    any fork pool exists).

    Memory discipline (fail-loud, no silent degradation):
      * install-time estimate (one direction block) vetoes at > 2x the
        whole-pool budget — a 2x margin absorbs the estimate's coarse
        S-term;
      * each real system is gated on TRUE resident bytes vs the
        per-system share (10% headroom), matching _check_mem_gate.
    """
    import types

    if mem_limit_gb is None:
        from . import runtime
        mem_limit_gb = runtime.DEFAULT_MEM_LIMIT_GB
    total_budget = float(mem_limit_gb) * 2**30
    nsys = psn.I * psn.ng
    per_system = total_budget / max(1, nsys) * 0.9

    from .memopt import install_compact_build
    if not hasattr(psn, "_compact_build"):
        install_compact_build(psn)
    psn._bterms_by_i = dict(psn._compact_bterms)
    psn._rhs_digests_by_i = {i: _digest(psn._bterms_by_i[i])
                             for i in range(psn.I)}
    psn._rhs_identity_checks = 0
    psn._lusys = {}
    psn._syscache = {}
    ds = {i: spd_row_scale(psn, i) for i in range(psn.I)}

    est_bytes, est_info = estimate_pool(psn, ordering)
    if est_bytes > total_budget * 2:
        raise AngSchurInfeasible(
            f"angschr: pool est. {est_bytes / 2**30:.1f} GB > 2x budget "
            f"{mem_limit_gb:.1f} GB ({est_info['per_system_GB']:.2f} GB "
            f"per system x {nsys} systems, "
            f"{est_info['per_block_MB']:.0f} MB per direction factor); "
            f"raise mem_limit_gb or reduce M")

    # OMP thread pool for the mrsolve kernel: sized from the env at the
    # FIRST parallel region, so set it now (before any factorization).
    from . import runtime
    if n_sform > 1:
        runtime.set_thread_limit(max(1, psn.threads // n_sform))

    reports = []
    verdicts = []

    def lu_opt(self, i, g):
        key = (i, g)
        if key not in self._lusys:
            t0 = time.perf_counter()
            A, _bt = self._compact_build(i, g)
            del _bt
            d = ds[i]
            B = A.multiply(d[:, None]).tocsc()
            diag0 = B.diagonal()
            sgn = -1.0 if diag0[diag0 != 0].min() < 0 else 1.0
            C = (sgn * B).tocsc()
            del A, B
            try:
                sys_obj = AngSchurSystem(
                    self, C, d, sgn, ordering=ordering,
                    verify=verify, n_sform=n_sform)
            finally:
                del C
            if sys_obj.resident_bytes() > per_system:
                sys_obj.close()
                msg = (f"angschr: (i={i}, g={g}) resident "
                       f"{sys_obj.resident_bytes() / 2**30:.2f} GB > "
                       f"per-system budget {per_system / 2**30:.2f} GB")
                verdicts.append((i, g, "gate", msg))
                raise MemoryError(msg + f"; certificate: {verdicts}")
            self._lusys[key] = (sys_obj, None)
            reports.append(dict(
                i=i, g=g, M=self.M, nI=sys_obj.nI, nB=sys_obj.nB,
                ncomp=sys_obj.ncomp, S_nnz=sys_obj.S_nnz,
                resident_GB=round(sys_obj.resident_bytes() / 2**30, 3),
                fac_s=round(sys_obj.fac_s, 1),
                sform_s=round(sys_obj.sform_s, 1),
                sfac_s=round(sys_obj.sfac_s, 1),
                total_s=round(time.perf_counter() - t0, 1),
                verify_rel=sys_obj.verify_rel))
        return self._lusys[key]

    def bterms_opt(self, i, g):
        self._lu(i, g)
        return self._bterms_by_i[i]

    def solve_i_opt(self, i, g, qnode):
        sys_obj, _ = self._lu(i, g)
        raw = np.empty((self.nodes, self.M, 4))
        for m in range(self.M):
            for grp in self._groups:
                S = self._state_matrix(i, g, m, grp)
                sel = self._grp_sel[grp]
                raw[sel, m, :] = qnode[sel] @ S[:4, 4:].T
        b = np.zeros(self.ncol)
        for (m, lf), (rws, sgns, nds) in self._bterms_for(i, g).items():
            np.add.at(b, rws, sgns * raw[nds, m, lf])
        return sys_obj.solve(b)

    psn._lu = types.MethodType(lu_opt, psn)
    psn._bterms_for = types.MethodType(bterms_opt, psn)
    psn.solve_i = types.MethodType(solve_i_opt, psn)

    # prefactor every (i, g) in the parent before any fork pool exists
    for i in range(psn.I):
        for g in range(psn.ng):
            psn._lu(i, g)

    # restore the parent's full budget for the outer iteration; the mrsolve
    # OMP pool was already sized during S formation (no further mrsolve
    # calls), and each forked sweep worker pins itself to 1 thread.
    if n_sform > 1:
        runtime.set_thread_limit(psn.threads)

    psn._memopt_backend = "angschr"
    return reports


def pool_estimate_angschur(psn, ordering=1):
    """For the auto dispatcher: (est_bytes, info) without building."""
    return estimate_pool(psn, ordering)
