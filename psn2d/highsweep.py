# -*- coding: utf-8 -*-
"""Inner-iteration node sweep (Gauss-Seidel, row/column alternating,
行列交替扫描) for the A>0 face system.
Added 2026-09-14.

INDEPENDENT RE-DERIVATION (checked row-by-row against
highorder.BlockAngleSystem in tools_ho/check_gs_rows.py):

Global (i, g) face system, row structure (BlockAngleSystem._assemble):
  internal face f (nodes na='a' side, nb='b' side), mode m, slot a:
      row = +1 * term(na, m, fa, a) + (-1) * term(nb, m, fb, a)
  reflect face f, pair (ma, mb), slot a, node n (the 'a' node):
      row = +1 * term(n, ma, lf, a) + (-parity[a]) * term(n, mb, lf, a)
  vacuum face f, mode m, slot a, node n:
      row = sum_b B[m][a,b] * term(n, m, lf, a, slot b)
            + ( -outward ) * (col (f, m, a))
  with  term(n, m, lf, a, scale) contributing
      scale * L[grp, m, lf, a, :4K] * signs[m, n] (one K-vector per
      local face) to n's 4 local-face columns and
      -scale * L[grp, m, lf, a, 4K:5K] to the source moments of node n;
      signs[m, n, lf] = outward * es * mode, with (es, mode) from the
      reflect elimination exactly as in BlockAngleSystem.__init__
      (es = elim_sign, mode = parity when colidx < 0, else +1/1).
  Vacuum rows DO carry source coupling (the -scale*L[4K:5K] part,
  same as every other row) plus the -outward closing term.

Node restriction.  For node n let x_n be the unknowns of n's four
local faces (internal/vacuum lf: M modes x K slots; reflect lf: M/2
master pairs x K slots, slave modes eliminated into the master column
with the SAME column and sign convention as the global matrix).  A
global row whose face belongs to n couples to x_n and, for internal
faces, to the local face unknowns of exactly ONE face-neighbour
(5-point stencil).  Writing sigma = +1 if n is the 'a' side of the
internal face else -1, the per-node local system is

      A_n x_n = Src_n q_n
                + sum_lf internal [  -sigma * Lnb[m, fb, a, :4K]
                                        * sgn_nb (3 non-shared faces)
                                        @ u_nb
                                  +   sigma * Lnb[m, fb, a, 4K:5K]
                                        @ q_nb ]

with the SHARED face column (colidx[f, m], common to both sides) in the
DIAGONAL block:
      A_n[row, shared col] = sigma * L_n[m, lf, a] * sgn_n
                             - sigma * Lnb[m, fb, a] * sown_nb
exactly as the two term() calls of the global row both write that one
column.  Every coefficient is copied verbatim from the same frozen
kernels (L), sign vectors (sg) and B blocks the global assembly uses
-- NO re-derived algebra.  Row/column Gauss-Seidel over the nodes is
therefore a legitimate inner iteration of the IDENTICAL discrete
system.

Two structural facts (used for the block form):
  * face-neighbour nodes ALWAYS have different local face masks (their
    sigma on the shared face is +1 vs -1), so nodes inside one
    micro-class (grp, face-mask, neighbour-groups) never share a face
    -> a whole class can be updated as one block, mathematically
    identical to updating its nodes one by one;
  * diagonal-block factors are shared by micro-class, so total factor
    memory is KB-scale per (i, g) -- versus the 21 x multi-GB resident
    SuperLU pool of the direct path (M8 assembly peak 54 GB, core M2
    ~187 GB).  This unlocks M >= 12, S >= 2 assemblies and the 1/4
    core.

Sweep pattern (行列交替): odd passes walk the micro-classes in
row-major centroid order (row sweep), even passes in column-major
centroid order (column sweep), alternating every pass.

Verified scope: same as highorder.py (axis-aligned conforming
rectangular nodes, uniform A, half-plane even M, isotropic sources,
reflect + vacuum boundaries only).
"""
import time

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .highorder import basis_gauss, local_directional_response, local_response
from .node import node_params_generic

# fixed outward signs of local faces (fxp, fxm, fyp, fym)
_SOWN = np.array([1.0, -1.0, 1.0, -1.0])
# neighbour's local face of the same global face, per own local face
_FB = (1, 0, 3, 2)


class HoNodeLocal:
    """Node-local factor table + GS sweep for one (i, g) of one A."""

    def __init__(self, psn, i, g, A, branch="directional", gs_tol=1e-12,
                 max_gs=500, omega=1.0, verbose=False):
        if branch not in ("directional", "original"):
            raise ValueError("branch must be directional or original")
        self.psn, self.i, self.g, self.A = psn, i, g, A
        self.K, self.branch = A + 1, branch
        self.gs_tol, self.max_gs, self.omega = gs_tol, max_gs, omega
        self.verbose = verbose
        self.kernel = (local_directional_response
                       if branch == "directional" else local_response)
        p = psn
        if not hasattr(p, "_groups"):
            p._vectorize_setup()
        self.M = p.M
        self.parity = (-1.0) ** np.arange(self.K)
        if not hasattr(p, "_ho_face_neighbor"):
            tab = np.zeros((p.nodes, 4, 2), dtype=np.int64)
            tab[:, :, 0] = -1
            for f, face in enumerate(p.faces):
                if face["type"] != "internal":
                    continue
                na, nb = face["a"], face["b"]
                la = p._local_face(na, f)
                lb = p._local_face(nb, f)
                tab[na, la, 0], tab[na, la, 1] = nb, 1
                tab[nb, lb, 0], tab[nb, lb, 1] = na, -1
            p._ho_face_neighbor = tab
        self._build_tables()
        self.u = None
        self._cls_cache = {}
        self._mc_cache = {}
        self._pmc_cache = {}

    # ------------------------------------------------------------------ #
    def _build_tables(self):
        p, M, K = self.psn, self.M, self.K
        # frozen kernel rows per (grp, m, lf): K rows x (4K+5); and V
        L = np.zeros((len(p._groups), M, 4, K, 4 * K + 5))
        Vm = np.zeros((len(p._groups), M, 5, 4 * K + 5))
        for gi, grp in enumerate(p._groups):
            n0 = p._node_of_grp[grp]
            for m in range(M):
                pr, _ = p._pr_alpha_rect(self.i, self.g, m,
                                         p.mat_of_node[n0], n0)
                Ll, Vl, _ = self.kernel(pr, m * p.dphi, (m + 1) * p.dphi,
                                        self.A)
                for lf in range(4):
                    L[gi, m, lf] = Ll[lf * K:(lf + 1) * K]
                Vm[gi, m] = Vl
        self.L, self.Vm = L, Vm
        # vacuum B blocks (axis, m): K x K
        self.Bmat = None
        if any(f["type"] == "vacuum" for f in p.faces):
            if p.generic:
                kappa = node_params_generic(1.0, 1.0, p.theta[self.i],
                                            p.dtheta)["alpha_polar"]
            else:
                kappa = float(p.mu[self.i])
            z, qw, Vg = basis_gauss(K - 1)
            blocks = {}
            for axis in ("x", "y"):
                mats = []
                for m in range(M):
                    phi = (m + 0.5 + 0.5 * z) * (np.pi / M)
                    normal = np.abs(np.cos(phi) if axis == "x"
                                    else np.sin(phi))
                    mats.append(kappa * (Vg.T @ ((qw * normal)[:, None] * Vg)))
                blocks[axis] = np.stack(mats)
            self.Bmat = blocks
        # per-node face info
        self.nf = []
        self.node_grp_idx = []
        for n in range(p.nodes):
            row = []
            for lf in range(4):
                f = p.node_fmap[n][lf][0]
                assert f >= 0
                t = p.faces[f]["type"]
                if t == "internal":
                    nb, fb = p._ho_face_neighbor[n, lf]
                    sigma = 1.0 if p.faces[f]["a"] == n else -1.0
                    row.append(("i", int(nb), int(fb), sigma))
                elif t == "reflect":
                    row.append(("r", tuple(p.face_pairs[f])))
                else:
                    row.append(("v", p.faces[f]["axis"]))
            self.nf.append(row)
            self.node_grp_idx.append(p._groups.index(
                (p.mat_of_node[n], p.shape_of_node[n])))
        # global column index + sign vectors (same as BlockAngleSystem)
        sg = np.ones((p.nodes, M, 4, K))
        colg = np.zeros((p.nodes, 4, M), dtype=np.int64)
        for n in range(p.nodes):
            for lf in range(4):
                f = p.node_fmap[n][lf][0]
                s_own = p.node_fmap[n][lf][1]
                for m in range(M):
                    c, es = p._col_of(f, m)
                    colg[n, lf, m] = c
                    mode = self.parity if p.colidx[f, m] < 0 else 1.0
                    sg[n, m, lf] = s_own * es * mode
        self.sg, self.colg = sg, colg
        # sign-pattern key per node: nodes in one micro-class must share
        # the SAME sign vectors (reflect elimination signs differ between
        # physical boundary faces, so the face-mask alone is not enough).
        self.sgkey = []
        for n in range(p.nodes):
            self.sgkey.append(
                tuple(np.round(self.sg[n], 10).ravel().tolist()))
        # pair positions for BOTH master and slave (slaves eliminate
        # into the master column / local block)
        self.pair_pos = {}
        for f, face in enumerate(p.faces):
            if face["type"] == "reflect":
                for pos, (a, b) in enumerate(p.face_pairs[f]):
                    self.pair_pos[(f, a)] = pos
                    self.pair_pos[(f, b)] = pos
        # local layout: per node, per lf: K-block offsets (rows == cols,
        # the local system is square)
        self.blockoff = np.zeros((p.nodes, 4), dtype=np.int64)
        self.d = np.zeros(p.nodes, dtype=np.int64)
        for n in range(p.nodes):
            off = 0
            for lf in range(4):
                t = self.nf[n][lf][0]
                self.blockoff[n, lf] = off
                off += (len(self.nf[n][lf][1]) if t == "r" else M)
            self.d[n] = off * K
        # micro-classes: (grp, face-mask, neighbour-group tuple, parity).
        # The red-black (checkerboard) parity is part of the class key:
        # a face ALWAYS connects opposite-parity nodes, so nodes within
        # one micro-class never share a face -> the batch update is a
        # genuine block Gauss-Seidel (red-black GS), never a divergent
        # block-Jacobi.
        self.mc_of = []
        for n in range(p.nodes):
            mask = tuple(self._mask_entry(n, lf) for lf in range(4))
            nbt = tuple(self.node_grp_idx[self.nf[n][lf][1]]
                        if self.nf[n][lf][0] == "i" else -1
                        for lf in range(4))
            parity = (int(p.j_idx[n]) + int(p.i_idx[n])) % 2
            self.mc_of.append((self.node_grp_idx[n], mask, nbt, parity,
                               self.sgkey[n]))
        self.mc_nodes = {}
        for n in range(p.nodes):
            self.mc_nodes.setdefault(self.mc_of[n], []).append(n)
        # per-face global row start (assembly row order)
        face_row_start = []
        r = 0
        for f, face in enumerate(p.faces):
            face_row_start.append(r)
            if face["type"] == "internal":
                r += p.M * K
            elif face["type"] == "reflect":
                r += len(p.face_pairs[f]) * K
            else:
                r += p.M * K
        assert r == p.ncol * K
        self.face_row_start = face_row_start

    def _mask_entry(self, n, lf):
        info = self.nf[n][lf]
        if info[0] == "i":
            return ("i", info[3])
        if info[0] == "r":
            return ("r",)
        return ("v", info[1])

    def _local_colpos(self, n, lf, mode):
        """K-block position of (local face lf, azimuth mode) in node n's
        local layout.  Reflect faces: BOTH master and slave modes map
        to their pair's single K-block.  Internal/vacuum: mode number."""
        t = self.nf[n][lf][0]
        f = self.psn.node_fmap[n][lf][0]
        if t == "r":
            pos = self.pair_pos[(f, mode)]
        else:
            pos = mode
        return self.blockoff[n, lf] + pos

    # ------------------------------------------------------------------ #
    def _class_system(self, grp, mask, sgk):
        """(A (d x d) WITHOUT the shared-face neighbour diagonal, lu of
        A, SRC (d x 5), rep) for a (grp, face-mask, sign-pattern) class.
        Coefficients verbatim from the term()/vacuum_row() expressions
        of the global assembly."""
        key = (grp, tuple(mask), sgk)
        if key in self._cls_cache:
            return self._cls_cache[key]
        p, M, K = self.psn, self.M, self.K
        rep = None
        for n in range(p.nodes):
            if self.node_grp_idx[n] == grp and \
                    tuple(self._mask_entry(n, lf) for lf in range(4)) \
                    == tuple(mask) and self.sgkey[n] == sgk:
                rep = n
                break
        assert rep is not None, "no representative node for class"
        Ln = self.L[grp]
        sg = self.sg[rep]
        d = int(self.d[rep])
        A = np.zeros((d, d))
        SRC = np.zeros((d, 5))
        for lf in range(4):
            info = self.nf[rep][lf]
            t = info[0]
            nrow = (len(info[1]) if t == "r" else M) * K
            r0 = int(self.blockoff[rep, lf]) * K
            for r in range(nrow):
                row = r0 + r
                a = r % K
                if t == "i":
                    m = r // K
                    s = info[3]
                    for lf2 in range(4):
                        cb = self._local_colpos(rep, lf2, m) * K
                        A[row, cb:cb + K] += s * Ln[m, lf, a,
                                                    lf2 * K:(lf2 + 1) * K] \
                            * sg[m, lf2]
                    SRC[row] = -s * Ln[m, lf, a, 4 * K:4 * K + 5]
                elif t == "r":
                    ma = info[1][r // K][0]
                    mb = info[1][r // K][1]
                    for (mx, sx) in ((ma, 1.0), (mb, -self.parity[a])):
                        for lf2 in range(4):
                            cb = self._local_colpos(rep, lf2, mx) * K
                            A[row, cb:cb + K] += sx \
                                * Ln[mx, lf, a, lf2 * K:(lf2 + 1) * K] \
                                * sg[mx, lf2]
                        SRC[row] += -sx * Ln[mx, lf, a, 4 * K:4 * K + 5]
                else:  # vacuum
                    axis = info[1]
                    m = r // K
                    Bm = self.Bmat[axis][m]
                    for b in range(K):
                        sc = Bm[a, b]
                        if sc == 0.0:
                            continue
                        for lf2 in range(4):
                            cb = self._local_colpos(rep, lf2, m) * K
                            A[row, cb:cb + K] += sc \
                                * Ln[m, lf, b, lf2 * K:(lf2 + 1) * K] \
                                * sg[m, lf2]
                        SRC[row] += -sc * Ln[m, lf, b, 4 * K:4 * K + 5]
                    A[row, self._local_colpos(rep, lf, m) * K + a] \
                        += -p.node_fmap[rep][lf][1]
        out = dict(A=A, fac=lu_factor(A), SRC=SRC, rep=rep, d=d)
        self._cls_cache[key] = out
        return out

    # ------------------------------------------------------------------ #
    def _microclass(self, key):
        """Batched sweep data for one micro-class (shared A/SRC/N)."""
        ck = ("mc",) + key
        if ck in self._mc_cache:
            return self._mc_cache[ck]
        p, M, K = self.psn, self.M, self.K
        grp, mask, nbt, _parity, _sgk = key
        nlist = np.array(self.mc_nodes[key], dtype=np.int64)
        Nn = len(nlist)
        cls = self._class_system(grp, mask, _sgk)
        A = cls["A"].copy()
        SRC, rep, d = cls["SRC"], cls["rep"], int(self.d[cls["rep"]])
        # --- shared-face neighbour diagonal: the global internal-face
        # row carries BOTH sides' L rows on the shared face column
        # (colidx[f, m] is one global column owned by x_n as n's local
        # face lf); the class A above has only n's own side there, add
        # the neighbour side  -sigma * Lnb[m, fb, a] * sown_nb
        # (internal face -> colidx >= 0, so the neighbour sign on the
        # shared face is its fixed outward sign).
        for lf in range(4):
            if self.nf[rep][lf][0] != "i":
                continue
            fb = _FB[lf]
            sigma = self.nf[rep][lf][3]
            Lnb = self.L[nbt[lf]]
            sown = _SOWN[fb]
            r0 = int(self.blockoff[rep, lf]) * K
            cblk = int(self.blockoff[rep, lf]) * K
            for m in range(M):
                for a in range(K):
                    row = r0 + m * K + a
                    A[row, cblk + m * K:cblk + (m + 1) * K] += \
                        -sigma * Lnb[m, fb, a, fb * K:(fb + 1) * K] * sown
        fac = lu_factor(A)
        # writeback: local unknown (blk, slot) -> global flat column
        wb = np.zeros((Nn, d), dtype=np.int64)
        for ii, n in enumerate(nlist):
            for lf in range(4):
                t = self.nf[n][lf][0]
                f = p.node_fmap[n][lf][0]
                if t == "r":
                    for pos, (ma, mb) in enumerate(self.nf[n][lf][1]):
                        blk = self.blockoff[n, lf] + pos
                        wb[ii, blk * K:(blk + 1) * K] = \
                            self.colg[n, lf, ma] * K + np.arange(K)
                else:
                    for m in range(M):
                        blk = self.blockoff[n, lf] + m
                        wb[ii, blk * K:(blk + 1) * K] = \
                            self.colg[n, lf, m] * K + np.arange(K)
        # neighbour data per internal lf, split by neighbour group
        nb = {}
        for lf in range(4):
            if self.nf[rep][lf][0] != "i":
                continue
            fb = _FB[lf]
            sigma = self.nf[rep][lf][3]
            nb_groups = {}
            for ii, n in enumerate(nlist):
                nbn = self.nf[n][lf][1]
                nb_groups.setdefault(self.node_grp_idx[nbn], []).append(ii)
            parts = []
            for gi, idxs in sorted(nb_groups.items()):
                idxs = np.array(idxs, dtype=np.int64)
                # the ACTUAL neighbour node ids (nlist[idxs] would be the
                # class's own nodes — a bug found via dbg_terms.py)
                nodes_nb = np.array(
                    [self.nf[nlist[ii]][lf][1] for ii in idxs],
                    dtype=np.int64)
                Lnb = self.L[gi]
                # Global internal-face row = term(na, +1) + term(nb, -1):
                #   L_n sgn_n u_n - Lnb sgn_nb u_nb = -L_n[4K:] q_n + Lnb[4K:] q_nb
                # (n = 'a' side, sigma = +1).  In n's local system
                # (own side: A diagonal = +sigma * L_n * sgn_n,
                #  SRC = -sigma * L_n[4K:]):
                #   shared face column (in A): -sigma * Lnb * sgn_nb[fb]
                #   neighbour u part (RHS, 3 non-shared faces):
                #       -sigma * Lnb * sgn_nb @ u_nb
                #   neighbour q part (RHS):    -sigma * Lnb[4K:5K] @ q_nb
                # (checked for n on either side via the sigma factor;
                #  VERIFIED against the global matrix at the exact
                #  solution in tools_ho/dbg_terms.py: G[nonlocal]@u =
                #  -Lnb*sgn_nb@u, and b = Sm@q - G[nonlocal]@u).
                # The shared face f == fb is zeroed here (it lives in the
                # diagonal A).  sgn_nb below is the neighbour's FULL sign
                # vector (sown x elim-sign x mirror mode), exactly the
                # BlockAngleSystem.signs of the neighbour node.
                C0 = sigma * Lnb[:, fb, :, :4 * K].reshape(M, K, 4, K)
                C0[:, :, fb, :] = 0.0
                src5 = np.empty((M, K, 5))
                for m in range(M):
                    src5[m] = Lnb[m, fb, :, 4 * K:4 * K + 5]
                sgn_nb = self.sg[nodes_nb].reshape(len(idxs), M, 4, K)
                gath = np.empty((len(idxs), 4 * M * K), dtype=np.int64)
                for s2, n_nb in enumerate(nodes_nb):
                    for lf2 in range(4):
                        for m in range(M):
                            gath[s2, (lf2 * M + m) * K:
                                 (lf2 * M + m + 1) * K] = \
                                self.colg[n_nb, lf2, m] * K + np.arange(K)
                rowidx = np.add(int(self.blockoff[rep, lf]) * K
                                + np.arange(M)[:, None] * K,
                                np.arange(K)[None, :]).ravel()
                parts.append(dict(idxs=idxs, C0=C0, src5=src5, sigma=sigma,
                                  sgn_nb=sgn_nb, gath=gath,
                                  rowidx=rowidx, nodes_nb=nodes_nb))
            nb[lf] = parts
        # global row indices of the class's local rows (for validation)
        fstart = self.face_row_start
        rowg = np.zeros(d, dtype=np.int64)
        for lf in range(4):
            t = self.nf[rep][lf][0]
            f = p.node_fmap[rep][lf][0]
            off = int(self.blockoff[rep, lf]) * K
            nrow = (len(self.nf[rep][lf][1]) if t == "r" else M) * K
            rowg[off:off + nrow] = fstart[f] + np.arange(nrow)
        # reduce data: per (node, mode m) the 4K J4 gather in canonical
        # face layout [f0|f1|f2|f3] (slave modes eliminate into the
        # master column with the same sign as the global apply)
        J4_idx = np.zeros((Nn, M, 4 * K), dtype=np.int64)
        J4_sgn = np.zeros((Nn, M, 4 * K))
        for ii, n in enumerate(nlist):
            for lf in range(4):
                for m in range(M):
                    c = self.colg[n, lf, m]
                    J4_idx[ii, m, lf * K:(lf + 1) * K] = \
                        c * K + np.arange(K)
                    J4_sgn[ii, m, lf * K:(lf + 1) * K] = self.sg[n, m, lf]
        Vj = self.Vm[grp, :, :, :4 * K].transpose(0, 2, 1).copy()  # (M, 4K, 5)
        Vs = self.Vm[grp, :, :, 4 * K:4 * K + 5].sum(axis=0)  # (5, 5)
        out = dict(nodes=nlist, wb=wb, A=A, fac=fac,
                   SRC=SRC, nb=nb, d=d, grp=grp, rowg=rowg,
                   J4_idx=J4_idx, J4_sgn=J4_sgn,
                   Vj=Vj, Vs=Vs, rep=rep)
        # numba flat pack for _mc_b (lever 3); None when unavailable or
        # disabled via OPSN_NO_NUMBA=1 (A/B isolation).  Built once here
        # (parent, pre-fork) and COW-shared by workers.
        try:
            from . import mckern
            if mckern.HAVE_NUMBA and mckern.ENABLED:
                out["pack"] = mckern.pack_microclass(out)
            else:
                out["pack"] = None
        except Exception:
            out["pack"] = None
        self._mc_cache[ck] = out
        return out

    # ------------------------------------------------------------------ #
    def _mc_order(self):
        """Micro-classes (red-black GS) in row-major / column-major
        centroid order, split by checkerboard parity."""
        p = self.psn
        cen = {}
        for key in self.mc_nodes:
            ns = np.array(self.mc_nodes[key])
            cen[key] = (float(np.mean(p.j_idx[ns])),
                        float(np.mean(p.i_idx[ns])))
        orders = {}
        for parity in (0, 1):
            keys = [k for k in self.mc_nodes if k[3] == parity]
            row_maj = sorted(keys, key=lambda k: (cen[k][0], cen[k][1]))
            col_maj = sorted(keys, key=lambda k: (cen[k][1], cen[k][0]))
            orders[parity] = (row_maj, col_maj)
        return orders

    def _mc_b(self, data, qn):
        """b = Src q + neighbour contributions (u and q).

        Numba path (lever 3): when the microclass carries a pre-packed
        flat form, the b-assembly runs as one prange kernel that
        releases the GIL (T threads set by the worker init).  Falls
        back to the NumPy path when numba is unavailable."""
        pack = data.get("pack")
        if pack is not None:
            from . import mckern
            return mckern.mc_b(pack, self.u, qn)
        u = self.u
        nlist = data["nodes"]
        b = qn[nlist] @ data["SRC"].T
        for lf, parts in data["nb"].items():
            for part in parts:
                idxs = part["idxs"]
                row1d = part["rowidx"].ravel()
                # un: (s, f, m, slot) -> (s, m, f, slot) to align with
                # sgn_nb (s, m, f, slot) and C0 (m, a, f, slot).
                un = u[part["gath"]].reshape(
                    len(idxs), 4, self.M, self.K).transpose(0, 2, 1, 3)
                # prod (n, m, f, s) = sgn_nb * un ; C0 (m, a, f, s)
                prod = part["sgn_nb"] * un
                contrib = np.einsum("mafs,nmfs->nma", part["C0"], prod)
                tmp = b[idxs]
                tmp[:, row1d] += contrib.reshape(len(idxs), -1)
                # neighbour source part: +sigma * (L_nb[4K:5K] @ q_nb)
                # src5 (m, a, q); row (m, a) wants q_nb @ src5[m, a]
                src = np.einsum("nq,maq->nma", qn[part["nodes_nb"]],
                                part["src5"]).reshape(len(idxs), -1)
                tmp[:, row1d] += part["sigma"] * src
                b[idxs] = tmp
        return b

    def _mc_pass(self, data, qn):
        xold = None
        if self.omega != 1.0:
            xold = self.u[data["wb"]].copy()
        b = self._mc_b(data, qn)
        x = lu_solve(data["fac"], b.T).T
        if self.omega != 1.0:
            x = xold + self.omega * (x - xold)
        self.u[data["wb"]] = x
        return x

    def _mc_residual(self, data, qn):
        x = self.u[data["wb"]]
        b = self._mc_b(data, qn)
        r = data["A"] @ x.T - b.T
        denom = np.abs(b).max() + 1e-300
        return float(np.abs(r).max() / denom)

    # ------------------------------------------------------------------ #
    def sweep(self, qn, sweeps=None):
        """GS inner sweep on the (i, g) system.
        Returns (u (ncol, K), iters, relres, out (nodes, 5) W/M-weighted
        5-moment reduce -- same contract as BlockAngleSystem.apply).
        ``sweeps`` overrides max_gs."""
        p = self.psn
        if self.u is None:
            self.u = np.zeros(p.ncol * self.K)
        orders = self._mc_order()
        data_cache = {k: self._microclass(k) for pty in (0, 1)
                      for k in orders[pty][0] + orders[pty][1]}
        mg = self.max_gs if sweeps is None else int(sweeps)
        it = 0
        res = None
        while True:
            it += 1
            # red-black GS: faces only connect opposite parity, so each
            # parity group is a face-independent set; direction
            # alternates row-major / column-major per pass (行列交替)
            direction = it % 2          # 0: column-major, 1: row-major
            for pty in (0, 1):
                seq = orders[pty][direction]
                for key in seq:
                    self._mc_pass(data_cache[key], qn)
            if it % 2 == 0 or it >= mg:
                res = 0.0
                for key in data_cache:
                    res = max(res, self._mc_residual(data_cache[key],
                                                     qn))
                if self.verbose:
                    print(f"    gs {it:3d} relres={res:.2e}")
                if res < self.gs_tol:
                    break
            if it >= mg:
                break
        out = self._reduce(qn)
        return self.u.reshape(p.ncol, self.K), it, res, out

    # ------------------------------------------------------------------ #
    def sweep_gmres(self, qn, rtol=1e-11, maxiter=800, verbose=False):
        """GMRES on the FIXED-SOURCE face system (2026-09-14).

        One full red-black pass (row-major + column-major) is the affine
        map F(u) = G u + c of the iteration; its fixed point
        (I - G) u = c IS the exact solution of the (i,g) face system.
        GMRES is applied to  A v = (I-G) v = v - F(v) + c  with
        b = c, i.e. Krylov over the red-black block iteration itself
        (the traditional red-black block preconditioner, in the cheaper
        matvec-free form).  The operator is fixed (q frozen), so plain
        GMRES is valid -- no FGMRES needed.

        * warm start: x0 = self.u from the previous call (the previous
          outer iteration's solution) -- production behaviour.
        * acceptance: the ORIGINAL residual  ||c - (I-G) u|| / ||c||
          (the face-current residual of the global system), computed
          with one final matvec after the Krylov iteration.

        Returns (u, matvecs, relres, out) -- same contract as sweep()."""
        p = self.psn
        K = self.K
        if self.u is None:
            self.u = np.zeros(p.ncol * K)
        orders = self._mc_order()
        seen = []
        for pty in (0, 1):
            for key in orders[pty][0] + orders[pty][1]:
                if key not in seen:
                    seen.append(key)
        data_cache = {k: self._microclass(k) for k in seen}
        u_work = self.u

        def one_pass():
            for direction in (0, 1):          # 行列交替: col-major + row-major
                for pty in (0, 1):
                    for key in orders[pty][direction]:
                        self._mc_pass(data_cache[key], qn)

        # x0: inherited solution (warm) or zero (cold)
        x0 = u_work.copy()
        # c = F(0)
        u_work[:] = 0.0
        one_pass()
        cvec = u_work.copy()
        u_work[:] = x0

        def matvec(v):
            w = v.copy()
            u_work[:] = v
            one_pass()
            return w - u_work + cvec          # (I - G) v

        from scipy.sparse.linalg import LinearOperator, gmres
        niter = [0]

        def _cb(*args):
            niter[0] += 1

        t0 = time.time()
        op = LinearOperator((u_work.size, u_work.size), matvec,
                            dtype=float)
        u_sol, info = gmres(op, cvec, x0=x0, rtol=rtol, atol=0.0,
                            maxiter=maxiter, callback=_cb,
                            callback_type='pr_norm')
        dt = time.time() - t0
        # ORIGINAL face-system residual:  r = b - A u = F(u) - u
        # (at the exact solution F(u*)=u*, so r=0; note A u* = c, so
        #  matvec(u*) = c and must NOT be used as the residual).
        u_work[:] = u_sol
        F_u = u_work.copy()
        one_pass()
        F_u = u_work                        # F(u_sol)
        r = F_u - u_sol
        res = float(np.abs(r).max() / max(np.abs(cvec).max(), 1e-300))
        u_work[:] = u_sol
        if verbose:
            print(f"    gmres matvecs={niter[0]} info={info} "
                  f"relres={res:.2e} ({dt:.2f}s)")
        out = self._reduce(qn)
        return self.u.reshape(p.ncol, self.K), niter[0], res, out

    # ------------------------------------------------------------------ #
    def _reduce(self, qn):
        """W[i]/M * sum_m (J4_m @ Vm[:4K].T + q @ Vm[4K:].T) per node
        -- identical expression to BlockAngleSystem.apply."""
        p, K = self.psn, self.K
        u = self.u  # flat (ncol*K,)
        out = np.zeros((p.nodes, 5))
        for key in self.mc_nodes:
            data = self._microclass(key)
            nlist = data["nodes"]
            J4 = u[data["J4_idx"]] * data["J4_sgn"]
            acc = np.einsum("nmd, mde -> ne", J4, data["Vj"])
            acc += qn[nlist] @ data["Vs"].T
            out[nlist] = acc
        out *= p.W[self.i] / p.M
        return out

    # ------------------------------------------------------------------ #
    # Angle-pair block GMRES (2026-09-14, "lever 1").
    #
    # The (i,g) face system H is STRICTLY block-diagonal over the M/2
    # azimuth mirror pairs {p, M-1-p} (verified probe dbg_pair_diag.py:
    # off-pair-block max|A| = 0 for every class and microclass, and
    # memopt.mirror_pair_columns partitions ncol).  Each pair is
    # therefore an independent system: its own Krylov GMRES, its own
    # warm start (disjoint global columns), identical contraction.
    # A pair's five-moment contribution is W/M * sum_{m in pair} ...;
    # the driver accumulates over pairs -> exact same flux.
    # ------------------------------------------------------------------ #
    def _pair_cols(self, pp):
        """Flat global indices of mirror pair {pp, M-1-pp} in u.

        Reuses memopt.mirror_pair_columns (probe-verified to partition
        ncol): the pair's master columns c map to flat slice
        c*K .. c*K+K-1 in u."""
        from .memopt import mirror_pair_columns
        p, K = self.psn, self.K
        if not hasattr(self, "_pmc_pcols"):
            self._pmc_pcols = mirror_pair_columns(p)
        cols = np.sort(self._pmc_pcols[pp]).astype(np.int64)
        idx = (cols[:, None] * K + np.arange(K)[None, :]).ravel()
        return idx

    def _pair_sel(self, rep, pp):
        """Local row/col indices of mirror pair {pp, M-1-pp} in the
        class layout (rows == cols).  blockoff is in BLOCK units
        (each block = K indices), so the block at (lf, pos) occupies
        flat indices (blockoff[lf]+pos)*K .. (blockoff[lf]+pos+1)*K-1.
        Reflect faces: the pair's single K-block sits at position pp
        (face_pairs are ordered (0,M-1),(1,M-2),...).  Internal/vacuum:
        the two modes of the pair sit at positions pp and M-1-pp."""
        M, K = self.M, self.K
        ms = (pp, M - 1 - pp)
        sel = []
        for lf in range(4):
            t = self.nf[rep][lf][0]
            off = int(self.blockoff[rep, lf])        # block units
            if t == "r":
                sel += list(range((off + pp) * K, (off + pp + 1) * K))
            else:
                for m2 in ms:
                    sel += list(range((off + m2) * K, (off + m2 + 1) * K))
        return np.array(sorted(sel), dtype=np.int64)

    def _pair_microclass(self, key, pp):
        """Pair-restricted microclass data.  Reuses the full
        _microclass (COW-shared coefficient tables) and extracts the
        {pp, M-1-pp} sub-system: block diagonal A_p (probe-verified
        zero off-blocks), its LU, pair source rows, pair writeback,
        pair-restricted neighbour parts and pair reduce data."""
        ck = ("pmc",) + key + (pp,)
        if ck in self._pmc_cache:
            return self._pmc_cache[ck]
        p, M, K = self.psn, self.M, self.K
        ms = (pp, M - 1 - pp)
        data = self._microclass(key)
        rep = data["rep"]
        sel = self._pair_sel(rep, pp)
        A = data["A"]
        Ap = A[np.ix_(sel, sel)].copy()
        offmask = np.zeros_like(A, bool)
        offmask[sel, :] = True
        offmask[:, sel] = False
        if float(np.abs(A[offmask]).max()) > 0.0:
            raise AssertionError(
                f"pair block {pp} of class {key} is not block-diagonal")
        nlist = data["nodes"]
        Nn = len(nlist)
        # neighbour parts, restricted to pair rows / pair columns.
        # The neighbour gath order must match the full version:
        # (node s, lf2, m) -> here (s, lf2, m') with m' in ms, so the
        # reshape (s, 4, 2, K) aligns with C0[ms] (2, K, 4, K) and
        # sgn_nb[:, ms] (s, 2, 4, K).
        nb = {}
        for lf, parts in data["nb"].items():
            pl = []
            for part in parts:
                idxs = part["idxs"]
                # pair row indices RELATIVE TO the pair layout (sel):
                abs_rows = np.add(int(self.blockoff[rep, lf]) * K
                                  + np.array(ms)[:, None] * K,
                                  np.arange(K)[None, :]).ravel()
                row1d = np.searchsorted(sel, abs_rows)
                # (neighbour node, 4 lf2 x 2 pair modes x K) gather
                gath = np.empty((len(idxs), 8 * K), dtype=np.int64)
                for s2, n_nb in enumerate(part["nodes_nb"]):
                    j = 0
                    for lf2 in range(4):
                        for m2 in ms:
                            c = self.colg[n_nb, lf2, m2]
                            gath[s2, j:j + K] = c * K + np.arange(K)
                            j += K
                pl.append(dict(idxs=idxs,
                               C0=part["C0"][list(ms)],
                               src5=part["src5"][list(ms)],
                               sigma=part["sigma"],
                               sgn_nb=part["sgn_nb"][:, list(ms), :, :],
                               gath=gath, rowidx=row1d,
                               nodes_nb=part["nodes_nb"]))
            nb[lf] = pl
        # writeback / reduce restricted to the pair
        grp = key[0]
        # Vm[grp, m] is (5, 4K+5); select pair modes + current cols,
        # then transpose to (2, 4K, 5) to match the full _reduce layout
        Vj = self.Vm[grp, list(ms), :, :4 * K].transpose(0, 2, 1)
        Vs = self.Vm[grp, list(ms), :, 4 * K:4 * K + 5].sum(axis=0)
        out = dict(nodes=nlist, wb=data["wb"][:, sel],
                   A=Ap, fac=lu_factor(Ap), SRC=data["SRC"][sel],
                   nb=nb, rep=rep, sel=sel, ms=ms,
                   J4_idx=data["J4_idx"][:, list(ms), :],
                   J4_sgn=data["J4_sgn"][:, list(ms), :],
                   Vj=Vj, Vs=Vs)
        # numba pack (lever 3): pair nb parts have the same flat layout
        try:
            from . import mckern
            if mckern.HAVE_NUMBA and mckern.ENABLED:
                out["pack"] = mckern.pack_microclass(out)
            else:
                out["pack"] = None
        except Exception:
            out["pack"] = None
        self._pmc_cache[ck] = out
        return out

    def _mc_b_pair(self, pdata, qn):
        """Pair-restricted b = SRC q + neighbour contributions."""
        pack = pdata.get("pack")
        if pack is not None:
            from . import mckern
            return mckern.mc_b(pack, self.u, qn)
        u = self.u
        nlist = pdata["nodes"]
        b = qn[nlist] @ pdata["SRC"].T
        for lf, parts in pdata["nb"].items():
            for part in parts:
                idxs = part["idxs"]
                row1d = part["rowidx"]
                un = u[part["gath"]].reshape(
                    len(idxs), 4, 2, self.K).transpose(0, 2, 1, 3)
                prod = part["sgn_nb"] * un
                contrib = np.einsum("mafs,nmfs->nma", part["C0"], prod)
                tmp = b[idxs]
                tmp[:, row1d] += contrib.reshape(len(idxs), -1)
                src = np.einsum("nq,maq->nma", qn[part["nodes_nb"]],
                                part["src5"]).reshape(len(idxs), -1)
                tmp[:, row1d] += part["sigma"] * src
                b[idxs] = tmp
        return b

    def _mc_pass_pair(self, pdata, qn):
        b = self._mc_b_pair(pdata, qn)
        x = lu_solve(pdata["fac"], b.T).T
        self.u[pdata["wb"]] = x
        return x

    def _reduce_pair(self, qn, pp):
        """Pair share of the five-moment flux: W/M * sum_{m in pair}.
        Driver accumulates over pp -> identical to _reduce."""
        p = self.psn
        out = np.zeros((p.nodes, 5))
        for key in self.mc_nodes:
            data = self._pair_microclass(key, pp)
            nlist = data["nodes"]
            J4 = self.u[data["J4_idx"]] * data["J4_sgn"]
            acc = np.einsum("nmd, mde -> ne", J4, data["Vj"])
            acc += qn[nlist] @ data["Vs"].T
            out[nlist] = acc
        out *= p.W[self.i] / p.M
        return out

    def sweep_gmres_pair(self, qn, pp, rtol=1e-11, maxiter=800,
                         verbose=False):
        """GMRES on ONE mirror-pair block of the fixed-source (i,g)
        system.  Krylov runs on the LOCAL pair vector (d = ncol_pair*K);
        the matvec scatters into the global u, runs the pair-restricted
        red-black pass, and gathers back.  Same contract as
        sweep_gmres but returns the pair's five-moment share."""
        p, K = self.psn, self.K
        if self.u is None:
            self.u = np.zeros(p.ncol * K)
        orders = self._mc_order()
        seen = []
        for pty in (0, 1):
            for key in orders[pty][0] + orders[pty][1]:
                if key not in seen:
                    seen.append(key)
        data_cache = {k: self._pair_microclass(k, pp) for k in seen}
        u_work = self.u
        # global indices of this pair's unknowns (flat u layout)
        if not hasattr(self, "_pmc_cols"):
            self._pmc_cols = {}
        if pp not in self._pmc_cols:
            self._pmc_cols[pp] = self._pair_cols(pp)
        idx = self._pmc_cols[pp]

        def one_pass():
            for direction in (0, 1):          # 行列交替
                for pty in (0, 1):
                    for key in orders[pty][direction]:
                        self._mc_pass_pair(data_cache[key], qn)

        x0 = u_work[idx].copy()                # warm start: pair only
        u_work[idx] = 0.0
        one_pass()
        cvec = u_work[idx].copy()              # c = F(0), pair rows
        u_work[idx] = x0

        def matvec(v):
            u_work[idx] = v
            one_pass()
            return v - u_work[idx] + cvec      # (I - G) v + c, pair

        from scipy.sparse.linalg import LinearOperator, gmres
        niter = [0]

        def _cb(*args):
            niter[0] += 1

        t0 = time.time()
        op = LinearOperator((len(idx), len(idx)), matvec, dtype=float)
        u_sol, info = gmres(op, cvec, x0=x0, rtol=rtol, atol=0.0,
                            maxiter=maxiter, callback=_cb,
                            callback_type='pr_norm')
        dt = time.time() - t0
        u_work[idx] = u_sol
        one_pass()                             # F(u_sol), pair rows
        r = u_work[idx] - u_sol
        res = float(np.abs(r).max() / max(np.abs(cvec).max(), 1e-300))
        u_work[idx] = u_sol
        if verbose:
            print(f"    gmres pair{pp} matvecs={niter[0]} info={info} "
                  f"relres={res:.2e} ({dt:.2f}s)")
        out = self._reduce_pair(qn, pp)
        return self.u.reshape(p.ncol, self.K), niter[0], res, out


# GS keff driver
# --------------------------------------------------------------------- #
def keff_highorder_gs(psn, A=3, branch="directional", max_outer=12000,
                      outer_tol=1e-10, gs_tol=1e-12, max_gs=500,
                      omega=1.0, verbose=False):
    """Power iteration with the node-GS inner sweep.

    Returns (lam, phi, qim, info) -- same contract as
    highorder.keff_highorder."""
    from .highorder import _validate_A
    A = _validate_A(A)
    if branch not in ("directional", "original"):
        raise ValueError("branch must be directional or original")
    if not psn.rect:
        raise ValueError("high-order closure requires rectangular nodes")
    t0 = time.time()
    matidx = psn.mat[psn.j_idx, psn.i_idx]
    ng = psn.ng
    area_w = psn.area.ravel()
    if not hasattr(psn, "_groups"):
        psn._vectorize_setup()
    locals_ = {}
    for i in range(psn.I):
        for g in range(ng):
            locals_[(i, g)] = HoNodeLocal(psn, i, g, A, branch,
                                          gs_tol=gs_tol, max_gs=max_gs,
                                          omega=omega)
    def fission_rate(phi):
        fr = np.zeros(psn.nodes)
        for g in range(ng):
            fr += psn.nuSf[g, matidx] * phi[g]
        fr *= area_w
        return fr

    def all_groups(qnode):
        phi = np.zeros((ng, psn.nodes))
        qim = np.zeros((ng, psn.nodes, 5))
        git = 0
        gres = 0.0
        for g in range(ng):
            for i in range(psn.I):
                _, it, res, out = locals_[(i, g)].sweep(qnode[g])
                git = max(git, it)
                if res is not None:
                    gres = max(gres, res)
                phi[g] += out[:, 0]
                qim[g] += out
        return phi, qim, git, gres

    lam = 1.0
    mom = np.zeros((ng, psn.nodes, 5))
    mom[:, :, 0] = 1.0
    qnode = psn._source_update(mom, 1.0, matidx)
    F_old = float(fission_rate(np.ones((ng, psn.nodes))).sum())
    phi = qim = None
    converged = False
    dlam = 1.0
    src_res = None
    total_gs = 0
    outer = 0
    for outer in range(max_outer):
        phi, qim, git, gres = all_groups(qnode)
        total_gs += git
        F_new = float(fission_rate(phi).sum())
        lam_new = lam * F_new / F_old
        dlam = abs(lam_new - lam) / lam_new
        lam = lam_new
        s_next = psn._source_update(qim, lam, matidx)
        src_res = float(np.linalg.norm(s_next - qnode)
                        / max(1.0, np.linalg.norm(qnode)))
        qnode = s_next
        F_old = F_new
        if verbose and (outer % 5 == 0 or dlam < outer_tol):
            print(f"gs it {outer:4d}  keff={lam:.7f}  dkeff={dlam:.2e}  "
                  f"src_res={src_res:.2e}  gs={git}")
        if dlam < outer_tol and src_res < outer_tol:
            converged = True
            break
    info = dict(A=int(A), K=A + 1, branch=branch, inner="gs",
                converged=converged, outer_iterations=outer + 1,
                keff_residual=dlam, source_residual=float(src_res),
                gs_inner_iterations_total=int(total_gs),
                seconds=round(time.time() - t0, 2))
    if not converged:
        info["warning"] = (f"NOT converged after {max_outer} outer "
                           f"iterations (dlam={dlam:.2e}, "
                           f"src_res={src_res:.2e})")
    return lam, phi, qim, info


# --------------------------------------------------------------------- #
# GMRES inner sweep + cross-(i,g) fork process pool (2026-09-15)
# --------------------------------------------------------------------- #
# module-level worker state (populated by keff_highorder_gmres BEFORE
# the pool is created; inherited through fork, not pickled)
_POOL = None
_IDX = None
_BLK = None
_RTOL = 1e-11
_MAXIT = 800
_NUM_THREADS = 1
_GMRES_POOL_LIM = None


def _gmres_pool_init():
    """Worker initializer: pin BLAS to one thread per worker and set
    the numba prange thread count for the b-assembly kernel."""
    global _GMRES_POOL_LIM
    try:
        from threadpoolctl import threadpool_limits
        _GMRES_POOL_LIM = threadpool_limits(limits=1, user_api="blas")
    except ImportError:
        _GMRES_POOL_LIM = None
    try:
        from . import mckern
        if mckern.HAVE_NUMBA and mckern.ENABLED:
            mckern.set_num_threads(_NUM_THREADS)
            # JIT warm-up HERE (worker = leaf, post-fork): the first
            # parallel call initialises this worker's own OpenMP pool,
            # which is safe because the worker never forks again.
            # Running it in the PARENT before the fork instead aborts
            # every child (libgomp fork guard, 2026-09-14 bench C).
            _u = np.zeros(16)
            _qn = np.zeros((2, 5))
            _pack = mckern.pack_microclass(
                {"nodes": np.array([0, 1]),
                 "SRC": np.zeros((8, 5)),
                 "A": np.zeros((8, 8)),
                 "nb": {0: [dict(idxs=np.array([0, 1]),
                                 gath=np.zeros((2, 32), np.int64),
                                 sgn_nb=np.zeros((2, 2, 4, 4)),
                                 C0=np.zeros((2, 4, 4, 4)),
                                 src5=np.zeros((2, 4, 5)),
                                 rowidx=np.zeros((2, 8), np.int64),
                                 sigma=1.0,
                                 nodes_nb=np.array([0, 1]))]}})
            mckern.mc_b(_pack, _u, _qn)
    except Exception:
        pass


def _gmres_pool_task(arg):
    idx, qg = arg
    if _POOL is None:
        raise RuntimeError("GMRES pool worker started without forked "
                           "state (fork start method required)")
    i, g = _IDX[idx]
    nl = _POOL[(i, g)]
    t = time.time()
    _, nv, res, out = nl.sweep_gmres(qg, rtol=_RTOL, maxiter=_MAXIT)
    return out, nv, res, time.time() - t


def _gmres_pool_task_pair(arg):
    """One (i, g, pair) block task: solve the mirror-pair sub-system."""
    idx, qg = arg
    if _POOL is None:
        raise RuntimeError("GMRES pool worker started without forked "
                           "state (fork start method required)")
    i, g, pp = _BLK[idx]
    nl = _POOL[(i, g)]
    t = time.time()
    _, nv, res, out = nl.sweep_gmres_pair(qg, pp, rtol=_RTOL,
                                          maxiter=_MAXIT)
    return out, nv, res, time.time() - t


def keff_highorder_gmres(psn, A=3, branch="directional", max_outer=4000,
                         outer_tol=1e-10, rtol=1e-11, maxiter=800,
                         nworkers=None, verbose=False, pairs=None):
    """Power iteration, GMRES inner sweep, cross-(i,g) process pool.

    Why a PROCESS pool (not a thread pool, measured 2026-09-15 M2):
    the red-black matvec is ~87% ``_mc_b`` (NumPy fancy-index/einsum
    glue that holds the GIL) + ~13% ``lu_solve``; Python threads
    measure ~2% efficiency (slower than serial), and 24 threads x
    auto-threaded OpenBLAS core-dumps the heap.  One fork per (i, g)
    gives each system its own GIL; each worker pins BLAS to 1 thread
    via threadpoolctl (verified: 2000^3 matmul 36.6 -> 286 ms).

    All (i, g) HoNodeLocal factor tables are built in the parent
    BEFORE the fork (pure-numpy data, COW-shared by every worker;
    workers only write their own ``nl.u`` warm start).  Workers are
    STATIC shards (one single-worker pool per shard, worker w owns
    systems w, w+NW, ...): the warm start ``nl.u`` is worker-private
    after the fork, so a queue-grabbing ``Pool.map`` that reassigns
    systems between outers would silently cold-start the slow system
    (cost measured ~25 extra matvec per migration on M2).  Static
    sharding costs nothing on the makespan — the slow system is the
    critical path either way — and the parent accumulates in the
    exact serial (i, g) order -> deterministic bit-wise.

    ``pairs`` (lever 1, 2026-09-14): split each (i, g) system into
    its M/2 mirror-pair blocks {p, M-1-p} (the face system is
    strictly block-diagonal over pairs, probe-verified) and run one
    GMRES per block as an independent pool task.  Block count =
    I*ng*(M/2); each (i, g) still has ONE worker that owns all its
    pairs in a nested single-worker map, so the warm start u stays
    worker-private and the flux accumulation order (i, g, pair) is
    deterministic.  ``pairs=None`` auto-enables for M >= 4
    (no benefit at M=2: a single pair is the whole system).

    Convergence contract is IDENTICAL to highorder.keff_highorder:
    dlam < outer_tol AND five-moment source residual < outer_tol
    (equal gates, user decision 2026-09-15: src and keff tolerance
    同等化, production outer_tol=1e-7); reaching max_outer without both
    is reported as non-converged, never returned silently.

    Returns (lam, phi, qim, info).
    """
    import multiprocessing as mp
    import os
    from .highorder import _validate_A
    A = _validate_A(A)
    if branch not in ("directional", "original"):
        raise ValueError("branch must be directional or original")
    if not psn.rect:
        raise ValueError("high-order closure requires rectangular nodes")
    ctx = mp.get_context("fork")
    t0 = time.time()
    # Thread budget: the host has 48 cores but the convention is to
    # use at most 24 (user rule 2026-09-14).  Override with
    # PSN_CORE_BUDGET.  NW2 workers x T numba threads <= BUDGET.
    try:
        BUDGET = int(os.environ.get('PSN_CORE_BUDGET', '24'))
    except ValueError:
        BUDGET = 24
    BUDGET = max(1, min(BUDGET, os.cpu_count() or 1))
    matidx = psn.mat[psn.j_idx, psn.i_idx]
    ng = psn.ng
    area_w = psn.area.ravel()
    if not hasattr(psn, "_groups"):
        psn._vectorize_setup()

    IDX = [(i, g) for i in range(psn.I) for g in range(ng)]
    npair = psn.M // 2
    # M=2: the single pair {0,1} spans ALL modes -> zero work splitting
    # and an empty off-block mask in _pair_microclass (crash, fixed
    # 2026-09-14).  Fall back to full-class blocks; pairs buy nothing
    # at npair < 2.
    use_pairs = npair >= 2 and (pairs if pairs is not None else True)
    # parent-side build: factor tables are KB-scale per (i, g) and
    # COW-shared after the fork (vs 21x multi-GB SuperLU pool)
    locals_ = {}
    for i in range(psn.I):
        for g in range(ng):
            locals_[(i, g)] = HoNodeLocal(psn, i, g, A, branch,
                                          gs_tol=rtol, max_gs=1)
    # ---- pre-fork table prebuild (2026-09-14 OOM fix) -----------------
    # OOM root cause (measured, tools_ho/dbg_memprobe.py): the worker-side
    # LAZY _microclass/_pair_microclass build accumulates forever in
    # _mc_cache/_pmc_cache.  Static shards give each worker 10-11 distinct
    # (i, g) systems; per (i, g) that is ~406 MB full tables + ~25 MB per
    # pair table -> ~4.5 GB/worker, x24 workers = ~108 GB > 62 GB host
    # (host kernel OOM log matched 24 python procs @ 4.4-4.6 GiB).
    # Fix: build ALL tables in the PARENT before the fork.  They are pure
    # numpy data, COW-inherited read-only by every worker (workers write
    # only their own u slice and Krylov basis), so physical memory drops
    # from 24x per-worker copies to ONE shared copy.  pack_microclass is
    # pure numpy (no numba execution), safe in the parent; the numba
    # kernel itself still first-runs in the worker initializer (post-fork
    # OpenMP rule, unchanged).
    if os.environ.get("OPSN_NO_PREBUILD", "") != "1":
        t_pb = time.time()
        for (i, g), nl in locals_.items():
            for key in nl.mc_nodes:
                nl._microclass(key)
            if use_pairs:
                for pp in range(npair):
                    for key in nl.mc_nodes:
                        nl._pair_microclass(key, pp)
        if verbose:
            print(f"    prebuild {len(locals_)} (i,g) x "
                  f"{'(full+all pairs)' if use_pairs else 'full'} "
                  f"tables: {time.time() - t_pb:.0f}s (COW pre-fork)",
                  flush=True)
    NW = len(IDX) if nworkers is None else int(nworkers)
    NW = max(1, min(NW, len(IDX), os.cpu_count()))

    global _POOL, _IDX, _RTOL, _MAXIT, _BLK, _NUM_THREADS
    _POOL, _IDX = locals_, IDX
    _RTOL, _MAXIT = float(rtol), int(maxiter)
    _BLK = None

    def fission_rate(phi):
        fr = np.zeros(psn.nodes)
        for g in range(ng):
            fr += psn.nuSf[g, matidx] * phi[g]
        fr *= area_w
        return fr

    lam = 1.0
    mom = np.zeros((ng, psn.nodes, 5))
    mom[:, :, 0] = 1.0
    qnode = psn._source_update(mom, 1.0, matidx)
    F_old = float(fission_rate(np.ones((ng, psn.nodes))).sum())
    phi = qim = None
    converged = False
    dlam = 1.0
    src_res = None
    max_res = 0.0
    matvec_total = 0
    outer = 0
    # ---- block sharding (lever 1) ------------------------------------
    # system mode: one block per (i, g);  pair mode: one block per
    # (i, g, mirror-pair pp).  Static shards over single-worker pools:
    # worker w forever owns blocks w, w+NW2, ... -> warm start u stays
    # worker-private (COW), no cold-start migration.  Pair blocks are
    # independent (strict block-diagonal H, probe-verified) and touch
    # disjoint columns of u, so several workers may share one (i, g)
    # HoNodeLocal safely (COW-private u copies hold their pair slice).
    if use_pairs:
        BLOCKS = [(i, g, pp) for (i, g) in IDX for pp in range(npair)]
    else:
        BLOCKS = [(i, g, -1) for (i, g) in IDX]
    _BLK = BLOCKS
    B = len(BLOCKS)
    NW2 = B if nworkers is None else int(nworkers)
    NW2 = max(1, min(NW2, B, BUDGET if nworkers is None else (os.cpu_count() or 1)))
    # lever 3: per-worker numba threads for the b-assembly prange.
    # T = budget / workers keeps NW2*T <= BUDGET (BLAS stays pinned to 1).
    _NUM_THREADS = max(1, BUDGET // NW2)
    task = _gmres_pool_task_pair if use_pairs else _gmres_pool_task
    # NOTE: no numba execution in the PARENT.  The first parallel
    # numba call initialises the OpenMP runtime in the calling
    # process; fork() from a process with an active OpenMP pool
    # makes libgomp abort every child ("fork() called from a
    # process already using GNU OpenMP, this is unsafe", 2026-09-14
    # combo bench C).  Warm-up therefore runs in each worker's
    # initializer (post-fork, leaf processes) instead.
    pools = [ctx.Pool(1, initializer=_gmres_pool_init)
             for _ in range(NW2)]
    try:
        for outer in range(max_outer):
            t1 = time.time()
            phi = np.zeros((ng, psn.nodes))
            qim = np.zeros((ng, psn.nodes, 5))
            args_w = [[] for _ in range(NW2)]
            for bidx, (i, g, pp) in enumerate(BLOCKS):
                args_w[bidx % NW2].append((bidx, qnode[g]))
            # all shards independent: map_async, then gather in order
            futs = [p.map_async(task, a) for p, a in zip(pools, args_w)]
            results = [None] * B
            for w in range(NW2):
                for bidx, res in zip(range(w, B, NW2), futs[w].get()):
                    results[bidx] = res
            dt = time.time() - t1
            # load-balance diagnostics: per-worker makespan this outer
            makespan_w = np.zeros(NW2)
            for w in range(NW2):
                makespan_w[w] = max(t for _, _, _, t in
                                    (results[b] for b in
                                     range(w, B, NW2)))
            wb_max = float(makespan_w.max())
            wb_min = float(makespan_w.min())
            wb_idle = dt - wb_max
            outer_mat = 0
            for (i, g, pp), (out, nv, r, t) in zip(BLOCKS, results):
                phi[g] += out[:, 0]
                qim[g] += out
                outer_mat = max(outer_mat, nv)
            matvec_total += sum(r[1] for r in results)
            max_res = max(max_res, max(r[2] for r in results))
            if verbose and (outer % 20 == 0 or outer < 3):
                print(f"    [lb] dt={dt:.1f}s worker makespan "
                      f"max={wb_max:.1f}s min={wb_min:.1f}s "
                      f"ratio={wb_max/max(wb_min,1e-9):.2f} "
                      f"prepool={wb_idle:.1f}s", flush=True)
            F_new = float(fission_rate(phi).sum())
            lam_new = lam * F_new / F_old
            dlam = abs(lam_new - lam) / lam_new
            lam = lam_new
            s_next = psn._source_update(qim, lam, matidx)
            src_res = float(np.linalg.norm(s_next - qnode)
                            / max(1.0, np.linalg.norm(qnode)))
            qnode = s_next
            F_old = F_new
            if verbose and (outer % 5 == 0 or dlam < outer_tol):
                print(f"gmres it {outer:4d}  keff={lam:.7f}  "
                      f"dkeff={dlam:.2e}  src_res={src_res:.2e}  "
                      f"dt={dt:.1f}s  matvecs={outer_mat}")
            if dlam < outer_tol and src_res < outer_tol:
                converged = True
                break
    finally:
        for p in pools:
            p.close()
            p.join()
        del pools
    _POOL = _IDX = None
    info = dict(A=int(A), K=A + 1, branch=branch,
                inner="gmres" + ("_pairs" if use_pairs else ""),
                converged=converged, outer_iterations=outer + 1,
                keff_residual=dlam, source_residual=float(src_res),
                max_face_system_residual=float(max_res),
                gmres_matvecs_total=int(matvec_total),
                nworkers=int(NW2), nsystems=len(IDX),
                nblocks=int(B), npairs=int(npair),
                seconds=round(time.time() - t0, 2))
    if not converged:
        info["warning"] = (f"NOT converged after {max_outer} outer "
                           f"iterations (dlam={dlam:.2e}, "
                           f"src_res={src_res:.2e})")
    return lam, phi, qim, info
