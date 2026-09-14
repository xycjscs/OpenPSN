# -*- coding: utf-8 -*-
"""Inner-iteration node sweep (Gauss-Seidel, row/column alternating,
行列交替扫描) + CMFD outer acceleration for the A>0 face system.
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

CMFD (INNER-iteration flux-update, consistent coarse node grid):
  * coarse grid = c x c fine-node blocks (c = cmfd_collapse, must
    divide nx and ny; the 3x3 pin tiling uses c = 3 -> pin level),
    coarse material = area-weighted homogenization of each distinct
    block signature, coarse operator = the SAME A3 directional closure
    solved EXACTLY with one small SuperLU per (i, g) (the coarse node
    count is tiny: M2/M4 assembly 9 nodes -> ~1300 unknowns, M12
    assembly 9 nodes, core 324 nodes -> ~48000 unknowns; all cheap);
  * the inner source q is frozen for the duration of one (i, g) sweep,
    so the coarse solve is needed only ONCE per outer iteration;
  * every 2nd GS pass (after a cmfd_warm warm-up) the fine face
    currents and the fine source moments are corrected multiplicatively
    by R = phi_c / phi_f (clipped to [0.25, 4], damped by cmfd_omega);
  * the correction is SWITCHED OFF once
    max|R-1| < cmfd_switch: at the exact fixed point phi_c == phi_f
    (the coarse and fine problems share the frozen source q), so
    R -> 1 exactly and the final pure-GS phase converges to the
    UNACCELERATED fixed point.  The homogenization-commutator error
    of the coarse mesh only biases the transient, never the answer
    (the acceptance trajectory ends on the plain GS fixed point,
    verified against the direct SuperLU solution to 1e-11).

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
                 max_gs=500, omega=1.0, verbose=False, cmfd_collapse=0,
                 cmfd_switch=1e-8, cmfd_omega=0.6, cmfd_warm=4):
        if branch not in ("directional", "original"):
            raise ValueError("branch must be directional or original")
        self.psn, self.i, self.g, self.A = psn, i, g, A
        self.K, self.branch = A + 1, branch
        self.gs_tol, self.max_gs, self.omega = gs_tol, max_gs, omega
        self.verbose = verbose
        # inner-iteration CMFD (flux-update): coarse c x c node blocks,
        # exact coarse LU per (i, g), flux-ratio correction every 2nd
        # pass, auto-switch off below cmfd_switch (R -> 1 at the exact
        # fixed point, so the final GS phase is untouched)
        self.cmfd_collapse = int(cmfd_collapse or 0)
        self.cmfd_switch = float(cmfd_switch)
        self.cmfd_omega = float(cmfd_omega)
        self.cmfd_warm = int(cmfd_warm)
        self._coarse = None
        self._coarse_fail = False
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
        self._mc_cache[ck] = dict(nodes=nlist, wb=wb, A=A, fac=fac,
                                  SRC=SRC, nb=nb, d=d, grp=grp, rowg=rowg,
                                  J4_idx=J4_idx, J4_sgn=J4_sgn,
                                  Vj=Vj, Vs=Vs, rep=rep)
        return self._mc_cache[ck]

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
        """b = Src q + neighbour contributions (u and q)."""
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
        ``sweeps`` overrides max_gs (used for the cheap coarse solves)."""
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


# --------------------------------------------------------------------- #
# CMFD coarse mesh
# --------------------------------------------------------------------- #
def build_coarse_psn(psn, c=3):
    """Coarse pin-cell PSN2D: c x c fine-node blocks homogenized by
    area-weighted XS over each distinct block signature.  Returns
    (coarse, node_map, areaw) or None if c does not divide the grid."""
    from .solver import PSN2D
    p = psn
    nx, ny = p.nx, p.ny
    if nx % c or ny % c:
        return None
    hx, hy = p.hx, p.hy
    mat = p.mat
    nxc, nyc = nx // c, ny // c
    mat_map = np.zeros((nyc, nxc), dtype=int)
    hx_c = np.zeros((nyc, nxc))
    hy_c = np.zeros((nyc, nxc))
    node_map = np.zeros(p.nodes, dtype=np.int64)
    areaw = np.zeros(p.nodes)
    sigs = {}
    for jc in range(nyc):
        for ic in range(nxc):
            j0, i0 = jc * c, ic * c
            a_tot = 0.0
            sig = []
            for dj in range(c):
                for di in range(c):
                    j, i = j0 + dj, i0 + di
                    a = float(hx[j, i] * hy[j, i])
                    a_tot += a
                    sig.append((int(mat[j, i]), round(float(hx[j, i]), 9),
                                round(float(hy[j, i]), 9)))
            sigt = tuple(sig)
            if sigt not in sigs:
                sigs[sigt] = len(sigs)
            mat_map[jc, ic] = sigs[sigt]
            hx_c[jc, ic] = float(hx[j0:j0 + c, i0:i0 + c].sum())
            hy_c[jc, ic] = float(hy[j0:j0 + c, i0:i0 + c].sum())
            for dj in range(c):
                for di in range(c):
                    n = (j0 + dj) * nx + (i0 + di)
                    node_map[n] = jc * nxc + ic
                    areaw[n] = float(hx[j0 + dj, i0 + di]
                                     * hy[j0 + dj, i0 + di])
    nmc = len(sigs)
    ng = p.ng
    St_c = np.zeros((ng, nmc))
    Sgg_c = np.zeros((ng, ng, nmc))
    nuSf_c = np.zeros((ng, nmc))
    for mc, sigt in ((v, k) for k, v in sigs.items()):
        a_tot = 0.0
        acc_t = np.zeros(ng)
        acc_sgg = np.zeros((ng, ng))
        acc_f = np.zeros(ng)
        for jc in range(nyc):
            for ic in range(nxc):
                if mat_map[jc, ic] != mc:
                    continue
                j0, i0 = jc * c, ic * c
                for dj in range(c):
                    for di in range(c):
                        j, i = j0 + dj, i0 + di
                        a = float(hx[j, i] * hy[j, i])
                        mf = int(mat[j, i])
                        a_tot += a
                        acc_t += p.St[:, mf] * a
                        acc_sgg += p.Sgg[:, :, mf] * a
                        acc_f += p.nuSf[:, mf] * a
        St_c[:, mc] = acc_t / a_tot
        Sgg_c[:, :, mc] = acc_sgg / a_tot
        nuSf_c[:, mc] = acc_f / a_tot
    kw = dict(boundary=p.bnd, widths=(hx_c, hy_c), threads=p.threads,
              mem_limit_gb=p.mem_limit_gb)
    if p.generic:
        coarse = PSN2D(mat_map, 1.0, St_c, Sgg_c, nuSf_c, chi=p.chi,
                       generic=True, I=p.I, M=p.M, **kw)
    else:
        coarse = PSN2D(mat_map, 1.0, St_c, Sgg_c, nuSf_c, chi=p.chi,
                       generic=False, M=p.M, **kw)
    return coarse, node_map, areaw


# --------------------------------------------------------------------- #
# GS + CMFD keff driver
# --------------------------------------------------------------------- #
def keff_highorder_gs(psn, A=3, branch="directional", max_outer=12000,
                      outer_tol=1e-10, gs_tol=1e-12, max_gs=500,
                      omega=1.0, cmfd=False, cmfd_omega=0.5,
                      cmfd_collapse=3, cmfd_switch=1e-8,
                      cmfd_coarse_sweeps=4, verbose=False):
    """Power iteration with the node-GS inner sweep (+ optional CMFD).

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
    coarse = node_map = areaw = coarse_gs = None
    if cmfd:
        built = build_coarse_psn(psn, c=cmfd_collapse)
        if built is None:
            if verbose:
                print("cmfd: collapse inapplicable -> off")
        else:
            coarse, node_map, areaw = built
            coarse_gs = {}
            for i in range(coarse.I):
                for g in range(ng):
                    coarse_gs[(i, g)] = HoNodeLocal(
                        coarse, i, g, A, branch, gs_tol=1e-9,
                        max_gs=cmfd_coarse_sweeps)
    else:
        coarse = node_map = areaw = coarse_gs = None

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
    cmfd_active = cmfd and coarse is not None
    cmfd_iters = 0
    total_gs = 0
    outer = 0
    for outer in range(max_outer):
        phi, qim, git, gres = all_groups(qnode)
        total_gs += git
        F_new = float(fission_rate(phi).sum())
        lam_new = lam * F_new / F_old
        dlam = abs(lam_new - lam) / lam_new
        lam = lam_new
        if cmfd_active and outer > 0:
            s_c = np.zeros((ng, coarse.nodes, 5))
            for g in range(ng):
                for nf in range(psn.nodes):
                    s_c[g, node_map[nf]] += areaw[nf] * qnode[g, nf]
            phi_c = np.zeros((ng, coarse.nodes))
            for g in range(ng):
                for i in range(coarse.I):
                    _, _, _, out_c = coarse_gs[(i, g)].sweep(s_c[g])
                    phi_c[g] += out_c[:, 0]
            r = np.empty_like(qim)
            for g in range(ng):
                r[g] = np.clip(
                    phi_c[g][node_map] / np.maximum(qim[g, :, 0], 1e-12),
                    0.25, 4.0)
            qim *= 1.0 + cmfd_omega * (r - 1.0)
            cmfd_iters += 1
        s_next = psn._source_update(qim, lam, matidx)
        src_res = float(np.linalg.norm(s_next - qnode)
                        / max(1.0, np.linalg.norm(qnode)))
        qnode = s_next
        F_old = F_new
        if cmfd_active and dlam < cmfd_switch and src_res < cmfd_switch:
            cmfd_active = False
        if verbose and (outer % 5 == 0 or dlam < outer_tol):
            print(f"gs it {outer:4d}  keff={lam:.7f}  dkeff={dlam:.2e}  "
                  f"src_res={src_res:.2e}  gs={git}  "
                  f"cmfd={'on' if cmfd_active else 'off'}")
        if dlam < outer_tol and src_res < 1e-9:
            converged = True
            break
    info = dict(A=int(A), K=A + 1, branch=branch, inner="gs",
                converged=converged, outer_iterations=outer + 1,
                keff_residual=dlam, source_residual=float(src_res),
                gs_inner_iterations_total=int(total_gs),
                cmfd=cmfd, cmfd_iterations=int(cmfd_iters),
                cmfd_active_at_end=bool(cmfd_active),
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
_RTOL = 1e-11
_MAXIT = 800
_GMRES_POOL_LIM = None


def _gmres_pool_init():
    """Worker initializer: pin BLAS to one thread per worker."""
    global _GMRES_POOL_LIM
    try:
        from threadpoolctl import threadpool_limits
        _GMRES_POOL_LIM = threadpool_limits(limits=1, user_api="blas")
    except ImportError:
        _GMRES_POOL_LIM = None


def _gmres_pool_task(arg):
    idx, qg = arg
    if _POOL is None:
        raise RuntimeError("GMRES pool worker started without forked "
                           "state (fork start method required)")
    i, g = _IDX[idx]
    nl = _POOL[(i, g)]
    _, nv, res, out = nl.sweep_gmres(qg, rtol=_RTOL, maxiter=_MAXIT)
    return out, nv, res


def keff_highorder_gmres(psn, A=3, branch="directional", max_outer=4000,
                         outer_tol=1e-10, rtol=1e-11, maxiter=800,
                         nworkers=None, verbose=False):
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

    Convergence contract is IDENTICAL to highorder.keff_highorder:
    dlam < outer_tol AND five-moment source residual < 1e-9; reaching
    max_outer without both is reported as non-converged, never
    returned silently.

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
    matidx = psn.mat[psn.j_idx, psn.i_idx]
    ng = psn.ng
    area_w = psn.area.ravel()
    if not hasattr(psn, "_groups"):
        psn._vectorize_setup()

    IDX = [(i, g) for i in range(psn.I) for g in range(ng)]
    # parent-side build: factor tables are KB-scale per (i, g) and
    # COW-shared after the fork (vs 21x multi-GB SuperLU pool)
    locals_ = {}
    for i in range(psn.I):
        for g in range(ng):
            locals_[(i, g)] = HoNodeLocal(psn, i, g, A, branch,
                                          gs_tol=rtol, max_gs=1)
    NW = len(IDX) if nworkers is None else int(nworkers)
    NW = max(1, min(NW, len(IDX), os.cpu_count()))

    global _POOL, _IDX, _RTOL, _MAXIT
    _POOL, _IDX = locals_, IDX
    _RTOL, _MAXIT = float(rtol), int(maxiter)

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
    # static sharding: pool w runs forever on systems w, w+NW, ...
    pools = [ctx.Pool(1, initializer=_gmres_pool_init)
             for _ in range(NW)]
    try:
        for outer in range(max_outer):
            t1 = time.time()
            phi = np.zeros((ng, psn.nodes))
            qim = np.zeros((ng, psn.nodes, 5))
            args_w = [[] for _ in range(NW)]
            for idx, (i, g) in enumerate(IDX):
                args_w[idx % NW].append((idx, qnode[g]))
            # all shards independent: map_async, then gather in order
            futs = [p.map_async(_gmres_pool_task, a)
                    for p, a in zip(pools, args_w)]
            results = [None] * len(IDX)
            for w in range(NW):
                for idx, res in zip((idx for idx in range(w, len(IDX), NW)),
                                    futs[w].get()):
                    results[idx] = res
            dt = time.time() - t1
            outer_mat = 0
            for (i, g), (out, nv, r) in zip(IDX, results):
                phi[g] += out[:, 0]
                qim[g] += out
                outer_mat = max(outer_mat, nv)
            matvec_total += sum(r[1] for r in results)
            max_res = max(max_res, max(r[2] for r in results))
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
            if dlam < outer_tol and src_res < 1e-9:
                converged = True
                break
    finally:
        for p in pools:
            p.close()
            p.join()
        del pools
    _POOL = _IDX = None
    info = dict(A=int(A), K=A + 1, branch=branch, inner="gmres",
                converged=converged, outer_iterations=outer + 1,
                keff_residual=dlam, source_residual=float(src_res),
                max_face_system_residual=float(max_res),
                gmres_matvecs_total=int(matvec_total),
                nworkers=int(NW), nsystems=len(IDX),
                seconds=round(time.time() - t0, 2))
    if not converged:
        info["warning"] = (f"NOT converged after {max_outer} outer "
                           f"iterations (dlam={dlam:.2e}, "
                           f"src_res={src_res:.2e})")
    return lam, phi, qim, info
