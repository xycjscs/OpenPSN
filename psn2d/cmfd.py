# -*- coding: utf-8 -*-
"""Coarse-mesh finite-difference (CMFD) acceleration of the fission source.

STATUS: experimental, OFF by default (``--cmfd 0``).  With the coarse
conductances available here it is either biased or inert, never useful: the
accelerator only pays once the coarse face terms reproduce the fine net face
currents (see the last paragraph).  Measured on
``c5g7_uo2_assembly.yaml::M8_S2`` (plain reference keff
1.3401628545523647, 205 iterations, 23.2 s), pin-cell coarse mesh
(block 2 = 289 cells of 1.26 cm, one per homogenized pin):

=========================  ============  ======  ===========================
prolongation ratio         keff          iters   note
=========================  ============  ======  ===========================
mode / fine cell average   +20861 pcm    250     max|R-1| pinned at clip
new eigenmode / old        +0.00 pcm     205     max|R-1| = 2.3e-07
=========================  ============  ======  ===========================

Both numbers are the same scheme with a different denominator, and together
they say what is missing.  The coarse operator's fundamental mode is NOT the
fine flux -- at the already converged fine solution its amplitude ratio has
median 0.698, 88.7% of the (cell, group) entries off by more than 10%, and its
own eigenvalue is 1.1909866 against the exact 1.3401629 (-14918 pcm).  Divide
by the fine flux and the fine solution stops being a fixed point, so the run
drifts by exactly that error (and under-relaxation only scales it: omega 1.0 /
0.5 gave +20861 / +7079 pcm).  Divide by the previous converged coarse
eigenmode -- what this module now does -- and R == 1 to 2e-07 because the
coarse coefficients barely move between sweeps, so the bias is gone and so is
the acceleration.  The shipped form is therefore a SAFE NO-OP on this case.

What separates the two columns in the shipping codes is a coarse face term
that reproduces the FINE net face currents, which makes the coarsened fine
solution satisfy the coarse equations exactly: OpenMOC's tallied-current
correction ``dif_surf_corr`` (src/Cmfd.cpp:1140-1145, matrix entries
:1451-1465, limiter :1147-1164), OpenSn's partial/net-current closure, OpenMC's
nonlinear face parameter solved from net current tallies.  This solver can have
the same thing -- ``PSN2D._sweep_o`` already forms the per-segment net face
currents (``X[:, :, :4] = J4``) and drops them when reducing over M -- but it
does not have them yet, so do not tune block/merge/omega/clip hoping for
acceleration; plumb the face currents first.

Why it should have worked: ``PSN2D.keff`` iterates the fission source with
plain power iteration, whose rate is set by the dominance ratio (measured
rho = 0.9893 on the C5G7 quarter core -> ~1450 sweeps for dkeff < 1e-10), and
the slow mode is the spatial fundamental, which a coarse-mesh balance is the
standard tool for.  ``node_state`` gives the exact within-node balance of the
paraboloidal solution, ``St * phi_bar = q0 - (1/a) * sum_j J4``
(psn2d/node.py), so summing over the nodes of a coarse cell C yields that
cell's exact fine balance with  A = sum St_g V phi_g (removal),
B = sum V sum_g' Ss_{g,g'} phi_g' (in-scatter),  P = chi_g sum V sum_g' Sf_{g'}
phi_g' (production) -- no face-current output is needed from the sweep, only
the reduced flux.  Every coarse coefficient here is a *ratio to the
volume-averaged fine flux*, so the coarse operator reproduces A, B and P term
by term when phi_bar == phiC; only the leakage is modelled, by a coarse
diffusion operator.

That modelling is the whole of the approximation, and it is what breaks the
scheme: the coarse balance has its own fundamental mode, whose shape differs
from phiC by the diffusion-model error, and R is exactly that ratio.  Making R
tend to 1 needs coarse conductances that reproduce the fine face currents --
equivalence-theory homogenization (discontinuity factors), which this solver
does not have.  Note also that an accelerator's internal residual validates
nothing: the fixed-source variant of this scheme converged to exactly
keff = k/2 (66000 pcm wrong) while satisfying the dkeff < 1e-10 contract.  Only
keff against the unaccelerated reference validates one, which is why every
skipped coarse step is recorded and published (see ``_skip``) rather than
quietly no-oped.
"""
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import splu


class Cmfd:
    """One coarse-mesh rebalance per fine sweep, fission-source formulation.

    ``block`` fine nodes per coarse cell per direction; ``merge`` merges that
    many x that many NON-fissile base cells into one coarse cell while every
    fissionable base cell stays its own cell (pin-wise in fuel, coarse in
    moderator/reflector); ``start`` skips the first outer iterations (a flat
    start flux makes the homogenized coefficients meaningless and the first
    ratios huge); ``omega`` under-relaxes the multiplier; ``clip`` bounds R so
    a bad coarse solve cannot wreck the fine iterate.
    """

    def __init__(self, psn, block=4, start=4, clip=20.0, merge=1, omega=1.0,
                 groups=0, denom="prev", current=False, verbose=False):
        self.p = psn
        self.ng = psn.ng
        # Coarse GROUP structure.  Production CMFD rarely keeps one coarse
        # group per fine group -- OpenMOC's C5G7 model collapses 7 transport
        # groups to 2 (profile/models/c5g7/c5g7-3d-cmfd.cpp:543-549) -- and it
        # matters here for a reason this file used to get wrong: the closer the
        # coarse operator gets to reproducing the fine balance group by group,
        # the more exactly the coarsened fine flux IS its dominant eigenvector,
        # so R -> 1 and the accelerator is inert.  Group condensation is the
        # standard knob that keeps a consistent coarse model predictive.
        # `groups` bins contiguous fine groups (0 = keep 1:1); the ratio is
        # applied per COARSE group, as in OpenMOC src/Cmfd.cpp:1549-1552.
        self.nG = self.ng if int(groups) <= 0 else min(int(groups), self.ng)
        self.bin = np.array([min(g * self.nG // self.ng, self.nG - 1)
                             for g in range(self.ng)], dtype=np.int64)
        # Prolongation denominator: "prev" = previous converged coarse
        # eigenmode (exactly unbiased, measured inert); "fine" = coarse average
        # of the fine flux from the sweep that just finished, which is the form
        # OpenMOC actually uses (src/Cmfd.cpp:3277-3279 with _old_flux written
        # only by collapseXS at :947-957).
        self.denom = str(denom)
        # Exact-balance one-sided (discontinuity) face couplings: see _closure.
        self.current = bool(current)
        self.cA = self.cB = None
        self.block = int(block)
        self.start = int(start)
        self.clip = float(clip)
        self.merge = int(merge)
        self.omega = float(omega)
        self.verbose = bool(verbose)
        self.n_applied = 0
        self.dev_last = 0.0
        self.dev_max = 0.0
        self.n_clip = 0
        self.ncoarse = 300            # cap on coarse power iterations
        self.coarse_tol = 1e-9        # relative kbar change to call it converged
        self.ncoarse_used = 0         # iterations used by the last update
        self.xprev = None             # warm start: coarse shape from last time
        self.k_coarse = None          # last coarse kbar (diagnostic only)
        self.skip = None              # last reason a coarse step was skipped
        self.n_skip = 0               # how often that happened
        nodes, nx, ny = psn.nodes, psn.nx, psn.ny
        if nodes != nx * ny:
            raise ValueError(f"CMFD needs a full {nx}x{ny} node grid, got "
                             f"{nodes} nodes")
        b = max(1, self.block)
        j, i = np.divmod(np.arange(nodes), nx)        # n = j*nx + i
        ncx0, ncy0 = int(np.ceil(nx / b)), int(np.ceil(ny / b))
        nC0 = ncx0 * ncy0
        cid0 = ((j // b) * ncx0 + (i // b)).astype(np.int64)
        self.matidx = psn.mat[psn.j_idx, psn.i_idx]    # node -> material row
        if psn.rect:
            hx = np.asarray(psn.hx, float).ravel()
            hy = np.asarray(psn.hy, float).ravel()
            self.V = np.asarray(psn.area, float).ravel()
        else:
            hx = np.full(nodes, float(psn.h))
            hy = np.full(nodes, float(psn.h))
            self.V = np.full(nodes, float(psn.h) ** 2)
        # ---- coarse partition: pin-cell resolution where it matters ---------
        # `block` x `block` node blocks are the base cells; with merge > 1 every
        # FISSIONABLE base cell keeps its own coarse cell (one pin cell) while
        # the non-fissile ones (moderator, guide tubes, reflector) are merged
        # into merge x merge groups.  That is the partition production codes
        # actually use -- pin-wise in fuel, coarse in reflector -- instead of
        # one uniform grid that smears a fuel pin, a water gap and a guide tube
        # into a single diffusion coefficient.
        fuel = np.zeros(nC0, bool)
        has_fis = np.asarray(psn.nuSf > 0.0).any(axis=0)
        np.add.at(fuel, cid0, has_fis[self.matidx])
        mg = max(1, self.merge)
        if mg == 1 or fuel.all() or not fuel.any():
            ids = np.arange(nC0, dtype=np.int64)
        else:
            cx0, cy0 = np.divmod(cid0, ncx0)
            wfuel = np.flatnonzero(fuel)
            compact = np.full(nC0, -1, np.int64)
            compact[wfuel] = np.arange(wfuel.size)
            ids = np.empty(nC0, np.int64)
            nxt, wkey = wfuel.size, {}
            nmx = int(np.ceil(ncx0 / mg))
            for c in range(nC0):
                if fuel[c]:
                    ids[c] = compact[c]
                else:
                    k = (cy0[c] // mg) * nmx + (cx0[c] // mg)
                    if k not in wkey:
                        wkey[k] = nxt
                        nxt += 1
                    ids[c] = wkey[k]
        self.cid = ids[cid0]
        self.nC = int(ids.max()) + 1
        self.nCbase = nC0
        self.fuel = fuel[ids[:self.nC]]        # per coarse cell (approx.)
        self.VC = np.bincount(self.cid, weights=self.V, minlength=self.nC)
        # Coarse-cell EXTENTS.  A block is `block` nodes wide, so the MEAN node
        # width is not the cell width: using it makes the diffusion distance
        # `block` times too small and the face length `block` times too big,
        # i.e. over-diffuses the coarse mesh by block^2.  That is not a
        # cosmetic error -- the face conductance then outweighs removal, the
        # coarse matrix stops being an M-matrix and the coarse flux goes
        # negative (measured: row sums down to -259 at block 8, 100% of coarse
        # steps refused).
        def _extents(cid, nC):
            def one(idx, w, nmax):
                key = cid * nmax + idx            # unique (cell, line) pair
                order = np.argsort(key, kind="stable")
                k = key[order]
                first = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
                tab = np.zeros(nC * nmax)
                tab[k[first]] = w[order[first]]   # that line's width, once
                return tab.reshape(nC, nmax).sum(axis=1)
            return one(i, hx, nx), one(j, hy, ny)   # x- and y-extents
        ex, ey = _extents(self.cid, self.nC)
        ex0, ey0 = _extents(cid0, nC0) if mg > 1 else (ex, ey)
        # Faces are built at BASE resolution and merged per coarse-cell pair:
        # two coarse cells can meet along several disjoint segments, and adding
        # their conductances (summed area, area-weighted distance) is exact for
        # the collinear segments that occur on a rectilinear partition.
        seg = {}

        def _add(a, bb, ar, di):
            key = (a, bb) if a < bb else (bb, a)
            v = seg.get(key)
            seg[key] = [ar, ar * di] if v is None else [v[0] + ar, v[1] + ar * di]
        for cy in range(ncy0):
            for cx in range(ncx0):
                c = cy * ncx0 + cx
                a = int(ids[c])
                if cx + 1 < ncx0:                 # x-normal face: length is
                    n = c + 1                     # the y-extent, the distance
                    if ids[n] != a:               # is the x-extent
                        _add(a, int(ids[n]), 0.5 * (ey0[c] + ey0[n]),
                             0.5 * (ex0[c] + ex0[n]))
                if cy + 1 < ncy0:                 # y-normal face: swapped
                    n = c + ncx0
                    if ids[n] != a:
                        _add(a, int(ids[n]), 0.5 * (ex0[c] + ex0[n]),
                             0.5 * (ey0[c] + ey0[n]))
        ks = sorted(seg)
        self.pa = np.asarray([k[0] for k in ks], np.int64)
        self.pb = np.asarray([k[1] for k in ks], np.int64)
        ar = np.asarray([seg[k][0] for k in ks])
        di = np.asarray([seg[k][1] for k in ks]) / ar
        self.p_geo = (ar / di).astype(float)
        self._off_pairs = [(g, gp) for g in range(self.nG)
                           for gp in range(self.nG) if gp != g]
        self._build_pattern()

    # ------------------------------------------------------------------ #
    def _build_pattern(self):
        """Fixed stencil + the slices the per-iteration data assembly fills.

        Unknown x[g, C] flattened group-major, index = g*nC + C.  Coefficients
        change every iteration, the pattern does not, so it is built once.
        ``Iadd``/``Isub`` scatter a per-pair array onto coarse cells (the
        face-divergence and its self-term), which keeps the assembly in BLAS
        instead of a Python loop over faces.
        """
        ng, nC, np_ = self.nG, self.nC, len(self.pa)
        idx = [g * nC + np.arange(nC) for g in range(ng)]
        rows, cols = [], []
        for g in range(ng):                       # 1. diagonal
            rows.append(idx[g]); cols.append(idx[g])
        self.cut_diag = (0, ng * nC)
        for (g, gp) in self._off_pairs:           # 2. intra-cell group coupling
            rows.append(idx[g]); cols.append(idx[gp])
        a = self.cut_diag[1]
        self.cut_off = (a, a + len(self._off_pairs) * nC)
        a = self.cut_off[1]
        # 3. face couplings: per group, then per pair, both orientations
        fr = np.tile(np.concatenate([self.pa, self.pb]), ng)
        fc = np.tile(np.concatenate([self.pb, self.pa]), ng)
        base = np.repeat(np.arange(ng) * nC, 2 * np_)
        rows.append(base + fr); cols.append(base + fc)
        self.cut_face = (a, a + 2 * np_ * ng)
        self.rows = np.concatenate(rows)
        self.cols = np.concatenate(cols)
        self.nnz_pattern = self.cut_face[1]
        # Iadd scatters a signed per-pair value onto cells (+1 on the pa side,
        # -1 on the pb side); Iall the unsigned incidence (both sides +1).
        one, k = np.ones(np_), np.arange(np_)
        self.Iadd = csr_matrix((np.concatenate([one, -one]),
                                (np.concatenate([self.pa, self.pb]),
                                 np.concatenate([k, k]))),
                               shape=(nC, np_))
        self.Iall = csr_matrix((np.concatenate([one, one]),
                                (np.concatenate([self.pa, self.pb]),
                                 np.concatenate([k, k]))),
                               shape=(nC, np_))

    # ------------------------------------------------------------------ #
    def _tally(self, phi):
        """Fine-mesh group sums per coarse cell -> coarse coefficients."""
        p, ng, nC = self.p, self.ng, self.nC
        matidx = p.mat[p.j_idx, p.i_idx]
        V, cid, bc, tiny = self.V, self.cid, np.bincount, np.finfo(float).tiny
        phiC = np.empty((ng, nC))          # volume-averaged fine flux
        Nrm = np.empty((ng, nC))
        Rem = np.empty((ng, nC))           # sum St V phi
        Sigt = np.empty((ng, nC))          # flux-weighted total xs
        Sca = np.empty((ng, ng, nC))       # sum Ss_{g,g'} V phi_{g'}
        Fis = np.empty((ng, ng, nC))       # chi_g sum Sf_{g'} V phi_{g'}
        wV = np.empty((ng, p.nodes))
        for g in range(ng):
            wV[g] = V * phi[g]
            Nrm[g] = bc(cid, weights=wV[g], minlength=nC)
            Stg = p.St[g, matidx]
            Rem[g] = bc(cid, weights=Stg * wV[g], minlength=nC)
            Sigt[g] = Rem[g] / np.maximum(Nrm[g], tiny)
            phiC[g] = Nrm[g] / self.VC
        self.ok = bool(np.all(phiC > tiny) and np.all(np.isfinite(phiC)))
        if not self.ok:
            return
        inv = 1.0 / np.maximum(phiC, tiny)              # inv[gp] scales row g
        # every coarse coefficient is an aggregate DIVIDED by the averaged
        # fine flux of the group it multiplies, so the operator reproduces the
        # fine aggregate term by term when phi_bar == phiC.  (Missing this on
        # Rem leaves a raw sum on the diagonal next to ratio-scaled off
        # diagonals and produces a garbage coarse solve -- measured: the run
        # converged to keff 0.67, i.e. 66000 pcm off, while still passing the
        # dkeff < 1e-10 contract.  Internal residuals do not validate an
        # accelerator; only comparing keff against the reference does.)
        Rem = Rem * inv
        for g in range(ng):
            for gp in range(ng):
                Sca[g, gp] = bc(cid, weights=p.Sgg[g, gp, matidx] * wV[gp],
                                minlength=nC) * inv[gp]
                Fis[g, gp] = p.chi[g] * bc(cid, weights=p.nuSf[gp, matidx]
                                           * wV[gp], minlength=nC) * inv[gp]
        # D = 1/(3 Sigma_tr), Sigma_tr = flux-weighted TOTAL cross section.
        # Subtracting the within-group removal (St - Ss_gg) instead is the
        # classic mistake: isotropic in-group scattering does not transport-
        # correct, so D = 1/(3(St - Ss_gg)) blows up exactly where absorption
        # is small (thermal groups in the reflector), the face conductance
        # then exceeds removal, the coarse matrix stops being an M-matrix and
        # the coarse flux goes negative (measured on the C5G7 UO2 assembly:
        # row sums down to -119, A^-1 entries down to -4.5e-3).
        D = 1.0 / (3.0 * np.maximum(Sigt, tiny))
        if self.nG != ng:
            Rem, Sca, Fis, phiC, D = self._condense(Rem, Sca, Fis, phiC, D)
        self.D = D
        self.Rem, self.Sca, self.Fis, self.phiC = Rem, Sca, Fis, phiC

    def _condense(self, Rem, Sca, Fis, phiC, D):
        """Collapse fine groups onto the coarse group structure.

        Reaction-rate consistent by construction: every condensed coefficient
        times the condensed flux equals the sum of the fine terms it replaces
        (OpenMOC does the same in collapseXS, src/Cmfd.cpp:782-926), and D is
        the flux-weighted average of the fine D, as in src/Cmfd.cpp:884-890.
        """
        ng, nG, nC = self.ng, self.nG, self.nC
        b, tiny = self.bin, np.finfo(float).tiny
        m = [b == G for G in range(nG)]
        phiG = np.array([phiC[k].sum(axis=0) for k in m])
        w = np.maximum(phiG, tiny)
        RemG = np.array([(Rem[k] * phiC[k]).sum(axis=0) for k in m]) / w
        DG = np.array([(D[k] * phiC[k]).sum(axis=0) for k in m]) / w
        ScaG = np.zeros((nG, nG, nC))
        FisG = np.zeros((nG, nG, nC))
        for G in range(nG):
            for Gp in range(nG):
                ScaG[G, Gp] = (Sca[np.ix_(m[G], m[Gp])]
                               * phiC[m[Gp]][None]).sum(axis=(0, 1)) / w[Gp]
                FisG[G, Gp] = (Fis[np.ix_(m[G], m[Gp])]
                               * phiC[m[Gp]][None]).sum(axis=(0, 1)) / w[Gp]
        return RemG, ScaG, FisG, phiG, DG

    def _closure(self, lam):
        """One-sided face conductances that make the coarse balance EXACT.

        A Fickian face term C_f (phiA - phiB) with C_f >= 0 cannot reproduce
        the fine leakage field on this mesh: the minimum achievable worst-cell
        residual is 6% of the peak leakage in every group (linear program,
        dev/cmlp.py).  Allowing one discontinuity per cell side --

            J_f = cA_f phiA - cB_f phiB,      cA_f, cB_f >= 0

        which is exactly what an interface discontinuity factor does, makes it
        exact to solver tolerance (1e-5 relative).  Given a face-current field
        J consistent with the cell leakages, the pair is closed in
        parentheses-free form:

            cA = max(J,0)/phiA + s phiB,   cB = max(-J,0)/phiB + s phiA,
            s  = Dhat/max(phiA, phiB)

        so cA phiA - cB phiB == J identically, both conductances stay
        non-negative for ANY J (a Fickian term needs J to run downhill in the
        coarse flux, the transport answer often does not), and when J happens to
        equal the diffusion current the pair collapses back to cA = cB = Dhat,
        i.e. plain diffusion is reproduced exactly rather than perturbed.

        J itself is any face-current field with div J == Leak; it is routed up a
        spanning tree of the coarse adjacency (_spanning_tree), which is exact
        to round-off and needs no solve.  Leak comes from the
        tallies, so no extra sweep output is needed -- and by construction the
        collapsed fine flux then satisfies the coarse equations term by term at
        the current lambda, which is the fixed-point property every prolongation
        form here has been missing.
        """
        ng, nC = self.nG, self.nC
        phiC, tiny = self.phiC, np.finfo(float).tiny
        # What the operator's face term has to equal, read off _operators: its
        # row reads  Rem phiC + face - Ss_gg phiC = (all production)/kbar,  with
        # P = in-scatter AND fission, both k-scaled (the coarse step is a
        # dominance-ratio power iteration on that pair).  So
        #     face = Prod + Ss_gg phiC - Rem phiC,
        # NOT the transport-style "removal minus sources": this pair scales
        # in-scatter by 1/kbar, and feeding it the other convention leaves a
        # residual of order 1 (measured) instead of machine zero.
        ScaIn = np.einsum("gpc,pc->gc", self.Sca, phiC)
        Prod = np.einsum("gpc,pc->gc", self.Fis, phiC) / lam
        # ScaIn is ALREADY sum_gp Sca[g,gp] phiC[gp] (scatter arrives with
        # the COLUMN group's flux), so it adds here; multiplying it by
        # phiC[g] again, as this line first did, squares the in-scatter.
        Leak = Prod - self.Rem * phiC + ScaIn
        self._Leak = Leak                         # diagnostic for probes
        if not hasattr(self, "_tree"):
            self._tree = self._spanning_tree()
        if self._tree is None:               # disconnected partition
            self.cA = self.cB = None
            return
        D = self.D
        Dp, Dn = D[:, self.pa], D[:, self.pb]
        Dh = 2.0 * Dp * Dn / np.maximum(Dp + Dn, 1e-300) * self.p_geo
        pa, pb = phiC[:, self.pa], phiC[:, self.pb]
        cA, cB = np.empty_like(Dh), np.empty_like(Dh)
        for g in range(ng):
            J = self._route(Leak[g])
            s = Dh[g] / np.maximum(np.maximum(pa[g], pb[g]), tiny)
            cA[g] = np.maximum(J, 0.0) / np.maximum(pa[g], tiny) + s * pb[g]
            cB[g] = np.maximum(-J, 0.0) / np.maximum(pb[g], tiny) + s * pa[g]
        self.cA, self.cB = cA, cB

    def _spanning_tree(self):
        """Leaves-first order of a spanning tree of the coarse adjacency graph.

        Pushing each subtree's leakage imbalance up its own tree edge gives a
        face-current field with div J == Leak to round-off, in O(cells), with no
        linear solve.  The previous version solved a weighted graph Laplacian
        instead and stopped at a balance residual of 0.18 rather than machine
        zero -- on a pin mesh the pin/mixer conductance ratio makes that system
        badly conditioned and the solve eats the accuracy it is supposed to
        deliver.  Faces off the tree carry J = 0, which the one-sided closure
        still turns into a positive (cA, cB) pair, so the operator keeps its
        diffusion-like background coupling; only the tree faces carry the
        correction current.
        """
        nC, nf = self.nC, len(self.pa)
        inc = [[] for _ in range(nC)]
        for f in range(nf):
            a, b = int(self.pa[f]), int(self.pb[f])
            if a != b:                        # a face inside one cell: no current
                inc[a].append((f, b))
                inc[b].append((f, a))
        seen = np.zeros(nC, dtype=bool)
        seen[0] = True
        parent = np.full(nC, -1, dtype=np.int64)
        pface = np.full(nC, -1, dtype=np.int64)
        order, stack = [], [0]
        while stack:                          # discovery order: parent before child
            v = stack.pop()
            order.append(v)
            for f, u in inc[v]:
                if not seen[u]:
                    seen[u] = True
                    parent[u], pface[u] = v, f
                    stack.append(u)
        if np.any(parent[1:] < 0):            # cell 0 is the tree root
            return None                       # disconnected coarse adjacency
        verts = np.asarray(order[1:], dtype=np.int64)[::-1]   # children first
        return verts, pface[verts], parent[verts]

    def _route(self, leak):
        """Face currents with div J == leak, routed up the spanning tree."""
        bal = np.asarray(leak, dtype=float).copy()
        J = np.zeros(len(self.pa))
        verts, pface, parent = self._tree
        for v, f, par in zip(verts, pface, parent):   # leaves before parents
            s = bal[v]
            J[f] = s if self.pa[f] == v else -s   # positive J flows pa -> pb
            bal[par] += s
        return J

    def _skip(self, why):
        """Record why a coarse step was refused and tell the caller to skip.

        A coarse step that silently does nothing is the worst failure mode of
        an accelerator -- the run still converges, the residual contract is
        still satisfied, and the only symptom is that nothing got faster.  The
        reason is therefore kept on the object and published in the JSON.
        """
        self.skip = why
        self.n_skip += 1
        return None

    def _operators(self, *, transport_split=False, boundary_leak=None):
        """Assemble (A, P) of the coarse eigenproblem from the current tally.

        A = removal + leakage (group-diagonal), P = in-scatter + fission.
        In-scatter OUT of a group stays on A's diagonal (St_g - Ss_gg =
        absorption + fission + out-of-group scattering > 0), but group-TO-
        group transfer is lagged into P -- a Jacobi split in energy.  That is
        what keeps the coarse solve physical: with the transfer in A the
        matrix stops being an M-matrix, because the homogenized removal is
        weighted by phi_g while the transfer is weighted by phi_gp, and those
        two weights disagree inside a heterogeneous coarse cell (measured: the
        coarse flux went negative at every iteration of the C5G7 UO2 assembly,
        min = -8.6 at block 2).  A Z-matrix whose row sums are absorption +
        fission + leakage > 0 IS an M-matrix, so A^-1 >= 0 and a non-negative
        source can only produce a non-negative phi_bar.

        The transport split puts all scattering on the left and fission alone
        on the right. A caller can also supply integrated boundary leakage.
        """
        ng, nC = self.nG, self.nC
        D = self.D                                        # (ng, nC)
        Dp, Dn = D[:, self.pa], D[:, self.pb]
        Df = (2.0 * Dp * Dn / np.maximum(Dp + Dn, 1e-300)) * self.p_geo
        if getattr(self, "cA", None) is not None:
            # leakage out of A is cA phiA - cB phiB: A's diagonal takes cA, B's
            # takes cB, and the two orientations of the face stencil take -cB
            # and -cA.  Still conservative (the pair cancels between the two
            # cells) and still non-positive off-diagonal.
            fa, fb = self.cA, self.cB
            leak = np.array([np.bincount(self.pa, weights=fa[g], minlength=nC)
                             + np.bincount(self.pb, weights=fb[g], minlength=nC)
                             for g in range(ng)])
        else:
            fa = fb = Df
            leak = Df @ self.Iall.T
        dg = np.arange(ng)
        # Exact-balance currents require the transport split. Interior faces
        # cancel in the domain sum, which would be violated by the factor
        # 1/k on in-scatter in the Jacobi-in-energy split.
        transport_split = transport_split or self.cA is not None
        if boundary_leak is not None:
            leak += boundary_leak
        diagA = self.Rem + leak - self.Sca[dg, dg]
        diagP = np.stack([self.Fis[g, g] for g in range(ng)])
        npair = len(self._off_pairs)
        offSca = (np.stack([self.Sca[g, gp] for (g, gp) in self._off_pairs])
                  if npair else np.zeros((0, nC)))
        offFis = (np.stack([self.Fis[g, gp] for (g, gp) in self._off_pairs])
                  if npair else np.zeros((0, nC)))
        offP = offFis if transport_split else offFis + offSca
        # Face slots are, per group, (row pa[f], col pb[f]) for every face f and
        # then the mirrored (row pb[f], col pa[f]): see _build_pattern.  The
        # value in BOTH orientations is the FACE conductance Df[f]; gathering it
        # by cell index instead (Df[g, self.pa], as this line used to do)
        # repeats the low-numbered faces, never uses faces >= nC at all, and
        # leaves the off-diagonal inconsistent with the diagonal's sum over
        # faces -- the leakage operator then is not a divergence at all.  On
        # c5g7_uo2_assembly::M8_S2 that pinned the coarse eigenvalue 14918 pcm
        # below the exact fine one, immovable by any face-conductance scale,
        # and kept median(mode/phiC) at 0.70 -- which is what made every
        # prolongation form here either biased or inert.
        face = -np.concatenate([np.concatenate([fb[g], fa[g]])
                                for g in range(ng)])
        # the flattened unknown is group-major (index = g*nC + C), which is
        # exactly (ng, nC) in C order -- hence the bare ravels.  The reaction
        # stencil slots are (diagonal, then off-diagonal pairs) in the order
        # _build_pattern laid them out.
        sl = np.r_[0:nC * ng, self.cut_off[0]:self.cut_off[1]]
        rA, cA = self.rows[sl], self.cols[sl]
        rD, cD = self.rows[:nC * ng], self.cols[:nC * ng]
        rf, cf = (self.rows[self.cut_face[0]:self.cut_face[1]],
                  self.cols[self.cut_face[0]:self.cut_face[1]])
        shape = (ng * nC, ng * nC)
        if transport_split:
            valA = np.concatenate([diagA.ravel(), -offSca.ravel(), face])
            idxA = (np.concatenate([rA, rf]), np.concatenate([cA, cf]))
        else:
            valA = np.concatenate([diagA.ravel(), face])
            idxA = (np.concatenate([rD, rf]), np.concatenate([cD, cf]))
        A = coo_matrix((valA, idxA), shape=shape).tocsc()
        P = coo_matrix((np.concatenate([diagP.ravel(), offP.ravel()]),
                        (rA, cA)), shape=shape).tocsc()
        return A, P

    def ratios(self, phi, lam):
        """Coarse eigen-solve -> (nC, ng) prolongation ratio, or None to skip.

        The coarse step is an EIGENVALUE problem ``A phi = (1/kbar) P phi``,
        power-iterated on the coarse mesh, warm-started from the previous
        coarse eigenmode.  What gets applied to the fine mesh is NOT that mode
        divided by the fine flux.  The first version here did exactly that and
        is wrong in kind, not in tuning: at the already converged fine solution
        of c5g7_uo2_assembly::M8_S2 the pin-cell coarse mode is not the fine
        flux (median mode/mean ratio 0.698, 88.7% of the (cell, group) entries
        off by more than 10%, kbar 1.190986649 against the exact fine
        1.340162855), so the fine solution is not a fixed point of the
        accelerated iteration at all, and the run drifted to +20861 pcm while
        satisfying its own dkeff < 1e-10 contract.

        The ratio shipping codes apply is that same mode divided by the coarse
        average of the FINE flux of the sweep that just finished.  In OpenMOC
        ``getFluxRatio`` is ``_new_flux / _old_flux`` (src/Cmfd.cpp:3277-3279)
        and ``_old_flux`` is written in exactly one place -- the collapse of
        the current MOC flux (src/Cmfd.cpp:947-957); there is no
        converged-flux swap anywhere in that file, and both operands are
        rescaled to unit total fission source before the ratio is taken
        (rescaleFlux, src/Cmfd.cpp:1304-1323).  So this module's first version
        had the standard ratio, and the measurement above is the reason it does
        not work HERE: OpenMOC can afford it because it solves its coarse face
        term D~ from the transport-tallied net current of that same face
        (src/Cmfd.cpp:1140-1145), which makes the coarsened fine flux satisfy
        the coarse equations, so R -> 1 *as the coupled iteration converges*.
        R -> 1 at convergence is how one tells CMFD is working; R == 1 at every
        iteration means it is doing nothing.

        What is applied below is therefore the weaker, safe ratio: the newly
        converged coarse eigenmode over the previous one.  Exactly unbiased --
        when the fine iterate stops changing so do the coefficients and the
        eigenmode, R is 1, and the plain source update is reproduced bit for
        bit -- and on this lattice exactly inert (max|R-1| = 2.3e-07 over a
        whole M8_S2 run).  It keeps the correctness half of this settled while
        the face currents are still missing: acceleration needs
        current-consistent coarse conductances, not a different denominator.
        Normalizing both vectors to unit production also conserves the
        volume-integrated fission source.

        kbar is a diagnostic only; the reported keff always comes from the fine
        solve, never from this operator, whose own eigenvalue sits 14918 pcm
        below the exact answer on this case.
        """
        self._tally(phi)
        if not self.ok:
            return self._skip("flux not positive/finite")
        ng, nC = self.nG, self.nC
        if self.current:
            self._closure(lam)                    # one-sided face couplings
        else:
            self.cA = self.cB = None
        A, P = self._operators()                  # see _operators for both
        try:                                                    # definitions
            luA = splu(A)
        except Exception as exc:
            return self._skip(f"LU failed: {exc}")
        n = ng * nC
        Vw = np.repeat(self.VC, ng)                       # coarse volumes
        if self.xprev is not None and self.xprev.size == n:
            x = self.xprev.copy()
            warm = False
        else:
            # No previous coarse flux to take a ratio against.  Build one and
            # apply nothing this iteration -- the alternative (mode / fine mean
            # flux) is exactly the biased form above.  OpenMOC likewise runs
            # its first sweep uncorrected, src/Cmfd.cpp:1175-1178.
            x = self.phiC.ravel().astype(float).copy()
            warm = True
        k, kc = float(lam), None
        nit = 0
        for nit in range(1, self.ncoarse + 1):
            xn = luA.solve(P @ x)
            if not np.all(np.isfinite(xn)):
                return self._skip("coarse solve not finite")
            if xn.min() <= 0.0:
                return self._skip(f"coarse flux not positive "
                                  f"(min={xn.min():.3e}, k~{k:.4f})")
            sn, s0 = float(Vw @ xn), float(Vw @ x)
            if not sn > 0.0 or not s0 > 0.0:
                return self._skip("production vanished")
            # x is an eigenvector iterate, so A^-1 P x = kbar x gives
            # kbar = (Vw.xn)/(Vw.x) at convergence -- a ratio, hence
            # independent of whatever scale x happens to carry.
            k = sn / s0                                   # coarse kbar
            if not 1e-100 < float(np.abs(xn).max()) < 1e100:
                xn /= float(np.abs(xn).max())             # overflow guard
            x = xn
            if kc is not None and abs(k - kc) <= self.coarse_tol * abs(k):
                break                                     # coarse mode converged
            kc = k
        self.ncoarse_used = nit
        self.k_coarse = k
        F = float(Vw @ (P @ x))
        if not F > 0.0:
            return self._skip("no production in coarse eigenmode")
        x = x / F                             # unit total production, as
        xprev, self.xprev = self.xprev, x     # rescaleFlux does
        if self.denom == "fine":
            Fc = float(Vw @ (P @ self.phiC.ravel()))
            if not Fc > 0.0:
                return self._skip("no production in the coarse fine flux")
            den = self.phiC.ravel() / Fc      # same normalization, OpenMOC's
        else:
            den = xprev
            if warm:
                self.skip = ("coarse warm-up (no previous coarse flux to "
                             "ratio)")
                return None
        Rr = (x / den).reshape(ng, nC).T                    # (nC, nG)
        if self.omega != 1.0:
            # Under-relaxation of the update, the OpenSn-style damping of
            # cmfd_correction.cc:124 (relaxation, damping-halving retries).
            Rr = 1.0 + self.omega * (Rr - 1.0)
        R = np.clip(Rr, 1.0 / self.clip, self.clip)
        self.n_clip += int(np.count_nonzero(Rr != R))
        self.dev_last = float(np.abs(R - 1.0).max())
        self.dev_max = max(self.dev_max, self.dev_last)
        self.n_applied += 1
        if self.verbose and self.n_applied % 10 == 0:
            print(f"  cmfd[{self.n_applied}] max|R-1|={self.dev_last:.3e} "
                  f"clipped={self.n_clip}")
        return R

    # ------------------------------------------------------------------ #
    def source(self, qim, lam, matidx=None):
        """CMFD-corrected fission source, or None to run the plain update.

        Prolongation: scale the flux moments inside each coarse cell by R --
        all five paraboloidal moments, as OpenMOC scales its linear-source
        moments too (src/Cmfd.cpp:1549-1561) -- then form the next source with
        the UNCHANGED ``PSN2D._source_update``.  Where R == 1 the products are
        bit-identical to the plain path.
        """
        R = self.ratios(qim[:, :, 0], lam)
        if R is None:
            return None
        p, ng = self.p, self.ng
        if matidx is None:
            matidx = p.mat[p.j_idx, p.i_idx]
        Rnode = R[self.cid][:, self.bin]        # (nodes, ng), one coarse
        return p._source_update(qim * Rnode.T[:, :, None], lam, matidx)
