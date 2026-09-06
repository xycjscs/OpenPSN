"""
PSN2D global solver — restricted 2D (TY polar quadrature), 1-group.
Chao et al., ANE 240 (2027) 112707.  keff power iteration + source update (2.16).

Angular convention (restricted 2D, z-invariant):
  in-plane direction of region (i, m):  mu_i * (cos phi_m, sin phi_m)
    mu_i  = TY 3-point sin(theta) set  (in-plane cosine magnitude)
    phi_m = (m + 0.5) * 2*pi / M       M segments over the FULL 2*pi
  quadrature weight per (i, m):  W_i / M          (sum over i,m = 1)
  (paper 2.9c: w_im = sin(theta_i) sin(dtheta_i/2) dphi_m / pi  =  W_i/M)

Mirror-reflective boundary (2.9f/g) — the mirror DEPENDS ON THE FACE NORMAL:
  x-face (normal along x):  phi' = -phi      ->  m' = (M - 1 - m) mod M
  y-face (normal along y):  phi' = pi - phi  ->  m' = (M//2 - m - 1) mod M
  Flux:  Phi_m = Phi_m'          (2.9f)  -- one row per mirror pair
  Current: J_m' = -J_m           (2.9g)  -- column elimination, sign -1
  (M must be even; no segment center sits exactly on a face normal.)

Per (polar i) a dense system A u = b over face-current unknowns u[f, m].
Node response: (1/alpha)Phi = R . J4 + s   (psn_node.node_response_matrix).
Rows:
  internal face:  continuity of (1/alpha)Phi on both sides  (2.9d),
                  current J shared (single unknown per (f, m))
  vacuum:         J = sgn(cos(phi_m - phi_n)) * 2 * (1/alpha)Phi   (2.9e)
  reflect:        Phi_m = Phi_m'                                   (2.9f)
Source update (2.16, all five paraboloidal moments):
  q_n[n] = (S_gg[n] + nuSf[n]/lam) * sum_{i,m} w_im * mom_n(i, m; n)

The inner (scattering) solve is done exactly per (i, m-set) for the current
source; the fission part is power-iterated with k-update (equivalent to the
paper's two-step iteration, 2.16 steps (1)-(4)).
"""
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from .node import (node_alphas, node_alphas_generic, node_params_generic,
                      node_state, node_response_matrix, ty_polar_set,
                      generic_polar_set)

PI = np.pi


def mirror_x(m, M):
    """Mirror of azimuth segment m about an x-face normal (phi' = -phi)."""
    return (M - 1 - m) % M


def mirror_y(m, M):
    """Mirror of azimuth segment m about a y-face normal (phi' = pi - phi)."""
    return (M // 2 - m - 1) % M


def face_pairs(mirror, M):
    """Disjoint mirror pairs (master, slave) over m = 0..M-1, master = min."""
    seen = set()
    pairs = []
    for m in range(M):
        if m in seen:
            continue
        p = mirror(m, M)
        a, b = (m, p) if m <= p else (p, m)
        pairs.append((a, b))
        seen.add(m)
        seen.add(p)
    return pairs


class PSN2D:
    def __init__(self, mat_map, h, St, Sgg, nuSf, chi=None,
                 boundary=("reflect",) * 4, M=12, TY=None, full_2pi=False,
                 generic=False, I=30):
        if M % 2 != 0:
            raise ValueError("M must be even (mirror pairs need paired segments)")
        self.mat = mat_map.astype(np.int64)
        self.h = h
        self.nx, self.ny = mat_map.shape[1], mat_map.shape[0]
        self.St = np.atleast_2d(St).astype(float)      # (ng, nmat)
        self.Sgg = np.atleast_3d(Sgg).astype(float)    # (ng, ng, nmat)
        self.nuSf = np.atleast_2d(nuSf).astype(float)  # (ng, nmat)
        self.ng = self.St.shape[0]
        self.chi = (np.atleast_1d(chi).astype(float) if chi is not None
                    else np.ones(self.ng))
        self.bnd = list(boundary)                       # x-, y-, x+, y+
        self.M = M
        self.full_2pi = full_2pi
        # paper convention (2.10b): M segments over [0, pi) (180 deg);
        # opposite directions are automatically included by the P1 base function
        # (cos2phi/sin2phi invariant under phi -> phi + pi).  M = paper's M.
        self.dphi = (2.0 * PI / M) if full_2pi else (PI / M)
        self.generic = generic
        if generic:
            # uniform polar partition (2.10d-f); in 2D z-invariant, opposite
            # pairs (theta, pi-theta) are identical -> keep I/2 distinct lines,
            # doubled weight (2.9c): w_i = 2 sin(th_i) sin(dth/2), sum = 1
            ps = generic_polar_set(I)
            self.theta = ps["theta"]
            self.dtheta = ps["dtheta"]
            self.W = ps["w"]
        else:
            if TY is None:
                TY = ty_polar_set(3)
            self.mu = np.asarray(TY["mu"])
            self.W = np.asarray(TY["w"])
            self.W = self.W / self.W.sum()              # 3.2b normalization
            self.theta = None
            self.dtheta = None
        self.I = len(self.W)
        self._build_topology()
        self._Rcache = {}
        self._syscache = {}
        self._lusys = {}

    # ------------------------------------------------------------------ #
    def _node_pr_alpha(self, i, m, mat):
        """Node params + alpha vector for direction (i, m), material mat."""
        if self.generic:
            pr = node_params_generic(self.St[0, mat], self.h,
                                     self.theta[i], self.dtheta)
            alpha = node_alphas_generic(self.theta[i], self.dtheta, self.h,
                                        self._phi_m(m), self.dphi)
        else:
            pr = None
            alpha = None
        return pr, alpha

    def _node_params_i(self, i, mat):
        if self.generic:
            return node_params_generic(self.St[0, mat], self.h,
                                       self.theta[i], self.dtheta)
        return None

    # ------------------------------------------------------------------ #
    def _build_topology(self):
        nx, ny = self.nx, self.ny
        self.nodes = nx * ny
        self.i_idx = (np.arange(self.nodes) % nx).astype(np.int64)
        self.j_idx = (np.arange(self.nodes) // nx).astype(np.int64)
        fxp = np.full(self.nodes, -1, np.int64)
        fxm = np.full(self.nodes, -1, np.int64)
        fyp = np.full(self.nodes, -1, np.int64)
        fym = np.full(self.nodes, -1, np.int64)
        faces = []
        for j in range(ny):
            for i in range(nx - 1):
                A = j * nx + i
                fid = len(faces)
                faces.append({"type": "internal", "a": A, "b": A + 1, "axis": "x"})
                fxp[A] = fid; fxm[A + 1] = fid
        for j in range(ny - 1):
            for i in range(nx):
                A = j * nx + i
                fid = len(faces)
                faces.append({"type": "internal", "a": A, "b": A + nx, "axis": "y"})
                fyp[A] = fid; fym[A + nx] = fid
        sides = {0: ("x-", fxm), 1: ("y-", fym), 2: ("x+", fxp), 3: ("y+", fyp)}
        for n in range(self.nodes):
            i, j = self.i_idx[n], self.j_idx[n]
            for k, (i0, j0) in enumerate([(i == 0, 0), (j == 0, 1),
                                          (i == nx - 1, 2), (j == ny - 1, 3)]):
                if i0:
                    name, arr = sides[k]
                    fid = len(faces)
                    faces.append({"type": self.bnd[k], "a": n, "axis": name[:1],
                                  "side": k})
                    arr[n] = fid
        self.faces = faces
        self.F = len(faces)
        self.fxp, self.fxm, self.fyp, self.fym = fxp, fxm, fyp, fym
        # face outward-normal angle (in the full 2*pi convention)
        self.face_phi = []
        for fc in faces:
            if fc["type"] == "internal":
                self.face_phi.append(0.0)
            else:
                self.face_phi.append({0: PI, 1: 1.5 * PI,
                                      2: 0.0, 3: 0.5 * PI}[fc["side"]])
        # per-face mirror map (reflect faces only)
        self.face_mirror = {}
        self.face_pairs = {}
        for f in range(self.F):
            if self.faces[f]["type"] == "reflect":
                if self.full_2pi:
                    # directed lines: x-face flips phi->-phi, y-face flips phi->pi-phi
                    mir = mirror_y if self.faces[f]["axis"] == "x" else mirror_x
                else:
                    # half-plane [0,pi): opposite directions combined, so BOTH
                    # x-face (phi->pi-phi) and y-face (phi->-phi ~ pi-phi mod pi)
                    # use the same mirror m' = M-1-m
                    mir = mirror_x
                self.face_mirror[f] = mir
                self.face_pairs[f] = face_pairs(mir, self.M)
        # unknown columns: internal/vacuum -> all M; reflect -> one per pair
        colidx = np.full((self.F, self.M), -1, np.int64)
        ncol = 0
        for f in range(self.F):
            t = self.faces[f]["type"]
            if t in ("internal", "vacuum"):
                for m in range(self.M):
                    colidx[f, m] = ncol; ncol += 1
            else:
                for (a, b) in self.face_pairs[f]:
                    colidx[f, a] = ncol; ncol += 1
        self.colidx = colidx
        self.ncol = ncol
        # reflect elimination: slave segment -> master column with sign -1
        self.elim_col = np.full((self.F, self.M), -1, np.int64)
        self.elim_sign = np.zeros((self.F, self.M))
        for f in range(self.F):
            if self.faces[f]["type"] != "reflect":
                continue
            for (a, b) in self.face_pairs[f]:
                self.elim_col[f, b] = colidx[f, a]
                self.elim_sign[f, b] = -1.0
        # per node: local face j -> (global face, outward sign)
        self.node_fmap = [None] * self.nodes
        for n in range(self.nodes):
            self.node_fmap[n] = [(self.fxp[n], 1.0), (self.fxm[n], -1.0),
                                 (self.fyp[n], 1.0), (self.fym[n], -1.0)]

    # ------------------------------------------------------------------ #
    def _col_of(self, f, m):
        """Column for u[f,m]; eliminated reflect cols mapped to master col
        (sign folded in by caller via elim_sign)."""
        c = self.colidx[f, m]
        if c >= 0:
            return c, 1.0
        return self.elim_col[f, m], self.elim_sign[f, m]

    def _phi_m(self, m):
        return (m + 0.5) * self.dphi

    def _pr_generic(self, i, g, mat):
        """Generic-mode node params for direction i, material mat (with St/a)."""
        pr = node_params_generic(self.St[g, mat], self.h,
                                 self.theta[i], self.dtheta)
        return {**pr, "St": self.St[g, mat], "a": self.h}

    def _alpha_for(self, i, m):
        if self.generic:
            return node_alphas_generic(self.theta[i], self.dtheta, self.h,
                                       self._phi_m(m), self.dphi)
        return None

    def _R(self, i, g, m, mat):
        k = (i, g, m, mat)
        if k not in self._Rcache:
            if self.generic:
                R, _ = node_response_matrix(self.St[g, mat], 1.0, self.h,
                                            self._phi_m(m), self.dphi,
                                            pr=self._pr_generic(i, g, mat),
                                            alpha=self._alpha_for(i, m))
            else:
                R, _ = node_response_matrix(self.St[g, mat], self.mu[i], self.h,
                                            self._phi_m(m), self.dphi)
            self._Rcache[k] = R
        return self._Rcache[k]

    def _state(self, J4, qn, i, g, n, m):
        """node_state with generic/restricted params selected automatically."""
        mat = self.mat[self.j_idx[n], self.i_idx[n]]
        if self.generic:
            return node_state(J4, qn, self.St[g, mat], 1.0, self.h,
                              self._phi_m(m), self.dphi,
                              pr=self._pr_generic(i, g, mat),
                              alpha=self._alpha_for(i, m))
        return node_state(J4, qn, self.St[g, mat], self.mu[i],
                          self.h, self._phi_m(m), self.dphi)

    # ------------------------------------------------------------------ #
    def build_system(self, i, g):
        key = (i, g)
        if key in self._syscache:
            return self._syscache[key]
        M = self.M
        # SPARSE assembly (the matrix has ~9 nonzeros per row; the former
        # dense np.zeros((ncol,ncol)) was the OOM source at large N).
        A_rows = {}     # row -> {col: val}
        rows_rhs = []     # list of (row, [(node, m, lf, sgn), ...])
        for f in range(self.F):
            fac = self.faces[f]
            t = fac["type"]
            if t == "internal":
                A_n, B_n = fac["a"], fac["b"]
                lfA = self._local_face(A_n, f)
                lfB = self._local_face(B_n, f)
                matA = self.mat[self.j_idx[A_n], self.i_idx[A_n]]
                matB = self.mat[self.j_idx[B_n], self.i_idx[B_n]]
                for m in range(M):
                    RA = self._R(i, g, m, matA)
                    RB = self._R(i, g, m, matB)
                    row = len(rows_rhs)
                    for j in range(4):
                        fj, sg = self.node_fmap[A_n][j]
                        c, es = self._col_of(fj, m)
                        A_rows.setdefault(row, {})
                        A_rows[row][c] = A_rows[row].get(c, 0.0) \
                            + RA[lfA, j] * sg * es
                    for j in range(4):
                        fj, sg = self.node_fmap[B_n][j]
                        c, es = self._col_of(fj, m)
                        A_rows.setdefault(row, {})
                        A_rows[row][c] = A_rows[row].get(c, 0.0) \
                            - RB[lfB, j] * sg * es
                    # L_A - L_B = 0  ->  R_A.J_A - R_B.J_B = s_B - s_A
                    rows_rhs.append((row, [(A_n, m, lfA, -1.0),
                                           (B_n, m, lfB, +1.0)]))
            elif t == "vacuum":
                n = fac["a"]
                lf = self._local_face(n, f)
                mat = self.mat[self.j_idx[n], self.i_idx[n]]
                phi_n = self.face_phi[f]
                for m in range(M):
                    R = self._R(i, g, m, mat)
                    al = self._alpha_for(i, m)
                    alpha_lf = (al if al is not None
                                else node_alphas(self.mu[i], self.h,
                                                 self._phi_m(m), self.dphi))[lf]
                    cfac = np.cos(self._phi_m(m) - phi_n)
                    sgn = 1.0 if cfac >= 0 else -1.0
                    row = len(rows_rhs)
                    for j in range(4):
                        fj, sg = self.node_fmap[n][j]
                        c, es = self._col_of(fj, m)
                        A_rows.setdefault(row, {})
                        A_rows[row][c] = A_rows[row].get(c, 0.0) \
                            + (-2.0 * sgn * alpha_lf * R[lf, j]) * sg * es
                    c, es = self._col_of(f, m)
                    A_rows.setdefault(row, {})
                    A_rows[row][c] = A_rows[row].get(c, 0.0) \
                        + self.node_fmap[n][lf][1] * es
                    rows_rhs.append((row, [(n, m, lf, 2.0 * sgn * alpha_lf)]))
            else:  # reflect
                n = fac["a"]
                lf = self._local_face(n, f)
                mat = self.mat[self.j_idx[n], self.i_idx[n]]
                for (a, b) in self.face_pairs[f]:
                    Ra = self._R(i, g, a, mat)
                    Rb = self._R(i, g, b, mat)
                    row = len(rows_rhs)
                    for j in range(4):
                        fj, sg = self.node_fmap[n][j]
                        c, es = self._col_of(fj, a)
                        A_rows.setdefault(row, {})
                        A_rows[row][c] = A_rows[row].get(c, 0.0) \
                            + Ra[lf, j] * sg * es
                    for j in range(4):
                        fj, sg = self.node_fmap[n][j]
                        c, es = self._col_of(fj, b)
                        A_rows.setdefault(row, {})
                        A_rows[row][c] = A_rows[row].get(c, 0.0) \
                            - Rb[lf, j] * sg * es
                    rows_rhs.append((row, [(n, a, lf, -1.0),
                                           (n, b, lf, +1.0)]))
        nr = len(rows_rhs)
        rr = np.concatenate([np.full(len(d), r, np.int32)
                             for r, d in sorted(A_rows.items())])
        cc = np.concatenate([np.array(list(d.keys()), np.int32)
                             for _, d in sorted(A_rows.items())])
        vv = np.concatenate([np.array(list(d.values()), float)
                             for _, d in sorted(A_rows.items())])
        A = csc_matrix((vv, (rr, cc)), shape=(nr, self.ncol))
        assert A.shape[0] == len(rows_rhs), (A.shape, len(rows_rhs))
        self._syscache[key] = (A, rows_rhs)
        return A, rows_rhs

    def _local_face(self, n, f):
        for j in range(4):
            if self.node_fmap[n][j][0] == f:
                return j
        raise ValueError("face not on node")

    # ------------------------------------------------------------------ #
    def _lu(self, i, g):
        key = (i, g)
        if key not in self._lusys:
            A, rows_rhs = self.build_system(i, g)
            self._lusys[key] = (splu(csc_matrix(A)), rows_rhs)
        return self._lusys[key]

    def _raw_flux(self, i, g, n, m, qn):
        """(1/alpha)Phi for node n seg m given source qn, J4=0 (the s part)."""
        st = self._state(np.zeros(4), qn, i, g, n, m)
        return st["Phibar"] / st["alpha"]

    # ------------------------------------------------------------------ #
    def _vectorize_setup(self):
        """One-off tables for the vectorized node-state path."""
        self.mat_of_node = self.mat[self.j_idx, self.i_idx]
        self._mats_present = sorted(set(self.mat_of_node.tolist()))
        self._mat_sel = {m: (self.mat_of_node == m) for m in self._mats_present}
        self._node_of_mat = {}
        for m in self._mats_present:
            self._node_of_mat[m] = int(np.argmax(self._mat_sel[m]))
        # J4 gather tables: J4[n,m,j] = esgn[m,j,n] * u[cidx[m,j,n]]
        M, nd = self.M, self.nodes
        self._cidx = np.zeros((M, 4, nd), np.int64)
        self._esgn = np.zeros((M, 4, nd))
        for m in range(M):
            for n in range(nd):
                for j in range(4):
                    f, sg = self.node_fmap[n][j]
                    c, es = self._col_of(f, m)
                    self._cidx[m, j, n] = c
                    self._esgn[m, j, n] = sg * es
        # b-term grouping is per (i, g) system (vacuum-row coefficients and
        # row numbering depend on the angular line); built lazily in
        # _bterms_for.
        self._bterms_by_sys = {}
        self._Scache = {}

    def _state_matrix(self, i, g, m, mat):
        """9x9 linear map  [Phi(4); phi_bar; mom[1..4]]  =  S @ [J4(4); q(5)]
        for direction (i,m), group g, material mat.
        Probed with the VALIDATED node_state (self._state) on a reference
        node of that material — no re-derived algebra."""
        k = (i, g, m, mat)
        if k in self._Scache:
            return self._Scache[k]
        n0 = self._node_of_mat[mat]
        S = np.zeros((9, 9))
        for e in range(4):                       # J4 basis columns
            J4 = np.zeros(4); J4[e] = 1.0
            st = self._state(J4, np.zeros(5), i, g, n0, m)
            S[:4, e] = st["Phi"]; S[4, e] = st["phi_bar"]; S[5:, e] = st["mom"][1:]
        for e in range(5):                       # source-moment columns
            q = np.zeros(5); q[e] = 1.0
            st = self._state(np.zeros(4), q, i, g, n0, m)
            S[:4, 4 + e] = st["Phi"]; S[4, 4 + e] = st["phi_bar"]
            S[5:, 4 + e] = st["mom"][1:]
        self._Scache[k] = S
        return S

    def _sweep(self, g, qnode):
        """Exact solve for the given source (vectorized)."""
        if not hasattr(self, '_Scache'):
            self._vectorize_setup()
        pb = np.zeros(self.nodes)
        qim_sum = np.zeros((self.nodes, 5))
        M = self.M
        for i in range(self.I):
            u = self.solve_i(i, g, qnode)
            w = self.W[i] / M
            J4 = np.empty((self.nodes, M, 4))
            for m in range(M):
                for j in range(4):
                    J4[:, m, j] = self._esgn[m, j] * u[self._cidx[m, j]]
            X = np.empty((self.nodes, M, 9))
            X[:, :, :4] = J4
            X[:, :, 4:] = qnode[:, None, :]
            Y = np.empty((self.nodes, M, 9))
            for m in range(M):
                for mat in self._mats_present:
                    S = self._state_matrix(i, g, m, mat)
                    sel = self._mat_sel[mat]
                    Y[sel, m, :] = X[sel, m, :] @ S.T
            pb += w * Y[:, :, 4].sum(axis=1)
            qim_sum[:, 0] += w * Y[:, :, 4].sum(axis=1)
            qim_sum[:, 1:] += w * Y[:, :, 5:].sum(axis=1)
        return pb, qim_sum

    def _bterms_for(self, i, g):
        """(m, lf) -> (rows, sgns, nodes) for THIS (i, g) system only.
        Row numbering and vacuum-row coefficients are per-system, so the
        table must not be shared across (i, g)."""
        key = (i, g)
        bt = self._bterms_by_sys.get(key)
        if bt is None:
            from collections import defaultdict
            _, rows_rhs = self._lu(i, g)
            acc = defaultdict(lambda: ([], [], []))
            for row, terms in rows_rhs:
                for (n, m, lf, sgn) in terms:
                    a = acc[(m, lf)]
                    a[0].append(row); a[1].append(sgn); a[2].append(n)
            bt = {k: (np.array(v[0], np.int64), np.array(v[1]),
                      np.array(v[2], np.int64)) for k, v in acc.items()}
            self._bterms_by_sys[key] = bt
        return bt

    def solve_i(self, i, g, qnode):
        lu, rows_rhs = self._lu(i, g)
        M = self.M
        raw = np.empty((self.nodes, M, 4))   # Phi at J4=0 (Phibar/alpha)
        for m in range(M):
            for mat in self._mats_present:
                S = self._state_matrix(i, g, m, mat)
                sel = self._mat_sel[mat]
                raw[sel, m, :] = qnode[sel] @ S[:4, 4:].T
        b = np.zeros(self.ncol)
        for (m, lf), (rws, sgns, nds) in self._bterms_for(i, g).items():
            np.add.at(b, rws, sgns * raw[nds, m, lf])
        return lu.solve(b)

    def _node_state_at(self, i, g, n, m, u, qn):
        J4 = np.zeros(4)
        for j in range(4):
            fj, sg = self.node_fmap[n][j]
            c, es = self._col_of(fj, m)
            J4[j] = sg * es * u[c]
        return self._state(J4, qn, i, g, n, m)

    def keff(self, max_outer=2000, outer_tol=1e-9, verbose=True,
             source="full"):
        """Multi-group keff driver.
        Source moments (paper B.3): the paraboloidal source in group g is
            qmom_g = sum_{g2} [ Sgg[g,g2] + chi_g * nuSf[g2]/lam ] * mom(phi_g2)
        i.e. fission source is local (same spatial parabola as the flux),
        spectrally distributed by chi.  Reduces exactly to the single-group
        'full' variant (S + nuSf/lam)*mom that reproduces Fig.3 to <1 pcm."""
        matidx = self.mat[self.j_idx, self.i_idx]
        ng = self.ng
        lam = 1.0
        phi = np.ones((ng, self.nodes))
        fission_rate = np.zeros(self.nodes)
        for g in range(ng):
            fission_rate += self.nuSf[g, matidx] * phi[g]
        # paraboloidal moments of uniform flux = [1,0,0,0,0]
        mom = np.zeros((ng, self.nodes, 5))
        mom[:, :, 0] = 1.0
        qnode = np.zeros((ng, self.nodes, 5))
        for g in range(ng):
            for g2 in range(ng):
                qnode[g] += (self.Sgg[g, g2, matidx]
                             + self.chi[g] * self.nuSf[g2, matidx] / lam
                             )[:, None] * mom[g2]
        F_old = float(fission_rate.sum())
        for outer in range(max_outer):
            phi = np.zeros((ng, self.nodes))
            qim = np.zeros((ng, self.nodes, 5))
            for g in range(ng):
                phi[g], qim[g] = self._sweep(g, qnode[g])
            fission_rate = np.zeros(self.nodes)
            for g in range(ng):
                fission_rate += self.nuSf[g, matidx] * phi[g]
            F_new = float(fission_rate.sum())
            lam_new = lam * F_new / F_old
            dlam = abs(lam_new - lam) / lam_new
            lam = lam_new
            # paraboloidal source update (B.3), all groups:
            for g in range(ng):
                qnode[g] = np.zeros((self.nodes, 5))
                for g2 in range(ng):
                    qnode[g] += (self.Sgg[g, g2, matidx]
                                 + self.chi[g] * self.nuSf[g2, matidx] / lam
                                 )[:, None] * qim[g2]
            F_old = F_new
            if verbose and (outer % 5 == 0 or dlam < outer_tol):
                print(f"it {outer:4d}  keff={lam:.7f}  dkeff={dlam:.2e}  "
                      f"F={F_new:.5e}")
            if dlam < outer_tol:
                break
        return lam, phi, qnode

    # ------------------------------------------------------------------ #
    def face_currents_report(self, i, g, u, m=None):
        """Per-face surface flux / current for diagnostics."""
        out = {}
        for f in range(self.F):
            if self.faces[f]["type"] == "internal":
                n = self.faces[f]["a"]
                lf = self._local_face(n, f)
                st = self._node_state_at(i, g, n, 0 if m is None else m, u,
                                         np.zeros(5))
                out[f] = (st["Phi"][lf], st["J4check"][lf])
        return out
