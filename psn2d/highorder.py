# -*- coding: utf-8 -*-
"""High-order directional PSN closure (angular degree A).

Integration of the OpenPSN_directional_A3_upgrade package (Yuchao Xu,
2026-09-13).  Adds a user-specified azimuthal Legendre degree A inside
every azimuthal segment: K=A+1 angular unknowns per segment per face,
with a direction-matrix body closure (Dx, Dy, Dxy are K x K matrices).

Method identities (DIRECTIONAL_COMPATIBILITY.md):
  * legacy             - the untouched original path (solver.PSN2D.keff)
  * scalar_direction   - directional branch at A=0 (this module)
  * high_order_direction - directional branch at A>=1 (this module)
  * original + A>=0    - original-scale angular Galerkin diagnostic branch

Verified scope: axis-aligned conforming rectangular nodes, uniform A,
half-plane even M, isotropic sources, reflect and vacuum boundaries
(vacuum: J = B L with the K x K hemispheric current matrix B, BOUNDARY
package 2026-09-13, eq. B9; at A0 B00 = 2|alpha_face| recovers the
legacy vacuum row bit-for-bit).  Albedo, periodic, incident and
mixed-order interfaces are NOT implemented: the solver must reject or
route them to the legacy path.

The frozen kernels below are byte-identical to the package's
code/_frozen_core.py (sha256 bd4b72f6... per MANIFEST.json) and
code/angle_forms.py; BlockAngleSystem/BlockSweep are ported from
code/block_sweep.py with only import-path changes.  The keff driver here
reuses the production source-update / power-iteration framework (plain
iteration, no Dombey extrapolation) as required by the package.
"""
import multiprocessing as _mp
import time

import numpy as np
from scipy import sparse
from scipy.linalg import solve
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from .node import node_params_generic


# --------------------------------------------------------------------- #
# Frozen angular basis (package code/angle_forms.py, verbatim)
# --------------------------------------------------------------------- #
def basis_gauss(degree, n=None):
    if n is None:
        n = max(32, 2 * degree + 8)
    z, w = np.polynomial.legendre.leggauss(n)
    V = np.polynomial.legendre.legvander(z, degree) \
        * np.sqrt(2 * np.arange(degree + 1) + 1)
    return z, w / 2, V


def angular_forms(mu0, mu1, phi0, phi1, polar_degree, azimuth_degree):
    z, w, V = basis_gauss(polar_degree)
    mu = (mu0 + mu1) / 2 + (mu1 - mu0) * z / 2
    T = V.T @ ((w * (1 - mu * mu))[:, None] * V)
    z, w, V = basis_gauss(azimuth_degree)
    phi = (phi0 + phi1) / 2 + (phi1 - phi0) * z / 2
    C = V.T @ ((w * np.cos(phi) ** 2)[:, None] * V)
    S = V.T @ ((w * np.sin(phi) ** 2)[:, None] * V)
    H = V.T @ ((w * np.sin(phi) * np.cos(phi))[:, None] * V)
    return (T + T.T) / 2, (C + C.T) / 2, (S + S.T) / 2, (H + H.T) / 2


# --------------------------------------------------------------------- #
# Frozen local kernels (package code/_frozen_core.py, verbatim)
# --------------------------------------------------------------------- #
def local_response(pr, lo, hi, degree):
    """[4*K face flux moments; 5 angular-mean spatial moments] vs [4*K J;5 q].

    Original-scale branch: the legacy spatial closure with an azimuthal
    Galerkin extension.  At A=0 this recovers the original discrete
    system (legacy regression target).
    """
    K = degree + 1
    eye = np.eye(K)
    _, C, S, H = angular_forms(0., 1., lo, hi, 0, degree)
    X, Y, Z = 2 * C, 2 * S, 2 * H
    p, k, st, hx, hy = (pr[v] for v in ['p', 'k', 'St', 'hx', 'hy'])
    sx, cx, sy, cy = (pr[v] for v in ['sx', 'cx', 'sy', 'cy'])
    zero = np.zeros((K, K))
    mat = np.block([
        [-p * k * sx * X, -p * k * cx * X, zero, -p * 2 * sy / hy * Z],
        [-p * k * sx * X, +p * k * cx * X, zero, +p * 2 * sy / hy * Z],
        [zero, -p * 2 * sx / hx * Z, -p * k * sy * Y, -p * k * cy * Y],
        [zero, +p * 2 * sx / hx * Z, -p * k * sy * Y, +p * k * cy * Y]])
    off = np.zeros((4 * K, 5))
    x, y, z = X[:, 0], Y[:, 0], Z[:, 0]
    off[0 * K:1 * K, 1] = -p * 2 / (hx * st) * x
    off[0 * K:1 * K, 3] = -p * 6 / (hx * st) * x
    off[0 * K:1 * K, 2] = -p * 2 / (hy * st) * z
    off[1 * K:2 * K, 1] = +p * 2 / (hx * st) * x
    off[1 * K:2 * K, 3] = -p * 6 / (hx * st) * x
    off[1 * K:2 * K, 2] = +p * 2 / (hy * st) * z
    off[2 * K:3 * K, 2] = -p * 2 / (hy * st) * y
    off[2 * K:3 * K, 4] = -p * 6 / (hy * st) * y
    off[2 * K:3 * K, 1] = -p * 2 / (hx * st) * z
    off[3 * K:4 * K, 2] = +p * 2 / (hy * st) * y
    off[3 * K:4 * K, 4] = -p * 6 / (hy * st) * y
    off[3 * K:4 * K, 1] = +p * 2 / (hx * st) * z
    rhs = np.concatenate([np.eye(4 * K), -off], axis=1)
    coeff = solve(mat, rhs)
    residual = np.max(abs(mat @ coeff - rhs)) / max(np.max(abs(rhs)), 1.)
    nvar = 4 * K + 5
    mean = np.zeros((K, nvar))
    mean[:, 0 * K:1 * K] -= eye / (hx * st)
    mean[:, 1 * K:2 * K] -= eye / (hx * st)
    mean[:, 2 * K:3 * K] -= eye / (hy * st)
    mean[:, 3 * K:4 * K] -= eye / (hy * st)
    mean[0, 4 * K] = 1 / st
    flux = np.tile(mean, (4, 1))
    tx = 2 * sx / pr['ka_x']
    ty = 2 * sy / pr['ka_y']
    flux[0 * K:1 * K] += (cx - tx) * coeff[0 * K:1 * K] + sx * coeff[1 * K:2 * K]
    flux[1 * K:2 * K] += (cx - tx) * coeff[0 * K:1 * K] - sx * coeff[1 * K:2 * K]
    flux[2 * K:3 * K] += (cy - ty) * coeff[2 * K:3 * K] + sy * coeff[3 * K:4 * K]
    flux[3 * K:4 * K] += (cy - ty) * coeff[2 * K:3 * K] - sy * coeff[3 * K:4 * K]
    for face, sgn, lin, quad in [(0, 1, 1, 3), (1, -1, 1, 3),
                                 (2, 1, 2, 4), (3, -1, 2, 4)]:
        flux[face * K, 4 * K + lin] += sgn / st
        flux[face * K, 4 * K + quad] += 1 / st
    mom = np.vstack([mean[0], 3 * pr['Il_x'] * coeff[K],
                     3 * pr['Il_y'] * coeff[3 * K],
                     5 * pr['Iq_x'] * coeff[0], 5 * pr['Iq_y'] * coeff[2 * K]])
    for j in range(1, 5):
        mom[j, 4 * K + j] += 1 / st
    return flux, mom, float(residual)


def local_directional_response(pr, lo, hi, degree):
    """Direction-matrix body closure: same five spatial source terms and
    four face means, but the hyperbolic body scales are the K x K matrix
    spectra of Dx = 2pC, Dy = 2pS, Dxy = 2pH.

    Scaled hyperbolic functions avoid overflow for grazing directions.
    This is a body-closure change, not the original-k p extension.
    """
    K = degree + 1
    eye = np.eye(K)
    zero = np.zeros((K, K))
    st, hx, hy = (pr[x] for x in ['St', 'hx', 'hy'])
    _, C, S, H = angular_forms(0., 1., lo, hi, 0, degree)
    Dx = 2 * pr['p'] * C
    Dy = 2 * pr['p'] * S
    Dxy = 2 * pr['p'] * H

    def axis(D, h):
        eigv, V = np.linalg.eigh(D)
        assert np.min(eigv) > 0
        kval = np.sqrt(st / eigv)
        u = kval * h / 2
        t = np.tanh(u)
        avg = t / u
        Il = (1 - avg) / u
        Iq = (t - 3 * Il) / u
        small = abs(u) < 1e-3
        v = u[small]
        Il[small] = v / 3 - 2 * v ** 3 / 15 + 17 * v ** 5 / 315
        Iq[small] = v ** 2 / 15 - v ** 4 / 35 + 11 * v ** 6 / 945
        def mat(f):
            return (V * f) @ V.T
        return mat(t), mat(avg), mat(Il), mat(Iq), mat(eigv * kval * t), mat(eigv * kval)

    tx, avx, ilx, iqx, dxs, dxc = axis(Dx, hx)
    ty, avy, ily, iqy, dys, dyc = axis(Dy, hy)
    crossy = 2 / hy * Dxy @ ty
    crossx = 2 / hx * Dxy @ tx
    mat = np.block([[-dxs, -dxc, zero, -crossy],
                    [-dxs, +dxc, zero, +crossy],
                    [zero, -crossx, -dys, -dyc],
                    [zero, +crossx, -dys, +dyc]])
    off = np.zeros((4 * K, 5))
    x, y, z = Dx[:, 0], Dy[:, 0], Dxy[:, 0]
    off[0 * K:1 * K, 1] = -2 / (hx * st) * x
    off[0 * K:1 * K, 3] = -6 / (hx * st) * x
    off[0 * K:1 * K, 2] = -2 / (hy * st) * z
    off[1 * K:2 * K, 1] = +2 / (hx * st) * x
    off[1 * K:2 * K, 3] = -6 / (hx * st) * x
    off[1 * K:2 * K, 2] = +2 / (hy * st) * z
    off[2 * K:3 * K, 2] = -2 / (hy * st) * y
    off[2 * K:3 * K, 4] = -6 / (hy * st) * y
    off[2 * K:3 * K, 1] = -2 / (hx * st) * z
    off[3 * K:4 * K, 2] = +2 / (hy * st) * y
    off[3 * K:4 * K, 4] = -6 / (hy * st) * y
    off[3 * K:4 * K, 1] = +2 / (hx * st) * z
    rhs = np.concatenate([np.eye(4 * K), -off], axis=1)
    coeff = solve(mat, rhs)
    residual = np.max(abs(mat @ coeff - rhs)) / max(np.max(abs(rhs)), 1.)
    mean = np.zeros((K, 4 * K + 5))
    for f, h in [(0, hx), (1, hx), (2, hy), (3, hy)]:
        mean[:, f * K:(f + 1) * K] -= eye / (h * st)
    mean[0, 4 * K] = 1 / st
    flux = np.tile(mean, (4, 1))
    ax, bx, ay, by = np.split(coeff, 4, axis=0)
    flux[0 * K:1 * K] += (eye - avx) @ ax + tx @ bx
    flux[1 * K:2 * K] += (eye - avx) @ ax - tx @ bx
    flux[2 * K:3 * K] += (eye - avy) @ ay + ty @ by
    flux[3 * K:4 * K] += (eye - avy) @ ay - ty @ by
    for face, sgn, lin, quad in [(0, 1, 1, 3), (1, -1, 1, 3),
                                 (2, 1, 2, 4), (3, -1, 2, 4)]:
        flux[face * K, 4 * K + lin] += sgn / st
        flux[face * K, 4 * K + quad] += 1 / st
    mom = np.vstack([mean[0], (3 * ilx @ bx)[0], (3 * ily @ by)[0],
                     (5 * iqx @ ax)[0], (5 * iqy @ ay)[0]])
    for j in range(1, 5):
        mom[j, 4 * K + j] += 1 / st
    return flux, mom, float(residual)


# --------------------------------------------------------------------- #
# Sparse five-moment sweep (port of package code/block_sweep.py)
# --------------------------------------------------------------------- #
class BlockAngleSystem:
    """One cached (polar index, energy group) face-current system.

    ``apply(q)`` takes q[node, (0,1x,1y,2x,2y)] and returns the five
    spatial flux moments, already weighted by W[i]/M and summed over M.
    Sum these returns over polar index exactly once for a group sweep.
    The topology and material arrays must stay immutable during its life.
    """

    def __init__(self, psn, i: int, g: int, A: int = 3,
                 branch: str = "directional", local_kernel=None):
        if isinstance(A, bool) or not isinstance(A, (int, np.integer)) or A < 0:
            raise ValueError("A must be a nonnegative integer")
        if branch not in ("directional", "original"):
            raise ValueError("branch must be directional or original")
        if not psn.rect or psn.full_2pi or psn.M < 2 or psn.M % 2:
            raise ValueError("Need rectangular nodes, half-plane even M")
        bad = {f["type"] for f in psn.faces} - {"internal", "reflect",
                                                "vacuum"}
        if bad:
            raise NotImplementedError(
                "High-order closure is validated for reflect and vacuum "
                f"boundaries only; unsupported types: {sorted(bad)}")
        if not (0 <= i < psn.I and 0 <= g < psn.ng):
            raise ValueError("Polar or energy-group index out of range")
        if not hasattr(psn, "_groups"):
            psn._vectorize_setup()
        self.p, self.i, self.g, self.A = psn, i, g, A
        self.K, self.n, self.branch = A + 1, psn.nodes, branch
        self.parity = (-1.0) ** np.arange(self.K)
        self.Bmat = self._vacuum_B(psn, i)
        self.kernel = local_kernel or (local_directional_response
                                       if branch == "directional"
                                       else local_response)
        self.local = {}
        self.local_residual = 0.0
        self.last_face_residual = None
        self.cols = np.empty((psn.M, self.n, 4 * self.K), dtype=np.int64)
        self.signs = np.empty_like(self.cols, dtype=float)
        for m in range(psn.M):
            for grp in psn._groups:
                n = psn._node_of_grp[grp]
                pr, _ = psn._pr_alpha_rect(i, g, m, psn.mat_of_node[n], n)
                L, V, residual = self.kernel(pr, m * psn.dphi,
                                             (m + 1) * psn.dphi, A)
                if L.shape != (4 * self.K, 4 * self.K + 5):
                    raise ValueError("Local L has wrong shape")
                if V.shape != (5, 4 * self.K + 5):
                    raise ValueError("Local V has wrong shape")
                self.local[m, grp] = (L, V)
                self.local_residual = max(self.local_residual, residual)
            for n in range(self.n):
                for lf, (face, outward_sign) in enumerate(psn.node_fmap[n]):
                    c, eliminated_sign = psn._col_of(face, m)
                    block = slice(lf * self.K, (lf + 1) * self.K)
                    self.cols[m, n, block] = c * self.K + np.arange(self.K)
                    reflect_modes = self.parity if psn.colidx[face, m] < 0 \
                        else 1.0
                    self.signs[m, n, block] = outward_sign * eliminated_sign \
                        * reflect_modes
        self._assemble()

    def _vacuum_B(self, psn, i):
        """Vacuum hemispheric current matrix B for polar index i.

        Block order (m, a): B[(m*K+a), (m*K+b)] =
        kappa * <b_a b_b |cos phi|> on x faces, |sin phi| on y faces
        (BOUNDARY package eq. B4; block-diagonal over polar segments
        because there is one polar index per system here).  Only needed
        when the topology has vacuum faces.  At A=0 B00 = 2|alpha_face|,
        exactly the legacy vacuum row coefficient (checked bit-for-bit
        in tools_ho/check_b_matrix.py).
        """
        if not any(f["type"] == "vacuum" for f in psn.faces):
            return None
        K, M = self.K, psn.M
        if psn.generic:
            kappa = node_params_generic(1.0, 1.0, psn.theta[i],
                                        psn.dtheta)["alpha_polar"]
        else:
            kappa = float(psn.mu[i])
        z, qw, V = basis_gauss(K - 1)
        blocks = {}
        for axis in ("x", "y"):
            mats = []
            for m in range(M):
                phi = (m + 0.5 + 0.5 * z) * (np.pi / M)
                normal = np.abs(np.cos(phi) if axis == "x"
                                else np.sin(phi))
                block = V.T @ ((qw * normal)[:, None] * V)
                mats.append(kappa * block)
            blocks[axis] = sparse.block_diag(mats, format="csc")
        return blocks

    def _assemble(self):
        p, K = self.p, self.K
        ar, ac, av, br, bc, bv = [], [], [], [], [], []
        row = 0

        def term(n, m, lf, a, scale):
            grp = (p.mat_of_node[n], p.shape_of_node[n])
            L, _ = self.local[m, grp]
            current = scale * L[lf * K + a, :4 * K] * self.signs[m, n]
            ar.extend([row] * (4 * K))
            ac.extend(self.cols[m, n])
            av.extend(current)
            br.extend([row] * 5)
            bc.extend(n * 5 + np.arange(5))
            bv.extend(-scale * L[lf * K + a, 4 * K:])

        def vacuum_row(n, m, lf, a, axis):
            """J_out(m,a) = B[(m,a),(m,b)] . L_face(m,b)  (eq. B9).

            L_face(m,b) is the same row-linear-combination term() uses,
            so at A=0 the row is the legacy vacuum row (J = B00 L,
            B00 = 2|alpha_face|) up to sign convention.
            """
            grp = (p.mat_of_node[n], p.shape_of_node[n])
            L, _ = self.local[m, grp]
            for b in range(K):
                scale = self.Bmat[axis][(m * K + a), m * K + b]
                if scale == 0.0:
                    continue
                current = scale * L[lf * K + b, :4 * K] * self.signs[m, n]
                ar.extend([row] * (4 * K))
                ac.extend(self.cols[m, n])
                av.extend(current)
                br.extend([row] * 5)
                bc.extend(n * 5 + np.arange(5))
                bv.extend(-scale * L[lf * K + b, 4 * K:])
            c, es = p._col_of(f_vac, m)
            outward = p.node_fmap[n][lf][1]
            ar.append(row)
            ac.append(c * K + a)
            av.append(-outward * es)
            br.append(row)
            bc.append(n * 5)
            bv.append(0.0)

        for f, face in enumerate(p.faces):
            if face["type"] == "internal":
                na, nb = face["a"], face["b"]
                fa, fb = p._local_face(na, f), p._local_face(nb, f)
                for m in range(p.M):
                    for a in range(K):
                        term(na, m, fa, a, 1.0)
                        term(nb, m, fb, a, -1.0)
                        row += 1
            elif face["type"] == "vacuum":
                f_vac = f
                n = face["a"]
                lf = p._local_face(n, f)
                axis = face["axis"]
                for m in range(p.M):
                    for a in range(K):
                        vacuum_row(n, m, lf, a, axis)
                        row += 1
            else:
                n = face["a"]
                lf = p._local_face(n, f)
                for ma, mb in p.face_pairs[f]:
                    for a in range(K):
                        term(n, ma, lf, a, 1.0)
                        term(n, mb, lf, a, -self.parity[a])
                        row += 1
        expected = p.ncol * K
        if row != expected:
            raise ValueError(f"Face equation count {row} != unknown count {expected}")
        self.matrix = coo_matrix((av, (ar, ac)), shape=(row, row)).tocsc()
        self.source_map = coo_matrix((bv, (br, bc)),
                                     shape=(row, self.n * 5)).tocsr()
        self.matrix.sum_duplicates()
        self.matrix.eliminate_zeros()
        self.source_map.sum_duplicates()
        self.source_map.eliminate_zeros()
        self.factor = splu(self.matrix, permc_spec="COLAMD")

    def apply(self, q, check_residual=True):
        q = np.asarray(q, dtype=float)
        if q.shape != (self.n, 5) or not np.all(np.isfinite(q)):
            raise ValueError(f"q must be finite with shape ({self.n}, 5)")
        rhs = self.source_map @ q.ravel()
        current = self.factor.solve(rhs)
        if check_residual:
            self.last_face_residual = float(
                np.linalg.norm(self.matrix @ current - rhs)
                / max(np.linalg.norm(rhs), np.finfo(float).tiny))
        out = np.zeros_like(q)
        for m in range(self.p.M):
            for grp in self.p._groups:
                sel = self.p._grp_sel[grp]
                _, V = self.local[m, grp]
                local_current = self.signs[m, sel] * current[self.cols[m, sel]]
                out[sel] += (local_current @ V[:, :4 * self.K].T
                             + q[sel] @ V[:, 4 * self.K:].T)
        return self.p.W[self.i] / self.p.M * out

    def diagnostics(self):
        return dict(A=self.A, K=self.K, i=self.i, g=self.g,
                    local_kernel_name=self.kernel.__name__,
                    face_unknowns=self.matrix.shape[0],
                    matrix_nnz=self.matrix.nnz,
                    source_map_nnz=self.source_map.nnz,
                    factor_nnz=self.factor.L.nnz + self.factor.U.nnz,
                    local_residual=float(self.local_residual),
                    face_residual=self.last_face_residual)


class BlockSweep:
    """Lazy reusable polar/group factors; source assembly stays with PSN2D.

    Cache is valid for this immutable topology/XS/angular configuration
    only.  Plain sparse LU; the scalar Cholesky/Schur shared-factor
    backends are NOT applicable for A > 0 and must not be used.
    """

    def __init__(self, psn, A=3, branch="directional"):
        self.psn, self.A, self.branch = psn, A, branch
        self.systems = {}

    def system(self, i, g):
        key = (i, g)
        if key not in self.systems:
            self.systems[key] = BlockAngleSystem(self.psn, i, g,
                                                 self.A, self.branch)
        return self.systems[key]

    def group(self, g, q, check_residual=True):
        return sum(self.system(i, g).apply(q, check_residual)
                   for i in range(self.psn.I))

    def all_groups(self, q, check_residual=True):
        q = np.asarray(q, float)
        if q.shape != (self.psn.ng, self.psn.nodes, 5):
            raise ValueError("q shape must be (ng, nodes, 5)")
        moments = np.stack([self.group(g, q[g], check_residual)
                            for g in range(self.psn.ng)])
        return moments[:, :, 0], moments

    def diagnostics(self):
        out = {}
        for (i, g), sys in self.systems.items():
            out[f"i{i}_g{g}"] = sys.diagnostics()
        return out

    def clear(self):
        self.systems.clear()


# --------------------------------------------------------------------- #
# Parallel sweep (fork + COW, mirrors the legacy process-pool design)
# --------------------------------------------------------------------- #
_HO_SW = None   # registry inherited by forked workers


def _ho_prefactor_job(hs, per_worker, i, g):
    """Prefactor one (i, g) system inside a pool worker with a capped
    BLAS budget (same pattern as solver._prefactor_job)."""
    from . import runtime
    prev = runtime.get_thread_limit()
    runtime.set_thread_limit(per_worker)
    try:
        hs.sweep.system(i, g)
    finally:
        runtime.set_thread_limit(prev)


def _ho_sweep_worker(args):
    """One (g, i) job: back-substitute the cached factor for that system.

    Reads the source from shared memory, writes the (nodes, 5) reduce to
    its own slot.  Workers must never populate or mutate shared caches
    (apply with check_residual=False to keep the inherited factor objects
    untouched apart from the private last_face_residual slot).
    """
    import multiprocessing.shared_memory as shm_mod
    g, i, qname, rname, ng, I, nodes = args
    q_shm = shm_mod.SharedMemory(name=qname)
    r_shm = shm_mod.SharedMemory(name=rname)
    try:
        q = np.frombuffer(q_shm.buf, dtype=np.float64) \
            .reshape(ng, nodes, 5)
        r = np.frombuffer(r_shm.buf, dtype=np.float64) \
            .reshape(ng, I, nodes, 5)
        out = _HO_SW.sweep.system(i, g).apply(q[g], check_residual=False)
        r[g, i] = out
    finally:
        # drop EVERY numpy view of the shared buffers before closing them
        # (a live view = an exported pointer -> BufferError on close);
        # explicit None-assignment releases the frame slots
        # deterministically (same discipline as solver._sweep_worker).
        q = None
        r = None
        out = None
        q_shm.close()
        r_shm.close()
    return (g, i)


class HighOrderSweep:
    """BlockSweep wrapper with an optional forked process-pool backend.

    Serial path (small problems): lazy BlockSweep factors, exact
    production order.  Parallel path: after ``prefactor()`` has built
    every (i, g) face system, a forked pool back-substitutes in workers
    (SuperLU's gssv does not release the GIL in this scipy build — same
    measurement as the legacy path, see solver.py).  Sources go in and
    per-job reduces go out through POSIX shared memory; the parent
    accumulates in the exact serial (g, i) order -> bit-identical.
    """

    # same effective unknown-count threshold as the legacy path
    _MIN_NCOL = 150000

    def __init__(self, psn, A=3, branch="directional"):
        self.psn = psn
        self.sweep = BlockSweep(psn, A=A, branch=branch)
        self._pool = None
        self._q_shm = None
        self._r_shm = None

    # ------------------------------ serial ----------------------------
    def all_groups_serial(self, qnode):
        return self.sweep.all_groups(qnode)

    # ------------------------------ pool ------------------------------
    def prefactor(self):
        """Build and factorize every (i, g) system up front.

        SuperLU's factorization releases the GIL (legacy measurement
        ~19x at 6 threads), so factorization overlaps on a small thread
        pool with the same budget rule as the legacy _prefactor:
        W workers x (threads // W) BLAS threads <= psn.threads."""
        psn = self.psn
        jobs = [(g, i) for g in range(psn.ng) for i in range(psn.I)]
        n = len(jobs)
        if n > 1:
            pf = max(1, min(6, psn.threads))
            per_worker = max(1, psn.threads // pf)
            from . import runtime
            from concurrent.futures import ThreadPoolExecutor
            pool = ThreadPoolExecutor(max_workers=pf)
            try:
                futs = [pool.submit(_ho_prefactor_job, self, per_worker,
                                    i, g) for g, i in jobs]
                for f in futs:
                    f.result()
            finally:
                pool.shutdown(wait=True)
            runtime.set_thread_limit(psn.threads)   # restore parent budget
        else:
            self.sweep.system(jobs[0][1], jobs[0][0])

    def _pool_ok(self):
        return self.psn.ncol * (self.sweep.A + 1) >= self._MIN_NCOL

    def start_pool(self):
        """Fork the worker pool AFTER every factor exists (COW-safe)."""
        global _HO_SW
        if self._pool is not None or not self._pool_ok():
            return
        from . import solver as _s
        psn = self.psn
        n = psn.ng * psn.I
        p = _s._pool_size(n)
        p = min(p, psn.threads)   # strict budget (user input 2026-09-12)
        if p <= 1:
            return
        q_size = psn.ng * psn.nodes * 5 * 8
        r_size = psn.ng * psn.I * psn.nodes * 5 * 8
        free = _s._shm_free_bytes()
        if q_size + r_size > free:
            raise MemoryError(
                f"high-order sweep pool needs "
                f"{(q_size + r_size) / 2**20:.1f} MiB of /dev/shm but only "
                f"{free / 2**20:.1f} MiB is free — enlarge the shm tmpfs "
                f"or run serial (PSN_PAR=1)")
        self._q_shm = _s._SharedBuf(q_size)
        self._r_shm = _s._SharedBuf(r_size)
        _HO_SW = self
        ctx = _mp.get_context("fork")
        self._pool = ctx.Pool(processes=p)

    def stop_pool(self):
        global _HO_SW
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
        for sh in (self._q_shm, self._r_shm):
            if sh is not None:
                sh.close()
                sh.unlink()
        self._q_shm = None
        self._r_shm = None
        _HO_SW = None

    # ------------------------------ dispatch --------------------------
    def all_groups(self, qnode):
        if self._pool is None:
            return self.all_groups_serial(qnode)
        psn = self.psn
        ng, I_, nodes = psn.ng, psn.I, psn.nodes
        q_view = np.frombuffer(self._q_shm.buf, dtype=np.float64).reshape(-1)
        q_view[:] = np.ascontiguousarray(qnode).ravel()
        args = [(g, i, self._q_shm.name, self._r_shm.name,
                 ng, I_, nodes) for g in range(ng) for i in range(I_)]
        try:
            self._pool.map_async(_ho_sweep_worker, args).get()
            out = np.frombuffer(self._r_shm.buf, dtype=np.float64)
            out = out.reshape(ng, I_, nodes, 5)
            phi = np.zeros((ng, nodes))
            qim = np.zeros((ng, nodes, 5))
            for g in range(ng):
                for i in range(I_):
                    o = out[g, i]
                    phi[g] += o[:, 0]
                    qim[g] += o
            return phi, qim
        finally:
            # release the shared-buffer views: on an exception this frame
            # stays on the traceback, and a live view is an exported
            # pointer that would make stop_pool()'s close() raise
            q_view = None
            out = None
            o = None


# --------------------------------------------------------------------- #
# High-order keff driver (production power iteration, plain sweeps)
# --------------------------------------------------------------------- #
def _validate_A(A):
    if isinstance(A, bool) or not isinstance(A, (int, np.integer)) or A < 0:
        raise ValueError(f"angular.A must be a nonnegative integer, got {A!r}")
    return int(A)


def keff_highorder(psn, A=3, branch="directional", max_outer=4000,
                   outer_tol=1e-10, verbose=False):
    """Plain power iteration with the high-order angular sweep.

    Reuses PSN2D._source_update and the production convergence bookkeeping
    (fission-rate ratio), with NO Dombey/Richardson extrapolation (the
    package requires the acceptance trajectory to be the plain one).
    Convergence requires BOTH dlam < outer_tol and the five-moment
    source fixed-point residual < 1e-9; reaching max_outer without both
    is reported as non-converged, never returned silently.

    Returns (lam, phi, qim, info).
    """
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
        # pre-warm on the main thread: the threaded prefactor below
        # would otherwise race on the first _vectorize_setup
        psn._vectorize_setup()
    sweep = HighOrderSweep(psn, A=A, branch=branch)

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
    qim = None
    phi = None
    s_next = None
    converged = False
    dlam = 1.0
    src_res = None
    warm = True
    try:
        for outer in range(max_outer):
            if warm:
                # serial warm sweep: builds every (i,g) factor (COW-safe
                # fork afterwards), exact production order
                sweep.prefactor()
                phi, qim = sweep.all_groups_serial(qnode)
                sweep.start_pool()
                warm = False
            else:
                phi, qim = sweep.all_groups(qnode)
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
                print(f"ho it {outer:4d}  keff={lam:.7f}  dkeff={dlam:.2e}  "
                      f"src_res={src_res:.2e}")
            if dlam < outer_tol and src_res < 1e-9:
                converged = True
                break
    finally:
        sweep.stop_pool()

    diag = sweep.sweep.diagnostics()
    max_local_res = max((d["local_residual"] for d in diag.values()),
                        default=0.0)
    max_face_res = max((d["face_residual"] for d in diag.values()
                        if d["face_residual"] is not None), default=None)
    info = dict(A=int(A), K=A + 1, branch=branch, converged=converged,
                outer_iterations=outer + 1,
                keff_residual=dlam, source_residual=float(src_res),
                max_local_kernel_residual=max_local_res,
                max_face_system_residual=max_face_res,
                seconds=round(time.time() - t0, 2))
    if not converged:
        info["warning"] = (f"NOT converged after {max_outer} outer "
                           f"iterations (dlam={dlam:.2e}, "
                           f"src_res={src_res:.2e})")
    return lam, phi, qim, info


def compare_A0_original_state(psn, A=0, branch="original"):
    """A0 regression (DIRECTIONAL_COMPATIBILITY 4.1): original-scale
    local_response(A=0) vs the untouched legacy 9x9 state matrix for
    every (i, g, m, group).  Returns the max relative difference."""
    if A != 0:
        raise ValueError("A0 regression requires A=0")
    if branch != "original":
        raise ValueError("A0 regression uses the original-scale branch")
    if not psn.rect:
        raise ValueError("rectangular topology required")
    if not hasattr(psn, "_groups"):
        psn._vectorize_setup()
    worst = 0.0
    for i in range(psn.I):
        for g in range(psn.ng):
            for m in range(psn.M):
                for grp in psn._groups:
                    n0 = psn._node_of_grp[grp]
                    pr, _ = psn._pr_alpha_rect(i, g, m,
                                               psn.mat_of_node[n0], n0)
                    L, V, _ = local_response(pr, m * psn.dphi,
                                             (m + 1) * psn.dphi, A)
                    combo = np.vstack([L, V])           # (9, 9)
                    orig = psn._state_matrix(i, g, m, grp)
                    d = np.max(abs(combo - orig)) / \
                        max(np.max(abs(orig)), 1.0)
                    worst = max(worst, float(d))
    return worst
