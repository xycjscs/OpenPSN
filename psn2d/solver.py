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
import os
import sys
import time

# Note: OpenBLAS/OMP are pinned to 1 thread in psn2d/__init__.py (imported
# before this module) — see the determinism guard there.

import mmap
import multiprocessing as _mp
import multiprocessing.shared_memory as _shm
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from .node import (node_alphas, node_alphas_generic, node_params_generic,
                      node_state, node_response_matrix, ty_polar_set,
                      generic_polar_set)
from .node_rect import (node_params_rect, node_params_rect_generic,
                        node_state_rect, node_response_matrix_rect)

PI = np.pi


def _cgroup_cpu_quota():
    """Effective CPU count: cgroup v2 quota if set, else os.cpu_count()."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


def _pool_size(njobs):
    """Sweep parallelism.  Default: half the available cores; override with
    PSN_PAR (1 = serial, 0/absent = auto).  Never exceeds the job count."""
    try:
        p = int(os.environ.get("PSN_PAR", ""))
    except ValueError:
        p = 0
    if p <= 0:
        p = max(1, _cgroup_cpu_quota() // 2)
    return max(1, min(njobs, p))


# Module handle to the currently-active PSN2D object; set by the parent
# right before forking the sweep worker pool.  Forked children inherit it
# read-only via copy-on-write (LU factors, state-matrix cache, b-term
# tables are all built before the fork and never mutated afterwards).
_SWEET_PSN = None


def _shm_free_bytes():
    """Free bytes on the POSIX-shm tmpfs (usually /dev/shm)."""
    try:
        st = os.statvfs("/dev/shm")
        return st.f_bavail * st.f_frsize
    except OSError:
        return 0


class _MmapBuf:
    """Parent-side handle around an open MAP_SHARED file mapping."""
    __slots__ = ("_fd", "_mm")

    def __init__(self, fd, mm):
        self._fd, self._mm = fd, mm

    @property
    def buf(self):
        return self._mm

    def close(self):
        try:
            self._mm.close()
        finally:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None


class _SharedBuf:
    """Shared byte buffer for the process-pool sweep.

    POSIX shm (``psm_*`` on /dev/shm) when the segment fits in the tmpfs;
    otherwise a file under ``$PSN_SWEEP_SHM_DIR`` (default /tmp) mmap'ed
    MAP_SHARED.  On this host /dev/shm is a 64 MiB tmpfs while core
    S>=5 sweep segments need 70-100 MiB: an over-cap shm segment SIGBUSes
    the worker on the first write past the limit and pool.map then hangs
    forever (C5G7 core M2_S6 deadlock, 2026-09-10).  The file path is
    byte-for-byte the same memory model (shared pages, buffer protocol),
    so the sweep result is unchanged.
    """
    __slots__ = ("name", "size", "_sm", "_mm", "_fd", "_path")

    def __init__(self, size, force_file=False):
        self.size = size
        self._sm = None
        self._mm = None
        self._fd = None
        self._path = None
        self.name = ""
        forced_file = force_file or os.environ.get("PSN_SWEEP_SHM", "1") == "0"
        if not forced_file and size <= _shm_free_bytes():
            try:
                sm = _shm.SharedMemory(create=True, size=size)
                self._sm = sm
                self.name = sm.name
                return
            except OSError:
                pass   # tmpfs full at alloc time -> fall through to /tmp
        d = os.environ.get("PSN_SWEEP_SHM_DIR", "/tmp")
        fd, path = tempfile.mkstemp(prefix="psn_sweep_", dir=d)
        try:
            os.ftruncate(fd, size)
            self._mm = mmap.mmap(fd, size, flags=mmap.MAP_SHARED)
        except Exception:
            os.close(fd)
            try:
                os.unlink(path)
            except OSError:
                pass
            raise
        self._fd = fd
        self._path = path
        self.name = path

    @property
    def buf(self):
        return self._sm.buf if self._sm is not None else self._mm

    def close(self):
        # Note: does NOT clear _sm — unlink() must still see it (the
        # sweep pool does close(); unlink(); and a cleared _sm would
        # silently skip the shm_unlink, leaking the 64 MiB /dev/shm).
        if self._sm is not None:
            self._sm.close()
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def unlink(self):
        if self._sm is not None:
            self._sm.unlink()
            self._sm = None
        if self._path is not None:
            try:
                os.unlink(self._path)
            except OSError:
                pass
            self._path = None


def _open_buf(name):
    """Worker side: open a shared sweep buffer by the name string the
    parent put in the job args ('psm_*' -> POSIX shm, absolute path ->
    /tmp file mapping)."""
    if os.path.isabs(name):
        fd = os.open(name, os.O_RDWR)
        mm = mmap.mmap(fd, 0, flags=mmap.MAP_SHARED)
        return _MmapBuf(fd, mm)
    return _shm.SharedMemory(name=name)


def _sweep_worker(args):
    """Forked sweep worker: one (i, g) direction, end to end.

    Runs in a child process (its own GIL — the whole reason for processes;
    the scipy build in use does NOT release the GIL around SuperLU's
    gssv back-substitution, so a thread pool serializes on the lock and
    MEASURES SLOWER than serial on the 21-job C5G7 core).  The source
    vector is read from, and the per-job reduce is written to, POSIX
    shared memory: no pickling of large arrays.  The local reduce below
    uses the SAME expression order as the serial path of ``_sweep_all``,
    so the parent's accumulation is bit-identical."""
    (g, i), qname, rname, ng, I_, nodes, w = args
    qs = _open_buf(qname)
    rs = _open_buf(rname)
    try:
        qnode = np.frombuffer(qs.buf, dtype=np.float64).reshape(ng, nodes, 5)
        q = qnode[g].copy()
        out = np.frombuffer(rs.buf, dtype=np.float64).reshape(ng, I_, nodes, 5)
        psn = _SWEET_PSN
        M_ = psn.M
        raw = np.empty((nodes, M_, 4))
        for m in range(M_):
            for grp in psn._groups:
                S = psn._state_matrix(i, g, m, grp)
                sel = psn._grp_sel[grp]
                raw[sel, m, :] = q[sel] @ S[:4, 4:].T
        b = np.zeros(psn.ncol)
        for (m, lf), (rws, sgns, nds) in psn._bterms_for(i, g).items():
            np.add.at(b, rws, sgns * raw[nds, m, lf])
        lu, _ = psn._lusys[(i, g)]
        u = lu.solve(b)
        Y = np.empty((nodes, M_, 9))
        J4 = np.empty((nodes, M_, 4))
        for m in range(M_):
            for j in range(4):
                J4[:, m, j] = psn._esgn[m, j] * u[psn._cidx[m, j]]
        X = np.empty((nodes, M_, 9))
        X[:, :, :4] = J4
        X[:, :, 4:] = q[:, None, :]
        for m in range(M_):
            for grp in psn._groups:
                S = psn._state_matrix(i, g, m, grp)
                sel = psn._grp_sel[grp]
                Y[sel, m, :] = X[sel, m, :] @ S.T
        o = out[g, i]
        o[:, 0] = w * Y[:, :, 4].sum(axis=1)
        o[:, 1:] = w * Y[:, :, 5:].sum(axis=1)
    finally:
        # drop EVERY numpy view of the shared buffers before closing them
        # (a live view = an exported pointer -> BufferError on close);
        # explicit None-assignment releases the frame slots deterministically
        qnode = None
        out = None
        o = None
        qs.close()
        rs.close()
    return (g, i)


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
                 generic=False, I=30, widths=None):
        if M % 2 != 0:
            raise ValueError("M must be even (mirror pairs need paired segments)")
        self.mat = mat_map.astype(np.int64)
        self.h = h
        self.nx, self.ny = mat_map.shape[1], mat_map.shape[0]
        # rectangular nodes: widths = (hx[nd], hy[nd]) per-node arrays.
        # scalar/None -> square path (unchanged, bit-identical).
        if widths is None:
            self.rect = False
            self.hx = self.hy = None
            self.area = None
        else:
            hx, hy = np.asarray(widths[0], float), np.asarray(widths[1], float)
            if hx.shape != mat_map.shape or hy.shape != mat_map.shape:
                raise ValueError("widths arrays must match mat_map shape")
            if (hx <= 0).any() or (hy <= 0).any():
                raise ValueError("widths must be positive")
            if not (np.allclose(hx, hx.ravel()[0])
                    and np.allclose(hy, hy.ravel()[0])):
                self.rect = True
            else:
                # uniform rectangles: still use the rect closed forms
                # (a true generalization) but with one shape class.
                self.rect = True
            self.hx = hx
            self.hy = hy
            self.area = hx * hy
            # shape classes: rounded (hx, hy) tuples -> per-node class id
            rx = np.round(hx, 9)
            ry = np.round(hy, 9)
            self._shape_keys = {}
            key_of = []
            for v in zip(rx.ravel(), ry.ravel()):
                if v not in self._shape_keys:
                    self._shape_keys[v] = len(self._shape_keys)
                key_of.append(self._shape_keys[v])
            self.shape_id = np.array(key_of, np.int64).reshape(mat_map.shape)
            self.nshape = len(self._shape_keys)
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
        # process-pool sweep (see _start_sweep_pool / _sweep_all)
        self._proc_pool = None
        self._q_shm = None
        self._r_shm = None

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

    def _R(self, i, g, m, mat, n=None):
        if self.rect:
            # rectangular: R depends on the node's shape class
            sh = int(self.shape_id[self.j_idx[n], self.i_idx[n]])
            k = (i, g, m, mat, sh)
            if k not in self._Rcache:
                pr, alpha = self._pr_alpha_rect(i, g, m, mat, n)
                R, _ = node_response_matrix_rect(pr, self._phi_m(m),
                                                 self.dphi, alpha)
                self._Rcache[k] = R
            return self._Rcache[k]
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
        if self.rect:
            pr, alpha = self._pr_alpha_rect(i, g, m, mat, n)
            return node_state_rect(J4, qn, pr, self._phi_m(m), self.dphi, alpha)
        if self.generic:
            return node_state(J4, qn, self.St[g, mat], 1.0, self.h,
                              self._phi_m(m), self.dphi,
                              pr=self._pr_generic(i, g, mat),
                              alpha=self._alpha_for(i, m))
        return node_state(J4, qn, self.St[g, mat], self.mu[i],
                          self.h, self._phi_m(m), self.dphi)

    def _pr_alpha_rect(self, i, g, m, mat, n):
        """Rectangular node params + alpha for node n (its hx, hy)."""
        hx = float(self.hx[self.j_idx[n], self.i_idx[n]])
        hy = float(self.hy[self.j_idx[n], self.i_idx[n]])
        if self.generic:
            pr = node_params_rect_generic(self.St[g, mat], hx, hy,
                                          self.theta[i], self.dtheta)
            alpha = node_alphas_generic(self.theta[i], self.dtheta, hx,
                                        self._phi_m(m), self.dphi)
        else:
            pr = node_params_rect(self.St[g, mat], self.mu[i], hx, hy)
            alpha = node_alphas(self.mu[i], 1.0, self._phi_m(m), self.dphi)
        return pr, alpha

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
                    RA = self._R(i, g, m, matA, A_n)
                    RB = self._R(i, g, m, matB, B_n)
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
                    R = self._R(i, g, m, mat, n)
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
                    Ra = self._R(i, g, a, mat, n)
                    Rb = self._R(i, g, b, mat, n)
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
            # COLAMD: for the C5G7 rect-core systems this cuts the LU fill
            # from ~205x to ~31x (factor 8.6 GB -> 1.3 GB per (i,g) system,
            # 21 min -> 23 s of factorization; see c5g7/_probe_ordering.log).
            lu = splu(csc_matrix(A), permc_spec="COLAMD")
            # The factor is self-contained (splu holds no reference to the
            # input matrix), so drop the sparse A (~90 MB per system).
            self._syscache.pop(key, None)
            self._lusys[key] = (lu, rows_rhs)
        return self._lusys[key]

    def _raw_flux(self, i, g, n, m, qn):
        """(1/alpha)Phi for node n seg m given source qn, J4=0 (the s part)."""
        st = self._state(np.zeros(4), qn, i, g, n, m)
        return st["Phibar"] / st["alpha"]

    # ------------------------------------------------------------------ #
    def _vectorize_setup(self):
        """One-off tables for the vectorized node-state path."""
        self.mat_of_node = self.mat[self.j_idx, self.i_idx]
        if self.rect:
            self.shape_of_node = self.shape_id.ravel()
            # groups = (mat, shape) pairs present in the model
            keys = set(zip(self.mat_of_node.tolist(),
                           self.shape_of_node.tolist()))
            self._groups = sorted(keys)
            self._grp_sel = {}
            self._node_of_grp = {}
            for (mat, sh) in self._groups:
                sel = (self.mat_of_node == mat) & (self.shape_of_node == sh)
                self._grp_sel[(mat, sh)] = sel
                self._node_of_grp[(mat, sh)] = int(np.argmax(sel))
            # representative widths per shape class
            self._shape_dims = {}
            for sh in range(self.nshape):
                n0 = int(np.argmax(self.shape_of_node == sh))
                self._shape_dims[sh] = (float(self.hx.ravel()[n0]),
                                        float(self.hy.ravel()[n0]))
        else:
            self.shape_of_node = None
            self._groups = [(m, 0) for m in sorted(set(self.mat_of_node.tolist()))]
            self._grp_sel = {(m, 0): (self.mat_of_node == m) for m, _ in self._groups}
            self._node_of_grp = {(m, 0): int(np.argmax(self._grp_sel[(m, 0)]))
                                 for m, _ in self._groups}
            self._shape_dims = {0: (float(self.h), float(self.h))}
        self._mats_present = sorted(set(self.mat_of_node.tolist()))
        self._mat_sel = {m: (self.mat_of_node == m) for m in self._mats_present}
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

    def _state_matrix(self, i, g, m, grp):
        """9x9 linear map  [Phi(4); phi_bar; mom[1..4]]  =  S @ [J4(4); q(5)]
        for direction (i,m), group g, (material, shape) group.
        Probed with the VALIDATED node_state on a reference node of that
        group — no re-derived algebra."""
        k = (i, g, m, grp)
        if k in self._Scache:
            return self._Scache[k]
        n0 = self._node_of_grp[grp]
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

    def _sweep_one(self, i, g, qnode):
        """LU back-substitution + node response for ONE (i, g) direction.
        Thread-safe: reads only shared tables, writes local arrays only."""
        u = self.solve_i(i, g, qnode)
        M = self.M
        J4 = np.empty((self.nodes, M, 4))
        for m in range(M):
            for j in range(4):
                J4[:, m, j] = self._esgn[m, j] * u[self._cidx[m, j]]
        X = np.empty((self.nodes, M, 9))
        X[:, :, :4] = J4
        X[:, :, 4:] = qnode[:, None, :]
        Y = np.empty((self.nodes, M, 9))
        for m in range(M):
            for grp in self._groups:
                S = self._state_matrix(i, g, m, grp)
                sel = self._grp_sel[grp]
                Y[sel, m, :] = X[sel, m, :] @ S.T
        return self.W[i] / M, Y

    def _sweep(self, g, qnode):
        """Exact solve for the given source (vectorized, single group)."""
        if not hasattr(self, '_Scache'):
            self._vectorize_setup()
        pb = np.zeros(self.nodes)
        qim_sum = np.zeros((self.nodes, 5))
        for i in range(self.I):
            w, Y = self._sweep_one(i, g, qnode)
            pb += w * Y[:, :, 4].sum(axis=1)
            qim_sum[:, 0] += w * Y[:, :, 4].sum(axis=1)
            qim_sum[:, 1:] += w * Y[:, :, 5:].sum(axis=1)
        return pb, qim_sum

    def _sweep_all(self, qnode):
        """All groups in one dispatch.  Independent work units are the
        (i, g) direction systems (I x ng = 21 for TY3 x 7-group).

        Two paths, bit-identical to each other:
        * serial — small problems (ncol < _PROCPOOL_MIN_NCOL or PSN_PAR=1);
          the original implementation, unchanged expression order.
        * process pool — large problems.  A thread pool does NOT work here:
          SuperLU's gssv back-substitution does not release the GIL in this
          scipy build (measured 0.58x at 21 threads, i.e. slower than
          serial), so the per-iteration solve runs in forked processes that
          share the pre-factorized object read-only (COW) and exchange
          sources/reduces through POSIX shared memory.  Workers reduce
          locally with the serial path's exact expression order and the
          parent accumulates in the same (g, i) order -> bit-identical."""
        if self._proc_pool is None:
            return self._sweep_all_serial(qnode)
        return self._sweep_all_proc(qnode)

    def _sweep_all_serial(self, qnode):
        if not hasattr(self, '_Scache'):
            self._vectorize_setup()
        jobs = [(g, i) for g in range(self.ng) for i in range(self.I)]
        res = {(g, i): self._sweep_one(i, g, qnode[g]) for g, i in jobs}
        phi = np.zeros((self.ng, self.nodes))
        qim = np.zeros((self.ng, self.nodes, 5))
        for g in range(self.ng):
            for i in range(self.I):
                w, Y = res[(g, i)]
                phi[g] += w * Y[:, :, 4].sum(axis=1)
                qim[g, :, 0] += w * Y[:, :, 4].sum(axis=1)
                qim[g, :, 1:] += w * Y[:, :, 5:].sum(axis=1)
        return phi, qim

    def _sweep_all_proc(self, qnode):
        """Process-pool sweep: sources in via shared memory, per-job local
        reduces out via shared memory; parent accumulates in the exact
        serial (g, i) order (bit-identical to _sweep_all_serial)."""
        q_view = np.frombuffer(self._q_shm.buf, dtype=np.float64)
        q_view.reshape(-1)[:] = np.ascontiguousarray(qnode).ravel()
        ng, I_, nodes = self.ng, self.I, self.nodes
        args = [((g, i), self._q_shm.name, self._r_shm.name,
                 ng, I_, nodes, self.W[i] / self.M)
                for g in range(ng) for i in range(I_)]
        self._proc_pool.map_async(_sweep_worker, args).get()
        out = np.frombuffer(self._r_shm.buf, dtype=np.float64)
        out = out.reshape(ng, I_, nodes, 5)
        phi = np.zeros((ng, nodes))
        qim = np.zeros((ng, nodes, 5))
        for g in range(ng):
            for i in range(I_):
                o = out[g, i]
                phi[g] += o[:, 0]
                qim[g, :, 0] += o[:, 0]
                qim[g, :, 1:] += o[:, 1:]
        return phi, qim

    def _prefactor_one(self, i, g):
        self._lu(i, g)
        self._bterms_for(i, g)
        # rows_rhs is fully consumed by _bterms_for — drop it (a few hundred
        # MB per system at core size).
        lu, _ = self._lusys[(i, g)]
        self._lusys[(i, g)] = (lu, None)

    def _prefactor(self):
        """Factorize every (i, g) system up front and build b-term tables,
        so the sweep does only back-substitution.  SuperLU's factorization
        releases the GIL (measured ~19x at 6 threads), so factorization
        overlaps on a thread pool.  The pool is deliberately small:
        build+factor carries ~3 GB of transient per job, a wave of 6 stays
        far inside the 62 GB cgroup."""
        jobs = [(g, i) for g in range(self.ng) for i in range(self.I)]
        n = len(jobs)
        if not hasattr(self, '_Scache'):
            self._vectorize_setup()   # creates _bterms_by_sys/_groups tables
        if n > 1:
            pf = max(1, min(6, _pool_size(n)))
            pool = ThreadPoolExecutor(max_workers=pf)
            try:
                futs = [pool.submit(self._prefactor_one, i, g)
                        for g, i in jobs]
                for f in futs:
                    f.result()
            finally:
                pool.shutdown(wait=True)
        else:
            self._prefactor_one(*jobs[0])

    # ------------------ process-pool sweep (fork + COW) ------------------
    # The back-substitution (SuperLU gssv) does NOT release the GIL in this
    # scipy build (measured 2026-09-12: 21-thread solve scaling = 0.58x,
    # i.e. SLOWER than serial; GIL-contention probe 1.61x under a busy
    # Python main thread).  Factorization DOES release the GIL (hence the
    # thread pool above), but the per-iteration sweep needs one GIL per
    # worker -> a forked PROCESS pool.  Children inherit the parent object
    # read-only via copy-on-write (21 COLAMD factors, 13 GB on the core,
    # are never copied); sources flow in and per-job reduces flow out
    # through POSIX shared memory, so the parent pays only a ~2 MB copy of
    # the source and a ~6.5 MB readback per iteration.  The worker's local
    # reduce uses the serial path's exact expression order, and the parent
    # accumulates jobs in the same (g, i) order -> bit-identical results.
    #
    # Small problems stay serial: the fork + per-iter copy cost only pays
    # off once one serial sweep exceeds ~0.2 s.
    _PROCPOOL_MIN_NCOL = 150000

    def _start_sweep_pool(self):
        """Fork the worker pool AFTER everything the workers read is warm:
        all LU factors, b-term tables and state-matrix caches (a warm
        first serial sweep guarantees the latter two are populated)."""
        global _SWEET_PSN
        if self._proc_pool is not None:
            return
        n = self.ng * self.I
        p = _pool_size(n)
        if p <= 1 or self.ncol < self._PROCPOOL_MIN_NCOL:
            return
        # warm every lazy cache so no worker ever populates a shared cache
        # (which would COW-diverge); the serial warm sweep already did this
        # for the (0, 0) state matrices used in probing — do it for all.
        for m in range(self.M):
            for grp in self._groups:
                self._state_matrix(0, 0, m, grp)
        for g in range(self.ng):
            for i in range(self.I):
                _ = self._bterms_for(i, g)
        # Both sweep segments live for the whole run, so the gate is on
        # the SUM: tmpfs is charged on page touch, and a per-segment
        # "fits" check is racy — a truncated-but-untouched segment shows
        # as free space until the workers write it (C5G7 core M4_S5
        # deadlock, 2026-09-10: q=52 MiB + r=17 MiB each passed the
        # per-segment check, together 69 MiB > the 64 MiB tmpfs).
        q_size = self.ng * self.nodes * 5 * 8
        r_size = self.ng * self.I * self.nodes * 5 * 8
        use_shm = (os.environ.get("PSN_SWEEP_SHM", "1") != "0"
                   and q_size + r_size <= _shm_free_bytes())
        self._q_shm = _SharedBuf(q_size, force_file=not use_shm)
        self._r_shm = _SharedBuf(r_size, force_file=not use_shm)
        _SWEET_PSN = self
        ctx = _mp.get_context("fork")
        self._proc_pool = ctx.Pool(processes=p)

    def _stop_sweep_pool(self):
        global _SWEET_PSN
        if self._proc_pool is not None:
            self._proc_pool.close()
            self._proc_pool.join()
            self._proc_pool = None
        for sh in (self._q_shm, self._r_shm):
            if sh is not None:
                sh.close()
                sh.unlink()
        self._q_shm = None
        self._r_shm = None
        _SWEET_PSN = None

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
            for grp in self._groups:
                S = self._state_matrix(i, g, m, grp)
                sel = self._grp_sel[grp]
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

    def _source_update(self, qim, lam, matidx):
        """Plain B.3 paraboloidal source update for the whole spectrum.
        qim: (ng, nodes, 5) paraboloidal moments of the current flux."""
        ng = self.ng
        s = np.empty((ng, self.nodes, 5))
        for g in range(ng):
            s[g] = np.zeros((self.nodes, 5))
            for g2 in range(ng):
                s[g] += (self.Sgg[g, g2, matidx]
                         + self.chi[g] * self.nuSf[g2, matidx] / lam
                         )[:, None] * qim[g2]
        return s

    def keff(self, max_outer=2000, outer_tol=1e-9, verbose=True,
             source="full", extrapolate=False):
        """Multi-group keff driver.
        Source moments (paper B.3): the paraboloidal source in group g is
            qmom_g = sum_{g2} [ Sgg[g,g2] + chi_g * nuSf[g2]/lam ] * mom(phi_g2)
        i.e. fission source is local (same spatial parabola as the flux),
        spectrally distributed by chi.  Reduces exactly to the single-group
        'full' variant (S + nuSf/lam)*mom that reproduces Fig.3 to <1 pcm.

        Dombey two-point source extrapolation (OFF by default; pass
        ``extrapolate=True`` to enable): when the power sequence is
        asymptotically geometric (dlam < 1e-4 and the two-point ratio
        rho_hat = (k0-k1)/(k1-k2) sits in (0.5, 0.999)) the next source is
        replaced by the Richardson estimate
            S_ex = (S_n - rho_hat*S_{n-1}) / (1 - rho_hat),
        which, for a purely geometric error e_n ~ a*rho^n, equals the
        fixed point S* exactly.  (A linear combination of paraboloidal
        sources is a valid paraboloidal source, so the next sweep is
        well-defined.)  Acceptance is the EXACT fixed-point residual
        ratio: with G the source-update map, accept iff
            ||G(S_ex) - S_ex|| < 0.5 * ||G(S_n) - S_n||.
        A rejected attempt costs one extra sweep and cools down 2
        iterations; 6 consecutive failures cool down 30 (no hard disable,
        the map keeps becoming more geometric).  Convergence is judged on
        the plain (un-extrapolated) dlam sequence.

        WARNING: the extrapolated trajectory is NOT bit-for-bit
        equivalent to the plain iteration.  On the C5G7 rectangular
        quarter-core M8 case it converged in 585 outer iterations (vs
        1454 plain, ~2.5x faster) but returned 1.1853736, 0.06 pcm below
        the plain value 1.1853742: the two-point extrapolation cancels the
        dominant geometric mode but leaves a slow subdominant mode whose
        per-iteration change then falls below the dkeff criterion before
        it has decayed.  Reference values therefore come from the plain
        iteration, which is why extrapolation is off by default; use it as
        a ~2.5x speedup on large core cases when ~0.1 pcm accuracy is
        acceptable (still within the 1 pcm regression tolerance)."""
        matidx = self.mat[self.j_idx, self.i_idx]
        ng = self.ng
        # physical fission RATE is nuSf * phi * node-area; for rectangular
        # nodes the areas differ so they must enter the balance (for square
        # nodes a constant area cancels in F_new/F_old — path unchanged).
        area_w = self.area.ravel() if self.rect else None
        # parallel pre-factorization (SuperLU factorize releases the GIL)
        self._prefactor()
        lam = 1.0
        phi = np.ones((ng, self.nodes))
        fission_rate = np.zeros(self.nodes)
        for g in range(ng):
            fission_rate += self.nuSf[g, matidx] * phi[g]
        if area_w is not None:
            fission_rate *= area_w
        # paraboloidal moments of uniform flux = [1,0,0,0,0]
        mom = np.zeros((ng, self.nodes, 5))
        mom[:, :, 0] = 1.0
        qnode = self._source_update(mom, 1.0, matidx)
        F_old = float(fission_rate.sum())
        lam_hist = []   # last 3 keff values (one per plain sweep)
        s_hist = []     # last 2 sources actually swept
        cool = 0        # iterations until extrapolation may retry
        n_fail = 0      # consecutive rejected attempts
        warm = True    # first sweep runs serial to warm every lazy cache
        try:
            for outer in range(max_outer):
                if warm:
                    # Serial warm sweep: populates _Scache/_bterms_by_sys
                    # etc. so that, after the fork, NO worker ever writes
                    # into a shared cache (a worker-side cache fill would
                    # COW-diverge from the parent and is wasted work).
                    phi, qim = self._sweep_all_serial(qnode)
                    self._start_sweep_pool()
                    warm = False
                else:
                    phi, qim = self._sweep_all(qnode)
                fission_rate = np.zeros(self.nodes)
                for g in range(ng):
                    fission_rate += self.nuSf[g, matidx] * phi[g]
                if area_w is not None:
                    fission_rate *= area_w
                F_new = float(fission_rate.sum())
                lam_new = lam * F_new / F_old
                dlam = abs(lam_new - lam) / lam_new
                lam = lam_new
                lam_hist.append(lam)
                if len(lam_hist) > 3:
                    lam_hist.pop(0)
                s_hist.append(qnode)
                if len(s_hist) > 2:
                    s_hist.pop(0)
                s_next = self._source_update(qim, lam, matidx)
                F_old = F_new
                # ---------------- Dombey two-point source extrapolation -------
                qnode = s_next
                if cool > 0:
                    cool -= 1
                elif (extrapolate and outer >= 8 and n_fail < 6
                      and dlam < 1e-4
                      and len(lam_hist) == 3 and len(s_hist) == 2
                      and lam_hist[1] != lam_hist[0]):
                    k2, k1, k0 = lam_hist
                    rhat = (k0 - k1) / (k1 - k2)
                    if 0.5 < rhat < 0.999:
                        s_n, s_nm1 = s_hist[-1], s_hist[-2]
                        s_ex = (s_n - rhat * s_nm1) / (1.0 - rhat)
                        phi_ex, qim_ex = self._sweep_all(s_ex)
                        fission_rate = np.zeros(self.nodes)
                        for g in range(ng):
                            fission_rate += self.nuSf[g, matidx] * phi_ex[g]
                        if area_w is not None:
                            fission_rate *= area_w
                        F_ex = float(fission_rate.sum())
                        lam_ex = lam * F_ex / F_old   # = F_ex/F_0 (telescoping)
                        # exact fixed-point residual test (no extra sweep):
                        r_n = float(np.linalg.norm(s_next - s_n) /
                                    max(1.0, float(np.linalg.norm(s_n))))
                        s_ex_up = self._source_update(qim_ex, lam_ex, matidx)
                        r_ex = float(np.linalg.norm(s_ex_up - s_ex) /
                                     max(1.0, float(np.linalg.norm(s_ex))))
                        if r_ex < 0.5 * r_n:
                            # accept: continue the power sequence from S_ex
                            qnode = s_ex
                            cool = 2
                            n_fail = 0
                            if verbose:
                                print(f"  dombey: rhat={rhat:.6f}  "
                                      f"residual {r_n:.2e} -> {r_ex:.2e}  "
                                      f"keff_ex={lam_ex:.7f}")
                        else:
                            n_fail += 1
                            if verbose:
                                print(f"  dombey reject: rhat={rhat:.6f}  "
                                      f"r_n={r_n:.2e} r_ex={r_ex:.2e}  "
                                      f"ratio={r_ex/max(r_n,1e-300):.2f} "
                                      f"(fail {n_fail})")
                            if n_fail >= 6:
                                # not a pure geometric mode yet: cool down and
                                # let it retry later, do not disable outright
                                cool = 30
                                n_fail = 0
                # -----------------------------------------------------------------
                if verbose and (outer % 5 == 0 or dlam < outer_tol):
                    print(f"it {outer:4d}  keff={lam:.7f}  dkeff={dlam:.2e}  "
                          f"F={F_new:.5e}")
                if dlam < outer_tol:
                    break

        finally:
            self._stop_sweep_pool()
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
