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

import multiprocessing as _mp
import multiprocessing.shared_memory as _shm
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from .node import (node_alphas, node_alphas_generic, node_params_generic,
                      node_state, node_response_matrix, ty_polar_set,
                      generic_polar_set)
from .node_rect import (node_params_rect, node_params_rect_generic,
                        node_state_rect, node_response_matrix_rect)
from . import node_strict
from . import numa

PI = np.pi

# dkeff at which ``--anderson M --extrapolate`` hands the iteration from Anderson
# to Dombey (see PSN2D.keff).  1e-4 is Dombey's own attempt gate, so the relay
# fires at the first iteration where the second accelerator can do anything;
# the env var exists because the crossover is a measured quantity, not a law.
RELAY_DLAM = float(os.environ.get("OPENPSN_AA_RELAY_DLAM", 1e-4))

# Two Dombey knobs, both env-overridable because both are measured quantities
# (see the tail trace in dev/tailtrace.py) rather than anything the method
# dictates:
#   DOMBEY_ACCEPT  how much better the extrapolated source must be than the
#                  plain one for the trial to be kept (0.5 = "must halve the
#                  source residual", Dombey's own test);
#   DOMBEY_COOL    sweeps to sit out after an acceptance before retrying, i.e.
#                  how often the accelerator is allowed to touch the tail.
#                  2 keeps three plain sweeps between attempts because an
#                  attempt needs a 3-point keff history; see the accepted-sweep
#                  memo below for why that cadence is now free to tighten.
#   DOMBEY_COOLFAIL  sweeps to sit out after DOMBEY_FAILS rejected attempts.
#                      30 was a guess; on M16_S2 it turned out to cost more
#                      than the extrapolations it protects against -- see the
#                      recorded tail in dev/xleg.sh / OPENPSN_TRACE=1.
DOMBEY_ACCEPT = float(os.environ.get("OPENPSN_DOMBEY_ACCEPT", 0.5))
DOMBEY_COOL = int(os.environ.get("OPENPSN_DOMBEY_COOL", 2))
DOMBEY_FAILS = int(os.environ.get("OPENPSN_DOMBEY_FAILS", 6))
DOMBEY_COOLFAIL = int(os.environ.get("OPENPSN_DOMBEY_COOLFAIL", 30))

# How far the SOURCE may still be from a fixed point when ``dlam`` stops the
# run, as a multiple of ``outer_tol``.  ``dlam`` watches the eigenvalue, and the
# eigenvalue is the *insensitive* direction of the source map: an iterate can sit
# 1e-5 away from a fixed point and still move keff by less than 1e-10, and
# Dombey's accept test (its extrapolated source must halve the source residual)
# can land the iteration exactly there.  Two stops of this run, both measured
# 2026-11-23 on M16_S2 / 60 threads / seed 1 with the gate off:
#
#   r_src at stop   sweeps    keff bias vs the 1.1886778 reference
#   1.07e-05        229+1     -0.79 pcm      (relay threshold moved to 1e-5)
#   3.67e-07        323       -0.004 pcm     (the shipped configuration)
#
# 29x the residual buys only 200x the error, so ``r_src`` is NOT a proportional
# error estimate -- the direction of the leftover matters, not just its size --
# and that is why it is used here as a guard and published per run rather than
# converted into a claimed accuracy.  The default 1e-4*outer_tol = 1e-6 sits a
# decade inside the refused stop and a decade outside the accepted one: it ends
# the shipped run on exactly the iteration it ended on before (bit-identical
# keff, same sweep count), and it refuses the -0.79 pcm stop.  Wanting a source
# converged for its own sake, pay: OPENPSN_RES_STOP_FACTOR=100 ran 671 sweeps /
# 211 s instead of 323 / 129 s to land r_src at 1.9e-9 -- it moved the bias from
# -0.0038 to +0.000003 pcm, +63% of work for 1/1500 of the error.  On the gate
# case M4_S2 the same relay stops at r_src 1.6e-7 and lands +0.013 pcm off the
# plain-iteration value there (Dombey alone +0.0000, Anderson alone +0.0000), so
# the residual is a loose bound -- 75x inside a 1-pcm budget, not a certificate.
RES_STOP_FACTOR = float(os.environ.get("OPENPSN_RES_STOP_FACTOR", 1e4))


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


class _SharedBuf:
    """Shared byte buffer for the process-pool sweep (POSIX shm ONLY).

    The parent gates on q_size + r_size <= free tmpfs before creating
    EITHER segment (tmpfs is charged on page touch, so the gate must
    cover the SUM — see the note in ``_start_sweep_pool``).  A too-small
    tmpfs raises MemoryError with an actionable message: there is
    deliberately NO disk fallback.  A PSN2D sweep host must ship a
    roomy /dev/shm (>= q+r for the largest sweep; the C5G7 quarter
    core at S=6 needs ~100 MiB, the 64 MiB container default does not
    fit core S>=5).  Run with ``PSN_PAR=1`` to stay serial and skip
    the pool (and its shared buffers) entirely.
    """
    __slots__ = ("name", "size", "_sm")

    def __init__(self, size):
        self.size = size
        free = _shm_free_bytes()
        if size > free:
            raise MemoryError(
                f"sweep pool needs {size / 2**20:.1f} MiB of /dev/shm but "
                f"only {free / 2**20:.1f} MiB is free — enlarge the shm "
                f"tmpfs (docker: --shm-size=4g) or run serial (PSN_PAR=1)")
        self._sm = _shm.SharedMemory(create=True, size=size)
        self.name = self._sm.name

    @property
    def buf(self):
        return self._sm.buf

    def close(self):
        # Note: does NOT clear _sm — unlink() must still see it (the
        # sweep pool does close(); unlink(); and a cleared _sm would
        # silently skip the shm_unlink, leaking the segment).
        self._sm.close()

    def unlink(self):
        self._sm.unlink()
        self._sm = None


def _open_buf(name):
    """Worker side: open the parent's POSIX-shm sweep segment by name."""
    return _shm.SharedMemory(name=name)


def _prefactor_job(psn, per_worker, i, g):
    """Prefactor inside a pool worker: cap THIS worker's BLAS threads at
    per_worker (so W workers x per_worker <= psn.threads), do the job,
    restore.  The parent re-asserts the full budget after the pool."""
    from . import runtime
    prev = runtime.get_thread_limit()
    runtime.set_thread_limit(per_worker)
    # NUMA: first-touch THIS system's factor on its home node so that the sweep
    # worker which is routed to it reads locally (psn2d/numa).  set_mempolicy is
    # per-thread, which is what lets W threads factorise W different systems
    # into W different nodes at the same time.  No-op unless OPENPSN_NUMA=1.
    nn = numa.width() if numa.mem_home() else 0
    if nn > 1:
        numa.set_mem(numa.home(i, g, psn.ng, nn))
    try:
        psn._prefactor_one(i, g)
    finally:
        runtime.set_thread_limit(prev)
        if nn > 1:
            numa.set_mem(None)


class _RoutedPool:
    """Several fork pools — one per NUMA node — behind the small subset of the
    ``multiprocessing.Pool`` API the sweep uses (``map_async``/``close``/
    ``join``), so ``_sweep_all_proc`` and the ``_proc_pool is None`` serial test
    stay exactly as they were.

    ``map_async`` routes every task to the pool whose node HOLDS that (i, g)
    factor — the same ``numa.home`` the prefactor used to place it — then waits
    for all of them.  Routing cannot perturb the arithmetic: the workers write
    their reduces into shared memory and the parent accumulates them in its own
    (g, i) order regardless of which pool produced which, so a placed run is
    bit-identical to an unplaced one and can be gated that way.
    """

    class _Handle:
        """``get()`` gives back the per-task results in the order the tasks were
        passed in (each batch remembers its original indices), and re-raises the
        first worker error — i.e. what Pool's AsyncResult.get() would have done.
        The sweep ignores the values (they are in shared memory), but returning
        them keeps the stand-in honest for any other caller and for probes."""

        def __init__(self, batches, n):
            self._batches = batches                   # [(indices, AsyncResult)]
            self._n = n

        def get(self):
            out, err = [None] * self._n, None
            for idxs, h in self._batches:
                try:
                    res = h.get()
                except BaseException as exc:          # noqa: BLE001
                    err = err if err is not None else exc
                    continue
                for j, r in zip(idxs, res):
                    out[j] = r
            if err is not None:
                raise err
            return out

    def __init__(self, pools, nn, ng):
        self.pools = pools                            # {node: Pool}
        self.nn, self.ng = nn, ng

    def map_async(self, fn, args):
        batches = {}
        for n, a in enumerate(args):
            g, i = a[0]
            batches.setdefault(numa.home(i, g, self.ng, self.nn),
                               []).append((n, a))
        sent = [(idxs, self.pools[k].map_async(fn, [a for _, a in grp]))
                for k, grp in sorted(batches.items()) if k in self.pools
                for idxs in [[j for j, _ in grp]]]
        return self._Handle(sent, len(args))

    def close(self):
        for p in self.pools.values():
            p.close()

    def join(self):
        for p in self.pools.values():
            p.join()


def _sweep_worker(args):
    """Forked sweep worker: one (i, g) direction, end to end.

    Runs in a child process (its own GIL — the whole reason for processes;
    the scipy build in use does NOT release the GIL around SuperLU's
    gssv back-substitution, so a thread pool serializes on the lock and
    MEASURES SLOWER than serial on the 21-job C5G7 core).  The source
    vector is read from, and the per-job reduce is written to, POSIX
    shared memory: no pickling of large arrays.  The work is done by
    ``PSN2D._sweep_o`` -- literally the same function the serial path
    calls -- so parent and child cannot drift apart."""
    (g, i), qname, rname, ng, I_, nodes, w = args
    # strict budget (2026-09-12 user input): each forked worker pins to 1
    # BLAS thread, so N workers x 1 <= the user's threads cap
    from . import runtime
    runtime.set_thread_limit(1)
    qs = _open_buf(qname)
    rs = _open_buf(rname)
    try:
        qnode = np.frombuffer(qs.buf, dtype=np.float64).reshape(ng, nodes, 5)
        q = qnode[g].copy()
        out = np.frombuffer(rs.buf, dtype=np.float64).reshape(ng, I_, nodes, 5)
        psn = _SWEET_PSN
        # one core for both sweep paths: the backend may install a
        # vectorised _sweep_o, and the serial path uses the same hook, so
        # parent and children stay bit-identical to each other
        _w, o_local = psn._sweep_o(i, g, q)
        o = out[g, i]
        o[:] = o_local
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
                 generic=False, I=30, widths=None, threads=None,
                 mem_limit_gb=None, strict=False):
        if M % 2 != 0:
            raise ValueError("M must be even (mirror pairs need paired segments)")
        # Runtime inputs (user settings 2026-09-12): total parallel-unit
        # budget and factor-pool memory cap.  Default threads = half the
        # system core count; default memory cap 32 GB.
        from . import runtime
        self.threads = max(1, int(threads or runtime.default_threads()))
        self.mem_limit_gb = float(mem_limit_gb or runtime.DEFAULT_MEM_LIMIT_GB)
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
        # strict phase-space diffusion node closure (Xu, strict PSN):
        # replaces the isotropic body with the exact sector tensor node
        # (node_strict); faces/alpha/closure unchanged.  M=2 restricted-
        # model anchor: node map == legacy bit-for-bit (see node_strict).
        self.strict = bool(strict)
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
                if self.strict:
                    R, _ = node_strict.strict_response_dispatch(
                        pr, self._phi_m(m), self.dphi, alpha)
                else:
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
                                            alpha=self._alpha_for(i, m),
                                            strict=self.strict)
            else:
                R, _ = node_response_matrix(self.St[g, mat], self.mu[i], self.h,
                                            self._phi_m(m), self.dphi,
                                            strict=self.strict)
            self._Rcache[k] = R
        return self._Rcache[k]

    def _state(self, J4, qn, i, g, n, m):
        """node_state with generic/restricted params selected automatically."""
        mat = self.mat[self.j_idx[n], self.i_idx[n]]
        if self.rect:
            pr, alpha = self._pr_alpha_rect(i, g, m, mat, n)
            if self.strict:
                return node_strict.strict_state_dispatch(
                    J4, qn, pr, self._phi_m(m), self.dphi, alpha)
            return node_state_rect(J4, qn, pr, self._phi_m(m), self.dphi, alpha)
        if self.generic:
            return node_state(J4, qn, self.St[g, mat], 1.0, self.h,
                              self._phi_m(m), self.dphi,
                              pr=self._pr_generic(i, g, mat),
                              alpha=self._alpha_for(i, m),
                              strict=self.strict)
        return node_state(J4, qn, self.St[g, mat], self.mu[i],
                          self.h, self._phi_m(m), self.dphi,
                          strict=self.strict)

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

    def _sweep_o(self, i, g, qnode):
        """(weight, o) for one (i, g) system, where o[n] holds the system's
        already-weighted contribution to [phi, moments(1..4)] of group g.

        This is the single reduce used by BOTH sweep paths (serial and
        forked pool), so the two stay bit-identical to each other.  The
        default is ``_sweep_one`` plus the same expression order the loops
        used before; a backend may install a vectorised equivalent (see
        memopt.install_shared_chol) that never materialises the (nodes, M,
        9) X/Y temporaries — those two arrays were ~45% of a sweep's
        memory traffic on the C5G7 core.
        """
        w, Y = self._sweep_one(i, g, qnode)
        o = np.empty((self.nodes, 5))
        o[:, 0] = w * Y[:, :, 4].sum(axis=1)
        o[:, 1:] = w * Y[:, :, 5:].sum(axis=1)
        return w, o

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
        res = {(g, i): self._sweep_o(i, g, qnode[g]) for g, i in jobs}
        phi = np.zeros((self.ng, self.nodes))
        qim = np.zeros((self.ng, self.nodes, 5))
        for g in range(self.ng):
            for i in range(self.I):
                _w, o = res[(g, i)]
                phi[g] += o[:, 0]
                qim[g, :, 0] += o[:, 0]
                qim[g, :, 1:] += o[:, 1:]
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
        far inside the 62 GB cgroup.  Thread budget (2026-09-12 user
        input): W workers x (threads // W) BLAS threads <= self.threads."""
        jobs = [(g, i) for g in range(self.ng) for i in range(self.I)]
        n = len(jobs)
        if not hasattr(self, '_Scache'):
            self._vectorize_setup()   # creates _bterms_by_sys/_groups tables
        if n > 1:
            # Pool width: 6 by default (the plain path's dict assembly carries
            # ~3 GB of transient per job).  A backend whose factorization is
            # GIL-free and BLAS-free (the C++ Eigen Cholesky) raises the cap
            # through ``_prefactor_workers`` — one worker per (i, g) system
            # then runs at 1 BLAS thread, which is what a many-core box wants.
            cap = int(getattr(self, '_prefactor_workers', 6))
            pf = max(1, min(cap, self.threads))
            per_worker = max(1, self.threads // pf)
            from . import runtime
            pool = ThreadPoolExecutor(max_workers=pf)
            try:
                futs = [pool.submit(_prefactor_job, self, per_worker, i, g)
                        for g, i in jobs]
                for f in futs:
                    f.result()
            finally:
                pool.shutdown(wait=True)
            runtime.set_thread_limit(self.threads)   # restore parent budget
        else:
            self._prefactor_one(*jobs[0])

    def _check_mem_gate(self):
        """Fail-loud gate (2026-09-12 user input): the resident factor pool
        — all (i, g) factors, COW-shared across workers — must fit the
        user's memory cap ``mem_limit_gb`` (default 32 GB).  Measured on
        the live factors, not estimated; over budget is an error, never a
        silent fallback."""
        total = 0
        for (i, g), (factor, _rows) in self._lusys.items():
            nnz = getattr(factor, "factor_nnz", None)
            if nnz is None:
                nnz = factor.L.nnz + factor.U.nnz
            total += getattr(factor, "pool_bytes", nnz * 16)
        limit = self.mem_limit_gb * 2**30
        if total > limit:
            raise MemoryError(
                f"factor pool {total / 2**30:.2f} GB "
                f"({len(self._lusys)} systems) exceeds mem_limit_gb="
                f"{self.mem_limit_gb:.0f}; increase the limit or use a "
                f"smaller M / coarser grid / tiled backend")

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
        p = min(p, self.threads)   # strict budget (2026-09-12 user input)
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
        # No disk fallback: a sweep host must have a roomy /dev/shm;
        # _SharedBuf raises MemoryError with an actionable message.
        q_size = self.ng * self.nodes * 5 * 8
        r_size = self.ng * self.I * self.nodes * 5 * 8
        if q_size + r_size > _shm_free_bytes():
            raise MemoryError(
                f"sweep pool needs (q+r)={ (q_size + r_size) / 2**20:.1f} MiB "
                f"of /dev/shm but only {_shm_free_bytes() / 2**20:.1f} MiB is "
                f"free — enlarge the shm tmpfs (docker: --shm-size=4g) or "
                f"run serial (PSN_PAR=1)")
        self._q_shm = _SharedBuf(q_size)
        self._r_shm = _SharedBuf(r_size)
        _SWEET_PSN = self
        ctx = _mp.get_context("fork")
        nn = numa.width()
        njobs = self.I * self.ng
        if numa.route_pools() and nn > 1 and p >= 2 * nn:
            # One pool per node, each forked while the PARENT's affinity is that
            # node's CPU set — Pool offers no per-worker hook, and fork is how
            # the mask reaches the children — sized by how many systems are
            # homed there.  A worker then reads its factor locally instead of
            # pulling it across QPI (measured: 21 jobs on 4 nodes inflate their
            # own run time 2.4x, dev/poolfloor.py).
            cnt = {}
            for g in range(self.ng):
                for i in range(self.I):
                    k = numa.home(i, g, self.ng, nn)
                    cnt[k] = cnt.get(k, 0) + 1
            all_cpus = os.sched_getaffinity(0)
            pools = {}
            try:
                for node, cpus in numa.node_cpus():
                    if node not in cnt:
                        continue
                    if not numa.set_cpu(cpus):
                        break
                    pools[node] = ctx.Pool(
                        processes=max(1, min(cnt[node],
                                             -(-p * cnt[node] // njobs))))
            finally:
                os.sched_setaffinity(0, all_cpus)
            if len(pools) == len(cnt):
                self._proc_pool = _RoutedPool(pools, nn, self.ng)
            else:                       # partial placement is worse than none
                for pl in pools.values():
                    pl.close()
                    pl.join()
        if self._proc_pool is None:
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
        qim: (ng, nodes, 5) paraboloidal moments of the current flux.

        Pure data motion, and the most expensive piece of an iteration after the
        sweep itself: 49 Python steps, 98 per-node gathers out of the
        (ng, ng, nmat) tables (``matidx`` is node -> material and fixed for the
        whole run, so each call redoes work done once before it), and 49 fresh
        (nodes, 5) temporaries -- 66 MB of traffic to produce 2.9 MB, ~19 ms per
        call in situ on M16_S2 (``dev/runphase.py`` times it in place), which is
        ~5 s of a 319-sweep run.  ``--opt chol-rs`` therefore replaces it with a
        fused kernel that never builds the coefficient array and takes the two
        scatter tables node-major (``memopt.install_fused_source_update``, 7 ms
        in situ).  This loop stays the definition of the answer: every other
        backend runs it, and the kernel is checked byte-for-byte against it
        (``dev/srcident.py``) because a NumPy reordering of the same sum was
        measured as fast warm (16.4 -> 9.6 ms, ``dev/srcupdate.py``) and made no
        difference at all once the arrays were cold -- 18-23 ms either way.
        """
        ng = self.ng
        s = np.empty((ng, self.nodes, 5))
        for g in range(ng):
            s[g] = np.zeros((self.nodes, 5))
            for g2 in range(ng):
                s[g] += (self.Sgg[g, g2, matidx]
                         + self.chi[g] * self.nuSf[g2, matidx] / lam
                         )[:, None] * qim[g2]
        return s

    def _aa_update(self, aa, s, f):
        """One Anderson-accelerated source step; None means take the plain step.

        AA(m) on the fixed-point map S = G(S): keep the last m+1 pairs
        (S_i, F_i = G(S_i)), minimise ||dF gamma - (S_n - F_n)|| over the
        residual differences dF_i = F_{i+1} - F_i, then step

            S_{n+1} = F_n - dF gamma.

        Two properties make it the right accelerator here, where Dombey's
        two-point Richardson step is not.  (1) Fixed points are preserved
        exactly: at S = G(S) the right-hand side is 0, gamma = 0 is a solution
        and the step returns F_n = S_n, so AA cannot move the answer -- the
        sub-0.1 pcm bias Dombey carries (a *wrong* assumed decay ratio rho_hat
        leaves a subdominant mode standing when dkeff cuts the iteration) is
        not reproduced.  (2) It costs no extra sweeps: F_n is the source
        update the loop already computes, whereas every Dombey attempt pays
        for a trial sweep and an accept/reject residual test.

        A linear combination of paraboloidal sources need not stay
        non-negative, and a negative fission source can drive F_new/F_old --
        the eigenvalue update -- anywhere, so two cheap protections fall back to
        the plain power step and clear the history: a non-finite or degenerate
        least-squares result, and a step longer than ``maxstep`` times the plain
        residual.  Positivity itself is not a veto: the source is lifted back to
        the floor the plain source already has (measured against a
        shortest-sub-step ratio test below, which loses -- see the comment
        there).  If the residual then grows ``bad_limit`` steps in a row,
        acceleration switches itself off and the run continues as the plain
        iteration, which always converges.
        """
        m = int(aa["m"])
        if not m:
            return None
        aa["s"].append(s)
        aa["f"].append(f)
        while len(aa["s"]) > m + 1:
            aa["s"].pop(0)
            aa["f"].pop(0)
        res = float(np.linalg.norm(f - s))
        if len(aa["s"]) < 3:
            return None
        k = len(aa["s"]) - 1
        if aa["buf"] is None or aa["buf"].shape[1] != f.size:
            # one (m x N) scratch row block, reused: a fresh 14 MB np.empty
            # per iteration is 3.5k first-touch page faults for nothing
            aa["buf"] = np.empty((m, f.size))
        dF = aa["buf"][:k]
        for j in range(k):
            dF[j] = (aa["f"][j + 1] - aa["f"][j]).ravel()
        if not np.all(np.isfinite(dF)):
            aa["why"]["finite"] += 1
            aa["s"].clear()
            aa["f"].clear()
            return None
        try:
            # Normal equations, not numpy's lstsq.  The system is (N x k) with
            # N = 364140 unknowns and k <= depth columns -- handing LAPACK the
            # (k x N) transpose instead asks for an under-determined solve,
            # which gelsd refuses outright (measured: 68 straight LinAlgErrors
            # with AA(5) on M8_S2, i.e. AA silently did nothing and reproduced
            # the plain 205-iteration trajectory).  Even solved properly,
            # gelsd's SVD costs 0.07 s per iteration on M16_S2 (measured 47.8
            # -> 52.3 s over 60 iterations), a sixth of the saving AA buys on
            # that case, so form the (k x k) Gram matrix -- one gemm, 2*N*k^2
            # flops -- and take the Tikhonov term scaled by its own diagonal as
            # the stand-in for the rcond=1e-10 cutoff the SVD used to apply.
            gram = dF @ dF.T
            sc = float(np.max(np.diag(gram)))
            gam = np.linalg.solve(gram + (1e-12 * sc if sc > 0.0 else 1.0)
                                  * np.eye(k), dF @ (s - f).ravel())
        except np.linalg.LinAlgError:
            aa["why"]["solve"] += 1
            aa["s"].clear()
            aa["f"].clear()
            return None
        step = gam @ dF
        if not np.all(np.isfinite(step)):
            aa["why"]["finite"] += 1
            aa["s"].clear()
            aa["f"].clear()
            return None
        if float(np.linalg.norm(step)) > aa["maxstep"] * max(res, 1e-300):
            aa["why"]["longstep"] += 1
            aa["s"].clear()
            aa["f"].clear()
            return None
        flat = f.ravel()
        # A linear combination of paraboloidal sources is not obliged to stay
        # non-negative (732 of 750 steps on M16_S2 undershoot somewhere), and a
        # negative fission source sends F_new/F_old -- the eigenvalue update --
        # anywhere.  Three ways of handling that have been measured here, and
        # the middle one wins:
        #   * veto the whole step: throws AA away, it fires on nearly every
        #     iteration (AA(3/5/8) reproduced the plain 205-iteration uo2
        #     trajectory exactly);
        #   * the interior-point RATIO test, alpha = min (f_i - floor)/step_i
        #     over the violating entries, shortening or discarding the step:
        #     principled, but on c5g7_mox_assembly_bww::M4_S2 it vetoed 51 steps
        #     as negligible and applied 4, leaving 157 iterations where the
        #     floor below gave 81;
        #   * FLOOR the entries at the bound the PLAIN source already has.  The
        #     pedestal it leaves where AA wanted to go negative is not noise --
        #     it is an over-relaxation of exactly the entries the positivity
        #     limit binds, and it converges fastest of the three.  It is still
        #     unbiased in the only sense that matters here: at a fixed point the
        #     step is 0, nothing undershoots, nothing is floored, and the answer
        #     is the plain iteration's own.
        floor = min(0.0, float(f.min()))
        shifted = flat - step
        if float(shifted.min()) < floor:
            aa["clipped"] += 1
            shifted = np.maximum(shifted, floor)
        s_new = shifted.reshape(f.shape)
        if aa["res"] and res > 1.02 * aa["res"][-1]:
            aa["bad"] += 1
        else:
            aa["bad"] = 0
        aa["res"].append(res)
        aa["applied"] += 1
        if aa["bad"] >= aa["bad_limit"]:
            aa["m"] = 0
            aa["off"] = True
        return s_new

    def keff(self, max_outer=2000, outer_tol=1e-9, verbose=True,
             source="full", extrapolate=False, anderson=0, ritz=False):
        """Multi-group keff driver.

        With ``ritz=True``, use a diffusion-preconditioned projection of the
        transport eigenproblem. In that mode ``max_outer`` caps total sweeps
        including the final verification sweep.

        Anderson acceleration (``anderson=m``, off by default) replaces the
        plain power step with AA(m) on the same source map, at no extra sweep
        cost -- see ``_aa_update`` for the formulation and its guards.
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
        acceptable (still within the 1 pcm regression tolerance).

        Anderson acceleration, for comparison, is the accelerator used by
        production transport codes for exactly this map; here it removes the
        bias above as well as more iterations, so ``--anderson`` is the
        recommended outer accelerator and ``--extrapolate`` is kept for
        reproducibility of the older numbers.

        Passing BOTH is a relay, not a contradiction: Anderson runs the early
        phase, where it is measurably ahead (M16_S2 at a matched 60 sweeps:
        -2637 pcm against -6314 for the plain and Dombey trajectories), and once
        dkeff falls to ``RELAY_DLAM`` (1e-4, Dombey's own attempt gate) the
        history is dropped and Dombey finishes the tail, where it is ahead (633
        sweeps to outer_tol against Anderson's 752).  A relayed run therefore
        inherits Dombey's sub-0.1 pcm bias, not Anderson's zero bias — that is
        the price of the extra speed, and ``aa_stats["relayed_to_dombey"]``
        records that it happened and where."""
        if ritz:
            if extrapolate or anderson or getattr(self, "_cmfd", None) is not None:
                raise ValueError("Ritz is a standalone outer solver")
            from .ritz import keff_ritz
            return keff_ritz(self, max_outer, outer_tol, verbose)
        matidx = self.mat[self.j_idx, self.i_idx]
        ng = self.ng
        # physical fission RATE is nuSf * phi * node-area; for rectangular
        # nodes the areas differ so they must enter the balance (for square
        # nodes a constant area cancels in F_new/F_old — path unchanged).
        area_w = self.area.ravel() if self.rect else None
        # runtime budget (user input 2026-09-12): factorization phase runs
        # with <= self.threads BLAS threads in the parent
        from . import runtime
        runtime.set_thread_limit(self.threads)
        # parallel pre-factorization (SuperLU factorize releases the GIL)
        self._prefactor()
        self._check_mem_gate()
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
        it_done = 0
        r_src = float("inf")   # source fixed-point residual (see RES_STOP_FACTOR)
        extra_sweeps = 0   # trial sweeps beyond the outer-loop count
        sweeps = 0         # sweeps actually EXECUTED (see the memo below)
        pending = None     # (source, phi, qim) of an accepted Dombey trial
        # one row per outer iteration: (it, dkeff, r_src, mode) with mode
        # p=plain/Anderson, d=Dombey attempt accepted, r=rejected.  A few
        # hundred rows of floats, and it is the only way the tail policy below
        # can be tuned from data instead of from another cold run.
        trace = []
        cmfd = getattr(self, "_cmfd", None)   # optional CMFD (see psn2d/cmfd)
        # Anderson state: history of (S_i, G(S_i)) plus the restart counters
        # described in _aa_update.  All of it lives in the PARENT (the keff
        # loop is parent-side), so nothing here crosses the fork.
        aa = {"m": int(anderson), "s": [], "f": [], "res": [], "bad": 0,
              "off": False, "maxstep": 4.0, "bad_limit": 4, "buf": None,
              "why": {"finite": 0, "longstep": 0, "solve": 0},
              "applied": 0, "clipped": 0, "relay": None}
        # Relay: asking for BOTH accelerators means "Anderson first, Dombey in
        # the tail", not an error.  The two were measured to win different
        # halves of M16_S2 (at a matched 60 sweeps Anderson(5) sits at -2637 pcm
        # where the plain and Dombey trajectories are still at -6314, yet run to
        # outer_tol Anderson needs 752 sweeps and Dombey 633), so each hands the
        # iteration over at the point where the other one's own gate opens:
        # Dombey only tries below dlam 1e-4, and that is also where Anderson's
        # least-squares stops adding anything the two-point ratio cannot.
        relay = bool(extrapolate) and int(anderson) > 0
        try:
            for outer in range(max_outer):
                it_done = outer + 1
                mode = "p"
                rhat = rn = rex = None      # trace diagnostics for this sweep
                if warm:
                    # Serial warm sweep: populates _Scache/_bterms_by_sys
                    # etc. so that, after the fork, NO worker ever writes
                    # into a shared cache (a worker-side cache fill would
                    # COW-diverge from the parent and is wasted work).
                    phi, qim = self._sweep_all_serial(qnode)
                    self._start_sweep_pool()
                    warm = False
                    sweeps += 1
                elif pending is not None and qnode is pending[0]:
                    # Dombey accepted this source on the previous iteration and
                    # its trial sweep already produced exactly what this
                    # iteration is about to recompute: the sweep is a pure
                    # function of its source argument, so the trial's (phi, qim)
                    # are moved here instead of being recomputed.  Bit-identical
                    # by construction (same arrays, same code path), and it is
                    # what makes the trial sweep cost ONE sweep instead of two.
                    phi, qim = pending[1], pending[2]
                    pending = None
                else:
                    phi, qim = self._sweep_all(qnode)
                    sweeps += 1
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
                # CMFD (when installed) replaces the fission source with the
                # coarse-mesh-corrected one; None => plain update, bit-identical
                s_next = None
                if cmfd is not None and outer >= cmfd.start:
                    s_next = cmfd.source(qim, lam, matidx)
                if s_next is None:
                    s_next = self._source_update(qim, lam, matidx)
                # Fixed-point residual of the source map AT the swept iterate
                # (s_hist[-1] is the source this sweep consumed), computed every
                # iteration because the stop test needs it -- see RES_STOP_FACTOR.
                r_src = float(np.linalg.norm(s_next - s_hist[-1]) /
                              max(1.0, float(np.linalg.norm(s_hist[-1]))))
                F_old = F_new
                # ---------------- Anderson acceleration of the source map ---
                if aa["m"] and not (relay and outer >= 8
                                    and dlam <= RELAY_DLAM):
                    qn_aa = self._aa_update(aa, qnode, s_next)
                    qnode = s_next if qn_aa is None else qn_aa
                    if aa["off"] and verbose:
                        print("  anderson: residual grew 3x in a row, "
                              "acceleration off (plain iteration continues)")
                else:
                    if aa["m"]:
                        # Hand the tail to Dombey for good.  Its history is
                        # dropped rather than kept as a fallback: AA is exactly
                        # the accelerator that is not helping at dlam 1e-4, and
                        # a live history would keep perturbing the lam sequence
                        # whose ratio Dombey is trying to measure.
                        aa["relay"] = {"it": outer + 1, "dlam": dlam}
                        aa["m"] = 0
                        del aa["s"][:], aa["f"][:], aa["res"][:]
                        if verbose:
                            print(f"  anderson -> dombey: relayed at "
                                  f"it {outer + 1}, dlam={dlam:.2e}")
                    qnode = s_next
                # ---------------- Dombey two-point source extrapolation -------
                if cool > 0:
                    cool -= 1
                elif (extrapolate and not aa["m"] and outer >= 8 and n_fail < 6
                      and dlam < 1e-4
                      and len(lam_hist) == 3 and len(s_hist) == 2
                      and lam_hist[1] != lam_hist[0]):
                    k2, k1, k0 = lam_hist
                    rhat = (k0 - k1) / (k1 - k2)
                    if 0.5 < rhat < 0.999:
                        s_n, s_nm1 = s_hist[-1], s_hist[-2]
                        s_ex = (s_n - rhat * s_nm1) / (1.0 - rhat)
                        phi_ex, qim_ex = self._sweep_all(s_ex)
                        extra_sweeps += 1
                        sweeps += 1
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
                        rn = r_n
                        s_ex_up = self._source_update(qim_ex, lam_ex, matidx)
                        r_ex = float(np.linalg.norm(s_ex_up - s_ex) /
                                     max(1.0, float(np.linalg.norm(s_ex))))
                        if r_ex < DOMBEY_ACCEPT * r_n:
                            # accept: continue the power sequence from S_ex
                            qnode = s_ex
                            # the NEXT iteration sweeps s_ex, and the trial
                            # sweep just above already computed its (phi, qim)
                            pending = (s_ex, phi_ex, qim_ex)
                            cool = DOMBEY_COOL
                            n_fail = 0
                            mode = "d"
                            rex = r_ex
                            if verbose:
                                print(f"  dombey: rhat={rhat:.6f}  "
                                      f"residual {r_n:.2e} -> {r_ex:.2e}  "
                                      f"keff_ex={lam_ex:.7f}")
                        else:
                            n_fail += 1
                            mode = "r"
                            rex = r_ex
                            if verbose:
                                print(f"  dombey reject: rhat={rhat:.6f}  "
                                      f"r_n={r_n:.2e} r_ex={r_ex:.2e}  "
                                      f"ratio={r_ex/max(r_n,1e-300):.2f} "
                                      f"(fail {n_fail})")
                            if n_fail >= DOMBEY_FAILS:
                                # not a pure geometric mode yet: cool down and
                                # let it retry later, do not disable outright
                                cool = DOMBEY_COOLFAIL
                                n_fail = 0
                # -----------------------------------------------------------------
                trace.append((outer + 1, dlam, r_src, mode, float(lam), rhat,
                             rn, rex))
                if verbose and (outer % 5 == 0 or dlam < outer_tol):
                    print(f"it {outer:4d}  keff={lam:.7f}  dkeff={dlam:.2e}  "
                          f"F={F_new:.5e}")
                if dlam < outer_tol and r_src <= RES_STOP_FACTOR * outer_tol:
                    break

        finally:
            self._stop_sweep_pool()
        self.outer_iterations = it_done
        # Dombey's trial sweeps are real sweeps: ``outer_iterations`` alone
        # understates the work, so both numbers are published and the CLI
        # prints them together.
        self.extra_sweeps = extra_sweeps
        # sweeps the solver actually paid for: ``outer_iterations +
        # trial_sweeps`` counts one iteration per accepted extrapolation whose
        # sweep had already been done by the trial above, so it overstates the
        # work by the number of acceptances.
        self.sweeps_executed = sweeps
        self.trace = trace
        # the source-map residual of the LAST swept iterate: a run that stopped
        # because dlam was small can now be read for whether its SOURCE was
        # converged too (see RES_STOP_FACTOR).
        self.source_residual = r_src
        self.aa_stats = {"depth": int(anderson), "applied": int(aa["applied"]),
                         "off": bool(aa["off"]),
                         # sweeps really executed, vs the outer+trial the report
                         # prints (the difference is the accepted trials whose
                         # sweep the memo above reused)
                         "sweeps_executed": int(sweeps),
                         "dombey_accept": DOMBEY_ACCEPT,
                         "dombey_cool": DOMBEY_COOL,
                         "dombey_fails": DOMBEY_FAILS,
                         "dombey_coolfail": DOMBEY_COOLFAIL,
                         # {it, dlam} of the handover on a relayed run, None if
                         # Anderson owned the whole iteration
                         "relayed_to_dombey": aa["relay"],
                         "source_residual": r_src,
                         "res_stop_factor": RES_STOP_FACTOR,
                         # steps whose linear combination undershot the plain
                         # source's own floor and had to be lifted back
                         "clipped": int(aa["clipped"]),
                         "rejected": {k: int(v) for k, v in aa["why"].items()}}
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
