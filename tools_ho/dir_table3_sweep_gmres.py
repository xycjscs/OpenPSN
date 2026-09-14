# -*- coding: utf-8 -*-
"""Table 3 S x M sweep — GMRES process-pool inner sweep (A=3, directional).

Rectangle 3x3 geometry (17x17 pins x 3x3 subdivide -> 51x51 base grid),
pure XS, all-reflect:
  DIRUO2 : c5g7_rect_uo2_assembly.yaml   (kref 1.3347260, OpenMC)
  DIRMOX : c5g7_rect_mox_assembly.yaml   (kref 1.1851053, OpenMC)

Sweeps S=1..6 (pin subdivide) x M=2,4,8,12,16,24 (face modes), A=3.

Why this script exists: the LU-direct path (dir_table3_sweep.py) hit a
memory wall — M8_S1 peaked at 53.8 GB (21 SuperLU factors resident)
and M>=12 / S>=2 never ran.  The GMRES pool path (keff_highorder_gmres)
measures 3.65 GB at M8_S1 and 6.41 GB at M12_S1, with keff bit-
identical to LU (DKEFF <= 2e-14, e2e3/e2e_A).  This sweep re-runs the
UNREACHED points on the GMRES pool; points already completed in the
LU log (S1: M2/M4/M8, both materials) are inherited, not re-run —
their keff values are path-independent to machine precision.

Protocol (inherited from dir_table3_sweep.py, 2026-09-10 policy):
  * fresh subprocess per point (VSZ-accumulation OOM cascade guard)
  * RLIMIT_AS = 128 GB virtual backstop
  * coordinator true-RSS watchdog (5 s poll) > 50 GB -> INFEASIBLE
  * plain power iteration, closure=directional_matrix, A=3, K=4, TY3
  * threads=1 (BLAS pinned to 1 per worker; parallelism is the
    process pool over 21 (i,g) systems, static shards)

Usage:
  dir_table3_sweep_gmres.py [uo2|mox|all]        # coordinator
  dir_table3_sweep_gmres.py point <key> M S      # single point (child)
"""
import os, re, sys, time, resource, subprocess, threading

sys.path.insert(0, '/opt/data/workspace/OpenPSN')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
EX = '/opt/data/workspace/OpenPSN/examples/c5g7/study2_rectangular'
LOG = '/opt/data/workspace/OpenPSN/tools_ho/dir_table3_sweep_gmres.log'
LEGACY_LOG = '/opt/data/workspace/c5g7/dir_table3_sweep.log'
VSZ_BACKSTOP_GB = 128.0
WATCHDOG_GB = 50.0
POINT_TIMEOUT_S = 12 * 3600

M_LIST = [2, 4, 8, 12, 16, 24]
S_LIST = [1, 2, 3, 4, 5, 6]
A_DEG = 3
RTOL = 1e-7     # keff tolerance, uniform 1e-7 (user, 2026-09-14)
MAXITER = 800

PROBS = {
    'uo2': {'spec': EX + '/c5g7_rect_uo2_assembly.yaml', 'tag': 'DIRUO2',
            'kref': 1.3347260},
    'mox': {'spec': EX + '/c5g7_rect_mox_assembly.yaml', 'tag': 'DIRMOX',
            'kref': 1.1851053},
}


def log(line):
    with open(LOG, 'a') as f:
        f.write(line + '\n'); f.flush(); os.fsync(f.fileno())
    print(line, flush=True)


def done_set(tag):
    """Points with a keff line in THIS log (fresh GMRES result) or in
    the legacy LU log (inherited: keff path-independent to ~1e-14).
    INFEASIBLE/OOM/ERROR entries do NOT count as done — they are
    exactly what the GMRES pool is here to unlock."""
    s = set()
    pat = re.compile(rf'{tag} M(\d+)_S(\d+) keff=')
    for path in (LOG, LEGACY_LOG):
        if os.path.exists(path):
            for ln in open(path):
                m = pat.search(ln)
                if m:
                    s.add((int(m.group(1)), int(m.group(2))))
    return s


def _child_rss_gb(pid):
    try:
        for ln in open(f'/proc/{pid}/status'):
            if ln.startswith('VmRSS:'):
                return int(ln.split()[1]) / 1e6
    except (OSError, ValueError):
        return None
    return None


def run_point(key, MM, S):
    """Single point, executed in this (fresh) process, GMRES pool.

    Lever config D (measured best, dbg_bench_combo24, 24-thread budget,
    2026-09-14): angle-pair sharding + numba kernels, 24 workers x 1T.
    M8: 3.83x vs serial; M2: numba on, 21 workers (pairs is a no-op at
    M=2: 1 block per (i,g))."""
    from psn2d import model
    from psn2d import __main__ as M
    from psn2d import highsweep
    from psn2d import mckern
    mckern.ENABLED = True
    P = PROBS[key]
    t0 = time.time()
    spec = model.load_spec(P['spec'])
    case = {'name': f'M{MM}_S{S}', 'model': 'ty3', 'I': 30, 'M': MM,
            'subdivide': S, 'kref': P['kref'], 'A': A_DEG,
            'closure': 'directional_matrix'}
    spec['solver'].update({'keff_tol': 1e-7, 'max_outer': 6000,
                           'threads': 1})
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    try:
        resource.setrlimit(resource.RLIMIT_AS,
                           (int(VSZ_BACKSTOP_GB * 1e9), hard))
        capped = True
    except (ValueError, OSError):
        capped = False
    try:
        psn = M.build_solver(spec, case, threads=1, mem_limit_gb=120)
        nworkers = min(21 * (MM // 2) if MM > 2 else 21, 24)
        k, phi, qim, hinfo = highsweep.keff_highorder_gmres(
            psn, A=A_DEG, branch='directional',
            max_outer=6000, outer_tol=1e-7,
            rtol=RTOL, maxiter=MAXITER, verbose=False,
            pairs=True, nworkers=nworkers)
        if capped:
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        dt = time.time() - t0
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        dref = (k - P['kref']) * 1e5
        tagline = ('' if hinfo['converged'] else ' NOT_CONVERGED')
        print(f"{P['tag']} M{MM}_S{S} keff={k:.7f} A={A_DEG} K={A_DEG+1} "
              f"inner=gmres nodes={psn.nodes} iters="
              f"{hinfo['outer_iterations']} "
              f"matvecs={hinfo['gmres_matvecs_total']} "
              f"dref={dref:+.1f}pcm wall={dt:.0f}s peak={peak:.2f}GB"
              f"{tagline}", flush=True)
    except MemoryError:
        if capped:
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        print(f"{P['tag']} M{MM}_S{S} OOM (hit {VSZ_BACKSTOP_GB:.0f}GB "
              f"VSZ backstop) at {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        if capped:
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        print(f"{P['tag']} M{MM}_S{S} ERROR {type(e).__name__} "
              f"{str(e)[:80]} at {time.time()-t0:.0f}s", flush=True)


def run_child(key, MM, S, t0):
    p = subprocess.Popen([sys.executable, '-u', os.path.abspath(__file__),
                          'point', key, str(MM), str(S)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True)
    killed = None
    t_end = time.time() + POINT_TIMEOUT_S
    while p.poll() is None:
        if time.time() > t_end:
            p.kill(); killed = f'TIMEOUT after {POINT_TIMEOUT_S//3600}h'
            break
        rss = _child_rss_gb(p.pid)
        if rss is not None and rss > WATCHDOG_GB:
            p.kill()
            killed = (f'INFEASIBLE (RSS {rss:.1f}GB > watchdog '
                      f'{WATCHDOG_GB:.0f}GB)')
            break
        time.sleep(5)
    try:
        out, err = p.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        out, err = '', ''
    if killed:
        return f"{PROBS[key]['tag']} M{MM}_S{S} {killed} at {time.time()-t0:.0f}s"
    lines = (out or '').strip().splitlines()
    tail = lines[-1] if lines else ''
    if not re.match(rf"{PROBS[key]['tag']} M{MM}_S{S} (keff=|OOM|ERROR)",
                    tail):
        errl = (err or '').strip().splitlines()
        tail = (errl[-1] if errl else '') or \
            f'child rc={p.returncode} (no result line)'
    return tail


def channels_for(cpu, MM):
    """Concurrent point channels under the 24-thread budget, M-aware.

    A sweep point forks I*ng = 3*7 = 21 workers (one per (i,g) system)
    in the serial era.  With lever-1 angle-pair sharding each (i,g)
    splits into M/2 blocks, so a point wants 21*(M/2) workers, capped
    at the thread budget.

      M=2 : 21 workers/point  -> 1 point (21 <= 24)
      M=4 : 42 -> capped 24   -> only 1 fits
      M>=8: 24 (capped)       -> only 1 fits

    Each point peaks <1.3 GB RSS (M24_S1 measured 1.20 GB), so the
    constraint is threads, not memory."""
    per_point = 21 if MM <= 2 else 21 * (MM // 2)
    per_point = min(per_point, cpu)
    return max(1, min(2, cpu // per_point))


_BUDGET = None  # set in __main__: core-count budget for point channels
_CORES = 24     # thread budget (host has 48 cores; convention: 24, 2026-09-14)


class CoreBudget:
    """Counted core budget: a point acquires its worker count before
    forking and releases it afterwards, so concurrent points never
    oversubscribe the thread budget (24, 2026-09-14 convention).  At
    24 only a single M2 point (21 workers) fits alongside nothing
    else; all M>=4 points take the whole budget and run exclusive."""

    def __init__(self, total):
        self.total = total
        self.free = total
        self.cv = threading.Condition()

    def acquire(self, n):
        with self.cv:
            while self.free < n:
                self.cv.wait()
            self.free -= n

    def release(self, n):
        with self.cv:
            self.free += n
            self.cv.notify_all()


def run_point_child_log(key, MM, S, t0):
    """Start line + run + result line, thread-safe (log() is
    append+fsync; print order between channels may interleave)."""
    budget = min(21 * (MM // 2) if MM > 2 else 21, _CORES)
    if _BUDGET is not None:
        _BUDGET.acquire(budget)
    try:
        log(f'--- {PROBS[key]["tag"]} M{MM}_S{S} start (fresh subprocess, '
            f'gmres pool, budget={budget}) t={time.time()-t0:.0f}s')
        return (run_child(key, MM, S, t0)
                + f" elapsed={time.time()-t0:.0f}s")
    finally:
        if _BUDGET is not None:
            _BUDGET.release(budget)


if __name__ == '__main__':
    if len(sys.argv) >= 5 and sys.argv[1] == 'point':
        run_point(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
        sys.exit(0)

    MODE = sys.argv[1] if len(sys.argv) > 1 else 'all'
    keys = [MODE] if MODE in PROBS else ['uo2', 'mox']
    CPU = _CORES  # thread budget (48-core host, 24-thread convention)
    _BUDGET = CoreBudget(CPU)
    log(f'--- DIR TABLE3 GMRES SWEEP start (A={A_DEG}, closure='
        f'directional_matrix, inner=gmres static-shard pool, '
        f'rtol={RTOL}, maxiter={MAXITER}, channels=budget-{CPU}) ---')
    log('NOTE: core points blocked (vacuum boundary not validated for '
        'A>0); legacy LU points (S1 M2/M4/M8) inherited from '
        'dir_table3_sweep.log (keff path-independent to ~1e-14)')
    t0 = time.time()
    for key in keys:
        P = PROBS[key]
        done = done_set(P['tag'])
        pending = []
        for S in S_LIST:
            for MM in M_LIST:
                if (MM, S) in done:
                    log(f"{P['tag']} M{MM}_S{S} INHERITED (legacy LU "
                        f'keff, not re-run)')
                else:
                    pending.append((MM, S))
        if not pending:
            continue
        # multi-channel with core-budget semaphore: two M2 points run
        # at once (21+21=42 workers <= 48 cores); larger-M points take
        # the whole budget so they stay exclusive.  Channel count is
        # generous (max points per key); the semaphore serializes
        # anything that does not fit.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        max_workers = max(1, min(len(pending), 2 * len(M_LIST)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut = {}
            for MM, S in pending:
                fut[ex.submit(run_point_child_log, key, MM, S, t0)] \
                    = (MM, S)
            for f in as_completed(fut):
                MM, S = fut[f]
                log(f.result())
    log(f'DIR TABLE3 GMRES SWEEP DONE in {time.time()-t0:.0f}s')
