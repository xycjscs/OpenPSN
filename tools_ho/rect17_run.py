# -*- coding: utf-8 -*-
"""rect4 (17x17 assembly, every pin fuel split 2x2 = 4 blocks) — A=3 sweep.

Geometry (user 2026-09-14: "rect4 = assembly, pin 燃料 2x2 分割"):
  c5g7_rect17_split4_{uo2,mox}_assembly.yaml — 68x68 base nodes
  (17x17 pins, every pin's center square split 2x2), pure XS,
  all-reflective.  S1 = 68x68, S2 = 136x136 (subdivide 2).

Sweep: S=1,2 x M=2,4,8,12,16,24 x UO2/MOX = 24 points, A=3
directional closure, GMRES pool (lever D: pairs+numba, 24w x 1T,
2026-09-14 combo bench winner 3.83x).

Order: per material, all S1 points first (dry-verified safe,
measured peak 21.2 GB), then S2 with M ascending (low M first —
smaller d, smaller LU — so any 32 GiB-cap infeasibility shows up
on the high-M points last, cheap points are not wasted after it).

Memory guards (2026-09-14 user rules): fresh subprocess per point,
RLIMIT_AS 32 GiB + CPU affinity 0-23 (inherited by pool workers),
so an infeasible point fails gracefully (logged OOM) without
touching the host.

Protocol: power iteration, closure=directional_matrix, keff tol
1e-7 (uniform, commit 79fda66).

Usage:
  rect17_run.py                       # coordinator (24 points)
  rect17_run.py point <key> <M> <S>   # single point (child)
"""
import os, re, sys, time, resource, subprocess

sys.path.insert(0, '/opt/data/workspace/OpenPSN')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('PSN_CORE_BUDGET', '24')

EX = '/opt/data/workspace/OpenPSN/examples/c5g7/study2_rectangular'
LOG = '/opt/data/workspace/OpenPSN/tools_ho/rect17_run.log'
WATCHDOG_GB = 32.0

def _pin_24():
    """Hard 24-core protection (2026-09-14 user rule): pin this process
    to CPUs 0-23 via sched_setaffinity — every thread created later
    (pool workers, numba/OpenMP) inherits the mask, so total CPU
    throughput is kernel-capped at 24 cores regardless of what any
    library spawns.  cgroups are read-only in this container, so
    affinity is the hard guard."""
    import os as _os
    _os.sched_setaffinity(0, set(range(24)))


def _mem_guard_32g():
    """Hard 32 GB memory protection: RLIMIT_AS = 32 GiB virtual
    backstop (kernel-enforced; RSS <= VSZ always)."""
    import resource as _resource
    _resource.setrlimit(_resource.RLIMIT_AS,
                        (int(32 * 2 ** 30), -1))
POINT_TIMEOUT_S = 12 * 3600

M_LIST = [2, 4, 8, 12, 16, 24]   # face modes
S_LIST = [1, 2]                  # S1=68x68, S2=136x136
A_DEG = 3
RTOL = 1e-7      # keff tolerance, uniform 1e-7 (user, 2026-09-14)
MAXITER = 800

PAIRS = True     # lever 1: angle-pair block GMRES (auto-off at M=2)
NUMBA = True     # lever 3: numba kernels
NWORKERS = 24    # pool workers (24-thread budget)

PROBS = {
    'uo2': {'spec': EX + '/c5g7_rect17_split4_uo2_assembly.yaml',
            'tag': 'R17UO2', 'kref': 1.3347260},   # rect3x3 anchor
    'mox': {'spec': EX + '/c5g7_rect17_split4_mox_assembly.yaml',
            'tag': 'R17MOX', 'kref': 1.1851053},
}


def log(line):
    with open(LOG, 'a') as f:
        f.write(line + '\n'); f.flush(); os.fsync(f.fileno())
    print(line, flush=True)


def done_set():
    """(key, M, S) already reported a CONVERGED keff in this log AND the
    field npz exists on disk carrying BOTH flux and fission (2026-09-15:
    every point must carry its flux + fission fields for the
    Figure-2-style comparison; pre-dump-run converged points and
    fission-only npz get re-run)."""
    s = set()
    pat = re.compile(r'(R17UO2|R17MOX) M(\d+)_S(\d+) keff=')
    if os.path.exists(LOG):
        for ln in open(LOG):
            m = pat.search(ln)
            if m and 'NOT_CONVERGED' not in ln:
                key = 'uo2' if m.group(1) == 'R17UO2' else 'mox'
                npz = (f'/opt/data/workspace/c5g7/'
                       f'psn_r17_{key}_M{m.group(2)}_S{m.group(3)}.npz')
                if os.path.exists(npz):
                    try:
                        import numpy as _np
                        d = _np.load(npz)
                        if 'flux' in d.files and 'fission' in d.files:
                            s.add((key, int(m.group(2)), int(m.group(3))))
                    except Exception:
                        pass
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
    """Single point in this fresh process, GMRES pool (config D)."""
    from psn2d import model, __main__ as M, highsweep, mckern
    from psn2d.dump import dump_fields
    import numpy as _np
    mckern.ENABLED = bool(NUMBA)
    P = PROBS[key]
    t0 = time.time()
    spec = model.load_spec(P['spec'])
    case = {'name': f'M{MM}_S{S}', 'model': 'ty3', 'I': 30, 'M': MM,
            'subdivide': S, 'A': A_DEG, 'closure': 'directional_matrix'}
    spec['solver'].update({'keff_tol': 1e-7, 'max_outer': 6000,
                           'threads': 1})
    _pin_24()          # hard: 24-core affinity (inherited by children)
    _mem_guard_32g()   # hard: RLIMIT_AS 32 GiB backstop
    try:
        psn = M.build_solver(spec, case, threads=1, mem_limit_gb=32)
        k, phi, qim, h = highsweep.keff_highorder_gmres(
            psn, A=A_DEG, branch='directional', max_outer=6000,
            outer_tol=1e-7, rtol=RTOL, maxiter=MAXITER,
            pairs=PAIRS, nworkers=NWORKERS, verbose=False)
        dt = time.time() - t0
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        dref = (k - P['kref']) * 1e5
        tagline = '' if h['converged'] else ' NOT_CONVERGED'
        # fission + flux fields: always save, every point (user 2026-09-15)
        # via the psn2d package feature (psn2d/dump.py)
        _out = f'/opt/data/workspace/c5g7/psn_r17_{key}_M{MM}_S{S}.npz'
        dump_fields(psn, phi, _out, block=4, metadata={
            'keff': float(k), 'M': _np.int64(MM), 'S': _np.int64(S),
            'converged': bool(h['converged']),
            'backend': _np.array('gmres'), 'time_s': float(dt)})
        print(f"{P['tag']} M{MM}_S{S} keff={k:.7f} A={A_DEG} K={A_DEG+1} "
              f"inner=gmres nodes={psn.nodes} "
              f"iters={h['outer_iterations']} "
              f"matvecs={h['gmres_matvecs_total']} "
              f"dref={dref:+.1f}pcm wall={dt:.0f}s "
              f"peak={peak:.2f}GB{tagline} dump={_out}", flush=True)
    except MemoryError:
        print(f"{P['tag']} M{MM}_S{S} OOM at {time.time()-t0:.0f}s",
              flush=True)
    except Exception as e:
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
            p.kill(); killed = f'TIMEOUT {POINT_TIMEOUT_S//3600}h'; break
        rss = _child_rss_gb(p.pid)
        if rss is not None and rss > WATCHDOG_GB:
            p.kill()
            killed = f'INFEASIBLE (RSS {rss:.1f}GB > {WATCHDOG_GB:.0f}GB)'
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
            f'child rc={p.returncode} (no result)'
    return tail


def run_point_child_log(key, MM, S, t0):
    log(f'--- {PROBS[key]["tag"]} M{MM}_S{S} start (fresh subprocess, '
        f'gmres pool, pairs={PAIRS} numba={NUMBA} w={NWORKERS}) '
        f't={time.time()-t0:.0f}s')
    return run_child(key, MM, S, t0) + f" elapsed={time.time()-t0:.0f}s"


if __name__ == '__main__':
    _pin_24()
    if len(sys.argv) >= 5 and sys.argv[1] == 'point':
        run_point(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
        sys.exit(0)

    log(f'--- RECT4 (17x17 split-2x2) A3 SWEEP start (A={A_DEG}, '
        f'M={M_LIST}, S={S_LIST}, closure=directional_matrix, '
        f'inner=gmres pool, pairs={PAIRS} numba={NUMBA} w={NWORKERS}, '
        f'budget=24, tol=1e-7, 24 points) ---')
    t0 = time.time()
    results = {}
    done = done_set()
    for key in ['uo2', 'mox']:
        for S in S_LIST:          # all S1 first (dry-verified), then S2
            for MM in M_LIST:     # M ascending: cheap points first
                if (key, MM, S) in done:
                    results[(key, MM, S)] = \
                        f'{PROBS[key]["tag"]} M{MM}_S{S} INHERITED ' \
                        f'(converged keff in log)'
                    log(results[(key, MM, S)])
                    continue
                r = run_point_child_log(key, MM, S, t0)
                results[(key, MM, S)] = r
                log(r)
    # final table
    for key in ['uo2', 'mox']:
        tag = PROBS[key]['tag']
        for S in S_LIST:
            cells = []
            for MM in M_LIST:
                m = re.search(r'M\d+_S\d+ keff=([0-9.]+)',
                              results.get((key, MM, S), ''))
                cells.append(f'{m.group(1)}' if m else '-')
            log(f'{tag} S{S}: ' + '  '.join(f'M{MM}={c}'
                                            for MM, c in zip(M_LIST,
                                                             cells)))
        for MM in M_LIST:
            m1 = re.search(r'M\d+_S\d+ keff=([0-9.]+)',
                           results.get((key, MM, 1), ''))
            m2 = re.search(r'M\d+_S\d+ keff=([0-9.]+)',
                           results.get((key, MM, 2), ''))
            if m1 and m2:
                d = (float(m2.group(1)) - float(m1.group(1))) * 1e5
                log(f'{tag} M{MM} S2-S1: {d:+.1f} pcm')
    log(f'RECT4 A3 SWEEP DONE in {time.time()-t0:.0f}s')
