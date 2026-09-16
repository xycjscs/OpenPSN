# -*- coding: utf-8 -*-
"""rect4x4 S x M sweep — UO2/MOX x S1/S2 x M=2..24, A=3, directional.

New geometry (2x2 fuel pins + 1 water ring), all-reflect, pure XS.
  S1 = 4x4 nodes   (subdivide=1, base grid 4x4)
  S2 = 8x8 nodes   (subdivide=2, uniform 2x2 split of every node)

OpenMC reference: user decision 2026-09-14 — reuse the rect3x3
anchors (UO2 1.3347260 / MOX 1.1851053).  Note: rect4x4 is a 2x2-pin
cluster, so its keff is NOT expected to match the 3x3-pin reference
closely; the dref is reported for the record.  The real validation
is the S1-vs-S2 cross-consistency: both meshes discretize the SAME
physics, so their keff must agree to the outer convergence
tolerance.  A large |S2 - S1| flags mesh under-resolution, not a bug.

Protocol: same as dir_table3_sweep_gmres.py — fresh subprocess per
point, RLIMIT_AS 128GB backstop, RSS watchdog 50GB, power iteration,
closure=directional_matrix, A=3.

Config (inner solver) chosen from the 24-thread combo bench once it
lands; P/NB below are set to the winning lever combination.

Sweep: S=1..2 x M=2,4,8,12,16,24 x UO2/MOX (24 points; the four
M24 points from the 2026-09-14 4-point run are inherited from log).

Usage:
  rect4x4_run.py                    # coordinator (full S x M sweep)
  rect4x4_run.py point <key> <M> <S>   # single point (child)
"""
import os, re, sys, time, resource, subprocess, threading

sys.path.insert(0, '/opt/data/workspace/OpenPSN')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('PSN_CORE_BUDGET', '24')

EX = '/opt/data/workspace/OpenPSN/examples/c5g7/study2_rectangular'
LOG = '/opt/data/workspace/OpenPSN/tools_ho/rect4x4_run.log'
VSZ_BACKSTOP_GB = 32.0   # HARD limit (user 2026-09-14)
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
    backstop (kernel-enforced; RSS <= VSZ always).  Replaces the old
    128 GB backstop."""
    import resource as _resource
    _resource.setrlimit(_resource.RLIMIT_AS,
                        (int(32 * 2 ** 30), -1))
POINT_TIMEOUT_S = 12 * 3600

M_LIST = [2, 4, 8, 12, 16, 24]  # face modes
S_LIST = [1, 2]  # S1=4x4, S2=8x8
A_DEG = 3
RTOL = 1e-7       # keff tolerance, uniform 1e-7 (user, 2026-09-14)
MAXITER = 800

# inner-solver lever combo (set after combo bench lands)
PAIRS = True     # lever 1: angle-pair block GMRES
NUMBA = True     # lever 3: numba _mc_b
NWORKERS = 24    # pool workers (<= 24-thread budget)

PROBS = {
    'uo2': {'spec': EX + '/c5g7_rect4x4_uo2_assembly.yaml', 'tag': 'R4UO2',
            'kref': 1.3347260},   # rect3x3 anchor (user decision 2026-09-14)
    'mox': {'spec': EX + '/c5g7_rect4x4_mox_assembly.yaml', 'tag': 'R4MOX',
            'kref': 1.1851053},
}


def log(line):
    with open(LOG, 'a') as f:
        f.write(line + '\n'); f.flush(); os.fsync(f.fileno())
    print(line, flush=True)


def done_set():
    """(key, M, S) already reported a CONVERGED keff in this log."""
    s = set()
    pat = re.compile(r'(R4UO2|R4MOX) M(\d+)_S(\d+) keff=')
    if os.path.exists(LOG):
        for ln in open(LOG):
            m = pat.search(ln)
            if m and 'NOT_CONVERGED' not in ln:
                s.add(('uo2' if m.group(1) == 'R4UO2' else 'mox',
                       int(m.group(2)), int(m.group(3))))
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
    """Single point in this fresh process, GMRES pool."""
    from psn2d import model, __main__ as M, highsweep, mckern
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
        nworkers = min(21 * (MM // 2) if MM > 2 else 21, NWORKERS)
        k, phi, qim, h = highsweep.keff_highorder_gmres(
            psn, A=A_DEG, branch='directional', max_outer=6000,
            outer_tol=1e-7, rtol=RTOL, maxiter=MAXITER,
            pairs=PAIRS, nworkers=nworkers, verbose=False)
        dt = time.time() - t0
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        dref = (k - P['kref']) * 1e5
        tagline = '' if h['converged'] else ' NOT_CONVERGED'
        print(f"{P['tag']} M{MM}_S{S} keff={k:.7f} A={A_DEG} K={A_DEG+1} "
              f"inner=gmres nodes={psn.nodes} iters={h['outer_iterations']} "
              f"matvecs={h['gmres_matvecs_total']} "
              f"dref={dref:+.1f}pcm wall={dt:.0f}s "
              f"peak={peak:.2f}GB{tagline}", flush=True)
    except MemoryError:
        print(f"{P['tag']} M{MM}_S{S} OOM at {time.time()-t0:.0f}s", flush=True)
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
    if not re.match(rf"{PROBS[key]['tag']} M{MM}_S{S} (keff=|OOM|ERROR)", tail):
        errl = (err or '').strip().splitlines()
        tail = (errl[-1] if errl else '') or f'child rc={p.returncode} (no result)'
    return tail


def run_point_child_log(key, MM, S, t0):
    budget = min(21 * (MM // 2) if MM > 2 else 21, NWORKERS)
    log(f'--- {PROBS[key]["tag"]} M{MM}_S{S} start (fresh subprocess, '
        f'gmres pool, pairs={PAIRS} numba={NUMBA} w={budget}) '
        f't={time.time()-t0:.0f}s')
    return run_child(key, MM, S, t0) + f" elapsed={time.time()-t0:.0f}s"


if __name__ == '__main__':
    _pin_24()
    if len(sys.argv) >= 5 and sys.argv[1] == 'point':
        run_point(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
        sys.exit(0)

    log(f'--- RECT4X4 SxM SWEEP start (A={A_DEG}, M={M_LIST}, '
        f'S={S_LIST}, closure=directional_matrix, inner=gmres pool, '
        f'pairs={PAIRS} numba={NUMBA} w<={NWORKERS}, budget=24) ---')
    t0 = time.time()
    results = {}
    done = done_set()
    for key in ['uo2', 'mox']:
        for S in S_LIST:
            for MM in M_LIST:
                if (key, MM, S) in done:
                    results[(key, MM, S)] = \
                        f'{PROBS[key]["tag"]} M{MM}_S{S} ' \
                        f'INHERITED (converged keff in log)'
                    log(results[(key, MM, S)])
                    continue
                r = run_point_child_log(key, MM, S, t0)
                results[(key, MM, S)] = r
                log(r)
    # cross-consistency: S2 vs S1 per (material, M)
    for key in ['uo2', 'mox']:
        for MM in M_LIST:
            m1 = re.search(r'keff=([0-9.]+)',
                           results.get((key, MM, 1), ''))
            m2 = re.search(r'keff=([0-9.]+)',
                           results.get((key, MM, 2), ''))
            if m1 and m2:
                d = (float(m2.group(1)) - float(m1.group(1))) * 1e5
                log(f"{PROBS[key]['tag']} M{MM} S2-S1 consistency: "
                    f"{d:+.1f} pcm")
    log(f'RECT4X4 SxM SWEEP DONE in {time.time()-t0:.0f}s')
