# -*- coding: utf-8 -*-
"""Table 3 sweep with the NEW high-order directional closure (A=3).

Study II rectangular nodes, pure XS (c5g7-mgxs.h5):
  DIRUO2 : 17x17 UO2 rect assembly (all-reflect)  M=2..24 x S=1..6
  DIRMOX : 17x17 MOX rect assembly (all-reflect)  M=2..24 x S=1..6
  DIRCORE: 51x51 1/4 core (reflect/vacuum)  — BLOCKED: the A3 closure is
           validated for all-reflect boundaries only; vacuum needs an
           independent Robin-condition derivation + Gauss verification
           before it may be used.  Core points are skipped, not run.

Protocol (mirrors rect_sweep.py, 2026-09-10 policy):
  * fresh subprocess per point (VSZ-accumulation OOM cascade guard)
  * RLIMIT_AS = 128 GB virtual backstop
  * coordinator true-RSS watchdog (5 s poll) > 50 GB -> INFEASIBLE
  * high-order closure = plain power iteration (no Dombey: the package
    requires the plain trajectory for acceptance; the legacy Dombey
    WARNING in solver.keff applies a fortiori)
  * closure=directional_matrix, A=3, K=4, TY3

Usage:
  dir_table3_sweep.py [uo2|mox|all]            # coordinate
  dir_table3_sweep.py point [uo2|mox] M S      # single point (child)
"""
import os, re, sys, time, resource, subprocess

sys.path.insert(0, '/opt/data/workspace/OpenPSN')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
EX = '/opt/data/workspace/OpenPSN/examples'
LOG = '/opt/data/workspace/OpenPSN/tools_ho/dir_table3_sweep.log'
VSZ_BACKSTOP_GB = 128.0
WATCHDOG_GB = 50.0
POINT_TIMEOUT_S = 12 * 3600
THREADS = 24   # strict budget (user input 2026-09-12): sweep pool <= 24

M_LIST = [2, 4, 8, 12, 16, 24]
S_LIST = [1, 2, 3, 4, 5, 6]
A_DEG = 3

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
    s = set()
    if os.path.exists(LOG):
        for ln in open(LOG):
            m = re.match(rf'{tag} M(\d+)_S(\d+) (?:keff=|INFEASIBLE|SKIPPED)',
                         ln)
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
    """Single point, executed in this (fresh) process."""
    from psn2d import model
    from psn2d import __main__ as M
    P = PROBS[key]
    t0 = time.time()
    spec = model.load_spec(P['spec'])
    case = {'name': f'M{MM}_S{S}', 'model': 'ty3', 'I': 30, 'M': MM,
            'subdivide': S, 'kref': P['kref'], 'A': A_DEG,
            'closure': 'directional_matrix'}
    # solver tolerances per the packaged sample (plain iteration)
    spec['solver'].update({'keff_tol': 1e-10, 'max_outer': 6000,
                           'threads': THREADS})
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (int(VSZ_BACKSTOP_GB * 1e9), hard))
        capped = True
    except (ValueError, OSError):
        capped = False
    try:
        psn = M.build_solver(spec, case, threads=THREADS, mem_limit_gb=32)
        from psn2d import highorder
        k, phi, qim, hinfo = highorder.keff_highorder(
            psn, A=A_DEG, branch='directional',
            max_outer=6000, outer_tol=1e-10, verbose=False)
        if capped:
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        dt = time.time() - t0
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        dref = (k - P['kref']) * 1e5
        tagline = ('' if hinfo['converged']
                   else ' NOT_CONVERGED')
        print(f"{P['tag']} M{MM}_S{S} keff={k:.7f} A={A_DEG} K={A_DEG+1} "
              f"nodes={psn.nodes} iters={hinfo['outer_iterations']} "
              f"dref={dref:+.1f}pcm wall={dt:.0f}s peak={peak:.2f}GB"
              f"{tagline}", flush=True)
    except MemoryError:
        if capped:
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        print(f"{P['tag']} M{MM}_S{S} OOM (hit {VSZ_BACKSTOP_GB:.0f}GB VSZ "
              f"backstop) at {time.time()-t0:.0f}s", flush=True)
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


if __name__ == '__main__':
    if len(sys.argv) >= 5 and sys.argv[1] == 'point':
        run_point(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
        sys.exit(0)

    MODE = sys.argv[1] if len(sys.argv) > 1 else 'all'
    keys = [MODE] if MODE in PROBS else ['uo2', 'mox']
    log(f'--- DIR TABLE3 SWEEP start (A={A_DEG}, closure=directional_matrix, '
        f'threads={THREADS}, plain iteration) ---')
    log('NOTE: core points blocked (vacuum boundary not validated for A>0)')
    t0 = time.time()
    for key in keys:
        P = PROBS[key]
        done = done_set(P['tag'])
        for S in S_LIST:
            for MM in M_LIST:
                if (MM, S) in done:
                    continue
                log(f'--- {P["tag"]} M{MM}_S{S} start (fresh subprocess) '
                    f't={time.time()-t0:.0f}s')
                log(run_child(key, MM, S, t0)
                    + f" elapsed={time.time()-t0:.0f}s")
    log(f'DIR TABLE3 SWEEP DONE in {time.time()-t0:.0f}s')
