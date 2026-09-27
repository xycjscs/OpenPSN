# -*- coding: utf-8 -*-
"""Dry check for the pre-fork prebuild fix (2026-09-14).

Production-exact layout: 68x68 S1 UO2, M24, A=3, closure=directional,
GMRES pool, pairs=True, 24 workers x 1T (BUDGET=24), pin24 +
RLIMIT_AS 32G — the same guards as rect17_run.py.  Runs only
DRY_OUTERS (default 2) outer iterations with an unreachable tol, so it
exercises prebuild + fork + a full task wave (each worker builds its
real GMRES solve) and then stops BEFORE real work time.

A monitor thread samples, every 2 s:
  * sum of VmRSS over the whole process tree (parent + 24 workers,
    found by argv match on this script name)
  * container cgroup v2 memory.current (delta vs start)
and reports peaks.  Pass criteria (vs 62 GB host / 100 GB cgroup):
  * tree RSS peak < 45 GB
  * no MemoryError, both outers complete.

Usage: DRY_OUTERS=2  tools_ho/dbg_mem_dry.py
"""
import os, sys, time, resource, threading

sys.path.insert(0, '/opt/data/workspace/OpenPSN')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ['PSN_CORE_BUDGET'] = '24'

EX = ('/opt/data/workspace/OpenPSN'
      '/examples/c5g7/study2_rectangular'
      '/c5g7_rect17_split4_uo2_assembly.yaml')
MAXOUT = int(os.environ.get('DRY_OUTERS', '2'))

os.sched_setaffinity(0, set(range(24)))          # pin24 (production)
resource.setrlimit(resource.RLIMIT_AS,            # 32G (production)
                   (int(32 * 2 ** 30), -1))

CG = '/sys/fs/cgroup/memory.current'


def cgroup_gb():
    try:
        with open(CG) as f:
            return int(f.read().strip()) / 2 ** 30
    except OSError:
        return float('nan')


def mem_available_gb():
    try:
        for ln in open('/proc/meminfo'):
            if ln.startswith('MemAvailable:'):
                return int(ln.split()[1]) / 1e6
    except OSError:
        pass
    return float('nan')


def tree_rss_gb():
    tot, n, vsz_max, vsz_max_pid = 0.0, 0, 0.0, ''
    for d in os.listdir('/proc'):
        if not d.isdigit():
            continue
        try:
            with open(f'/proc/{d}/cmdline', 'rb') as f:
                if b'dbg_mem_dry' not in f.read():
                    continue
            rss, vsz = 0, 0
            with open(f'/proc/{d}/status') as f:
                for ln in f:
                    if ln.startswith('VmRSS:'):
                        rss = int(ln.split()[1])
                    elif ln.startswith('VmSize:'):
                        vsz = int(ln.split()[1])
            tot += rss / 1e6
            if vsz > vsz_max:
                vsz_max, vsz_max_pid = vsz, d
            n += 1
        except OSError:
            pass
    return tot, n, vsz_max / 1e6, vsz_max_pid


peak_tree = [0.0, 0, 0.0, 0.0]     # rss, nprocs, cgroup_delta, t
avail_min = [float('inf'), 0.0]    # min MemAvailable, time of min
vsz_peak, vsz_peak_pid = 0.0, ''   # max VmSize in tree (RLIMIT_AS guard)
cg_base = cgroup_gb()
t_start = time.time()
_stop = threading.Event()


def monitor():
    global vsz_peak, vsz_peak_pid
    while not _stop.is_set():
        rss, n, vsz, vsz_pid = tree_rss_gb()
        cg = cgroup_gb() - cg_base
        avail = mem_available_gb()
        if avail < avail_min[0]:
            avail_min[:] = [avail, time.time() - t_start]
        if vsz > vsz_peak:
            vsz_peak, vsz_peak_pid = vsz, vsz_pid
        if rss > peak_tree[0]:
            peak_tree[:] = [rss, n, cg, time.time() - t_start]
        print(f"  [mon] t={time.time()-t_start:6.0f}s tree_rss={rss:5.2f}GB "
              f"procs={n:2d} maxVSZ={vsz:5.2f}GB(pid {vsz_pid}) "
              f"cg_delta={cg:5.2f}GB avail={avail:5.2f}GB", flush=True)
        _stop.wait(2)


def main():
    from psn2d import model, __main__ as M, highsweep, mckern
    mckern.ENABLED = True
    spec = model.load_spec(EX)
    case = {'name': 'dry', 'model': 'ty3', 'I': 30, 'M': 24,
            'subdivide': 1, 'A': 3, 'closure': 'directional_matrix'}
    spec['solver'].update({'keff_tol': 1e-7, 'max_outer': MAXOUT,
                           'threads': 1})
    th = threading.Thread(target=monitor, daemon=True)
    th.start()
    t0 = time.time()
    psn = M.build_solver(spec, case, threads=1, mem_limit_gb=32)
    t_build = time.time() - t0
    _, _, _, info = highsweep.keff_highorder_gmres(
        psn, A=3, branch='directional', max_outer=MAXOUT,
        outer_tol=1e-30, rtol=1e-7, maxiter=800,
        pairs=True, nworkers=24, verbose=True)
    _stop.set()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    hwm = 0
    with open('/proc/self/status') as f:
        for ln in f:
            if ln.startswith('VmHWM:'):
                hwm = int(ln.split()[1]) / 1e6
    print(f"DRY-DONE wall={time.time()-t0:.0f}s (build={t_build:.0f}s) "
          f"parent_rss_peak={hwm:.2f}GB ru_maxrss={peak:.2f}GB "
          f"tree_rss_peak={peak_tree[0]:.2f}GB (procs={peak_tree[1]} "
          f"cg_delta={peak_tree[2]:.2f}GB at t={peak_tree[3]:.0f}s) "
          f"host_avail_min={avail_min[0]:.2f}GB (at t={avail_min[1]:.0f}s) "
          f"outers={info['outer_iterations']} "
          f"converged={info['converged']} "
          f"nworkers={info['nworkers']} nblocks={info['nblocks']}",
          flush=True)


if __name__ == '__main__':
    main()
