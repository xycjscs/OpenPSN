# -*- coding: utf-8 -*-
"""dbg_m8mem.py — M8 assembly memory benchmark: GMRES process pool path.

Benchmark protocol (per the established convention): each measured
configuration runs in an independent subprocess; peaks are read from
/proc/<pid>/status VmHWM (ru_maxrss in-process is contaminated by
high-water).  Here the parent HWM is the COW-shared baseline (psn +
factor tables) and each worker's PRIVATE memory is its VmHWM minus the
parent HWM at fork (workers only write their own nl.u / GMRES Arnoldi
workspace).  System peak = parent private + shared + sum(worker
private), approximated as parent VmHWM + sum(worker VmHWM - baseline).

Reference: the direct-LU path's M8 peak was 54 GB (Table 3 record,
21 SuperLU factors resident in the parent).
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import sys, time, json

sys.path.insert(0, '/opt/data/workspace/OpenPSN')
import numpy as np
from psn2d import model as mdl
from psn2d import __main__ as PM

EX = '/opt/data/workspace/OpenPSN/examples/c5g7/study2_rectangular'
N_OUT = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def hwm_kb(pid=None):
    with open(f'/proc/{pid or os.getpid()}/status') as fh:
        for line in fh:
            if line.startswith('VmHWM:'):
                return int(line.split()[1])
    return -1


def rss_kb(pid=None):
    with open(f'/proc/{pid or os.getpid()}/status') as fh:
        for line in fh:
            if line.startswith('VmRSS:'):
                return int(line.split()[1])
    return -1


def children():
    me = os.getpid()
    out = []
    for d in os.listdir('/proc'):
        if not d.isdigit():
            continue
        try:
            with open(f'/proc/{d}/stat') as fh:
                parts = fh.read().rsplit(')', 1)[1].split()
            if int(parts[1]) == me:      # ppid field
                out.append(int(d))
        except (OSError, IndexError):
            pass
    return out


spec = mdl.load_spec(EX + '/c5g7_rect_uo2_assembly.yaml')
case = {'name': 'M8', 'model': 'ty3', 'I': 30, 'M': 8,
        'subdivide': 1, 'A': 3, 'closure': 'directional_matrix'}
spec['solver'].update({'keff_tol': 1e-10, 'max_outer': 200, 'threads': 1})
t0 = time.time()
psn = PM.build_solver(spec, case, threads=1, mem_limit_gb=200)
if not hasattr(psn, '_groups'):
    psn._vectorize_setup()
h_build = hwm_kb()
print(f"grid {psn.nx}x{psn.ny} nodes={psn.nodes} M={psn.M} "
      f"K={psn.M + 1} ncol={psn.ncol} systems={psn.I * psn.ng}",
      flush=True)
print(f"build: {time.time()-t0:.1f}s  HWM={h_build/1e6:.3f} GB", flush=True)

from psn2d import highsweep as hs

# factor tables (parent, pre-fork) -- KB-scale per (i,g)
t0 = time.time()
locals_ = {(i, g): hs.HoNodeLocal(psn, i, g, 3, 'directional',
                                  gs_tol=1e-11, max_gs=1)
           for i in range(psn.I) for g in range(psn.ng)}
h_fac = hwm_kb()
print(f"factor tables ({len(locals_)}): {time.time()-t0:.1f}s  "
      f"HWM={h_fac/1e6:.3f} GB", flush=True)

# run the pool for N_OUT outer iterations; poll worker HWM in parallel
import multiprocessing as mp
IDX = [(i, g) for i in range(psn.I) for g in range(psn.ng)]
hs._POOL, hs._IDX = locals_, IDX
hs._RTOL, hs._MAXIT = 1e-11, 800

ctx = mp.get_context('fork')
pool = ctx.Pool(len(IDX), initializer=hs._gmres_pool_init)

matidx = psn.mat[psn.j_idx, psn.i_idx]
ng = psn.ng
mom = np.zeros((ng, psn.nodes, 5))
mom[:, :, 0] = 1.0
qnode = psn._source_update(mom, 1.0, matidx)
h_fork = hwm_kb()
worker_hwm = {}
worker_rss = {}
stop = [False]

def poll():
    while not stop[0]:
        for c in children():
            try:
                h = hwm_kb(c)
                r = rss_kb(c)
                worker_hwm[c] = max(worker_hwm.get(c, 0), h)
                worker_rss[c] = max(worker_rss.get(c, 0), r)
            except OSError:
                pass
        time.sleep(0.05)

pt = None
try:
    import threading
    pt = threading.Thread(target=poll, daemon=True)
    pt.start()
    for outer in range(N_OUT):
        t1 = time.time()
        args = [(idx, qnode[g]) for idx, (i, g) in enumerate(IDX)]
        results = pool.map(hs._gmres_pool_task, args)
        dt = time.time() - t1
        phi = np.zeros((ng, psn.nodes))
        qim = np.zeros((ng, psn.nodes, 5))
        for (i, g), (o, nv, r) in zip(IDX, results):
            phi[g] += o[:, 0]
            qim[g] += o
        area_w = psn.area.ravel()
        fr = sum(psn.nuSf[g, matidx] * phi[g] for g in range(ng)) * area_w
        print(f"outer {outer}: {dt:.2f}s  max_matvec="
              f"{max(r[1] for r in results)}  "
              f"worst_res={max(r[2] for r in results):.1e}", flush=True)
        lam = 1.0
        qnode = psn._source_update(qim, lam, matidx)
finally:
    stop[0] = True
    pool.close()
    pool.join()
    hs._POOL = hs._IDX = None

h_end = hwm_kb()
mw = max(worker_hwm.values()) if worker_hwm else 0
mr = max(worker_rss.values()) if worker_rss else 0
n_w = len(worker_hwm)
total_gb = (h_end + sum(v - h_fork for v in worker_hwm.values())) / 1e6
print(f"\nparent HWM (shared baseline at fork) = {h_fork/1e6:.3f} GB",
      flush=True)
print(f"parent HWM at end                      = {h_end/1e6:.3f} GB",
      flush=True)
print(f"workers observed: {n_w}", flush=True)
print(f"worker peak HWM (incl. inherited)      = {mw/1e6:.3f} GB",
      flush=True)
print(f"worker peak PRIVATE (HWM - baseline)   = "
      f"{(mw - h_fork)/1e6:.3f} GB", flush=True)
print(f"worker peak RSS (incl. shared pages)   = {mr/1e6:.3f} GB",
      flush=True)
print(f"SYSTEM PEAK approx (parent + sum private) = "
      f"{total_gb:.2f} GB", flush=True)
print("reference: direct-LU M8 peak = 54 GB (Table 3)", flush=True)
print("DONE", flush=True)
