# -*- coding: utf-8 -*-
"""prof_pass.py — profile one red-black pass: b-assembly vs lu_solve,
per-call cost, system size, and GIL/BLAS-oversubscription behaviour
of batched lu_solve under 24 threads x {24,1} OpenBLAS threads."""
import sys, time
sys.path.insert(0, '/opt/data/workspace/OpenPSN')
import numpy as np
from scipy.linalg import lu_solve
from threadpoolctl import threadpool_limits
from psn2d import highsweep as hs
from psn2d import model as mdl
from psn2d import __main__ as PM

EX = '/opt/data/workspace/OpenPSN/examples/c5g7/study2_rectangular'
spec = mdl.load_spec(EX + '/c5g7_rect_uo2_assembly.yaml')
case = {'name': 'M2', 'model': 'ty3', 'I': 30, 'M': 2, 'subdivide': 1,
        'A': 3, 'closure': 'directional_matrix'}
spec['solver'].update({'keff_tol': 1e-10, 'max_outer': 200, 'threads': 1})
psn = PM.build_solver(spec, case, threads=1, mem_limit_gb=32)
if not hasattr(psn, '_groups'):
    psn._vectorize_setup()

nl = hs.HoNodeLocal(psn, 2, 0, 3, 'directional', gs_tol=1e-11, max_gs=8000)
q0 = np.random.default_rng(42).normal(size=(psn.nodes, 5))
nl.u = np.zeros(psn.ncol * nl.K)
orders = nl._mc_order()
seen = []
for pty in (0, 1):
    for k in orders[pty][0] + orders[pty][1]:
        if k not in seen:
            seen.append(k)
dc = {k: nl._microclass(k) for k in seen}
sizes = {k: dc[k]['wb'].shape[0] for k in seen}
print(f"micro-classes: {len(seen)}, nodes total: {psn.nodes}, ncol: {psn.ncol}")
print("system dims per class:", sorted({dc[k]['A'].shape[0] for k in seen}))

# ---- cost breakdown of one full round ----
t_b = t_s = 0.0
ncalls = 0
for direction in (0, 1):
    for pty in (0, 1):
        for key in orders[pty][direction]:
            d = dc[key]
            t0 = time.time(); b = nl._mc_b(d, q0); t_b += time.time() - t0
            t0 = time.time(); x = lu_solve(d['fac'], b.T).T; t_s += time.time() - t0
            nl.u[d['wb']] = x
            ncalls += 1
print(f"one full round (2 passes): b-assembly {t_b*1e3:.1f} ms | "
      f"lu_solve {t_s*1e3:.1f} ms | total {(t_b+t_s)*1e3:.1f} ms")
print(f"lu_solve calls/round: {ncalls} (batched per class), "
      f"per-call ~{t_s/ncalls*1e6:.1f} us")

# ---- GIL / oversubscription test on one large class ----
key_big = max(seen, key=lambda k: sizes[k])
fac = dc[key_big]['fac']
d = dc[key_big]['A'].shape[0]
n_rhs = sizes[key_big]
Bs = np.random.default_rng(0).normal(size=(d, n_rhs))

def batch(n):
    for _ in range(n):
        lu_solve(fac, Bs)

t0 = time.time(); batch(400); t_serial = time.time() - t0
print(f"\n[main thread, BLAS auto] 400 solves: {t_serial*1e3:.0f} ms")

import concurrent.futures as cf
NT = 24
N = 1200

def worker(lo, hi):
    for _ in range(lo, hi):
        lu_solve(fac, Bs)

# 24 py threads x BLAS 24 threads (oversubscribed)
t0 = time.time()
with cf.ThreadPoolExecutor(max_workers=NT) as ex:
    ex.map(worker, range(0, NT), [0, N // NT] * NT)
t_over = time.time() - t0
print(f"[24 thr x BLAS auto] {N} solves: {t_over*1e3:.0f} ms  "
      f"(speedup {t_serial*400/N/t_over:.2f}x)")

# 24 py threads x BLAS 1 thread (clean)
t0 = time.time()
with cf.ThreadPoolExecutor(max_workers=NT) as ex:
    futs = []
    def w2():
        with threadpool_limits(1, worker_thread=True):
            for _ in range(N // NT):
                lu_solve(fac, Bs)
        return None
    for _ in range(NT):
        futs.append(ex.submit(w2))
    for f in futs:
        f.result()
t_clean = time.time() - t0
print(f"[24 thr x BLAS=1]    {N} solves: {t_clean*1e3:.0f} ms  "
      f"(speedup {t_serial*400/N/t_clean:.2f}x)")
