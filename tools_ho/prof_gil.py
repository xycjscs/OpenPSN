# -*- coding: utf-8 -*-
"""prof_gil.py — decisive GIL + cross-(i,g) parallelism probe.
Run with OPENBLAS_NUM_THREADS=1 so the ONLY parallelism is our own
Python threads (removes OpenBLAS oversubscription/crash + isolates
the NumPy GIL behaviour, which is the real question).

  Part 1: _mc_b (the ~87% of a matvec, NumPy glue) — serial vs N thr
  Part 2: one_pass (full matvec: _mc_b + lu_solve) — serial vs 24 thr
  Part 3: the dominant cost's GIL fraction -> predicted parallel cap
"""
import sys, time
sys.path.insert(0, '/opt/data/workspace/OpenPSN')
import numpy as np
import concurrent.futures as cf

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
ng, I = psn.ng, psn.I
print(f"built: I={I} ng={ng} -> {I*ng} (i,g) systems, nodes={psn.nodes}")

rng = np.random.default_rng(7)
q0_all = {(i, g): rng.normal(size=(psn.nodes, 5))
          for i in range(I) for g in range(ng)}

def make_sys(i, g):
    nl = hs.HoNodeLocal(psn, i, g, 3, 'directional', gs_tol=1e-11, max_gs=1)
    nl.u = np.zeros(psn.ncol * nl.K)
    orders = nl._mc_order()
    seen = []
    for pty in (0, 1):
        for k in orders[pty][0] + orders[pty][1]:
            if k not in seen:
                seen.append(k)
    dc = {k: nl._microclass(k) for k in seen}
    return nl, orders, dc, q0_all[(i, g)]

def b_round(sys, n=5):
    nl, orders, dc, qn = sys
    t = 0.0
    for _ in range(n):
        t0 = time.time()
        for direction in (0, 1):
            for pty in (0, 1):
                for key in orders[pty][direction]:
                    nl._mc_b(dc[key], qn)
        t += time.time() - t0
    return t / n

def one_pass(sys):
    nl, orders, dc, qn = sys
    for direction in (0, 1):
        for pty in (0, 1):
            for key in orders[pty][direction]:
                nl._mc_pass(dc[key], qn)

systems = [make_sys(i, g) for i in range(I) for g in range(ng)]
N = len(systems)
print(f"{N} (i,g) systems ready")

# ---------- Part 1: _mc_b GIL probe (8 threads) ----------
print("\n--- Part 1: _mc_b (b-assembly) scaling ---")
t_ser_b = sum(b_round(s) for s in systems) / N
print(f"  per-system b-assembly round: {t_ser_b*1e3:.1f} ms")
def run_b(pool):
    return b_round(pool)
for NT in (2, 4, 8):
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=NT) as ex:
        list(ex.map(run_b, systems[:NT]))
    t_par = time.time() - t0
    # each thread did 1 round; serial cost for NT rounds = NT*t_ser_b
    print(f"  {NT:2d} thr: {t_par*1e3:7.1f} ms for {NT} rounds  "
          f"(speedup {NT*t_ser_b/t_par:.2f}x, eff {NT*t_ser_b/t_par/NT*100:.0f}%)")

# ---------- Part 2: full one_pass scaling (24 threads) ----------
print(f"\n--- Part 2: one_pass (full matvec) scaling, {N} systems ---")
t0 = time.time()
for s in systems:
    one_pass(s)
t_ser = time.time() - t0
print(f"  serial {N} one_pass: {t_ser:.2f}s  ({t_ser/N*1e3:.1f} ms each)")

NT = 24
t0 = time.time()
with cf.ThreadPoolExecutor(max_workers=NT) as ex:
    list(ex.map(one_pass, systems))
t_par = time.time() - t0
sp = t_ser / t_par
print(f"  {NT} thr: {t_par:.2f}s  speedup {sp:.2f}x  efficiency {sp/NT*100:.0f}%")

# ---------- Part 3: predicted outer-iter time ----------
print(f"\n--- Part 3: outer-iteration cost model ---")
for nv in (150, 120):
    print(f"  ~{nv} matvec/outer: serial {t_ser*nv:.0f}s | "
          f"24-thread {t_par*nv:.0f}s")
