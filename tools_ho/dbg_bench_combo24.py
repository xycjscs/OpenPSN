# -*- coding: utf-8 -*-
"""Combined lever benchmark — 24-THREAD budget (user rule 2026-09-14:
the 48-core host is used with at most 24 threads).

M8 (S=1) configurations, full convergence each:
  A: system, numba off, 21w x T-irrelevant   (INHERITED 1063s — config is
     bit-identical to the 48-thread run: 21 workers, numba disabled,
     thread count unused; re-running wastes 18 min)
  B: pair,   numba off,  24w x 1T            (baseline, 24-thread)
  C: system, numba on,   21w x 1T
  D: pair,   numba on,   24w x 1T
  E: pair,   numba on,   12w x 2T
M2 two configurations:
  F: system, numba off,  21w                 (baseline)
  G: system, numba on,   21w x 1T
Speedups reported against A (M8) and F (M2).
Run: python dbg_bench_combo24.py
"""
import os, sys, time, resource
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ['PSN_CORE_BUDGET'] = '24'
sys.path.insert(0, '/opt/data/workspace/OpenPSN')
import numpy as np
from psn2d import model
from psn2d import __main__ as M
from psn2d import highsweep
from psn2d import mckern

EX = '/opt/data/workspace/OpenPSN/examples/c5g7/study2_rectangular'
REF = {'M2': 1.3342495, 'M8': 1.3352414}
A_INHERITED = 1063.0  # M8 system no-numba, 21w — 2026-09-14 combo bench A


def build(MM):
    spec = model.load_spec(EX + '/c5g7_rect_uo2_assembly.yaml')
    case = {'name': f'M{MM}_S1', 'model': 'ty3', 'I': 30, 'M': MM,
            'subdivide': 1, 'kref': 1.3347260, 'A': 3,
            'closure': 'directional_matrix'}
    spec['solver'].update({'keff_tol': 1e-10, 'max_outer': 6000,
                           'threads': 1})
    return M.build_solver(spec, case, threads=1, mem_limit_gb=120)


def run(tag, MM, pairs, numba, nworkers=None):
    mckern.ENABLED = bool(numba)
    psn = build(MM)
    t0 = time.time()
    k, phi, qim, h = highsweep.keff_highorder_gmres(
        psn, A=3, branch='directional', max_outer=6000, outer_tol=1e-10,
        rtol=1e-11, maxiter=800, pairs=pairs, nworkers=nworkers)
    wall = time.time() - t0
    dref = abs(k - REF[f'M{MM}'])
    print(f'{tag}: keff={k:.7f} dref={dref:.1e} wall={wall:.0f}s '
          f'workers={h["nworkers"]} blocks={h["nblocks"]} T={highsweep._NUM_THREADS} '
          f'iters={h["outer_iterations"]} '
          f'matvecs={h["gmres_matvecs_total"]} '
          f'peak={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f}GB',
          flush=True)
    del psn, k, phi, qim, h
    return wall


t_start = time.time()
print('=== COMBO BENCH 24-THREAD start ===', flush=True)
w = {}
w['A_sys_noN'] = A_INHERITED
w['B_pair_noN'] = run('M8 B pair-noN(24w)', 8, True, False)
w['C_sys_N'] = run('M8 C sys-N(21wT1)', 8, False, True)
w['D_pair_N_24'] = run('M8 D pair-N(24wT1)', 8, True, True, nworkers=24)
w['E_pair_N_12'] = run('M8 E pair-N(12wT2)', 8, True, True, nworkers=12)
w['F_m2sys_noN'] = run('M2 F sys-noN(21w)', 2, False, False)
w['G_m2sys_N'] = run('M2 G sys-N(21wT1)', 2, False, True)
print('\n=== SUMMARY (wall seconds, 24-thread budget) ===', flush=True)
for k_, v in w.items():
    print(f'  {k_:14s} {v:7.0f}s', flush=True)
print('\n=== SPEEDUPS ===', flush=True)
print(f'  lever1 M8  (A->B)   : {w["A_sys_noN"]/w["B_pair_noN"]:.2f}x', flush=True)
print(f'  lever3 M8  (A->C)   : {w["A_sys_noN"]/w["C_sys_N"]:.2f}x', flush=True)
print(f'  lever1+3 M8 (A->D)  : {w["A_sys_noN"]/w["D_pair_N_24"]:.2f}x', flush=True)
print(f'  lever1+3 M8 (A->E)  : {w["A_sys_noN"]/w["E_pair_N_12"]:.2f}x', flush=True)
print(f'  best   M8 (A->min)  : {w["A_sys_noN"]/min(w.values()):.2f}x', flush=True)
print(f'  lever3 M2  (F->G)   : {w["F_m2sys_noN"]/w["G_m2sys_N"]:.2f}x', flush=True)
print(f'total {time.time()-t_start:.0f}s', flush=True)
print('COMBO BENCH 24 DONE', flush=True)
