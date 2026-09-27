# -*- coding: utf-8 -*-
"""Lever-3 A/B benchmark: M2 system mode, numba on vs off.

M2 => M/2 = 1 => system mode (no pair split), so the ONLY difference
between the two runs is the b-assembly kernel: NumPy (GIL) vs numba
prange (GIL-released).  Run both in one process (same load).
Speedup = wall_numpy / wall_numba.

Reference keff (legacy LU M2_S1): 1.3342495
Run: python dbg_bench_numba.py
"""
import os, sys, time, resource
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, '/opt/data/workspace/OpenPSN')
import numpy as np
from psn2d import model
from psn2d import __main__ as M
from psn2d import highsweep
from psn2d import mckern

EX = '/opt/data/workspace/OpenPSN/examples/c5g7/study2_rectangular'
REFK = 1.3342495
MM = 2


def build():
    spec = model.load_spec(EX + '/c5g7_rect_uo2_assembly.yaml')
    case = {'name': f'M{MM}_S1', 'model': 'ty3', 'I': 30, 'M': MM,
            'subdivide': 1, 'kref': 1.3347260, 'A': 3,
            'closure': 'directional_matrix'}
    spec['solver'].update({'keff_tol': 1e-10, 'max_outer': 6000,
                           'threads': 1})
    return M.build_solver(spec, case, threads=1, mem_limit_gb=120)


def run(enable_numba):
    mckern.ENABLED = bool(enable_numba)   # runtime toggle
    psn = build()
    t0 = time.time()
    k, phi, qim, h = highsweep.keff_highorder_gmres(
        psn, A=3, branch='directional', max_outer=6000, outer_tol=1e-10,
        rtol=1e-11, maxiter=800, pairs=False)
    wall = time.time() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    return k, wall, h, peak


print(f'=== M2 numba A/B (ref LU keff={REFK}) ===', flush=True)
print('running NUMPY (numba off) ...', flush=True)
k_n, w_n, h_n, _ = run(False)
print(f'  NUMPY  keff={k_n:.7f} dkeff={abs(k_n-REFK):.2e} '
      f'wall={w_n:.0f}s workers={h_n["nworkers"]} '
      f'iters={h_n["outer_iterations"]} '
      f'matvecs={h_n["gmres_matvecs_total"]}', flush=True)
print('running NUMBA (numba on) ...', flush=True)
k_b, w_b, h_b, _ = run(True)
print(f'  NUMBA  keff={k_b:.7f} dkeff={abs(k_b-REFK):.2e} '
      f'wall={w_b:.0f}s workers={h_b["nworkers"]} '
      f'iters={h_b["outer_iterations"]} '
      f'matvecs={h_b["gmres_matvecs_total"]}', flush=True)
sp = w_n / w_b
print(f'\n  SPEEDUP (numpy/numba) = {w_n:.0f}s / {w_b:.0f}s = {sp:.2f}x',
      flush=True)
print(f'  keff agreement |k_n - k_b| = {abs(k_n-k_b):.2e}', flush=True)
ok = (abs(k_n - k_b) < 1e-9 and abs(k_n - REFK) < 1e-8
      and abs(k_b - REFK) < 1e-8)
print('E2E', 'PASS' if ok else 'FAIL', flush=True)
sys.exit(0 if ok else 1)
