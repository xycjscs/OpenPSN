# -*- coding: utf-8 -*-
"""Lever-1 A/B benchmark: same point, same epoch, system vs pair mode.

For a given M, run full-convergence keff in BOTH modes back-to-back in
this process (machine load identical) and report wall + keff + worker
count for each.  Speedup = wall_system / wall_pair.

Reference keff (legacy LU, dir_table3_sweep.log):
  M4_S1 = 1.3349468
  M8_S1 = 1.3352414
Run: python dbg_bench_pairs.py M4 | M8
"""
import os, sys, time, resource
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, '/opt/data/workspace/OpenPSN')
import numpy as np
from psn2d import model
from psn2d import __main__ as M
from psn2d import highsweep

EX = '/opt/data/workspace/OpenPSN/examples/c5g7/study2_rectangular'
REF = {'M4': 1.3349468, 'M8': 1.3352414}
STAGE = sys.argv[1] if len(sys.argv) > 1 else 'M8'
MM = int(STAGE[1:])
REFK = REF[STAGE]


def build():
    spec = model.load_spec(EX + '/c5g7_rect_uo2_assembly.yaml')
    case = {'name': f'M{MM}_S1', 'model': 'ty3', 'I': 30, 'M': MM,
            'subdivide': 1, 'kref': 1.3347260, 'A': 3,
            'closure': 'directional_matrix'}
    spec['solver'].update({'keff_tol': 1e-10, 'max_outer': 6000,
                           'threads': 1})
    return M.build_solver(spec, case, threads=1, mem_limit_gb=120)


def run(pairs):
    psn = build()
    t0 = time.time()
    k, phi, qim, h = highsweep.keff_highorder_gmres(
        psn, A=3, branch='directional', max_outer=6000, outer_tol=1e-10,
        rtol=1e-11, maxiter=800, pairs=pairs)
    wall = time.time() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    return k, wall, h, peak


print(f'=== {STAGE} A/B benchmark (ref LU keff={REFK}) ===', flush=True)
print('running SYSTEM mode (pairs=False) ...', flush=True)
k_s, w_s, h_s, pk_s = run(False)
print(f'  SYSTEM keff={k_s:.7f} dkeff={abs(k_s-REFK):.2e} '
      f'wall={w_s:.0f}s blocks={h_s["nblocks"]} workers={h_s["nworkers"]} '
      f'iters={h_s["outer_iterations"]} matvecs={h_s["gmres_matvecs_total"]} '
      f'peak={pk_s:.2f}GB', flush=True)
print('running PAIR mode (pairs=True) ...', flush=True)
k_p, w_p, h_p, pk_p = run(True)
print(f'  PAIR   keff={k_p:.7f} dkeff={abs(k_p-REFK):.2e} '
      f'wall={w_p:.0f}s blocks={h_p["nblocks"]} workers={h_p["nworkers"]} '
      f'iters={h_p["outer_iterations"]} matvecs={h_p["gmres_matvecs_total"]} '
      f'peak={pk_p:.2f}GB', flush=True)
sp = w_s / w_p
print(f'\n  SPEEDUP (system/pair) = {w_s:.0f}s / {w_p:.0f}s = {sp:.2f}x',
      flush=True)
print(f'  keff agreement |k_s - k_p| = {abs(k_s-k_p):.2e}', flush=True)
# Gates: same-run pair-vs-system agreement (Krylov spaces differ at the
# inner rtol level) and both against the legacy-LU reference.  The
# outer iteration pins keff to dlam<1e-10, so ~1e-9 is the natural
# agreement floor (measured: system mode dkeff=8.05e-10 vs LU ref).
ok = (abs(k_s - k_p) < 1e-9 and abs(k_s - REFK) < 1e-8
      and abs(k_p - REFK) < 1e-8)
print('E2E', 'PASS' if ok else 'FAIL', flush=True)
sys.exit(0 if ok else 1)
