# -*- coding: utf-8 -*-
"""Reproduce the M=2 'zero-size array to reduction maximum' crash with a
full traceback (2026-09-14: rect4x4 pin-cell sweep, M2 S1/S2 both die in
~0s).  Runs on the ACTUAL sweep geometry (17x17 split4 assembly) so the
fix is validated where it matters.

Usage: dbg_m2.py [S]   (default S=1)
"""
import os, sys, time, traceback

sys.path.insert(0, '/opt/data/workspace/OpenPSN')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('PSN_CORE_BUDGET', '24')

S = int(sys.argv[1]) if len(sys.argv) > 1 else 1
EX = '/opt/data/workspace/OpenPSN/examples/c5g7/study2_rectangular'
os.sched_setaffinity(0, set(range(24)))
import resource as _r
_r.setrlimit(_r.RLIMIT_AS, (int(64 * 2 ** 30), -1))

t0 = time.time()
from psn2d import model, __main__ as M, highsweep, mckern
mckern.ENABLED = True
spec = model.load_spec(EX + '/c5g7_rect17_split4_uo2_assembly.yaml')
case = {'name': f'M2_S{S}', 'model': 'ty3', 'I': 30, 'M': 2,
        'subdivide': S, 'A': 3, 'closure': 'directional_matrix'}
spec['solver'].update({'keff_tol': 1e-7, 'max_outer': 6000, 'threads': 1})
psn = M.build_solver(spec, case, threads=1, mem_limit_gb=64)
print(f"built: nodes={psn.nodes} t={time.time()-t0:.0f}s", flush=True)
try:
    k, phi, qim, h = highsweep.keff_highorder_gmres(
        psn, A=3, branch='directional', max_outer=6000,
        outer_tol=1e-7, rtol=1e-7, maxiter=800,
        pairs=True, nworkers=24, verbose=False)
    print(f"OK keff={k:.7f} iters={h['outer_iterations']} "
          f"converged={h['converged']} t={time.time()-t0:.0f}s", flush=True)
except Exception:
    traceback.print_exc()
    sys.exit(1)
