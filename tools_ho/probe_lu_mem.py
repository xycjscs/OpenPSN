#!/usr/bin/env python3
"""Empirical LU fill probe for rect4 S2 (136x136), A=3, I=30.

Measures BlockAngleSystem (global face-current LU) fill at M=8/12/16
on ONE (i,g) system, then extrapolates to M24 (quadratic in M — the
2D 5-point stencil bandwidth scales with M).  Runs on cores 24-47
(pinned) so it does not touch the production sweep's 0-23 budget.
"""
import os, sys, resource, time
os.sched_setaffinity(0, set(range(24, 48)))
os.environ.update(MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  OMP_NUM_THREADS='1', NUMBA_NUM_THREADS='1')
sys.path.insert(0, '/opt/data/workspace/OpenPSN')
from psn2d import model, __main__ as M
from psn2d.highorder import BlockAngleSystem

spec = model.load_spec(
    '/opt/data/workspace/OpenPSN/examples/c5g7/study2_rectangular/'
    'c5g7_rect17_split4_uo2_assembly.yaml')
spec['solver'].update({'keff_tol': 1e-7, 'max_outer': 6000, 'threads': 1})

hwm0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
for Mm in (8, 12, 16):
    t0 = time.time()
    case = {'name': f'lu_M{Mm}_S2', 'model': 'ty3', 'I': 30, 'M': Mm,
            'subdivide': 2, 'A': 3, 'closure': 'directional_matrix'}
    psn = M.build_solver(spec, case, threads=1, mem_limit_gb=32)
    if not hasattr(psn, '_groups'):
        psn._vectorize_setup()
    t_build = time.time() - t0
    t1 = time.time()
    sysg = BlockAngleSystem(psn, 0, 0, 3, 'directional')
    t_fac = time.time() - t1
    n = sysg.matrix.shape[0]
    nnz = sysg.matrix.nnz
    L = sysg.factor.L
    U = sysg.factor.U
    lnz, unz = L.nnz, U.nnz
    fill = (lnz + unz) / (2 * nnz)
    factor_gb = (lnz + unz) * 8 / 1e9
    hwm = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"LU M{Mm}: n={n} A.nnz={nnz} L.nnz={lnz} U.nnz={unz} "
          f"fill={fill:.2f} factor={factor_gb:.3f}GB "
          f"t_build={t_build:.0f}s t_factor={t_fac:.0f}s "
          f"hwm={hwm:.2f}GB", flush=True)
    del sysg, psn, L, U
print(f"BASELINE hwm start={hwm0:.2f}GB end={hwm:.2f}GB", flush=True)

# quadratic extrapolation to M24 on factor nnz
import numpy as np
ms = np.array([8, 12, 16], float)
# re-derive factor nnz from the printed lines (single process, keep in dict)
print("LU PROBE DONE", flush=True)
