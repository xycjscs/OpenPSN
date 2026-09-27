# -*- coding: utf-8 -*-
"""Verify the OpenMP-fork fix: numba ENABLED + pair pool, 1 outer
iteration, M4.  Before the fix the first numba call in the parent
initialised OpenMP there and libgomp aborted every forked worker
("fork() called from a process already using GNU OpenMP")."""
import os, sys, time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, '/opt/data/workspace/OpenPSN')

from psn2d import model
from psn2d import __main__ as M
from psn2d import highsweep
from psn2d import mckern

EX = '/opt/data/workspace/OpenPSN/examples/c5g7/study2_rectangular'
print('numba ENABLED =', mckern.ENABLED, '(must be True)', flush=True)
assert mckern.ENABLED, 'OPSN_NO_NUMBA is set? fix env'


def build(MM):
    spec = model.load_spec(EX + '/c5g7_rect_uo2_assembly.yaml')
    case = {'name': f'M{MM}_S1', 'model': 'ty3', 'I': 30, 'M': MM,
            'subdivide': 1, 'kref': 1.3347260, 'A': 3,
            'closure': 'directional_matrix'}
    spec['solver'].update({'keff_tol': 1e-10, 'max_outer': 6000,
                           'threads': 1})
    return M.build_solver(spec, case, threads=1, mem_limit_gb=120)


psn = build(4)
t0 = time.time()
k, phi, qim, info = highsweep.keff_highorder_gmres(
    psn, A=3, branch='directional', max_outer=1, outer_tol=1e-10,
    rtol=1e-11, maxiter=200, pairs='auto')
dt = time.time() - t0
print(f'M4 pair+numba 1-outer: keff={k:.7f} dt={dt:.1f}s '
      f'workers={info["nworkers"]} blocks={info["nblocks"]} '
      f'mode={info["inner"]}', flush=True)
print('FORK-FIX PASS' if dt < 600 else 'SUSPICIOUS: too slow (dead workers?)')
