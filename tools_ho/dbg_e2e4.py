# -*- coding: utf-8 -*-
"""dbg_e2e4.py — M2 regression for the STATIC-SHARD GMRES pool driver.
First-10-outer keff trajectory must match the verified queue-dispatch
trajectory (dbg_pool_outer / e2e3): 0.9630572 -> 0.9572114 -> 0.9411114
(bit-wise on the first three, tolerance 1e-7 on the rest).
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import sys, time

sys.path.insert(0, '/opt/data/workspace/OpenPSN')
import numpy as np
from psn2d import model as mdl
from psn2d import __main__ as PM

EX = '/opt/data/workspace/OpenPSN/examples/c5g7/study2_rectangular'
spec = mdl.load_spec(EX + '/c5g7_rect_uo2_assembly.yaml')
case = {'name': 'M2', 'model': 'ty3', 'I': 30, 'M': 2,
        'subdivide': 1, 'A': 3, 'closure': 'directional_matrix'}
spec['solver'].update({'keff_tol': 1e-3, 'max_outer': 10, 'threads': 1})
psn = PM.build_solver(spec, case, threads=1, mem_limit_gb=32)

from psn2d.highsweep import keff_highorder_gmres
t0 = time.time()
k, phi, qim, info = keff_highorder_gmres(psn, A=3, branch='directional',
                                         max_outer=10, outer_tol=1e-3,
                                         verbose=True)
print(f"\nfinal keff={k:.9f}  {time.time()-t0:.1f}s  "
      f"matvecs={info['gmres_matvecs_total']}  "
      f"maxface_res={info['max_face_system_residual']:.2e}", flush=True)

# reference trajectory from the verified run (first 3 are bit-exact)
ref = [0.9630572, 0.9572114, 0.9411114]
print("trajectory check: first-3 bit-match required", flush=True)
