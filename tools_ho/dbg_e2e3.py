# -*- coding: utf-8 -*-
"""dbg_e2e3.py — M2 end-to-end: full-convergence LU path vs GMRES pool.

Two independently built psn objects (pool state isolation); each driver
runs to its production convergence criterion (dlam < 1e-10 AND
src_res < 1e-9).  Compares keff, outer counts and the flux field.
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


def build():
    spec = mdl.load_spec(EX + '/c5g7_rect_uo2_assembly.yaml')
    case = {'name': 'M2', 'model': 'ty3', 'I': 30, 'M': 2,
            'subdivide': 1, 'A': 3, 'closure': 'directional_matrix'}
    spec['solver'].update({'keff_tol': 1e-10, 'max_outer': 400,
                           'threads': 1})
    return PM.build_solver(spec, case, threads=1, mem_limit_gb=32)


psn_lu = build()
from psn2d.highorder import keff_highorder
t0 = time.time()
k_lu, phi_lu, qim_lu, info_lu = keff_highorder(psn_lu, A=3,
                                               branch='directional')
dt_lu = time.time() - t0
print(f"LU   : keff={k_lu:.9f}  outers={info_lu['outer_iterations']}  "
      f"dlam={info_lu['keff_residual']:.2e}  src={info_lu['source_residual']:.2e}"
      f"  converged={info_lu['converged']}  {dt_lu:.1f}s", flush=True)

psn_gm = build()
from psn2d.highsweep import keff_highorder_gmres
t0 = time.time()
k_gm, phi_gm, qim_gm, info_gm = keff_highorder_gmres(
    psn_gm, A=3, branch='directional', rtol=1e-11)
dt_gm = time.time() - t0
print(f"GMRES: keff={k_gm:.9f}  outers={info_gm['outer_iterations']}  "
      f"dlam={info_gm['keff_residual']:.2e}  src={info_gm['source_residual']:.2e}"
      f"  converged={info_gm['converged']}  {dt_gm:.1f}s  "
      f"workers={info_gm['nworkers']}/{info_gm['nsystems']}  "
      f"matvecs={info_gm['gmres_matvecs_total']}  "
      f"maxface_res={info_gm['max_face_system_residual']:.2e}", flush=True)

dkeff = abs(k_gm - k_lu)
dnorm = phi_gm.sum()
phi_diff = float(np.abs(phi_gm - phi_lu).max())
phi_rel = float(np.abs(phi_gm - phi_lu).max() /
                max(1.0, np.abs(phi_lu).max()))
print(f"\nDKEFF      = {dkeff:.3e}", flush=True)
print(f"PHI max|d| = {phi_diff:.3e}  (rel {phi_rel:.3e})", flush=True)
ok = dkeff < 1e-7 and info_lu['converged'] and info_gm['converged']
print("VERDICT:", "PASS" if ok else "FAIL", flush=True)
