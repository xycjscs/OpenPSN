# -*- coding: utf-8 -*-
"""dbg_e2e_A.py — M2 end-to-end for A=1 and A=2: GMRES pool vs direct LU.

Same protocol as dbg_e2e3.py (which covered A=3, PASS).  Runs both A
values in one subprocess: for each A, build two independent psn
objects, run each driver to the production convergence criterion,
compare keff + flux field.  Verdict per A.
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
from psn2d.highorder import keff_highorder
from psn2d.highsweep import keff_highorder_gmres

EX = '/opt/data/workspace/OpenPSN/examples/c5g7/study2_rectangular'
AS = [int(a) for a in sys.argv[1:]] or [1, 2]


def build(A):
    spec = mdl.load_spec(EX + '/c5g7_rect_uo2_assembly.yaml')
    case = {'name': f'M2_A{A}', 'model': 'ty3', 'I': 30, 'M': 2,
            'subdivide': 1, 'A': A, 'closure': 'directional_matrix'}
    spec['solver'].update({'keff_tol': 1e-10, 'max_outer': 400,
                           'threads': 1})
    return PM.build_solver(spec, case, threads=1, mem_limit_gb=32)


all_ok = True
for A in AS:
    print(f"===== A = {A} =====", flush=True)
    psn_lu = build(A)
    t0 = time.time()
    k_lu, phi_lu, qim_lu, info_lu = keff_highorder(psn_lu, A=A,
                                                   branch='directional')
    dt_lu = time.time() - t0
    print(f"LU   : keff={k_lu:.9f}  outers={info_lu['outer_iterations']}  "
          f"dlam={info_lu['keff_residual']:.2e}  src={info_lu['source_residual']:.2e}"
          f"  converged={info_lu['converged']}  {dt_lu:.1f}s", flush=True)

    psn_gm = build(A)
    t0 = time.time()
    k_gm, phi_gm, qim_gm, info_gm = keff_highorder_gmres(
        psn_gm, A=A, branch='directional', rtol=1e-11)
    dt_gm = time.time() - t0
    print(f"GMRES: keff={k_gm:.9f}  outers={info_gm['outer_iterations']}  "
          f"dlam={info_gm['keff_residual']:.2e}  src={info_gm['source_residual']:.2e}"
          f"  converged={info_gm['converged']}  {dt_gm:.1f}s  "
          f"workers={info_gm['nworkers']}/{info_gm['nsystems']}  "
          f"matvecs={info_gm['gmres_matvecs_total']}  "
          f"maxface_res={info_gm['max_face_system_residual']:.2e}",
          flush=True)

    dkeff = abs(k_gm - k_lu)
    phi_diff = float(np.abs(phi_gm - phi_lu).max())
    phi_rel = float(phi_diff / max(1.0, np.abs(phi_lu).max()))
    print(f"DKEFF      = {dkeff:.3e}", flush=True)
    print(f"PHI max|d| = {phi_diff:.3e}  (rel {phi_rel:.3e})", flush=True)
    ok = (dkeff < 1e-7 and info_lu['converged'] and info_gm['converged'])
    print(f"VERDICT A={A}:", "PASS" if ok else "FAIL", flush=True)
    all_ok = all_ok and ok
    # release before next A
    del psn_lu, psn_gm, phi_lu, phi_gm, qim_lu, qim_gm

print("\nOVERALL:", "PASS" if all_ok else "FAIL", flush=True)
