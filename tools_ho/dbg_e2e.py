# -*- coding: utf-8 -*-
"""dbg_e2e.py — end-to-end: keff_highorder_gs (plain + CMFD) vs
keff_highorder (direct LU) on the M2 uo2 assembly.  Same acceptance
protocol (plain power iteration, dlam<1e-10 & src_res<1e-9).
Measures keff agreement, wall time, peak RSS."""
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, '/opt/data/workspace/OpenPSN')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '24')

from psn2d import highorder as ho
from psn2d import highsweep as hsw
from psn2d import model as mdl
from psn2d import __main__ as PM

EX = '/opt/data/workspace/OpenPSN/examples/c5g7/study2_rectangular'


def build_psn(key='uo2', MM=2):
    spec = mdl.load_spec(EX + f'/c5g7_rect_{key}_assembly.yaml')
    case = {'name': f'M{MM}', 'model': 'ty3', 'I': 30, 'M': MM,
            'subdivide': 1, 'A': 3, 'closure': 'directional_matrix'}
    spec['solver'].update({'keff_tol': 1e-10, 'max_outer': 60, 'threads': 24})
    psn = PM.build_solver(spec, case, threads=24, mem_limit_gb=32)
    if not hasattr(psn, '_groups'):
        psn._vectorize_setup()
    return psn


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


# 1) direct LU reference
print("== LU reference (keff_highorder) ==", flush=True)
psn = build_psn()
t0 = time.time()
lam_lu, phi_lu, qim_lu, info_lu = ho.keff_highorder(psn, A=3,
                                                    branch='directional',
                                                    max_outer=60,
                                                    outer_tol=1e-10,
                                                    verbose=True)
t_lu = time.time() - t0
print(f"LU: keff={lam_lu:.7f} outer={info_lu['outer_iterations']} "
      f"t={t_lu:.0f}s peak={peak_gb():.1f}GB")

# 2) GS plain (SOR omega=1.02)
print("\n== GS plain (omega=1.02) ==", flush=True)
psn = build_psn()
t0 = time.time()
lam_gs, phi_gs, qim_gs, info_gs = hsw.keff_highorder_gs(
    psn, A=3, branch='directional', max_outer=60, outer_tol=1e-10,
    gs_tol=1e-12, max_gs=8000, omega=1.02, cmfd=False, verbose=True)
t_gs = time.time() - t0
print(f"GS: keff={lam_gs:.7f} outer={info_gs['outer_iterations']} "
      f"gs_total={info_gs['gs_inner_iterations_total']} "
      f"t={t_gs:.0f}s peak={peak_gb():.1f}GB")
print(f"keff diff (pcm) = {(lam_gs-lam_lu)*1e5:.4f}")
print(f"phi max rel diff = {np.abs(phi_gs-phi_lu).max()/phi_lu.max():.3e}")

# 3) GS + CMFD
print("\n== GS + CMFD ==", flush=True)
psn = build_psn()
t0 = time.time()
lam_c, phi_c, qim_c, info_c = hsw.keff_highorder_gs(
    psn, A=3, branch='directional', max_outer=60, outer_tol=1e-10,
    gs_tol=1e-12, max_gs=8000, omega=1.02, cmfd=True, cmfd_omega=0.5,
    cmfd_collapse=3, cmfd_switch=1e-8, cmfd_coarse_sweeps=6, verbose=True)
t_c = time.time() - t0
print(f"CMFD: keff={lam_c:.7f} outer={info_c['outer_iterations']} "
      f"gs_total={info_c['gs_inner_iterations_total']} "
      f"cmfd_iters={info_c['cmfd_iterations']} "
      f"cmfd_active_end={info_c['cmfd_active_at_end']} "
      f"t={t_c:.0f}s peak={peak_gb():.1f}GB")
print(f"keff diff vs LU (pcm) = {(lam_c-lam_lu)*1e5:.4f}")
print(f"phi max rel diff = {np.abs(phi_c-phi_lu).max()/phi_lu.max():.3e}")
