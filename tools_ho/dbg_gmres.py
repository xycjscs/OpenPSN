# -*- coding: utf-8 -*-
"""dbg_gmres.py — GMRES(red-black matvec) vs plain GS on the slow i2
(i,g) system; cold + warm start; agreement with the SuperLU reference.
Also the fast i0/g0 system for contrast."""
import sys, time

import numpy as np

sys.path.insert(0, '/opt/data/workspace/OpenPSN')
from psn2d import highorder as ho
from psn2d import highsweep as hs
from psn2d import model as mdl
from psn2d import __main__ as PM

EX = '/opt/data/workspace/OpenPSN/examples/c5g7/study2_rectangular'


def build_psn(key='uo2', MM=2):
    spec = mdl.load_spec(EX + f'/c5g7_rect_{key}_assembly.yaml')
    case = {'name': f'M{MM}', 'model': 'ty3', 'I': 30, 'M': MM,
            'subdivide': 1, 'A': 3, 'closure': 'directional_matrix'}
    spec['solver'].update({'keff_tol': 1e-10, 'max_outer': 200,
                           'threads': 1})
    psn = PM.build_solver(spec, case, threads=1, mem_limit_gb=32)
    if not hasattr(psn, '_groups'):
        psn._vectorize_setup()
    return psn


psn = build_psn()
B_ref = None
q0 = np.random.default_rng(42).normal(size=(psn.nodes, 5))

for (ii, gg, slow) in ((psn.I - 1, 0, True), (0, 0, False)):
    print(f"== system (i={ii}, g={gg}) ==", flush=True)
    B_ref = ho.BlockAngleSystem(psn, ii, gg, 3, 'directional')
    b = B_ref.source_map @ q0.ravel()
    u_ref = B_ref.factor.solve(b)

    # ---- GMRES cold start ----
    nl = hs.HoNodeLocal(psn, ii, gg, 3, 'directional', gs_tol=1e-11,
                        max_gs=8000)
    t0 = time.time()
    u, nv, res, out = nl.sweep_gmres(q0, rtol=1e-11, maxiter=600,
                                     verbose=True)
    dt = time.time() - t0
    du = float(np.abs(u.ravel() - u_ref).max() / np.abs(u_ref).max())
    print(f"  GMRES cold : matvecs={nv} res={res:.2e} du={du:.2e} "
          f"({dt:.1f}s)  -> {dt/max(nv,1):.2f}s/matvec", flush=True)

    # ---- GMRES warm start (2nd source, like the next outer iter) ----
    q1 = q0 * (1.0 + 0.02 * np.random.default_rng(7)
               .normal(size=(psn.nodes, 1)))
    b1 = B_ref.source_map @ q1.ravel()
    u_ref1 = B_ref.factor.solve(b1)
    t0 = time.time()
    u1, nv1, res1, out1 = nl.sweep_gmres(q1, rtol=1e-11, maxiter=600,
                                         verbose=True)
    dt1 = time.time() - t0
    du1 = float(np.abs(u1.ravel() - u_ref1).max() / np.abs(u_ref1).max())
    print(f"  GMRES warm : matvecs={nv1} res={res1:.2e} du={du1:.2e} "
          f"({dt1:.1f}s)  -> {dt1/max(nv1,1):.2f}s/matvec", flush=True)

    # ---- plain GS warm start (same workload, bounded to 600 passes) ----
    nl2 = hs.HoNodeLocal(psn, ii, gg, 3, 'directional', gs_tol=1e-11,
                         max_gs=8000)
    nl2.u = u.ravel().copy()          # same warm start
    t0 = time.time()
    u2, it2, res2, out2 = nl2.sweep(q1, sweeps=600)
    dt2 = time.time() - t0
    print(f"  GS  warm   : passes={it2} res={res2:.2e} "
          f"({dt2:.1f}s)  -> {dt2/max(it2,1)*1000:.0f}ms/pass  "
          f"[600-pass bound, full convergence was 3300+]", flush=True)
    print(flush=True)

print("DONE", flush=True)
