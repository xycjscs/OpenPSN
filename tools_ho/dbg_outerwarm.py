# -*- coding: utf-8 -*-
"""dbg_outerwarm.py — REALISTIC warm-start measurement.

Run the LU direct driver (keff_highorder) for 10 outer iterations,
storing the source qnode at each outer step.  Then, for each
consecutive source pair (outer k -> k+1), solve the slow (i2, g0)
(i,g) system with warm-started node-GS (u inherited from the k-1
solution) and count the passes needed to reach gs_tol.  This is the
production inner-iteration workload.
"""
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
ng = psn.ng
matidx = psn.mat[psn.j_idx, psn.i_idx]
area_w = psn.area.ravel()


def fission_rate(phi):
    fr = np.zeros(psn.nodes)
    for g in range(ng):
        fr += psn.nuSf[g, matidx] * phi[g]
    return fr * area_w


# ---- LU outer trajectory (store sources) ----
locals_lu = {}
for i in range(psn.I):
    for g in range(ng):
        locals_lu[(i, g)] = ho.BlockAngleSystem(psn, i, g, 3,
                                                'directional')
lam = 1.0
mom = np.zeros((ng, psn.nodes, 5))
mom[:, :, 0] = 1.0
qnode = psn._source_update(mom, 1.0, matidx)
F_old = float(fission_rate(np.ones((ng, psn.nodes))).sum())
sources = [qnode.copy()]
print("LU outer trajectory:", flush=True)
t0 = time.time()
for outer in range(10):
    phi = np.zeros((ng, psn.nodes))
    qim = np.zeros((ng, psn.nodes, 5))
    for g in range(ng):
        for i in range(psn.I):
            out = locals_lu[(i, g)].apply(qnode[g])
            phi[g] += out[:, 0]
            qim[g] += out
    F_new = float(fission_rate(phi).sum())
    lam_new = lam * F_new / F_old
    lam = lam_new
    qnode = psn._source_update(qim, lam, matidx)
    F_old = F_new
    sources.append(qnode.copy())
    d = abs(lam - lam_new) / lam_new
    print(f"  outer {outer:2d} keff={lam:.6f} dkeff={d:.2e} "
          f"(+{time.time()-t0:6.1f}s)", flush=True)

# ---- warm-start GS on the slow system for each source pair ----
i2, g0 = psn.I - 1, 0
nl = hs.HoNodeLocal(psn, i2, g0, 3, 'directional',
                    gs_tol=1e-11, max_gs=8000)
print(f"\nwarm-start GS on (i={i2}, g={g0}), tol 1e-11:", flush=True)
u_inherited = None
for k in range(1, len(sources)):
    # perturb check: how much did the source change?
    qk, qprev = sources[k], sources[k - 1]
    dq = float(np.linalg.norm(qk[g0] - qprev[g0]) /
               np.linalg.norm(qprev[g0]))
    if u_inherited is not None:
        nl.u = u_inherited
    else:
        nl.u = np.zeros(psn.ncol * nl.K)
        cold = True
    t0 = time.time()
    u, it, res, out = nl.sweep(qk[g0])
    dt = time.time() - t0
    u_inherited = nl.u.copy()
    tag = "COLD" if (k == 1) else "warm "
    print(f"  outer {k:2d}  dq={dq:.3e}  {tag} iters={it:5d} "
          f"res={res:.2e}  {dt:7.1f}s", flush=True)

# also the fast system (i=0, g=0) for contrast
i0, g0b = 0, 0
nl2 = hs.HoNodeLocal(psn, i0, g0b, 3, 'directional',
                     gs_tol=1e-11, max_gs=8000)
print(f"\ncontrast system (i={i0}, g={g0b}):", flush=True)
u_inherited = None
for k in range(1, len(sources)):
    qk, qprev = sources[k], sources[k - 1]
    dq = float(np.linalg.norm(qk[g0b] - qprev[g0b])
               / np.linalg.norm(qprev[g0b]))
    if u_inherited is not None:
        nl2.u = u_inherited
    else:
        nl2.u = np.zeros(psn.ncol * nl2.K)
    t0 = time.time()
    u, it, res, out = nl2.sweep(qk[g0b])
    dt = time.time() - t0
    u_inherited = nl2.u.copy()
    tag = "COLD" if (k == 1) else "warm "
    print(f"  outer {k:2d}  dq={dq:.3e}  {tag} iters={it:5d} "
          f"res={res:.2e}  {dt:7.1f}s", flush=True)
