# -*- coding: utf-8 -*-
"""dbg_gmres_outer.py — wall time of one FULL outer iteration on the
M2 uo2 assembly with the GMRES(red-black) inner sweep (90 (i,g)
systems per outer, warm-started between outers)."""
import sys, time

import numpy as np

sys.path.insert(0, '/opt/data/workspace/OpenPSN')
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


def frate(phi):
    fr = np.zeros(psn.nodes)
    for g in range(ng):
        fr += psn.nuSf[g, matidx] * phi[g]
    return fr * area_w


locals_ = {(i, g): hs.HoNodeLocal(psn, i, g, 3, 'directional',
                                  gs_tol=1e-11, max_gs=800)
           for i in range(psn.I) for g in range(ng)}
t0 = time.time()
lam = 1.0
mom = np.zeros((ng, psn.nodes, 5))
mom[:, :, 0] = 1.0
qnode = psn._source_update(mom, 1.0, matidx)
F_old = float(frate(np.ones((ng, psn.nodes))).sum())
for outer in range(4):
    phi = np.zeros((ng, psn.nodes))
    qim = np.zeros((ng, psn.nodes, 5))
    t1 = time.time()
    worst = ('', 0, 0.0)
    for g in range(ng):
        for i in range(psn.I):
            u, nv, res, out = locals_[(i, g)].sweep_gmres(qnode[g],
                                                          rtol=1e-11,
                                                          maxiter=600)
            if nv > worst[1]:
                worst = (f'(i={i},g={g})', nv, time.time() - t1)
            phi[g] += out[:, 0]
            qim[g] += out
    dt = time.time() - t1
    F_new = float(frate(phi).sum())
    lam_new = lam * F_new / F_old
    lam = lam_new
    qnode = psn._source_update(qim, lam, matidx)
    F_old = F_new
    tag = 'COLD' if outer == 0 else 'warm'
    print(f"outer {outer} [{tag}]: {dt:7.1f}s  keff={lam:.7f}  "
          f"slowest system {worst[0]} matvecs={worst[1]}", flush=True)
print(f"total {time.time()-t0:.1f}s", flush=True)
