# -*- coding: utf-8 -*-
"""dbg_e2e2.py — end-to-end keff agreement: node-GS (red-black,
row/column alternating) inner sweep  vs  direct-SuperLU inner solve,
same plain power-iteration outer driver, M2 uo2 assembly, 15 outer.
Prints both keff trajectories + final diff."""
import sys, time

import numpy as np

sys.path.insert(0, '/opt/data/workspace/OpenPSN')
from psn2d import highorder as ho
from psn2d import highsweep as hs
from psn2d import model as mdl
from psn2d import __main__ as PM

EX = '/opt/data/workspace/OpenPSN/examples/c5g7/study2_rectangular'
NOUT = 15


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


def run(psn, use_gs, gs_kw):
    ng = psn.ng
    matidx = psn.mat[psn.j_idx, psn.i_idx]
    area_w = psn.area.ravel()

    def frate(phi):
        fr = np.zeros(psn.nodes)
        for g in range(ng):
            fr += psn.nuSf[g, matidx] * phi[g]
        return fr * area_w

    if use_gs:
        locals_ = {(i, g): hs.HoNodeLocal(psn, i, g, 3, 'directional',
                                          **gs_kw)
                   for i in range(psn.I) for g in range(ng)}

        def agroups(qnode):
            phi = np.zeros((ng, psn.nodes))
            qim = np.zeros((ng, psn.nodes, 5))
            git = 0
            for g in range(ng):
                for i in range(psn.I):
                    _, it, res, out = locals_[(i, g)].sweep(qnode[g])
                    git = max(git, it)
                    phi[g] += out[:, 0]
                    qim[g] += out
            return phi, qim, git
    else:
        locals_ = {(i, g): ho.BlockAngleSystem(psn, i, g, 3,
                                               'directional')
                   for i in range(psn.I) for g in range(ng)}

        def agroups(qnode):
            phi = np.zeros((ng, psn.nodes))
            qim = np.zeros((ng, psn.nodes, 5))
            for g in range(ng):
                for i in range(psn.I):
                    out = locals_[(i, g)].apply(qnode[g])
                    phi[g] += out[:, 0]
                    qim[g] += out
            return phi, qim, 0

    lam = 1.0
    mom = np.zeros((ng, psn.nodes, 5))
    mom[:, :, 0] = 1.0
    qnode = psn._source_update(mom, 1.0, matidx)
    F_old = float(frate(np.ones((ng, psn.nodes))).sum())
    traj = []
    t0 = time.time()
    for outer in range(NOUT):
        phi, qim, git = agroups(qnode)
        F_new = float(frate(phi).sum())
        lam_new = lam * F_new / F_old
        dlam = abs(lam_new - lam) / lam_new
        lam = lam_new
        qnode = psn._source_update(qim, lam, matidx)
        F_old = F_new
        traj.append(lam)
        tag = 'GS' if use_gs else 'LU'
        print(f"  [{tag}] outer {outer:2d} keff={lam:.7f} "
              f"dkeff={dlam:.2e} gs={git} "
              f"({time.time()-t0:6.1f}s)", flush=True)
    return lam, traj


print("== LU direct (reference) ==", flush=True)
lam_lu, traj_lu = run(build_psn('uo2', 2), False, {})
print("\n== GS node-sweep ==", flush=True)
gs_kw = dict(gs_tol=1e-10, max_gs=6000, omega=1.0, verbose=False)
lam_gs, traj_gs = run(build_psn('uo2', 2), True, gs_kw)
print(f"\n== FINAL: LU keff={lam_lu:.7f}   GS keff={lam_gs:.7f}   "
      f"diff={abs(lam_lu-lam_gs):.2e} ==", flush=True)
print(f"traj max |d keff| = {max(abs(a-b) for a, b in zip(traj_lu, traj_gs)):.2e}")
