# -*- coding: utf-8 -*-
"""dbg_pool_outer.py — cross-(i,g) parallel outer iteration via a fork
process pool (21 workers, BLAS pinned to 1 thread each).

Math: within one outer iteration all sources are frozen, so the 21
(i,g) face systems are independent (zero shared writes).  Each worker
keeps its own HoNodeLocal.u (warm start) in a per-idx state dict, so
the system-to-worker assignment can change between outers without
losing the warm start.  Driver logic is identical to dbg_gmres_outer.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import sys, time
import multiprocessing as mp

sys.path.insert(0, '/opt/data/workspace/OpenPSN')
import numpy as np
from psn2d import highsweep as hs
from psn2d import model as mdl
from psn2d import __main__ as PM

EX = '/opt/data/workspace/OpenPSN/examples/c5g7/study2_rectangular'

psn = None
IDX = []
_STATE = {}


def _task(arg):
    idx, qg = arg
    i, g = IDX[idx]
    nl = _STATE.get(idx)
    if nl is None:
        nl = hs.HoNodeLocal(psn, i, g, 3, 'directional',
                            gs_tol=1e-11, max_gs=1)
        _STATE[idx] = nl
    t0 = time.time()
    u, nv, res, out = nl.sweep_gmres(qg, rtol=1e-11, maxiter=600)
    return out, nv, res, time.time() - t0


def main():
    global psn, IDX
    spec = mdl.load_spec(EX + '/c5g7_rect_uo2_assembly.yaml')
    case = {'name': 'M2', 'model': 'ty3', 'I': 30, 'M': 2,
            'subdivide': 1, 'A': 3, 'closure': 'directional_matrix'}
    spec['solver'].update({'keff_tol': 1e-10, 'max_outer': 200,
                           'threads': 1})
    psn = PM.build_solver(spec, case, threads=1, mem_limit_gb=32)
    if not hasattr(psn, '_groups'):
        psn._vectorize_setup()
    ng, I = psn.ng, psn.I
    IDX = [(i, g) for i in range(I) for g in range(ng)]
    matidx = psn.mat[psn.j_idx, psn.i_idx]
    area_w = psn.area.ravel()
    print(f"I={I} ng={ng} -> {len(IDX)} (i,g) systems, "
          f"nodes={psn.nodes}, ncol={psn.ncol}", flush=True)

    def frate(phi):
        fr = np.zeros(psn.nodes)
        for g in range(ng):
            fr += psn.nuSf[g, matidx] * phi[g]
        return fr * area_w

    mom = np.zeros((ng, psn.nodes, 5))
    mom[:, :, 0] = 1.0
    qnode = psn._source_update(mom, 1.0, matidx)
    lam = 1.0
    F_old = float(frate(np.ones((ng, psn.nodes))).sum())

    NT = len(IDX)
    t_all = time.time()
    pool = mp.Pool(NT)
    for outer in range(3):
        phi = np.zeros((ng, psn.nodes))
        qim = np.zeros((ng, psn.nodes, 5))
        args = [(idx, qnode[g]) for idx, (i, g) in enumerate(IDX)]
        t1 = time.time()
        res = pool.map(_task, args)
        dt = time.time() - t1
        for (i, g), (out, nv, r, tv) in zip(IDX, res):
            phi[g] += out[:, 0]
            qim[g] += out
        F_new = float(frate(phi).sum())
        lam = lam * F_new / F_old
        qnode = psn._source_update(qim, lam, matidx)
        F_old = F_new
        nvs = [r[1] for r in res]
        tvs = [r[3] for r in res]
        tag = 'COLD' if outer == 0 else 'warm '
        print(f"outer {outer} [{tag}]: {dt:6.2f}s  keff={lam:.7f}  "
              f"matvecs {min(nvs)}-{max(nvs)}  "
              f"worst {max(tvs):.2f}s  ({len(IDX)} systems on "
              f"{NT} procs)", flush=True)
    pool.close()
    pool.join()
    print(f"total {time.time()-t_all:.1f}s", flush=True)


if __name__ == '__main__':
    main()
