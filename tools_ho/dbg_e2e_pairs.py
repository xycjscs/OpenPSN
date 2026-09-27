# -*- coding: utf-8 -*-
"""Lever-1 smoke + e2e: pair-block GMRES vs system GMRES / legacy LU.

Stage A (smoke, ~1 min): 1 outer iteration, M4.  Compares the pair
mode's five-moment flux against the legacy system mode (same warm
state) -- catches shape/index bugs before any long run.
Stage B (e2e, ~20-40 min each): full convergence, M4 and M8.
  reference: legacy LU keff (dir_table3_sweep.log):
    M4_S1  keff=1.3349468
    M8_S1  keff=1.3352414
  same-epoch old GMRES pool anchor (bench_m8_old.log):
    M8_S1  keff=1.3352414 wall=1059s
Run: python dbg_e2e_pairs.py [smoke|M4|M8]
"""
import os, sys, time, resource
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, '/opt/data/workspace/OpenPSN')

import numpy as np
from psn2d import model
from psn2d import __main__ as M
from psn2d import highsweep

EX = '/opt/data/workspace/OpenPSN/examples/c5g7/study2_rectangular'
REF = {'M4': 1.3349468, 'M8': 1.3352414}
STAGE = sys.argv[1] if len(sys.argv) > 1 else 'smoke'


def build(MM):
    spec = model.load_spec(EX + '/c5g7_rect_uo2_assembly.yaml')
    case = {'name': f'M{MM}_S1', 'model': 'ty3', 'I': 30, 'M': MM,
            'subdivide': 1, 'kref': 1.3347260, 'A': 3,
            'closure': 'directional_matrix'}
    spec['solver'].update({'keff_tol': 1e-10, 'max_outer': 6000,
                           'threads': 1})
    return M.build_solver(spec, case, threads=1, mem_limit_gb=120)


def one_outer(psn, pairs):
    """Single inner sweep (1 GMRES over all blocks), returns the
    five-moment qim -- enough to compare the two code paths."""
    t0 = time.time()
    matidx = psn.mat[psn.j_idx, psn.i_idx]
    ng = psn.ng
    locals_ = {}
    for i in range(psn.I):
        for g in range(ng):
            locals_[(i, g)] = highsweep.HoNodeLocal(
                psn, i, g, 3, 'directional', gs_tol=1e-11, max_gs=1)
    qnode = psn._source_update(np.zeros((ng, psn.nodes, 5)), 1.0, matidx)
    qnode[:, :, 0] = qnode[:, :, 0]  # no-op keep shape
    mom = np.zeros((ng, psn.nodes, 5))
    mom[:, :, 0] = 1.0
    qnode = psn._source_update(mom, 1.0, matidx)
    # one inner sweep via the driver's block machinery
    import multiprocessing as mp
    ctx = mp.get_context('fork')
    npair = psn.M // 2
    use_pairs = (npair >= 2) if pairs is None else bool(pairs)
    IDX = [(i, g) for i in range(psn.I) for g in range(ng)]
    BLOCKS = ([(i, g, pp) for (i, g) in IDX for pp in range(npair)]
              if use_pairs else [(i, g, -1) for (i, g) in IDX])
    highsweep._POOL, highsweep._IDX = locals_, IDX
    highsweep._RTOL, highsweep._MAXIT = 1e-11, 800
    highsweep._BLK = BLOCKS
    B = len(BLOCKS)
    NW2 = min(B, os.cpu_count() or 1)
    task = (highsweep._gmres_pool_task_pair if use_pairs
            else highsweep._gmres_pool_task)
    pools = [ctx.Pool(1, initializer=highsweep._gmres_pool_init)
             for _ in range(NW2)]
    qim = np.zeros((ng, psn.nodes, 5))
    try:
        args_w = [[] for _ in range(NW2)]
        for bidx in range(B):
            args_w[bidx % NW2].append((bidx, qnode[BLOCKS[bidx][1]]))
        futs = [p.map_async(task, a) for p, a in zip(pools, args_w)]
        results = [None] * B
        for w in range(NW2):
            for bidx, res in zip(range(w, B, NW2), futs[w].get()):
                results[bidx] = res
        for (i, g, pp), (out, nv, r, t) in zip(BLOCKS, results):
            qim[g] += out
        dt = time.time() - t0
        maxres = max(r[2] for r in results)
    finally:
        for p in pools:
            p.close(); p.join()
    return qim, dt, maxres


if STAGE == 'smoke':
    psn = build(4)
    print('M4 smoke: system mode 1 outer ...', flush=True)
    qim_s, dt_s, res_s = one_outer(psn, pairs=False)
    # rebuild identical psn for pair mode (fresh HoNodeLocal state)
    psn2 = build(4)
    print('M4 smoke: pair mode 1 outer ...', flush=True)
    qim_p, dt_p, res_p = one_outer(psn2, pairs=True)
    d = float(np.abs(qim_s - qim_p).max())
    rel = d / float(np.abs(qim_s).max())
    print(f'smoke M4: system dt={dt_s:.2f}s maxres={res_s:.2e} | '
          f'pair dt={dt_p:.2f}s maxres={res_p:.2e}')
    print(f'smoke M4: max|dqim| = {d:.3e}  rel = {rel:.3e}')
    # gate 1e-8: catches shape/index bugs (O(1) errors); the two
    # Krylov spaces legitimately differ at the rtol ~1e-11 level
    ok = rel < 1e-8
    print('SMOKE', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)

MM = int(STAGE[1:])
psn = build(MM)
t0 = time.time()
k, phi, qim, hinfo = highsweep.keff_highorder_gmres(
    psn, A=3, branch='directional', max_outer=6000, outer_tol=1e-10,
    rtol=1e-11, maxiter=800, pairs=True)
wall = time.time() - t0
ref = REF[STAGE]
dk = abs(k - ref)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f'e2e {STAGE}_S1 PAIRS keff={k:.7f} ref(legacy LU)={ref:.7f} '
      f'dkeff={dk:.3e} iters={hinfo["outer_iterations"]} '
      f'blocks={hinfo["nblocks"]} workers={hinfo["nworkers"]} '
      f'matvecs={hinfo["gmres_matvecs_total"]} wall={wall:.0f}s '
      f'peak={peak:.2f}GB', flush=True)
print('E2E', 'PASS' if dk < 2e-12 else 'FAIL')
sys.exit(0 if dk < 2e-12 else 1)
