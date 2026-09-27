# -*- coding: utf-8 -*-
"""Lever-3 microbenchmark: b-assembly and full matvec in isolation,
one thread, no pool overhead (kernel-level lever-3 number).

Per (i, g) system, over EVERY microclass:
  b-assembly : numpy _mc_b (pack stripped) vs mckern.mc_b (T=1)
  full matvec: 4-order red-black pass, numpy b vs numba b
Median over repeated calls.  M4 default; ~3-5 min.
Run: python dbg_bench_mcmatvec.py [M]
"""
import os, sys, time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, '/opt/data/workspace/OpenPSN')
import numpy as np
from psn2d import model
from psn2d import __main__ as M
from psn2d import highsweep
from psn2d import mckern

EX = '/opt/data/workspace/OpenPSN/examples/c5g7/study2_rectangular'
MM = int(sys.argv[1]) if len(sys.argv) > 1 else 4


def build(MM):
    spec = model.load_spec(EX + '/c5g7_rect_uo2_assembly.yaml')
    case = {'name': f'M{MM}_S1', 'model': 'ty3', 'I': 30, 'M': MM,
            'subdivide': 1, 'kref': 1.3347260, 'A': 3,
            'closure': 'directional_matrix'}
    spec['solver'].update({'keff_tol': 1e-10, 'max_outer': 6000,
                           'threads': 1})
    return M.build_solver(spec, case, threads=1, mem_limit_gb=120)


def timeit(fn, n):
    ts = []
    for _ in range(n):
        t0 = time.time()
        fn()
        ts.append(time.time() - t0)
    return float(np.median(ts))


psn = build(MM)
if not hasattr(psn, '_groups'):
    psn._vectorize_setup()
matidx = psn.mat[psn.j_idx, psn.i_idx]
ng = psn.ng
mom = np.zeros((ng, psn.nodes, 5))
mom[:, :, 0] = 1.0
qnode = psn._source_update(mom, 1.0, matidx)
IDX = [(i, g) for i in range(psn.I) for g in range(ng)]

assert mckern.HAVE_NUMBA, "numba required for this benchmark"
mckern.set_num_threads(1)

tot_b_np = tot_b_nb = 0.0
tot_mv_np = tot_mv_nb = 0.0
n_mc = n_mv = 0
for (i, g) in IDX:
    local = highsweep.HoNodeLocal(psn, i, g, 3, 'directional',
                                  gs_tol=1e-11, max_gs=1)
    local.u = np.zeros(psn.ncol * (3 + 1))
    qn = qnode[g]
    orders = local._mc_order()
    seen = []
    for pty in (0, 1):
        for key in orders[pty][0] + orders[pty][1]:
            if key not in seen:
                seen.append(key)
    cache = {k: local._microclass(k) for k in seen}
    data_np = {k: {kk: vv for kk, vv in d.items() if kk != 'pack'}
               for k, d in cache.items()}
    for key in seen:
        d = data_np[key]
        dn = cache[key]
        if not d["nodes"].size:
            continue
        n_mc += 1
        tot_b_np += timeit(lambda: local._mc_b(d, qn), 30)
        tot_b_nb += timeit(lambda: mckern.mc_b(dn["pack"], local.u, qn), 30)
        def mv_np():
            for direction in (0, 1):
                for pty in (0, 1):
                    for k2 in orders[pty][direction]:
                        local._mc_pass(data_np[k2], qn)
        def mv_nb():
            for direction in (0, 1):
                for pty in (0, 1):
                    for k2 in orders[pty][direction]:
                        local._mc_pass(cache[k2], qn)
        n_mv += 1
        tot_mv_np += timeit(mv_np, 10)
        tot_mv_nb += timeit(mv_nb, 10)
        del local
    print(f'  (i={i},g={g}) done, running totals: '
          f'b_np {tot_b_np*1e3:.0f}ms b_nb {tot_b_nb*1e3:.0f}ms '
          f'mv_np {tot_mv_np*1e3:.0f}ms mv_nb {tot_mv_nb*1e3:.0f}ms',
          flush=True)

print(f'\nM{MM} b-assembly: {n_mc} microclasses | numpy '
      f'{tot_b_np/n_mc*1e3:.3f}ms -> numba {tot_b_nb/n_mc*1e3:.3f}ms '
      f'= {tot_b_np/tot_b_nb:.2f}x (T=1)')
print(f'M{MM} full matvec (4-order pass): numpy '
      f'{tot_mv_np/n_mv*1e3:.3f}ms -> numba {tot_mv_nb/n_mv*1e3:.3f}ms '
      f'= {tot_mv_np/tot_mv_nb:.2f}x (T=1)')
print('MCMATVEC BENCH DONE')
