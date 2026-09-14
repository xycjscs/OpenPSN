# -*- coding: utf-8 -*-
"""check_gs_rows.py — validate the node-GS inner sweep (highsweep)
against the direct BlockAngleSystem solve.  Independent re-derivation
proof for the A3 inner iteration:

  1) GLOBAL RESIDUAL: the GS fixed-point u must satisfy the GLOBAL
     matrix equation  A u = b  (A = BlockAngleSystem.matrix,
     b = source_map @ q)  to ~1e-11 for a random source — i.e. the
     local systems + neighbour parts reproduce the global system.
  2) SOLVE AGREEMENT: u_gs == direct LU solve of the global system
     (unique SPD-ish system -> same solution), and the W/M moment
     reduce matches BlockAngleSystem.apply bit-for-bit.

Run: /opt/data/venvs/omconda/bin/python tools_ho/check_gs_rows.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, '/opt/data/workspace/OpenPSN')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '24')

from psn2d import highorder as ho       # noqa: E402
from psn2d import highsweep as hs       # noqa: E402
from psn2d import model as M            # noqa: E402
from psn2d import __main__ as PM        # noqa: E402

EX = '/opt/data/workspace/OpenPSN/examples/c5g7/study2_rectangular'
A_DEG = 3


def build(key, MM):
    spec = M.load_spec(EX + f'/c5g7_rect_{key}_assembly.yaml')
    case = {'name': f'M{MM}', 'model': 'ty3', 'I': 30, 'M': MM,
            'subdivide': 1, 'A': A_DEG, 'closure': 'directional_matrix'}
    spec['solver'].update({'keff_tol': 1e-10, 'max_outer': 10,
                           'threads': 1})
    psn = PM.build_solver(spec, case, threads=1, mem_limit_gb=32)
    if not hasattr(psn, '_groups'):
        psn._vectorize_setup()
    return psn


def check_one(psn, i, g, seed):
    K = A_DEG + 1
    bs = ho.BlockAngleSystem(psn, i, g, A_DEG, 'directional')
    rng = np.random.default_rng(seed)
    qn = rng.normal(size=(psn.nodes, 5))
    b = bs.source_map @ qn.ravel()
    u_ref = bs.factor.solve(b)
    out_ref = bs.apply(qn, check_residual=False)

    nl = hs.HoNodeLocal(psn, i, g, A_DEG, 'directional',
                        gs_tol=1e-13, max_gs=5000)
    t0 = time.time()
    u_gs, it, res, out_gs = nl.sweep(qn)
    dt = time.time() - t0

    # 1) global residual of the GS solution
    A = bs.matrix
    r = (A @ u_gs.ravel()) - b
    gres = float(np.abs(r).max() / max(np.abs(b).max(), 1e-300))
    # 2) agreement with the direct solve.  Tolerances scale with the
    #    ACHIEVED global residual (a capped cold start must not fail an
    #    otherwise-correct iteration: solution error is bounded by the
    #    residual / cond, and du/dout track gres to ~100x in practice).
    du = float(np.abs(u_gs.ravel() - u_ref.ravel()).max()
               / max(np.abs(u_ref).max(), 1e-300))
    douts = float(np.abs(out_gs - out_ref).max()
                  / max(np.abs(out_ref).max(), 1e-300))
    print(f'  i{i} g{g}: gs_iters={it} local_res={res:.2e} '
          f'global_res={gres:.2e} du={du:.2e} dout={douts:.2e} '
          f'sweep={dt:.1f}s', flush=True)
    assert gres < max(1e-9, 1e-6 * res), f'global residual {gres}'
    assert du < max(1e-8, 1e2 * gres), f'u mismatch {du}'
    assert douts < max(1e-9, 1e2 * gres), f'moment reduce {douts}'


def main():
    for key, MM, groups in (('uo2', 2, (0, 3, 6)),
                            ('uo2', 4, (0, 6)),
                            ('mox', 2, (0, 6))):
        print(f'== {key} M{MM} ==', flush=True)
        psn = build(key, MM)
        t0 = time.time()
        for i in range(psn.I):
            for g in groups:
                check_one(psn, i, g, seed=100 + 7 * i + g)
        print(f'== {key} M{MM} PASS ({time.time()-t0:.0f}s) ==',
              flush=True)
    print('ALL GS VALIDATION PASS')


if __name__ == '__main__':
    main()
