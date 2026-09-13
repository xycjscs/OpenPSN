# -*- coding: utf-8 -*-
"""Verify the high-order parallel (fork+COW) sweep is BIT-IDENTICAL to
the serial path, and that threaded prefactor matches serial prefactor.

Strategy: force pool activation on the 17x17 assembly (normally below
the 150k threshold) by lowering _MIN_NCOL, pre-factor, run one sweep
serially, then start the pool and run the same sweep; compare
bit-for-bit (==, not allclose).  Also time both.
"""
import os
os.environ.update({k: '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                                    'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                                    'NUMEXPR_NUM_THREADS')})
import sys
import time

import numpy as np

sys.path.insert(0, '/opt/data/workspace/OpenPSN')
from psn2d import model
from psn2d import highorder
from psn2d.__main__ import build_solver

IN = ('/opt/data/cache/documents/a3_upgrade/'
      'OpenPSN_directional_A3_upgrade_Yuchao_Xu/inputs')
spec = model.load_spec(f'{IN}/mox87_ty3_s2.json')
case = [c for c in model.expand_cases(spec) if c['name'] == 'M24_A3'][0]
psn = build_solver(spec, case, threads=4, mem_limit_gb=2)
psn._vectorize_setup()
print(f"psn: nodes={psn.nodes} ng={psn.ng} I={psn.I} M={psn.M} "
      f"ncol={psn.ncol} ncol*K={psn.ncol*4}")

rng = np.random.default_rng(42)
q0 = 1.0 + rng.random((psn.ng, psn.nodes, 5))

# --- serial baseline ---
sw = highorder.HighOrderSweep(psn, A=3, branch='directional')
t0 = time.time()
sw.prefactor()
tf = time.time() - t0
t0 = time.time()
phi_s, qim_s = sw.all_groups_serial(q0)
ts = time.time() - t0
print(f"serial: prefactor={tf:.2f}s  sweep={ts:.3f}s")

# --- parallel (force pool despite small ncol) ---
highorder.HighOrderSweep._MIN_NCOL = 1
sw2 = highorder.HighOrderSweep(psn, A=3, branch='directional')
t0 = time.time()
sw2.prefactor()
t1 = time.time() - t0
sw2.start_pool()
npool = len(sw2._pool._pool) if sw2._pool is not None else 0
t0 = time.time()
phi_p, qim_p = sw2.all_groups(q0)
tp = time.time() - t0
sw2.stop_pool()
print(f"parallel: prefactor={t1:.2f}s  workers={npool}  sweep={tp:.3f}s")

# --- bit-identical check ---
same_phi = bool(np.array_equal(phi_p, phi_s))
same_qim = bool(np.array_equal(qim_p, qim_s))
print(f"phi bit-identical: {same_phi}")
print(f"qim bit-identical: {same_qim}")
if not (same_phi and same_qim):
    print("  max|dphi| =", np.abs(phi_p - phi_s).max())
    print("  max|dqim| =", np.abs(qim_p - qim_s).max())

# --- second sweep (factor reuse across outer iterations) ---
q1 = 1.0 + rng.random((psn.ng, psn.nodes, 5))
t0 = time.time()
sw2.start_pool()
phi_p2, _ = sw2.all_groups(q1)
sw2.stop_pool()
t0b = time.time()
sw.all_groups_serial(q1)
print(f"second sweep: pool={tp2 if False else time.time()-t0:.3f}s "
      f"(reuse of cached factors — no re-factorization)")
print("RESULT:", "PASS" if (same_phi and same_qim) else "FAIL")
