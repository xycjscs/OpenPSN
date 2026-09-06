"""Equivalence test: vectorized node-state path vs original slow path.

Runs the same problem with the ORIGINAL (backed-up) solver module and with
the modified psn_solver, comparing keff bit-closely on several configs:
  - BWR 2-group (N=2 M=8, N=1 M=24, Gd N=1 M=12)
  - generic checkerboard 4x4 M=8 weak
Any difference > 1e-9 (relative) => FAIL.
"""
import sys, time
import os as _os; sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'core'))
import numpy as np
import importlib

import test_bwr as tb
from test_bwr import bwr_map


def run_fast(N, M, gd=False, generic=False, I=30):
    import psn_solver
    importlib.reload(psn_solver)
    p = psn_solver.PSN2D(bwr_map(gd), 1.5 / N, tb.St, tb.Sgg, tb.nuSf,
                         chi=tb.chi, boundary=('reflect',) * 4, M=M,
                         generic=generic, I=I)
    k, _, _ = p.keff(max_outer=4000, outer_tol=1e-10, verbose=False)
    return k


def run_slow(N, M, gd=False, generic=False, I=30):
    import psn_solver_slow_backup as ps
    p = ps.PSN2D(bwr_map(gd), 1.5 / N, tb.St, tb.Sgg, tb.nuSf,
                 chi=tb.chi, boundary=('reflect',) * 4, M=M,
                 generic=generic, I=I)
    k, _, _ = p.keff(max_outer=4000, outer_tol=1e-10, verbose=False)
    return k


def checker_fast(M):
    import psn_solver
    importlib.reload(psn_solver)
    FUEL = (1.5, 0.15, 0.24); ABS = (1.0, 0.07, 0.0)
    St = np.array([[FUEL[0], ABS[0]]]); Sa = np.array([[FUEL[1], ABS[1]]])
    nuf = np.array([[FUEL[2], ABS[2]]])
    Sgg = np.zeros((1, 1, 2)); Sgg[0, 0, :] = St[0, :] - Sa[0, :]
    S = 4
    g = np.kron(np.array([[0, 1], [1, 0]]), np.ones((S, S), int))
    p = psn_solver.PSN2D(g, 1.0 / S, St, Sgg, nuf, generic=True, I=30, M=M)
    k, _, _ = p.keff(max_outer=4000, outer_tol=1e-10, verbose=False)
    return k


def checker_slow(M):
    import psn_solver_slow_backup as ps
    FUEL = (1.5, 0.15, 0.24); ABS = (1.0, 0.07, 0.0)
    St = np.array([[FUEL[0], ABS[0]]]); Sa = np.array([[FUEL[1], ABS[1]]])
    nuf = np.array([[FUEL[2], ABS[2]]])
    Sgg = np.zeros((1, 1, 2)); Sgg[0, 0, :] = St[0, :] - Sa[0, :]
    S = 4
    g = np.kron(np.array([[0, 1], [1, 0]]), np.ones((S, S), int))
    p = ps.PSN2D(g, 1.0 / S, St, Sgg, nuf, generic=True, I=30, M=M)
    k, _, _ = p.keff(max_outer=4000, outer_tol=1e-10, verbose=False)
    return k


CASES = [
    ('bwr N1 M4  noGd', dict(N=1, M=4)),
    ('bwr N2 M8  noGd', dict(N=2, M=8)),
    ('bwr N1 M24 noGd', dict(N=1, M=24)),
    ('bwr N1 M12 Gd  ', dict(N=1, M=12, gd=True)),
]
ok = True
for name, kw in CASES:
    kf = run_fast(**kw)
    t0 = time.time()
    ks = run_slow(**kw)
    dt = time.time() - t0
    d = abs(kf - ks) / ks
    flag = 'OK ' if d < 1e-9 else 'FAIL'
    if d >= 1e-9:
        ok = False
    print(f'{flag} {name}: fast={kf:.9f} slow={ks:.9f} rel={d:.2e} (slow {dt:.1f}s)')

for M in (8,):
    kf = checker_fast(M)
    t0 = time.time()
    ks = checker_slow(M)
    dt = time.time() - t0
    d = abs(kf - ks) / ks
    flag = 'OK ' if d < 1e-9 else 'FAIL'
    if d >= 1e-9:
        ok = False
    print(f'{flag} checker4x4 M{M} generic: fast={kf:.9f} slow={ks:.9f} '
          f'rel={d:.2e} (slow {dt:.1f}s)')

print('RESULT:', 'PASS — vectorized path identical to validated original'
      if ok else 'FAIL — do NOT use the modified solver')
