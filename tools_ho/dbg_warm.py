# -*- coding: utf-8 -*-
"""dbg_warm.py — warm-start GS iteration counts (production scenario):
after a cold solve, perturb q by small amounts (like consecutive power
iterations) and count sweeps to re-converge from the inherited u."""
import sys
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
    spec['solver'].update({'keff_tol': 1e-10, 'max_outer': 10, 'threads': 1})
    psn = PM.build_solver(spec, case, threads=1, mem_limit_gb=32)
    if not hasattr(psn, '_groups'):
        psn._vectorize_setup()
    return psn


psn = build_psn()
i, g = 0, 0  # slowest system for the fast polar direction
B = ho.BlockAngleSystem(psn, i, g, 3, 'directional')
nl = hs.HoNodeLocal(psn, i, g, 3, 'directional', gs_tol=1e-13, max_gs=5000)

rng = np.random.default_rng(42)
q0 = rng.normal(size=(psn.nodes, 5))
u, it_cold, res, out = nl.sweep(q0)
print(f"cold: iters={it_cold} res={res:.2e}")

# warm: perturb q at several amplitudes (power-iteration-like changes)
for amp in (0.5, 0.1, 0.05, 0.01, 0.001):
    q1 = q0 * (1 + amp * rng.normal(size=(psn.nodes, 1)))
    u, it_warm, res, out = nl.sweep(q1)
    print(f"amp={amp:6.3f}: iters={it_warm} res={res:.2e}")

# and the slow polar direction
i2 = psn.I - 1
nl2 = hs.HoNodeLocal(psn, i2, 0, 3, 'directional', gs_tol=1e-13, max_gs=8000)
u, it_cold, res, out = nl2.sweep(q0)
print(f"\ni2 g0 cold: iters={it_cold} res={res:.2e}")
for amp in (0.1, 0.01, 0.001):
    q1 = q0 * (1 + amp * rng.normal(size=(psn.nodes, 1)))
    u, it_warm, res, out = nl2.sweep(q1)
    print(f"i2 g0 amp={amp:6.3f}: iters={it_warm} res={res:.2e}")
