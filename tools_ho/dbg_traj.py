# -*- coding: utf-8 -*-
"""dbg_traj.py — GS residual trajectory (is it converging or diverging?)."""
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
i, g = 0, 0
B = ho.BlockAngleSystem(psn, i, g, 3, 'directional')
nl = hs.HoNodeLocal(psn, i, g, 3, 'directional', gs_tol=1e-13,
                    max_gs=40, verbose=True)

rng = np.random.default_rng(100)
qn = rng.normal(size=(psn.nodes, 5))
try:
    u_gs, it, res, out_gs = nl.sweep(qn)
    print("converged?", it, res)
except FloatingPointError:
    print("overflow")
# inspect u growth
print("max|u| =", np.max(np.abs(nl.u)))
