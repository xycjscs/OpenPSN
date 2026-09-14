# -*- coding: utf-8 -*-
"""dbg_sor.py — SOR (omega>1) sweep-count scan for the slow polar
direction i2: find the fastest stable omega."""
import sys
import time

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
i, g = psn.I - 1, 0   # slowest polar direction
B = ho.BlockAngleSystem(psn, i, g, 3, 'directional')
rng = np.random.default_rng(42)
q0 = rng.normal(size=(psn.nodes, 5))
b_ref = B.source_map @ q0.ravel()
u_ref = B.factor.solve(b_ref)

for omega in (1.0, 1.005, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.08):
    nl = hs.HoNodeLocal(psn, i, g, 3, 'directional', gs_tol=1e-12,
                        max_gs=3000, omega=omega)
    t0 = time.time()
    try:
        u, it, res, out = nl.sweep(q0)
        du = float(np.abs(u.ravel() - u_ref).max() / np.abs(u_ref).max())
        print(f"omega={omega:5.3f}: iters={it:5d} res={res:.2e} "
              f"du={du:.2e} t={time.time()-t0:.1f}s", flush=True)
    except (FloatingPointError, ValueError) as e:
        print(f"omega={omega:5.3f}: FAILED ({type(e).__name__}) "
              f"t={time.time()-t0:.1f}s", flush=True)
