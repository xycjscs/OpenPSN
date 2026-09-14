# -*- coding: utf-8 -*-
"""dbg_sweepdir.py — test sweep-direction reversal for the slow polar
direction i2: alternating forward/backward row sweeps (the classic
行列交替 of node-method source iteration).  Baseline = current
row/col-major without reversal."""
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
i, g = psn.I - 1, 0
B = ho.BlockAngleSystem(psn, i, g, 3, 'directional')
rng = np.random.default_rng(42)
q0 = rng.normal(size=(psn.nodes, 5))
b_ref = B.source_map @ q0.ravel()
u_ref = B.factor.solve(b_ref)


def order_4way(self):
    p = self.psn
    cen = {}
    for key in self.mc_nodes:
        ns = np.array(self.mc_nodes[key])
        cen[key] = (float(np.mean(p.j_idx[ns])),
                    float(np.mean(p.i_idx[ns])))
    orders = {}
    for parity in (0, 1):
        keys = [k for k in self.mc_nodes if k[3] == parity]
        orders[parity] = (
            sorted(keys, key=lambda k: (cen[k][0], cen[k][1])),
            sorted(keys, key=lambda k: (cen[k][0], -cen[k][1])),
            sorted(keys, key=lambda k: (cen[k][1], cen[k][0])),
            sorted(keys, key=lambda k: (-cen[k][1], cen[k][0])),
        )
    return orders


def sweep_dir(nl, qn, mg=600):
    if nl.u is None:
        nl.u = np.zeros(nl.psn.ncol * nl.K)
    u = nl.u
    orders = order_4way(nl)
    data_cache = {k: nl._microclass(k) for pty in (0, 1)
                  for k in orders[pty][0]}
    it = 0
    res = None
    while it < mg:
        it += 1
        direction = (it - 1) % 4
        for pty in (0, 1):
            for key in orders[pty][direction]:
                nl._mc_pass(data_cache[key], qn)
        if it % 5 == 0:
            res = 0.0
            for key in data_cache:
                res = max(res, nl._mc_residual(data_cache[key], qn))
            if it <= 60 or it % 50 == 0:
                print(f"  it={it:4d} res={res:.2e}", flush=True)
            if res < nl.gs_tol:
                break
    return it, res


for label, patch in (("current (row/col fwd)", None),
                     ("4-way reversal", order_4way)):
    nl = hs.HoNodeLocal(psn, i, g, 3, 'directional', gs_tol=1e-12,
                        max_gs=600)
    if patch is not None:
        nl._mc_order = patch
    t0 = time.time()
    it, res = sweep_dir(nl, q0, mg=600)
    du = float(np.abs(nl.u - u_ref).max() / np.abs(u_ref).max())
    print(f"{label}: it={it} res={res:.2e} du={du:.2e} "
          f"t={time.time()-t0:.1f}s\n", flush=True)
