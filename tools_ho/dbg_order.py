# -*- coding: utf-8 -*-
"""dbg_order.py — node-ordering experiment for the slow i2 system:
compare sweep cycles [row, col] (current) vs [diag45, diag135] vs
[diag45, diag135, row, col], on the class-batch GS (mathematically
identical, only the update order changes)."""
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
A_f = B.matrix

nl = hs.HoNodeLocal(psn, i, g, 3, 'directional', gs_tol=1e-12, max_gs=100)

# build per-order class sequences.  Node position: (j = row, i = col)
jidx = psn.j_idx
iidx = psn.i_idx
keypos = {}
for key in nl.mc_nodes:
    ns = np.array(nl.mc_nodes[key])
    # representative position: min of (j, i) lexicographic per order type
    keypos[key] = (jidx[ns], iidx[ns])


def order_by(sortkey):
    orders = {}
    for parity in (0, 1):
        keys = [k for k in nl.mc_nodes if k[3] == parity]
        ks = sorted(keys, key=lambda k: sortkey(k))
        orders[parity] = ks
    return orders


def sortkey_row(k):
    j, i = keypos[k]
    return (int(j.min()), int(i.min()))


def sortkey_col(k):
    j, i = keypos[k]
    return (int(i.min()), int(j.min()))


def sortkey_d1(k):     # 45-deg ray (+x+y): sort by (j+i) then (j-i)
    j, i = keypos[k]
    return (int((j + i).min()), int((j - i).min()))


def sortkey_d2(k):     # 135-deg ray (-x+y): sort by (j-i) then (j+i)
    j, i = keypos[k]
    return (int((j - i).min()), int((j + i).min()))


cycles = {
    "row,col": (sortkey_row, sortkey_col),
    "d45,d135": (sortkey_d1, sortkey_d2),
    "row,col,d45,d135": (sortkey_row, sortkey_col, sortkey_d1, sortkey_d2),
}

data_cache = {k: nl._microclass(k) for k in nl.mc_nodes}

for name, sks in cycles.items():
    seqs = [order_by(sk) for sk in sks]
    u = np.zeros(psn.ncol * nl.K)
    nl.u = u
    it = 0
    mg = 1500
    res = None
    t0 = time.time()
    while it < mg:
        it += 1
        direction = (it - 1) % len(seqs)
        for pty in (0, 1):
            for key in seqs[direction][pty]:
                nl._mc_pass(data_cache[key], q0)
        if it % 10 == 0:
            r_f = A_f @ u - b_ref
            res = float(np.abs(r_f).max() / np.abs(b_ref).max())
            if it <= 40 or it % 50 == 0:
                print(f"  [{name}] it={it:4d} res={res:.2e}", flush=True)
            if res < 1e-12:
                break
    du = float(np.abs(u - u_ref).max() / np.abs(u_ref).max())
    print(f"[{name}] done it={it} res={res:.2e} du={du:.2e} "
          f"t={time.time()-t0:.1f}s\n", flush=True)
