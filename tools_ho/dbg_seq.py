# -*- coding: utf-8 -*-
"""dbg_seq.py — sequential per-node true Gauss-Seidel prototype:
update nodes ONE BY ONE in row-major / column-major alternating order
(reading the freshest neighbour values), reusing the microclass factors.
Measure sweep counts for the slow polar direction i2."""
import sys
import time

import numpy as np

sys.path.insert(0, '/opt/data/workspace/OpenPSN')
from psn2d import highorder as ho
from psn2d import highsweep as hs
from psn2d import model as mdl
from psn2d import __main__ as PM
from scipy.linalg import lu_solve

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
K, M = 4, psn.M
i, g = psn.I - 1, 0
B = ho.BlockAngleSystem(psn, i, g, 3, 'directional')
nl = hs.HoNodeLocal(psn, i, g, 3, 'directional', gs_tol=1e-12, max_gs=100)

rng = np.random.default_rng(42)
q0 = rng.normal(size=(psn.nodes, 5))
b_ref = B.source_map @ q0.ravel()
u_ref = B.factor.solve(b_ref)

# per-node b builder using microclass data (one node at a time)
u = np.zeros(psn.ncol * K)


def node_b(n, qn):
    """RHS for node n's local system at the current u."""
    key = nl.mc_of[n]
    data = nl._microclass(key)
    nlist = data["nodes"]
    ii = int(np.searchsorted(nlist, n))
    d = data["d"]
    b = (data["SRC"] @ qn[n]).copy()
    for lf, parts in data["nb"].items():
        # find the part containing this node's neighbour
        nb = nl.nf[n][lf][1]
        for part in parts:
            pos = np.searchsorted(part["nodes_nb"], nb)
            if pos < len(part["nodes_nb"]) and \
                    part["nodes_nb"][pos] == nb:
                s2 = int(pos)
                # u-part: C0 (m,a,f,s) x sgn_nb[s2] x u_nb
                un = u[part["gath"][s2]].reshape(4, M, K).transpose(1, 0, 2)
                prod = part["sgn_nb"][s2] * un
                contrib = np.einsum("mafs, mfs -> ma", part["C0"], prod)
                b[part["rowidx"]] += contrib.ravel()
                # q-part
                src = np.einsum("q,maq->ma", qn[nb], part["src5"])
                b[part["rowidx"]] += part["sigma"] * src.ravel()
                break
    return b


# node order: row-major (j then i) and column-major (i then j)
jidx = psn.j_idx
iidx = psn.i_idx
row_order = np.argsort(jidx * 1000 + iidx)
col_order = np.argsort(iidx * 1000 + jidx)


def res_global():
    A = B.matrix
    r = A @ u - b_ref
    return float(np.abs(r).max() / max(np.abs(b_ref).max(), 1e-300))


for name, order in (("row", row_order), ("col", col_order)):
    u.fill(0.0)
    t0 = time.time()
    it = 0
    mg = 3000
    while it < mg:
        it += 1
        for n in order:
            key = nl.mc_of[n]
            data = nl._microclass(key)
            nlist = data["nodes"]
            ii = int(np.searchsorted(nlist, n))
            b = node_b(n, q0)
            x = lu_solve(data["fac"], b).ravel()
            u[data["wb"][ii]] = x
        if it % 10 == 0:
            r = res_global()
            print(f"{name} it={it} res={r:.2e} t={time.time()-t0:.1f}s",
                  flush=True)
            if r < 1e-12:
                break
    print(f"{name}: converged it={it} t={time.time()-t0:.1f}s")
    du = float(np.abs(u - u_ref).max() / np.abs(u_ref).max())
    print(f"{name}: du vs LU = {du:.2e}")
