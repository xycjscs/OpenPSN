# -*- coding: utf-8 -*-
"""dbg_rhs3.py — verify _mc_b at the exact global solution.

A has been proven equal to G[gr, localcols] (dbg_node_row: 0 mismatches),
so at the exact solution u_ref the local system requires
    b_true[n, lr] = G[gr, localcols] @ x_n
Compare that against nl._mc_b.  Row/col mapping via blockoff (handles
reflect-face nodes correctly)."""
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
K, M = B.K, psn.M
nl = hs.HoNodeLocal(psn, i, g, 3, 'directional', verbose=False)

rng = np.random.default_rng(7)
qn = rng.normal(size=(psn.nodes, 5))
b = B.source_map @ qn.ravel()
u_ref = B.factor.solve(b)
G = B.matrix
nl.u = u_ref.copy()

rows = {}
r = 0
for f, face in enumerate(psn.faces):
    if face["type"] == "internal":
        for m in range(M):
            for a in range(K):
                rows[("i", f, m, a)] = r
                r += 1
    elif face["type"] == "vacuum":
        for m in range(M):
            for a in range(K):
                rows[("v", f, m, a)] = r
                r += 1
    else:
        for pos in range(len(psn.face_pairs[f])):
            for a in range(K):
                rows[("r", f, pos, a)] = r
                r += 1
assert r == G.shape[0]

worst = 0.0
nbad = 0
for key, nodes in nl.mc_nodes.items():
    data = nl._microclass(key)
    nlist = data["nodes"]
    d = data["d"]
    wb = data["wb"]
    x = u_ref[wb]
    bdata = nl._mc_b(data, qn)
    for ii, n in enumerate(nlist):
        off = 0
        for lfl in range(4):
            t = nl.nf[n][lfl][0]
            nblk = len(nl.nf[n][lfl][1]) if t == "r" else M
            f = psn.node_fmap[n][lfl][0]
            for mb in range(nblk):
                for a in range(K):
                    lr = off + mb * K + a
                    if t == "i":
                        gr = rows[("i", f, mb, a)]
                    elif t == "v":
                        gr = rows[("v", f, mb, a)]
                    else:
                        gr = rows[("r", f, mb, a)]
                    btrue = float(
                        np.asarray(G[gr, wb[ii]].toarray().ravel()
                                   @ x[ii]))
                    db = abs(bdata[ii, lr] - btrue)
                    worst = max(worst, db)
                    if db > 1e-8:
                        nbad += 1
                        if nbad <= 12:
                            print(f"n={n} key{key[:2]} lr={lr} lf={lfl} "
                                  f"t={t} mb={mb} a={a}: "
                                  f"bmine={bdata[ii, lr]:.6e} "
                                  f"btrue={btrue:.6e}")
        off += nblk * K
print(f"nmc={len(nl.mc_nodes)} bad rows: {nbad}  worst |b_mine-b_true| = {worst:.3e}")
