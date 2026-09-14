# -*- coding: utf-8 -*-
"""dbg_rhs2.py — compare (a) _microclass A rows vs G restricted to the
node's local columns, and (b) _mc_b vs the exact required RHS."""
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
nl.u = None  # set per test below

rng = np.random.default_rng(7)
qn = rng.normal(size=(psn.nodes, 5))
b = B.source_map @ qn.ravel()
u_ref = B.factor.solve(b)
G = B.matrix
Sm = B.source_map
nl.u = u_ref.copy()

# rowg from data; verify against an independent row map
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

worst_A = 0.0
worst_b = 0.0
nbad = 0
for key, nodes in nl.mc_nodes.items():
    data = nl._microclass(key)
    nlist = data["nodes"]
    d = data["d"]
    A = data["A"]
    wb = data["wb"]
    x = u_ref[wb]  # (Nn, d)
    bdata = nl._mc_b(data, qn)
    # expected b from the global matrix exactly:
    #   b_exp[n, lr] = Sm[rowg, :] @ q_flat - G[rowg, localcols] @ x[n]
    qflat = qn.ravel()
    for ii, n in enumerate(nlist[:2]):
        localcols = wb[ii]  # d distinct global cols
        for lr in range(d):
            fl = lr // (M * K) if False else None
            # find which face this local row belongs to
            off = 0
            lfl = None
            for l2 in range(4):
                t2 = nl.nf[n][l2][0]
                nblk = len(nl.nf[n][l2][1]) if t2 == "r" else M
                if off + nblk > lr // K:
                    lfl = l2
                    break
                off += nblk
            f = psn.node_fmap[n][lfl][0]
            t = nl.nf[n][lfl][0]
            if t == "i":
                m = (lr // K) % M
                a = lr % K
                gr = rows[("i", f, m, a)]
            elif t == "v":
                m = (lr // K) % M
                a = lr % K
                gr = rows[("v", f, m, a)]
            else:
                pos = (lr // K) % (M // 2)
                a = lr % K
                gr = rows[("r", f, pos, a)]
            # A row check
            arow = A[lr]
            gvals = G[gr, localcols].toarray().ravel()
            da = np.max(np.abs(arow - gvals))
            worst_A = max(worst_A, da)
            # b check
            bexp = float(np.asarray(Sm[gr] @ qflat).ravel()[0]) - (
                G[gr, localcols].toarray().ravel() @ x[ii])
            db = abs(bdata[ii, lr] - bexp)
            worst_b = max(worst_b, db)
            if da > 1e-9 or db > 1e-8:
                nbad += 1
                if nbad <= 10:
                    print(f"n={n} key={key[:1]} lr={lr} face={f} t={t} "
                          f"gr={gr}: |dA|={da:.3e} |db|={db:.3e} "
                          f"bmine={bdata[ii, lr]:.4e} bexp={bexp:.4e}")
print(f"worst |A - G_local| = {worst_A:.3e}")
print(f"worst |b_mine - b_exp| = {worst_b:.3e}")
print("bad rows (first 2 nodes per mc):", nbad)
