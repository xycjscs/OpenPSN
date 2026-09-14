# -*- coding: utf-8 -*-
"""dbg_final.py — full comparison for node 0's two internal faces,
printing every component.  Single ground truth: b_true = G[gr, local]@x."""
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
G, Sm = B.matrix, B.source_map

rng = np.random.default_rng(7)
qn = rng.normal(size=(psn.nodes, 5))
b = Sm @ qn.ravel()
u_ref = B.factor.solve(b)
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

n = 0
key0 = nl.mc_of[n]
data = nl._microclass(key0)
ii = list(data["nodes"]).index(n)
wb = data["wb"]
x = u_ref[wb[ii]]
bdata = nl._mc_b(data, qn)
qflat = qn.ravel()

off = 0
print(f"node {n}, sigma per lf: "
      f"{{{', '.join(f'{lf}: {nl.nf[n][lf][3]}' for lf in range(4) if nl.nf[n][lf][0]=='i')}}}")
for lfl in range(4):
    t = nl.nf[n][lfl][0]
    nblk = len(nl.nf[n][lfl][1]) if t == "r" else M
    f = psn.node_fmap[n][lfl][0]
    for mb in range(nblk):
        for a in range(K):
            lr = off + mb * K + a
            keyt = {"i": "i", "v": "v", "r": "r"}[t]
            gr = rows[(keyt, f, mb, a)]
            btrue = float(np.asarray(G[gr, wb[ii]].toarray().ravel() @ x))
            qsrc = float(np.asarray(Sm[gr] @ qflat).ravel()[0])
            d = bdata[ii, lr] - btrue
            flag = " <<<" if abs(d) > 1e-9 else ""
            print(f"lf={lfl} {t} mb={mb} a={a} lr={lr}: bdata={bdata[ii, lr]:+12.6f} "
                  f"btrue={btrue:+12.6f} qsrc={qsrc:+12.6f} d={d:+10.3e}{flag}")
    off += nblk * K
