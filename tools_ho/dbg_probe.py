# -*- coding: utf-8 -*-
"""dbg_probe.py — decompose _mc_b into its pieces for one microclass."""
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
nl.u = u_ref.copy()

# the microclass containing node 0
key0 = nl.mc_of[0]
data = nl._microclass(key0)
nlist = data["nodes"]
ii0 = list(nlist).index(0)
print("key:", key0[:4])
print("nlist first 5:", nlist[:5], "ii0:", ii0, "Nn:", len(nlist))
print("d:", data["d"])
print("blockoff[rep]:", [int(v) for v in nl.blockoff[data["rep"]]])
print("mask entries rep:", [nl._mask_entry(data["rep"], lf) for lf in range(4)])
print("mask entries n=0:", [nl._mask_entry(0, lf) for lf in range(4)])

# ground-truth b from the global matrix (the only trustworthy definition)
G, Sm = B.matrix, B.source_map
wb = data["wb"]
x = u_ref[wb]
qflat = qn.ravel()
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
b_true = np.zeros(data["d"])
off = 0
for lfl in range(4):
    t = nl.nf[n][lfl][0]
    nblk = len(nl.nf[n][lfl][1]) if t == "r" else M
    f = psn.node_fmap[n][lfl][0]
    for mb in range(nblk):
        for a in range(K):
            lr = off + mb * K + a
            keyt = {"i": "i", "v": "v", "r": "r"}[t]
            gr = rows[(keyt, f, mb, a)]
            b_true[lr] = float(np.asarray(Sm[gr] @ qflat).ravel()[0]) \
                - np.asarray(G[gr, wb[ii0]].toarray().ravel() @ x[ii0])
    off += nblk * K

bdata = nl._mc_b(data, qn)
qsrc = qn[nlist] @ data["SRC"].T
print("\nrows   q@SRC.T     b_mine    b_true")
for lr in range(16):
    print(f"{lr:4d}  {qsrc[ii0, lr]:+12.6f}  {bdata[ii0, lr]:+12.6f}  "
          f"{b_true[lr]:+12.6f}")

# neighbour contributions per lf
print("\nneighbour parts (rowidx ranges):")
for lf, parts in data["nb"].items():
    for part in parts:
        print(f"  lf={lf} n_nb={len(part['idxs'])} rowidx={part['rowidx'][:3]}...")
        un = nl.u[part["gath"]].reshape(
            len(part["idxs"]), 4, M, K).transpose(0, 2, 1, 3)
        prod = part["sgn_nb"] * un
        contrib = np.einsum("mafs,nmfs->nma", part["C0"], prod)
        src = np.einsum("nq,maq->nma", qn[part["nodes_nb"]], part["src5"])
        print(f"    contrib[0][:4]={contrib[0][:4]}")
        print(f"    sigma*src[0][:4]={part['sigma'] * src[0][:4]}")
