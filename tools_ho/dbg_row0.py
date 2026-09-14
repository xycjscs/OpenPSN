# -*- coding: utf-8 -*-
"""dbg_row0.py — for global row (face f, m=0, a=0) decompose the
neighbour u-part: compare (a) the exact G[gr,nonlocal]@u, (b) my
_mc_b contrib, (c) a manual sum, face by face, to locate the bug."""
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
G, Sm = B.matrix, B.source_map

# pick node 0, its internal face lf=0 (sigma=+1 'a' side), mode m=0 slot a=0
n = 0
lf = 0
info = nl.nf[n][lf]
assert info[0] == 'i', info
sigma = info[3]
fb = hs._FB[lf]
nb = info[1]            # neighbour node
f = psn.node_fmap[n][lf][0]
m = 0
a = 0
# global row
rows = {}
r = 0
for ff, face in enumerate(psn.faces):
    if face["type"] == "internal":
        for mm in range(M):
            for aa in range(K):
                rows[(ff, mm, aa)] = r
                r += 1
    elif face["type"] == "vacuum":
        for mm in range(M):
            for aa in range(K):
                r += K
    else:
        for pos in range(len(psn.face_pairs[ff])):
            for aa in range(K):
                r += K
gr = None
r = 0
for ff, face in enumerate(psn.faces):
    if face["type"] == "internal":
        if ff == f:
            gr = r + m * K + a
            break
        r += M * K
    else:
        r += 0
print(f"n={n} lf={lf} nb={nb} fb={fb} f={f} sigma={sigma} gr={gr}")

# (a) exact nonlocal u-part: all global cols with G[gr,col]!=0 that are NOT
#     in node n's local writeback
key0 = nl.mc_of[n]
data = nl._microclass(key0)
wb = data["wb"]
localcols = set(int(c) for c in wb[0])
nnz = G[gr].tocoo()
nonlocal_u = 0.0
print("\ncol          G[gr,col]      u[col]        colowner(n,lf,m,slot)")
for cidx, cv in zip(nnz.col, nnz.data):
    c = int(cidx)
    if c in localcols:
        continue
    # decode owner: find node/lf/m whose colg gives c
    owner = None
    for nn in range(psn.nodes):
        for l2 in range(4):
            mm_ = nl.colg[nn, l2]
            for slot in range(K):
                pass
    nonlocal_u += cv * u_ref[c]
    if abs(cv) > 1e-12:
        # find which node/face this col belongs to
        dec = None
        for nn in range(psn.nodes):
            for l2 in range(4):
                for mm in range(M):
                    if int(nl.colg[nn, l2, mm]) * K + (a) == c:
                        dec = (nn, l2, mm)
        print(f"  col={c:6d} G={cv:+.6f} u={u_ref[c]:+.6f} owner={dec}")
print(f"\n(a) exact nonlocal u-part for row {gr}: {nonlocal_u:+.8f}")

# (b) my _mc_b contrib (u part only) for this row
u_part_mine = None
for l2, parts in data["nb"].items():
    if l2 != lf:
        continue
    for part in parts:
        if n not in data["nodes"]:
            continue
        ii = list(data["nodes"]).index(n)
        if ii not in part["idxs"]:
            continue
        un = nl.u[part["gath"]].reshape(len(part["idxs"]), 4, M, K)
        un = un.transpose(0, 2, 1, 3)
        prod = part["sgn_nb"] * un
        contrib = np.einsum("mafs,nmfs->nma", part["C0"], prod)
        s_i = list(part["idxs"]).index(n)
        u_part_mine = contrib[s_i, m, a]
print(f"(b) my _mc_b u-part for row {gr}: {u_part_mine:+.8f}")

# (c) manual: sum over nb's non-shared faces lf2!=fb, all slots
Lnb = nl.L[nl.node_grp_idx[nb]]
manual = 0.0
per_face = {}
for lf2 in range(4):
    if lf2 == fb:
        continue
    acc = 0.0
    for slot in range(K):
        c = int(nl.colg[nb, lf2, m]) * K + slot
        sg = nl.sg[nb, m, lf2, slot]
        Lval = Lnb[m, fb, a, lf2 * K + slot]
        acc += Lval * sg * u_ref[c]
    per_face[lf2] = acc
    manual += acc
print(f"(c) manual sum Lnb*sgn_nb@u_nb[non-shared]: {manual:+.8f}")
print("    per-face:", {k: round(v, 6) for k, v in per_face.items()})
print(f"    fb (shared, excluded) = {fb}")

# (d) what if we DIDN'T zero the shared face?
manual_all = manual
for slot in range(K):
    c = int(nl.colg[nb, fb, m]) * K + slot
    manual_all += Lnb[m, fb, a, fb * K + slot] * nl.sg[nb, m, fb, slot] * u_ref[c]
print(f"(d) manual INCLUDING shared face: {manual_all:+.8f}")
