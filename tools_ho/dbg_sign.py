# -*- coding: utf-8 -*-
"""dbg_sign.py — DEFINITIVE sign check for the neighbour RHS terms.
For several nodes and their internal faces, compare the exact global
matrix's non-local part against my two terms separately:
    b_true = G[gr, local] @ x
    b_mine = Src@q + C0*sgn@u + sigma*src5@q_nb
Decompose G[gr, nonlocal] @ u into (a) shared-face col and (b) other
cols, and the source rows' non-local q part, to pin every sign."""
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

# pick a few nodes with internal faces on both sides
test_nodes = []
for n in range(psn.nodes):
    ifps = [lf for lf in range(4) if nl.nf[n][lf][0] == 'i']
    if ifps:
        test_nodes.append((n, ifps[0]))
    if len(test_nodes) >= 6:
        break

for n, lf in test_nodes:
    info = nl.nf[n][lf]
    sigma = info[3]
    nb = info[1]
    fb = hs._FB[lf]
    f = psn.node_fmap[n][lf][0]
    key0 = nl.mc_of[n]
    data = nl._microclass(key0)
    ii = list(data["nodes"]).index(n)
    wb = data["wb"]
    x = u_ref[wb[ii]]
    bdata = nl._mc_b(data, qn)
    m, a = 0, 0
    lr = int(nl.blockoff[n, lf]) * K + m * K + a
    gr = rows[("i", f, m, a)]
    btrue = float(np.asarray(G[gr, wb[ii]].toarray().ravel() @ x))
    print(f"\nnode={n} lf={lf} sigma={sigma:+.0f} nb={nb} fb={fb} "
          f"lr={lr} gr={gr}")
    print(f"  b_true(G@x_local) = {btrue:+.8f}")
    print(f"  b_mine            = {bdata[ii, lr]:+.8f}  "
          f"d={bdata[ii, lr]-btrue:+.2e}")
    # decompose: global row = G_local@x + G_nonlocal@u ; source row =
    # Sm_local@q_n + Sm_nonlocal@q_nb
    nlnnz = G[gr].tocoo()
    shared_col = int(nl.colg[nb, fb, m]) * K + a
    gnl_u = 0.0
    gnl_u_shared = 0.0
    for cidx, cv in zip(nlnnz.col, nlnnz.data):
        c = int(cidx)
        if c in set(int(cc) for cc in wb[ii]):
            continue
        gnl_u += cv * u_ref[c]
        if c == shared_col:
            gnl_u_shared += cv * u_ref[c]
    smnz = Sm[gr].tocoo()
    gnl_q = 0.0
    for cidx, cv in zip(smnz.col, smnz.data):
        c = int(cidx)
        if c != nb * 5:          # not the neighbour's source block
            gnl_q += cv * qflat[c]
    # my two terms
    part = data["nb"][lf][0]
    s2 = list(part["nodes_nb"]).index(nb)
    un = nl.u[part["gath"][s2]].reshape(4, M, K).transpose(1, 0, 2)
    prod = part["sgn_nb"][s2] * un
    contrib = float(np.einsum("fs,fs->", part["C0"][m, a], prod[m]))
    src = float(np.einsum("q,maq->", qn[nb], part["src5"][m, a]))
    # A-matrix shared-face diagonal entry (what the local solve assumes)
    A_shared = float(data["A"][lr, int(nl.blockoff[n, lf]) * K + m * K + a])
    print(f"  G_nonlocal@u (all)  = {gnl_u:+.8f}  (shared col part {gnl_u_shared:+.8f})")
    print(f"  my u-term           = {contrib:+.8f}   (excl. shared)")
    print(f"  G_src_nonlocal@q    = {gnl_q:+.8f}")
    print(f"  my q-term           = {part['sigma'] * src:+.8f}")
    print(f"  A[shared col]       = {A_shared:+.8f}   "
          f"(global: G[gr,shared]={float(G[gr, shared_col]):+.8f})")
