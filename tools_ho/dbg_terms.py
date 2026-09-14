# -*- coding: utf-8 -*-
"""dbg_terms.py — term-by-term comparison for the neighbour u-part of
global row (face 0, m=0, a=0), node n=0 (sigma=+1)."""
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

n, lf, m, a = 0, 0, 0, 0
info = nl.nf[n][lf]
sigma = info[3]
fb = hs._FB[lf]
nb = info[1]
print(f"n={n} lf={lf} nb={nb} fb={fb} sigma={sigma}")

key0 = nl.mc_of[n]
data = nl._microclass(key0)
part = data["nb"][lf][0]
print("part idxs:", part["idxs"], "nodes_nb:", part["nodes_nb"])

# --- module arrays
C0 = part["C0"]            # (M, K, 4, K)  [m, a, f, s]
sgn_nb = part["sgn_nb"]    # (N, M, 4, K)
un = nl.u[part["gath"]].reshape(len(part["idxs"]), 4, M, K)
un = un.transpose(0, 2, 1, 3)   # (N, M, 4, K)
prod = sgn_nb * un
contrib = np.einsum("mafs,nmfs->nma", C0, prod)
print(f"\nmodule einsum contrib[0,{m},{a}] = {contrib[0, m, a]:+.8f}")

# --- expected C0 from raw L
Lnb = nl.L[nl.node_grp_idx[nb]]
C0_exp = -sigma * Lnb[:, fb, :, :4 * K].reshape(M, K, 4, K)
C0_exp[:, :, fb, :] = 0.0
print(f"C0 vs expected: max diff = {np.max(np.abs(C0 - C0_exp)):.3e}")

# --- manual 12-term loop
tot = 0.0
print(f"\n{'f':>2} {'s':>2}  {'C0':>14} {'sgn':>8} {'u':>14} {'term':>14}")
for f in range(4):
    if f == fb:
        continue
    for s in range(K):
        c = int(nl.colg[nb, f, m]) * K + s
        sg = nl.sg[nb, m, f, s]
        cv = C0[m, a, f, s]
        uv = nl.u[c]
        t = cv * sg * uv
        tot += t
        print(f"{f:2d} {s:2d}  {cv:+14.8f} {sg:+8.2f} {uv:+14.8f} {t:+14.8f}")
print(f"manual total = {tot:+.8f}")

# --- what does un[0,m,f,s] contain vs u[colg]?
print("\nun[0, m, f, s] vs u[colg[nb,f,m]*K+s]:")
ok = True
for f in range(4):
    for s in range(K):
        c = int(nl.colg[nb, f, m]) * K + s
        if un[0, m, f, s] != nl.u[c]:
            ok = False
            print(f"  MISMATCH f={f} s={s}: un={un[0, m, f, s]} u[c={c}]={nl.u[c]}")
print("  all match" if ok else "  see above")
print(f"\nsgn_nb[0,0,0:3] = {sgn_nb[0, 0, 0]}  sg[nb,0,0] = {nl.sg[nb, 0, 0]}")
print(f"un[0,0,0] = {un[0, 0, 0]}")
print(f"C0[0,0] (m=0,a=0, over f,s):\n{C0[0, 0]}")
print(f"C0_exp[0,0]:\n{C0_exp[0, 0]}")
