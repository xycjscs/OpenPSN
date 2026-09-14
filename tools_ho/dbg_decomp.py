# -*- coding: utf-8 -*-
"""dbg_decomp.py — decisive decomposition of the slow GS mode vs the
Galerkin coarse (face-current) space:

  1) identity check:  P A_c^{-1} R A_f P == P ?  (must hold exactly)
  2) slow-mode smoothness:  s = ||(I - proj_P) e|| / ||e||
  3) one-correction contraction on the mode:
       ||e - P A_c^{-1} R A_f e|| / ||e||   (should be ~s if consistent)
"""
import sys, time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

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
K, M = 4, psn.M
i, g = psn.I - 1, 0
B = ho.BlockAngleSystem(psn, i, g, 3, 'directional')
A_f = B.matrix.tocsr()
q0 = np.random.default_rng(42).normal(size=(psn.nodes, 5))
b = B.source_map @ q0.ravel()
u_ref = B.factor.solve(b)

c = 3
coarse, node_map, areaw = hs.build_coarse_psn(psn, c=c)


def face_rows(p, M_, K_):
    r = 0
    rows = {}
    for f, face in enumerate(p.faces):
        if face["type"] == "internal":
            for m in range(M_):
                for a in range(K_):
                    rows[(f, m, a)] = r
                    r += 1
        elif face["type"] == "vacuum":
            for m in range(M_):
                for a in range(K_):
                    rows[(f, m, a)] = r
                    r += 1
        else:
            for pi in range(len(p.face_pairs[f])):
                for a in range(K_):
                    rows[(f, pi, a)] = r
                    r += 1
    return rows


frow = face_rows(psn, M, K)
# which fine face types exist?
types = {}
for f, face in enumerate(psn.faces):
    types[face["type"]] = types.get(face["type"], 0) + 1
print("fine face types:", types)
print("coarse face types:",
      {t: sum(1 for x in coarse.faces if x["type"] == t)
       for t in set(x["type"] for x in coarse.faces)})

cfaces = [f for f, face in enumerate(coarse.faces)
          if face["type"] == "internal"]
crow = {}
for idx, cf in enumerate(cfaces):
    for m in range(M):
        for a in range(K):
            crow[(cf, m, a)] = idx * M * K + m * K + a
nrow_c = len(cfaces) * M * K

cpair = {}
for cf in cfaces:
    cface = coarse.faces[cf]
    a, bb = int(cface["a"]), int(cface["b"])
    cpair[(a, bb)] = cf
    cpair[(bb, a)] = cf
f2c = {}
for f, face in enumerate(psn.faces):
    if face["type"] != "internal":
        continue
    ca = int(node_map[int(face["a"])])
    cb = int(node_map[int(face["b"])])
    if ca != cb:
        f2c[f] = cpair[(ca, cb)]

rr, rc = [], []
for f, cf in f2c.items():
    for m in range(M):
        for a in range(K):
            rr.append(crow[(cf, m, a)])
            rc.append(frow[(f, m, a)])
Rmat = coo_matrix((np.ones(len(rr)), (rr, rc)),
                  shape=(nrow_c, A_f.shape[0])).tocsr()
pc, pr = [], []
ncols_c = nrow_c
for f, cf in f2c.items():
    for m in range(M):
        col = int(psn.colidx[f, m])
        if col < 0:
            continue
        for a in range(K):
            pc.append(col * K + a)
            pr.append(crow[(cf, m, a)])
Pmat = coo_matrix((np.ones(len(pc)), (pc, pr)),
                  shape=(A_f.shape[1], ncols_c)).tocsr()
print(f"R {Rmat.shape}  P {Pmat.shape}  (P cols used: {np.count_nonzero(Pmat)} / {Pmat.nnz})")
A_c = (Rmat @ A_f @ Pmat).tocsc()
fac_c = splu(A_c, permc_spec="COLAMD")

# ---------- 1) identity ----------
e_c_rand = np.random.default_rng(1).normal(size=ncols_c)
Pe = Pmat @ e_c_rand
M_Pe = Pmat @ fac_c.solve(Rmat @ (A_f @ Pe))
print(f"\n1) identity: ||P A_c^-1 R A_f P - P||_F = "
      f"{np.abs(M_Pe - Pe).max():.2e}  (||P e|| max {np.abs(Pe).max():.1e})")

# ---------- 2) extract the slow mode ----------
nl = hs.HoNodeLocal(psn, i, g, 3, 'directional', gs_tol=1e-14, max_gs=4000)
nl.u = np.zeros(A_f.shape[1])
data_cache = {k: nl._microclass(k) for k in nl.mc_nodes}
orders = nl._mc_order()


def one_pass(u):
    nl.u = u
    for pty in (0, 1):
        for direction in (0, 1):
            for key in orders[pty][direction]:
                nl._mc_pass(data_cache[key], q0)
    return nl.u


t0 = time.time()
prev = nl.u.copy()
e = None
for k in range(1500):
    one_pass(nl.u)
    if k >= 1496:
        e = nl.u - prev
    prev = nl.u.copy()
e = e / np.linalg.norm(e)
print(f"\n2) slow mode extracted ({time.time()-t0:.0f}s)")

# energy by column group
n_int_cross, n_int_inblk = 0, 0
for f, face in enumerate(psn.faces):
    if face["type"] != "internal":
        continue
    ca = int(node_map[int(face["a"])])
    cb = int(node_map[int(face["b"])])
    for m in range(M):
        col = int(psn.colidx[f, m])
        if col < 0:
            continue
        grp = "cross" if ca != cb else "inblock"
        n_int_cross += 1 if grp == "cross" else 0
        n_int_inblk += 0 if grp == "cross" else 1
print(f"   internal crossing cols ~{n_int_cross}, in-block cols ~{n_int_inblk}")

Pe_full = Pmat @ (Pmat.T @ e) / (Pmat.T @ Pmat).diagonal()[np.newaxis, :] if False else None
# orthogonal projection onto range(P)
PtP = (Pmat.T @ Pmat).todense()
PtP_inv_e = np.linalg.solve(PtP, Pmat.T @ e)
proj = np.asarray(Pmat @ PtP_inv_e).ravel()
s = np.linalg.norm(e - proj) / np.linalg.norm(e)
print(f"   mode smoothness: ||(I-proj_P) e||/||e|| = {s:.4f}")

# ---------- 3) correction on the mode ----------
corr = Pmat @ fac_c.solve(Rmat @ (A_f @ e))
red = np.linalg.norm(e - corr) / np.linalg.norm(e)
print(f"\n3) one V-correction on the mode: ||e - P A_c^-1 R A_f e||/||e|| = {red:.4f}")
print(f"   (if consistent Galerkin, expect ~ s = {s:.4f})")
