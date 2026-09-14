# -*- coding: utf-8 -*-
"""dbg_slowmode.py — DECISIVE: measure single-step contraction of the
slowest mode under (a) plain GS pass, (b) GS pass + CMFD correction.

The slowest GS mode direction is extracted by iterating from zero for
2000 passes and taking the last error (dominated by the slowest
eigenmode).  Then we measure:
    e_next_plain = T_gs e           (one full red-black row/col cycle)
    e_next_cmfd  = (T_gs + omega*P A_c^-1 R A_f) e   (one cycle + CMFD)
and the L2 contraction  ||e_next||/||e||  for each.
If CMFD contraction << plain, it works.  If equal, it's invisible.
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
ncol = A_f.shape[1]

# ---------- extract slowest mode direction ----------
nl = hs.HoNodeLocal(psn, i, g, 3, 'directional', gs_tol=1e-14, max_gs=4000)
nl.u = np.zeros(ncol)
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
e = None
prev = nl.u.copy()
for k in range(2000):
    one_pass(nl.u)
    if k >= 1996:
        e = nl.u - prev
    prev = nl.u.copy()
e = e / np.linalg.norm(e)
print(f"slowest mode extracted in {time.time()-t0:.0f}s", flush=True)

# ---------- build Galerkin coarse (full, weight 1/c on R) ----------
c = 3
coarse, node_map, areaw = hs.build_coarse_psn(psn, c=c)


def face_rows(p):
    r = 0
    rows = {}
    for f, face in enumerate(p.faces):
        if face["type"] == "internal":
            for m in range(M):
                for a in range(K):
                    rows[(f, m, a)] = r
                    r += 1
        elif face["type"] == "vacuum":
            r += M * K
        else:
            r += len(p.face_pairs[f]) * K
    return rows


cfaces = [f for f, face in enumerate(coarse.faces)
          if face["type"] == "internal"]
crow = {}
for idx, cf in enumerate(cfaces):
    for m in range(M):
        for a in range(K):
            crow[(cf, m, a)] = idx * M * K + m * K + a
nrow_c = len(cfaces) * M * K

frow = face_rows(psn)
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
    ca, cb = int(node_map[int(face["a"])]), int(node_map[int(face["b"])])
    if ca == cb:
        continue
    f2c[f] = cpair[(ca, cb)]

wR = 1.0 / c
rr, rc = [], []
for f, cf in f2c.items():
    for m in range(M):
        for a in range(K):
            rr.append(crow[(cf, m, a)])
            rc.append(frow[(f, m, a)])
Rmat = coo_matrix((np.full(len(rr), wR), (rr, rc)),
                  shape=(nrow_c, A_f.shape[0])).tocsr()
pc, pr = [], []
for f, cf in f2c.items():
    for m in range(M):
        for a in range(K):
            pr.append(crow[(cf, m, a)])
            pc.append(psn.colidx[f, m] * K + a)
Pmat = coo_matrix((np.ones(len(pc)), (pc, pr)),
                  shape=(A_f.shape[1], nrow_c)).tocsr()
A_c = (Rmat @ A_f @ Pmat).tocsc()
fac_c = splu(A_c, permc_spec="COLAMD")
print(f"A_c {A_c.shape} nnz={A_c.nnz}", flush=True)


def cmfd_corr(v, omega=1.0):
    return omega * (Pmat @ fac_c.solve(Rmat @ (A_f @ v)))


# ---------- contraction measurement ----------
n = 6
e_p = e.copy()
e_c = e.copy()
for k in range(n):
    e_p = one_pass(e_p)
    e_c = one_pass(e_c) + cmfd_corr(e_c, 1.0)
c_plain = np.linalg.norm(e_p) / np.linalg.norm(e)
c_cmfd = np.linalg.norm(e_c) / np.linalg.norm(e)
print(f"\nafter {n} cycles on the slowest mode:", flush=True)
print(f"  plain GS : ||e|| ratio = {c_plain:.6f}  "
      f"(per-cycle {c_plain**(1/n):.6f})", flush=True)
print(f"  GS+CMFD  : ||e|| ratio = {c_cmfd:.6f}  "
      f"(per-cycle {c_cmfd**(1/n):.6f})", flush=True)

# also: is the CMFD correction itself aligned with the slow mode?
d = cmfd_corr(e, 1.0)
align = float(np.abs(np.dot(e, d))) / (
    np.linalg.norm(e) * max(np.linalg.norm(d), 1e-300))
mag = np.linalg.norm(d) / np.linalg.norm(e)
print(f"  CMFD corr |d|/|e| = {mag:.4e}   |e·d|/(|e||d|) = {align:.4f}",
      flush=True)
