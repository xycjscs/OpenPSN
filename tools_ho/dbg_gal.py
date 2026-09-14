# -*- coding: utf-8 -*-
"""dbg_gal.py — Galerkin-coarse V-cycle for the slow i2 system.
A_c := R A_f P (consistent by construction).  V-cycle:
    r = b - A_f u ;  u += omega * P A_c^{-1} R r   every 2 GS passes.
Measures sweep counts to 1e-12 vs plain GS, and the consistency
(e·Me)/(e·e) on smooth vectors (should be ~+1 now)."""
import sys
import time

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
b_ref = B.source_map @ q0.ravel()
u_ref = B.factor.solve(b_ref)

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


# restriction target rows: need the COARSE grid's internal-face row
# structure.  But the Galerkin A_c rows are indexed by the COARSE
# internal faces (cf) — same (cf, m, a) -> row as face_rows(coarse)
# would give IF the coarse grid's row order matches.  We do NOT use the
# native coarse A3 at all; the coarse "grid" here is purely the set of
# internal coarse faces.  Row index for a coarse internal face = its
# position in the sorted list of coarse internal faces (m, a).
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
    a, b = int(cface["a"]), int(cface["b"])
    cpair[(a, b)] = cf
    cpair[(b, a)] = cf
f2c = {}
for f, face in enumerate(psn.faces):
    if face["type"] != "internal":
        continue
    ca, cb = int(node_map[int(face["a"])]), int(node_map[int(face["b"])])
    if ca == cb:
        continue
    f2c[f] = cpair[(ca, cb)]

# R: (nrow_c, nrow_f) coarse row <- sum fine rows (w=1)
rr, rc = [], []
for f, cf in f2c.items():
    for m in range(M):
        for a in range(K):
            rr.append(crow[(cf, m, a)])
            rc.append(frow[(f, m, a)])
Rmat = coo_matrix((np.ones(len(rr)), (rr, rc)),
                  shape=(nrow_c, A_f.shape[0])).tocsr()
# P: (ncol_f, ncol_c) fine col <- coarse col (w=1)
pc, pr = [], []
ncol_c = nrow_c
for f, cf in f2c.items():
    for m in range(M):
        for a in range(K):
            pr.append(crow[(cf, m, a)])
            pc.append(psn.colidx[f, m] * K + a)
Pmat = coo_matrix((np.ones(len(pc)), (pc, pr)),
                  shape=(A_f.shape[1], ncol_c)).tocsr()

print(f"R {Rmat.shape} nnz={Rmat.nnz}   P {Pmat.shape} nnz={Pmat.nnz}")
t0 = time.time()
A_c = (Rmat @ A_f @ Pmat).tocsc()
print(f"A_c = R A_f P: {A_c.shape} nnz={A_c.nnz} build={time.time()-t0:.2f}s")
t0 = time.time()
fac_c = splu(A_c, permc_spec="COLAMD")
print(f"splu A_c {time.time()-t0:.2f}s")
# invertibility probe
x0 = np.random.default_rng(0).normal(size=ncol_c)
d0 = fac_c.solve(x0)
print(f"A_c solve probe residual = "
      f"{np.abs(A_c @ d0 - x0).max()/np.abs(x0).max():.2e}")

# consistency on smooth vectors
def apply_M(v):
    return Pmat @ fac_c.solve(Rmat @ v)


jidx, iidx = psn.j_idx, psn.i_idx
nx, ny = psn.nx, psn.ny
for name, fn in {
    "const": lambda j, i: 1.0,
    "lin_j": lambda j, i: j / ny,
    "lin_i": lambda j, i: i / nx,
    "cos1": lambda j, i: np.cos(np.pi * (j + .5) / ny)
                        * np.cos(np.pi * (i + .5) / nx),
}.items():
    e = np.zeros(A_f.shape[1])
    for f, face in enumerate(psn.faces):
        if face["type"] != "internal":
            continue
        n = int(face["a"])
        e[psn.colidx[f, 0] * K + 0] = fn(float(jidx[n]), float(iidx[n]))
    Me = apply_M(e)
    ray = float(e @ Me) / float(e @ e)
    rel = float(np.linalg.norm(Me - e)) / max(float(np.linalg.norm(e)), 1e-300)
    print(f"  {name:6s}: (e·Me)/(e·e)={ray:+.3f}  ||Me-e||/||e||={rel:.3f}")

# ---------------- V-cycle runs ----------------
nl = hs.HoNodeLocal(psn, i, g, 3, 'directional', gs_tol=1e-12, max_gs=3000)
nl.u = None
data_cache = {k: nl._microclass(k) for k in nl.mc_nodes}
orders = nl._mc_order()


def run(use_cmfd, omega, label):
    u = np.zeros(psn.ncol * K)
    nl.u = u
    it = 0
    mg = 3000
    res = None
    t0 = time.time()
    while it < mg:
        it += 1
        direction = (it - 1) % 2
        for pty in (0, 1):
            for key in orders[pty][direction]:
                nl._mc_pass(data_cache[key], q0)
        if use_cmfd and it % 2 == 0:
            r = b_ref - A_f @ u
            u += omega * (Pmat @ fac_c.solve(Rmat @ r))
        if it % 10 == 0:
            res = float(np.abs(A_f @ u - b_ref).max()
                        / np.abs(b_ref).max())
            if it <= 40 or it % 40 == 0:
                print(f"  [{label}] it={it:4d} res={res:.2e}", flush=True)
            if res < 1e-12:
                break
    du = float(np.abs(u - u_ref).max() / np.abs(u_ref).max())
    print(f"[{label}] done it={it} res={res:.2e} du={du:.2e} "
          f"t={time.time()-t0:.1f}s\n", flush=True)
    return it


run(False, 0.0, "plain")
for omega in (0.5, 1.0):
    run(True, omega, f"gal-vcycle w={omega}")
