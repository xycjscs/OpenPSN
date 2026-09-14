# -*- coding: utf-8 -*-
"""dbg_gal3.py — corrected Galerkin-coarse V-cycle.

R: internal + vacuum fine rows -> coarse rows (weight 1/c).
P: internal crossing + vacuum fine cols <- coarse cols (weight 1).
Reflex rows/cols EXCLUDED on both sides (the previous singularity was
reflect rows without reflect columns -> zero rows in A_c).
A_c = R A_f P is block-diagonal [internal-block, boundary-diagonal]
and consistent by construction.

Decisive test: extract the actual slow GS mode e, form its residual
r = A_f e, apply ONE correction corr = P A_c^{-1} R r, and measure
||e - corr|| / ||e||.  Then full V-cycle runs vs plain.
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
ft = {}
for f, face in enumerate(psn.faces):
    ft[face["type"]] = ft.get(face["type"], 0) + 1
ct = {}
for f, face in enumerate(coarse.faces):
    ct[face["type"]] = ct.get(face["type"], 0) + 1
print("fine faces:", ft, " coarse faces:", ct, flush=True)


def face_rows(p, M_, K_):
    r = 0
    rows = {}
    for f, face in enumerate(p.faces):
        if face["type"] == "internal":
            for m in range(M_):
                for a in range(K_):
                    rows[("i", f, m, a)] = r
                    r += 1
        elif face["type"] == "vacuum":
            for m in range(M_):
                for a in range(K_):
                    rows[("v", f, m, a)] = r
                    r += 1
        else:
            for pi in range(len(p.face_pairs[f])):
                for a in range(K_):
                    rows[("r", f, pi, a)] = r
                    r += 1
    return rows


frow = face_rows(psn, M, K)
crown = face_rows(coarse, M, K)

# coarse row index restricted to internal+vacuum rows (in crown order)
cr = 0
crow = {}
for f, face in enumerate(coarse.faces):
    if face["type"] == "internal":
        for m in range(M):
            for a in range(K):
                crow[("i", f, m, a)] = cr
                cr += 1
    elif face["type"] == "vacuum":
        for m in range(M):
            for a in range(K):
                crow[("v", f, m, a)] = cr
                cr += 1
nrow_c = cr
print(f"coarse rows (i+v): {nrow_c} of {A_f.shape[0]} fine rows", flush=True)

cpair = {}
for cf, cface in enumerate(coarse.faces):
    if cface["type"] == "internal":
        a, bb = int(cface["a"]), int(cface["b"])
        cpair[(a, bb)] = cf
        cpair[(bb, a)] = cf
cbnd = {}
for cn in range(coarse.nodes):
    for lf in range(4):
        f = coarse.node_fmap[cn][lf][0]
        if coarse.faces[f]["type"] == "vacuum":
            cbnd[(cn, lf)] = f

face2c = {}
for f, face in enumerate(psn.faces):
    t = face["type"]
    if t == "internal":
        ca = int(node_map[int(face["a"])])
        cb = int(node_map[int(face["b"])])
        if ca != cb:
            face2c[f] = cpair[(ca, cb)]
    elif t == "vacuum":
        n = int(face["a"])
        lf = psn._local_face(n, f)
        face2c[f] = cbnd[(int(node_map[n]), lf)]
print(f"mapped fine faces: {len(face2c)}", flush=True)

rr, rc, rw = [], [], []
for f, cf in face2c.items():
    t = psn.faces[f]["type"]
    key_c = ("i" if t == "internal" else "v", cf)
    if t == "internal":
        for m in range(M):
            for a in range(K):
                rr.append(crow[("i", cf, m, a)])
                rc.append(frow[("i", f, m, a)])
                rw.append(1.0 / c)
    else:
        for m in range(M):
            for a in range(K):
                rr.append(crow[("v", cf, m, a)])
                rc.append(frow[("v", f, m, a)])
                rw.append(1.0 / c)
Rmat = coo_matrix((rw, (rr, rc)), shape=(nrow_c, A_f.shape[0])).tocsr()

pc, pr, pw = [], [], []
for f, cf in face2c.items():
    t = psn.faces[f]["type"]
    for m in range(M):
        col = int(psn.colidx[f, m])
        if col < 0:
            continue
        for a in range(K):
            pc.append(col * K + a)
            pr.append(crow[("i" if t == "internal" else "v", cf, m, a)])
            pw.append(1.0)
Pmat = coo_matrix((pw, (pc, pr)), shape=(A_f.shape[1], nrow_c)).tocsr()
print(f"R {Rmat.shape} nnz={Rmat.nnz}   P {Pmat.shape} nnz={Pmat.nnz}")
A_c = (Rmat @ A_f @ Pmat).tocsc()
t0 = time.time()
fac_c = splu(A_c, permc_spec="COLAMD")
print(f"A_c {A_c.shape} nnz={A_c.nnz} splu {time.time()-t0:.2f}s")
x0 = np.random.default_rng(0).normal(size=nrow_c)
d0 = fac_c.solve(x0)
print(f"A_c probe residual = "
      f"{np.abs(A_c @ d0 - x0).max()/np.abs(x0).max():.2e}", flush=True)


def correct(v, omega=1.0):
    return omega * (Pmat @ fac_c.solve(Rmat @ v))


# ---------------- extract the slow mode ----------------
nl = hs.HoNodeLocal(psn, i, g, 3, 'directional', gs_tol=1e-14,
                    max_gs=4000)
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
print(f"\nslow mode extracted ({time.time()-t0:.0f}s)", flush=True)

r = A_f @ e
corr = correct(r, 1.0)
red = np.linalg.norm(e - corr) / np.linalg.norm(e)
print(f"one correction on the slow mode: "
      f"||e - corr||/||e|| = {red:.4f}  (want << 1)", flush=True)
# where is the mode's energy: crossing vs in-block internal cols?
ne_c = ne_b = 0.0
for f, face in enumerate(psn.faces):
    if face["type"] != "internal":
        continue
    ca = int(node_map[int(face["a"])])
    cb = int(node_map[int(face["b"])])
    for m in range(M):
        col = int(psn.colidx[f, m])
        if col < 0:
            continue
        s = float(np.sum(e[col * K: col * K + K] ** 2))
        if ca != cb:
            ne_c += s
        else:
            ne_b += s
print(f"mode energy: crossing cols {ne_c:.3e}  in-block cols {ne_b:.3e} "
      f"  (ratio {ne_b/max(ne_c,1e-300):.3f})", flush=True)

# ---------------- V-cycle runs ----------------
def run(use_cmfd, omega, label, mg=2000, every=2):
    u = np.zeros(A_f.shape[1])
    it = 0
    res = None
    t0 = time.time()
    while it < mg:
        it += 1
        direction = (it - 1) % 2
        for pty in (0, 1):
            for key in orders[pty][direction]:
                nl._mc_pass(data_cache[key], q0)
        if use_cmfd and it % every == 0:
            r = b - A_f @ u
            u += correct(r, omega)
            nl.u = u
        if it % 25 == 0:
            res = float(np.abs(A_f @ u - b).max() / np.abs(b).max())
            if it <= 75 or it % 100 == 0:
                print(f"  [{label}] it={it:4d} res={res:.2e}", flush=True)
            if res < 1e-12:
                break
    du = float(np.abs(u - u_ref).max() / np.abs(u_ref).max())
    print(f"[{label}] done it={it} res={res:.2e} du={du:.2e} "
          f"t={time.time()-t0:.1f}s\n", flush=True)
    return it


run(False, 0.0, "plain")
for omega in (0.5, 1.0):
    run(True, omega, f"gal3 w={omega}")
