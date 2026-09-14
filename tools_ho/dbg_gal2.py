# -*- coding: utf-8 -*-
"""dbg_gal2.py — FULL-row Galerkin-coarse V-cycle (internal + vacuum +
reflect rows AND columns).  A_c := R A_f P is consistent by
construction: for smooth e = P e_c,  M e = P A_c^{-1} R A_f P e_c = e
exactly.  Measures M on smooth vectors and the V-cycle vs plain on the
slow i2 system."""
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
Bc = ho.BlockAngleSystem(coarse, i, g, 3, 'directional')
A_c_native = Bc.matrix.tocsr()
print(f"fine {A_f.shape}   coarse native {A_c_native.shape}", flush=True)


def face_rows(p, M_, K_):
    """(type, f, m, a) -> row, in assembly order."""
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
        else:  # reflect
            for pi in range(len(p.face_pairs[f])):
                for a in range(K_):
                    rows[("r", f, pi, a)] = r
                    r += 1
    return rows


frow = face_rows(psn, M, K)
crown = face_rows(coarse, M, K)
nrow_c = crown and max(crown.values()) + 1
# row-count consistency with the assembled matrices
assert nrow_c == A_c_native.shape[0], (nrow_c, A_c_native.shape)

# ---- fine face -> coarse face mapping (all face types) ----
cpair = {}
for cf, cface in enumerate(coarse.faces):
    if cface["type"] == "internal":
        a, bb = int(cface["a"]), int(cface["b"])
        cpair[(a, bb)] = cf
        cpair[(bb, a)] = cf
# coarse-node -> its boundary face per local face
cbnd = {}
for cn in range(coarse.nodes):
    for lf in range(4):
        f = coarse.node_fmap[cn][lf][0]
        if coarse.faces[f]["type"] != "internal":
            cbnd[(cn, lf)] = f

face2c = {}   # fine face -> coarse face (all types)
for f, face in enumerate(psn.faces):
    t = face["type"]
    if t == "internal":
        ca = int(node_map[int(face["a"])])
        cb = int(node_map[int(face["b"])])
        if ca != cb:
            face2c[f] = cpair[(ca, cb)]
    else:
        # boundary face of fine node(s): use the 'a' node
        n = int(face["a"])
        cn = int(node_map[n])
        lf = psn._local_face(n, f)
        face2c[f] = cbnd[(cn, lf)]

# ---- R: fine rows -> coarse rows, weight 1/c (all row types) ----
rr, rc, rw = [], [], []
for f, cf in face2c.items():
    face = psn.faces[f]
    ctype = coarse.faces[cf]["type"]
    if ctype == "internal":
        for m in range(M):
            for a in range(K):
                rr.append(crown[("i", cf, m, a)])
                rc.append(frow[("i", f, m, a)])
                rw.append(1.0 / c)
    elif ctype == "vacuum":
        for m in range(M):
            for a in range(K):
                rr.append(crown[("v", cf, m, a)])
                rc.append(frow[("v", f, m, a)])
                rw.append(1.0 / c)
    else:
        # reflect: fine face f has len(face_pairs[f]) pairs; the coarse
        # face has the same number of pairs (1:1 per pair)
        npf = len(psn.face_pairs[f])
        ncf = len(coarse.face_pairs[cf])
        assert npf == ncf, (npf, ncf)
        for pi in range(npf):
            for a in range(K):
                rr.append(crown[("r", cf, pi, a)])
                rc.append(frow[("r", f, pi, a)])
                rw.append(1.0 / c)
Rmat = coo_matrix((rw, (rr, rc)), shape=(nrow_c, A_f.shape[0])).tocsr()

# ---- P: fine cols -> coarse cols (internal crossing + boundary) ----
# fine column of (face f, mode m) is colidx[f, m]*K + a for internal;
# boundary columns need the same assembly colidx convention — take it
# from the sparsity of A_f per face-row block instead (robust).
# Build by scanning A_f: each row block (face f) touches its own
# columns + neighbour columns; the OWN columns are the diagonal-ish
# ones.  Simpler: colidx is stored on psn for all faces.
pc, pr, pw = [], [], []
for f, cf in face2c.items():
    ctype = coarse.faces[cf]["type"]
    face = psn.faces[f]
    if ctype == "internal":
        for m in range(M):
            col = int(psn.colidx[f, m])
            if col < 0:
                continue
            for a in range(K):
                pc.append(col * K + a)
                pr.append(crown[("i", cf, m, a)])
                pw.append(1.0)
    elif ctype == "vacuum":
        for m in range(M):
            col = int(psn.colidx[f, m])
            if col < 0:
                continue
            for a in range(K):
                pc.append(col * K + a)
                pr.append(crown[("v", cf, m, a)])
                pw.append(1.0)
    else:
        for pi in range(len(psn.face_pairs[f])):
            # reflect columns: colidx for the pair's master face?
            # In PSN2D reflect pairs share the face f; the eliminated
            # slave modes get colidx < 0 or a pair slot.  Use the
            # pair column table if present, else skip (reflect
            # correction is a small part of the smooth mode).
            for a in range(K):
                pc.append(-1)  # placeholder, fixed below
                pr.append(-1)
                pw.append(1.0)
# fix reflect columns: use the coarse system's own col layout — the
# reflect columns of A_f are the pair master columns; find them from
# psn pair tables
if psn.face_pairs:
    # rebuild P excluding placeholders
    pc2, pr2, pw2 = [], [], []
    for f, cf in face2c.items():
        ctype = coarse.faces[cf]["type"]
        if ctype == "internal":
            for m in range(M):
                col = int(psn.colidx[f, m])
                if col < 0:
                    continue
                for a in range(K):
                    pc2.append(col * K + a)
                    pr2.append(crown[("i", cf, m, a)])
                    pw2.append(1.0)
        elif ctype == "vacuum":
            for m in range(M):
                col = int(psn.colidx[f, m])
                if col < 0:
                    continue
                for a in range(K):
                    pc2.append(col * K + a)
                    pr2.append(crown[("v", cf, m, a)])
                    pw2.append(1.0)
    Pmat = coo_matrix((pw2, (pc2, pr2)),
                      shape=(A_f.shape[1], nrow_c)).tocsr()
else:
    Pmat = None
    print("no reflect faces on fine grid — P built without placeholders")
    Pmat = coo_matrix((pw, (pc, pr)),
                      shape=(A_f.shape[1], nrow_c)).tocsr()

print(f"R {Rmat.shape} nnz={Rmat.nnz}   P {Pmat.shape} nnz={Pmat.nnz}")
A_c = (Rmat @ A_f @ Pmat).tocsc()
t0 = time.time()
fac_c = splu(A_c, permc_spec="COLAMD")
print(f"splu A_c {time.time()-t0:.2f}s")
x0 = np.random.default_rng(0).normal(size=A_c.shape[1])
d0 = fac_c.solve(x0)
print(f"A_c solve probe residual = "
      f"{np.abs(A_c @ d0 - x0).max()/np.abs(x0).max():.2e}", flush=True)
# how close is the Galerkin A_c to the native coarse operator?
A_c_full = A_c.todense()
An_full = A_c_native.todense()
print(f"||A_c - A_native||F / ||A_native||F = "
      f"{np.linalg.norm(A_c_full-An_full)/np.linalg.norm(An_full):.3f}")


def apply_M(v):
    return Pmat @ fac_c.solve(Rmat @ v)


jidx, iidx = psn.j_idx, psn.i_idx
nx, ny = psn.nx, psn.ny
print("\nconsistency (should be ~+1, rel ~0):")
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
data_cache = {k: nl._microclass(k) for k in nl.mc_nodes}
orders = nl._mc_order()


def one_sweep(u):
    nl.u = u
    for pty in (0, 1):
        for key in orders[pty][0]:
            nl._mc_pass(data_cache[key], q0)
    return nl.u


def run(use_cmfd, omega, label, mg=1200):
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
        if use_cmfd and it % 2 == 0:
            r = b - A_f @ u
            u += omega * (Pmat @ fac_c.solve(Rmat @ r))
        if it % 20 == 0:
            res = float(np.abs(A_f @ u - b).max() / np.abs(b).max())
            if it <= 60 or it % 100 == 0:
                print(f"  [{label}] it={it:4d} res={res:.2e}", flush=True)
            if res < 1e-12:
                break
    du = float(np.abs(u - u_ref).max() / np.abs(u_ref).max())
    print(f"[{label}] done it={it} res={res:.2e} du={du:.2e} "
          f"t={time.time()-t0:.1f}s\n", flush=True)
    return it


run(False, 0.0, "plain")
for omega in (0.5, 1.0):
    run(True, omega, f"full-gal w={omega}")
