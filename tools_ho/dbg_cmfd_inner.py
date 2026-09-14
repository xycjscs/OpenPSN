# -*- coding: utf-8 -*-
"""dbg_cmfd_inner.py — PROTOTYPE of the inner-iteration CMFD
(coarse-grid residual correction, algebraic-multigrid consistent form):

  each pass: red-black block-GS (2 half-passes)
  every 2nd pass:
      r_f = A_f u - b_f          (fine global residual, (i,g))
      r_c[n] = sum_{f faces of n} (area ratio) r_f[eq(f)]
      d_c = A_c^{-1} r_c         (coarse A3 solve, small)
      u_f[eq(f)] += omega * d_c[coarse face of f]
      q_f[n] += omega_cq * d_c_q[n]

Fixed point: u exact => r_f = 0 => d = 0 (identity).  Measures
sweep counts vs the plain GS baseline on the slow polar i2.
"""
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix

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
Sm = B.source_map
rng = np.random.default_rng(42)
q0 = rng.normal(size=(psn.nodes, 5))
b_ref = Sm @ q0.ravel()
u_ref = B.factor.solve(b_ref)

# ---------------- coarse mesh + A3 operator ----------------
c = 3
coarse, node_map, areaw = hs.build_coarse_psn(psn, c=c)
Cc = ho.BlockAngleSystem(coarse, i, g, 3, 'directional')
A_c = Cc.matrix.tocsr()
print(f"coarse nodes={coarse.nodes} ncol_c={coarse.ncol} "
      f"A_c={A_c.shape} nnz={A_c.nnz}")
t0 = time.time()
Cc.factor.solve(np.zeros(A_c.shape[1]))
print(f"coarse first solve {time.time()-t0:.2f}s")

# coarse row order (assembly): same structure as fine
def face_rows(p):
    rows = []   # (kind, f, m, a) -> global row index
    r = 0
    for f, face in enumerate(p.faces):
        if face["type"] == "internal":
            for m in range(M):
                for a in range(K):
                    rows.append(("i", f, m, a))
                    r += 1
        elif face["type"] == "vacuum":
            for m in range(M):
                for a in range(K):
                    rows.append(("v", f, m, a))
                    r += 1
        else:
            for pos in range(len(p.face_pairs[f])):
                for a in range(K):
                    rows.append(("r", f, pos, a))
                    r += 1
    return rows


fine_rows = face_rows(psn)
coarse_rows = face_rows(coarse)
cref = {key: k for k, key in enumerate(coarse_rows)}

# fine face f (internal) -> (coarse face cf, coarse material group)
cpair = {}
for cf, cface in enumerate(coarse.faces):
    if cface["type"] == "internal":
        a, b = int(cface["a"]), int(cface["b"])
        cpair[(a, b)] = cf
        cpair[(b, a)] = cf
f2c = {}
cgrp_of_face = {}
for cf, cface in enumerate(coarse.faces):
    if cface["type"] == "internal":
        n = int(cface["a"])
        cgrp_of_face[cf] = coarse._groups.index(
            (coarse.mat_of_node[n], coarse.shape_of_node[n]))
for f, face in enumerate(psn.faces):
    if face["type"] != "internal":
        continue
    na, nb = int(face["a"]), int(face["b"])
    ca, cb = int(node_map[na]), int(node_map[nb])
    if ca == cb:
        continue
    cf = cpair[(ca, cb)]
    f2c[f] = cf

# area of fine node / coarse node (sum of fine areas)
A_fine = psn.area.ravel()
A_cnode = np.zeros(coarse.nodes)
np.add.at(A_cnode, node_map, A_fine)

# face LENGTHS (the L-row is the AVERAGE current, normalised by the
# face length): fine face f length and the coarse face length it
# belongs to.  Restriction weight = L_f / L_c (= 1/c for uniform);
# injection (prolongation) = copy (weight 1) — the standard
# full-weighting consistent pair.
hx, hy = psn.hx, psn.hy
fine_flen = {}
for f, face in enumerate(psn.faces):
    if face["type"] != "internal":
        continue
    na, nb = int(face["a"]), int(face["b"])
    # face between (na, nb): perpendicular to x if i differs
    if psn.i_idx[na] != psn.i_idx[nb]:      # x-face, length = hy
        fine_flen[f] = float(hy[psn.j_idx[na], psn.i_idx[na]])
    else:                                    # y-face, length = hx
        fine_flen[f] = float(hx[psn.j_idx[na], psn.i_idx[na]])
coarse_flen = {}
for cf, cface in enumerate(coarse.faces):
    if cface["type"] != "internal":
        continue
    na, nb = int(cface["a"]), int(cface["b"])
    if coarse.i_idx[na] != coarse.i_idx[nb]:
        coarse_flen[cf] = float(coarse.hy[coarse.j_idx[na],
                                           coarse.i_idx[na]])
    else:
        coarse_flen[cf] = float(coarse.hx[coarse.j_idx[na],
                                          coarse.i_idx[na]])

# node source: coarse q_c[n] = sum areaw * q_f  (areaw = area of fine node)
# (areaw from build_coarse_psn is the fine node area, see /a_tot removed)
# fine source equation index: node n block n*5..n*5+5; coarse: nc*5

# ---- residual of the fine system, restricted to coarse ----
def restrict_residual(u_flat, q):
    r_f = A_f @ u_flat - b_ref
    # r_c: coarse face equations = sum over fine faces crossing it,
    #      weighted by area ratio A_fine(n)/A_cnode(n_c)
    r_c = np.zeros(A_c.shape[0])
    for f, cf in f2c.items():
        face = psn.faces[f]
        na = int(face["a"])
        a_ratio = A_fine[na] / A_cnode[int(node_map[na])]
        for m in range(M):
            for a in range(K):
                fr = frow_idx.get((f, m, a))
                if fr is None:
                    continue
                cr = cref[("i", cf, m, a)]
                r_c[cr] += a_ratio * r_f[fr]
    return r_c



crows_idx = cref  # (kind, f, m, a) -> row
frow_idx = {}
for k, key in enumerate(fine_rows):
    if key[0] == "i":
        frow_idx[(key[1], key[2], key[3])] = k

t0 = time.time()
r_c = restrict_residual(b_ref, q0)
print(f"restrict_residual build+run {time.time()-t0:.2f}s")
d_c = Cc.factor.solve(r_c)
print(f"coarse correction norm = {np.abs(d_c).max():.3e}")

# inject: u_f[fine row (f,m,a) col?] -> the correction lands on the
# FINE FACE EQUATION's unknown: for face f (nodes na, nb), mode m slot a:
# global column = colidx[f, m] * K + a  (same for both nodes)
# u_f[col] += omega * d_c[cr], cr = cref[("i", cf, m, a)]
# Build the injection once: for each fine face f, m, a:
inj_rows = []
inj_cols = []
inj_coef = []
for f, cf in f2c.items():
    face = psn.faces[f]
    na = int(face["a"])
    a_ratio = A_fine[na] / A_cnode[int(node_map[na])]
    for m in range(M):
        for a in range(K):
            col = psn.colidx[f, m] * K + a
            cr = crows_idx[("i", cf, m, a)]
            inj_rows.append(cr)
            inj_cols.append(col)
            inj_coef.append(1.0)
inj = coo_matrix((inj_coef, (inj_cols, inj_rows)),
                 shape=(psn.ncol * K, A_c.shape[0])).tocsr()
d_f = inj @ d_c
print(f"fine correction max = {np.abs(d_f).max():.3e} (coarse {np.abs(d_c).max():.3e})")

# ---------------- run GS with/without the correction ----------------
nl = hs.HoNodeLocal(psn, i, g, 3, 'directional', gs_tol=1e-12, max_gs=400)
nl.u = np.zeros(psn.ncol * K)
data_cache = {k: nl._microclass(k) for k in nl.mc_nodes}
orders = nl._mc_order()


def run(use_cmfd, omega=0.6, label=""):
    u = np.zeros(psn.ncol * K)
    nl.u = u
    it = 0
    res = None
    t0 = time.time()
    while it < 400:
        it += 1
        direction = it % 2
        for pty in (0, 1):
            for key in orders[pty][direction]:
                nl._mc_pass(data_cache[key], q0)
        if use_cmfd and it >= 4 and it % 2 == 0:
            r_c = restrict_residual(u, q0)
            d_c = Cc.factor.solve(r_c)
            u -= omega * (inj @ d_c)
        if it % 4 == 0 or it >= 399:
            r_f = A_f @ u - b_ref
            res = float(np.abs(r_f).max() / max(np.abs(b_ref).max(), 1e-300))
            print(f"  [{label}] it={it:4d} res={res:.2e} "
                  f"t={time.time()-t0:.1f}s", flush=True)
            if res < 1e-12:
                break
    du = float(np.abs(u - u_ref).max() / np.abs(u_ref).max())
    print(f"[{label}] done it={it} res={res:.2e} du={du:.2e} "
          f"t={time.time()-t0:.1f}s", flush=True)
    return it, res


run(False, label="plain")
print()
run(True, label="cmfd-omega=0.6")
