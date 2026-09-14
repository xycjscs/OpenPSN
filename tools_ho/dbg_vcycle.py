# -*- coding: utf-8 -*-
"""dbg_vcycle.py — measure the ASYMPTOTIC contraction factor of
plain GS vs GS+residual-CMFD on the slow i-system, using the L2
residual in the tail region (where the smooth slow mode dominates).
Reports the local 2-step contraction factor res[t+2]/res[t]."""
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
b_ref = B.source_map @ np.random.default_rng(42).normal(size=(psn.nodes, 5)).ravel()
q0 = np.random.default_rng(42).normal(size=(psn.nodes, 5))

# coarse
c = 3
coarse, node_map, areaw = hs.build_coarse_psn(psn, c=c)
Cc = ho.BlockAngleSystem(coarse, i, g, 3, 'directional')
A_c = Cc.matrix

# face-row mapping (internal only)
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


frow = face_rows(psn)
crow = face_rows(coarse)
cpair = {}
for cf, cface in enumerate(coarse.faces):
    if cface["type"] == "internal":
        a, b = int(cface["a"]), int(cface["b"])
        cpair[(a, b)] = cpair.get((a, b), cf)
        cpair[(b, a)] = cpair.get((a, b), cf)
f2c = {}
for f, face in enumerate(psn.faces):
    if face["type"] != "internal":
        continue
    ca, cb = int(node_map[int(face["a"])]), int(node_map[int(face["b"])])
    if ca == cb:
        continue
    f2c[f] = cpair[(ca, cb)]

# injection: fine col (f,m,a) <- coarse row (cf,m,a)
inj_rows, inj_cols = [], []
for f, cf in f2c.items():
    for m in range(M):
        for a in range(K):
            inj_cols.append(psn.colidx[f, m] * K + a)
            inj_rows.append(crow[(cf, m, a)])
Pmat = coo_matrix((np.ones(len(inj_rows)), (inj_cols, inj_rows)),
                  shape=(psn.ncol * K, A_c.shape[0])).tocsr()

# restriction: coarse row (cf,m,a) <- average of fine rows (f,m,a)
rest_rows, rest_cols, rest_w = [], [], []
for f, cf in f2c.items():
    for m in range(M):
        for a in range(K):
            rest_rows.append(crow[(cf, m, a)])
            rest_cols.append(frow[(f, m, a)])
            rest_w.append(1.0)
Rmat = coo_matrix((rest_w, (rest_rows, rest_cols)),
                  shape=(A_c.shape[0], A_f.shape[0])).tocsr()


def vcycle(u, q, use_cmfd, data_cache, orders, it):
    direction = it % 2
    for pty in (0, 1):
        for key in orders[pty][direction]:
            HoNodeLocal_pass(data_cache[key], q)
    if use_cmfd:
        r_c = Rmat @ (b_ref - A_f @ u)
        u += Pmat @ (Cc.factor.solve(r_c))
    return u


# We'll drive directly with the microclass passes.  Set up HoNodeLocal.
nl = hs.HoNodeLocal(psn, i, g, 3, 'directional', gs_tol=1e-14, max_gs=10)
nl.u = None
data_cache = {k: nl._microclass(k) for k in nl.mc_nodes}
orders = nl._mc_order()

def HoNodeLocal_pass(data, q):
    nl._mc_pass(data, q)

N = 1200
for use_cmfd in (False, True):
    u = np.zeros(psn.ncol * K)
    nl.u = u
    res_hist = []
    t0 = time.time()
    for it in range(1, N + 1):
        direction = (it - 1) % 2
        for pty in (0, 1):
            for key in orders[pty][direction]:
                nl._mc_pass(data_cache[key], q0)
        if use_cmfd and it % 2 == 0:
            r_c = Rmat @ (b_ref - A_f @ u)
            u += Pmat @ (Cc.factor.solve(r_c))
        if it % 10 == 0:
            res_hist.append(float(np.linalg.norm(A_f @ u - b_ref)))
    tail = res_hist[-40:]
    # local 2-step (20-iter-blocked) contraction in the tail
    factors = []
    for t in range(len(tail) - 2):
        if tail[t] > 0:
            factors.append(tail[t + 2] / tail[t])
    med = float(np.median(factors)) if factors else float('nan')
    print(f"{'CMFD' if use_cmfd else 'plain'}: "
          f"res[10]={res_hist[0]:.2e} res[{N}]={res_hist[-1]:.2e} "
          f"tail-contraction(2-step)={med:.4f} t={time.time()-t0:.1f}s",
          flush=True)
