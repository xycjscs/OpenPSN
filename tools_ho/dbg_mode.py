# -*- coding: utf-8 -*-
"""dbg_mode.py — extract the dominant slow MODE of the i2 GS iteration
(run from zero, take the tail error, which is dominated by the slowest
eigenmode) and characterize its spatial shape.  Also omega-scan the
residual-CMFD correction to find the stable/fastest damping."""
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
q0 = np.random.default_rng(42).normal(size=(psn.nodes, 5))
b_ref = B.source_map @ q0.ravel()
u_ref = B.factor.solve(b_ref)

# --- run GS from zero, capture the tail error to isolate the slow mode
nl = hs.HoNodeLocal(psn, i, g, 3, 'directional', gs_tol=1e-16, max_gs=10000)
nl.u = np.zeros(psn.ncol * K)
data_cache = {k: nl._microclass(k) for k in nl.mc_nodes}
orders = nl._mc_order()
N = 700
# capture u every 10 sweeps in the tail
snap = {}
for it in range(1, N + 1):
    direction = (it - 1) % 2
    for pty in (0, 1):
        for key in orders[pty][direction]:
            nl._mc_pass(data_cache[key], q0)
    if it >= N - 100 and it % 10 == 0:
        snap[it] = nl.u.copy()

# dominant mode = the slowest-decaying component of the tail error.
# Use two tail snapshots far apart: e_a, e_b; the slow mode direction is
# ~ e_b (the fast modes have decayed).  Contraction from a->b:
its = sorted(snap)
e_last = snap[its[-1]] - u_ref
# normalize spatially: this mode lives on the face-current unknowns.
# Reduce to a per-node scalar to visualize the spatial shape: for each
# node take the max |.| over its own face-current blocks (approx).
# Simpler: look at the error on a few representative nodes' internal faces.
print(f"tail iters captured: {its[0]}..{its[-1]}  ({len(its)} snaps)")
# contraction factor over the tail window (10-sweep steps)
errs = [float(np.linalg.norm(snap[t] - u_ref)) for t in its]
factors = [errs[k + 1] / errs[k] for k in range(len(errs) - 1)
           if errs[k] > 1e-300]
if factors:
    print(f"tail 10-sweep contraction: median={np.median(factors):.4f} "
          f"min={min(factors):.4f} max={max(factors):.4f}")
    print(f"  => per-sweep ~{np.median(factors)**0.1:.5f}")

# spatial shape of the dominant mode: e_last on internal-face current of
# node n, mode m=0 slot a=0 (the leading azimuthal mode).  Map to a
# coarse (pin) grid to see if it's smooth.
# Build node index -> (j, i) and pin -> (j//3, i//3)
jidx = psn.j_idx
iidx = psn.i_idx
# For each internal face, the current col = colidx[f,m]*K + a.
# Collect the leading mode (m=0,a=0) value per internal face, then
# downsample to pin level.
nx, ny = psn.nx, psn.ny
# accumulate per pin
pin_vals = np.zeros((ny // 3, nx // 3))
pin_cnt = np.zeros((ny // 3, nx // 3))
for f, face in enumerate(psn.faces):
    if face["type"] != "internal":
        continue
    n = int(face["a"])
    c = psn.colidx[f, 0] * K + 0
    pin_vals[jidx[n] // 3, iidx[n] // 3] += e_last[c]
    pin_cnt[jidx[n] // 3, iidx[n] // 3] += 1
pin_vals = pin_vals / np.maximum(pin_cnt, 1)
# smoothness: compare adjacent pin values (relative difference)
dh = np.abs(np.diff(pin_vals, axis=1)).mean()   # (17,16)
dv = np.abs(np.diff(pin_vals, axis=0)).mean()   # (16,17)
scale = np.abs(pin_vals).mean() + 1e-300
print(f"\nDominant slow mode (m=0 internal-face current), pin level:")
print(f"  pin grid {pin_vals.shape}, |val| mean={np.abs(pin_vals).mean():.2e} "
      f"max={np.abs(pin_vals).max():.2e}")
print(f"  horiz/vert adjacent mean abs diff: {dh/scale:.3f} / {dv/scale:.3f} "
      f"(<<1 => SMOOTH spatial mode; ~1 => oscillatory)")
# Also: is it concentrated on one pin / a boundary (eigenlayer) or spread?
frac = (np.abs(pin_vals) > 0.5 * np.abs(pin_vals).max()).mean()
print(f"  fraction of pins with |val|>50% of max: {frac:.2f} "
      f"(1.0 => spread/uniform; small => localized/boundary)")

# ============ omega scan for residual CMFD ============
c = 3
coarse, node_map, areaw = hs.build_coarse_psn(psn, c=c)
Cc = ho.BlockAngleSystem(coarse, i, g, 3, 'directional')
A_c = Cc.matrix


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

inj_rows, inj_cols = [], []
for f, cf in f2c.items():
    for m in range(M):
        for a in range(K):
            inj_cols.append(psn.colidx[f, m] * K + a)
            inj_rows.append(crow[(cf, m, a)])
Pmat = coo_matrix((np.ones(len(inj_rows)), (inj_cols, inj_rows)),
                  shape=(psn.ncol * K, A_c.shape[0])).tocsr()
rest_rows, rest_cols, rest_w = [], [], []
for f, cf in f2c.items():
    for m in range(M):
        for a in range(K):
            rest_rows.append(crow[(cf, m, a)])
            rest_cols.append(frow[(f, m, a)])
            rest_w.append(1.0)
Rmat = coo_matrix((rest_w, (rest_rows, rest_cols)),
                  shape=(A_c.shape[0], A_f.shape[0])).tocsr()

Nscan = 500
print(f"\nomega scan (500 sweeps, CMFD every 2 sweeps, residual form):")
for omega in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
    u = np.zeros(psn.ncol * K)
    nl.u = u
    hist = []
    for it in range(1, Nscan + 1):
        direction = (it - 1) % 2
        for pty in (0, 1):
            for key in orders[pty][direction]:
                nl._mc_pass(data_cache[key], q0)
        if it % 2 == 0:
            r_c = Rmat @ (b_ref - A_f @ u)
            u += omega * (Pmat @ Cc.factor.solve(r_c))
        if it % 25 == 0:
            hist.append(float(np.linalg.norm(A_f @ u - b_ref)))
    tailf = [hist[k + 1] / hist[k] for k in range(len(hist) - 1)
             if hist[k] > 1e-300]
    med = float(np.median(tailf)) if tailf else float('nan')
    print(f"  omega={omega:.1f}: res[end]={hist[-1]:.2e} "
          f"tail-25step-contraction={med:.4f}", flush=True)
