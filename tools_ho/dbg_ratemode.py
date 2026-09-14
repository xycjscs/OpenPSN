# -*- coding: utf-8 -*-
"""dbg_ratemode.py — test the TRADITIONAL CMFD flux-ratio correction on
the slow i2 mode:

  R_n = phi_c(n_c) / phi_f(n)   (node fluxes, node c = c x c block)
  u_new(face of node n) = u(face) * R_n

Measure (1) does the slow mode have a nonzero fine flux?  (2) does one
ratio-kill reduce the mode?  (3) full GS + ratio correction every 2
sweeps vs plain GS.
"""
import sys, time

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
    spec['solver'].update({'keff_tol': 1e-10, 'max_outer': 200,
                           'threads': 1})
    psn = PM.build_solver(spec, case, threads=1, mem_limit_gb=32)
    if not hasattr(psn, '_groups'):
        psn._vectorize_setup()
    return psn


psn = build_psn()
K, M = 4, psn.M
i2, g0 = psn.I - 1, 0
B = ho.BlockAngleSystem(psn, i2, g0, 3, 'directional')
A_f = B.matrix.tocsr()
rng = np.random.default_rng(42)
q0 = rng.normal(size=(psn.nodes, 5))
b = B.source_map @ q0.ravel()
u_ref = B.factor.solve(b)

c = 3
coarse, node_map, areaw = hs.build_coarse_psn(psn, c=c)
print(f"coarse built: {coarse.nodes} nodes", flush=True)

nl = hs.HoNodeLocal(psn, i2, g0, 3, 'directional', gs_tol=1e-14,
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


# ---- extract slow mode ----
t0 = time.time()
prev = nl.u.copy()
e = None
for k in range(1500):
    one_pass(nl.u)
    if k >= 1496:
        e = nl.u - prev
    prev = nl.u.copy()
e = e / np.linalg.norm(e)
print(f"slow mode extracted ({time.time()-t0:.0f}s)", flush=True)

# ---- fine flux of the mode (nl._reduce with q=0) ----
nl.u = e
phi_mode = nl._reduce(np.zeros((psn.nodes, 5)))
phi_mode = phi_mode[:, 0]
print(f"mode fine flux: max={np.abs(phi_mode).max():.3e} "
      f" rms={np.sqrt((phi_mode**2).mean()):.3e}")

# ---- coarse flux of q0 with the high-order closure ----
t0 = time.time()
nlc = hs.HoNodeLocal(coarse, i2, g0, 3, 'directional', gs_tol=1e-12,
                     max_gs=400)
# coarse source = area-weighted restriction
s_c = np.zeros((coarse.nodes, 5))
for nf in range(psn.nodes):
    s_c[node_map[nf]] += areaw[nf] * q0[nf]
u_c, itc, resc, out_c = nlc.sweep(s_c)
phi_c = out_c[:, 0]
print(f"coarse sweep: it={itc} res={resc:.2e} ({time.time()-t0:.1f}s)",
      flush=True)

# ratio on fine nodes
phi_f = nl._reduce(q0)[:, 0]  # fine flux of the reference solution
R = phi_c[node_map] / np.maximum(phi_f, 1e-300)
print(f"flux ratio R: min={R.min():.3f} max={R.max():.3f} "
      f" rmsdev={np.sqrt(((R-1)**2).mean()):.3e}", flush=True)

# node-wise multiplier on the mode
n_of_col = np.zeros(A_f.shape[1], dtype=np.int64)
for f, face in enumerate(psn.faces):
    t = face["type"]
    if t == "internal":
        n = int(face["a"])
        n_of_col[psn.colidx[f, :] * K + np.arange(K)] = \
            np.repeat(n, M * K)
    elif t == "vacuum":
        n = int(face["a"])
        col = int(psn.colidx[f, 0])
        if col >= 0:
            n_of_col[col * K:(col + 1) * K] = n
    else:
        n = int(face["a"])
        for m in range(M):
            col = int(psn.colidx[f, m])
            if col >= 0:
                n_of_col[col * K:(col + 1) * K] = n
Rnode = R[n_of_col]
killed = e * Rnode
red = np.linalg.norm(e - killed) / np.linalg.norm(e)
print(f"\nONE ratio-kill on the slow mode: ||e - e*R||/||e|| = {red:.4f}"
      f"  (want << 1)", flush=True)

# ---- full runs ----
def run(mode, mg=1500, every=2, omega=1.0, switch=1e-9, label=""):
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
        if mode == "ratio" and it % every == 0:
            phi_f = nl._reduce(q0)[:, 0]
            phi_c_arr = phi_c  # fixed q -> fixed coarse flux
            Rr = phi_c_arr[node_map] / np.maximum(phi_f, 1e-300)
            Rr = np.clip(omega * Rr + (1 - omega), 0.0, 5.0)
            u *= Rr[n_of_col]
            nl.u = u
        if it % 25 == 0:
            res = float(np.abs(A_f @ u - b).max() / np.abs(b).max())
            if it <= 50 or it % 100 == 0:
                print(f"  [{label}] it={it:4d} res={res:.2e}", flush=True)
            if res < 1e-12:
                break
            if mode == "ratio" and res < switch:
                mode = "plain"
                print(f"  [{label}] it={it:4d} switched to plain "
                      f"(res={res:.2e})", flush=True)
    du = float(np.abs(u - u_ref).max() / np.abs(u_ref).max())
    print(f"[{label}] done it={it} res={res:.2e} du={du:.2e} "
          f"t={time.time()-t0:.1f}s\n", flush=True)
    return it


run("plain", label="plain")
run("ratio", label="ratio-kill w=1")
run("ratio", omega=0.6, label="ratio-kill w=0.6")
