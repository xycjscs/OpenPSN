# -*- coding: utf-8 -*-
"""dbg_cmfd.py — de-risk inner CMFD: build coarse PSN2D, solve the
coarse A0 state-matrix system (cheap), and measure how many fine GS
passes a coarse flux-ratio correction saves on the slow i2 system."""
import sys
import time

import numpy as np

sys.path.insert(0, '/opt/data/workspace/OpenPSN')
from psn2d import highorder as ho
from psn2d import highsweep as hs
from psn2d import model as mdl
from psn2d import __main__ as PM
from scipy.sparse import coo_matrix

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
c = 3
coarse, node_map, areaw = hs.build_coarse_psn(psn, c=c)
print(f"fine {psn.nx}x{psn.ny}={psn.nodes} -> coarse {coarse.nx}x{coarse.ny}"
      f"={coarse.nodes}")
print(f"coarse boundary={coarse.bnd}")
print(f"coarse ncol={coarse.ncol} M={coarse.M} mat_set={set(int(m) for m in coarse.mat.ravel())}")
print(f"coarse St shape={coarse.St.shape} Sgg={coarse.Sgg.shape}")

# --- coarse A0 operator (the legacy 9x9 state-matrix assembly, A=0) ---
i = psn.I - 1   # slow polar direction
g = 0
CoarseA0 = ho.BlockAngleSystem(coarse, i, g, A=0, branch='original')
t0 = time.time()
x0 = CoarseA0.matrix @ 0
print(f"coarse A0 matrix {CoarseA0.matrix.shape} nnz={CoarseA0.matrix.nnz} "
      f"build={time.time()-t0:.2f}s")

# random source
rng = np.random.default_rng(1)
q_fine = rng.normal(size=(psn.nodes, 5))
# coarse source = area-weighted sum of fine source
s_c = np.zeros((coarse.nodes, 5))
for nf in range(psn.nodes):
    s_c[node_map[nf]] += areaw[nf] * q_fine[nf]
b_c = CoarseA0.source_map @ s_c.ravel()
t0 = time.time()
u_c = CoarseA0.factor.solve(b_c)
out_c = CoarseA0.apply(s_c, check_residual=False)   # (coarse.nodes, 5) W/M
t_solve = time.time() - t0
print(f"coarse A0 solve+apply = {t_solve*1000:.1f} ms  "
      f"phi_c max={out_c[:,0].max():.3e}")

# fine A3 reference solution at the same source (for the ratio)
B = ho.BlockAngleSystem(psn, i, g, 3, 'directional')
b_f = B.source_map @ q_fine.ravel()
u_f_ref = B.factor.solve(b_f)
out_f_ref = B.apply(q_fine, check_residual=False)
phi_f_ref = out_f_ref[:, 0]

# ratio R at fine nodes (from coarse phi via node_map)
R = np.clip(out_c[:, 0][node_map] / np.maximum(phi_f_ref, 1e-12), 0.25, 4.0)
print(f"R = phi_c/phi_f: min={R.min():.3f} max={R.max():.3f} "
      f"mean={R.mean():.3f} std={R.std():.3f}")

# --- measure: GS passes to converge i2, with/without CMFD correction ---
nl = hs.HoNodeLocal(psn, i, g, 3, 'directional', gs_tol=1e-12, max_gs=2000)

# baseline trajectory (no correction)
u = np.zeros(psn.ncol * nl.K)
nl.u = u
data_cache = {k: nl._microclass(k) for k in nl.mc_nodes}
orders = nl._mc_order()
b_ref = B.source_map @ q_fine.ravel()
res = None
t0 = time.time()
for it in range(1, 2001):
    direction = it % 2
    for pty in (0, 1):
        for key in orders[pty][direction]:
            nl._mc_pass(data_cache[key], q_fine)
    if it % 10 == 0:
        res = max(nl._mc_residual(data_cache[k], q_fine)
                  for k in data_cache)
        if it <= 40 or it % 40 == 0:
            print(f"  base it={it:4d} res={res:.2e}", flush=True)
        if res < nl.gs_tol:
            break
t_base = time.time() - t0
print(f"BASELINE converged it={it} res={res:.2e} t={t_base:.1f}s")
