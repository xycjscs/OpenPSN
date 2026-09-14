# -*- coding: utf-8 -*-
"""dbg_rhs.py — check (1) microclass A invertibility and (2) the
neighbour RHS part: at the exact global solution u_ref, the local
system must satisfy  A x_n + nb_terms(u_ref, q) = 0."""
import sys
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
    spec['solver'].update({'keff_tol': 1e-10, 'max_outer': 10, 'threads': 1})
    psn = PM.build_solver(spec, case, threads=1, mem_limit_gb=32)
    if not hasattr(psn, '_groups'):
        psn._vectorize_setup()
    return psn


psn = build_psn()
i, g = 0, 0
B = ho.BlockAngleSystem(psn, i, g, 3, 'directional')
K, M = B.K, psn.M
nl = hs.HoNodeLocal(psn, i, g, 3, 'directional', verbose=False)

rng = np.random.default_rng(7)
qn = rng.normal(size=(psn.nodes, 5))
b = B.source_map @ qn.ravel()
u_ref = B.factor.solve(b)

print(f"ncol={psn.ncol} nmc={len(nl.mc_nodes)}")

# (1) invertibility of every microclass A
worst = 1e300
for key, nodes in nl.mc_nodes.items():
    data = nl._microclass(key)
    s = np.linalg.svd(data["A"], compute_uv=False)
    worst = min(worst, s.min())
    if s.min() < 1e-6:
        print(f"  ILL-CONDITIONED mc {key[:2]} nodes={len(nodes)} "
              f"smin={s.min():.3e} n={nodes[:4]}")
print("min singular value over all mc:", worst)

# (2) residual of the local system at the exact solution
nl.u = u_ref.copy()  # flat (ncol*K,)
worst_res = 0.0
for key, nodes in nl.mc_nodes.items():
    data = nl._microclass(key)
    x = u_ref[data["wb"]]  # (Nn, d)
    bdata = nl._mc_b(data, qn)
    r = data["A"] @ x.T - bdata.T  # (d, Nn)
    m = np.max(np.abs(r))
    worst_res = max(worst_res, m)
    if m > 1e-9:
        print(f"  BAD mc {key[:2]} n={len(nodes)} maxres={m:.3e} "
              f"bmax={np.max(np.abs(bdata)):.3e}")
print("max local-system residual at exact solution:", worst_res)
