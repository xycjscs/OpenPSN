import os
os.environ.update({k: '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                                    'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                                    'NUMEXPR_NUM_THREADS', 'PSN_PAR')})
import sys
import numpy as np

sys.path.insert(0, '/opt/data/workspace/OpenPSN')
sys.path.insert(0,
    '/opt/data/cache/documents/a3_upgrade/OpenPSN_directional_A3_upgrade_Yuchao_Xu/code')
from psn2d import model
from psn2d import highorder
from psn2d.__main__ import build_solver
from _frozen_core import local_response as pkg_lr
from psn2d.node_rect import node_params_rect

IN = ('/opt/data/cache/documents/a3_upgrade/'
      'OpenPSN_directional_A3_upgrade_Yuchao_Xu/inputs')
spec = model.load_spec(f'{IN}/mox87_ty3_s2.json')
case = [c for c in model.expand_cases(spec) if c['name'] == 'M24_A3'][0]
psn = build_solver(spec, case, threads=1, mem_limit_gb=2)
psn._vectorize_setup()

i, g, m = 1, 1, 7
grp = (0, 2)
n0 = psn._node_of_grp[grp]
pr, _ = psn._pr_alpha_rect(i, g, m, psn.mat_of_node[n0], n0)
lo, hi = m * psn.dphi, (m + 1) * psn.dphi
L1, V1, r1 = highorder.local_response(pr, lo, hi, 0)
L2, V2, r2 = pkg_lr(pr, lo, hi, 0)
dL = L1 - L2
dV = V1 - V2
print("per-row max|dL| (faces 0..3, each K=1):")
for f in range(4):
    print(f"  face{f}: {np.abs(dL[f]).max():.6e}")
print("per-col max|dL|:", [f"{v:.3e}" for v in np.abs(dL).max(axis=0)])
print("V bad rows:", [f"{np.abs(dV[f]).max():.3e}" for f in range(5)])
print("V bad cols:", [f"{np.abs(dV[:, c]).max():.3e}" for c in range(9)])

# recompute off table by hand for both and compare
st, hx, hy, p = pr['St'], pr['hx'], pr['hy'], pr['p']
from psn2d.highorder import angular_forms
_, C, S, H = angular_forms(0., 1., lo, hi, 0, 0)
X, Y, Z = 2 * C, 2 * S, 2 * H
x, y, z = X[:, 0], Y[:, 0], Z[:, 0]
print("\nhand off rows (mine vs pkg expected):")
print(f" x+ q1x: -2p/(hx st) x = {-2*p/(hx*st)*x[0]:+.6e}")
print(f" x- q1x: +2p/(hx st) x = {+2*p/(hx*st)*x[0]:+.6e}")
print(f" y+ q1y: -2p/(hy st) y = {-2*p/(hy*st)*y[0]:+.6e}")
print(f" y- q1y: +2p/(hy st) y = {+2*p/(hy*st)*y[0]:+.6e}")
print(f" y+ q2y: -6p/(hy st) y = {-6*p/(hy*st)*y[0]:+.6e}")

# also check mean/flux part: isolate by zeroing off
# compare L[:, :4] (J columns) only
print("\nJ-column block max diff:", np.abs(dL[:, :4]).max())
print("source-column block max diff:", np.abs(dL[:, 4:]).max())
