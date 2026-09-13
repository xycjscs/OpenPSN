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
from block_sweep import BlockSweep as PkgBlockSweep  # package original

IN = ('/opt/data/cache/documents/a3_upgrade/'
      'OpenPSN_directional_A3_upgrade_Yuchao_Xu/inputs')
spec = model.load_spec(f'{IN}/mox87_ty3_s2.json')
case = [c for c in model.expand_cases(spec) if c['name'] == 'M24_A3'][0]
psn = build_solver(spec, case, threads=1, mem_limit_gb=2)
psn._vectorize_setup()

# --- (1) local kernel comparison: mine vs package, A=0 ---
from _frozen_core import local_response as pkg_lr
i, g, m = 0, 0, 0
grp = psn._groups[0]
n0 = psn._node_of_grp[grp]
pr, _ = psn._pr_alpha_rect(i, g, m, psn.mat_of_node[n0], n0)
L1, V1, _ = highorder.local_response(pr, 0, psn.dphi, 0)
L2, V2, _ = pkg_lr(pr, 0, psn.dphi, 0)
print("kernel A0: L diff =", np.max(np.abs(L1 - L2)),
      " V diff =", np.max(np.abs(V1 - V2)))

# --- (2) sweep comparison: package BlockSweep vs mine, A=0 original ---
rng = np.random.default_rng(7)
q0 = 1.0 + rng.random((psn.ng, psn.nodes, 5))
sw_pkg = PkgBlockSweep(psn, A=0, branch="original")
sw_mine = highorder.BlockSweep(psn, A=0, branch="original")
phi_p, qim_p = sw_pkg.all_groups(q0)
phi_m, qim_m = sw_mine.all_groups(q0)
print("sweep A0 original: phi rel L2 =",
      np.linalg.norm(phi_p - phi_m) / np.linalg.norm(phi_p),
      " qim0 rel L2 =",
      np.linalg.norm(qim_p[:, :, 0] - qim_m[:, :, 0])
      / np.linalg.norm(qim_p[:, :, 0]))

# --- (3) assembled matrix comparison for one (i,g) ---
sp = sw_pkg.system(0, 0)
sm = sw_mine.system(0, 0)
d = (sp.matrix - sm.matrix).tocoo()
print("matrix (i0,g0): nnz_pkg =", sp.matrix.nnz, " nnz_mine =", sm.matrix.nnz,
      " max|diff| =", np.max(np.abs(d.data)) if d.nnz else 0.0,
      " n_diff =", d.nnz)
sp2 = (sp.source_map - sm.source_map).tocoo()
print("source_map: max|diff| =",
      np.max(np.abs(sp2.data)) if sp2.nnz else 0.0, " n_diff =", sp2.nnz)
