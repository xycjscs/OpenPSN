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
from _frozen_core import local_response as pkg_lr, local_directional_response as pkg_ldr

IN = ('/opt/data/cache/documents/a3_upgrade/'
      'OpenPSN_directional_A3_upgrade_Yuchao_Xu/inputs')
spec = model.load_spec(f'{IN}/mox87_ty3_s2.json')
case = [c for c in model.expand_cases(spec) if c['name'] == 'M24_A3'][0]
psn = build_solver(spec, case, threads=1, mem_limit_gb=2)
psn._vectorize_setup()

for (i, g, m) in [(0, 0, 0), (1, 1, 7), (2, 3, 20)]:
    for grp in psn._groups:
        n0 = psn._node_of_grp[grp]
        pr, _ = psn._pr_alpha_rect(i, g, m, psn.mat_of_node[n0], n0)
        lo, hi = m * psn.dphi, (m + 1) * psn.dphi
        L1, V1, _ = highorder.local_response(pr, lo, hi, 0)
        L2, V2, _ = pkg_lr(pr, lo, hi, 0)
        dL = np.abs(L1 - L2).max(axis=0)   # per column
        dV = np.abs(V1 - V2).max(axis=0)
        badL = np.where(dL > 1e-9)[0]
        badV = np.where(dV > 1e-9)[0]
        print(f"i{i} g{g} m{m} grp{grp}: L bad cols={badL.tolist()} "
              f"max={dL.max():.3e} | V bad cols={badV.tolist()} max={dV.max():.3e}")
