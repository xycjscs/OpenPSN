import os
os.environ.update({k: '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                                    'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                                    'NUMEXPR_NUM_THREADS', 'PSN_PAR')})
import sys
import numpy as np

sys.path.insert(0, '/opt/data/workspace/OpenPSN')
from psn2d import model
from psn2d import highorder
from psn2d.__main__ import build_solver

IN = '/opt/data/cache/documents/a3_upgrade/OpenPSN_directional_A3_upgrade_Yuchao_Xu/inputs'
spec = model.load_spec(f'{IN}/mox87_ty3_s2.json')
case = [c for c in model.expand_cases(spec) if c['name'] == 'M24_A3'][0]
psn = build_solver(spec, case, threads=1, mem_limit_gb=2)
print("rect:", psn.rect, "nodes:", psn.nodes, "ng:", psn.ng,
      "I:", psn.I, "M:", psn.M, "ncol:", psn.ncol)

sweep = highorder.BlockSweep(psn, A=3, branch='directional')
matidx = psn.mat[psn.j_idx, psn.i_idx]
area_w = psn.area.ravel()
ng = psn.ng


def fission_rate(phi):
    fr = np.zeros(psn.nodes)
    for g in range(ng):
        fr += psn.nuSf[g, matidx] * phi[g]
    fr *= area_w
    return fr


mom = np.zeros((ng, psn.nodes, 5))
mom[:, :, 0] = 1.0
qnode = psn._source_update(mom, 1.0, matidx)
print("init qnode: norm=%.4e min=%.4e max=%.4e"
      % (np.linalg.norm(qnode), qnode.min(), qnode.max()))

lam = 1.0
F_old = float(fission_rate(np.ones((ng, psn.nodes))).sum())
for outer in range(6):
    phi, qim = sweep.all_groups(qnode)
    F_new = float(fission_rate(phi).sum())
    lam_new = lam * F_new / F_old
    s_next = psn._source_update(qim, lam_new, matidx)
    print(f"it{outer}: lam={lam_new:.8f}  phi0 min={phi[:, 0].min():+.4e} "
          f"max={phi[:, 0].max():+.4e}  snext norm={np.linalg.norm(s_next):.4e} "
          f"min={s_next.min():+.3e} max={s_next.max():+.3e}")
    qnode = s_next
    lam, F_old = lam_new, F_new
