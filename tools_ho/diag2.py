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

# --- identical random 5-moment source ---
rng = np.random.default_rng(2026)
q0 = rng.random((psn.ng, psn.nodes, 5))
q0[:, :, 0] = 1.0 + rng.random((psn.ng, psn.nodes))  # positive base

# 1) legacy vectorized sweep (the trusted production path)
phi_leg, qim_leg = psn._sweep_all_serial(q0)

# 2) high-order original branch A=0 (same discrete system expected)
sw = highorder.BlockSweep(psn, A=0, branch='original')
phi_ho, qim_ho = sw.all_groups(q0)

# 3) high-order directional branch A=3 (different method, sanity only)
sw3 = highorder.BlockSweep(psn, A=3, branch='directional')
phi_ho3, qim_ho3 = sw3.all_groups(q0)

print("== A0 original vs legacy (should be ~0) ==")
print("phi  rel L2:", np.linalg.norm(phi_ho - phi_leg) / np.linalg.norm(phi_leg))
for l in range(5):
    d = np.linalg.norm(qim_ho[:, :, l] - qim_leg[:, :, l]) \
        / max(np.linalg.norm(qim_leg[:, :, l]), 1e-30)
    print(f"  mom{l}: rel L2 = {d:.3e}")

print("== A3 directional (different method, just magnitude) ==")
print("phi0 mean leg vs ho3:", phi_leg[:, 0].mean(), phi_ho3[:, 0].mean())
for l in range(5):
    print(f"  mom{l} mean: leg={qim_leg[:, :, l].mean():+.5e} "
          f"ho3={qim_ho3[:, :, l].mean():+.5e}")

# per-system diagnostics
diag = sw3.diagnostics()
for key, d in diag.items():
    print(key, "face_unknowns:", d["face_unknowns"],
          "local_res:", f"{d['local_residual']:.2e}",
          "face_res:", f"{d['face_residual']:.2e}")
