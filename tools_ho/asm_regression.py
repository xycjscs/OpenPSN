"""Full-reflect regression after vacuum port (3 points vs golden)."""
import os, sys

os.chdir('/opt/data/workspace/OpenPSN')
sys.path.insert(0, '/opt/data/workspace/OpenPSN')
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['OPENBLAS_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '16'

from psn2d.model import load_spec, expand_cases
from psn2d import __main__ as M, highorder

spec = load_spec('examples/mox87_ty3_s2.json')
cases = expand_cases(spec)
want = {'M24_A3', 'M24_A4', 'M48_A3'}
golden = {'M24_A3': 1.1732714250722445,
          'M24_A4': 1.173287965672,
          'M48_A3': 1.173295267646}
for case in cases:
    if case['name'] not in want:
        continue
    psn = M.build_solver(spec, case, threads=16)
    psn._vectorize_setup()  # pre-warm on main thread (avoid prefactor race)
    k, phi, qim, info = highorder.keff_highorder(
        psn, A=case['A'], branch='directional',
        max_outer=10000, outer_tol=1e-11)
    g = golden[case['name']]
    print(f"{case['name']:10s} keff={k:.16f} golden={g:.16f} "
          f"dk={abs(k-g):.2e} converged={info['converged']}")
