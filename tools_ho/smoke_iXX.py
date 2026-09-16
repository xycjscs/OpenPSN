# -*- coding: utf-8 -*-
"""Smoke: generic polar set I=60 / I=120 on checkerboard weak, S4 M4."""
import os, sys
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from psn2d import model, __main__ as M_, mckern
mckern.ENABLED = True
from psn2d import node
from psn2d.highsweep import keff_highorder_gmres

spec = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'examples', 'psn_repro', 'checkerboard_1g_rect.yaml')
for I in (60, 120):
    s = node.generic_polar_set(I)
    print(f'I={I}: theta len={len(s["theta"])} wsum={s["w"].sum():.8f} '
          f'theta[0]={s["theta"][0]:.4f} dtheta={s["dtheta"]:.4f}', flush=True)
    spec_d = model.load_spec(spec)
    case = {'name': f'I{I}smoke', 'model': 'generic', 'I': I, 'M': 4,
            'subdivide': 4, 'A': 3, 'closure': 'directional_matrix'}
    spec_d['solver'].update({'keff_tol': 1e-7, 'max_outer': 6000, 'threads': 8})
    psn = M_.build_solver(spec_d, case, threads=8)
    k, phi, qim, h = keff_highorder_gmres(psn, A=3, branch='directional',
        max_outer=6000, outer_tol=1e-7, rtol=1e-7, maxiter=800,
        pairs=True, nworkers=8, verbose=False)
    dref = (k - 1.12974) * 1e5
    print(f'I={I} S4 M4 keff={k:.7f} dref={dref:+.1f}pcm '
          f'iters={h["outer_iterations"]} conv={h["converged"]}', flush=True)
print('SMOKE OK')
