# -*- coding: utf-8 -*-
"""Smoke: generic polar (I=30) + A=3 directional closure, checkerboard rect."""
import os, sys, time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from psn2d import model, __main__ as M_
from psn2d import mckern
mckern.ENABLED = True

spec_p = 'examples/psn_repro/checkerboard_1g_rect.yaml'
kref = 1.12974
for M in (4, 24):
    for S in (1, 6):
        spec_d = model.load_spec(spec_p)
        case = {'name': f'M{M}_S{S}', 'model': 'generic', 'I': 30, 'M': M,
                'subdivide': S, 'A': 3, 'closure': 'directional_matrix'}
        spec_d['solver'].update({'keff_tol': 1e-7, 'max_outer': 6000,
                                 'threads': 8})
        psn = M_.build_solver(spec_d, case, threads=8)
        t0 = time.time()
        from psn2d.highsweep import keff_highorder_gmres
        k, phi, qim, h = keff_highorder_gmres(
            psn, A=3, branch='directional', max_outer=6000,
            outer_tol=1e-7, rtol=1e-7, maxiter=800,
            pairs=True, nworkers=8, verbose=False)
        print(f"GEN I30 A3 weak M{M}_S{S}: nodes={psn.nodes} I_eff={psn.I} "
              f"keff={k:.7f} dref={(k-kref)*1e5:+.1f}pcm "
              f"conv={h['converged']} wall={time.time()-t0:.1f}s", flush=True)
