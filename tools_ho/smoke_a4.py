# -*- coding: utf-8 -*-
"""Smoke: A=4 / A=5 directional closure, checkerboard rect weak, GMRES."""
import os, sys, time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from psn2d import model, __main__ as M_
from psn2d import mckern
mckern.ENABLED = True
from psn2d.highsweep import keff_highorder_gmres

spec_p = 'examples/psn_repro/checkerboard_1g_rect.yaml'
kref = 1.12974
for A in (4, 5):
    for M, S in [(4, 2), (24, 6)]:
        spec_d = model.load_spec(spec_p)
        case = {'name': f'A{A}M{M}_S{S}', 'model': 'ty3', 'I': 30, 'M': M,
                'subdivide': S, 'A': A, 'closure': 'directional_matrix'}
        spec_d['solver'].update({'keff_tol': 1e-7, 'max_outer': 6000,
                                 'threads': 8})
        psn = M_.build_solver(spec_d, case, threads=8)
        t0 = time.time()
        k, phi, qim, h = keff_highorder_gmres(
            psn, A=A, branch='directional', max_outer=6000,
            outer_tol=1e-7, rtol=1e-7, maxiter=800,
            pairs=True, nworkers=8, verbose=False)
        print(f"A{A} weak M{M}_S{S}: keff={k:.7f} dref={(k-kref)*1e5:+.1f}pcm "
              f"K={h['K']} conv={h['converged']} "
              f"src_res={h.get('source_residual', float('nan')):.1e} "
              f"wall={time.time()-t0:.1f}s", flush=True)
