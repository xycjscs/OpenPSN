# -*- coding: utf-8 -*-
"""Smoke: checkerboard rect path.
1) legacy closure: rect vs square, weak+strong, M4 S1 -> keff must agree
2) A=3 directional GMRES pool: weak M2_S1, M24_S6; strong M2_S1
"""
import os, sys, time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from psn2d import model, __main__ as M_
from psn2d import mckern
mckern.ENABLED = True

def run_a3(spec_p, M, S, threads=8):
    spec_d = model.load_spec(spec_p)
    case = {'name': f'M{M}_S{S}', 'model': 'ty3', 'I': 30, 'M': M,
            'subdivide': S, 'A': 3, 'closure': 'directional_matrix'}
    spec_d['solver'].update({'keff_tol': 1e-7, 'max_outer': 6000,
                             'threads': threads})
    psn = M_.build_solver(spec_d, case, threads=threads)
    t0 = time.time()
    from psn2d.highsweep import keff_highorder_gmres
    k, phi, qim, h = keff_highorder_gmres(
        psn, A=3, branch='directional', max_outer=6000,
        outer_tol=1e-7, rtol=1e-7, maxiter=800,
        pairs=True, nworkers=threads, verbose=False)
    return k, h, time.time() - t0, psn

# --- 1) legacy consistency: square vs rect (run_case, default opt) ---
for tag, sq, rc, kref in [
    ('weak',  'examples/psn_repro/checkerboard_1g.yaml',
              'examples/psn_repro/checkerboard_1g_rect.yaml', 1.12974),
    ('strong','examples/psn_repro/checkerboard_1g_strong.yaml',
              'examples/psn_repro/checkerboard_1g_rect_strong.yaml', 0.51673),
]:
    spec1 = model.load_spec(sq)
    spec2 = model.load_spec(rc)
    for sp in (spec1, spec2):
        sp['angular'] = {'model': 'ty3', 'M': 4}
        sp['solver'].update({'keff_tol': 1e-7, 'max_outer': 6000})
        sp['cases'] = [{'name': 'm4', 'model': 'ty3', 'I': 30,
                        'M': 4, 'subdivide': 1, 'kref': kref}]
    r1 = M_.run_case(spec1, spec1['cases'][0], verbose=False)
    r2 = M_.run_case(spec2, spec2['cases'][0], verbose=False)
    print(f"LEGACY {tag} M4_S1: square keff={r1['keff']:.7f}  "
          f"rect keff={r2['keff']:.7f}  dk={r2['keff']-r1['keff']:+.2e}  "
          f"dref={(r2['keff']-kref)*1e5:+.1f}pcm", flush=True)

# --- 2) A=3 directional GMRES pool on rect ---
for tag, rc, kref in [
    ('weak',   'examples/psn_repro/checkerboard_1g_rect.yaml',        1.12974),
    ('strong', 'examples/psn_repro/checkerboard_1g_rect_strong.yaml', 0.51673),
]:
    for M, S in [(2, 1), (24, 6)]:
        k, h, dt, psn = run_a3(rc, M, S)
        print(f"A3 {tag} M{M}_S{S}: nodes={psn.nodes} keff={k:.7f} "
              f"dref={(k-kref)*1e5:+.1f}pcm conv={h['converged']} "
              f"iters={h['outer_iterations']} wall={dt:.1f}s", flush=True)
