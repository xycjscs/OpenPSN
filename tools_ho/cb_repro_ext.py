# -*- coding: utf-8 -*-
"""Checkerboard reproduction-table M-extension (legacy PSN, generic I=30).

Extends the paper's Fig.3 reproduction tables (appA/appB, M=2..24) with
M = 48, 96, 192 for S = 1..10, weak and strong absorber, EXACTLY on the
original Chao-PSN protocol: closure=legacy (psn.keff plain power
iteration), angular model=generic I=30, tol 1e-10, max_outer 4000.

Usage:  python tools_ho/cb_repro_ext.py weak|strong
Line format (same shape as other cb logs):
  <TAG> M<M>_S<S> keff=<9dp> dref=<+-x.x>pcm iter=<n> t=<s>s
"""
import os, sys, time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from psn2d import model
from psn2d.__main__ import run_case

VAR = sys.argv[1]
assert VAR in ('weak', 'strong')
if VAR == 'weak':
    YAML = 'examples/psn_repro/checkerboard_1g.yaml'
    TAG, KREF = 'CBRE', 1.12974
else:
    YAML = 'examples/psn_repro/checkerboard_1g_strong.yaml'
    TAG, KREF = 'CBRS', 0.51673

spec = model.load_spec(os.path.join(REPO, YAML))

M_EXT = (48, 96, 192)
S_LIST = tuple(range(1, 11))

print(f"=== cb_repro_ext {VAR}: S={S_LIST} x M={M_EXT} legacy generic I30 ===",
      flush=True)
print(f"nodes: {__import__('socket').gethostname()}", flush=True)

for S in S_LIST:
    for M in M_EXT:
        case = dict(spec.get('angular', {}))
        case.update({'name': f'S{S}_M{M}', 'M': M, 'subdivide': S,
                     'kref': KREF})
        t0 = time.time()
        try:
            r = run_case(spec, case, verbose=False, threads=1)
            dt = time.time() - t0
            print(f"{TAG} M{M}_S{S} keff={r['keff']:.9f} "
                  f"dref={r['pcm']:+.1f}pcm t={dt:.1f}s", flush=True)
        except Exception as e:
            dt = time.time() - t0
            print(f"{TAG} M{M}_S{S} ERROR {type(e).__name__}: {e} t={dt:.1f}s",
                  flush=True)
print("=== cb_repro_ext ALL DONE ===", flush=True)
