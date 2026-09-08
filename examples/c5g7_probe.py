# -*- coding: utf-8 -*-
"""Probe C5G7-2D 1/4 core: run S/M case, report keff, wall time, peak RSS."""
import resource
import sys
import time

import numpy as np

sys.path.insert(0, '/opt/data/workspace/OpenPSN')
from psn2d import model
from psn2d import __main__ as M

S = int(sys.argv[1])
MM = int(sys.argv[2])
TOL = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-10

spec = model.load_spec('/opt/data/workspace/OpenPSN/examples/c5g7_2d_quarter_core.yaml')
case = {'name': f'M{MM}_S{S}', 'model': 'ty3', 'I': 30, 'M': MM,
        'subdivide': S, 'kref': 1.18646}
t0 = time.time()
psn = M.build_solver(spec, case)
k, phi, qnode = psn.keff(max_outer=6000, outer_tol=TOL, verbose=True)
dt = time.time() - t0
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # MB -> GB
print(f"\nPROBE_DONE M={MM} S={S} nodes={psn.nodes} keff={k:.7f} "
      f"pcm={(k-1.18646)*1e5:+.1f} wall={dt:.0f}s peak_RSS={peak:.2f}GB", flush=True)
