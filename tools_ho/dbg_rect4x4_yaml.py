# -*- coding: utf-8 -*-
"""Smoke v2: rect4x4 YAML — correct side metrics (per-row x-sum)."""
import sys
sys.path.insert(0, '/opt/data/workspace/OpenPSN')
import numpy as np
from psn2d import model, __main__ as M

EX = 'examples/c5g7/study2_rectangular/'
for f, fuel in [('c5g7_rect4x4_uo2_assembly.yaml', 1),
                ('c5g7_rect4x4_mox_assembly.yaml', 2)]:
    spec = model.load_spec(EX + f)
    for case in model.expand_cases(spec):
        psn = M.build_solver(spec, case, threads=1, mem_limit_gb=120)
        mat = psn.mat
        fuel_nodes = int((mat == fuel).sum())
        side_x = float(psn.hx.sum(axis=1)[0])
        side_y = float(psn.hy.sum(axis=0)[0])
        # material layout printout (S1 only, compact)
        print(f"{case['name']}: shape={mat.shape} fuel={fuel_nodes} "
              f"side_x={side_x:.5f} side_y={side_y:.5f}")
        if case['subdivide'] == 1:
            for r in mat:
                print('   ', ' '.join('.' if v == 0 else '#' for v in r))
        del psn
# expected: side = w + 2f + w = 0.1514375 + 2*0.9571251 + 0.1514375
print('expected side:', 2*0.15143746025551064 + 2*0.9571250794889787)
print('YAML SMOKE2 OK')
