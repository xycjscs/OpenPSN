# -*- coding: utf-8 -*-
"""Check: per-node width convention in rect3x3 vs my rect4x4."""
import sys
sys.path.insert(0, '/opt/data/workspace/OpenPSN')
import numpy as np
from psn2d import model, __main__ as M

EX = 'examples/c5g7/study2_rectangular/'
spec = model.load_spec(EX + 'c5g7_rect_uo2_assembly.yaml')
case = {'name': 'chk', 'model': 'ty3', 'I': 30, 'M': 4, 'subdivide': 1,
        'kref': 1.0, 'A': 3, 'closure': 'directional_matrix'}
psn = M.build_solver(spec, case, threads=1, mem_limit_gb=120)
hx = np.array(psn.hx)
print('rect3x3 hx shape:', hx.shape)
print('row0 first 9:', [round(v, 5) for v in hx[0][:9]])
print('row0 sum:', round(float(hx[0].sum()), 5))
print('row1 first 9:', [round(v, 5) for v in hx[1][:9]])
print('row1 sum:', round(float(hx[1].sum()), 5))
print('per-pin fuel node width (expect 0.957 or 0.319?):',
      round(float(hx[1][1]), 5))

# now mine
spec2 = model.load_spec(EX + 'c5g7_rect4x4_uo2_assembly.yaml')
c2 = {'name': 'chk2', 'model': 'ty3', 'I': 30, 'M': 4, 'subdivide': 1,
      'kref': 1.0, 'A': 3, 'closure': 'directional_matrix'}
psn2 = M.build_solver(spec2, c2, threads=1, mem_limit_gb=120)
hx2 = np.array(psn2.hx)
print('\nrect4x4 hx shape:', hx2.shape)
print('row0:', [round(v, 5) for v in hx2[0]])
print('row1:', [round(v, 5) for v in hx2[1]])
print('row1 sum:', round(float(hx2[1].sum()), 5))
