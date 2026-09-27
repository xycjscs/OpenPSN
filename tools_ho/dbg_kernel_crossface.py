# -*- coding: utf-8 -*-
"""Is the local directional kernel L block-diagonal over mirror pairs?
L[m, lf, a, lf2*K+b] couples face lf to face lf2 at the same node.
If L mixes m -> m' across faces for some (i,g), the global (i,g) face
system is NOT block-diagonal over mirror pairs and lever 1 is invalid.
Check ALL (i, g) x all m x all (lf, lf2) cross-face energy per material."""
import os, sys
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, '/opt/data/workspace/OpenPSN')
import numpy as np
from psn2d import model
from psn2d import __main__ as M
from psn2d.highorder import local_directional_response

EX = '/opt/data/workspace/OpenPSN/examples/c5g7/study2_rectangular'
spec = model.load_spec(EX + '/c5g7_rect_uo2_assembly.yaml')
case = {'name': 'M4_S1', 'model': 'ty3', 'I': 30, 'M': 4, 'subdivide': 1,
        'kref': 1.3347260, 'A': 3, 'closure': 'directional_matrix'}
spec['solver'].update({'keff_tol': 1e-10, 'max_outer': 6000, 'threads': 1})
psn = M.build_solver(spec, case, threads=1, mem_limit_gb=120)
p = psn
Mm, A = p.M, 3
print(f'M={Mm} I={p.I} ng={p.ng}; mirror map: m -> {Mm-1-m}')
# distinct material/shape groups and a representative node each
groups = {}
for gi, grp in enumerate(p._groups):
    n = p._node_of_grp[grp]
    groups[gi] = (grp, n)

worst_global = 0.0
detail = 0
for i in range(p.I):
    for g in range(p.ng):
        for gi, (grp, n) in groups.items():
            mat = p.mat_of_node[n]
            for m in range(Mm):
                pr, _ = p._pr_alpha_rect(i, g, m, mat, n)
                L, V, res = local_directional_response(
                    pr, m * p.dphi, (m + 1) * p.dphi, A)
                # cross-face energy: rows lf vs cols lf2, lf2 != lf,
                # restricted to the SAME mirror pair (pair-diagonal
                # cross-face is allowed; cross-PAIR is the violation)
                for lf in range(4):
                    for lf2 in range(4):
                        if lf2 == lf:
                            continue
                        blk = L[lf * (A+1):(lf+1)*(A+1),
                                lf2*(A+1):(lf2+1)*(A+1)]
                        e = float(np.abs(blk).max())
                        # which (a, b) are nonzero? a row mode, b col mode
                        worst_global = max(worst_global, e)
            # one structural detail per (i,g): the (0,1) cross block
            if detail < 3:
                pr, _ = p._pr_alpha_rect(i, g, 0, p.mat_of_node[n], n)
                L0, _, _ = local_directional_response(
                    pr, 0.0, p.dphi, A)
                d1 = float(np.abs(L0[0:(A+1), (A+1):2*(A+1)]).max())
                d2 = float(np.abs(L0[0:(A+1), 2*(A+1):3*(A+1)]).max())
                d3 = float(np.abs(L0[0:(A+1), 3*(A+1):4*(A+1)]).max())
                print(f'  i={i} g={g}: L[m0, lf0, :] cross-face max |'
                      f'lf1|={d1:.3e} |lf2|={d2:.3e} |lf3|={d3:.3e}')
                detail += 3 if (i == 0 and g == 0) else 0
print(f'GLOBAL worst cross-face L (any i,g,m,lf!=lf2) = {worst_global:.3e}')
print('ZERO cross-face => local kernel is face-diagonal => '
      'mirror-pair decomposition EXACT' if worst_global == 0
      else 'CROSS-FACE COUPLING PRESENT => lever-1 pair split INVALID')
