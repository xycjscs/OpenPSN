# -*- coding: utf-8 -*-
"""dbg_coarse.py — de-risk the CMFD coarse-mesh build: build_coarse_psn
on the M2 uo2 assembly, build the coarse A3 BlockAngleSystem, check
shapes, and verify the fine-face->coarse-face mapping counts."""
import sys
import numpy as np

sys.path.insert(0, '/opt/data/workspace/OpenPSN')
from psn2d import highorder as ho
from psn2d import highsweep as hs
from psn2d import model as mdl
from psn2d import __main__ as PM

EX = '/opt/data/workspace/OpenPSN/examples/c5g7/study2_rectangular'


def build_psn(key='uo2', MM=2):
    spec = mdl.load_spec(EX + f'/c5g7_rect_{key}_assembly.yaml')
    case = {'name': f'M{MM}', 'model': 'ty3', 'I': 30, 'M': MM,
            'subdivide': 1, 'A': 3, 'closure': 'directional_matrix'}
    spec['solver'].update({'keff_tol': 1e-10, 'max_outer': 10, 'threads': 1})
    psn = PM.build_solver(spec, case, threads=1, mem_limit_gb=32)
    if not hasattr(psn, '_groups'):
        psn._vectorize_setup()
    return psn


psn = build_psn()
c = 3
built = hs.build_coarse_psn(psn, c=c)
assert built is not None, "coarse build failed"
coarse, node_map, areaw = built
print(f"fine nodes={psn.nodes} ({psn.nx}x{psn.ny})  "
      f"coarse nodes={coarse.nodes} ({coarse.nx}x{coarse.ny})")
print(f"fine ncol={psn.ncol}  coarse ncol={coarse.ncol}")
print(f"fine K=4 M={psn.M}   coarse M={coarse.M}  coarse faces={len(coarse.faces)}")

# coarse A3 system
cs = ho.BlockAngleSystem(coarse, 0, 0, 3, 'directional')
print(f"coarse A3 matrix: {cs.matrix.shape} nnz={cs.matrix.nnz}")
x = np.random.default_rng(0).normal(size=cs.matrix.shape[0])
y = cs.factor.solve(x)
res = float(np.abs(cs.matrix @ y - x).max())
print(f"coarse LU solve residual = {res:.2e}")

# fine-face -> coarse-face mapping
cpair = {}
for cf, cface in enumerate(coarse.faces):
    if cface["type"] == "internal":
        a, b = int(cface["a"]), int(cface["b"])
        cpair[(a, b)] = cf
        cpair[(b, a)] = cf
fine2coarse = {}
for f, face in enumerate(psn.faces):
    if face["type"] != "internal":
        continue
    na, nb = int(face["a"]), int(face["b"])
    ca, cb = int(node_map[na]), int(node_map[nb])
    if ca == cb:
        continue
    cf = cpair.get((ca, cb))
    if cf is None:
        print(f"  WARNING: fine face {f} crosses coarse boundary but no "
              f"coarse face for pair ({ca},{cb})")
        continue
    fine2coarse[f] = cf
print(f"fine internal faces crossing a coarse face: {len(fine2coarse)} "
      f"(of {sum(1 for f in psn.faces if f['type']=='internal')} internal)")
# each coarse internal face should be crossed by c=3 fine faces
from collections import Counter
cnt = Counter(fine2coarse.values())
print(f"coarse faces used: {len(cnt)}  min/max fine-faces per coarse face: "
      f"{min(cnt.values())}/{max(cnt.values())}")
