# -*- coding: utf-8 -*-
"""Break down the 60 micro-classes of the 17x17 split4 assembly by key
field (2026-09-14: user asked 'where do the 60 classes come from')."""
import os, sys
from collections import Counter

sys.path.insert(0, '/opt/data/workspace/OpenPSN')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

from psn2d import model, __main__ as M, highsweep

EX = '/opt/data/workspace/OpenPSN/examples/c5g7/study2_rectangular'
spec = model.load_spec(EX + '/c5g7_rect17_split4_uo2_assembly.yaml')
case = {'name': 'cls', 'model': 'ty3', 'I': 30, 'M': 24, 'subdivide': 1,
        'A': 3, 'closure': 'directional_matrix'}
spec['solver'].update({'keff_tol': 1e-7, 'max_outer': 1, 'threads': 1})
psn = M.build_solver(spec, case, threads=1, mem_limit_gb=32)
nl = highsweep.HoNodeLocal(psn, 0, 0, 3, 'directional', gs_tol=1e-7,
                           max_gs=1)

ncls = len(nl.mc_nodes)
print(f"nodes={psn.nodes}  ncls={ncls}")
print(f"material/shape groups ({len(psn._groups)}):")
for gi, (mat, shape) in enumerate(psn._groups):
    print(f"  grp{gi}: mat={mat} shape={shape}")

grp_c, mask_c, nbt_c, par_c, sg_c = (Counter() for _ in range(5))
combo = Counter()
for key, nlist in nl.mc_nodes.items():
    grp, mask, nbt, parity, sgk = key
    grp_c[grp] += len(nlist)
    mask_c[mask] += len(nlist)
    nbt_c[nbt] += len(nlist)
    par_c[parity] += len(nlist)
    sg_c[sgk] += len(nlist)
    combo[(grp, parity)] += len(nlist)

def cnt(c, label):
    print(f"{label}: {len(c)} distinct")
    for k, v in sorted(c.items(), key=lambda x: -x[1])[:12]:
        ks = k if len(str(k)) < 70 else str(k)[:67] + '...'
        print(f"   {v:5d}  {ks}")

cnt(grp_c, "by grp")
cnt(mask_c, "by face-mask")
cnt(nbt_c, "by neighbour-group tuple")
cnt(par_c, "by parity")
cnt(sg_c, "by sign-pattern")
print("\nby (grp, parity):")
for k, v in sorted(combo.items(), key=lambda x: (x[0][0], x[0][1])):
    print(f"   grp{k[0]} parity{k[1]}: {v} nodes")

sizes = sorted(len(v) for v in nl.mc_nodes.values())
print(f"\nclass sizes: min={sizes[0]} max={sizes[-1]} sum={sum(sizes)}")
print(f"classes with Nn=1: {sum(1 for s in sizes if s == 1)}")
