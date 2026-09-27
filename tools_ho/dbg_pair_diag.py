# -*- coding: utf-8 -*-
"""Lever-1 foundation probe (fast, no pair-pass implementation):
  [1] node-local class matrix A is STRICTLY block-diagonal over mirror
      pairs {p, M-1-p}  (both the base class A and after the shared-face
      neighbour diagonal is added in _microclass)
  [2] memopt.mirror_pair_columns is a partition of ncol
If [1] holds, a pair-restricted red-black pass is exact (no cross-pair
coupling), so 21*(M/2) independent GMRES blocks are valid.
Run: python dbg_pair_diag.py
"""
import os, sys, time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, '/opt/data/workspace/OpenPSN')

import numpy as np
from psn2d import model
from psn2d import __main__ as M
from psn2d.highsweep import HoNodeLocal
from psn2d.memopt import mirror_pair_columns

EX = '/opt/data/workspace/OpenPSN/examples/c5g7/study2_rectangular'
MM = int(sys.argv[1]) if len(sys.argv) > 1 else 4
spec = model.load_spec(EX + '/c5g7_rect_uo2_assembly.yaml')
case = {'name': f'M{MM}_S1', 'model': 'ty3', 'I': 30, 'M': MM,
        'subdivide': 1, 'kref': 1.3347260, 'A': 3,
        'closure': 'directional_matrix'}
spec['solver'].update({'keff_tol': 1e-10, 'max_outer': 6000, 'threads': 1})
psn = M.build_solver(spec, case, threads=1, mem_limit_gb=120)
p = psn
print(f"nodes={p.nodes} ncol={p.ncol} M={p.M} K=4 full_2pi={p.full_2pi}")

nl = HoNodeLocal(p, 2, 0, 3, 'directional', gs_tol=1e-11, max_gs=1)
Mm = p.M

# owner of each local (lf,pos) block = mirror-pair id = min(m, M-1-m)
def block_owners(nl, rep):
    own = []
    for lf in range(4):
        t = nl.nf[rep][lf][0]
        nblk = len(nl.nf[rep][lf][1]) if t == 'r' else Mm
        for pos in range(nblk):
            if t == 'r':
                a, b = nl.nf[rep][lf][1][pos]
                own.extend([min(a, b)] * nl.K)
            else:
                own.extend([min(pos, Mm - 1 - pos)] * nl.K)
    return np.array(own)

# ---- [1] base class A block-diagonal over mirror pairs ----------------
worst = 0.0
ncls = 0
for grp in range(len(p._groups)):
    for n in range(p.nodes):
        if nl.node_grp_idx[n] != grp:
            continue
        mask = tuple(nl._mask_entry(n, lf) for lf in range(4))
        sgk = nl.sgkey[n]
        cls = nl._class_system(grp, mask, sgk)
        A = cls['A']
        own = block_owners(nl, cls['rep'])
        assert len(own) == A.shape[0], (len(own), A.shape)
        offmask = own[:, None] != own[None, :]
        worst = max(worst, float(np.abs(A[offmask]).max()))
        ncls += 1
print(f"[1a] base class A: {ncls} classes, off-pair-block max|A| = "
      f"{worst:.3e} {'(STRICT)' if worst == 0 else '(VIOLATION)'}")

# ---- [1b] full microclass A (with shared-face neighbour diagonal) ------
orders = nl._mc_order()
keys = []
for pty in (0, 1):
    for key in orders[pty][0] + orders[pty][1]:
        if key not in keys:
            keys.append(key)
worst = 0.0
for key in keys:
    data = nl._microclass(key)
    A = data['A']
    own = block_owners(nl, data['rep'])
    assert len(own) == A.shape[0]
    offmask = own[:, None] != own[None, :]
    worst = max(worst, float(np.abs(A[offmask]).max()))
print(f"[1b] microclass A (incl. neighbour diag): {len(keys)} micro-classes, "
      f"off-pair-block max|A| = {worst:.3e} "
      f"{'(STRICT)' if worst == 0 else '(VIOLATION)'}")

# ---- [2] pair-column partition -----------------------------------------
pcols = mirror_pair_columns(p)
tot = np.sort(np.concatenate(pcols))
ok = np.array_equal(tot, np.arange(p.ncol))
sizes = sorted(set(int(len(c)) for c in pcols))
print(f"[2] mirror_pair_columns: {len(pcols)} pairs (expect {Mm//2}), "
      f"partition of ncol={p.ncol}: {ok}, per-pair sizes {sizes}")

print("PROBE DONE", "OK" if worst == 0 and ok else "PROBLEM")
