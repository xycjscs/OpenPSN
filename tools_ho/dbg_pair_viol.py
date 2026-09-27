# -*- coding: utf-8 -*-
"""Find the EXACT (i, g, key, pp) where _pair_microclass's off-block
check fires, and print the offending A entries with their local-block
coordinates.  Run over every (i, g) to catch (i,g)-dependence."""
import os, sys
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, '/opt/data/workspace/OpenPSN')
import numpy as np
from psn2d import model
from psn2d import __main__ as M
from psn2d.highsweep import HoNodeLocal

EX = '/opt/data/workspace/OpenPSN/examples/c5g7/study2_rectangular'
spec = model.load_spec(EX + '/c5g7_rect_uo2_assembly.yaml')
case = {'name': 'M4_S1', 'model': 'ty3', 'I': 30, 'M': 4, 'subdivide': 1,
        'kref': 1.3347260, 'A': 3, 'closure': 'directional_matrix'}
spec['solver'].update({'keff_tol': 1e-10, 'max_outer': 6000, 'threads': 1})
psn = M.build_solver(spec, case, threads=1, mem_limit_gb=120)
p = psn
K = 4
Mm = 4


def block_of(nl, rep, local):
    """Return (lf, pos) for local index `local` (in K units)."""
    off = 0
    for lf in range(4):
        t = nl.nf[rep][lf][0]
        nblk = len(nl.nf[rep][lf][1]) if t == 'r' else Mm
        if local < off + nblk:
            return lf, local - off
        off += nblk
    return None, None


def owner(nl, rep, local):
    lf, pos = block_of(nl, rep, local)
    t = nl.nf[rep][lf][0]
    if t == 'r':
        a, b = nl.nf[rep][lf][1][pos]
        return min(a, b)
    return min(pos, Mm - 1 - pos)


for i in range(p.I):
    for g in range(p.ng):
        nl = HoNodeLocal(p, i, g, 3, 'directional', gs_tol=1e-11, max_gs=1)
        orders = nl._mc_order()
        keys = []
        for pty in (0, 1):
            for key in orders[pty][0] + orders[pty][1]:
                if key not in keys:
                    keys.append(key)
        for key in keys:
            data = nl._microclass(key)
            A = data['A']
            rep = data['rep']
            d = A.shape[0]
            # per-pair off-block test, exactly as _pair_microclass does
            for pp in range(Mm // 2):
                sel = nl._pair_sel(rep, pp)
                offmask = np.zeros_like(A, bool)
                offmask[sel, :] = True
                offmask[:, sel] = False
                bad = A[offmask]
                w = float(np.abs(bad).max()) if bad.size else 0.0
                if w > 0.0:
                    print(f'VIOLATION i={i} g={g} pp={pp} key={key[:4]} '
                          f'rep={rep} d={d} worst={w:.3e}')
                    # print up to 6 offending entries with block coords
                    rows, cols = np.where(offmask & (np.abs(A) > 0))
                    for (r, c) in list(zip(rows, cols))[:6]:
                        lf_r, pos_r = block_of(nl, rep, r // K)
                        lf_c, pos_c = block_of(nl, rep, c // K)
                        print(f'   A[r={r},c={c}]={A[r,c]:+.3e}  '
                              f'row(lf{lf_r},pos{pos_r},own'
                              f'{owner(nl,rep,r//K)}) '
                              f'col(lf{lf_c},pos{pos_c},own'
                              f'{owner(nl,rep,c//K)})')
                    break
            # stop after first pp with violation for this key
            if False:
                break
print('SCAN DONE')
