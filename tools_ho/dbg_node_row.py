# -*- coding: utf-8 -*-
"""dbg_node_row.py — per-node row identity between the local GS system
and the global BlockAngleSystem matrix.  For node n build the local
(d x d) system exactly as highsweep does (for a SINGLE node), and check
every (local row, local col) entry against G[global_row, global_col]."""
import sys
import numpy as np

sys.path.insert(0, "/opt/data/workspace/OpenPSN")
from psn2d import highorder as ho
from psn2d import __main__ as PM

EX = '/opt/data/workspace/OpenPSN/examples/c5g7/study2_rectangular'


def build_psn(key='uo2', MM=2):
    from psn2d import model as M
    spec = M.load_spec(EX + f'/c5g7_rect_{key}_assembly.yaml')
    case = {'name': f'M{MM}', 'model': 'ty3', 'I': 30, 'M': MM,
            'subdivide': 1, 'A': 3, 'closure': 'directional_matrix'}
    spec['solver'].update({'keff_tol': 1e-10, 'max_outer': 10,
                           'threads': 1})
    psn = PM.build_solver(spec, case, threads=1, mem_limit_gb=32)
    if not hasattr(psn, '_groups'):
        psn._vectorize_setup()
    return psn


psn = build_psn()

S = 1e-13


def colpos(nl, n, lf, mode):
    t = nl.nf[n][lf][0]
    f = nl.psn.node_fmap[n][lf][0]
    pos = nl.pair_pos[(f, mode)] if t == "r" else mode
    return nl.blockoff[n, lf] + pos


def gcol_of(nl, n, lf, mode):
    t = nl.nf[n][lf][0]
    f = nl.psn.node_fmap[n][lf][0]
    pos = nl.pair_pos[(f, mode)] if t == "r" else mode
    c = nl.colg[n, lf, pos if t == "r" else mode]
    return c * nl.K + (mode if t != "r" else mode)


def main():
    from psn2d.highsweep import HoNodeLocal, _FB, _SOWN
    global psn
    i, g = 0, 0
    B = ho.BlockAngleSystem(psn, i, g)
    G = B.matrix.toarray()
    K, M = B.K, psn.M
    nl = HoNodeLocal(psn, i, g, 3, 'directional', verbose=False)

    # global rows per (face, m, a): row index in G
    rows = {}
    r = 0
    for f, face in enumerate(psn.faces):
        if face["type"] == "internal":
            for m in range(M):
                for a in range(K):
                    rows[("i", f, m, a)] = r
                    r += 1
        elif face["type"] == "vacuum":
            for m in range(M):
                for a in range(K):
                    rows[("v", f, m, a)] = r
                    r += 1
        else:
            for pos, (ma, mb) in enumerate(psn.face_pairs[f]):
                for a in range(K):
                    rows[("r", f, pos, a)] = r
                    r += 1
    assert r == G.shape[0]

    total_bad = 0
    for n in range(psn.nodes):
        # local layout
        d = int(nl.d[n])
        L = nl.L[nl.node_grp_idx[n]]
        sg = nl.sg[n]
        # A_local
        A = np.zeros((d, d))
        src = np.zeros((d, 5))
        for lf in range(4):
            info = nl.nf[n][lf]
            t = info[0]
            nrow = (len(info[1]) if t == "r" else M) * K
            r0 = int(nl.blockoff[n, lf]) * K
            for r in range(nrow):
                row = r0 + r
                a = r % K
                if t == "i":
                    m = r // K
                    sigma = info[3]
                    for lf2 in range(4):
                        cp = colpos(nl, n, lf2, m) * K
                        A[row, cp:cp + K] += sigma * L[m, lf, a,
                                                     lf2 * K:(lf2 + 1) * K] \
                            * sg[m, lf2]
                    src[row] = -sigma * L[m, lf, a, 4 * K:4 * K + 5]
                    # shared-face neighbour diagonal
                    fb = _FB[lf]
                    cp2 = colpos(nl, n, lf, m) * K
                    A[row, cp2:cp2 + K] += \
                        -sigma * nl.L[nl.node_grp_idx[info[1]],
                                      m, fb, a, fb * K:(fb + 1) * K] \
                        * _SOWN[fb]
                    gkey = ("i", nl.psn.node_fmap[n][lf][0], m, a)
                elif t == "r":
                    ma = info[1][r // K][0]
                    mb = info[1][r // K][1]
                    for (mx, sx) in ((ma, 1.0), (mb, -nl.parity[a])):
                        for lf2 in range(4):
                            cp = colpos(nl, n, lf2, mx) * K
                            A[row, cp:cp + K] += sx * L[mx, lf, a,
                                                       lf2 * K:(lf2 + 1) * K] \
                                * sg[mx, lf2]
                        src[row] += -sx * L[mx, lf, a, 4 * K:4 * K + 5]
                    gkey = ("r", nl.psn.node_fmap[n][lf][0], r // K, a)
                else:
                    axis = info[1]
                    m = r // K
                    Bm = nl.Bmat[axis][m]
                    for b2 in range(K):
                        sc = Bm[a, b2]
                        if sc == 0.0:
                            continue
                        for lf2 in range(4):
                            cp = colpos(nl, n, lf2, m) * K
                            A[row, cp:cp + K] += sc * L[m, lf, b2,
                                                       lf2 * K:(lf2 + 1) * K] \
                                * sg[m, lf2]
                        src[row] += -sc * L[m, lf, b2, 4 * K:4 * K + 5]
                    A[row, colpos(nl, n, lf, m) * K + a] += \
                        -nl.psn.node_fmap[n][lf][1]
                    gkey = ("v", nl.psn.node_fmap[n][lf][0], m, a)
                gr = rows[gkey]
                # compare row: local col lc = (blk, slot) with blk = lc//K
                for lc in range(d):
                    blk = lc // K
                    slot = lc % K
                    lf2 = None
                    for l2 in range(4):
                        t2 = nl.nf[n][l2][0]
                        bo = int(nl.blockoff[n, l2])
                        nblk = len(nl.nf[n][l2][1]) if t2 == "r" else M
                        if bo <= blk < bo + nblk:
                            lf2 = l2
                            break
                    f2 = nl.psn.node_fmap[n][lf2][0]
                    bo = int(nl.blockoff[n, lf2])
                    rel = blk - bo
                    if nl.nf[n][lf2][0] == "r":
                        mode = nl.nf[n][lf2][1][rel][0]  # master
                    else:
                        mode = rel
                    c, es = nl.psn._col_of(f2, mode)
                    gc = c * K + slot
                    aval = A[row, lc]
                    gval = G[gr, gc]
                    if abs(aval - gval) > S * max(1.0, abs(gval), abs(aval)):
                        total_bad += 1
                        if total_bad <= 15:
                            print(f"node {n}: local row {row} ({gkey}) "
                                  f"col lc={lc} (lf2={lf2},mode={mode} "
                                  f"slot={slot} gcol={gc}): "
                                  f"A={aval!r} G={gval!r}")
    print("total coefficient mismatches:", total_bad)


if __name__ == "__main__":
    main()
