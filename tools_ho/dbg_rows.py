# -*- coding: utf-8 -*-
"""dbg_rows.py — row-by-row comparison of microclass A vs global matrix."""
import sys
import numpy as np

sys.path.insert(0, "/opt/data/workspace/OpenPSN")
from psn2d.solver import build_solver
from psn2d.highsweep import NodeLineSweep
from psn2d import highorder as ho

psn = build_solver("examples/c5g7/study2_rectangular/c5g7_rect_uo2_assembly.yaml")
psn.M = 2
i, g = 0, 0
B = ho.BlockAngleSystem(psn, i, g)
G = B.A.toarray()
nc = B.ncol
nl = NodeLineSweep(psn, i, g, verbose=False)
print("ncol", nc, "nrows", G.shape[0])

bad = 0
for key, nodes in list(nl.mc_nodes.items())[:6]:
    data = nl._microclass(key)
    A = data["A"]
    nlist = data["nodes"]
    d = len(nlist)
    for ln in range(d):
        n = nlist[ln]
        lf = ln // (nl.M * nl.K)
        f = psn.node_fmap[n][lf][0]
        mrow = (ln % (nl.M * nl.K)) // nl.K
        b = (ln % (nl.M * nl.K)) % nl.K
        r0 = nl.face_row_start[f] * nl.K + mrow * nl.K + b
        for lc in range(d):
            gcol = nl._local_colpos(n, nl._lf_of_lc(n, lc) if False else None, None) if False else None
    # instead: compare each local column via explicit col positions
    for lc in range(d):
        # find which (lf2, mode) this local col is
        pass
    # do it properly:
    for lc in range(d):
        colmap = data["colmap"][lc]  # (lf2, mode) or None(=q)
        if colmap is None:
            gcol = nc + n
        else:
            lf2, mode = colmap
            t = psn.nf[n][lf2][0]
            f2 = psn.node_fmap[n][lf2][0]
            if t == "r":
                gcol = nl.pair_pos[(f2, mode)]
            else:
                gcol = nl._col_of(f2, mode)
        # local row index for comparison
        for lr in range(d):
            lfl = lr // (nl.M * nl.K)
            mrl = (lr % (nl.M * nl.K)) // nl.K
            brl = (lr % (nl.M * nl.K)) % nl.K
            # global row of this local row: face f_of(lfl), mode mrl, b brl
            fl = psn.node_fmap[n][lfl][0]
            rl = nl.face_row_start[fl] * nl.K + mrl * nl.K + brl
            aval = A[lr, lc]
            gval = G[rl, gcol]
            if abs(aval - gval) > 1e-13 * max(1.0, abs(gval)):
                if bad < 25:
                    print(f"MISMATCH key={key} n={n} local row lr={lr} (lf={lfl},m={mrl},b={brl}) "
                          f"local col lc={lc} (colmap={colmap}): A={aval!r} G={gval!r}")
                bad += 1
print("total mismatches in first 6 classes:", bad)
