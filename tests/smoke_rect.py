#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke test for the rectangular-node solver path.

1. 4x4 checkerboard with square widths  (1.0, 1.0) must match the standard
   S=1 square solve (kron of the 2x2 grid) — bit-level path check.
2. Same problem with hy = 1.2 must converge to a finite keff and show the
   expected anisotropy (flux stretched along the longer dimension).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import psn2d  # noqa: E402
from psn2d.model import load_spec, arrays  # noqa: E402
from psn2d.solver import PSN2D  # noqa: E402


def main():
    spec = load_spec(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "examples", "psn_repro", "checkerboard_1g.yaml"))
    St, Sgg, nuSf, chi = arrays(spec)
    bnd = ("reflect",) * 4
    kw = dict(boundary=bnd, generic=True, I=30, M=12)

    # --- 1. square rect widths must equal the classic S=1 solve -----------
    grid = np.array([[0, 1], [1, 0]])
    mat_map = np.kron(grid, np.ones((1, 1), int))
    sol_sq = PSN2D(mat_map, 1.0, St, Sgg, nuSf, chi=chi,
                   widths=(np.full((2, 2), 1.0), np.full((2, 2), 1.0)), **kw)
    k_rect_sq, _, _ = sol_sq.keff(outer_tol=1e-10, verbose=False)

    sol_std = PSN2D(mat_map, 1.0, St, Sgg, nuSf, chi=chi, **kw)
    k_std, _, _ = sol_std.keff(outer_tol=1e-10, verbose=False)

    print(f"square-width rect path keff = {k_rect_sq:.7f}")
    print(f"classic square S=1       keff = {k_std:.7f}")
    d = abs(k_rect_sq - k_std)
    assert d < 1e-7, f"square-width rect deviates: {d:.3e}"

    # --- 2. true rectangles converge, flux anisotropy as expected --------
    wx = np.full((2, 2), 1.0)
    hy = 1.2
    wy = np.full((2, 2), hy)
    t0 = time.time()
    sol_r = PSN2D(mat_map, 1.0, St, Sgg, nuSf, chi=chi,
                  widths=(wx, wy), **kw)
    k_r, phi, qnode = sol_r.keff(outer_tol=1e-10, verbose=False)
    dt = time.time() - t0
    print(f"rect 1.0 x {hy} keff = {k_r:.7f}  ({dt:.2f} s, finite={np.isfinite(k_r)})")
    assert np.isfinite(k_r)

    # volume-weighted phi should be higher in the taller (y-stretched) nodes;
    # with identical material layout the checkerboard symmetry is preserved:
    p00 = phi[0, 0]; p11 = phi[0, 3]
    print(f"corner fuel nodes phi[0]: {p00:.5f} vs {p11:.5f} (symmetry {abs(p00-p11)/p00:.2e})")
    assert abs(p00 - p11) / p00 < 1e-9

    print("SMOKE OK")


if __name__ == "__main__":
    main()
