#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the C5G7-2D 1/4 CORE rectangular-node test problem.

Layout (same as examples/c5g7_2d_quarter_core.yaml, pin-level):
  51x51 pin grid; 2x2 fuel block (UO2 main-diagonal, MOX anti-diagonal)
  at bottom-left (core center), L-shaped pure-water reflector on
  right + top.  Mirror on left/bottom, vacuum on right/top.
  Reference: MCNP5 7-group 1.18646, nTRACER 1.18653.

Each pin is subdivided 3x3 (same recipe as the single-assembly problem):
  center node s x s  = pin material (uo2/mox43/mox7/mox87/gt/fc/water)
  edge nodes  s x w / w x s = water
  corner nodes w x w = water,  w = (1.26 - s)/2, s = sqrt(pi*0.54^2)

Node grid: 153x153.  Assembly templates are the same 17x17 rodded
templates as gen_c5g7_yaml.py (UO2 / MOX), copied from OpenMOC lattices.
"""
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(HERE, "examples")

PITCH = 1.26
R = 0.54
S_FUEL = float(np.sqrt(np.pi * R * R))        # 0.9571068 cm
W_GAP = (PITCH - S_FUEL) / 2.0                # 0.1514465 cm

# 17x17 templates (bottom row first) — same as gen_c5g7_yaml.py
UO2 = [
    "u" * 17,
    "u" * 17,
    "u u u u u g u u g u u g u u u u u".split(),
    "u u u g u u u u u u u u u g u u u".split(),
    "u" * 17,
    "u u g u u g u u g u u g u u g u u".split(),
    "u" * 17,
    "u" * 17,
    "u u g u u g u u f u u g u u g u u".split(),
    "u" * 17,
    "u" * 17,
    "u u g u u g u u g u u g u u g u u".split(),
    "u" * 17,
    "u u u g u u u u u u u u u g u u u".split(),
    "u u u u u g u u g u u g u u u u u".split(),
    "u" * 17,
    "u" * 17,
]
MOX = []
for r in range(17):
    if r == 0 or r == 16:
        MOX.append("m" * 17)
    elif r == 1 or r == 15:
        MOX.append(("m " + "o " * 15 + "m").split())
    elif r == 2 or r == 14:
        MOX.append("m o o o o g o o g o o g o o o o m".split())
    elif r == 3 or r == 13:
        MOX.append("m o o g o x x x x x x x o g o o m".split())
    elif r == 4 or r == 12:
        MOX.append("m o o o x x x x x x x x x o o o m".split())
    elif r == 5 or r == 11:
        MOX.append("m o g x x g x x g x x g x x g o m".split())
    elif r in (6, 7):
        MOX.append("m o o x x x x x x x x x x x o o m".split())
    elif r == 8:
        MOX.append("m o g x x g x x f x x g x x g o m".split())
    elif r in (9, 10):
        MOX.append("m o o x x x x x x x x x x x o o m".split())

# pure-material indices (c5g7_materials_pure.yaml order)
PURE = {"water": 0, "uo2": 1, "mox43": 2, "mox7": 3, "mox87": 4,
        "gd": 5, "fc": 6, "gt": 7}
PIN_MAT = {"u": PURE["uo2"], "o": PURE["mox7"], "x": PURE["mox87"],
           "m": PURE["mox43"], "g": PURE["gt"], "f": PURE["fc"]}


def quarter_core_pins():
    """51x51 pin map, row 0 = y- (bottom, mirror plane)."""
    grid: list = [["p"] * 51 for _ in range(51)]
    def block(asm, r0, c0):
        for i in range(17):
            for j in range(17):
                grid[r0 + i][c0 + j] = asm[i][j]
    block(UO2, 0, 0)
    block(MOX, 17, 0)
    block(MOX, 0, 17)
    block(UO2, 17, 17)
    return grid


def build():
    pin = quarter_core_pins()
    n = 3 * 51                       # 153
    mat_grid = np.zeros((n, n), int)   # [yy, xx]
    wx = np.empty((n, n))
    wy = np.empty((n, n))
    counts = {}
    for py in range(51):
        for px in range(51):
            ch = pin[py][px]
            matc = PURE["water"] if ch == "p" else PIN_MAT[ch]
            for sy in range(3):
                for sx in range(3):
                    yy = 3 * py + sy
                    xx = 3 * px + sx
                    is_center = (sx == 1 and sy == 1)
                    mat_grid[yy, xx] = matc if is_center else PURE["water"]
                    wx[yy, xx] = S_FUEL if sx == 1 else W_GAP
                    wy[yy, xx] = S_FUEL if sy == 1 else W_GAP
                    if is_center:
                        counts[matc] = counts.get(matc, 0) + 1
    # audits
    assert np.allclose(wx * wy, (wx * wy), equal_nan=True)
    total = float((wx * wy).sum())
    assert abs(total - 51 * 51 * PITCH ** 2) < 1e-9, total
    fuel_area = 0.0
    n_center = 0
    for py in range(51):
        for px in range(51):
            if pin[py][px] != "p":
                n_center += 1
                fuel_area += np.pi * R * R if pin[py][px] in "uoxm" else np.pi * R * R
    print(f"audit: {n_center} center (non-water) nodes, "
          f"per-material counts {counts}")
    print(f"audit: total area {total:.6f} cm2 (51x51 pin = "
          f"{51*51*PITCH**2:.6f})")
    return mat_grid, wx, wy, counts


def rows(arr):
    return [[float(v) for v in row] for row in arr]


def main():
    mat_grid, wx, wy, counts = build()
    spec = {
        "problem": "c5g7_rect_quarter_core",
        "description": ("C5G7-2D 1/4 core, 153x153 rectangular nodes "
                        "(51x51 pins x 3x3), PURE material XS, "
                        "mirror L/bottom, vacuum R/top"),
        "geometry": {
            "rect": True,
            "mat_grid": mat_grid.tolist(),
            "widths_x": rows(wx),
            "widths_y": rows(wy),
        },
        "boundaries": {"left": "reflect", "bottom": "reflect",
                       "right": "vacuum", "top": "vacuum"},
        "materials_file": "c5g7_materials_pure.yaml",
        "angular": {"model": "ty3", "M": 12},
        "solver": {"keff_tol": 1e-10, "max_outer": 6000},
        "cases": [
            {"name": "M8", "M": 8, "kref": 1.18646},
            {"name": "M12", "M": 12, "kref": 1.18646},
            {"name": "M16", "M": 16, "kref": 1.18646},
        ],
    }
    out = os.path.join(EX, "c5g7_rect_quarter_core.yaml")
    with open(out, "w") as f:
        f.write("# PSN2D — C5G7-2D 1/4 core, rectangular nodes, PURE XS\n"
                f"# 153x153 nodes (51x51 pins x 3x3); fuel center s={S_FUEL:.7f} cm\n"
                f"# (equal area to r={R} cm); water gap w={W_GAP:.7f} cm\n"
                "# 2x2 fuel (UO2 diag / MOX anti-diag) + L water reflector\n"
                "# XS: c5g7_materials_pure.yaml (unhomogenized, c5g7-mgxs.h5)\n"
                "# kref 1.18646 = MCNP5 7-group (C5G7-TD Part I 2D steady)\n")
        yaml.safe_dump(spec, f, sort_keys=False)
    print(f"wrote {out}")
    print(f"mat_grid {mat_grid.shape}, unique mats {np.unique(mat_grid)}")
    nshape = set()
    for yy in range(mat_grid.shape[0]):
        for xx in range(mat_grid.shape[1]):
            nshape.add((mat_grid[yy, xx], round(wx[yy, xx], 6), round(wy[yy, xx], 6)))
    print(f"(mat, wx, wy) shape classes: {len(nshape)}")


if __name__ == "__main__":
    main()
