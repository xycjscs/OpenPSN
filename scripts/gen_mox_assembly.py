#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the C5G7 MOX assembly problems (both study types).

1) Study I (pinwise BWW homogenised, square nodes):
   examples/c5g7_mox_assembly_bww.yaml
   17x17 grid, unit_size 1.26, all-reflective, materials from
   c5g7_materials_bww_om.yaml (pinwise flux-weighted self-shielded XS,
   same BWW library as the UO2 sweep).  Grid indices follow the UO2
   BWW yaml convention: 0=UO2 1=MOX-7% 2=MOX-8.7% 3=MOX-4.3%
   4=Guide Tube 5=Fission Chamber 6=Water.

2) Study II (rectangular nodes, PURE material XS):
   examples/c5g7_rect_mox_assembly.yaml
   51x51 nodes (17x17 pins x 3x3), same equal-area-square recipe as
   gen_rect_c5g7_asm.py, PURE 7-group XS (c5g7_materials_pure.yaml),
   all-reflective.  Pure indices: 0=water 1=uo2 2=mox43 3=mox7
   4=mox87 5=gd 6=fc 7=gt.
"""
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(HERE, "examples")

# 17x17 MOX template (bottom row first) — same as gen_c5g7_yaml.py
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
for row in MOX:
    assert len(row) == 17, (len(row), row)

# ---- Study I: BWW homogenised grid (same index convention as UO2 BWW) ----
BWW = {"u": 0, "o": 1, "x": 2, "m": 3, "g": 4, "f": 5, "p": 6}


def mox_grid_bww():
    return [[BWW[c] for c in MOX[i]] for i in range(17)]


def dump_grid(grid):
    return "\n".join("    - [" + ", ".join(str(v) for v in row) + "]"
                     for row in grid)


def write_bww():
    grid = mox_grid_bww()
    txt = f"""# PSN2D — C5G7 single MOX assembly (BWW flux-weighted self-shielded pin XS)
# (17x17, reflective, pin-homogenized) (7-group, c5g7-mgxs.h5 family)
#
# one 17x17 MOX assembly; all-reflective boundary; pin = fuel(r=0.54) + water ring.
# XS: c5g7_materials_bww_om.yaml (pinwise BWW self-shielded, same library as
# the UO2 BWW sweep).
problem: c5g7_mox_assembly_bww
description: C5G7 single MOX assembly, 7-group PSN2D, BWW pinwise XS.

geometry:
  grid:
{dump_grid(grid)}
  unit_size: 1.26
  subdivide: 1

boundaries:
  left: reflect
  bottom: reflect
  right: reflect
  top: reflect

materials_file: c5g7_materials_bww_om.yaml

angular:
  model: ty3
  M: 12

solver:
  keff_tol: 1e-10
  max_outer: 6000
"""
    out = os.path.join(EX, "c5g7_mox_assembly_bww.yaml")
    open(out, "w").write(txt)
    print("wrote", out)


# ---- Study II: rect pure-XS (51x51 nodes) ----
PITCH = 1.26
R = 0.54
S_FUEL = float(np.sqrt(np.pi * R * R))
W_GAP = (PITCH - S_FUEL) / 2.0

PURE = {"water": 0, "uo2": 1, "mox43": 2, "mox7": 3, "mox87": 4,
        "gd": 5, "fc": 6, "gt": 7}
PIN_MAT = {"u": PURE["uo2"], "o": PURE["mox7"], "x": PURE["mox87"],
           "m": PURE["mox43"], "g": PURE["gt"], "f": PURE["fc"]}


def build_rect():
    n = 3 * 17
    mat_grid = np.zeros((n, n), int)   # [yy, xx]
    wx = np.empty((n, n))
    wy = np.empty((n, n))
    counts = {}
    for py in range(17):
        for px in range(17):
            matc = PIN_MAT[MOX[py][px]]
            for sy in range(3):
                for sx in range(3):
                    yy = 3 * py + sy
                    xx = 3 * px + sx
                    is_fuel = (sx == 1 and sy == 1)
                    mat_grid[yy, xx] = matc if is_fuel else PURE["water"]
                    wx[yy, xx] = S_FUEL if sx == 1 else W_GAP
                    wy[yy, xx] = S_FUEL if sy == 1 else W_GAP
                    if is_fuel:
                        counts[matc] = counts.get(matc, 0) + 1
    assert (mat_grid > 0).sum() == 289
    assert np.all(mat_grid[1::3, 1::3] > 0)
    total = float((wx * wy).sum())
    assert abs(total - 17 * 17 * PITCH ** 2) < 1e-9
    fuel = float((wx * wy)[mat_grid > 0].sum())
    assert abs(fuel - 289 * np.pi * R ** 2) < 1e-9
    print(f"audit: 289 fuel nodes, total {total:.6f} cm2, "
          f"fuel {fuel:.4f} cm2, per-material {counts}")
    return mat_grid, wx, wy


def rows(arr):
    return [[float(v) for v in row] for row in arr]


def write_rect():
    mat_grid, wx, wy = build_rect()
    spec = {
        "problem": "c5g7_rect_mox_assembly",
        "description": ("C5G7 single MOX assembly, 51x51 rectangular nodes "
                        "(17x17 pins x 3x3), PURE material XS, all-reflective"),
        "geometry": {
            "rect": True,
            "mat_grid": mat_grid.tolist(),
            "widths_x": rows(wx),
            "widths_y": rows(wy),
        },
        "boundaries": {"left": "reflect", "bottom": "reflect",
                       "right": "reflect", "top": "reflect"},
        "materials_file": "c5g7_materials_pure.yaml",
        "angular": {"model": "ty3", "M": 12},
        "solver": {"keff_tol": 1e-10, "max_outer": 6000},
        "cases": [
            {"name": "M8", "M": 8, "kref": None},
            {"name": "M12", "M": 12, "kref": None},
            {"name": "M16", "M": 16, "kref": None},
        ],
    }
    out = os.path.join(EX, "c5g7_rect_mox_assembly.yaml")
    with open(out, "w") as f:
        f.write("# PSN2D — C5G7 single MOX assembly, rectangular nodes, PURE XS\n"
                f"# 51x51 nodes (17x17 pins x 3x3); fuel s={S_FUEL:.7f} cm "
                f"(equal area to r={R} cm); water gap w={W_GAP:.7f} cm\n"
                "# XS: c5g7_materials_pure.yaml (unhomogenized, c5g7-mgxs.h5)\n")
        yaml.safe_dump(spec, f, sort_keys=False)
    print("wrote", out)


if __name__ == "__main__":
    write_bww()
    write_rect()
