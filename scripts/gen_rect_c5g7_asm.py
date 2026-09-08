#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the C5G7 rectangular-node test problem (t4).

Single UO2 assembly (17x17 pins, all-reflective) with PURE material XS:
  - each pin (pitch 1.26 cm) is subdivided 3x3:
      center node  s x s  fuel (s = sqrt(pi*0.54^2) = 0.957107 cm,
                     equal area to the r = 0.54 cm circle)
      4 edge nodes s x w / w x s  water
      4 corner nodes w x w        water,  w = 1.26 - s = 0.1514465 cm
  - corners meet corners only; the 17x17 pin grid -> 51x51 node grid.
  - pin layout (UO2 / Guide Tube x12 / Fission Chamber center) is copied
    bit-for-bit from examples/c5g7_uo2_assembly.yaml (verified vs the
    official C5G7 UO2 rodded template).

XS: examples/c5g7_materials_pure.yaml (pure, UNhomogenized 7-group XS from
c5g7-mgxs.h5: water, uo2, mox43, mox7, mox87, gd, fc, gt).
"""
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(HERE, "examples")

PITCH = 1.26
R = 0.54
S_FUEL = float(np.sqrt(np.pi * R * R))        # 0.9571068 cm
W_GAP = (PITCH - S_FUEL) / 2.0                # 0.1514465 cm (w + s + w = pitch)

# 17x17 pin fuel map, copied from c5g7_uo2_assembly.yaml grid
# (0=UO2, 4=Guide Tube, 5=Fission Chamber; first row = y-)
PIN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 0, 0, 0],
    [0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 4, 0, 0, 4, 0, 0, 5, 0, 0, 4, 0, 0, 4, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
    [0, 0, 0, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])

# pure-material indices (c5g7_materials_pure.yaml order)
PURE = {"water": 0, "uo2": 1, "mox43": 2, "mox7": 3, "mox87": 4,
        "gd": 5, "fc": 6, "gt": 7}
PIN_MAT = {0: PURE["uo2"], 4: PURE["gt"], 5: PURE["fc"]}


def build():
    # node grid indexed mat_grid[yy, xx] (matches solver mat[j_idx, i_idx]).
    # pin (py, px): y-pin index py, x-pin index px.
    # node y index = 3*py + sy, x index = 3*px + sx.
    # x-width depends on sx (1 = center fuel column), y-width on sy.
    n = 3 * 17
    mat_grid = np.zeros((n, n), int)   # [yy, xx]
    wx = np.empty((n, n))
    wy = np.empty((n, n))
    for py in range(17):
        for px in range(17):
            matc = PIN_MAT[int(PIN[py, px])]
            for sy in range(3):
                for sx in range(3):
                    yy = 3 * py + sy
                    xx = 3 * px + sx
                    is_fuel = (sx == 1 and sy == 1)
                    mat_grid[yy, xx] = matc if is_fuel else PURE["water"]
                    wx[yy, xx] = S_FUEL if sx == 1 else W_GAP
                    wy[yy, xx] = S_FUEL if sy == 1 else W_GAP
    # geometry audit: 289 fuel squares at pin centers, rest water;
    # total area exactly 17x17 pin pitches; fuel area 289*pi*r^2
    n_fuel = int((mat_grid > 0).sum())
    assert n_fuel == 289
    assert np.all(mat_grid[1::3, 1::3] > 0)          # every pin center is fuel
    total = float((wx * wy).sum())
    fuel = float((wx * wy)[mat_grid > 0].sum())
    assert abs(total - 17 * 17 * PITCH ** 2) < 1e-9
    assert abs(fuel - 289 * np.pi * R ** 2) < 1e-9
    print(f"audit: {n_fuel} fuel nodes, total {total:.6f} cm2 "
          f"({17 * 17 * PITCH ** 2:.6f}), fuel {fuel:.4f} "
          f"({289 * np.pi * R ** 2:.4f})")
    return mat_grid, wx, wy


def rows(arr):
    return [[float(v) for v in row] for row in arr]


def main():
    mat_grid, wx, wy = build()
    spec = {
        "problem": "c5g7_rect_uo2_assembly",
        "description": ("C5G7 single UO2 assembly, 51x51 rectangular nodes "
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
            {"name": "M8", "M": 8, "kref": 1.3411780},
            {"name": "M12", "M": 12, "kref": 1.3411780},
            {"name": "M16", "M": 16, "kref": 1.3411780},
        ],
    }
    out = os.path.join(EX, "c5g7_rect_uo2_assembly.yaml")
    with open(out, "w") as f:
        f.write("# PSN2D — C5G7 single UO2 assembly, rectangular nodes, PURE XS\n"
                f"# 51x51 nodes (17x17 pins x 3x3); fuel s={S_FUEL:.7f} cm "
                f"(equal area to r={R} cm); water gap w={W_GAP:.7f} cm\n"
                "# XS: c5g7_materials_pure.yaml (unhomogenized, c5g7-mgxs.h5)\n"
                "# kref 1.3411780 = OpenMOC equal-HOMOGENIZED 17x17 "
                "(area-weighted, reflective) — geometry effect expected to be "
                "small vs this\n")
        yaml.safe_dump(spec, f, sort_keys=False)
    print(f"wrote {out}")
    print(f"mat_grid {mat_grid.shape}, unique mats {np.unique(mat_grid)}")
    print(f"s={S_FUEL:.7f} w={W_GAP:.7f} s+w*2={S_FUEL + 2*W_GAP:.7f} "
          f"(pitch {PITCH})")


if __name__ == "__main__":
    main()
