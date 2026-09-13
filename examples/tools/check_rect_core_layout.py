#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-check rect quarter-core layout vs the homogenized core grid."""
import os

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_EX = os.path.join(os.path.dirname(_HERE), "examples")
old = yaml.safe_load(open(os.path.join(_EX, "c5g7", "study1_homogenised", "c5g7_2d_quarter_core.yaml")))
new = yaml.safe_load(open(os.path.join(_EX, "c5g7", "study2_rectangular", "c5g7_rect_quarter_core.yaml")))
g_old = np.array(old["geometry"]["grid"], int)
g_new = np.array(new["geometry"]["mat_grid"], int)
wx = np.array(new["geometry"]["widths_x"])
wy = np.array(new["geometry"]["widths_y"])

# old: 0=UO2 1=MOX-7 2=MOX-8.7 3=MOX-4.3 4=GT 5=FC 6=water
# new pure: 0=water 1=uo2 2=mox43 3=mox7 4=mox87 5=gd 6=fc 7=gt
MAP = {0: 1, 1: 3, 2: 4, 3: 2, 4: 7, 5: 6, 6: 0}
g_old_mapped = np.vectorize(MAP.get)(g_old)

center_ok = 0
total = 0
mismatch = 0
for py in range(51):
    for px in range(51):
        yy, xx = 3 * py + 1, 3 * px + 1
        total += 1
        if g_new[yy, xx] == g_old_mapped[py, px]:
            center_ok += 1
        else:
            mismatch += 1
            if mismatch <= 5:
                print(f"MISMATCH pin({px},{py}) old={g_old[py, px]} new={g_new[yy, xx]}")
print(f"center nodes: {center_ok}/{total} match, {mismatch} mismatch")

noncenter = np.zeros_like(g_new, bool)
noncenter[:] = True
for py in range(51):
    for px in range(51):
        noncenter[3 * py + 1, 3 * px + 1] = False
assert (g_new[noncenter] == 0).all(), "non-center node not water!"
print("all non-center nodes = water: OK")

S = float(np.sqrt(np.pi * 0.54 ** 2))
W = (1.26 - S) / 2.0
assert set(np.unique(wx).tolist()) == {S, W}
assert set(np.unique(wy).tolist()) == {S, W}
assert abs(float((wx * wy).sum()) - 51 * 51 * 1.26 ** 2) < 1e-9
print("widths audit OK: only {S, W}, total area exact")
print("boundaries new:", new["boundaries"])
print("boundaries old:", old["boundaries"])
