#!/usr/bin/env python3
"""rect4 pin geometry figure (2026-09-15 user request: "pin 的几何结构发图").

C5G7 17x17 assembly, every pin's fuel square split 2x2 (4 fuel blocks),
68x68 base grid, 21.42 cm.  Three panels:
  A: assembly schematic, one pin circled
  B: single pin S1 (4x4 blocks) with real dimensions
  C: single pin S2 (8x8, each block subdivided 2x2)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

import os
_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "paper", "figures", "rect4_pin_geometry.png")

# real widths from the yaml (cm)
W = 0.15143746025551064      # water border block
F = 0.47856253974448937      # fuel half-block (fuel = 2F = 0.9571)
PITCH = W + 2 * F + W        # 1.26 cm
N = 17
TOTAL = N * PITCH            # 21.42 cm

WATER = "#cfe6f5"
FUEL = "#f2b263"
HL = "tab:red"

fig, (a1, a2, a3) = plt.subplots(
    1, 3, figsize=(16.5, 5.6),
    gridspec_kw={"width_ratios": [1.15, 1, 1]})
fig.suptitle("C5G7 17×17 assembly: every pin fuel square split 2×2 "
             "(68×68 base grid, 21.42 cm, all-reflective)",
             fontsize=14)

# ---- A: assembly schematic -------------------------------------------
hx, hy = 4, 7   # highlighted pin (col, row)
for px in range(N):
    for py in range(N):
        a1.add_patch(Rectangle((px * PITCH, py * PITCH), PITCH, PITCH,
                               facecolor=WATER, edgecolor="w", lw=0.6))
        m = (PITCH - 2 * F) / 2
        a1.add_patch(Rectangle((px * PITCH + m, py * PITCH + m), 2 * F, 2 * F,
                               facecolor=FUEL, edgecolor="none"))
a1.add_patch(Rectangle((hx * PITCH, hy * PITCH), PITCH, PITCH,
                       facecolor="none", edgecolor=HL, lw=2.2, ls="--"))
a1.set_xlim(-0.3, TOTAL + 0.3)
a1.set_ylim(-0.3, TOTAL + 0.3)
a1.set_aspect("equal")
a1.set_title("A  assembly: 17×17 pins (one circled)", fontsize=12)
a1.set_xlabel("x (cm)")
a1.set_ylabel("y (cm)")
a1.grid(alpha=0.0)
for t in (a1.xaxis, a1.yaxis):
    t.set_visible(False)

# ---- B: single pin S1 (4x4 blocks) -----------------------------------
blk = [W, F, F, W]
cw, ch = sum(blk), sum(blk)
for ix, w in enumerate(blk):
    for iy, h in enumerate(blk):
        fuel = ix in (1, 2) and iy in (1, 2)
        a2.add_patch(Rectangle((sum(blk[:ix]), sum(blk[:iy])), w, h,
                               facecolor=FUEL if fuel else WATER,
                               edgecolor="#5b7b93", lw=0.8))
# dimension lines (bottom: x, right: y)
y0 = -0.34
a2.add_patch(FancyArrowPatch((0, y0), (W, y0), arrowstyle="<|-|>",
                             mutation_scale=9, lw=1.0, color="k"))
a2.add_patch(FancyArrowPatch((W, y0), (3 * W + F, y0), arrowstyle="<|-|>",
                             mutation_scale=9, lw=1.0, color="k"))
a2.add_patch(FancyArrowPatch((3 * W + F, y0), (cw, y0), arrowstyle="<|-|>",
                             mutation_scale=9, lw=1.0, color="k"))
a2.text(W / 2, y0 + 0.13, "0.151", ha="center", va="bottom", fontsize=8.5)
a2.text(3 * W + F + W / 2, y0 + 0.13, "0.151", ha="center", va="bottom",
        fontsize=8.5)
a2.text((W + 3 * W + F) / 2, y0 - 0.13, "0.957 (= 2×0.479)", ha="center",
        va="top", fontsize=8.5)
a2.add_patch(FancyArrowPatch((cw + 0.34, 0), (cw + 0.34, cw),
                             arrowstyle="<|-|>", mutation_scale=9,
                             lw=1.0, color="k"))
a2.text(cw + 0.44, cw / 2, "1.26 (pitch)", ha="left", va="center",
        fontsize=8.5, rotation=90)
a2.text(cw / 2, ch / 2, "fuel\n2×2", ha="center", va="center", fontsize=9,
        color="#7a4a12")
a2.text(W / 2, ch - W / 2, "", fontsize=7)
a2.set_xlim(-0.5, cw + 1.35)
a2.set_ylim(-0.75, ch + 0.15)
a2.set_aspect("equal")
a2.set_title("B  one pin, S1: 4×4 blocks\n[water | fuel/2 | fuel/2 | water] × 2",
             fontsize=11)
for t in (a2.xaxis, a2.yaxis):
    t.set_visible(False)

# ---- C: single pin S2 (8x8, each block halved) ------------------------
blk2 = []
for v in blk:
    blk2 += [v / 2, v / 2]
cw2 = sum(blk2)
for ix, w in enumerate(blk2):
    for iy, h in enumerate(blk2):
        fuel = (ix // 2) in (1, 2) and (iy // 2) in (1, 2)
        a3.add_patch(Rectangle((sum(blk2[:ix]), sum(blk2[:iy])), w, h,
                               facecolor=FUEL if fuel else WATER,
                               edgecolor="#5b7b93",
                               lw=0.8 if (ix % 2) or (iy % 2) else 0.4))
a3.set_xlim(-0.15, cw2 + 0.15)
a3.set_ylim(-0.15, cw2 + 0.15)
a3.set_aspect("equal")
a3.set_title("C  one pin, S2: 8×8 blocks\n(each S1 block subdivided 2×2)",
             fontsize=11)
for t in (a3.xaxis, a3.yaxis):
    t.set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(_OUT, dpi=160, bbox_inches="tight")
print("saved", OUT)
