#!/usr/bin/env python3
"""Checkerboard A=3 directional sweep — GENERIC I=30 angular set:
table + symlog figure, plus TY3-vs-I30 comparison at S=6/S=12.
"""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
WLOG = os.path.join(_HERE, "cb_i30_weak.log")
SLOG = os.path.join(_HERE, "cb_i30_strong.log")
WLOGX = os.path.join(_HERE, "cb_i30_weak_ext.log")
SLOGX = os.path.join(_HERE, "cb_i30_strong_ext.log")
WLOG3 = os.path.join(_HERE, "cb_a3_weak.log")
SLOG3 = os.path.join(_HERE, "cb_a3_strong.log")
WLOG3x = os.path.join(_HERE, "cb_a3_weak_ext.log")
SLOG3x = os.path.join(_HERE, "cb_a3_strong_ext.log")
OUT = os.path.join(_REPO, "paper", "figures", "checkerboard_a3_i30_sm.png")
M_LIST = [2, 4, 8, 12, 16, 24]
S_LIST = list(range(1, 21))


def parse(paths, tag):
    data = {}
    for path in paths:
        for ln in open(path):
            m = re.match(rf"{tag} M(\d+)_S(\d+) keff=([\d.]+).*dref=([+-][\d.]+)pcm",
                         ln)
            if not m:
                continue
            M, S, keff, dref = int(m.group(1)), int(m.group(2)), \
                float(m.group(3)), float(m.group(4))
            data.setdefault(S, {})[M] = (keff, dref)
    return data


weak = parse([WLOG, WLOGX], "CBW30")
strong = parse([SLOG, SLOGX], "CBS30")
weak3 = parse([WLOG3, WLOG3x], "CBW")
strong3 = parse([SLOG3, SLOG3x], "CBS")

for name, data in [("WEAK I30 (kref 1.12974)", weak),
                   ("STRONG I30 (kref 0.51673)", strong)]:
    print(f"\n【{name}】 dkeff (pcm), A=3 directional, generic I=30")
    print("S\\M  " + "  ".join(f"M{m:<3}" for m in M_LIST))
    for S in S_LIST:
        cells = [f"{data.get(S, {}).get(M, (0, float('nan')))[1]:+6.1f}"
                 for M in M_LIST]
        print(f"S{S:<2} " + "  ".join(cells))

# TY3 vs I30 comparison at fine grids
print("\n【TY3 vs I30 对照】dkeff (pcm)")
print("S\\M  " + "  ".join(f"{'TY3/I30 M'+str(m):<16}" for m in (4, 8, 24)))
for S in (6, 8, 10, 12):
    row = f"S{S:<2} "
    for M in (4, 8, 24):
        a = weak3.get(S, {}).get(M, (0, float('nan')))[1]
        b = weak.get(S, {}).get(M, (0, float('nan')))[1]
        c = strong3.get(S, {}).get(M, (0, float('nan')))[1]
        d = strong.get(S, {}).get(M, (0, float('nan')))[1]
        row += f"  {a:+6.1f}/{b:+6.1f}  "
        row += f"  {c:+6.1f}/{d:+6.1f}   "
    print(row)

# --- symlog figure ---
style = dict(linewidth=1.8, markersize=4.5, zorder=3)
fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2))
fig.suptitle("Checkerboard problem (source paper Fig. 3) — $A{=}3$ "
             "directional closure, generic $I{=}30$ polar set, "
             "tolerance $10^{-7}$", fontsize=15)

for ax, (name, data) in zip(
        axes, [("Weak absorber (Sa = 0.07)\nOpenMC ref k = 1.12974", weak),
               ("Strong absorber (Sa = 0.70)\nOpenMC ref k = 0.51673", strong)]):
    for i, S in enumerate(S_LIST):
        d = data.get(S, {})
        Ms = [m for m in M_LIST if m in d]
        ys = [d[m][1] for m in Ms]
        color = cm.tab10(i % 10)
        ls = "-" if S <= 3 else (":" if S <= 8 else "--")
        ax.plot(Ms, ys, ls, color=color, marker="o",
                label=f"S={S} ({2*S}×{2*S})", **style)
    ys_all = [d[m][1] for d in data.values() for m in M_LIST if m in d]
    lim = 1.15 * max(abs(min(ys_all)), abs(max(ys_all)))
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("M (azimuthal fan count)")
    ax.set_ylabel(r"$\Delta k_{eff}$ vs OpenMC (pcm, log)")
    ax.set_xticks(M_LIST)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylim(-lim, lim)
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.grid(alpha=0.35)
    ax.legend(fontsize=8.5, loc="best")

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"\nsaved {OUT}")
