#!/usr/bin/env python3
"""Checkerboard (Chao paper Fig.3) A=3 directional S x M sweep: table + figure.

Parses cb_a3_weak.log / cb_a3_strong.log (HPC 32-thread GMRES, 72/72 pts).
Outputs:
  1) S x M tables (delta-keff pcm vs OpenMC, weak 1.12974 / strong 0.51673)
  2) paper/figures/checkerboard_a3_sm.png — two panels (weak/strong),
     delta-keff vs M, one curve per S=1..6, house style.
"""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
WLOG = os.path.join(_HERE, "cb_a3_weak.log")
SLOG = os.path.join(_HERE, "cb_a3_strong.log")
WLOG2 = os.path.join(_HERE, "cb_a3_weak_ext.log")
SLOG2 = os.path.join(_HERE, "cb_a3_strong_ext.log")
OUT = os.path.join(_REPO, "paper", "figures", "checkerboard_a3_sm.png")
KREF = {"CBW": 1.12974, "CBS": 0.51673}
M_LIST = [2, 4, 8, 12, 16, 24]
S_LIST = list(range(1, 13))


def parse(paths, tag):
    data = {}  # S -> {M: (keff, dref)}
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


weak = parse([WLOG, WLOG2], "CBW")
strong = parse([SLOG, SLOG2], "CBS")

# --- text tables (Feishu plain-text style) ---
for name, data, kref in [("WEAK absorber (Sa=0.07, kref 1.12974)", weak, 1.12974),
                         ("STRONG absorber (Sa=0.70, kref 0.51673)", strong, 0.51673)]:
    print(f"\n【{name}】 dkeff (pcm), A=3 directional, GMRES 32thr")
    hdr = "S\\M  " + "  ".join(f"M{m:<3}" for m in M_LIST)
    print(hdr)
    for S in S_LIST:
        row = f"S{S}  "
        cells = []
        for M in M_LIST:
            if M in data.get(S, {}):
                keff, dref = data[S][M]
                cells.append(f"{dref:+6.1f}")
            else:
                cells.append("   -- ")
        print(row + "  ".join(cells))
    # keff row for completeness
    print("keff (S6 row): " +
          "  ".join(f"{data.get(6, {}).get(M, (0, 0))[0]:.7f}" for M in M_LIST))

# --- figure: two panels, dkeff vs M, curve per S ---
style = dict(linewidth=1.8, markersize=4.5, zorder=3)
from matplotlib import cm
fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2))
fig.suptitle("Checkerboard (paper Fig.3) — A=3 directional PSN, "
             "GMRES 32-thread, tol 1e-7, S=1..12", fontsize=15)

for ax, (name, data, kref) in zip(
        axes, [("Weak absorber (Sa = 0.07)\nOpenMC ref k = 1.12974", weak, 1.12974),
               ("Strong absorber (Sa = 0.70)\nOpenMC ref k = 0.51673", strong, 0.51673)]):
    for i, S in enumerate(S_LIST):
        d = data.get(S, {})
        Ms = [m for m in M_LIST if m in d]
        ys = [d[m][1] for m in Ms]
        color = cm.tab10(i % 10)
        ls = "-" if S <= 3 else (":" if S <= 8 else "--")
        ax.plot(Ms, ys, ls, color=color, marker="o",
                label=f"S={S} ({2*S}×{2*S})", **style)
    ys_all = [d[m][1] for d in data.values() for m in M_LIST if m in d]
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("M (azimuthal fan count)")
    ax.set_ylabel(r"$\Delta k_{eff}$ vs OpenMC (pcm, log)")
    ax.set_xticks(M_LIST)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylim(-1.15 * max(abs(min(ys_all)), abs(max(ys_all))),
                1.15 * max(abs(min(ys_all)), abs(max(ys_all))))
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.grid(alpha=0.35)
    ax.legend(fontsize=8.5, loc="best")

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"\nsaved {OUT}")
