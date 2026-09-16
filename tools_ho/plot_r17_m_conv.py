#!/usr/bin/env python3
"""R17 (17x17 assembly, pin 2x2 split) M-convergence figure, house style.

Parses /tmp/rect4_a3_sweep.log (live sweep log) and draws:
  left : delta-keff vs OpenMC anchor, curves per (material, S)
  right: S-spread (max-min over S=1..2) per material
House style = paper/figures/pin6_M_convergence.png (linear y in pcm,
categorical M x-axis, tab10, grid, point value labels on right panel).
"""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "/tmp/rect4_a3_sweep.log"
OUT = "/opt/data/workspace/OpenPSN/paper/figures/r17_M_convergence.png"
KREF = {"UO2": 1.3347260, "MOX": 1.1851053}
M_LIST = [2, 4, 8, 12, 16, 24]

data = {}  # (mat, S) -> {M: dref_pcm}
for ln in open(LOG):
    m = re.match(r"R17(\w+) M(\d+)_S(\d+) keff=([\d.]+) .*dref=([+-][\d.]+)pcm", ln)
    if not m:
        continue
    mat, M, S, keff, dref = m.group(1), int(m.group(2)), int(m.group(3)), \
        float(m.group(4)), float(m.group(5))
    data.setdefault((mat, S), {})[M] = dref

style = dict(linewidth=2.0, markersize=6.5, zorder=3)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.2))
fig.suptitle("17×17 assembly (pin 2×2 split, 68×68 base grid) — "
             "OpenPSN vs OpenMC 1.3347260 / 1.1851053", fontsize=15)

curves = [
    ("UO2", 1, "tab:blue",  "o", "UO2  S=1"),
    ("UO2", 2, "tab:orange", "s", "UO2  S=2"),
    ("MOX", 1, "tab:green",  "^", "MOX  S=1"),
    ("MOX", 2, "tab:red",    "D", "MOX  S=2"),
]
spread = {}  # mat -> {M: d(S2)-d(S1)}
for mat in ("UO2", "MOX"):
    s1, s2 = data.get((mat, 1), {}), data.get((mat, 2), {})
    spread[mat] = {m: s2[m] - s1[m] for m in M_LIST
                   if m in s1 and m in s2}

for mat, S, color, mk, label in curves:
    d = data.get((mat, S), {})
    Ms = [m for m in M_LIST if m in d]
    ys = [d[m] for m in Ms]
    ax1.plot(Ms, ys, "-" if S == 1 else "--", color=color,
             marker=mk, label=label, **style)

for mat, (color, mk, off) in (("UO2", ("tab:orange", "s", 9)),
                              ("MOX", ("tab:red", "D", -14))):
    d = spread[mat]
    Ms = [m for m in M_LIST if m in d]
    ax2.plot(Ms, [d[m] for m in Ms], "-", color=color, marker=mk,
             label=f"{mat}  (S=2 − S=1)", **style)
    for m in Ms:
        ax2.annotate(f"{d[m]:.1f}", (m, d[m]),
                     textcoords="offset points", xytext=(0, off),
                     ha="center", fontsize=8.5)

ax1.set_title("keff convergence in M (A=3, directional)", fontsize=12)
ax1.set_xlabel("M (azimuthal fan count)")
ax1.set_ylabel(r"$\Delta k_{eff}$ vs OpenMC (pcm)")
ax1.set_xticks(M_LIST)
ax1.axhline(0.0, color="k", lw=0.8, ls="--")
ax1.grid(alpha=0.35)
ax1.legend(fontsize=10, loc="lower center")

ax2.set_title("S-subdivision sensitivity (S=2 − S=1)", fontsize=12)
ax2.set_xlabel("M (azimuthal fan count)")
ax2.set_ylabel("S-spread (S=2 − S=1, pcm)")
ax2.set_xticks(M_LIST)
ax2.grid(alpha=0.35)

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"saved {OUT}")
for (mat, S), d in sorted(data.items()):
    print(mat, f"S={S}", {m: d[m] for m in M_LIST if m in d})
