#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C5G7-2D: power distribution (PSN2D) + relative error vs OpenMOC.

Two .npz inputs (51x51 pin fission-rate maps):
  psn_c5g7_2d_power.npz        (keys: power, keff)
  openmoc_c5g7_2d_power.npz    (keys: power, keff)

Orientation: PSN node grid and OpenMOC mesh may differ by x/y flips.
We auto-align by trying 4 flips and keeping the one with minimum mean |rel diff|.

Outputs (300 dpi, print-ready):
  c5g7_power_and_error.png     3 panels: PSN power | OpenMOC power | rel error %
  c5g7_power_psn.png           PSN power alone (large, for sending)
  c5g7_power_openmoc.png       OpenMOC power alone (large)
  c5g7_relerr_openmoc.png      rel error vs OpenMOC (large)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "/opt/data/workspace/c5g7"
OUT  = "/opt/data/output/2026-09"
os.makedirs(OUT, exist_ok=True)

psn     = np.load(os.path.join(HERE, "psn_c5g7_2d_power.npz"))
openmoc = np.load(os.path.join(HERE, "openmoc_c5g7_2d_power.npz"))

p_psn = psn["power"].astype(float)
p_om  = openmoc["power"].astype(float)
k_psn, k_om = float(psn["keff"]), float(openmoc["keff"])

# normalize both to same total (compare shape, not absolute scale)
def norm(a):
    a = a.copy()
    a = a - a.min()
    return a / a.sum()

pn, on = norm(p_psn), norm(p_om)

def relerr(a, b):
    b = b + 1e-30
    return (a - b) / b * 100.0

# ---- auto-align: try 4 flips on openmoc map, min mean|rel| ----
cands = {
    "id":   on,
    "fx":   on[:, ::-1],
    "fy":   on[::-1, :],
    "fx_y": on[::-1, ::-1],
}
best, bestkey = None, None
for key, om_t in cands.items():
    e = np.abs(relerr(pn, om_t)).mean()
    print(f"orient {key:5s}  mean|rel| = {e:.4f} %")
    if best is None or e < best:
        best, bestkey = e, key
print(">> chosen orientation:", bestkey)
om_a = cands[bestkey]
pn_a = pn

rel = relerr(pn_a, om_a)   # (PSN - OpenMOC)/OpenMOC, %

def colorbar_frac(ax):
    ax.figure.canvas.draw()
    cb = ax.images[0].colorbar
    return cb

# ================= Panel set: 3-up =================
fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), dpi=300)

im0 = axes[0].imshow(pn_a, origin="lower", cmap="inferno", vmin=0,
                     vmax=pn_a.max() * 0.98)
axes[0].set_title("PSN2D  (node method, M=12, S=1)\n$k_{eff}$ = %.6f" % k_psn, fontsize=13)
colorbar_frac(axes[0]).set_label("rel. pin fission rate", fontsize=11)

im1 = axes[1].imshow(om_a, origin="lower", cmap="inferno", vmin=0,
                     vmax=om_a.max() * 0.98)
axes[1].set_title("OpenMOC  (MOC, 4az×6pol)\n$k_{eff}$ = %.6f" % k_om, fontsize=13)
colorbar_frac(axes[1]).set_label("rel. pin fission rate", fontsize=11)

cmax = np.percentile(np.abs(rel), 99)
im2 = axes[2].imshow(rel, origin="lower", cmap="RdBu_r", vmin=-cmax, vmax=cmax)
axes[2].set_title("Relative error (PSN − OpenMOC)/OpenMOC\n(mean = %+.2f%%,  max|·| = %.1f%%)"
                  % (rel.mean(), np.abs(rel).max()), fontsize=13)
colorbar_frac(axes[2]).set_label("%", fontsize=11)

for ax in axes:
    ax.set_xlabel("x (pin)  [1.26 cm/cell]", fontsize=11)
    ax.set_ylabel("y (pin)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("C5G7-2D MOX fuel-assembly benchmark  (51×51 pins, 2×2 fuel + L reflector, 7-group)",
             fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
p1 = os.path.join(OUT, "c5g7_power_and_error.png")
fig.savefig(p1, dpi=300, bbox_inches="tight"); plt.close(fig)
print("saved", p1)

# ================= single large panels =================
def big_panel(arr, title, cmap, vmin, vmax, label, fname, colorbar=True):
    fig, ax = plt.subplots(figsize=(8.6, 8.0), dpi=300)
    im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("x (pin)   1.26 cm / cell", fontsize=12)
    ax.set_ylabel("y (pin)", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    if colorbar:
        ax.figure.canvas.draw()
        im.colorbar().set_label(label, fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print("saved", p)

big_panel(pn_a,
          "C5G7-2D  fission power — PSN2D node method\n"
          r"($k_{eff}=%.6f$;  M=12 azimuthal, S=1)" % k_psn,
          "inferno", 0, pn_a.max() * 0.98, "rel. pin fission rate",
          "c5g7_power_psn.png")

big_panel(om_a,
          "C5G7-2D  fission power — OpenMOC\n"
          r"($k_{eff}=%.6f$;  4 azimuthal × 6 polar, CMFD)" % k_om,
          "inferno", 0, om_a.max() * 0.98, "rel. pin fission rate",
          "c5g7_power_openmoc.png")

big_panel(rel,
          "Relative error of PSN2D vs OpenMOC\n"
          r"$\left(P_{PSN}-P_{OpenMOC}\right)/P_{OpenMOC}$   "
          "(mean %+.2f%%,  max |·| %.1f%%)" % (rel.mean(), np.abs(rel).max()),
          "RdBu_r", -cmax, cmax, "percent", "c5g7_relerr_openmoc.png")

# quick stats for the report
print("\n=== stats ===")
print("keff PSN=%.6f  OpenMOC=%.6f  diff=%+.0f pcm" % (k_psn, k_om, (k_psn-k_om)*1e5))
print("rel error: mean=%+.3f%%  rms=%.3f%%  max|·|=%.2f%%" %
      (rel.mean(), np.sqrt((rel**2).mean()), np.abs(rel).max()))
print("95%% of pins within %.2f%%" % np.percentile(np.abs(rel), 95))
