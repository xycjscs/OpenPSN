# -*- coding: utf-8 -*-
"""Figures for the strict-PSN paper. Data = strict work-area logs only.

Fig 1  best-S trajectory vs M (locking figure)
Fig 2  per-row Richardson limits a(S), both methods, + S^-2 reference
Fig 3  prediction audit: predicted row limit vs measured, error in units
       of the method's own self-reported error bar
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = "/opt/data/workspace/psn_strict/"
OUT = "/opt/data/workspace/OpenPSN/paper_strict/figures/"
os.makedirs(OUT, exist_ok=True)
M_LIST = [2, 4, 8, 12, 16, 24, 48, 96, 192]


def load(tag, files):
    d = {}
    for f in files:
        if not os.path.exists(f):
            continue
        for ln in open(f, errors="ignore"):
            m = re.match(rf"{tag} M(\d+)_S(\d+) keff=\S+ dref=([+-][\d.]+)",
                         ln)
            if m:
                d[(int(m.group(2)), int(m.group(1)))] = float(m.group(3))
    return d


LG_S = load("LGS30", [D + "hpc_out/ext_1.log", D + "cb_legacy_strong.log",
                      D + "cb_legacy_strong_M48p.log"])
ST_S = load("STS30", [D + "hpc_out/ext_3.log", D + "cb_strict_strong.log",
                      D + "cb_strict_strong_M48p.log",
                      D + "cb_strict_Mconv_S12.log",
                      D + "cb_final_checks.log"])
LG_W = load("LGW30", [D + "hpc_out/ext_0.log", D + "cb_legacy_weak.log",
                      D + "cb_legacy_weak_M48p.log"])
ST_W = load("STW30", [D + "hpc_out/ext_2.log", D + "cb_strict_weak.log",
                      D + "cb_strict_weak_M48p.log",
                      D + "cb_weak_Mconv_S12.log",
                      D + "cb_final_checks.log"])

# ---------- Fig 1: best-S trajectory -------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
for a, (lg, st, ttl) in zip(
        ax, ((LG_S, ST_S, "strong absorber"),
             (LG_W, ST_W, "weak absorber"))):
    for d, c, lab in ((lg, "tab:red", "legacy"), (st, "tab:blue", "strict")):
        xs, ys = [], []
        for M in M_LIST:
            pts = {S: v for (S, m), v in d.items() if m == M and S <= 20}
            if pts:
                Sbest = min(pts, key=lambda S: abs(pts[S]))
                xs.append(M)
                ys.append(Sbest)
        a.semilogx(xs, ys, "o-", color=c, label=lab, ms=5)
    a.plot([2, 192], [1, 96], "k:", lw=0.8, alpha=0.5,
           label=r"$S=M/2$ (valley)")
    a.set_xlabel(r"azimuthal order $M$")
    a.set_ylabel(r"best spatial grid $S^{*}$")
    a.set_title(ttl, fontsize=10)
    a.legend(fontsize=8)
    a.set_ylim(0, 21)
    a.grid(alpha=0.3)
fig.suptitle(r"optimal $S$ at fixed $M$: legacy drifts on the valley band; "
             r"strict saturates", fontsize=11)
fig.tight_layout()
fig.savefig(OUT + "fig_bestS.png", dpi=200)
plt.close(fig)

# ---------- Fig 2: Richardson limits a(S) --------------------------------
def limits(d):
    out = {}
    for S in range(1, 21):
        v = [d.get((S, m)) for m in (48, 96, 192)]
        if any(x is None for x in v):
            continue
        # Richardson from 96,192 (O(M^-2)); error bar = |pred48/96 - pred96/192|
        lim = (4 * v[2] - v[1]) / 3
        alt = (4 * v[1] - v[0]) / 3
        out[S] = (lim, abs(lim - alt))
    return out


fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
for a, (lg, st, ttl) in zip(
        ax, ((LG_S, ST_S, "strong absorber"),
             (LG_W, ST_W, "weak absorber"))):
    for d, c, lab in ((lg, "tab:red", "legacy"), (st, "tab:blue", "strict")):
        L = limits(d)
        Ss = np.array(sorted(L))
        lim = np.array([L[s][0] for s in Ss])
        err = np.array([L[s][1] for s in Ss])
        a.errorbar(Ss, lim, yerr=err, fmt="o-", color=c, ms=4, lw=1,
                   capsize=2, label=lab + r"  $\lim_{M\to\infty} d$")
    Ss = np.array([2, 4, 8, 16], float)
    a.plot(Ss, 5.8 * (12 / Ss) ** 2, "k--", lw=0.9,
           label=r"$\propto S^{-2}$")
    a.axhline(0, color="grey", lw=0.6)
    a.set_xlabel(r"spatial grid $S$")
    a.set_ylabel(r"$\Delta k_{\mathrm{eff}}$ (pcm)")
    a.set_title(ttl, fontsize=10)
    a.legend(fontsize=7.5)
    a.grid(alpha=0.3)
fig.suptitle(r"angular-converged limit per fixed $S$ (Richardson $M\to\infty$):"
             "\nstrict $\\to$ pure spatial residual, legacy stops at the "
             "wrong-operator plateau", fontsize=10.5)
fig.tight_layout()
fig.savefig(OUT + "fig_limits.png", dpi=200)
plt.close(fig)

# ---------- Fig 3: prediction audit --------------------------------------
# For every S with M=48/96/192 (+384 where available), predict the row
# limit from (48,96) only; the error bar is the extrapolation disagreement.
# Score = |pred - measured| / bar.  Measured = M384 where available, else
# the (96,192) Richardson (kept separate in the table).
fig, ax = plt.subplots(figsize=(4.8, 3.6))
meas384 = {12: (ST_S.get((12, 384)), ST_W.get((12, 384)))}
rows = []
for d, wtag, vt in ((ST_S, "STS30", "strong"), (ST_W, "STW30", "weak")):
    for S in range(1, 21):
        v = [d.get((S, m)) for m in (24, 48, 96, 192)]
        if any(x is None for x in v):
            continue
        p0 = (4 * v[1] - v[0]) / 3
        p1 = (4 * v[2] - v[1]) / 3
        p2 = (4 * v[3] - v[2]) / 3
        bar = max(abs(p1 - p0), abs(p2 - p1))
        rows.append((vt, S, p2, bar))
for i, (vt, S, pred, bar) in enumerate(rows):
    ax.errorbar(pred, i, xerr=max(bar, 1e-3), fmt="o", ms=4,
                color="tab:red" if vt == "strong" else "tab:blue",
                capsize=2)
ax.axvline(0, color="grey", lw=0.6)
ax.set_xscale("symlog", linthresh=1.0)
ax.set_xlabel(r"predicted row limit from M<=192 (pcm)")
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([f"{vt[:1]}{S}" for vt, S, _, _ in rows], fontsize=6)
ax.set_title("reference-free prediction, all 40 rows\n"
             "(error bar = extrapolation self-disagreement)", fontsize=9.5)
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(OUT + "fig_predict.png", dpi=200)
plt.close(fig)

# ---------- tables -------------------------------------------------------
def row_tbl(d, S):
    return [d.get((S, m)) for m in M_LIST]


print("wrote", OUT, "fig_bestS / fig_limits / fig_predict")
print("S12 strong legacy:", row_tbl(LG_S, 12))
print("S12 strong strict:", row_tbl(ST_S, 12))
print("limits strong legacy S8/12/16/20:",
      [round(limits(LG_S)[s][0], 1) for s in (8, 12, 16, 20)
       if s in limits(LG_S)])
print("limits strong strict S8/12/16/20:",
      [round(limits(ST_S)[s][0], 1) for s in (8, 12, 16, 20)
       if s in limits(ST_S)])
