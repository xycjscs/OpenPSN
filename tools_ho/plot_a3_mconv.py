#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate c5g7_rect4_a3_m_conv.png as a 2x3 figure (2026-09-16, user ask).

Top row    : Delta k versus M (unchanged from the original figure).
Bottom row : pin-averaged fission-rate Sigma_f*phi MAX relative error (%)
             versus M, same three panels / line conventions, metric
             convention identical to rect4_fission_sm.py (pin average,
             /nu, 4-orientation best-align, unit-sum normalised, fuel pins).
"""
import os
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

import os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.join(_REPO, 'tools_ho', 'data', 'r17_core4')
FIGDIR = os.path.join(_REPO, 'paper', 'figures')
D = os.path.join(_REPO, 'examples', 'c5g7', 'study2_rectangular')

IDX2NU = {0: 0.0, 1: 2.441061, 2: 2.865292, 3: 2.871281,
          4: 2.873560, 5: 0.0, 6: 2.434001, 7: 0.0}
FUEL = ('1', '2', '3', '4')
MS = [2, 4, 8, 12, 16, 24]
KREF = {'UO2': 1.3347260, 'MOX': 1.1851053, 'CORE': 1.1872176}
ASM_REF = {'UO2': os.path.join(W, '..', 'omc_rect_asm_power.npy'),
           'MOX': os.path.join(W, '..', 'omc_mox_rect_asm_power.npy')}
ASM_SPEC = {'UO2': f'{D}/c5g7_rect17_split4_uo2_assembly.yaml',
            'MOX': f'{D}/c5g7_rect17_split4_mox_assembly.yaml'}
CORE_REF = os.path.join(W, '..', 'omc_rect_core_power.npy')
CORE_SPEC = f'{D}/c5g7_core_split4_uo2_mox_quarter_core.yaml'

# ------------------------------------------------------------- top row
DASM = {
    'UO2': dict(S1=[-119.2, -35.0, -2.8, +5.3, +8.3, +10.3],
                S2=[-128.7, -50.2, -14.7, -6.3, -2.8, -0.5]),
    'MOX': dict(S1=[-274.8, -74.1, -14.5, -0.9, +4.1, +7.4],
                S2=[-303.3, -103.2, -33.9, -19.6, -13.7, -9.8]),
}
CORE_S1 = dict(M=[2, 4, 8, 12, 16, 24], dk=[-135.6, -34.2, -1.7, +6.2, +9.1, +11.0])

# ---------------------------------------------------------- bottom row
def norm(a):
    a = a - a.min()
    return a / a.sum()


def pick_orientation(P, O):
    Pn = norm(P)
    best, bcorr = O, -2.0
    for cand in (O, O[:, ::-1], O[::-1, :], O[::-1, ::-1]):
        c = np.corrcoef(Pn.ravel(), norm(cand).ravel())[0, 1]
        if c > bcorr:
            best, bcorr = cand, c
    return best, bcorr


def pin_nu_grid(spec_path, center, step):
    y = yaml.safe_load(open(spec_path))
    mg = np.array([list(r) for r in y['geometry']['mat_grid']], dtype=int)
    pm = mg[center[0]::step, center[1]::step]
    nu = np.vectorize(lambda i: IDX2NU[int(i)])(pm)
    return nu, pm.astype(str)


def max_err(psn_full, om_map, nu, mat, block, crop=None):
    ny = psn_full.shape[0]
    pin = (psn_full.reshape(ny // block, block, ny // block, block)
           .mean(axis=(1, 3)))
    if crop is not None:
        r0, r1, c0, c1 = crop
        pin = pin[r0:r1, c0:c1]
        om_map = om_map[r0:r1, c0:c1]
        nu = nu[r0:r1, c0:c1]
        mat = mat[r0:r1, c0:c1]
    p = pin / np.where(nu > 0, nu, 1.0)
    o, corr = pick_orientation(p, om_map)
    P, O = norm(p), norm(o)
    rel = (P - O) / O * 100.0
    fuel = np.isin(mat, FUEL)
    r = rel[fuel]
    r = r[np.isfinite(r)]
    return float(np.abs(r).max())


fmax = {}
for mat in ('UO2', 'MOX'):
    for S in (1, 2):
        for M in MS:
            f = os.path.join(W, f'R17{mat}_M{M}_S{S}.npz')
            if not os.path.exists(f):
                continue
            full = np.load(f, allow_pickle=True)['full']
            sub = 4 if S == 1 else 8
            nu, matg = pin_nu_grid(ASM_SPEC[mat], (1, 1), 4)
            fmax[(mat, S, M)] = max_err(full, np.load(ASM_REF[mat]), nu, matg, sub)
            print(f'{mat} S{S} M{M:2d}: max={fmax[(mat, S, M)]:.4f}%')
core_f = {'S1': {}, 'S2': {}}
for S, sub in ((1, 4), (2, 8)):
    for M in MS:
        f = os.path.join(W, f'CORE4_M{M}_S{S}.npz')
        if not os.path.exists(f):
            continue
        full = np.load(f, allow_pickle=True)['full']
        nu, matg = pin_nu_grid(CORE_SPEC, (1, 1), 4)
        core_f[f'S{S}'][M] = max_err(full, np.load(CORE_REF), nu, matg, sub,
                                     crop=(0, 34, 0, 34))
        print(f'CORE S{S} M{M:2d}: max={core_f[f"S{S}"][M]:.4f}%')

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.5))
titles = [r'UO$_2$ assembly (ref. 1.3347260)',
          r'MOX assembly (ref. 1.1851053)',
          r'1/4 core (ref. 1.1872176)']

for j, key in enumerate(('UO2', 'MOX')):
    ax = axes[0, j]
    for S, st in ((1, '-'), (2, '--')):
        ax.plot(MS, DASM[key][f'S{S}'], st, marker='o', ms=4, lw=1.3,
                color=('tab:blue' if S == 1 else 'tab:red'), label=f'S={S}')
    ax.axhline(0, color='0.5', lw=0.7)
    ax.set_title(titles[j], fontsize=9.5, fontweight='bold')
    ax.tick_params(labelsize=8.5)
axes[0, 0].set_ylabel(r'$\Delta k$ (pcm)')
axes[0, 0].legend(fontsize=8.5, framealpha=0.9)

ax = axes[0, 2]
ax.plot(CORE_S1['M'], CORE_S1['dk'], '-', marker='o', ms=4.5, lw=1.3,
        color='tab:blue', label='S=1')
ax.axhline(0, color='0.5', lw=0.7)
ax.set_xlim(1, 26)
ax.text(0.95, 0.35, 'S=2 row: in progress',
        transform=ax.transAxes, ha='right', va='center', fontsize=7.5,
        style='italic', color='0.3')
ax.set_title(titles[2], fontsize=9.5, fontweight='bold')
ax.tick_params(labelsize=8.5)

for j, key in enumerate(('UO2', 'MOX')):
    ax = axes[1, j]
    for S, st in ((1, '-'), (2, '--')):
        Ms_, ys_ = [], []
        for M in MS:
            v = fmax.get((key, S, M))
            if v is not None:
                Ms_.append(M)
                ys_.append(v)
        ax.plot(Ms_, ys_, st, marker='o', ms=4, lw=1.3,
                color=('tab:blue' if S == 1 else 'tab:red'), label=f'S={S}')
    ax.set_yscale('linear')
    ax.set_ylim(0.16, 0.25)
    ax.set_xlim(1, 26)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.2f}'))
    ax.set_xlabel('M (azimuthal segments)')
    ax.set_title(titles[j], fontsize=9.5, fontweight='bold')
    ax.tick_params(labelsize=8.5)
axes[1, 0].set_ylabel(r'fission-rate max error $|\Delta|_{\max}$ (%)')
axes[1, 0].legend(fontsize=8.5, framealpha=0.9)

ax = axes[1, 2]
for S, st in ((1, '-'), (2, '--')):
    Ms_, ys_ = [], []
    for M in MS:
        v = core_f[f'S{S}'].get(M)
        if v is not None:
            Ms_.append(M)
            ys_.append(v)
    ax.plot(Ms_, ys_, st, marker='o', ms=4, lw=1.3,
            color=('tab:blue' if S == 1 else 'tab:red'), label=f'S={S}')
ax.set_yscale('linear')
ax.set_ylim(0.6, 0.9)
ax.set_xlim(1, 26)
ax.yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.2f}'))
ax.set_xlabel('M (azimuthal segments)')
ax.text(0.95, 0.35, 'S=2 row: in progress',
        transform=ax.transAxes, ha='right', va='center', fontsize=7.5,
        style='italic', color='0.3')
ax.tick_params(labelsize=8.5)
for a in axes.flat:
    a.grid(alpha=0.25, lw=0.5, which='both')

fig.suptitle(r'$A{=}3$ directional closure, $2\times2$ fuel-split tiling: '
             r'top, $\Delta k$ versus $M$; bottom, pin-averaged fission-rate '
             r'max error versus $M$ (same-geometry OpenMC references, '
             r'$\approx\pm 3.3$--$3.6$ pcm)',
             fontsize=10, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.945])
out = f'{FIGDIR}/c5g7_rect4_a3_m_conv.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print('saved', out)
