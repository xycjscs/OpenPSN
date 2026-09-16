# -*- coding: utf-8 -*-
"""3-row pin-averaged fission figure for the paper (Fig. 9 bottom / studyIII).

Regenerates paper/figures/c5g7_rect4_a3_fission.png:
  UO2 assembly S=2 M=24, MOX assembly S=2 M=24, 1/4 core S=1 M=8
  (OpenPSN | OpenMC same-geometry reference | pin-by-pin relative error).

Data: tools_ho/data/r17_core4/*.npz (HPC dumps) + tools_ho/data/omc_*.npy.
Ported from the c5g7 workdir (2026-09-16); paths are repo-relative.
"""
import os
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.join(_REPO, 'tools_ho', 'data')
FIGDIR = os.path.join(_REPO, 'paper', 'figures')

import sys
# (no sys.path needed)


def norm(a):
    a = a - a.min()
    return a / a.sum()


def pick_orientation(P, O):
    Pn = P - P.min(); Pn = Pn / Pn.sum()
    best, bcorr = O, -2
    for cand in (O, O[:, ::-1], O[::-1, :], O[::-1, ::-1]):
        On = cand - cand.min(); On = On / On.sum()
        c = np.corrcoef(Pn.ravel(), On.ravel())[0, 1]
        if c > bcorr:
            best, bcorr = cand, c
    return best, bcorr


def pin_nu_grid(spec_path, center):
    y = yaml.safe_load(open(spec_path))
    mg = np.array([list(r) for r in y['geometry']['mat_grid']], dtype=int)
    pm = mg[center[0]::4, center[1]::4]
    nu = np.vectorize(lambda i: IDX2NU[int(i)])(pm)
    return nu, pm.astype(str)


def make_row(label, keff, kref, psn_full, om_map, spec, crop=None,
             sub=None, title_cfg='S=2  M=24'):
    ny = psn_full.shape[0]
    n = sub if sub else 8
    psn_pin = (psn_full.reshape(ny // n, n, ny // n, n).mean(axis=(1, 3)))
    nu, mat = pin_nu_grid(spec, (1, 1))
    if crop is not None:
        r0, r1, c0, c1 = crop
        psn_pin = psn_pin[r0:r1, c0:c1]
        nu = nu[r0:r1, c0:c1]
        mat = mat[r0:r1, c0:c1]
        om_map = om_map[r0:r1, c0:c1]
    p = psn_pin / np.where(nu > 0, nu, 1.0)
    o, corr = pick_orientation(p, om_map)
    P, O = norm(p), norm(o)
    rel = (P - O) / O * 100
    fuel = np.isin(mat, FUEL)
    r = rel[fuel]
    r = r[np.isfinite(r)]
    dk = (keff - kref) * 1e5
    row = dict(label=label, dref=dk, corr=corr, P=P, O=O, rel=rel,
               fuel=fuel, rms=np.sqrt((r ** 2).mean()), mx=np.abs(r).max(),
               p95=np.percentile(np.abs(r), 95), n_fuel=r.size,
               title_cfg=title_cfg)
    print(f"{label}: keff={keff:.7f} dref={dk:+.1f} pcm orient={corr:.6f} "
          f"n_fuel={row['n_fuel']} rms={row['rms']:.4f}% max={row['mx']:.3f}%")
    return row


D = os.path.join(_REPO, 'examples', 'c5g7', 'study2_rectangular')
rows = []
rows.append(make_row(
    r'UO$_2$ assembly', 1.3347213, 1.3347260,
    np.load(f'{os.path.join(W, "r17_core4")}/R17UO2_M24_S2.npz')['full'],
    np.load(f'{W}/omc_rect_asm_power.npy'),
    f'{D}/c5g7_rect17_split4_uo2_assembly.yaml'))
rows.append(make_row(
    'MOX assembly', 1.1850074, 1.1851053,
    np.load(f'{os.path.join(W, "r17_core4")}/R17MOX_M24_S2.npz')['full'],
    np.load(f'{W}/omc_mox_rect_asm_power.npy'),
    f'{D}/c5g7_rect17_split4_mox_assembly.yaml'))
rows.append(make_row(
    '1/4 core (S1, M8)', 1.1872011, 1.1872176,
    np.load(f'{os.path.join(W, "r17_core4")}/CORE4_M8_S1.npz')['full'],
    np.load(f'{W}/omc_rect_core_power.npy'),
    f'{D}/c5g7_core_split4_uo2_mox_quarter_core.yaml',
    crop=(0, 34, 0, 34), sub=4, title_cfg='S=1  M=8'))

n = len(rows)
fig, axes = plt.subplots(n, 3, figsize=(9.6, 3.0 * n))
for i, R in enumerate(rows):
    lim = max(1.0, np.percentile(np.abs(R['rel'][R['fuel']]), 98))
    relm = np.ma.masked_where(~R['fuel'], R['rel'])
    im0 = axes[i, 0].imshow(R['P'], cmap='viridis', origin='lower')
    im1 = axes[i, 1].imshow(R['O'], cmap='viridis', origin='lower')
    im2 = axes[i, 2].imshow(relm, cmap='RdBu_r', origin='lower',
                            vmin=-lim, vmax=lim)
    for ax in axes[i]:
        ax.set_xticks([]); ax.set_yticks([])
    axes[i, 0].set_title(f'OpenPSN  {R["title_cfg"]}', fontsize=9)
    axes[i, 1].set_title('OpenMC (ref)', fontsize=9)
    axes[i, 2].set_title(f'PSN $-$ OM rel. err (%)  '
                         f'rms={R["rms"]:.3f}  max={R["mx"]:.2f}', fontsize=9)
    axes[i, 0].set_ylabel(f'{R["label"]}\n'
                          r'$\Delta k=' + f'{R["dref"]:+.1f}' + r'~\mathrm{pcm}$',
                          fontsize=9, fontweight='bold')
    for im, axidx in ((im0, 0), (im1, 1), (im2, 2)):
        cb = fig.colorbar(im, ax=axes[i, axidx], fraction=0.046, pad=0.04)
        for lb in cb.ax.get_yticklabels():
            lb.set_fontweight('bold')
fig.suptitle(r'$2\times2$ fuel-split tiling, $A{=}3$ closure: pin-averaged fission '
             r'$\Sigma_f\phi$ vs. same-geometry OpenMC '
             r'(assemblies S=2 M=24; core S=1 M=8, cropped to the 2$\times$2 fuel block)',
             fontsize=10.5, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.94])
out2 = f'{FIGDIR}/c5g7_rect4_a3_fission.png'
fig.savefig(out2, dpi=200, bbox_inches='tight')
print('saved', out2)
