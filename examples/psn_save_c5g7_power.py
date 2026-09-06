#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PSN2D C5G7-2D: re-run M12_S1 and save the 51x51 pin power distribution.

power_pin(i,j) = sum_g  nu*Sf[g, mat] * phi[g, node]   (fission rate, S=1 so
node == pin).  Also saves the material grid for orientation verification.
Writes:  /opt/data/workspace/c5g7/psn_c5g7_2d_power.npz
         keys: power (ny,nx), matgrid (ny,nx), keff
"""
import sys, time
import numpy as np

sys.path.insert(0, '/opt/data/workspace/psn2d')
from psn2d import model
from psn2d import __main__ as M

SPEC = '/opt/data/workspace/psn2d/examples/c5g7_2d_quarter_core.yaml'
OUT  = '/opt/data/workspace/c5g7/psn_c5g7_2d_power.npz'

spec  = model.load_spec(SPEC)
cases = model.expand_cases(spec)
case  = [c for c in cases if c['name'] == 'M12_S1'][0]

psn = M.build_solver(spec, case)
t0 = time.time()
k, phi, qnode = psn.keff(max_outer=int(spec['solver']['max_outer']),
                         outer_tol=float(spec['solver']['keff_tol']),
                         verbose=True)
dt = time.time() - t0

matidx = psn.mat[psn.j_idx, psn.i_idx]          # (ny, nx)
power  = np.zeros(psn.nodes)
for g in range(psn.ng):
    power += psn.nuSf[g, matidx.ravel()] * phi[g]

nx, ny = psn.nx, psn.ny
power2d = power.reshape(ny, nx)                 # node idx = j*nx + i
matgrid = matidx

np.savez(OUT, power=power2d, matgrid=matgrid, keff=float(k),
         time_s=float(dt))
print(f"\nPSN power saved: {OUT}")
print(f"keff={k:.7f}  nodes={psn.nodes} ({nx}x{ny})  t={dt:.0f}s")
print(f"power range: {power2d.min():.3e} .. {power2d.max():.3e}")
# orientation check: assembly block means (each assembly = 17x17 pins)
blk = power2d.reshape(3, 17, 3, 17).mean(axis=(1, 3))
print("assembly means (j,i):")
for j in range(3):
    print('  ', ['%.4f' % v for v in blk[j]])
