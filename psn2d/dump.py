# -*- coding: utf-8 -*-
"""Post-convergence field dump (feature added 2026-09-15, user request).

After every high-order solve, call :func:`dump_fields` to persist the
neutron flux and fission-rate distributions alongside the sweep log, so
downstream comparisons (e.g. Figure-2-style pin-by-pin vs OpenMC) need
no separate dump run and the "best configuration" can be picked
afterwards from the complete, field-annotated table.

Saved arrays (npz):
  flux        (ng, ny, nx)  per-group scalar flux [n^-2 s^-1]
  fission     (ny, nx)      full-node fission rate sum_g nu*Sigma_f*phi
  fission_pin (ny//b, nx//b) block-averaged fission per pin
  fission_pin_fx / _fy / _fxy   mirrored pin fields (assembly symmetry)
  flux_pin    (ny//b, nx//b) all-group flux, block-averaged per pin
  pin, full   backward-compat aliases (bestfig pipeline reads these)
  keff, M, S, block, converged, backend, time_s   (via metadata)
"""
import numpy as np


def _block_mean(p2d, b):
    ny, nx = p2d.shape
    if ny % b or nx % b:
        return p2d
    return p2d.reshape(ny // b, b, nx // b, b).mean(axis=(1, 3))


def _as_2d(a, ny, nx):
    """Accept (ny, nx) or flat (nodes,) -> (ny, nx)."""
    if a.shape == (ny, nx):
        return a
    return a.reshape(ny, nx)


def dump_fields(psn, phi, out_path, block=4, metadata=None):
    """Persist flux + fission fields for a converged (or not) solve.

    psn       -- the PSN2D solver object (provides nx, ny, ng, nuSf,
                 i_idx, j_idx, mat)
    phi       -- per-group scalar flux, shape (ng, ny*nx) [as returned
                 by keff_highorder_gmres] or (ng, ny, nx)
    out_path  -- npz destination
    block     -- fuel blocks per pin (2x2 pin split -> 4)
    metadata  -- optional dict of scalar/array extras (keff, M, S, ...)
    Returns out_path.
    """
    midx = psn.mat[psn.j_idx, psn.i_idx].ravel()
    ny, nx = psn.ny, psn.nx
    fission = np.zeros(psn.nodes)
    flux_all = np.zeros(psn.nodes)
    for g in range(psn.ng):
        _pg = np.asarray(phi[g]).ravel()   # flat (nodes,) or (ny, nx)
        fission += psn.nuSf[g, midx] * _pg
        flux_all += _pg
    f2d = _as_2d(fission, ny, nx)
    f2d_all = _as_2d(flux_all, ny, nx)
    fp = _block_mean(f2d, block)
    pp = _block_mean(f2d_all, block)
    data = {
        'flux': np.asarray(phi, dtype=np.float64),
        'fission': f2d.astype(np.float64),
        'fission_pin': fp.astype(np.float64),
        'fission_pin_fx': fp[:, ::-1].astype(np.float64),
        'fission_pin_fy': fp[::-1, :].astype(np.float64),
        'fission_pin_fxy': fp[::-1, ::-1].astype(np.float64),
        'flux_pin': pp.astype(np.float64),
        # backward-compat aliases (bestfig pipeline reads pin/full)
        'pin': fp.astype(np.float64),
        'pin_fx': fp[:, ::-1].astype(np.float64),
        'pin_fy': fp[::-1, :].astype(np.float64),
        'pin_fxy': fp[::-1, ::-1].astype(np.float64),
        'full': f2d.astype(np.float64),
        'block': np.int64(block),
    }
    if metadata:
        data.update(metadata)
    np.savez(out_path, **data)
    return out_path
