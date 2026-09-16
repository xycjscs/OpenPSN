# -*- coding: utf-8 -*-
"""HPC single-point fission-field dumper (SLURM compute node).

Same production protocol as hpc_point.py (GMRES pool, outer_tol=rtol=1e-7),
plus a pin-averaged fission map on disk for Figure-2-style comparison plots.

quantity: node power = sum_g nu*Sf[g,mat]*phi[g]; pin = mean over the 4x4
base blocks (rect4 convention: every pin fuel 2x2 split); saved nu-grid so
the plotting step can divide to Sigma_f*phi (as bestflux_figs.py does).

Usage (from a SLURM job):
  python hpc_flux_dump.py <key> <spec_yaml> <M> <S> <out_npz>
  python tools_ho/hpc_flux_dump.py core <repo>/examples/c5g7/study2_rectangular/c5g7_core_split4_uo2_mox_quarter_core.yaml 8 1 ./psn_core_M8_S1.npz
"""
import os, sys, time, resource

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _hpc_threads():
    cp = os.environ.get('SLURM_CPUS_PER_TASK')
    if cp:
        try:
            return int(cp)
        except ValueError:
            pass
    return 96


def main():
    key, spec, M, S, out = (sys.argv[1], sys.argv[2],
                            int(sys.argv[3]), int(sys.argv[4]),
                            sys.argv[5])
    N = _hpc_threads()

    from psn2d import model, __main__ as M_
    from psn2d import highsweep, mckern
    import numpy as np
    mckern.ENABLED = True

    t0 = time.time()
    spec_d = model.load_spec(spec)
    case = {'name': f'M{M}_S{S}', 'model': 'ty3', 'I': 30, 'M': M,
            'subdivide': S, 'A': 3, 'closure': 'directional_matrix'}
    spec_d['solver'].update({'keff_tol': 1e-7, 'max_outer': 6000,
                             'threads': 1})
    psn = M_.build_solver(spec_d, case, threads=1)
    k, phi, qim, h = highsweep.keff_highorder_gmres(
        psn, A=3, branch='directional', max_outer=6000,
        outer_tol=1e-7, rtol=1e-7, maxiter=800,
        pairs=True, nworkers=N, verbose=False)
    dt = time.time() - t0

    matidx = psn.mat[psn.j_idx, psn.i_idx]
    power = np.zeros(psn.nodes)
    for g in range(psn.ng):
        power += psn.nuSf[g, matidx.ravel()] * phi[g]
    ny, nx = psn.ny, psn.nx
    p2d = power.reshape(ny, nx)
    BLOCK = 4
    assert ny % BLOCK == 0 and nx % BLOCK == 0, (ny, nx)
    pin = p2d.reshape(ny // BLOCK, BLOCK, nx // BLOCK, BLOCK).mean(axis=(1, 3))

    np.savez(out,
             pin=pin.astype(np.float64),
             pin_fx=pin[:, ::-1].astype(np.float64),
             pin_fy=pin[::-1, :].astype(np.float64),
             pin_fxy=pin[::-1, ::-1].astype(np.float64),
             full=p2d.astype(np.float64),
             keff=float(k), M=np.int64(M), S=np.int64(S),
             block=np.int64(BLOCK),
             backend=np.array('gmres'), time_s=float(dt))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    tagline = '' if h['converged'] else ' NOT_CONVERGED'
    print(f"DUMP {key} M{M}_S{S} keff={k:.7f} nodes={psn.nodes} "
          f"iters={h['outer_iterations']} wall={dt:.0f}s "
          f"peak={peak:.2f}GB threads={N} pin={pin.shape} "
          f"saved={out}{tagline}", flush=True)


if __name__ == '__main__':
    main()
