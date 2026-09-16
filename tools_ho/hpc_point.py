# -*- coding: utf-8 -*-
"""HPC single-point runner (SLURM compute node), LU or GMRES.

Usage (from a SLURM job):
  python hpc_point.py <TAG> <spec_yaml> <M> <S> <method> <threads> <kref> [block] [model]
  e.g.
  python tools_ho/hpc_point.py R17UO2 <repo>/examples/c5g7/study2_rectangular/c5g7_rect17_split4_uo2_assembly.yaml 12 2 lu 64 1.3347260

  [block]  optional fuel-blocks-per-pin for the field dump (default 4 =
           2x2 pin split; use 1 for pin-less benchmark geometries like
           the checkerboard, where the 2x2 base grid IS the pin).
  [model]  optional angular model: ty3 (default) or generic.
  [I]      optional generic polar-set count (even; default 30).  Ignored
           for the ty3 model.

method:
  lu     -> psn2d.highorder.keff_highorder        (global face-current LU,
           all (i,g) SuperLU factors cached; plain power iteration outer)
  gmres  -> psn2d.highsweep.keff_highorder_gmres  (A=3 directional GMRES
           pool, pairs+numba; the local production config D)

Both use the equivalent acceptance gate: dlam < outer_tol AND
src_res < outer_tol, outer_tol = 1e-7 (user decision 2026-09-15,
commit 79fda66).  No local 32 GiB memory guard — the HPC node has
~1007 GB and the user said memory is not a constraint.

The node core count is taken from SLURM_NTASKS (or --cpus-per-task),
falling back to the requested `threads` arg.  BLAS is pinned single-
threaded; the sweep pool (LU back-sub / GMRES angle-pair) carries the
parallelism.
"""
import os, sys, time, resource

# --- thread env: single-threaded BLAS, numba pool handles parallelism ---
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _hpc_threads(threads_arg):
    """Number of worker threads for this point.  These jobs are single-task
    OMP-style (ntasks=1, cpus-per-task=N), so the budget is
    SLURM_CPUS_PER_TASK; SLURM_NTASKS (always 1 here) must NOT shadow it.
    Falls back to the arg outside SLURM (login-node / interactive path)."""
    cp = os.environ.get('SLURM_CPUS_PER_TASK')
    if cp:
        try:
            return int(cp)
        except ValueError:
            pass
    nt = os.environ.get('SLURM_NTASKS')
    if nt:
        try:
            return int(nt)
        except ValueError:
            pass
    return int(threads_arg)


def main():
    (tag, spec, M, S, method, threads, kref) = \
        sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), \
        sys.argv[5].lower(), sys.argv[6], float(sys.argv[7])
    block = int(sys.argv[8]) if len(sys.argv) > 8 else 4
    ang_model = sys.argv[9] if len(sys.argv) > 9 else 'ty3'
    if ang_model not in ('ty3', 'generic'):
        raise SystemExit(f'unknown angular model {ang_model!r}')
    I_generic = int(sys.argv[10]) if len(sys.argv) > 10 else 30
    if ang_model == 'generic' and I_generic % 2 != 0:
        raise SystemExit(f'generic polar set I must be even, got {I_generic}')
    N = _hpc_threads(threads)

    from psn2d import model, __main__ as M_
    from psn2d import mckern
    mckern.ENABLED = True   # numba kernels on for both paths

    t0 = time.time()
    spec_d = model.load_spec(spec)
    case = {'name': f'M{M}_S{S}', 'model': ang_model, 'I': I_generic, 'M': M,
            'subdivide': S, 'A': 3, 'closure': 'directional_matrix'}
    spec_d['solver'].update({'keff_tol': 1e-7, 'max_outer': 6000,
                             'threads': N})
    psn = M_.build_solver(spec_d, case, threads=N)

    if method == 'lu':
        from psn2d.highorder import keff_highorder
        k, phi, qim, h = keff_highorder(psn, A=3, branch='directional',
                                        max_outer=6000, outer_tol=1e-7,
                                        verbose=False)
        matvecs = '-'
    elif method == 'gmres':
        from psn2d.highsweep import keff_highorder_gmres
        k, phi, qim, h = keff_highorder_gmres(
            psn, A=3, branch='directional', max_outer=6000,
            outer_tol=1e-7, rtol=1e-7, maxiter=800,
            pairs=True, nworkers=N, verbose=False)
        matvecs = h.get('gmres_matvecs_total', '-')
    else:
        raise SystemExit(f'unknown method {method!r}')

    dt = time.time() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    dref = (k - kref) * 1e5
    tagline = '' if h['converged'] else ' NOT_CONVERGED'
    print(f"{tag} M{M}_S{S} keff={k:.7f} A=3 K=4 inner={method} "
          f"nodes={psn.nodes} iters={h['outer_iterations']} "
          f"matvecs={matvecs} dref={dref:+.1f}pcm wall={dt:.0f}s "
          f"peak={peak:.2f}GB threads={N}{tagline}", flush=True)

    # --- fission + flux fields: always save, every point (user
    #     2026-09-15: dump inside the sweep; best config selected from
    #     the full table) -- via the psn2d package feature (dump.py) ---
    import numpy as np
    from psn2d.dump import dump_fields
    runs_dir = os.environ.get('PSN_RUNS_DIR', '.')
    os.makedirs(runs_dir, exist_ok=True)
    out = os.path.join(runs_dir, f'{tag}_M{M}_S{S}.npz')
    dump_fields(psn, phi, out, block=block, metadata={
        'keff': float(k), 'M': np.int64(M), 'S': np.int64(S),
        'converged': bool(h['converged']),
        'backend': np.array(method), 'angular': np.array(ang_model),
        'I': np.int64(I_generic), 'time_s': float(dt)})
    import numpy as _np
    d = _np.load(out)
    print(f"DUMP saved {out} pin={d['pin'].shape} flux={d['flux'].shape}",
          flush=True)


if __name__ == '__main__':
    main()
