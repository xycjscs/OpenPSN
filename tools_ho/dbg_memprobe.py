# -*- coding: utf-8 -*-
"""Memory probe: 17x17 split4, 68x68 S1, M24, A=3 (dry, NO pool, NO GMRES).

Static OOM investigation (2026-09-14): measures the REAL table-construction
cost per (i,g) instead of extrapolating from estimates:

  1. build_solver cost (model, vectorize setup)
  2. ONE HoNodeLocal init cost (L/Vm/sg/colg/sgkey + class grouping)
  3. ALL 210 HoNodeLocal inits (= parent-side build, as in production)
  4. full microclass tables for (0,0)  [_mc_cache]
  5. all 12 pair tables for (0,0)      [_pmc_cache]
  6. extrapolation to the 2520-block sweep under the candidate layouts

No sweep, no fork — pure table construction.  Backstop RLIMIT_AS 48 GB.
"""
import os, sys, time, resource, collections

sys.path.insert(0, '/opt/data/workspace/OpenPSN')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

resource.setrlimit(resource.RLIMIT_AS, (int(48 * 2 ** 30), -1))

EX = ('/opt/data/workspace/OpenPSN'
      '/examples/c5g7/study2_rectangular')


def rss_gb():
    with open('/proc/self/status') as f:
        for ln in f:
            if ln.startswith('VmRSS:'):
                return int(ln.split()[1]) / 1e6
    return -1.0


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def nbytes_of(obj):
    """Logical byte count (arrays, dicts, lists)."""
    if hasattr(obj, 'nbytes'):
        return obj.nbytes
    tot = 0
    if isinstance(obj, dict):
        for v in obj.values():
            tot += nbytes_of(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            tot += nbytes_of(v)
    return tot


def cache_bytes(c):
    return sum(nbytes_of(v) for v in c.values()) / 1e6


def main():
    from psn2d import model, __main__ as M, highsweep
    spec = model.load_spec(
        EX + '/c5g7_rect17_split4_uo2_assembly.yaml')
    case = {'name': 'memprobe', 'model': 'ty3', 'I': 30, 'M': 24,
            'subdivide': 1, 'A': 3, 'closure': 'directional_matrix'}
    spec['solver'].update({'keff_tol': 1e-7, 'max_outer': 1, 'threads': 1})
    t0 = time.time()
    psn = M.build_solver(spec, case, threads=1, mem_limit_gb=32)
    if not hasattr(psn, '_groups'):
        psn._vectorize_setup()
    print(f"build_solver {time.time()-t0:.0f}s  nodes={psn.nodes} "
          f"ncol={psn.ncol} I={psn.I} M={psn.M} ng={psn.ng} "
          f"ngrp={len(psn._groups)} rss={rss_gb():.2f}GB", flush=True)

    # ---- 210 inits (= production parent-side build) ------------------
    t1 = time.time()
    r1 = rss_gb()
    locals_ = {}
    for i in range(psn.I):
        for g in range(psn.ng):
            locals_[(i, g)] = highsweep.HoNodeLocal(psn, i, g, 3,
                                                    "directional")
    ninit = rss_gb() - r1
    print(f"210x HoNodeLocal init {time.time()-t1:.0f}s  "
          f"dRSS={ninit:.2f}GB rss={rss_gb():.2f}GB  "
          f"(per-(i,g) init ~{ninit/210*1000:.0f}MB, "
          f"{(time.time()-t1)/210:.1f}s)", flush=True)

    nl = locals_[(0, 0)]
    sizes = [len(v) for v in nl.mc_nodes.values()]
    ds = [int(nl.d[v[0]]) for v in nl.mc_nodes.values()]
    print(f"class structure: ncls={len(sizes)} "
          f"Nn max={max(sizes)} min={min(sizes)} sum={sum(sizes)} "
          f"d max={max(ds)} min={min(ds)}", flush=True)
    print(f"  L={nl.L.nbytes/1e6:.0f}MB Vm={nl.Vm.nbytes/1e6:.0f}MB "
          f"sg={nl.sg.nbytes/1e6:.1f}MB colg={nl.colg.nbytes/1e6:.1f}MB "
          f"sgkey_tuples={len(nl.sgkey)}", flush=True)

    # ---- full microclass tables for (0,0) ----------------------------
    r0 = rss_gb()
    t2 = time.time()
    for k in nl.mc_nodes:
        nl._microclass(k)
    F = rss_gb() - r0
    print(f"full tables (0,0): {time.time()-t2:.0f}s  "
          f"mc_cache={len(nl._mc_cache)} bytes="
          f"{cache_bytes(nl._mc_cache):.0f}MB dRSS={F:.2f}GB "
          f"rss={rss_gb():.2f}GB", flush=True)

    # ---- all 12 pair tables for (0,0) --------------------------------
    r0 = rss_gb()
    t2 = time.time()
    for pp in range(psn.M // 2):
        for k in nl.mc_nodes:
            nl._pair_microclass(k, pp)
    P12 = rss_gb() - r0
    P = P12 / (psn.M // 2)
    print(f"12x pair tables (0,0): {time.time()-t2:.0f}s  "
          f"pmc_cache={len(nl._pmc_cache)} bytes="
          f"{cache_bytes(nl._pmc_cache):.0f}MB dRSS={P12:.2f}GB "
          f"({P*1000:.0f}MB/pair) rss={rss_gb():.2f}GB", flush=True)

    # ---- extrapolation ------------------------------------------------
    F, P = float(F), float(P)
    print("=== extrapolation: 210 (i,g), 2520 blocks, 24 workers ===",
          flush=True)
    print(f"parent 210x init            : {ninit:.1f}GB (measured above)",
          flush=True)
    print(f"current lazy (105 (i,g)/worker kept forever): "
          f"105*(F+P)={105*(F+P):.1f}GB/worker -> "
          f"{24*105*(F+P):.0f}GB total  <-- OOM root cause", flush=True)
    print(f"LRU-1 (evict (i,g) after its task): "
          f"~{F+P:.2f}GB/worker -> {24*(F+P):.1f}GB total", flush=True)
    print(f"parent prebuild ALL (F+12P)x210: "
          f"{210*(F+12*P):.0f}GB", flush=True)
    print(f"PROBE DONE peak={peak_gb():.2f}GB "
          f"total_wall={time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
