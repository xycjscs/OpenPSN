import sys, os, time, gc
sys.path.insert(0, os.environ["PSN_REPO"]); sys.path.insert(0, os.path.expanduser("~/psn_strict"))
import psn2d
from psn2d.memopt import install_compact_lu
from pypardiso import PyPardisoSolver
M, S = 96, 6
c = {"name": "scan_M96_S6", "M": M, "subdivide": S, "model": "ty3",
     "closure": "strict_phase_space", "kref": 1.1851058, "I": 3}
install_compact_lu(backend="mmd")
from psn2d.systems import build_all_systems
A = build_all_systems(c)[0][0].tocsr()
NT = int(os.environ.get("PD_THREADS", "32"))
cpu = "unknown"
try:
    for l in open("/proc/cpuinfo"):
        if l.startswith("model name"): cpu = l.split(":",1)[1].strip(); break
except Exception: pass
t0 = time.time(); res = "FAIL"
try:
    ps = PyPardisoSolver(mtype=11); ps.iparm[0] = 1; ps.iparm[2] = NT
    ps.factorize(A); res = "OK"
    ps.free_memory(everything=True); del ps; gc.collect()
except Exception: res = "FAIL"
print(f"SCAN host={os.uname().nodename} cpu={cpu} n={A.shape[0]} res={res} t={time.time()-t0:.0f}s", flush=True)
