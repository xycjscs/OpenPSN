# Equivalence check v2: compact-LU backends (lu/mmd) vs plain, MOX M24_S2 strict.
# off-baseline measured 2026-09-29 job 45491640: keff=1.183683567087
# Also prints per-run peak RSS (VmHWM of this process grows monotonically,
# so run each backend in a subprocess instead -- here we keep it simple and
# report final VmHWM per child via subprocess module).
import subprocess, sys, os

REPO = os.environ.get("PSN_REPO") or ""
BASE = 1.183683567087

CHILD = r'''
import sys, os, time
REPO = os.environ["PSN_REPO"]
sys.path.insert(0, REPO)
from psn2d.__main__ import run_case
from psn2d.model import load_spec
opt = sys.argv[1]
spec = load_spec(f"{REPO}/examples/c5g7/study2_rectangular/c5g7_rect_mox_assembly.yaml")
case = {"name": "eq", "M": 24, "subdivide": 2, "model": "ty3", "I": 30,
        "closure": "strict_phase_space"}
t0 = time.time()
r = run_case(spec, case, verbose=False,
             threads=int(os.environ.get("THREADS", "32")),
             opt=opt, mem_limit_gb=None)
hwm = 0
for line in open("/proc/self/status"):
    if line.startswith("VmHWM"):
        hwm = int(line.split()[1]) / 1048576.0
print(f"RESULT opt={opt} keff={r['keff']:.12f} t={time.time()-t0:.0f}s peakRSS={hwm:.1f}GB")
'''

for opt in ("lu", "mmd"):
    p = subprocess.run([sys.executable, "-c", CHILD, opt],
                       env={**os.environ, "PSN_REPO": REPO},
                       capture_output=True, text=True)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
    if not line:
        print(f"opt={opt}: FAILED\n{p.stderr[-600:]}")
        continue
    print(line[0], flush=True)
    k = float(line[0].split("keff=")[1].split()[0])
    d = abs(k - BASE)
    verdict = "OK" if d < 1e-10 else "MISMATCH"
    print(f"  |diff vs off-baseline| = {d:.3e}  [{verdict}]", flush=True)
