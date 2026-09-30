# One-shot: make c5g7_one.py accept OPT env var (default off), for M192 MOX rerun.
import sys
p = sys.argv[1]
src = open(p).read()
if 'os.environ.get("OPT"' in src:
    print("ALREADY PATCHED"); sys.exit(0)
old = 'r = run_case(spec, c, verbose=False,\n             threads=int(os.environ.get("THREADS", "24")),\n             mem_limit_gb=float(os.environ.get("MEM_LIMIT_GB", "0")) or None)'
new = 'r = run_case(spec, c, verbose=False,\n             threads=int(os.environ.get("THREADS", "24")),\n             opt=os.environ.get("OPT", "off"),\n             mem_limit_gb=float(os.environ.get("MEM_LIMIT_GB", "0")) or None)'
assert old in src, "pattern not found"
open(p, "w").write(src.replace(old, new))
print("PATCHED OPT env")
