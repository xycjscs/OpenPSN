# One-shot patcher for remote c5g7_one.py kref resolution bug. Idempotent.
import sys
p = sys.argv[1]
src = open(p).read()
if "unanimous" in src:
    print("ALREADY PATCHED")
    sys.exit(0)
old = """kref = None
for cc in spec.get("cases", []):
    if int(cc.get("M", -1)) == M:
        kref = cc.get("kref"); break
if kref is None:
    kref = spec.get("kref", 1.3347260)"""
v1 = """# kref is geometry/material-level (top-level); per-case kref lookup by M was
# a silent-wrong-reference bug (MOX high-M fell back to UO2 default). Prefer
# top-level, forbid fallback entirely.
kref = spec.get("kref")
if kref is None:
    sys.exit(f"FATAL: no top-level kref in spec for {case}")"""
new = """# kref is geometry/material-level: every cases[] entry carries the same value.
# Old code matched cases by M and silently fell back to the UO2 default on a
# miss (bogus -11221 pcm on MOX high-M). Read ALL entries, require unanimous.
krefs = {float(cc["kref"]) for cc in spec.get("cases", []) if "kref" in cc}
kref = spec.get("kref")
if kref is not None:
    krefs.add(float(kref))
if len(krefs) != 1:
    sys.exit(f"FATAL: ambiguous/missing kref for {case}: {sorted(krefs)}")
kref = krefs.pop()"""
if old in src:
    open(p, "w").write(src.replace(old, new)); print("PATCHED from original")
elif v1 in src:
    open(p, "w").write(src.replace(v1, new)); print("PATCHED from v1")
else:
    print("PATTERN NOT FOUND"); sys.exit(1)
