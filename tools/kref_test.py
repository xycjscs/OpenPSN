# Smoke test: patched kref resolution must give MOX=1.1851053, UO2=1.3347260
import sys, os
REPO = os.environ.get("PSN_REPO")
sys.path.insert(0, REPO)
from psn2d.model import load_spec
for mat, want in (("mox", 1.1851053), ("uo2", 1.3347260)):
    case = f"c5g7_rect_{mat}_assembly"
    spec = load_spec(f"{REPO}/examples/c5g7/study2_rectangular/{case}.yaml")
    krefs = {float(cc["kref"]) for cc in spec.get("cases", []) if "kref" in cc}
    if spec.get("kref") is not None:
        krefs.add(float(spec["kref"]))
    assert len(krefs) == 1, (case, sorted(krefs))
    got = krefs.pop()
    assert abs(got - want) < 1e-9, (mat, got, want)
    print(f"{mat}: kref={got} OK ({len(spec.get('cases', []))} cases unanimous)")
print("ALL OK")
