import sys, json, time
sys.path.insert(0, '/opt/data/workspace/OpenPSN/snapshot/core')
import numpy as np
from psn_solver import PSN2D

FUEL = (1.5, 0.15, 0.24)
ABS_W = (1.0, 0.07, 0.0); ABS_S = (1.0, 0.70, 0.0)
KREF_W = 1.12974; KREF_S = 0.51673
base = np.array([[0, 1], [1, 0]], int)

out = {}
t0 = time.time()
for case, ABS, KREF in (('weak', ABS_W, KREF_W), ('strong', ABS_S, KREF_S)):
    St = np.array([[FUEL[0], ABS[0]]]); Sa = np.array([[FUEL[1], ABS[1]]])
    nu = np.array([[FUEL[2], ABS[2]]])
    Sgg = np.zeros((1, 1, 2)); Sgg[0, 0, :] = St[0, :] - Sa[0, :]
    for S in (1, 2, 3, 4):
        for M in (4, 12, 24):
            g = np.kron(base, np.ones((S, S), int)); h = 1.0 / S
            p = PSN2D(g, h, St, Sgg, nu, generic=True, I=30, M=M)
            t1 = time.time()
            k, _, _ = p.keff(max_outer=4000, outer_tol=1e-10, verbose=False)
            dt = time.time() - t1
            key = f"{case}_S{S}_M{M}"
            out[key] = {"keff": round(float(k), 9), "abs_pcm": round(float((k - KREF) * 1e5), 2), "sec": round(dt, 1)}
            print(f"{key}: keff={k:.7f} abs={(k-KREF)*1e5:+8.1f} pcm [{dt:5.1f}s]", flush=True)
            json.dump(out, open('/opt/data/workspace/OpenPSN/web_ref.json', 'w'), indent=1)
print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)
