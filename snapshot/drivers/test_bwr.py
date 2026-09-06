"""BWR bundle 2-group benchmark (paper Sec 4.3, Fig. 8/10, Table 2).

Geometry: 6x6 cells of 1.5 cm. Outer ring = moderator (water),
central 4x4 = normal fuel (16 pins). Reflective boundary on all 4 sides.
Gd case: the pin at x,y in [0,1.5] (i=3, j=3) becomes Gd fuel.

Each 1.5cm cell is subdivided into N x N sub-nodes (paper rows 1x1..10x10),
node size h = 1.5/N cm.  Restricted PSN (TY 3-point polar set, I=3),
azimuthal M in {4,8,12,16,20,24}.

Reference (OpenMC, Stepanek et al. 1982):
    no-Gd  kref = 1.18797
    Gd     kref = 0.86688
Paper pcm = ABSOLUTE error (k - kref) * 1e5.
"""
import sys, numpy as np
import os as _os; sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'core'))
from psn_solver import PSN2D

# material indices: 0 = normal fuel, 1 = Gd fuel, 2 = moderator
# ---- Table 2 cross sections (2 groups x 3 materials) -------------------- #
# NOTE: read exponents carefully (E-01 = 0.1x, not 0.01x).
# St[g, mat]
St = np.array([
    [0.196647, 0.196657, 0.222064],        # g1: fuel, Gd, mod
    [0.596159, 3.53100,  0.887874],        # g2
])
# Sgg[g, g2, mat] = scattering cross-section from group g2 -> group g
Sgg = np.zeros((2, 2, 3))
for mat in (0, 1):  # normal fuel & Gd fuel share scattering
    Sgg[0, 0, mat] = 0.178000   # from g1 -> g1
    Sgg[1, 0, mat] = 0.0100200  # from g1 -> g2
    Sgg[0, 1, mat] = 0.001089   # from g2 -> g1
    Sgg[1, 1, mat] = 0.525500   # from g2 -> g2
Sgg[0, 0, 2] = 0.199500         # mod from g1 -> g1
Sgg[1, 0, 2] = 0.0218800        # mod from g1 -> g2
Sgg[0, 1, 2] = 0.001558         # mod from g2 -> g1
Sgg[1, 1, 2] = 0.878300         # mod from g2 -> g2
# nuSf[g, mat]
nuSf = np.array([
    [0.006203, 0.006203, 0.0],   # g1
    [0.110100, 0.110100, 0.0],   # g2
])
# fission spectrum: all into group 1
chi = np.array([1.0, 0.0])

KREF_NO_GD = 1.18797
KREF_GD = 0.86688

# Fig. 10-A (PSN, no-Gd) printed k-eff error (pcm), rows N=1..10, cols M=4,8,12,16,20,24
PAPER_NOGD = {
    1:  [123, -78, -125, -144, -152, -157],
    2:  [203, 25, -19, -35, -43, -48],
    3:  [223, 51, 9, -7, -15, -20],
    4:  [230, 62, 20, 5, -3, -8],
    5:  [234, 67, 26, 10, 2, -2],
    6:  [236, 70, 29, 14, 6, 1],
    7:  [237, 72, 31, 16, 8, 4],
    8:  [238, 73, 33, 17, 9, 5],
    9:  [239, 74, 34, 18, 10, 6],
    10: [239, 74, 34, 19, 11, 7],
}

def bwr_map(gd=False):
    base = np.zeros((6, 6), int)
    base[:, :] = 2           # all moderator
    base[1:5, 1:5] = 0       # central 4x4 normal fuel
    if gd:
        base[3, 3] = 1       # Gd pin at (0~1.5, 0~1.5)
    return base

def run_bwr(N, M, gd=False, generic=False, I=30, tol=1e-10):
    base = bwr_map(gd)
    g = np.kron(base, np.ones((N, N), int))
    h = 1.5 / N
    kws = dict(generic=False, M=M)
    if generic:
        kws = dict(generic=True, I=I, M=M)
    p = PSN2D(g, h, St, Sgg, nuSf, chi=chi,
              boundary=("reflect",) * 4, **kws)
    k, phi, _ = p.keff(max_outer=4000, outer_tol=tol, verbose=False)
    return k

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sanity"
    if mode == "sanity":
        # N=1, M=4, no-Gd -> expect ~1.18797 + 123e-5
        k = run_bwr(N=1, M=4, gd=False)
        err = (k - KREF_NO_GD) * 1e5
        print(f"SANITY N=1 M=4 no-Gd: keff={k:.6f}  err={err:+.1f} pcm  "
              f"(paper +123)")
    elif mode == "diag":
        # quick convergence scan to check trend vs paper
        for N in (1, 2, 4, 6, 8, 10):
            row = []
            for M in (4, 8, 12, 16, 20, 24):
                k = run_bwr(N=N, M=M, gd=False)
                err = (k - KREF_NO_GD) * 1e5
                row.append(err)
            pap = PAPER_NOGD[N]
            print(f"N={N:2d}: " + "  ".join(
                f"{e:+6.1f}/{p:+5d}" for e, p in zip(row, pap)), flush=True)
    elif mode == "full":
        # full 10x6 matrix, no-Gd
        for N in range(1, 11):
            row = []
            for M in (4, 8, 12, 16, 20, 24):
                k = run_bwr(N=N, M=M, gd=False)
                row.append((k - KREF_NO_GD) * 1e5)
            pap = PAPER_NOGD[N]
            d = [e - p for e, p in zip(row, pap)]
            print(f"N={N:2d}: " + "  ".join(
                f"{e:+7.1f}" for e in row) +
                "   |paper " + "  ".join(f"{p:+5d}" for p in pap) +
                "   |dmax " + f"{max(abs(x) for x in d):5.1f}", flush=True)
