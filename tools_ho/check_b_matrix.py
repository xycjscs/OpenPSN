"""Read-only equivalence check: planned local vacuum B blocks vs package.

Does NOT touch psn2d/ files. Compares:
  1. local polar-node dicts (TY3 / generic) fed to the package's own
     face_current_matrix — verifies our node data satisfies its contract;
  2. the planned local B construction (same formula, local basis_gauss)
     vs the package's face_current_matrix — must be identical;
  3. A0 identity B00 == 2*|alpha_face| against legacy node_alphas —
     predicts bit-level A0 vacuum regression.
"""
import sys
import numpy as np
from scipy import sparse

sys.path.insert(0, "/opt/data/workspace/OpenPSN")
PKG = "/opt/data/cache/documents/boundary_upgrade/OpenPSN_boundary_upgrade_Yuchao_Xu/code"
sys.path.insert(0, PKG)

from psn2d.node import (ty_polar_set, generic_polar_set, node_params_generic,
                        node_alphas, node_alphas_generic)
from boundary_conditions import face_current_matrix as pkg_face_current_matrix

def local_nodes_ty3():
    ty = ty_polar_set(3)
    w = np.asarray(ty["w"]); w = w / w.sum()
    return [dict(rule="TY3", index=i, weight=float(w[i]),
                 mu=float(ty["mu"][i])) for i in range(3)]

def local_nodes_generic(I=64):
    ps = generic_polar_set(I)
    out = []
    for i in range(len(ps["theta"])):
        out.append(dict(rule="generic", index=i,
                        theta=float(ps["theta"][i]),
                        dtheta=float(ps["dtheta"]),
                        weight=float(ps["w"][i])))
    return out

def local_b_matrix(nodes, M, A, axis):
    """Planned local construction: same formula as package, local basis."""
    import importlib
    hf = importlib.import_module("psn2d.highorder")
    K = A + 1
    z, qw, V = hf.basis_gauss(K - 1)
    # kappa per polar node
    kappa = []
    for nd in nodes:
        if nd["rule"] == "TY3":
            kappa.append(nd["mu"])
        else:
            # alpha_polar for unit St, unit size
            pr = node_params_generic(1.0, 1.0, nd["theta"], nd["dtheta"])
            kappa.append(pr["alpha_polar"])
    kappa = np.asarray(kappa)
    blocks = []
    for m in range(M):
        phi = (m + 0.5 + 0.5 * z) * np.pi / M
        normal = np.abs(np.cos(phi) if axis == "x" else np.sin(phi))
        block = V.T @ ((qw * normal)[:, None] * V)
        blocks.append((block + block.T) / 2)
    B = sparse.block_diag([kap * b for kap in kappa for b in blocks],
                          format="csr")
    B.eliminate_zeros()
    return B, kappa

worst = 0.0
for rule, nodes in (("TY3", local_nodes_ty3()),
                    ("generic I64", local_nodes_generic(64))):
    for axis in ("x", "y"):
        for M, A in ((4, 0), (8, 0), (8, 3), (12, 4)):
            B_loc, kappa = local_b_matrix(nodes, M, A, axis)
            B_pkg, meta = pkg_face_current_matrix(nodes, M, A, axis)
            d = max((B_loc - B_pkg).toarray().max(),
                    abs((B_loc - B_pkg).toarray()).max())
            worst = max(worst, d)
            print(f"{rule:12s} {axis} M{M:<3d} A{A}: B max|diff| = {d:.3e}  "
                  f"(shape {B_pkg.shape})  kappa_local_max={kappa.max():.9f}")

# A0 identity: B00 == 2*|alpha_face| for every (i, m, face)
dphi = np.pi / 8
M = 8
print("\nA0 identity check vs legacy alpha_face (TY3):")
ty = ty_polar_set(3); w = np.asarray(ty["w"]); w = w / w.sum()
bad = 0
for i in range(3):
    for m in range(M):
        alpha = node_alphas(ty["mu"][i], 1.0, (m + 0.5) * dphi, dphi)
        for lf in range(4):
            axis = "x" if lf < 2 else "y"
            B, _ = local_b_matrix(local_nodes_ty3(), M, 0, axis)
            # row index: polar i, segment m, mode 0 -> i*M*1 + m
            b00 = B[i * M + m, i * M + m]
            ok = np.isclose(b00, 2 * abs(alpha[lf]), rtol=0, atol=1e-15)
            if not ok:
                bad += 1
                print(f"  MISMATCH i={i} m={m} lf={lf}: B00={b00!r} "
                      f"2|alpha|={2*abs(alpha[lf])!r}")
print("A0 identity mismatches:", bad, "/ 96")
print("\nWORST B diff (local vs package):", worst)
sys.exit(1 if (worst > 1e-14 or bad) else 0)
