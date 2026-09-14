# -*- coding: utf-8 -*-
"""dbg_consistency.py — definitive CMFD consistency diagnostic.
Build M = P A_c^{-1} R A_f (as operator applications) and apply it to
smooth test vectors.  Consistency requires M e ~= e on the smooth
(spatial diffusion) modes.  Also compare R A_f P vs A_c directly.
Also measures the A_f vs A_c diagonal scaling (the likely c-factor bug).
"""
import sys
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

sys.path.insert(0, '/opt/data/workspace/OpenPSN')
from psn2d import highorder as ho
from psn2d import highsweep as hs
from psn2d import model as mdl
from psn2d import __main__ as PM

EX = '/opt/data/workspace/OpenPSN/examples/c5g7/study2_rectangular'


def build_psn(key='uo2', MM=2):
    spec = mdl.load_spec(EX + f'/c5g7_rect_{key}_assembly.yaml')
    case = {'name': f'M{MM}', 'model': 'ty3', 'I': 30, 'M': MM,
            'subdivide': 1, 'A': 3, 'closure': 'directional_matrix'}
    spec['solver'].update({'keff_tol': 1e-10, 'max_outer': 10, 'threads': 1})
    psn = PM.build_solver(spec, case, threads=1, mem_limit_gb=32)
    if not hasattr(psn, '_groups'):
        psn._vectorize_setup()
    return psn


psn = build_psn()
K, M = 4, psn.M
i, g = psn.I - 1, 0
B = ho.BlockAngleSystem(psn, i, g, 3, 'directional')
A_f = B.matrix.tocsr()
c = 3
coarse, node_map, areaw = hs.build_coarse_psn(psn, c=c)
Cc = ho.BlockAngleSystem(coarse, i, g, 3, 'directional')
A_c = Cc.matrix.tocsr()

# --- face row maps (internal only) ---
def face_rows(p):
    r = 0
    rows = {}
    for f, face in enumerate(p.faces):
        if face["type"] == "internal":
            for m in range(M):
                for a in range(K):
                    rows[(f, m, a)] = r
                    r += 1
        elif face["type"] == "vacuum":
            r += M * K
        else:
            r += len(p.face_pairs[f]) * K
    return rows


frow = face_rows(psn)
crow = face_rows(coarse)
cpair = {}
for cf, cface in enumerate(coarse.faces):
    if cface["type"] == "internal":
        a, b = int(cface["a"]), int(cface["b"])
        cpair[(a, b)] = cf
        cpair[(b, a)] = cf
f2c = {}
for f, face in enumerate(psn.faces):
    if face["type"] != "internal":
        continue
    ca, cb = int(node_map[int(face["a"])]), int(node_map[int(face["b"])])
    if ca == cb:
        continue
    f2c[f] = cpair[(ca, cb)]

# R: (ncol_c, nrow_f)  coarse row <- sum fine rows (weight 1)
rr, rc, rw = [], [], []
for f, cf in f2c.items():
    for m in range(M):
        for a in range(K):
            rr.append(crow[(cf, m, a)])
            rc.append(frow[(f, m, a)])
            rw.append(1.0)
Rmat = coo_matrix((rw, (rr, rc)), shape=(A_c.shape[0], A_f.shape[0])).tocsr()
# P: (ncol_f, ncol_c)  fine col <- coarse col (weight 1)
pr, pc, pw = [], [], []
for f, cf in f2c.items():
    for m in range(M):
        for a in range(K):
            pc.append(psn.colidx[f, m] * K + a)
            pr.append(crow[(cf, m, a)])
            pw.append(1.0)
Pmat = coo_matrix((pw, (pc, pr)), shape=(A_f.shape[1], A_c.shape[0])).tocsr()

# --- scaling check ---
d_f = A_f.diagonal()
d_c = A_c.diagonal()
print(f"A_f {A_f.shape} nnz={A_f.nnz}  diag mean|.|={np.abs(d_f).mean():.4e}")
print(f"A_c {A_c.shape} nnz={A_c.nnz}  diag mean|.|={np.abs(d_c).mean():.4e}")
print(f"  A_c/A_f diag ratio = {np.abs(d_c).mean()/np.abs(d_f).mean():.4f}  (expect ~1 if consistent)")
print(f"  c (collapse) = {c}")

# --- M = P A_c^{-1} R applied to smooth vectors ---
def apply_M(v):
    rv = Rmat @ v
    dc = Cc.factor.solve(rv)
    return Pmat @ dc


# smooth test vectors: low-order spatial polynomial on the m=0,a=0
# internal-face columns (the slow mode lives here).  j=row, i=col.
jidx, iidx = psn.j_idx, psn.i_idx
nx, ny = psn.nx, psn.ny
smooth = {}
for name, fn in {
    "const": lambda j, i: 1.0,
    "lin_j": lambda j, i: j / ny,
    "lin_i": lambda j, i: i / nx,
    "quad": lambda j, i: (j / ny) * (i / nx),
    "cos1": lambda j, i: np.cos(np.pi * (j + 0.5) / ny) * np.cos(np.pi * (i + 0.5) / nx),
}.items():
    e = np.zeros(A_f.shape[1])
    for f, face in enumerate(psn.faces):
        if face["type"] != "internal":
            continue
        n = int(face["a"])
        val = fn(float(jidx[n]), float(iidx[n]))
        # assign to the m=0,a=0 column (the leading slow component);
        # put the same value on the paired column for symmetry
        e[psn.colidx[f, 0] * K + 0] = val
    e_norm = np.linalg.norm(e)
    Me = apply_M(e)
    # Rayleigh-like consistency: (e·Me)/(e·e);  also elementwise corr
    ray = float(e @ Me) / float(e @ e)
    # how well does Me match e (relative residual of the correction):
    rel = float(np.linalg.norm(Me - e)) / max(e_norm, 1e-300)
    # per-component: Me / e where |e| is significant
    sig = np.abs(e) > 0.3 * np.abs(e).max()
    ratio = Me[sig] / (e[sig] + 1e-300)
    print(f"  {name:6s}: (e·Me)/(e·e)={ray:+.3f}  ||Me-e||/||e||={rel:.3f}  "
          f"median(Me/e)={np.median(ratio):+.3f}")
print("\nCONSISTENCY: need (e·Me)/(e·e)≈+1 and Me/e≈+1 for smooth modes.")
print("If (e·Me)/(e·e) is negative or far from +1, the coarse operator")
print("is inconsistent -> residual CMFD diverges (the observed bug).")

# --- also: R A_f P vs A_c on a smooth coarse vector ---
print("\nGalerkin check: R A_f P  vs  A_c  (should be ~equal if consistent)")
rng = np.random.default_rng(0)
for trial in range(2):
    vc = rng.normal(size=A_c.shape[1])
    # make it smooth: low-pass by keeping only the cos1 pattern
    vc2 = np.zeros_like(vc)
    # just use a random coarse vector for the operator comparison
    RAfP = Rmat @ (A_f @ (Pmat @ vc))
    Ac = A_c @ vc
    rel = float(np.linalg.norm(RAfP - Ac)) / max(float(np.linalg.norm(Ac)), 1e-300)
    print(f"  trial{trial}: ||R A_f P vc - A_c vc||/||A_c vc|| = {rel:.3f}")
