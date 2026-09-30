# Micro-benchmark: threaded PARDISO vs single-thread scipy splu,
# C5G7-(i,g)-class sparse system (n=512^2 grid, shifted Laplacian:
# SPD, non-singular; fill stats representative of 5pt face-coupled matrices).
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

n = 512
N = n * n
I_ = sp.eye(N, format="csr")
# 5-point stencil via flattened offsets + shift (nonsingular; same nnz class)
A = (4.0 * I_
     - sp.eye(N, k=1, format="csr") - sp.eye(N, k=-1, format="csr")
     - sp.eye(N, k=n, format="csr") - sp.eye(N, k=-n, format="csr")
     + 0.05 * I_).tocsc()
rng = np.random.default_rng(0)
b = rng.random(N)
print(f"N={N} nnz={A.nnz}", flush=True)

t0 = time.time()
lu = splu(A)
x1 = lu.solve(b)
print(f"splu factor+solve: {time.time()-t0:.1f}s (single-thread)", flush=True)

import pypardiso
for nt in (1, 8, 32):
    ps = pypardiso.PyPardisoSolver()
    ps.set_n_threads(nt)
    t0 = time.time()
    x2 = ps.solve(A, b)
    t_pd = time.time() - t0
    err = np.abs(x2 - x1).max() / np.abs(x1).max()
    ps.release_all_mem()
    print(f"pypardiso nt={nt}: factor+solve {t_pd:.1f}s  relerr={err:.2e}", flush=True)
