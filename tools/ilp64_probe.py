import sys; sys.path.insert(0, "/work/home/xycjscs/psn_strict")
import ctypes, os
lib = ctypes.CDLL("/work/home/xycjscs/mf/envs/psn/lib/libmkl_rt.so.2")
i64 = ctypes.POINTER(ctypes.c_int64)
vp = ctypes.c_void_p
lib.pardiso.argtypes = [ctypes.POINTER(ctypes.c_int64),
    i64, i64, i64, i64, i64, vp, i64, i64, i64, i64, i64, i64, vp, vp, i64]
import pardiso64
def _load_patched():
    pardiso64._lib = lib
    return lib
pardiso64._load = _load_patched
from pardiso64 import Pardiso64LU
import numpy as np, scipy.sparse as sp
n = 50000
rng = np.random.default_rng(1)
A = sp.diags([rng.uniform(-1,0,n-1), rng.uniform(4,5,n), rng.uniform(-1,0,n-1)],[-1,0,1],format="csr")
lu = Pardiso64LU(A, n_threads=8)
b = np.ones(n); x = lu.solve(b)
print("ILP64-RT probe OK; resid", np.abs(A@x-b).max(), "nnz14", lu._nnz14, "gate-nnz", lu.factor_nnz)
