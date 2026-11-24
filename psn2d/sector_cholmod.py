"""Optional FP64 supernodal solver for the finite-sector diffusion sweep."""
import ctypes as c
import os
from pathlib import Path
from threading import Lock

import numpy as np


def _load_kernel():
    default = Path(__file__).with_name("libsector_cholmod.so")
    path = os.environ.get("OPENPSN_SECTOR_CHOLMOD", str(default))
    try:
        lib = c.CDLL(path)
    except OSError as exc:
        raise RuntimeError(f"Cannot load the optional sector CHOLMOD bridge at {path}. "
                           "See the OpenPSN build instructions.") from exc
    ptr, size = c.c_void_p, c.c_size_t
    lib.psn_cholmod_analyze.argtypes = [size, size, ptr, ptr]
    lib.psn_cholmod_analyze.restype = ptr
    lib.psn_cholmod_factor.argtypes = [ptr, size, size, ptr, ptr, ptr, c.POINTER(size)]
    lib.psn_cholmod_factor.restype = ptr
    lib.psn_cholmod_solve.argtypes = [ptr, ptr, ptr]
    lib.psn_cholmod_solve.restype = c.c_int
    lib.psn_cholmod_free.argtypes = [ptr]
    lib.psn_cholmod_free.restype = None
    return lib


class _Factor:
    def __init__(self, lib, handle, size):
        self.lib, self.handle, self.size = lib, handle, size

    def solve(self, rhs):
        rhs = np.ascontiguousarray(rhs, dtype=np.float64)
        if rhs.shape != (self.size,):
            raise ValueError(f"expected CHOLMOD RHS with shape ({self.size},)")
        x = np.empty_like(rhs)
        if not self.lib.psn_cholmod_solve(self.handle, rhs.ctypes.data, x.ctypes.data):
            raise RuntimeError("sector CHOLMOD solve failed")
        return x

    def __del__(self):
        self.lib.psn_cholmod_free(self.handle)


class SectorCholmod:
    """Reuse symbolic analysis for the common graph of the phase systems."""

    def __init__(self):
        self.lib = _load_kernel()
        self.symbolic = None
        self.lock = Lock()

    def __call__(self, matrix):
        matrix = matrix.tocsc()
        n = matrix.shape[0]
        if n == 0 or matrix.shape[1] != n:
            raise ValueError("sector CHOLMOD requires a nonempty square matrix")
        if max(n, matrix.nnz) > np.iinfo(np.int32).max:
            raise ValueError("sector CHOLMOD matrix exceeds the int32 index range")
        matrix.sort_indices()
        p = np.ascontiguousarray(matrix.indptr, dtype=np.int32)
        i = np.ascontiguousarray(matrix.indices, dtype=np.int32)
        with self.lock:
            if self.symbolic is None:
                handle = self.lib.psn_cholmod_analyze(n, matrix.nnz, p.ctypes.data, i.ctypes.data)
                if not handle:
                    raise RuntimeError("sector CHOLMOD symbolic analysis failed (METIS required)")
                self.symbolic = _Factor(self.lib, handle, n)
                self.pattern = (p, i)
            elif not (np.array_equal(p, self.pattern[0]) and np.array_equal(i, self.pattern[1])):
                raise ValueError("sector CHOLMOD requires a common sparse pattern across phase systems")
        values = np.ascontiguousarray(matrix.data, dtype=np.float64)
        bytes_ = c.c_size_t()
        handle = self.lib.psn_cholmod_factor(self.symbolic.handle, n, matrix.nnz, p.ctypes.data,
                                             i.ctypes.data, values.ctypes.data, c.byref(bytes_))
        if not handle:
            raise RuntimeError("sector CHOLMOD numerical factorization failed")
        return _Factor(self.lib, handle, n).solve, bytes_.value
