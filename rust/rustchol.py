"""ctypes loader for the Rust sparse-Cholesky library (`libpsn_chol.so`).

Mirrors ``psn2d.memopt.EigenChol`` / ``psn2d.memopt.SharedChol`` one for one so
it can be wired in as a drop-in backend; the only additions are the fused
multi-RHS solves (``RustChol.solve2`` / ``solve_k``), which ``RustSharedChol``
uses for the mirror pair.

Library search order: ``OPENPSN_RUST_CHOL``, ``$RUST_CHOL`` ,
``<this dir>/libpsn_chol.so``.  METIS is resolved inside the .so at
``chol_factor`` time (RTLD_DEFAULT, then ``$OPENPSN_METIS_LIB``, then the usual
SONAMEs); a missing METIS with ``ordering=1`` raises, it never silently falls
back to natural order.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_LIB = None


def _find_lib():
    for env in ("OPENPSN_RUST_CHOL", "RUST_CHOL"):
        p = os.environ.get(env)
        if p:
            return p
    return str(_HERE / "libpsn_chol.so")


def load_rust_chol():
    """dlopen + one argtype setup per process (same shape as _load_eigen_chol)."""
    global _LIB
    if _LIB is not None:
        return _LIB
    path = _find_lib()
    if not os.path.exists(path):
        raise RuntimeError(f"libpsn_chol.so not found at {path} (run rust/build.sh)")
    lib = ctypes.CDLL(path)
    ip, dp = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)
    lib.chol_factor.argtypes = [ctypes.c_int, ctypes.c_int, ip, ip, dp,
                                ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.chol_factor.restype = ctypes.c_void_p
    lib.chol_solve.argtypes = [ctypes.c_void_p, ctypes.c_int, dp, dp]
    lib.chol_solve.restype = None
    if hasattr(lib, "chol_solve2"):
        lib.chol_solve2.argtypes = [ctypes.c_void_p, ctypes.c_int, dp, dp, dp, dp]
        lib.chol_solve2.restype = None
    if hasattr(lib, "chol_solve_k"):
        lib.chol_solve_k.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                     ctypes.POINTER(dp), ctypes.POINTER(dp),
                                     ctypes.c_int]
        lib.chol_solve_k.restype = None
    lib.chol_nnz.argtypes = [ctypes.c_void_p]
    lib.chol_nnz.restype = ctypes.c_longlong
    lib.chol_free.argtypes = [ctypes.c_void_p]
    lib.chol_free.restype = None
    if hasattr(lib, "chol_diag"):
        lib.chol_diag.argtypes = [ctypes.c_void_p,
                                  ctypes.POINTER(ctypes.c_longlong), ctypes.c_int]
        lib.chol_diag.restype = ctypes.c_int
    if hasattr(lib, "chol_last_error"):
        lib.chol_last_error.argtypes = []
        lib.chol_last_error.restype = ctypes.c_char_p
    _LIB = lib
    return lib


class RustChol:
    """Handle to a Rust sparse LL^T factor of a CSC matrix (EigenChol clone)."""

    def __init__(self, B, ordering=1):
        lib = load_rust_chol()
        self._lib = lib
        self.h = None
        if B.dtype != np.float64 or B.indices.dtype != np.int32 \
                or B.indptr.dtype != np.int32:
            raise TypeError("C ABI requires FP64 values and int32 sparse indices")
        if B.shape[0] != B.shape[1]:
            raise ValueError("Cholesky matrix must be square")
        B = B.tocsc()
        B.sort_indices()
        status = ctypes.c_int()
        self.n = B.shape[0]
        self.h = lib.chol_factor(self.n, B.nnz,
                                 B.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                 B.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                 B.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                 ordering, ctypes.byref(status))
        if not self.h:
            msg = b""
            if hasattr(lib, "chol_last_error"):
                msg = lib.chol_last_error() or b""
            raise RuntimeError(f"Rust Cholesky failed: status={status.value} {msg.decode(errors='replace')}")

    def solve(self, b):
        b = np.ascontiguousarray(b, dtype=np.float64)
        if b.shape != (self.n,):
            raise ValueError("Expected a one-dimensional RHS of matrix dimension")
        x = np.empty_like(b)
        self._lib.chol_solve(self.h, self.n,
                             b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                             x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return x

    def solve2(self, b0, b1):
        """Two right-hand sides through the factor in ONE pass."""
        b0 = np.ascontiguousarray(b0, dtype=np.float64)
        b1 = np.ascontiguousarray(b1, dtype=np.float64)
        if b0.shape != (self.n,) or b1.shape != (self.n,):
            raise ValueError("Expected one-dimensional RHS of matrix dimension")
        x0, x1 = np.empty_like(b0), np.empty_like(b1)
        P = ctypes.POINTER(ctypes.c_double)
        self._lib.chol_solve2(self.h, self.n,
                              b0.ctypes.data_as(P), x0.ctypes.data_as(P),
                              b1.ctypes.data_as(P), x1.ctypes.data_as(P))
        return x0, x1

    def solve_k(self, rhs):
        """k right-hand sides (list of (n,) arrays) in one pass over the factor."""
        k = len(rhs)
        cols = [np.ascontiguousarray(r, dtype=np.float64) for r in rhs]
        outs = [np.empty_like(c) for c in cols]
        dp = ctypes.POINTER(ctypes.c_double)
        bp = (dp * k)(*[c.ctypes.data_as(dp) for c in cols])
        xp = (dp * k)(*[o.ctypes.data_as(dp) for o in outs])
        self._lib.chol_solve_k(self.h, self.n, bp, xp, ctypes.c_int(k))
        return outs

    @property
    def nnz(self):
        return self._lib.chol_nnz(self.h)

    def diag(self):
        """[nnz_L, npanels, row_entries, stored_values, bytes, n]."""
        if not hasattr(self._lib, "chol_diag"):
            return None
        buf = (ctypes.c_longlong * 6)()
        self._lib.chol_diag(self.h, buf, 6)
        return list(buf)

    def close(self):
        if self.h:
            self._lib.chol_free(self.h)
            self.h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class RustSharedChol:
    """``psn2d.memopt.SharedChol`` on the Rust factor, fused mirror-pair solve.

    Same construction, same ``solve(b)`` contract (A x = b for the ORIGINAL
    unscaled A), same ``factor_nnz`` accounting; ``fused=True`` replaces the two
    back-to-back single-RHS solves per mirror pair with one ``chol_solve2``.
    """

    factor_cls = RustChol

    def __init__(self, A, d, pairs, ordering=1, check=True, fused=True):
        self.d = d
        self.n = A.shape[0]
        self.pairs = pairs
        self.fused = bool(fused and hasattr(self.factor_cls, "solve2"))
        self.factors = []
        self.symmetry_errors = []
        B = A.multiply(d[:, None]).tocsc()
        if check:
            delta = B - B.T
            err = float(abs(delta).max()) / float(abs(B).max()) if delta.nnz else 0.
            if not np.isfinite(err) or err > 1e-12:
                raise ValueError(f"Row-scaled matrix not finite/symmetric: {err}")
            self.scaled_asymmetry = err
            del delta
        for c, ct, s in pairs:
            block = B[c, :][:, c].tocsc()
            if check:
                twin = B[ct, :][:, ct].multiply(s[:, None]).multiply(s[None, :]).tocsc()
                delta = block - twin
                err = float(abs(delta).max()) / float(abs(block).max()) if delta.nnz else 0.
                if not np.isfinite(err) or err > 1e-12:
                    raise ValueError(f"Geometry sharing not finite/exact: {err}")
                self.symmetry_errors.append(err)
                del twin, delta
            self.factors.append(self.factor_cls(block, ordering=ordering))
        self.factor_nnz = sum(f.nnz for f in self.factors)
        self.L_and_permutations_payload_bytes = sum(
            f.nnz * 12 + (len(c) + 1) * 4 + len(c) * 8
            for f, (c, ct, s) in zip(self.factors, self.pairs))

    def solve(self, b):
        rhs = self.d * b
        x = np.empty(self.n, dtype=np.float64)
        # ``fused`` may be set after construction (a runtime patch of
        # ``memopt.SharedChol.solve`` has no __init__ to set it in)
        if getattr(self, "fused", False):
            for f, (c, ct, s) in zip(self.factors, self.pairs):
                x[c], t = f.solve2(rhs[c], s * rhs[ct])
                x[ct] = s * t
            return x
        for f, (c, ct, s) in zip(self.factors, self.pairs):
            x[c] = f.solve(rhs[c])
            x[ct] = s * f.solve(s * rhs[ct])
        return x

    @property
    def nnz(self):
        return self.factor_nnz

    def close(self):
        for f in self.factors:
            f.close()
        self.factors = []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# fused B.3 source update (not a Cholesky method -- it lives in the same .so
# because that is the library the accelerated path already loads)
# --------------------------------------------------------------------------- #

_SRC = None          # (fn, status) once resolved, False if the .so lacks it


def has_source_update():
    """True when the loaded library exports ``psn_src_update``."""
    global _SRC
    if _SRC is None:
        try:
            lib = load_rust_chol()
            fn = lib.psn_src_update
        except (AttributeError, OSError):
            _SRC = False
            return False
        dp = ctypes.POINTER(ctypes.c_double)
        fn.argtypes = [ctypes.c_int, ctypes.c_int, dp, dp, dp, ctypes.c_double,
                       dp, dp, ctypes.POINTER(ctypes.c_int)]
        fn.restype = None
        _SRC = (fn, ctypes.c_int())
    return _SRC is not False


def source_update(ng, nodes, base, nu, chi, lam, qim, out):
    """``out[g,n,k] = sum_G (base[g,G,n] + chi[g]*nu[G,n]/lam) * qim[G,n,k]``.

    Bit-for-bit the NumPy loop it replaces (see the header comment in
    ``src/lib.rs``); the point is that the coefficient array is never built.
    All arrays must be C-contiguous FP64 of the shapes documented there.
    """
    if not has_source_update():
        raise RuntimeError("libpsn_chol.so has no psn_src_update (rebuild it)")
    fn, status = _SRC
    fn(ctypes.c_int(ng), ctypes.c_int(nodes),
       base.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
       nu.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
       chi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
       ctypes.c_double(float(lam)),
       qim.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
       out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
       ctypes.byref(status))
    if status.value:
        msg = b""
        if hasattr(_LIB, "chol_last_error"):
            msg = _LIB.chol_last_error() or b""
        raise RuntimeError(f"psn_src_update failed: status={status.value} "
                           f"{msg.decode(errors='replace')}")
