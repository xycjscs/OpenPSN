# -*- coding: utf-8 -*-
"""Runtime parallelism budget.

``threads`` is a USER INPUT (YAML ``solver.threads`` / CLI ``--threads``,
default 24, user setting 2026-09-12): it is the TOTAL parallel-unit budget
for the whole process tree — never exceeded:

  * parent factorization phase: one process, BLAS/OpenMP <= N threads
  * parent prefactor thread pool: W workers x (N // W) BLAS threads <= N
  * forked sweep pool: <= N worker processes, each pinned to 1 BLAS thread

``set_thread_limit`` is safe to call at any time (OpenBLAS reads its env
only at load, so we also call the C runtime setter for the already-loaded
library).
"""
import ctypes
import os

_LIB = None


def _openblas():
    global _LIB
    if _LIB is None:
        import numpy  # noqa: F401  (loads libopenblas into the address space)
        lib = ctypes.CDLL(None, mode=ctypes.RTLD_GLOBAL)
        # CDLL(None) = the process itself; resolve the already-loaded lib
        try:
            lib.openblas_set_num_threads
        except AttributeError:
            raise OSError("libopenblas not loaded in this process")
        lib.openblas_set_num_threads.argtypes = [ctypes.c_int]
        lib.openblas_get_num_threads.restype = ctypes.c_int
        _LIB = lib
    return _LIB


def set_thread_limit(n):
    """Cap intra-op parallelism at n (>= 1).  Env vars cover child
    processes and future library loads; the C setter covers the
    already-loaded OpenBLAS in THIS process."""
    n = int(n)
    if n < 1:
        raise ValueError(f"threads must be >= 1, got {n}")
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    try:
        _openblas().openblas_set_num_threads(n)
    except OSError:
        pass
    return n


def get_thread_limit():
    try:
        return int(_openblas().openblas_get_num_threads())
    except Exception:
        return int(os.environ.get("OPENBLAS_NUM_THREADS", "1"))


def default_threads():
    """Default threads input = half the system's core count (user setting
    2026-09-12: default should follow the machine, not be hard-coded)."""
    return max(1, (os.cpu_count() or 1) // 2)


DEFAULT_MEM_LIMIT_GB = 32.0
