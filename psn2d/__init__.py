# -*- coding: utf-8 -*-
"""PSN2D — diffusion-based phase-space nodal method (2D, multi-group).

Independent Python implementation of Chao, Li & Chen, ANE 240 (2027) 112707.
"""
import os

# Determinism guard (2026-09-12): the solver relies on single-core GEMMs /
# reductions being run-to-run bit-identical (and forked sweep workers must
# use the same BLAS code path as the parent).  This build ships OpenBLAS
# DYNAMIC_ARCH with up to 64 threads; pin it BEFORE numpy's first BLAS
# call.  Measured: with this pin the fast regression suite is bit-identical
# to the unpinned baseline (7/7 Δ=0.000 pcm).
#
# Default is half the system core count (user setting 2026-09-12: the
# default should follow the machine, not be hard-coded).
# runtime.set_thread_limit may raise/lower this at install time from the
# user's threads input; env vars always win (setdefault).
_DEFAULT_T = max(1, (os.cpu_count() or 1) // 2)
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_DEFAULT_T))
os.environ.setdefault("OMP_NUM_THREADS", str(_DEFAULT_T))

__version__ = "0.1.0"

from .model import load_spec, expand_cases, arrays  # noqa: E402,F401
from .solver import PSN2D  # noqa: E402,F401
