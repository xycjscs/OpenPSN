# -*- coding: utf-8 -*-
"""PSN2D — diffusion-based phase-space nodal method (2D, multi-group).

Independent Python implementation of Chao, Li & Chen, ANE 240 (2027) 112707.
"""
__version__ = "0.1.0"

from .model import load_spec, expand_cases, arrays  # noqa: F401
from .solver import PSN2D  # noqa: F401
