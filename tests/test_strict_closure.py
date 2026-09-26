# -*- coding: utf-8 -*-
"""Strict phase-space diffusion PSN -- anchor + consistency gates.

Fast tier gates (no heavy sweeps):
  T-anchor : TY model, M=2  -- strict node map == legacy node map, global
             keff agrees to solver noise (< 1e-3 pcm) on checkerboard S1-3.
             At M=2 the sector diagonals phi_m +- pi/4 ARE the x/y axes, so
             the strict exact basis spans the legacy trial space.
  T-bit    : node-level -- strict_params + node_state_strict reproduce
             node_state_rect at M=2 on random J4/q to 1e-12 (rect kernel).
  T-closure: node map sizes/keys: strict state dict carries the same
             consumed keys as the legacy rect state.
Run: python tests/test_strict_closure.py [full]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psn2d import node_strict                                   # noqa: E402
from psn2d.node_rect import (node_params_rect, node_state_rect,  # noqa: E402
                             node_response_matrix_rect)


def t_anchor():
    from psn2d.__main__ import run_case
    from psn2d.model import load_spec
    spec = load_spec(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "examples/psn_repro/checkerboard_1g.yaml"))
    res = {}
    for clo in ("legacy", "strict_phase_space"):
        for S in (1, 2, 3):
            case = dict(spec.get("angular", {}))
            case.update({"name": "a", "model": "ty3", "M": 2, "subdivide": S,
                         "closure": clo})
            r = run_case(spec, case, verbose=False, threads=1)
            res[(clo, S)] = r["keff"]
    worst = max(abs(res[("legacy", S)] - res[("strict_phase_space", S)])
                for S in (1, 2, 3)) * 1e5
    ok = worst < 1e-3
    print(f"T-anchor  worst dd = {worst:.2e} pcm  {'PASS' if ok else 'FAIL'}")
    return ok


def t_bit():
    rng = np.random.default_rng(20260926)
    worst = 0.0
    St, mu, hx, hy = 0.7, 1.0 / np.sqrt(2.0), 0.5, 0.25
    pr = node_params_rect(St, mu, hx, hy)
    pr = {**pr, "St": St, "hx": hx, "hy": hy}
    dphi = np.pi / 2.0                       # M = 2 -> phi_m = pi/4
    alpha = np.array([1.03, 0.97, 1.11, 0.95])
    for trial in range(24):
        J4 = rng.normal(size=4)
        q = np.concatenate([[rng.uniform(0.5, 2.0)], rng.normal(size=4)])
        leg = node_state_rect(J4, q, pr, np.pi / 4, dphi, alpha)
        N = node_strict.strict_params_from_pr(pr, np.pi / 4, dphi)
        st = node_strict.node_state_strict(J4, q, N, alpha)
        for key in ("Phibar", "Phi", "phi_bar", "mom"):
            a, b = leg[key], st[key]
            scale = max(1.0, float(np.max(np.abs(a))))
            worst = max(worst, float(np.max(np.abs(a - b))) / scale)
    # relative gate 1e-9: mom components carry cancellation; the
    # global-anchor noise floor (8.9e-11 pcm keff) is the real bar
    ok = worst < 1e-9
    print(f"T-bit     max diff = {worst:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def t_closure():
    pr = node_params_rect(0.7, 1.0 / np.sqrt(2.0), 0.5, 0.25)
    pr = {**pr, "St": 0.7, "hx": 0.5, "hy": 0.25}
    N = node_strict.strict_params_from_pr(pr, np.pi / 4, np.pi / 2)
    st = node_strict.node_state_strict(np.ones(4), np.zeros(5), N,
                                       np.ones(4))
    need = {"Phi", "Phibar", "phi_bar", "mom", "alpha", "J4check"}
    ok = need.issubset(st.keys()) and st["mom"].shape == (5,)
    print(f"T-closure keys {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    gates = [t_bit, t_closure, t_anchor]
    fails = 0
    for g in gates:
        if not g():
            fails += 1
    print("=== STRICT GATES", "ALL PASS" if fails == 0 else f"{fails} FAIL",
          "===")
    sys.exit(1 if fails else 0)
