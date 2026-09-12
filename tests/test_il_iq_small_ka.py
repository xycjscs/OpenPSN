#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression test: small-ka (thin-node) branch of Il/Iq source moments.

2026-09-11: _il_iq / node_params / node_params_generic had a small-ka
(ka <= 1e-3) branch returning the CONSTANTS Il = 1/3, Iq = 1/5.  The correct
small-ka limits are Il -> ka/6, Iq -> ka^2/60 (both -> 0), so the old branch
was off by a factor ~2000 (Il) and ~3e7 (Iq) at the branch boundary.

Never triggered by the C5G7 studies (min ka = 4.8e-3 in RECTUO2/MOX/CORE and
PIN6 at S=6, TY model, smallest mu, smallest node), but this test guards the
branch so it cannot silently regress.

Reference: the defining closed forms evaluated in 60-digit decimal
arithmetic (c = cosh(ka/2), v = sinh(ka/2)/(ka/2) point values):
    Il = (2/ka)(c - v)      = u/3 + u^3/30 + u^5/840 + O(u^7),  u = ka/2
    Iq = (2/ka)(s - 3 Il)   = u^2/15 + u^4/210 + O(u^6)
Note the full-precision double formula itself loses accuracy as ka -> 0
(~5% relative on Iq at ka = 1e-3 by cancellation), which is exactly why the
series branch exists.

Run:  /opt/data/venvs/om11/bin/python tests/test_il_iq_small_ka.py
"""
import os
import sys
import decimal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from psn2d.node import node_params, node_params_generic
from psn2d.node_rect import _il_iq

decimal.getcontext().prec = 60


def ref_il_iq(ka_str):
    """60-digit reference for Il, Iq at optical thickness ka (decimal)."""
    D = decimal.Decimal
    ka = D(ka_str)
    u = ka / 2
    eu = u.exp()
    c = (eu + 1 / eu) / 2        # cosh(u)
    s = (eu - 1 / eu) / 2        # sinh(u)
    v = s / u
    Il = (2 / ka) * (c - v)
    Iq = (2 / ka) * (s - 3 * Il)
    return float(Il), float(Iq)


def series_il_iq(ka):
    u = 0.5 * ka
    Il = u * (1.0 / 3.0 + u * u * (1.0 / 30.0 + u * u / 840.0))
    Iq = u * u * (1.0 / 15.0 + u * u / 210.0)
    return Il, Iq


def full_il_iq(ka):
    c = np.cosh(0.5 * ka)
    s = np.sinh(0.5 * ka)
    Il = (2.0 / ka) * (c - 2.0 * s / ka)
    Iq = (2.0 / ka) * (s - 3.0 * Il)
    return Il, Iq


maxerr = 0.0
# 1. branch value vs 60-digit reference (inside the branch)
print("1. branch vs 60-digit reference:")
for ka_s in ("1e-4", "3e-4", "9.99e-4"):
    Il_b, Iq_b = _il_iq(float(ka_s))  # all < 1e-3 -> inside the branch
    Il_r, Iq_r = ref_il_iq(ka_s)
    rel = max(abs(Il_b - Il_r) / Il_r, abs(Iq_b - Iq_r) / Iq_r)
    maxerr = max(maxerr, rel)
    print(f"    ka={ka_s:>7s}: Il {Il_b:.6e} (ref {Il_r:.6e})  "
          f"Iq {Iq_b:.6e} (ref {Iq_r:.6e})")
print(f"   max rel = {maxerr:.3e}  {'PASS' if maxerr < 1e-10 else 'FAIL'}")

# 2. continuity at the branch boundary (branch side vs reference)
Il_b, Iq_b = _il_iq(1e-3 - 1e-15)
Il_r, Iq_r = ref_il_iq("1e-3")
c2 = max(abs(Il_b - Il_r) / Il_r, abs(Iq_b - Iq_r) / Iq_r)
print(f"2. boundary ka=1e-3 (branch side vs ref): rel = {c2:.3e}  "
      f"{'PASS' if c2 < 1e-9 else 'FAIL'}")

# 3. the full double formula is the inaccurate one at small ka (documents why
#    the branch exists): report, don't gate on it.
Il_f, Iq_f = full_il_iq(1e-3)
print(f"3. full double formula at ka=1e-3: Il rel err "
      f"{abs(Il_f - Il_r) / Il_r:.2e}, Iq rel err {abs(Iq_f - Iq_r) / Iq_r:.2e}"
      f"  (cancellation; branch corrects this)")

# 4. moments -> 0 as ka -> 0
Il_t, Iq_t = _il_iq(1e-8)
ok4 = Il_t < 1e-6 and Iq_t < 1e-12
print(f"4. ka=1e-8: Il={Il_t:.3e} (expect ~1.667e-9), Iq={Iq_t:.3e} "
      f"(expect ~1.667e-18)  {'PASS' if ok4 else 'FAIL'}")

# 5. node-parameter entry points (TY: k = sqrt(2) St/mu; generic: k = sqrt(3) St)
k_ty = np.sqrt(2.0) * 1.0 / 0.932954
pr = node_params(St=1.0, mu=0.932954, a=1e-4)
ok5 = abs(pr["Il"] - series_il_iq(k_ty * 1e-4)[0]) < 1e-15 \
    and abs(pr["p"] - 0.5 * 0.932954 ** 2) < 1e-15
print(f"5. node_params:      Il={pr['Il']:.6e} p={pr['p']:.8f}  "
      f"{'PASS' if ok5 else 'FAIL'}")
prg = node_params_generic(St=1.0, a=1e-4, theta_i=0.5, dtheta_i=0.1)
k_g = np.sqrt(3.0)
ok6 = abs(prg["Il"] - series_il_iq(k_g * 1e-4)[0]) < 1e-15
print(f"6. node_params_generic: Il={prg['Il']:.6e}  "
      f"{'PASS' if ok6 else 'FAIL'}")

allok = maxerr < 1e-10 and c2 < 1e-9 and ok4 and ok5 and ok6
print("\nALL PASS" if allok else "\nFAILURES PRESENT")
sys.exit(0 if allok else 1)
