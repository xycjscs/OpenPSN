#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify the RECTANGULAR node closed forms (psn2d/node_rect.py).

Two independent gates:
  A. Square reduction: node_state_rect(hx=a, hy=a) must match the validated
     square node.node_state term-by-term (same math, ~1e-15 association noise).
  B. Brute-force quadrature: for a genuinely rectangular node (hx != hy),
     every closed-form output is checked against direct high-order numerical
     integration of the base function
        phi = Ax cosh(kx) + Bx sinh(kx) + Ay cosh(ky) + By sinh(ky) + C
              + (q0 + q1x P1(2x/hx) + q1y P1(2y/hy)
                 + q2x P2(2x/hx) + q2y P2(2y/hy))/St
        k^2 = 2 St^2/mu^2,   Jvec = -(mu^2/2St) grad phi
   Face current  J_f = (1/L) int_f [n + cc*u_mn].Jvec dS
   Face flux     (1/L) int_f phi dS
   phi_bar       (1/A) int phi dA
   m1x,m2x       3*<P1(2x/hx) phi>, 5*<P2(2x/hx) phi>  (area average)
   m1y,m2y       3*<P1(2y/hy) phi>, 5*<P2(2y/hy) phi>

Run:  /opt/data/venvs/omconda/bin/python tests/verify_rect_node.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from psn2d.node import (node_state, P1, P2, ty_polar_set, node_alphas)
from psn2d.node_rect import (node_state_rect, node_params_rect,
                             face_currents_rect, face_fluxes_rect)

rng = np.random.default_rng(20260908)
TY = ty_polar_set(3)
M = 12
dphi = np.pi / M

MAXREL = 0.0
NFAIL = 0


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.max(np.abs(a - b) / np.maximum(1e-12, np.abs(b)))
    global MAXREL, NFAIL
    MAXREL = max(MAXREL, d)
    if d > 1e-9:
        NFAIL += 1
        print(f"    !! rel diff {d:.3e}  closed={a}  num={b}")
    return d


# ------------------------------------------------------------------ #
# A. square reduction
# ------------------------------------------------------------------ #
print("A. square reduction  node_state_rect(hx=hy) vs node.node_state")
for trial in range(40):
    a = float(rng.uniform(0.3, 3.0))
    St = float(rng.uniform(0.2, 3.0))
    i = int(rng.integers(3))
    m = int(rng.integers(M))
    mu = TY["mu"][i]
    phi_m = (m + 0.5) * dphi
    J4 = rng.standard_normal(4) * 0.01
    q = rng.standard_normal(5) * 0.01
    ref = node_state(J4, q, St, mu, a, phi_m, dphi)
    new = node_state_rect(J4, q, node_params_rect(St, mu, a, a), phi_m, dphi,
                          node_alphas(mu, 1.0, phi_m, dphi))
    for key in ("Phi", "phi_bar", "mom"):
        rel(new[key], ref[key])
    rel(new["Phibar"], ref["Phibar"])
    rel(new["J4check"], ref["J4check"])
    for key in ("Ax", "Bx", "Ay", "By", "C"):
        rel(new[key], ref[key])
print(f"    40 random trials, max rel diff = {MAXREL:.3e}  "
      f"{'PASS' if MAXREL < 1e-9 else 'FAIL'}")

# ------------------------------------------------------------------ #
# B. quadrature on genuinely rectangular nodes
# ------------------------------------------------------------------ #
print("B. quadrature  rectangular closed forms vs brute-force integration")
GLN = 300
GLX, GLW = np.polynomial.legendre.leggauss(GLN)
# integrands are sums of x-only * y-only factors (separable), so 1D
# Gauss-Legendre in each direction is exact to machine precision.


def face1d(F, L):
    """(1/L) int_{-L/2}^{L/2} F(t) dt via Gauss-Legendre."""
    return 0.5 * np.dot(GLW, F(0.5 * L * GLX))


def quadrature(hx, hy, St, mu, Ax, Bx, Ay, By, C, q, phi_m):
    k = np.sqrt(2.0) * St / mu
    q1x, q1y, q2x, q2y = q[1], q[2], q[3], q[4]

    def phi(x, y):
        u = 2.0 * x / hx
        v = 2.0 * y / hy
        return (Ax * np.cosh(k * x) + Bx * np.sinh(k * x)
                + Ay * np.cosh(k * y) + By * np.sinh(k * y) + C
                + (q[0] + q1x * P1(u) + q1y * P1(v)
                   + q2x * P2(u) + q2y * P2(v)) / St)

    dphix = lambda x: (k * Ax * np.sinh(k * x) + k * Bx * np.cosh(k * x)
                       + (2.0 * q1x / hx + 6.0 * q2x * (2.0 * x / hx) / hx) / St)
    dphiy = lambda y: (k * Ay * np.sinh(k * y) + k * By * np.cosh(k * y)
                       + (2.0 * q1y / hy + 6.0 * q2y * (2.0 * y / hy) / hy) / St)
    p = mu * mu / (2.0 * St)
    Jx = lambda x: -p * dphix(x)
    Jy = lambda y: -p * dphiy(y)
    c_c = np.sin(dphi) / dphi
    C2, S2 = np.cos(2 * phi_m), np.sin(2 * phi_m)
    Jxp = (1 + c_c * C2) * Jx(hx / 2) + c_c * S2 * face1d(Jy, hy)
    Jxm = -(1 + c_c * C2) * Jx(-hx / 2) - c_c * S2 * face1d(Jy, hy)
    Jyp = (1 - c_c * C2) * Jy(hy / 2) + c_c * S2 * face1d(Jx, hx)
    Jym = -(1 - c_c * C2) * Jy(-hy / 2) - c_c * S2 * face1d(Jx, hx)
    J4 = np.array([Jxp, Jxm, Jyp, Jym])
    Phi = np.array([
        face1d(lambda y: phi(hx / 2, y), hy),
        face1d(lambda y: phi(-hx / 2, y), hy),
        face1d(lambda x: phi(x, hy / 2), hx),
        face1d(lambda x: phi(x, -hy / 2), hx),
    ])

    def area_avg(F):
        X = 0.5 * hx * GLX
        Y = 0.5 * hy * GLX
        M = np.stack([F(float(x), Y) for x in X])
        return 0.25 * (GLW @ M) @ GLW

    phi_bar = area_avg(phi)
    mom = np.array([
        phi_bar,
        3 * area_avg(lambda x, y: P1(2.0 * x / hx) * phi(x, y)),
        3 * area_avg(lambda x, y: P1(2.0 * y / hy) * phi(x, y)),
        5 * area_avg(lambda x, y: P2(2.0 * x / hx) * phi(x, y)),
        5 * area_avg(lambda x, y: P2(2.0 * y / hy) * phi(x, y)),
    ])
    return J4, Phi, phi_bar, mom


# realistic parameter range (St ~ 0.05-1.5, widths ~ 0.1-1.3 cm, TY mu >= 0.167)
for trial in range(40):
    hx = float(rng.uniform(0.1, 1.3))
    hy = float(rng.uniform(0.1, 1.3))
    if abs(hx - hy) < 0.15:      # keep them genuinely rectangular
        hy += 0.5
    St = float(rng.uniform(0.05, 1.5))
    i = int(rng.integers(3))
    m = int(rng.integers(M))
    mu = TY["mu"][i]
    phi_m = (m + 0.5) * dphi
    Ax = float(rng.normal(0, 0.05))
    Bx = float(rng.normal(0, 0.05))
    Ay = float(rng.normal(0, 0.05))
    By = float(rng.normal(0, 0.05))
    C = float(rng.normal(0, 0.05))
    q = rng.standard_normal(5) * 0.01
    pr = node_params_rect(St, mu, hx, hy)

    J4_ref, Phi_ref, pb_ref, mom_ref = quadrature(
        hx, hy, St, mu, Ax, Bx, Ay, By, C, q, phi_m)
    J4_cf = np.array([
        face_currents_rect(Ax, Bx, Ay, By, q, pr, phi_m, dphi)[j]
        for j in range(4)])
    Phi_cf = face_fluxes_rect(Ax, Bx, Ay, By, C, q, pr)
    pb_cf = C + q[0] / St + (2 * pr["sx"] / pr["ka_x"]) * Ax \
        + (2 * pr["sy"] / pr["ka_y"]) * Ay
    # volume-average closed form: A.8c generalized
    mom_cf = np.array([
        pb_cf,
        3 * pr["Il_x"] * Bx + q[1] / St,
        3 * pr["Il_y"] * By + q[2] / St,
        5 * pr["Iq_x"] * Ax + q[3] / St,
        5 * pr["Iq_y"] * Ay + q[4] / St,
    ])
    if trial < 3 or NFAIL:
        print(f"  trial {trial}: hx={hx:.4f} hy={hy:.4f} St={St:.3f} "
              f"i={i} m={m}")
    rel(J4_cf, J4_ref)
    rel(Phi_cf, Phi_ref)
    rel(pb_cf, pb_ref)
    rel(mom_cf, mom_ref)

# self-consistency: J4 -> coeffs -> J4 round trip
print("C. round trip  J4 -> node_coeffs_rect -> J4check")
RT = 0.0
for trial in range(30):
    hx = float(rng.uniform(0.3, 2.5))
    hy = hx + float(rng.uniform(0.2, 1.5))
    St = float(rng.uniform(0.2, 3.0))
    i = int(rng.integers(3))
    m = int(rng.integers(M))
    mu = TY["mu"][i]
    phi_m = (m + 0.5) * dphi
    J4 = rng.standard_normal(4) * 0.01
    q = rng.standard_normal(5) * 0.01
    st = node_state_rect(J4, q, node_params_rect(St, mu, hx, hy), phi_m, dphi,
                         node_alphas(mu, 1.0, phi_m, dphi))
    d = np.max(np.abs(st["J4check"] - J4)) / np.max(np.abs(J4))
    RT = max(RT, d)
    if d > 1e-10:
        NFAIL += 1
        print(f"    !! round-trip rel {d:.3e}")
print(f"    30 trials, max round-trip rel = {RT:.3e}  "
      f"{'PASS' if RT < 1e-10 else 'FAIL'}")

print()
print(f"TOTAL: max rel diff {MAXREL:.3e}; failures={NFAIL}")
print("ALL PASS" if (MAXREL < 1e-9 and NFAIL == 0 and RT < 1e-10) else "FAILURES PRESENT")
sys.exit(0 if (MAXREL < 1e-9 and NFAIL == 0 and RT < 1e-10) else 1)
