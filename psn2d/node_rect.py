# -*- coding: utf-8 -*-
"""PSN2D RECTANGULAR node model — generalization of node.py (square) to
nodes of width hx (x) and height hy (y), centered at the origin:
    x in [-hx/2, hx/2],  y in [-hy/2, hy/2].

Same base function / angular conventions as the square node
(Chao et al., ANE 240 (2027) 112707):

    phi = Ax cosh(kx) + Bx sinh(kx) + Ay cosh(ky) + By sinh(ky) + C
          + (q0 + q1x P1(2x/hx) + q1y P1(2y/hy)
             + q2x P2(2x/hx) + q2y P2(2y/hy))/St

Generalization rule: every length scale splits by direction.
  x-direction quantities use hx (sx = sinh(k hx/2), cx = cosh(k hx/2),
  ka_x = k hx, Il_x/Iq_x from ka_x)
  y-direction quantities use hy (sy, cy, ka_y, Il_y/Iq_y)
  cross terms (e.g. the Ay part of the x-face flux) use the PERPENDICULAR
  width.

Validated (tests/verify_rect_node.py):
  A. hx = hy -> matches the validated square node to ~2e-14
  B. genuinely rectangular nodes -> matches 300-pt Gauss-Legendre
     quadrature of the base function to ~5e-12 (face currents, face flux,
     volume average, all four paraboloidal moments)
  C. J4 -> coefficients -> J4 round trip 7.5e-15

Face quantities (surface averages):
  JXp = -p [k Ax sx + k Bx cx + (2 q1x + 6 q2x)/(hx St)]
  JXm = -p [-k Ax sx + k Bx cx + (2 q1x - 6 q2x)/(hx St)]
  JYP = -p [k Ay sy + k By cy + (2 q1y + 6 q2y)/(hy St)]
  JYM = -p [-k Ay sy + k By cy + (2 q1y - 6 q2y)/(hy St)]
  JYavg = -p (2 By sy + 2 q1y/St)/hy          (G_y identity, x-faces)
  JXavg = -p (2 Bx sx + 2 q1x/St)/hx          (G_x identity, y-faces)
  Jxp = (1 + cc C2) JXp + cc S2 JYavg         (cc = sin dphi/dphi)
  Jxm = -(1 + cc C2) JXm - cc S2 JYavg
  Jyp = (1 - cc C2) JYP + cc S2 JXavg
  Jym = -(1 - cc C2) JYM - cc S2 JXavg

Coefficients from J4 (sequential exact solve):
  Bx = -(Jxp+Jxm + 4(1+cc2) p q1x/(St hx)) / (2 (1+cc2) p k cx)
  By = -(Jyp+Jym + 4(1-cc2) p q1y/(St hy)) / (2 (1-cc2) p k cy)
  Ax = (Jxm-Jxp - 12(1+cc2) p q2x/(St hx) - 2 cc2 S2 p (2 By sy + 2 q1y/St)/hy)
       / (2 (1+cc2) p k sx)
  Ay = (Jym-Jyp - 12(1-cc2) p q2y/(St hy) - 2 cc2 S2 p (2 Bx sx + 2 q1x/St)/hx)
       / (2 (1-cc2) p k sy)

Volume balance (all-outward currents, face lengths hy, hx):
  phi_bar = q0/St - [(Jxp+Jxm)/hx + (Jyp+Jym)/hy]/St
  C       = phi_bar - q0/St - (2 sx/ka_x) Ax - (2 sy/ka_y) Ay

Face flux (surface average, before alpha):
  Phi_x+ = cx Ax + sx Bx + (2 sy/ka_y) Ay + C + (q0 + q1x + q2x)/St
  Phi_x- = cx Ax - sx Bx + (2 sy/ka_y) Ay + C + (q0 - q1x + q2x)/St
  Phi_y+ = cy Ay + sy By + (2 sx/ka_x) Ax + C + (q0 + q1y + q2y)/St
  Phi_y- = cy Ay - sy By + (2 sx/ka_x) Ax + C + (q0 - q1y + q2y)/St

Paraboloidal source moments (B.1-B.2, x-moments use ka_x, y-moments ka_y):
  m0  = phi_bar
  m1x = 3 Il_x Bx + q1x/St      m1y = 3 Il_y By + q1y/St
  m2x = 5 Iq_x Ax + q2x/St      m2y = 5 Iq_y Ay + q2y/St

Two angular models share the SAME rectangular closed forms; only the
parameter dictionary differs:
  TY (restricted):  k = sqrt(2) St/mu,  p = mu^2/(2 St)   (node_params_rect)
  generic:          k = sqrt(3) St,     p = beta_i/(3 St) (node_params_rect_generic)
alpha (angular surface factor) is geometry-free: use node.node_alphas /
node.node_alphas_generic as in the square model.
"""
import numpy as np

from .node import (P1, P2, node_alphas, node_alphas_generic,
                   node_params_generic)


def _il_iq(ka):
    if ka > 1e-3:
        c = np.cosh(0.5 * ka)
        s = np.sinh(0.5 * ka)
        Il = (2.0 / ka) * (c - 2.0 * s / ka)
        Iq = (2.0 / ka) * (s - 3.0 * Il)
    else:
        # small-ka series (cancellation-free, continuous at ka = 1e-3):
        #   u = ka/2
        #   Il = (1/u)(cosh u - sinh u/u) = u/3 + u^3/30 + u^5/840 + O(u^7)
        #   Iq = v - (6/ka) Il             = u^2/15 + u^4/210 + O(u^6)
        # Both moments -> 0 as ka -> 0.  (2026-09-11: the old branch returned
        # constants Il=1/3, Iq=1/5 — off by O(1/ka), a factor ~2000 at the
        # branch boundary.  Never triggered by c5g7 runs (min ka = 4.8e-3);
        # regression test: tests/test_il_iq_small_ka.py.)
        u = 0.5 * ka
        Il = u * (1.0 / 3.0 + u * u * (1.0 / 30.0 + u * u / 840.0))
        Iq = u * u * (1.0 / 15.0 + u * u / 210.0)
    return Il, Iq


def node_params_rect(St, mu, hx, hy):
    """Rectangular node parameters, TY (restricted 2D) model."""
    k = np.sqrt(2.0) * St / mu
    ka_x = k * hx
    ka_y = k * hy
    sx = np.sinh(0.5 * ka_x)
    cx = np.cosh(0.5 * ka_x)
    sy = np.sinh(0.5 * ka_y)
    cy = np.cosh(0.5 * ka_y)
    p = mu * mu / (2.0 * St)
    Il_x, Iq_x = _il_iq(ka_x)
    Il_y, Iq_y = _il_iq(ka_y)
    return {"k": k, "ka_x": ka_x, "ka_y": ka_y, "sx": sx, "cx": cx,
            "sy": sy, "cy": cy, "p": p, "Il_x": Il_x, "Il_y": Il_y,
            "Iq_x": Iq_x, "Iq_y": Iq_y, "St": St, "hx": hx, "hy": hy}


def node_params_rect_generic(St, hx, hy, theta_i, dtheta):
    """Rectangular node parameters, generic PSN model (paper Sec 2).

    Same closed forms, diffusion-limited k = sqrt(3) St and paper (2.8d)
    beta_i — the geometry enters only through hx/hy (see module docstring).
    """
    k = np.sqrt(3.0) * St
    beta = (1.0 - 0.25 * np.cos(dtheta) - 0.25 * np.cos(2.0 * theta_i)
            - 0.5 * np.cos(2.0 * theta_i) * np.cos(dtheta))
    p = beta / (3.0 * St)
    ka_x = k * hx
    ka_y = k * hy
    sx = np.sinh(0.5 * ka_x)
    cx = np.cosh(0.5 * ka_x)
    sy = np.sinh(0.5 * ka_y)
    cy = np.cosh(0.5 * ka_y)
    Il_x, Iq_x = _il_iq(ka_x)
    Il_y, Iq_y = _il_iq(ka_y)
    alpha_polar = 0.25 * (dtheta - np.cos(2.0 * theta_i) * np.sin(dtheta)) \
        / (np.sin(theta_i) * np.sin(0.5 * dtheta))
    return {"k": k, "ka_x": ka_x, "ka_y": ka_y, "sx": sx, "cx": cx,
            "sy": sy, "cy": cy, "p": p, "Il_x": Il_x, "Il_y": Il_y,
            "Iq_x": Iq_x, "Iq_y": Iq_y, "St": St, "hx": hx, "hy": hy,
            "beta": beta, "alpha_polar": alpha_polar}


def _angles(phi_m, dphi):
    c_c = np.sin(dphi) / dphi
    C2 = np.cos(2.0 * phi_m)
    S2 = np.sin(2.0 * phi_m)
    return c_c, c_c * C2, S2


def face_currents_rect(Ax, Bx, Ay, By, q, pr, phi_m, dphi):
    """Outward surface-averaged net currents on [x+, x-, y+, y-] (EXACT)."""
    k, p = pr["k"], pr["p"]
    sx, cx = pr["sx"], pr["cx"]
    sy, cy = pr["sy"], pr["cy"]
    St, hx, hy = pr["St"], pr["hx"], pr["hy"]
    q0, q1x, q1y, q2x, q2y = q
    c_c, cc2, S2 = _angles(phi_m, dphi)
    JXp = -p * (k * Ax * sx + k * Bx * cx + (2 * q1x + 6 * q2x) / (hx * St))
    JXm = -p * (-k * Ax * sx + k * Bx * cx + (2 * q1x - 6 * q2x) / (hx * St))
    JYP = -p * (k * Ay * sy + k * By * cy + (2 * q1y + 6 * q2y) / (hy * St))
    JYM = -p * (-k * Ay * sy + k * By * cy + (2 * q1y - 6 * q2y) / (hy * St))
    JYavg = -p * (2 * By * sy + 2 * q1y / St) / hy
    JXavg = -p * (2 * Bx * sx + 2 * q1x / St) / hx
    return np.array([
        (1 + cc2) * JXp + c_c * S2 * JYavg,
        -(1 + cc2) * JXm - c_c * S2 * JYavg,
        (1 - cc2) * JYP + c_c * S2 * JXavg,
        -(1 - cc2) * JYM - c_c * S2 * JXavg,
    ])


def _j4_matrix_rect(St, mu, hx, hy, phi_m, dphi, pr=None):
    """Exact linear map M : [Ax,Bx,Ay,By] -> J4 (source-free), rectangular."""
    if pr is None:
        pr = node_params_rect(St, mu, hx, hy)
    k, p = pr["k"], pr["p"]
    sx, cx = pr["sx"], pr["cx"]
    sy, cy = pr["sy"], pr["cy"]
    c_c, cc2, S2 = _angles(phi_m, dphi)
    g1x = p * k * sx
    g2x = p * k * cx
    g1y = p * k * sy
    g2y = p * k * cy
    offd_by = 2.0 * c_c * S2 * p * sy / hy
    offd_bx = 2.0 * c_c * S2 * p * sx / hx
    M = np.array([
        [-(1 + cc2) * g1x, -(1 + cc2) * g2x, 0.0, -offd_by],
        [-(1 + cc2) * g1x, +(1 + cc2) * g2x, 0.0, +offd_by],
        [0.0, -offd_bx, -(1 - cc2) * g1y, -(1 - cc2) * g2y],
        [0.0, +offd_bx, -(1 - cc2) * g1y, +(1 - cc2) * g2y],
    ])
    return M, cc2, c_c, S2


def node_coeffs_rect(J4, q, pr, phi_m, dphi):
    """(Ax, Bx, Ay, By, C) from outward J4 and source q, rectangular.

    pr = node_params_rect(...) or node_params_rect_generic(...).
    """
    p, k = pr["p"], pr["k"]
    sx, cx = pr["sx"], pr["cx"]
    sy, cy = pr["sy"], pr["cy"]
    St, hx, hy = pr["St"], pr["hx"], pr["hy"]
    q0, q1x, q1y, q2x, q2y = q
    Jxp, Jxm, Jyp, Jym = J4
    M, cc2, c_c, S2 = _j4_matrix_rect(St, 1.0, hx, hy, phi_m, dphi, pr)
    off = np.array([
        -(1 + cc2) * p * (2 * q1x + 6 * q2x) / (hx * St)
        - c_c * S2 * p * (2 * q1y / St) / hy,
        +(1 + cc2) * p * (2 * q1x - 6 * q2x) / (hx * St)
        + c_c * S2 * p * (2 * q1y / St) / hy,
        -(1 - cc2) * p * (2 * q1y + 6 * q2y) / (hy * St)
        - c_c * S2 * p * (2 * q1x / St) / hx,
        +(1 - cc2) * p * (2 * q1y - 6 * q2y) / (hy * St)
        + c_c * S2 * p * (2 * q1x / St) / hx,
    ])
    Ax, Bx, Ay, By = np.linalg.solve(M, np.asarray(J4) - off)
    phi_bar = q0 / St - ((Jxp + Jxm) / hx + (Jyp + Jym) / hy) / St
    C = phi_bar - q0 / St - 2.0 * sx * Ax / pr["ka_x"] \
        - 2.0 * sy * Ay / pr["ka_y"]
    return Ax, Bx, Ay, By, C


def face_fluxes_rect(Ax, Bx, Ay, By, C, q, pr):
    """Surface-averaged base flux on [x+, x-, y+, y-], before alpha."""
    sx, cx = pr["sx"], pr["cx"]
    sy, cy = pr["sy"], pr["cy"]
    ka_x, ka_y = pr["ka_x"], pr["ka_y"]
    St = pr["St"]
    q0, q1x, q1y, q2x, q2y = q
    tx = 2.0 * sx / ka_x     # x-width average (enters y-face fluxes)
    ty = 2.0 * sy / ka_y     # y-width average (enters x-face fluxes)
    return np.array([
        cx * Ax + sx * Bx + ty * Ay + C + (q0 + q1x + q2x) / St,
        cx * Ax - sx * Bx + ty * Ay + C + (q0 - q1x + q2x) / St,
        cy * Ay + sy * By + tx * Ax + C + (q0 + q1y + q2y) / St,
        cy * Ay - sy * By + tx * Ax + C + (q0 - q1y + q2y) / St,
    ])


def node_state_rect(J4, q, pr, phi_m, dphi, alpha):
    """Full rectangular node state (same dict layout as node.node_state).

    pr  = node_params_rect(...) or node_params_rect_generic(...)
    alpha = geometry-free angular factor (node.node_alphas /
            node.node_alphas_generic), shape (4,).
    """
    Ax, Bx, Ay, By, C = node_coeffs_rect(J4, q, pr, phi_m, dphi)
    Phi = face_fluxes_rect(Ax, Bx, Ay, By, C, q, pr)
    St, hx, hy = pr["St"], pr["hx"], pr["hy"]
    phi_bar = q[0] / St - ((J4[0] + J4[1]) / hx
                           + (J4[2] + J4[3]) / hy) / St
    mom = np.array([
        phi_bar,
        3 * pr["Il_x"] * Bx + q[1] / St,
        3 * pr["Il_y"] * By + q[2] / St,
        5 * pr["Iq_x"] * Ax + q[3] / St,
        5 * pr["Iq_y"] * Ay + q[4] / St,
    ])
    return {"Ax": Ax, "Bx": Bx, "Ay": Ay, "By": By, "C": C,
            "Phi": Phi, "Phibar": alpha * Phi, "phi_bar": phi_bar,
            "mom": mom, "alpha": alpha, "J4check": face_currents_rect(
                Ax, Bx, Ay, By, q, pr, phi_m, dphi)}


def node_response_matrix_rect(pr, phi_m, dphi, alpha, q=None):
    """Linear node response Phibar_f/alpha_f = R.J4 + s, rectangular.

    Probed from the closed-form state (same strategy as the square node
    response matrix) — robust, no re-derived algebra.
    """
    if q is None:
        q = np.zeros(5)
    R = np.empty((4, 4))
    base = node_state_rect(np.zeros(4), q, pr, phi_m, dphi, alpha)
    s = base["Phibar"] / alpha
    for j in range(4):
        J4 = np.zeros(4)
        J4[j] = 1.0
        st = node_state_rect(J4, q, pr, phi_m, dphi, alpha)
        R[:, j] = (st["Phibar"] - base["Phibar"]) / alpha
    return R, s
