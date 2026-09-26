# -*- coding: utf-8 -*-
"""STRICT phase-space diffusion PSN -- node kernel (production module).

Selected with YAML angular.closure: strict_phase_space (legacy closures
untouched).  The solver calls strict_state_dispatch / strict_response_dispatch
in place of node_rect.node_state_rect / node_response_matrix_rect (rect path)
and node.node_state / node.node_response_matrix (square path).

Derivation and verification chain T1-T6 + the M=2 anchor property:
DERIVATION.md in the strict work area; paper text in paper/ (strict method
chapter).  Module docstring below carries the full method summary.
"""
"""STRICT phase-space diffusion PSN — node kernel v3 (THE method).

DERIVATION (strictened Chao chain; independent, see DERIVATION.md):

Strict cell tensor.  Per phase-space cell (band i, sector m) take the
UNWEIGHTED cell average of the second angular moment,
C_m = <Omega_2d Omega_2d^T>_cell, Omega_2d = mu_i (cos f, sin f):

    <cos^2> = 1/2 + s cos(2 phi_m)/2,  <sin^2> = 1/2 - s cos(2 phi_m)/2
    <cos sin> = s sin(2 phi_m)/2,      s = sin(dphi)/dphi  (sinc)

    D_m = p [[1 + s C2, s S2], [s S2, 1 - s C2]],   p = mu_i^2/(2 St)

(tr I /2 normalization: M=1 -> p I = legacy body exactly.)
IDENTITY: this volume tensor equals the paper's FACE tensor (2.7) with the
same s = sin dphi/dphi - the half-range |cos| weighting and the unweighted
volume weighting coincide for the 2x2 moments over a phi-sector.  Legacy
uses it at the FACES ONLY and solves an ISOTROPIC body; the strict method
solves the body WITH D_m.  That body/face inconsistency is the analytical
source of the checkerboard diagonal-valley cancellation.

DIAGONAL EXACT BASIS (the key lemma).  For a plane wave e^{k w.r}, the
tensor dispersion relation is w^T D_m w = p, i.e.
    1 + s cos(2 theta_w - 2 phi_m) = 1   <=>   theta_w = phi_m +- pi/4.
So for EVERY sector and EVERY anisotropy the two sector-diagonal directions
carry EXACT solutions with the SINGLE decay
    k = sqrt(St/p)   (TY: sqrt(2) St/mu = legacy k; identical for generic)
basis: ch(k u), sh(k u), ch(k v), sh(k v),  u = r . w(phi_m+pi/4),
                                              v = r . w(phi_m-pi/4).
Same status as the legacy additive family (4 exact modes of its body
operator), same conditioning profile (single k), and at M=2 (phi_m = pi/4,
dphi = pi/2) the axes ARE the x/y axes: the node kernel becomes the LEGACY
rect node bit-for-bit (anchor, T4).

EXACT PARTICULAR SOLUTION:
    phi_p = [q0 + q1x P1(x/a) + q1y P1(y/b) + q2x P2(x/a) + q2y P2(y/b)]/St
            + Cp,   Cp = 3 (Dxx q2x/a^2 + Dyy q2y/b^2)/St^2
    L phi_p = q(x,y) pointwise EXACT (legacy: cell-mean only).

FACE CURRENTS: J_f = <-D_m . grad phi>_f (divergence-theorem exact), which
reproduces the legacy mixing (1+-s C2) JX +- s S2 Javg identically when the
basis is the legacy one (M=2), and is the consistent continuation for M>2.

Closure identical in SIZE to legacy: J4 (4 outward face net currents) ->
4 mode coefficients; C from exact volume balance; moments m0..m4 with the
same conventions; alpha (angular chord factor) unchanged, applied outside.
"""
import numpy as np

# ---- <e^{kt}>, <t e^{kt}>, <P2 e^{kt}> on [-1,1], series-safe ------------
def m0(k):
    if abs(k) < 1e-4:
        k2 = k * k
        return 1.0 + k2 * (1.0 / 6.0 + k2 * (1.0 / 120.0 + k2 / 5040.0))
    return np.sinh(k) / k


def m1(k):
    if abs(k) < 1e-4:
        k2 = k * k
        return k * (1.0 / 3.0 + k2 * (1.0 / 30.0 + k2 * (1.0 / 840.0
                                                    + k2 / 45360.0)))
    return np.cosh(k) / k - np.sinh(k) / (k * k)


def m2(k):
    # <P2(t) cosh(k t)> line mean; == 5 Iq(2k)/5 of the production node
    # (identity Iq(2k) = s/k - 3c/k^2 + 3s/k^3, k = ka/2).
    if abs(k) < 1e-3:
        k2 = k * k
        return k2 * (1.0 / 15.0 + k2 * (1.0 / 210.0 + k2 / 45360.0))
    return np.sinh(k) / k - 3 * np.cosh(k) / k ** 2 + 3 * np.sinh(k) / k ** 3


def _eq(al, be, a, b):
    """Means of E = exp(al x + be y) on centered rect [-a,a]x[-b,b]."""
    ep, em = np.exp(al * a), np.exp(-al * a)
    fp, fm = np.exp(be * b), np.exp(-be * b)
    M0x, M0y = m0(al * a), m0(be * b)
    M1x, M1y = m1(al * a), m1(be * b)
    M2x, M2y = m2(al * a), m2(be * b)
    return {
        "pbx_p": ep * M0y, "pbx_m": em * M0y,
        "pby_p": fp * M0x, "pby_m": fm * M0x,
        "V": M0x * M0y, "Vx": M1x * M0y, "Vy": M0x * M1y,
        "Vxx": M2x * M0y, "Vyy": M0x * M2y,
        "gx_xp": al * ep * M0y, "gx_xm": al * em * M0y,
        "gy_xp": be * ep * M0y, "gy_xm": be * em * M0y,
        "gx_yp": al * fp * M0x, "gx_ym": al * fm * M0x,
        "gy_yp": be * fp * M0x, "gy_ym": be * fm * M0x,
    }


def _mode(al, be, a, b, even):
    Qp = _eq(al, be, a, b)
    Qm = _eq(-al, -be, a, b)
    if even:
        return {k: 0.5 * (Qp[k] + Qm[k]) for k in Qp}
    return {k: 0.5 * (Qp[k] - Qm[k]) for k in Qp}


def strict_params(St, p, gamma, phi_m, hx, hy):
    """gamma = sin(dphi)/dphi (sinc), dphi = pi/M: the EXACT sector tensor
    that reproduces the paper face-current mixing (1+-cc C2)JX +- cc S2
    Javg as J = -D_im.grad, verified to 1e-15 (T6 probe, DERIVATION.md).

    ANCHOR PROPERTY (M=2, NOT isotropy): with dphi=pi/2 the sector center
    is phi_m=pi/4 and the diagonal mode directions phi_m+-pi/4 are EXACTLY
    the x and y axes, so the strict basis {ch,sh(k . x/y)} spans the SAME
    function space as the legacy basis at the same k=sqrt(St/p); JR/sJ/Phi/
    mom then match the production node to 1e-15 (T4) and keff to solver
    noise (global anchor 8.9e-11 pcm).  The body tensor stays anisotropic
    (gamma=2/pi) - what matters is that the 4 exact modes of the strict
    body span the legacy trial space, making the node map identical."""
    a, b = 0.5 * hx, 0.5 * hy
    C2, S2 = np.cos(2.0 * phi_m), np.sin(2.0 * phi_m)
    Dxx, Dxy, Dyy = p * (1 + gamma * C2), p * gamma * S2, p * (1 - gamma * C2)
    k = np.sqrt(St / p)                      # == legacy k (TY and generic)
    th1 = phi_m + np.pi / 4.0
    th2 = phi_m - np.pi / 4.0
    mdefs = [(k * np.cos(th1), k * np.sin(th1), True),
             (k * np.cos(th1), k * np.sin(th1), False),
             (k * np.cos(th2), k * np.sin(th2), True),
             (k * np.cos(th2), k * np.sin(th2), False)]
    modes = [_mode(al, be, a, b, ev) for (al, be, ev) in mdefs]
    # ---- J4 matrix: J_f = JR.T @ coef + sJ @ q  (faces x+,x-,y+,y-) ------
    JR = np.zeros((4, 4))                    # [mode, face]
    for j, F in enumerate(modes):
        JR[j, 0] = -(Dxx * F["gx_xp"] + Dxy * F["gy_xp"])
        JR[j, 1] = +((Dxx * F["gx_xm"] + Dxy * F["gy_xm"]))
        JR[j, 2] = -(Dxy * F["gx_yp"] + Dyy * F["gy_yp"])
        JR[j, 3] = +((Dxy * F["gx_ym"] + Dyy * F["gy_ym"]))
    JRt = np.zeros((4, 4))                   # tangent currents (x-faces: y
    for j, F in enumerate(modes):            #  comp; y-faces: x comp)
        JRt[j, 0] = -(Dxy * F["gx_xp"] + Dyy * F["gy_xp"])
        JRt[j, 1] = +((Dxy * F["gx_xm"] + Dyy * F["gy_xm"]))
        JRt[j, 2] = -(Dxx * F["gx_yp"] + Dxy * F["gy_yp"])
        JRt[j, 3] = +((Dxx * F["gx_ym"] + Dxy * F["gy_ym"]))
    VV = np.array([F["V"] for F in modes])
    X1 = np.array([3.0 * F["Vx"] for F in modes])     # <3(x/a) mode>
    Y1 = np.array([3.0 * F["Vy"] for F in modes])
    X2 = np.array([5.0 * F["Vxx"] for F in modes])    # <5 P2(x/a) mode>
    Y2 = np.array([5.0 * F["Vyy"] for F in modes])
    PBx_p = np.array([F["pbx_p"] for F in modes])
    PBx_m = np.array([F["pbx_m"] for F in modes])
    PBy_p = np.array([F["pby_p"] for F in modes])
    PBy_m = np.array([F["pby_m"] for F in modes])
    # ---- exact particular solution ----------------------------------------
    q0c = 1.0 / St
    Cp = np.array([0.0, 0.0, 0.0,
                   3.0 * Dxx / (a * a * St * St),
                   3.0 * Dyy / (b * b * St * St)])
    sPB = np.array([
        [q0c, 0, 0, 1 / St, 0],
        [q0c, 0, 0, 1 / St, 0],
        [q0c, 0, 0, 0, 1 / St],
        [q0c, 0, 0, 0, 1 / St]])
    sPB[0, 1] = +1.0 / St
    sPB[1, 1] = -1.0 / St
    sPB[2, 2] = +1.0 / St
    sPB[3, 2] = -1.0 / St
    sPB += Cp[None, :]                    # Cp enters every face mean
    sJ = np.zeros((4, 5))
    sJ[0] = -(Dxx * np.array([0, 1, 0, 3, 0]) / (a * St)
              + Dxy * np.array([0, 0, 1, 0, 0]) / (b * St))
    sJ[1] = +(Dxx * np.array([0, 1, 0, -3, 0]) / (a * St)
              + Dxy * np.array([0, 0, 1, 0, 0]) / (b * St))
    sJ[2] = -(Dyy * np.array([0, 0, 1, 0, 3]) / (b * St)
              + Dxy * np.array([0, 1, 0, 0, 0]) / (a * St))
    sJ[3] = +(Dyy * np.array([0, 0, 1, 0, -3]) / (b * St)
              + Dxy * np.array([0, 1, 0, 0, 0]) / (a * St))
    return dict(St=St, p=p, gamma=gamma, phi_m=phi_m, hx=hx, hy=hy, k=k,
                Dxx=Dxx, Dxy=Dxy, Dyy=Dyy, th1=th1, th2=th2,
                JR=JR, JRt=JRt, VV=VV, X1=X1, Y1=Y1, X2=X2, Y2=Y2,
                PBx_p=PBx_p, PBx_m=PBx_m, PBy_p=PBy_p, PBy_m=PBy_m,
                Cp=Cp, sPB=sPB, sJ=sJ,
                Jmat=np.linalg.inv(JR.T),
                a=a, b=b)


def node_state_strict(J4, q, N, alpha):
    """Drop-in for node_state_rect (same dict layout)."""
    St, hx, hy = N["St"], N["hx"], N["hy"]
    q = np.asarray(q, float)
    coef = N["Jmat"] @ (np.asarray(J4, float) - N["sJ"] @ q)
    divJ = (J4[0] + J4[1]) / hx + (J4[2] + J4[3]) / hy
    phi_bar = (q[0] - divJ) / St
    # <phi> = VV.coef + C + <phi_p>,  <phi_p> = q0/St + Cp.q   (P1,P2 mean 0)
    C = phi_bar - q[0] / St - N["VV"] @ coef - N["Cp"] @ q
    Phi = np.array([
        N["PBx_p"] @ coef + C + N["sPB"][0] @ q,
        N["PBx_m"] @ coef + C + N["sPB"][1] @ q,
        N["PBy_p"] @ coef + C + N["sPB"][2] @ q,
        N["PBy_m"] @ coef + C + N["sPB"][3] @ q])
    mom = np.array([
        phi_bar,
        N["X1"] @ coef + q[1] / St,
        N["Y1"] @ coef + q[2] / St,
        N["X2"] @ coef + q[3] / St,
        N["Y2"] @ coef + q[4] / St])
    J4check = N["JR"].T @ coef + N["sJ"] @ q
    return {"coef": coef, "C": C, "Phi": Phi, "Phibar": alpha * Phi,
            "phi_bar": phi_bar, "mom": mom, "alpha": alpha,
            "J4check": J4check}


def node_response_matrix_strict(N, alpha, q=None):
    """Probe-based R, s (same strategy as node_response_matrix_rect)."""
    if q is None:
        q = np.zeros(5)
    R = np.empty((4, 4))
    base = node_state_strict(np.zeros(4), q, N, alpha)
    s = base["Phibar"] / alpha
    for j in range(4):
        J4 = np.zeros(4)
        J4[j] = 1.0
        st = node_state_strict(J4, q, N, alpha)
        R[:, j] = (st["Phibar"] - base["Phibar"]) / alpha
    return R, s


def _p_from_pr(pr):
    """Band weight p for the sector tensor from production node params."""
    if "beta" in pr:                     # generic model
        return pr["beta"] / (3.0 * pr["St"])
    return pr["p"]                        # TY: mu^2 / (2 St)


def strict_params_from_pr(pr, phi_m, dphi):
    """strict_params() built from production pr + sector geometry."""
    return strict_params(pr["St"], _p_from_pr(pr), np.sin(dphi) / dphi,
                         phi_m, pr["hx"], pr["hy"])


def strict_state_dispatch(J4, q, pr, phi_m, dphi, alpha):
    """Drop-in for node_rect.node_state_rect."""
    N = strict_params_from_pr(pr, phi_m, dphi)
    return node_state_strict(J4, q, N, alpha)


def strict_response_dispatch(pr, phi_m, dphi, alpha, q=None):
    """Drop-in for node_rect.node_response_matrix_rect."""
    N = strict_params_from_pr(pr, phi_m, dphi)
    return node_response_matrix_strict(N, alpha, q)


def strict_state_square(J4, q, St, mu, a, phi_m, dphi, pr=None, alpha=None):
    """Drop-in for node.node_state (square nodes)."""
    pp = pr["p"] if pr is not None else mu * mu / (2.0 * St)
    N = strict_params(St, pp, np.sin(dphi) / dphi, phi_m, a, a)
    al = np.ones(4) if alpha is None else alpha
    return node_state_strict(J4, q, N, al)


def strict_response_square(St, mu, a, phi_m, dphi, pr=None, alpha=None,
                           q=None):
    """Drop-in for node.node_response_matrix (square nodes)."""
    pp = pr["p"] if pr is not None else mu * mu / (2.0 * St)
    N = strict_params(St, pp, np.sin(dphi) / dphi, phi_m, a, a)
    al = np.ones(4) if alpha is None else alpha
    return node_response_matrix_strict(N, al, q)
