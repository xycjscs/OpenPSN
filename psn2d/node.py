"""
PSN node model — restricted 2D (TY polar quadrature), clean closed form.

Chao et al., ANE 240 (2027) 112707.  All formulas derived from the
base-function definition (eqs 2.11a/3.3d, 3.6, 3.7, 2.15, A.3-A.10, B.2)
and cross-verified against brute-force numerical quadrature.

Node: square, side a (extent -a/2..a/2), centered at origin.
Base function (3.3d / 2.11a):
    psi = 1/(4pi) [ phi - (1/St) mu (Omega_phi.grad) phi ],  mu = sin(theta_i)
    phi = Ax cosh(kx) + Bx sinh(kx) + Ay cosh(ky) + By sinh(ky) + C
          + (q0 + q1x P1(2x/a) + q1y P1(2y/a) + q2x P2(2x/a) + q2y P2(2y/a))/St
    k^2 = 2 St^2 / mu^2                     (3.9)
Current vector (3.6c):  Jvec = -(mu^2/(2 St)) grad phi

Face quantities (surface-averaged), face normal n_hat at angle phi_n,
u_mn = (cos(2 phi_m - phi_n), sin(2 phi_m - phi_n)),  c_c = sin(dphi)/dphi:
    J_f  (outward net current)  = (1/a) ∫_f [n_hat + c_c u_mn] . Jvec dS    (3.6b/3.7b)
    Phi_f (surface flux)        = alpha_f * (1/a) ∫_f phi dS,
    alpha_f = mu sin(dphi/2)/dphi * cos(phi_m - phi_n)                        (3.7a/3.7c)

KEY IDENTITY (A.3c/d), used for the tangential line integrals on the faces:
    (1/a) ∫_{-a/2}^{a/2} dphi/dy dy = 2 By s/a + 2 q1y/(a St)   (= G_y)
    (1/a) ∫_{-a/2}^{a/2} dphi/dx dx = 2 Bx s/a + 2 q1x/(a St)   (= G_x)
  (the cosh(kx) integrals give 2s/k*a*(1/a) -> 2B s/a, independent of k;
   equivalently G = (s/(ka c)) phi'_ys + (2/(St a))(1-2s/(ka c)) q1y, where
   phi'_ys = 2 k By c + 4 q1y/(St a) is the sum of y-face derivatives.)

Face current closed form (EXACT, verified vs quadrature):
    p = mu^2/(2 St);  C2 = cos(2 phi_m); S2 = sin(2 phi_m)
    JXp = -p [k Ax s + k Bx c + (2 q1x + 6 q2x)/(a St)]
    JXm = -p [-k Ax s + k Bx c + (2 q1x - 6 q2x)/(a St)]
    JYP = -p [k Ay s + k By c + (2 q1y + 6 q2y)/(a St)]
    JYM = -p [-k Ay s + k By c + (2 q1y - 6 q2y)/(a St)]
    JYavg = -p (2 By s + 2 q1y/St) / a     (G_y term, x-faces)
    JXavg = -p (2 Bx s + 2 q1x/St) / a     (G_x term, y-faces)
    Jxp = (1 + c_c C2) JXp + c_c S2 JYavg          (phi_n = 0)
    Jxm = -(1 + c_c C2) JXm - c_c S2 JYavg         (phi_n = pi)
    Jyp = (1 - c_c C2) JYP + c_c S2 JXavg          (phi_n = pi/2)
    Jym = -(1 - c_c C2) JYM - c_c S2 JXavg         (phi_n = 3pi/2)

Solving for (Ax,Bx,Ay,By) from J4 = [Jxp,Jxm,Jyp,Jym] (exact, sequential):
    Bx = -(Jxp+Jxm + 4(1+cC)p q1x/(St a)) / (2 (1+cC) p k c)
    By = -(Jyp+Jym + 4(1-cC)p q1y/(St a)) / (2 (1-cC) p k c)
    Ax = (Jxm-Jxp - 12(1+cC)p q2x/(St a) - 2 cC S2 p (2 By s + 2 q1y/St)/a)
         / (2 (1+cC) p k s)
    Ay = (Jym-Jyp - 12(1-cC)p q2y/(St a) - 2 cC S2 p (2 Bx s + 2 q1x/St)/a)
         / (2 (1-cC) p k s)
    cC = c_c C2 (cos), S2 = sin(2 phi_m)

Balance (2.9a, all-outward currents):
    phi_bar = q0/St - (Jxp+Jxm+Jyp+Jym)/(St a)          (2.15)
    phi_bar = C + q0/St + (2s/ka)(Ax+Ay)                 (A.8c)
    -> C = phi_bar - q0/St - (2s/ka)(Ax+Ay)              (A.9)

Face flux closed form (A.8a,b):
    Phi_x+ = c Ax + s Bx + (2s/ka) Ay + C + (q0 + q1x + q2x)/St
    Phi_x- = c Ax - s Bx + (2s/ka) Ay + C + (q0 - q1x + q2x)/St
    Phi_y+ = c Ay + s By + (2s/ka) Ax + C + (q0 + q1y + q2y)/St
    Phi_y- = c Ay - s By + (2s/ka) Ax + C + (q0 - q1y + q2y)/St
    (s = sinh(ka/2), c = cosh(ka/2); surface AVERAGES, before alpha)

T-matrix relation (A.10a), revised 2D (3.11a-c):
    [Phi_x+/alpha_x+ ; Phi_y+/alpha_y+]
      = -T1 [Jxd;Jyd] - T2 [Jxs;Jys] + q0/St [1;1]
        + (1/St)(1 - 6c/(ka s) + 12 k^2 a^2)[q2x;q2y]
        + (1/St)((1-2s/(ka c)) I - T3)[q1x;q1y]
    minus faces: flip sign of the T2 term and of the q1 term.
    Jxd=Jxp-Jxm, Jxs=Jxp+Jxm, Jyd=Jyp-Jym, Jys=Jyp+Jym
    T1 = (c St/(mu^2 k s) - 1/(St a)) diag(1/(1+cC C2), 1/(1-cC C2))
         + (1/(St a)) [[1,1],[1,1]]
    T2 = (s St/(mu^2 k c)) [[1+cC C2, cC S2 2s/(ka c)],[cC S2 2s/(ka c), 1-cC C2]]^-1
    T3 = (1-2s/(ka c)) (2 mu^2/(St a)) cC S2  T2 [[0,1],[1,0]]

Legendre moments of the base function phi (source update, B.1-B.2):
    m0  = phi_bar
    m1x = 3 Il Bx + q1x/St      (Il = (2/ka)(c - 2s/ka))     (B.2a)
    m1y = 3 Il By + q1y/St
    m2x = 5 Iq Ax + q2x/St      (Iq = (2/ka)(s - 3Il))       (B.2b)
    m2y = 5 Iq Ay + q2y/St
"""
import numpy as np

PI = np.pi


def ty_polar_set(npts=3):
    """Tabuchi-Yamamoto 3-point polar set (Yamamoto et al., JNST 44(2):120, 2007).

    Authoritative values as tabulated in OpenMOC (mit-crpg/OpenMOC,
    src/Quadrature.cpp, TYPolarQuad::initialize/precomputeWeights):
      sin(theta_i) = [0.166648, 0.537707, 0.932954]
      W_i          = [0.046233, 0.283619, 0.670148]
    In PSN (restricted 2D), mu = sin(theta) is the in-plane direction factor
    that multiplies the gradient in the transport/diffusion equation (3.3c).
    Weights normalized over the pi/2 region: sum(W) = 1  (paper eq. 3.2b).
    Moment check: sum(W*mu^2) = 0.66659 ~= 2/3  (paper eq. 3.3c diffusion limit).
    """
    mu = np.array([0.166648, 0.537707, 0.932954])  # sin(theta), ascending
    w = np.array([0.046233, 0.283619, 0.670148])
    return {"mu": mu, "w": w}


def P1(u):
    return u


def P2(u):
    return 0.5 * (3.0 * u * u - 1.0)


def node_params(St, mu, a):
    """k, ka, s=sinh(ka/2), c=cosh(ka/2), p, Il, Iq.  (restricted TY model)"""
    k = np.sqrt(2.0) * St / mu
    ka = k * a
    s = np.sinh(0.5 * ka)
    c = np.cosh(0.5 * ka)
    p = mu * mu / (2.0 * St)
    if ka > 1e-3:
        Il = (2.0 / ka) * (c - 2.0 * s / ka)
        Iq = (2.0 / ka) * (s - 3.0 * Il)
    else:
        Il = 1.0 / 3.0 * (1.0 - ka * ka / 40.0)
        Iq = 1.0 / 5.0
    return {"k": k, "ka": ka, "s": s, "c": c, "p": p,
            "Il": Il, "Iq": Iq, "R": 1.0 / mu, "beta": 1.0}


def generic_polar_set(I=30):
    """Generic PSN uniform polar partition (2.10d-f): I even, theta over 0..pi.

    In the 2D z-invariant model, segments i and I+1-i (theta and pi-theta)
    have identical physics (same sin theta).  We keep the I/2 distinct
    segments in 0..pi/2 and double their (2.9c) weight:
        w_i = 2 sin(theta_i) sin(dtheta_i/2),   theta_i = (i-1/2) dtheta,
        dtheta = pi/I,   sum_i w_i = 1  (normalizes the scalar flux).
    """
    I = int(I)
    if I % 2 != 0:
        raise ValueError("I must be even (0..pi partition)")
    dtheta = np.pi / I
    th = (np.arange(1, I // 2 + 1) - 0.5) * dtheta
    w = 2.0 * np.sin(th) * np.sin(0.5 * dtheta)
    w = w / w.sum()
    return {"theta": th, "w": w, "dtheta": dtheta}


def node_params_generic(St, a, theta_i, dtheta_i):
    """Generic PSN (paper Sec 2) node parameters.

    Node function solves the STANDARD diffusion equation (2.3c):
        -(1/3 St) lap(phi) + St phi = Q      ->  k^2 = 3 St^2  (angle-free)
    Face current (2.8b):  J = beta_i (n + cc*U) . (-(1/3 St) <grad phi>_face)
        p = beta_i/(3 St),   beta_i = 1 - 0.5 cos(2th)[cos(dth)+cos^2(dth/2)] (2.8d)
    Face flux (2.8a):  Phi = alpha * <phi>_face,
        alpha = 0.25 * (dth - cos(2th) sin(dth)) / (sin(th) sin(dth/2))
                * (sin(dphi/2)/dphi) * cos(phi_m - phi_n)      (2.8c)
    """
    # beta_i = exact angular integral of (2.6)/(2.7c) pair net current,
    # verified to 2e-7 vs direct (theta,phi) quadrature (verify_beta_decisive.py):
    #   beta = 1 - 1/4 cos(dth) - 1/4 cos(2th) - 1/2 cos(2th) cos(dth)
    # (fine-mesh limit = (3/2) sin^2(theta_i), the mono-directional P1 closure.)
    # NOTE: the naive text-extraction 1-0.5cos2th(cos dth+cos^2(dth/2)) is WRONG
    # (up to +33% at high theta -> 600-900 pcm keff offset in checkerboard).
    beta = 1.0 - 0.25 * np.cos(dtheta_i) - 0.25 * np.cos(2.0 * theta_i) \
        - 0.5 * np.cos(2.0 * theta_i) * np.cos(dtheta_i)
    p = beta / (3.0 * St)
    k = np.sqrt(3.0) * St
    ka = k * a
    s = np.sinh(0.5 * ka)
    c = np.cosh(0.5 * ka)
    if ka > 1e-3:
        Il = (2.0 / ka) * (c - 2.0 * s / ka)
        Iq = (2.0 / ka) * (s - 3.0 * Il)
    else:
        Il = 1.0 / 3.0 * (1.0 - ka * ka / 40.0)
        Iq = 1.0 / 5.0
    # (2.8c) polar factor (the azimuthal part is applied per face)
    alpha_polar = 0.25 * (dtheta_i - np.cos(2.0 * theta_i) * np.sin(dtheta_i)) \
        / (np.sin(theta_i) * np.sin(0.5 * dtheta_i))
    return {"k": k, "ka": ka, "s": s, "c": c, "p": p,
            "Il": Il, "Iq": Iq, "beta": beta, "alpha_polar": alpha_polar}


def node_alphas_generic(theta_i, dtheta_i, a, phi_m, dphi):
    """alpha for faces [x+, x-, y+, y-] from (2.8c)."""
    ap = node_params_generic(1.0, 1.0, theta_i, dtheta_i)["alpha_polar"]
    f = ap * np.sin(0.5 * dphi) / dphi
    return np.array([f * np.cos(phi_m),
                     f * np.cos(phi_m - PI),
                     f * np.cos(phi_m - 0.5 * PI),
                     f * np.cos(phi_m - 1.5 * PI)])


def node_alphas(mu, a, phi_m, dphi):
    """alpha for faces [x+, x-, y+, y-]  (3.7c)."""
    f = mu * np.sin(0.5 * dphi) / dphi
    return np.array([f * np.cos(phi_m),
                     f * np.cos(phi_m - PI),
                     f * np.cos(phi_m - 0.5 * PI),
                     f * np.cos(phi_m - 1.5 * PI)])


def face_currents(Ax, Bx, Ay, By, q, pr, phi_m, dphi):
    """Outward surface-averaged net currents on [x+, x-, y+, y-] (EXACT)."""
    s, c, p = pr["s"], pr["c"], pr["p"]
    St, a, k = pr["St"], pr["a"], pr["k"]
    q0, q1x, q1y, q2x, q2y = q
    c_c = np.sin(dphi) / dphi
    C2, S2 = np.cos(2 * phi_m), np.sin(2 * phi_m)
    JXp = -p * (k * Ax * s + k * Bx * c + (2 * q1x + 6 * q2x) / (a * St))
    JXm = -p * (-k * Ax * s + k * Bx * c + (2 * q1x - 6 * q2x) / (a * St))
    JYP = -p * (k * Ay * s + k * By * c + (2 * q1y + 6 * q2y) / (a * St))
    JYM = -p * (-k * Ay * s + k * By * c + (2 * q1y - 6 * q2y) / (a * St))
    JYavg = -p * (2 * By * s + 2 * q1y / St) / a
    JXavg = -p * (2 * Bx * s + 2 * q1x / St) / a
    return np.array([
        (1 + c_c * C2) * JXp + c_c * S2 * JYavg,
        -(1 + c_c * C2) * JXm - c_c * S2 * JYavg,
        (1 - c_c * C2) * JYP + c_c * S2 * JXavg,
        -(1 - c_c * C2) * JYM - c_c * S2 * JXavg,
    ])


def _j4_matrix(St, mu, a, phi_m, dphi, pr=None):
    """Exact closed-form linear map  M : [Ax,Bx,Ay,By] -> J4 (source-free).

    M = [[-(1+cc2)g1, -(1+cc2)g2,        0, -2 cc S2 ps/a],
         [-(1+cc2)g1, +(1+cc2)g2,        0, +2 cc S2 ps/a],
         [         0, -2 cc S2 ps/a, -(1-cc2)g1, -(1-cc2)g2],
         [         0, +2 cc S2 ps/a, -(1-cc2)g1, +(1-cc2)g2]]
    and the source offset off(q) with g1=pps, g2=ppc, cc=c_c*C2, S2=sin2phi_m,
    ps/a = p*s/a."""
    if pr is None:
        pr = node_params(St, mu, a)
    k, s, c = pr["k"], pr["s"], pr["c"]
    p = pr["p"]
    c_c = np.sin(dphi) / dphi
    C2, S2 = np.cos(2 * phi_m), np.sin(2 * phi_m)
    cc2 = c_c * C2
    g1 = p * k * s
    g2 = p * k * c
    offd = 2.0 * c_c * S2 * p * s / pr["a"]
    M = np.array([
        [-(1 + cc2) * g1, -(1 + cc2) * g2, 0.0, -offd],
        [-(1 + cc2) * g1, +(1 + cc2) * g2, 0.0, +offd],
        [0.0, -offd, -(1 - cc2) * g1, -(1 - cc2) * g2],
        [0.0, +offd, -(1 - cc2) * g1, +(1 - cc2) * g2],
    ])
    return M, cc2, c_c, S2


def node_coeffs(J4, q, St, mu, a, phi_m, dphi, pr=None):
    """Given outward face currents J4=[Jxp,Jxm,Jyp,Jym] and source q,
    return (Ax, Bx, Ay, By, C).  Exact closed form (linear solve of the
    source-free map plus a closed-form source offset).

    J4 must be in the SAME current convention as face_currents/pr:
      restricted: J4 = physical net current (p = mu^2/2St)
      generic:    J4 = physical paper current  J = beta(n+cc U).(-(1/3St)<grad phi>)
                  (the beta_i factor is already inside p of node_params_generic,
                   so the balance (2.15) uses J4 directly, no extra factor)
    """
    if pr is None:
        pr = node_params(St, mu, a)
        pr = {**pr, "St": St, "a": a}
    p, s, c = pr["p"], pr["s"], pr["c"]
    q0, q1x, q1y, q2x, q2y = q
    Jxp, Jxm, Jyp, Jym = J4
    M, cc2, c_c, S2 = _j4_matrix(St, mu, a, phi_m, dphi, pr)
    inv_aSt = 1.0 / (pr["a"] * St)
    off = np.array([
        -(1 + cc2) * p * (2 * q1x + 6 * q2x) * inv_aSt - c_c * S2 * p * (2 * q1y / St) / pr["a"],
        +(1 + cc2) * p * (2 * q1x - 6 * q2x) * inv_aSt + c_c * S2 * p * (2 * q1y / St) / pr["a"],
        -(1 - cc2) * p * (2 * q1y + 6 * q2y) * inv_aSt - c_c * S2 * p * (2 * q1x / St) / pr["a"],
        +(1 - cc2) * p * (2 * q1y - 6 * q2y) * inv_aSt + c_c * S2 * p * (2 * q1x / St) / pr["a"],
    ])
    sol = np.linalg.solve(M, np.asarray(J4) - off)
    Ax, Bx, Ay, By = sol
    # C from neutron balance (2.15) + volume average (A.8c):
    phi_bar = q[0] / St - (Jxp + Jxm + Jyp + Jym) / (St * a)
    C = phi_bar - q[0] / St - 2.0 * s * (Ax + Ay) / pr["ka"]
    return Ax, Bx, Ay, By, C


def face_fluxes(Ax, Bx, Ay, By, C, q, pr):
    """Surface-averaged base flux on [x+, x-, y+, y-]  (A.8a,b) -- BEFORE alpha."""
    s, c, ka = pr["s"], pr["c"], pr["ka"]
    St = pr["St"]
    q0, q1x, q1y, q2x, q2y = q
    t = 2.0 * s / ka
    return np.array([
        c * Ax + s * Bx + t * Ay + C + (q0 + q1x + q2x) / St,
        c * Ax - s * Bx + t * Ay + C + (q0 - q1x + q2x) / St,
        c * Ay + s * By + t * Ax + C + (q0 + q1y + q2y) / St,
        c * Ay - s * By + t * Ax + C + (q0 - q1y + q2y) / St,
    ])


def node_state(J4, q, St, mu, a, phi_m, dphi, pr=None, alpha=None):
    """
    Full node state: returns dict with
      Ax,Bx,Ay,By,C, phi_bar,
      Phi[4]  = raw surface flux (A.8, before alpha),
      Phibar[4] = alpha * Phi  (the 3.7a quantity used in interface conditions),
      J4check = face currents recomputed from coefficients (consistency),
      mom[5]  = Legendre moments (m0,m1x,m1y,m2x,m2y) of the base function.
    """
    if pr is None:
        pr = node_params(St, mu, a)
        pr = {**pr, "St": St, "a": a}
    if alpha is None:
        alpha = node_alphas(mu, a, phi_m, dphi)
    Ax, Bx, Ay, By, C = node_coeffs(J4, q, St, mu, a, phi_m, dphi, pr)
    Phi = face_fluxes(Ax, Bx, Ay, By, C, q, pr)
    phi_bar = q[0] / St - (J4[0] + J4[1] + J4[2] + J4[3]) / (St * a)
    Il, Iq = pr["Il"], pr["Iq"]
    mom = np.array([
        phi_bar,
        3 * Il * Bx + q[1] / pr["St"],
        3 * Il * By + q[2] / pr["St"],
        5 * Iq * Ax + q[3] / pr["St"],
        5 * Iq * Ay + q[4] / pr["St"],
    ])
    return {"Ax": Ax, "Bx": Bx, "Ay": Ay, "By": By, "C": C,
            "Phi": Phi, "Phibar": alpha * Phi, "phi_bar": phi_bar,
            "mom": mom, "alpha": alpha, "J4check": face_currents(
                Ax, Bx, Ay, By, q, pr, phi_m, dphi)}


def node_response_matrix(St, mu, a, phi_m, dphi, q=None, pr=None, alpha=None):
    """
    Linear node response:  Phibar_f/alpha_f = R_f . J4 + s_f   (f in 0..3)
    R (4x4), s (4,).  Built by probing the closed-form model (robust).
    """
    if q is None:
        q = np.zeros(5)
    if pr is None:
        pr = node_params(St, mu, a)
        pr = {**pr, "St": St, "a": a}
    if alpha is None:
        alpha = node_alphas(mu, a, phi_m, dphi)
    R = np.empty((4, 4))
    base = node_state(np.zeros(4), q, St, mu, a, phi_m, dphi, pr, alpha)
    s = base["Phibar"] / alpha
    for j in range(4):
        J4 = np.zeros(4)
        J4[j] = 1.0
        st = node_state(J4, q, St, mu, a, phi_m, dphi, pr, alpha)
        R[:, j] = (st["Phibar"] - base["Phibar"]) / alpha
    return R, s


def tmat_matrices(St, mu, a, phi_m, dphi, pr=None):
    """Revised 2D T-matrices (3.11a-c).  Returns (T1, T2, T3)."""
    if pr is None:
        pr = node_params(St, mu, a)
    k, s, c = pr["k"], pr["s"], pr["c"]
    mu2 = mu * mu
    c_c = np.sin(dphi) / dphi
    C2, S2 = np.cos(2 * phi_m), np.sin(2 * phi_m)
    d1 = c * St / (mu2 * k * s) - 1.0 / (St * a)
    T1 = np.array([[d1 / (1 + c_c * C2), 1.0 / (St * a)],
                   [1.0 / (St * a), d1 / (1 - c_c * C2)]])
    A = np.array([[1 + c_c * C2, c_c * S2 * 2 * s / (k * a * c)],
                  [c_c * S2 * 2 * s / (k * a * c), 1 - c_c * C2]])
    T2 = (s * St / (mu2 * k * c)) * np.linalg.inv(A)
    T3 = (1 - 2 * s / (k * a * c)) * (2 * mu2 / (St * a)) * (c_c * S2) \
        * (T2 @ np.array([[0.0, 1.0], [1.0, 0.0]]))
    return T1, T2, T3


def tmat_flux(St, mu, a, phi_m, dphi, q, J4, pr=None):
    """(1/alpha)*Phi on [x+, x-, y+, y-] from the T-matrix (A.10a)."""
    if pr is None:
        pr = node_params(St, mu, a)
    T1, T2, T3 = tmat_matrices(St, mu, a, phi_m, dphi, pr)
    s_, c_, ka, k = pr["s"], pr["c"], pr["ka"], pr["k"]
    q0, q1x, q1y, q2x, q2y = q
    Jxd = J4[0] - J4[1]
    Jxs = J4[0] + J4[1]
    Jyd = J4[2] - J4[3]
    Jys = J4[2] + J4[3]
    b0 = (q0 / St) * np.array([1.0, 1.0])
    # NOTE: paper (A.10a) prints "+12 k^2 a^2" in the q2 coefficient;
    # numerical verification (machine precision) shows the correct term is
    # +12/(ka)^2, i.e. the P2 particular-solution constant.  Use that.
    b2 = (1.0 / St) * (1 - 6 * c_ / (ka * s_) + 12.0 / (ka * ka)) \
        * np.array([q2x, q2y])
    b1 = (1.0 / St) * ((1 - 2 * s_ / (ka * c_)) * np.eye(2) - T3) \
        @ np.array([q1x, q1y])
    Lp = -T1 @ np.array([Jxd, Jyd]) - T2 @ np.array([Jxs, Jys]) \
        + b0 + b2 + b1
    Lm = -T1 @ np.array([Jxd, Jyd]) + T2 @ np.array([Jxs, Jys]) \
        + b0 + b2 - b1
    return np.array([Lp[0], Lm[0], Lp[1], Lm[1]])
