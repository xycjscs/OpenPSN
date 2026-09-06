"""DECISIVE beta_i test: direct (theta,phi) quadrature of the PHYSICAL net
node current (outgoing minus incoming of the plus/minus directions), with
the correct dOmega = sin(th) dth dph, vs the two closed-form candidates.

Physical net through the +x face of the phase-space node (box + antipode):
  leaving = int_{box} (n.O+) psi(O+) dO      (plus directions, n.O+>0 region)
  entering= int_{box} (-n.O-) psi(O-) dO     (minus directions entering)
  net = leaving - entering
with psi(O) = (1/4pi)[phi - (1/St) O.grad phi];  set phi=0, grad phi = gx*xhat.

n.O+ = sin(th)cos(phi-phin),  O+.grad = sin(th)cos(phi)  (gx=1)
n.O- = -sin(th)cos(phi-phin), O-.grad = -sin(th)cos(phi)

Model (2.8b) divided:  J4_x = beta (1 + cc cos(2 phim - phin)) (-1/3St)
raw = F * J4,  F = sin(th_i) sin(dth/2) dphi/pi.

Candidates:
  beta_exact = 3*S3/(4 sin(th_i) sin(dth/2)),  S3 = int sin^3(th) dth
  beta_paper = 1 - 0.5 cos(2 th_i)(cos dth + cos^2(dth/2))
"""
import numpy as np

Nt = Np = 3000
St = 1.0

def phys_net(th_i, dth, phi_m, dphi, phi_n=0.0):
    th = np.linspace(th_i - dth / 2, th_i + dth / 2, Nt)
    ph = np.linspace(phi_m - dphi / 2, phi_m + dphi / 2, Np)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    sT = np.sin(TH)
    c = np.cos(PH - phi_n)
    cosph = np.cos(PH)
    dO = sT                                   # the sin(theta) factor!
    # leaving (plus direction): n.O+ = sT*c ;  psi+ = -(1/4pi St) sT*cosph
    leav = np.trapezoid(np.trapezoid((sT * c) * (-(1.0 / (4 * np.pi * St)) * sT * cosph) * dO, ph, axis=1), th)
    # entering (minus direction): -n.O- = sT*c ;  psi- = +(1/4pi St) sT*cosph
    entr = np.trapezoid(np.trapezoid((sT * c) * (+ (1.0 / (4 * np.pi * St)) * sT * cosph) * dO, ph, axis=1), th)
    return leav - entr

def beta_exact(th_i, dth):
    # S3 = [ -cos + cos^3/3 ] over the box
    t1, t2 = th_i - dth / 2, th_i + dth / 2
    F = lambda t: -np.cos(t) + np.cos(t) ** 3 / 3.0
    S3 = F(t2) - F(t1)
    return 3.0 * S3 / (4.0 * np.sin(th_i) * np.sin(dth / 2.0))

def beta_paper(th_i, dth):
    return 1.0 - 0.5 * np.cos(2 * th_i) * (np.cos(dth) + np.cos(dth / 2.0) ** 2)

print(f"{'th':>6} {'dth/pi':>7} {'phim/pi':>8} | {'net_qd':>10} {'raw_ex':>10} {'raw_pp':>10} | ex/qd pp/qd")
worst_e = worst_p = 0.0
for th_i in np.deg2rad([5, 20, 45, 70, 85]):
    for dth in (np.pi / 30, 2 * np.pi / 30, np.pi / 6):
        for phi_m in (0.0, np.pi / 8, np.pi / 4):
            dphi = np.pi / 8
            net = phys_net(th_i, dth, phi_m, dphi)
            F = np.sin(th_i) * np.sin(dth / 2.0) * dphi / np.pi
            cc = np.sin(dphi) / dphi
            fac = 1.0 + cc * np.cos(2 * phi_m)          # phi_n = 0
            # model: raw = F * beta * (1+cc cos2phi_m) * (-1/3St)  (outward J4)
            raw_ex = -F * beta_exact(th_i, dth) * fac / (3 * St)
            raw_pp = -F * beta_paper(th_i, dth) * fac / (3 * St)
            re, rp = raw_ex / net, raw_pp / net
            worst_e = max(worst_e, abs(re - 1.0))
            worst_p = max(worst_p, abs(rp - 1.0))
            print(f"{np.rad2deg(th_i):6.1f} {dth/np.pi:7.4f} {phi_m/np.pi:8.4f} | "
                  f"{net:+10.4e} {raw_ex:+10.4e} {raw_pp:+10.4e} | {re:7.5f} {rp:7.5f}")
print(f"\nworst deviation from 1:  beta_exact {worst_e:.2e}   beta_paper {worst_p:.2e}")
