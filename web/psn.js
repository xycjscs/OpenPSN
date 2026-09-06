/*
 * psn.js — Phase Space Nodal method (PSN), 2D, 1-group, generic (2.10) model.
 * Faithful JavaScript port of OpenPSN (psn_solver.py / psn_node.py),
 * verified against the Python implementation to machine precision.
 *
 * Method source: Chao, Li & Chen, "Diffusion-based phase space nodal method
 * (PSN): Solving the neutron transport equation with a diffusion code",
 * Annals of Nuclear Energy 240 (2027) 112707, doi:10.1016/j.anucene.2026.112707.
 *
 * Node model (paper Sec 2, generic PSN):
 *   Base function (2.11a/3.3d):  psi = (1/4pi)[phi - (1/3St)(cos theta)(n.g)phi]
 *   phi satisfies the standard diffusion equation, k^2 = 3 St^2  (angle-free),
 *   phi = Ax cosh(kx) + Bx sinh(kx) + Ay cosh(ky) + By sinh(ky) + C
 *         + (q0 + q1x P1(2x/a) + q1y P1(2y/a) + q2x P2(2x/a) + q2y P2(2y/a))/St
 *   Face current  J = beta_i (n + cc U).(-(1/3St) <grad phi>_face)   (2.8b/d)
 *   Face flux     Phi = alpha_f <phi>_face                           (2.8a/c)
 *   beta_i, alpha_f in closed form — no quadrature anywhere.
 *
 * Global system per polar line i (single energy group):
 *   unknowns = face currents u[f, m].
 *   internal face : (1/alpha)Phi continuous across the face, J shared (2.9d)
 *   reflect face  : Phi_m = Phi_m', J_m' = -J_m                      (2.9f/g)
 *   vacuum face   : J = sgn(cos(phi_m - phi_n)) 2 (1/alpha)Phi       (2.9e)
 *   In the half-plane convention (dphi = pi/M) the system is block-diagonal
 *   over azimuth mirror pairs {a, b} = {m, M-1-m}: each block is a dense
 *   system whose coefficients are source-independent -> one-time LU,
 *   outer power-iteration steps only do back-substitution.
 *
 * Eigenvalue: power iteration with k-update and paraboloidal source moments
 * (paper 2.16, B.3).
 */
'use strict';

const PI = Math.PI;

/* ---------------- generic polar set (paper 2.10d-f) ---------------- */
function genericPolarSet(I) {
  I |= 0;
  if (I % 2) throw new Error('I must be even');
  const dtheta = PI / I;
  const theta = [], w0 = [];
  for (let i = 1; i <= I / 2; i++) {
    const th = (i - 0.5) * dtheta;
    theta.push(th);
    w0.push(2 * Math.sin(th) * Math.sin(0.5 * dtheta));
  }
  const sw = w0.reduce((a, b) => a + b, 0);
  return { theta, w: w0.map(x => x / sw), dtheta, n: theta.length };
}

/* ---------------- node parameters, generic mode (2.8) ---------------- */
function nodeParamsGeneric(St, a, th, dth) {
  // beta_i: exact angular integral of the (2.6)/(2.7c) pair net current
  const beta = 1.0 - 0.25 * Math.cos(dth) - 0.25 * Math.cos(2 * th)
             - 0.5 * Math.cos(2 * th) * Math.cos(dth);
  const p = beta / (3 * St);
  const k = Math.sqrt(3) * St, ka = k * a;
  const s = Math.sinh(0.5 * ka), c = Math.cosh(0.5 * ka);
  let Il, Iq;
  if (ka > 1e-3) {
    Il = (2 / ka) * (c - 2 * s / ka);
    Iq = (2 / ka) * (s - 3 * Il);
  } else {
    Il = (1 / 3) * (1 - ka * ka / 40);
    Iq = 1 / 5;
  }
  const alphaPolar = 0.25 * (dth - Math.cos(2 * th) * Math.sin(dth))
                    / (Math.sin(th) * Math.sin(0.5 * dth));
  return { k, ka, s, c, p, Il, Iq, alphaPolar, St, a, beta };
}

function nodeAlphasGeneric(pr, phiM, dphi) {
  const f = pr.alphaPolar * Math.sin(0.5 * dphi) / dphi;
  return [f * Math.cos(phiM), f * Math.cos(phiM - PI),
          f * Math.cos(phiM - 0.5 * PI), f * Math.cos(phiM - 1.5 * PI)];
}

/* node coefficients [Ax,Bx,Ay,By,C] from face currents J4 + source q
 * (exact closed form; faces order [x+, x-, y+, y-]) */
function nodeCoeffs(J4, q, pr, phiM, dphi) {
  const p = pr.p, s = pr.s, c = pr.c, a = pr.a, St = pr.St;
  const cc = Math.sin(dphi) / dphi;
  const C2 = Math.cos(2 * phiM), S2 = Math.sin(2 * phiM);
  const cc2 = cc * C2;
  const g1 = p * pr.k * s, g2 = p * pr.k * c;
  const offd = 2 * cc * S2 * p * s / a;
  const A = [
    [-(1 + cc2) * g1, -(1 + cc2) * g2, 0, -offd],
    [-(1 + cc2) * g1, +(1 + cc2) * g2, 0, +offd],
    [0, -offd, -(1 - cc2) * g1, -(1 - cc2) * g2],
    [0, +offd, -(1 - cc2) * g1, +(1 - cc2) * g2],
  ];
  const invASt = 1 / (a * St);
  const [q0, q1x, q1y, q2x, q2y] = q;
  const b = [
    J4[0] + (1 + cc2) * p * (2 * q1x + 6 * q2x) * invASt + cc * S2 * p * (2 * q1y / St) / a,
    J4[1] - (1 + cc2) * p * (2 * q1x - 6 * q2x) * invASt - cc * S2 * p * (2 * q1y / St) / a,
    J4[2] + (1 - cc2) * p * (2 * q1y + 6 * q2y) * invASt + cc * S2 * p * (2 * q1x / St) / a,
    J4[3] - (1 - cc2) * p * (2 * q1y - 6 * q2y) * invASt - cc * S2 * p * (2 * q1x / St) / a,
  ];
  // solve 4x4 (partial pivoting)
  for (let col = 0; col < 4; col++) {
    let piv = col;
    for (let row = col + 1; row < 4; row++)
      if (Math.abs(A[row][col]) > Math.abs(A[piv][col])) piv = row;
    const ap = A[piv], ac = A[col];
    if (piv !== col) A[col] = ap, A[piv] = ac;
    const bp = b[piv], bc = b[col];
    if (piv !== col) { b[col] = bp; b[piv] = bc; }
    for (let row = col + 1; row < 4; row++) {
      const f = A[row][col] / A[col][col];
      if (f) {
        A[row][col] = f;
        for (let cj = col + 1; cj < 4; cj++) A[row][cj] -= f * A[col][cj];
        b[row] -= f * b[col];
      }
    }
  }
  const x = [0, 0, 0, 0];
  for (let i = 3; i >= 0; i--) {
    let sm = b[i];
    for (let j = i + 1; j < 4; j++) sm -= A[i][j] * x[j];
    x[i] = sm / A[i][i];
  }
  const [Ax, Bx, Ay, By] = x;
  const phibar = q0 / St - (J4[0] + J4[1] + J4[2] + J4[3]) / (St * a);
  const C = phibar - q0 / St - 2 * s * (Ax + Ay) / pr.ka;
  return [Ax, Bx, Ay, By, C];
}

/* surface-averaged base flux [x+, x-, y+, y-] before alpha (A.8a,b) */
function faceFluxes(co, q, pr) {
  const [Ax, Bx, Ay, By, C] = co;
  const s = pr.s, c = pr.c, t = 2 * s / pr.ka, St = pr.St;
  const [q0, q1x, q1y, q2x, q2y] = q;
  return [
    c * Ax + s * Bx + t * Ay + C + (q0 + q1x + q2x) / St,
    c * Ax - s * Bx + t * Ay + C + (q0 - q1x + q2x) / St,
    c * Ay + s * By + t * Ax + C + (q0 + q1y + q2y) / St,
    c * Ay - s * By + t * Ax + C + (q0 - q1y + q2y) / St,
  ];
}

/* node state: J4(4) + q(5) -> {Phi(4), phibar, m1x, m1y, m2x, m2y} */
function nodeState(J4, q, pr, phiM, dphi) {
  const co = nodeCoeffs(J4, q, pr, phiM, dphi);
  const [Ax, Bx, Ay, By] = co;
  return {
    Phi: faceFluxes(co, q, pr),
    phibar: q[0] / pr.St - (J4[0] + J4[1] + J4[2] + J4[3]) / (pr.St * pr.a),
    m1x: 3 * pr.Il * Bx + q[1] / pr.St,
    m1y: 3 * pr.Il * By + q[2] / pr.St,
    m2x: 5 * pr.Iq * Ax + q[3] / pr.St,
    m2y: 5 * pr.Iq * Ay + q[4] / pr.St,
  };
}

/* dense LU (partial pivoting) */
function luFactor(A, n) {
  const a = new Float64Array(A);
  const piv = new Int32Array(n);
  for (let i = 0; i < n; i++) piv[i] = i;
  for (let col = 0; col < n; col++) {
    let p = col;
    for (let r = col + 1; r < n; r++)
      if (Math.abs(a[r * n + col]) > Math.abs(a[p * n + col])) p = r;
    if (Math.abs(a[p * n + col]) < 1e-300) throw new Error('singular system');
    if (p !== col) {
      for (let c2 = 0; c2 < n; c2++) {
        const t = a[col * n + c2]; a[col * n + c2] = a[p * n + c2]; a[p * n + c2] = t;
      }
      const tp = piv[col]; piv[col] = piv[p]; piv[p] = tp;
    }
    for (let r = col + 1; r < n; r++) {
      a[r * n + col] /= a[col * n + col];
      for (let c2 = col + 1; c2 < n; c2++)
        a[r * n + c2] -= a[r * n + col] * a[col * n + c2];
    }
  }
  return { a, piv, n };
}

function luSolve(lu, b) {
  const { a, piv, n } = lu;
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = b[piv[i]];
  for (let i = 0; i < n; i++) {
    let sm = x[i];
    for (let j = 0; j < i; j++) sm -= a[i * n + j] * x[j];
    x[i] = sm;
  }
  for (let i = n - 1; i >= 0; i--) {
    let sm = x[i];
    for (let j = i + 1; j < n; j++) sm -= a[i * n + j] * x[j];
    x[i] = sm / a[i * n + i];
  }
  return x;
}

/* ------------------------------------------------------------------ */
class PSN2D {
  /* matMap: ny x nx array of material indices; h: node side;
   * St, Sa, nuSf: per-material; boundary: [x-, y-, x+, y+] each
   * 'reflect' or 'vacuum'. */
  constructor(matMap, h, St, Sa, nuSf, M, opts = {}) {
    if (M % 2) throw new Error('M must be even');
    this.mat = matMap;
    this.h = h;
    this.nx = matMap[0].length;
    this.ny = matMap.length;
    this.nodes = this.nx * this.ny;
    this.St = St; this.Sa = Sa; this.nuSf = nuSf;
    this.Sg = St.map((s, i) => s - Sa[i]);
    this.M = M;
    this.bnd = opts.boundary || ['reflect', 'reflect', 'reflect', 'reflect'];
    const ps = genericPolarSet(opts.I || 30);
    this.theta = ps.theta; this.W = ps.w; this.dtheta = ps.dtheta;
    this.Ieff = ps.n;
    this.dphi = PI / M;
    this._buildTopology();
    this._prCache = new Map();
    this._alphaCache = new Map();
    this._luCache = new Map();
    this._Rcache = new Map();
    this._RfaceCache = new Map();
    this._Smap = new Map();
  }

  phiM(m) { return (m + 0.5) * this.dphi; }
  matOf(n) { return this.mat[(n / this.nx) | 0][n % this.nx]; }

  _pr(mat, i) {
    const k = mat * 1000 + i;
    let v = this._prCache.get(k);
    if (!v) {
      v = nodeParamsGeneric(this.St[mat], this.h, this.theta[i], this.dtheta);
      this._prCache.set(k, v);
    }
    return v;
  }
  _alpha(i, m) {
    const k = i * 1000 + m;
    let v = this._alphaCache.get(k);
    if (!v) {
      v = nodeAlphasGeneric(this._pr(0, i), this.phiM(m), this.dphi);
      this._alphaCache.set(k, v);
    }
    return v;
  }

  _buildTopology() {
    const { nx, ny, nodes } = this;
    const fxp = new Int32Array(nodes).fill(-1);
    const fxm = new Int32Array(nodes).fill(-1);
    const fyp = new Int32Array(nodes).fill(-1);
    const fym = new Int32Array(nodes).fill(-1);
    const faces = [];
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx - 1; i++) {
        const A = j * nx + i, fid = faces.length;
        faces.push({ t: 'internal', a: A, b: A + 1 });
        fxp[A] = fid; fxm[A + 1] = fid;
      }
    for (let j = 0; j < ny - 1; j++)
      for (let i = 0; i < nx; i++) {
        const A = j * nx + i, fid = faces.length;
        faces.push({ t: 'internal', a: A, b: A + nx });
        fyp[A] = fid; fym[A + nx] = fid;
      }
    for (let j = 0; j < ny; j++) {           // x- side
      const n = j * nx;
      faces.push({ t: this.bnd[0], a: n }); fxm[n] = faces.length - 1;
    }
    for (let i = 0; i < nx; i++) {           // y- side
      const n = i;
      faces.push({ t: this.bnd[1], a: n }); fym[n] = faces.length - 1;
    }
    for (let j = 0; j < ny; j++) {           // x+ side
      const n = j * nx + nx - 1;
      faces.push({ t: this.bnd[2], a: n }); fxp[n] = faces.length - 1;
    }
    for (let i = 0; i < nx; i++) {           // y+ side
      const n = (ny - 1) * nx + i;
      faces.push({ t: this.bnd[3], a: n }); fyp[n] = faces.length - 1;
    }
    this.faces = faces;
    this.F = faces.length;
    this.fxp = fxp; this.fxm = fxm; this.fyp = fyp; this.fym = fym;
    // local face index (0..3 = x+, x-, y+, y-) for each node
    this.localFace = [];
    for (let n = 0; n < nodes; n++) {
      const l = [-1, -1, -1, -1];
      l[0] = fxp[n]; l[1] = fxm[n]; l[2] = fyp[n]; l[3] = fym[n];
      this.localFace.push(l);
    }
    // face -> index within its class
    const intIdx = new Int32Array(this.F).fill(-1);
    const bndIdx = new Int32Array(this.F).fill(-1);
    let ci = 0, cb = 0;
    for (let f = 0; f < this.F; f++) {
      if (faces[f].t === 'internal') intIdx[f] = ci++;
      else bndIdx[f] = cb++;
    }
    this._intIdx = intIdx; this._bndIdx = bndIdx;
    this.nInt = ci; this.nBnd = cb;
    // block column map (built in GLOBAL face order).  Internal AND vacuum
    // faces keep 2 real columns per block [a,b]; reflect faces collapse to
    // 1 column (slave az sign -1 via elimB).  Matches psn_solver.py colidx.
    this._colA = new Int32Array(this.F);
    this._colB = new Int32Array(this.F);
    this._elimB = new Float64Array(this.F);
    {
      let c = 0;
      for (let f = 0; f < this.F; f++) {
        if (faces[f].t === 'reflect') {
          this._colA[f] = c; this._colB[f] = c; this._elimB[f] = -1; c += 1;
        } else {
          this._colA[f] = c; this._colB[f] = c + 1; this._elimB[f] = 1; c += 2;
        }
      }
      this.nPerBlock = c;
    }
    // mirror pairs {a,b} = {m, M-1-m}
    this.pairs = [];
    this.pairsOfAz = new Int32Array(this.M);
    this.azIsA = new Uint8Array(this.M);
    for (let m = 0; m < this.M; m++) {
      const p = (this.M - 1 - m + this.M) % this.M;
      if (p <= m) continue;   // already emitted when processing the smaller index
      const a = m, b = p;
      const pi = this.pairs.length;
      this.pairs.push([a, b]);
      this.pairsOfAz[a] = pi; this.pairsOfAz[b] = pi;
      this.azIsA[a] = 1;
    }
    this._facePhi = faces.map((fc, f) => {
      if (fc.t === 'internal') return 0;
      if (f < ny) return PI;               // x-
      if (f < 2 * ny) return 1.5 * PI;     // y-
      if (f < 2 * ny + nx) return 0;       // x+
      return 0.5 * PI;                     // y+
    });
  }

  _localFaceOf(n, f) {
    const l = this.localFace[n];
    for (let j = 0; j < 4; j++) if (l[j] === f) return j;
    throw new Error('face not on node');
  }

  /* R[lf][j] = d(Phi_raw at face lf)/d J4[j], for (mat, polar i, az). */
  _Rface(mat, i, az) {
    const k = mat * 100000 + i * 1000 + az;
    let R = this._RfaceCache.get(k);
    if (!R) {
      R = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
      const pr = this._pr(mat, i);
      const phiM = this.phiM(az);
      const base = nodeState([0, 0, 0, 0], [0, 0, 0, 0, 0], pr, phiM, this.dphi);
      for (let jj = 0; jj < 4; jj++) {
        const J4 = [0, 0, 0, 0]; J4[jj] = 1;
        const st = nodeState(J4, [0, 0, 0, 0, 0], pr, phiM, this.dphi);
        for (let lf = 0; lf < 4; lf++)
          R[lf][jj] = st.Phi[lf] - base.Phi[lf];
      }
      this._RfaceCache.set(k, R);
    }
    return R;
  }

  /* Build + LU-factorise the block system for (polar line i, mirror pair).
   * Block unknowns, in order:
   *   internal faces: u[f, a], u[f, b]   (2 each)
   *   boundary faces: u[f, master]       (1 each; slave az has sign -1)
   */
  _lu(i, pair) {
    const key = i * 1000 + pair[0];
    const hit = this._luCache.get(key);
    if (hit) return hit;
    const [a, b] = pair;
    const n = this.nPerBlock;
    const colA = this._colA, colB = this._colB, elimB = this._elimB;
    const Aflat = new Float64Array(n * n);
    const bterms = [];   // per row: list of {n, az, lf, sgn, vacFactor}
    let row = 0;
    for (let f = 0; f < this.F; f++) {
      const fc = this.faces[f];
      if (fc.t === 'internal') {
        const A_n = fc.a, B_n = fc.b;
        const lfA = this._localFaceOf(A_n, f), lfB = this._localFaceOf(B_n, f);
        const matA = this.matOf(A_n), matB = this.matOf(B_n);
        for (const az of [a, b]) {
          const rA = this._Rface(matA, i, az), rB = this._Rface(matB, i, az);
          const lfArrA = this.localFace[A_n], lfArrB = this.localFace[B_n];
          const Arow = row * n;
          for (let j = 0; j < 4; j++) {
            const fA = lfArrA[j];
            if (fA >= 0) {
              const sgnOut = [1, -1, 1, -1][j];
              const cIdx = (az === a) ? colA[fA] : colB[fA];
              const elimS = (az === a) ? 1 : elimB[fA];
              Aflat[Arow + cIdx] += rA[lfA][j] * sgnOut * elimS;
            }
            const fB = lfArrB[j];
            if (fB >= 0) {
              const sgnOut = [1, -1, 1, -1][j];
              const cIdx = (az === a) ? colA[fB] : colB[fB];
              const elimS = (az === a) ? 1 : elimB[fB];
              Aflat[Arow + cIdx] -= rB[lfB][j] * sgnOut * elimS;
            }
          }
          bterms.push({ row, terms: [
            { n: A_n, az, lf: lfA, sgn: -1 },
            { n: B_n, az, lf: lfB, sgn: +1 },
          ]});
          row++;
        }
      } else if (fc.t === 'reflect') {
        const n0 = fc.a;
        const lf = this._localFaceOf(n0, f);
        const mat = this.matOf(n0);
        const rA = this._Rface(mat, i, a), rB = this._Rface(mat, i, b);
        const lfArr = this.localFace[n0];
        const Arow = row * n;
        for (let j = 0; j < 4; j++) {
          const fj = lfArr[j];
          if (fj < 0) continue;
          const sgnOut = [1, -1, 1, -1][j];
          Aflat[Arow + colA[fj]] += rA[lf][j] * sgnOut;
          // b-az term maps to colB with the per-face elimination sign
          Aflat[Arow + colB[fj]] -= rB[lf][j] * sgnOut * elimB[fj];
        }
        bterms.push({ row, terms: [
          { n: n0, az: a, lf, sgn: -1 },
          { n: n0, az: b, lf, sgn: +1 },
        ]});
        row++;
      } else { // vacuum (2.9e): J = sgn 2 (1/alpha)Phi  ->  J - 2 sgn alpha Phi_raw = 0
        const n0 = fc.a;
        const lf = this._localFaceOf(n0, f);
        const mat = this.matOf(n0);
        const phiN = this._facePhi[f];
        const sgnOut = [1, -1, 1, -1][lf];
        for (const az of [a, b]) {
          const rV = this._Rface(mat, i, az);
          const alphaL = this._alpha(i, az)[lf];
          const sgn = Math.cos(this.phiM(az) - phiN) >= 0 ? 1 : -1;
          const lfArr = this.localFace[n0];
          const Arow = row * n;
          for (let j = 0; j < 4; j++) {
            const fj = lfArr[j];
            if (fj < 0) continue;
            const sgnOutJ = [1, -1, 1, -1][j];
            const cIdx = (az === a) ? colA[fj] : colB[fj];
            Aflat[Arow + cIdx] += -2 * sgn * alphaL * rV[lf][j] * sgnOutJ * ((az === a) ? 1 : elimB[fj]);
          }
          Aflat[Arow + ((az === a) ? colA[f] : colB[f])] += sgnOut * ((az === a) ? 1 : elimB[f]);
          bterms.push({ row, terms: [{ n: n0, az, lf, sgn: 2 * sgn * alphaL }] });
          row++;
        }
      }
    }
    const lu = luFactor(Aflat, n);
    const sys = { lu, n, bterms };
    this._luCache.set(key, sys);
    return sys;
  }

  /* 9x9 state map [Phi(4), phibar, m1x, m1y, m2x, m2y] = S @ [J4(4), q(5)]
   * for (mat, polar i, az). Probed with nodeState — same trick as the
   * Python vectorised path (107x in the reference implementation). */
  _stateMat(mat, i, az) {
    const k = (mat * 10000 + i) * this.M + az;
    let S = this._Smap.get(k);
    if (!S) {
      S = new Float64Array(81);
      const pr = this._pr(mat, i);
      const phiM = this.phiM(az);
      for (let e = 0; e < 4; e++) {
        const J4 = [0, 0, 0, 0]; J4[e] = 1;
        const st = nodeState(J4, [0, 0, 0, 0, 0], pr, phiM, this.dphi);
        S[e] = st.Phi[0]; S[9 + e] = st.Phi[1]; S[18 + e] = st.Phi[2]; S[27 + e] = st.Phi[3];
        S[36 + e] = st.phibar;
        S[45 + e] = st.m1x; S[54 + e] = st.m1y; S[63 + e] = st.m2x; S[72 + e] = st.m2y;
      }
      for (let e = 0; e < 5; e++) {
        const q = [0, 0, 0, 0, 0]; q[e] = 1;
        const st = nodeState([0, 0, 0, 0], q, pr, phiM, this.dphi);
        const col = 4 + e;
        S[col] = st.Phi[0]; S[9 + col] = st.Phi[1]; S[18 + col] = st.Phi[2]; S[27 + col] = st.Phi[3];
        S[36 + col] = st.phibar;
        S[45 + col] = st.m1x; S[54 + col] = st.m1y; S[63 + col] = st.m2x; S[72 + col] = st.m2y;
      }
      this._Smap.set(k, S);
    }
    return S;
  }

  /* one node's state from J4(4) + q(5) via the 9x9 map */
  _applyState(S, J4, q, out) {
    const X0 = J4[0], X1 = J4[1], X2 = J4[2], X3 = J4[3],
          X4 = q[0], X5 = q[1], X6 = q[2], X7 = q[3], X8 = q[4];
    out[0] = S[0] * X0 + S[1] * X1 + S[2] * X2 + S[3] * X3 + S[4] * X4 + S[5] * X5 + S[6] * X6 + S[7] * X7 + S[8] * X8;
    out[1] = S[36] * X0 + S[37] * X1 + S[38] * X2 + S[39] * X3 + S[40] * X4 + S[41] * X5 + S[42] * X6 + S[43] * X7 + S[44] * X8;
    out[2] = S[45] * X0 + S[46] * X1 + S[47] * X2 + S[48] * X3 + S[49] * X4 + S[50] * X5 + S[51] * X6 + S[52] * X7 + S[53] * X8;
    out[3] = S[54] * X0 + S[55] * X1 + S[56] * X2 + S[57] * X3 + S[58] * X4 + S[59] * X5 + S[60] * X6 + S[61] * X7 + S[62] * X8;
    out[4] = S[63] * X0 + S[64] * X1 + S[65] * X2 + S[66] * X3 + S[67] * X4 + S[68] * X5 + S[69] * X6 + S[70] * X7 + S[71] * X8;
    out[5] = S[72] * X0 + S[73] * X1 + S[74] * X2 + S[75] * X3 + S[76] * X4 + S[77] * X5 + S[78] * X6 + S[79] * X7 + S[80] * X8;
  }

  /* One exact angular sweep for the node source moments qnode[n][5].
   * Returns { phi[n], qim[n][5] }. */
  _sweep(qnode) {
    const N = this.nodes;
    const phi = new Float64Array(N);
    const qim = new Array(N);
    for (let n = 0; n < N; n++) qim[n] = [0, 0, 0, 0, 0];
    const Y = [0, 0, 0, 0, 0, 0];
    const J4a = [0, 0, 0, 0], J4b = [0, 0, 0, 0];
    for (let i = 0; i < this.Ieff; i++) {
      const w = this.W[i] / this.M;
      // precompute raw (J4=0) face fluxes: q @ S(:,4:9) via the state map
      const raw = new Array(N);
      for (let n = 0; n < N; n++) {
        const mat = this.matOf(n);
        const q = qnode[n];
        raw[n] = new Array(this.M);
        for (let az = 0; az < this.M; az++) {
          const S = this._stateMat(mat, i, az);
          const r = new Float64Array(4);
          for (let lf = 0; lf < 4; lf++)
            r[lf] = S[lf * 9 + 4] * q[0] + S[lf * 9 + 5] * q[1] + S[lf * 9 + 6] * q[2]
                  + S[lf * 9 + 7] * q[3] + S[lf * 9 + 8] * q[4];
          raw[n][az] = r;
        }
      }
      // solve all mirror-pair blocks
      const uOfPair = this.pairs.map(pair => {
        const sys = this._lu(i, pair);
        const rhs = new Float64Array(sys.n);
        for (const bt of sys.bterms) {
          let s = 0;
          for (const t of bt.terms) s += t.sgn * raw[t.n][t.az][t.lf];
          rhs[bt.row] = s;
        }
        return luSolve(sys.lu, rhs);
      });
      // accumulate node states via the 9x9 state maps
      for (let n = 0; n < N; n++) {
        const mat = this.matOf(n);
        const q = qnode[n];
        let ph = 0, q0 = 0, q1x = 0, q1y = 0, q2x = 0, q2y = 0;
        const lfArr = this.localFace[n];
        for (let p = 0; p < this.pairs.length; p++) {
          const [a, b] = this.pairs[p];
          const u = uOfPair[p];
          for (let j = 0; j < 4; j++) {
            const f = lfArr[j];
            if (f < 0) continue;
            const sgnOut = [1, -1, 1, -1][j];
            J4a[j] = sgnOut * u[this._colA[f]];
            J4b[j] = sgnOut * this._elimB[f] * u[this._colB[f]];
          }
          this._applyState(this._stateMat(mat, i, a), J4a, q, Y);
          ph += Y[1];
          q1x += Y[2]; q1y += Y[3]; q2x += Y[4]; q2y += Y[5];
          this._applyState(this._stateMat(mat, i, b), J4b, q, Y);
          ph += Y[1];
          q1x += Y[2]; q1y += Y[3]; q2x += Y[4]; q2y += Y[5];
        }
        phi[n] += w * ph;
        qim[n][0] += w * ph;
        qim[n][1] += w * q1x;
        qim[n][2] += w * q1y;
        qim[n][3] += w * q2x;
        qim[n][4] += w * q2y;
      }
    }
    return { phi, qim };
  }

  /* --- stepwise keff driver (for UI progress / live convergence plots) --- */
  startKeff(maxOuter = 4000, tol = 1e-10) {
    const N = this.nodes;
    const st = {
      lam: 1.0, qnode: new Array(N), outer: 0, maxOuter, tol,
      done: false, keff: 1.0, dlam: 1, phi: null, iters: 0, hist: [],
    };
    let F0 = 0;
    for (let n = 0; n < N; n++) F0 += this.nuSf[this.matOf(n)];
    for (let n = 0; n < N; n++) {
      const c = this.Sg[this.matOf(n)] + this.nuSf[this.matOf(n)] / st.lam;
      st.qnode[n] = [c, 0, 0, 0, 0];
    }
    st.F_old = F0;
    return st;
  }

  /* one outer power-iteration step; returns the state (st.done when
   * converged; st.phi holds the final scalar flux). */
  stepKeff(st) {
    if (st.done) return st;
    const N = this.nodes;
    const { phi, qim } = this._sweep(st.qnode);
    let F_new = 0;
    for (let n = 0; n < N; n++) F_new += this.nuSf[this.matOf(n)] * phi[n];
    const lam_new = st.lam * F_new / st.F_old;
    const dlam = Math.abs(lam_new - st.lam) / lam_new;
    st.lam = lam_new;
    for (let n = 0; n < N; n++) {
      const c = this.Sg[this.matOf(n)] + this.nuSf[this.matOf(n)] / st.lam;
      const q = qim[n];
      st.qnode[n] = [c * q[0], c * q[1], c * q[2], c * q[3], c * q[4]];
    }
    st.F_old = F_new;
    st.outer += 1;
    st.iters = st.outer;
    st.keff = st.lam;
    st.dlam = dlam;
    st.hist.push(dlam);
    if (dlam < st.tol || st.outer >= st.maxOuter) {
      st.done = true;
      st.phi = this._sweep(st.qnode).phi;
    }
    return st;
  }

  /* keff power iteration (paper 2.16, single group). */
  keff(maxOuter = 4000, tol = 1e-10, onIter = null) {
    const st = this.startKeff(maxOuter, tol);
    while (!st.done) {
      this.stepKeff(st);
      if (onIter) onIter(st.outer - 1, st.keff, st.dlam);
    }
    return { keff: st.keff, phi: st.phi, iters: st.iters, hist: st.hist };
  }
}

if (typeof module !== 'undefined')
  module.exports = { PSN2D, genericPolarSet, nodeState, nodeParamsGeneric,
                     nodeAlphasGeneric, nodeCoeffs, faceFluxes };
