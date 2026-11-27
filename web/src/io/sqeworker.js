// web/src/io/sqeworker.js
//
// Off-main-thread simulated Inelastic Neutron Scattering (INS) worker:
// computes 2D powder-averaged S(|Q|, E) and 1D phonon DOS.
//
// Physical Model Implemented Here
// ===============================
// When full C-type eigenvectors (`eigRe`, `eigIm`) and site scattering tables
// (`siteBetaOverSqrtM`, `siteSigmaInc`, `siteFrac`, `siteUiso`) are supplied by
// `compute/ins.js`, `computePowderSqE` evaluates the complete powder cross-section
// in physical units of [barn / meV / atom]:
//
//   1. Coherent 1-Phonon Powder S_coh(|Q|, E):
//      At total wave vector Q = q_cart + G_cart (with G = h*b_1 + k*b_2 + l*b_3):
//        F_coh(q + G, nu) = sum_s (beta_coh,s / sqrt(M_s)) * exp(-0.5 |Q|^2 U_iso,s(T))
//                           * (Q . e_{s,nu}^C(q)) * exp(-i 2*pi G . x_s)
//      Averaging over |Q|-shells on the periodic q + G lattice captures acoustic
//      Bragg cones, longitudinal/transverse (Q . e) polarization projections,
//      inter-sublattice structure-factor selection rules, and Debye-Waller damping.
//
//   2. Nuclear Incoherent 1-Phonon S_inc(|Q|, E):
//      Weighted by sigma_inc,s = max(0, sigma_tot,s - sigma_coh,s), recoil Q^2 / M_s,
//      and Debye-Waller factor exp(-|Q|^2 U_iso,s(T)).
//
//   3. Sjolander Multiphonon (n = 2..4) Background S_multi(|Q|, E):
//      Constructed without free parameters via two-sided self-convolution of the
//      unit-normalized single-phonon thermal spectrum f_1(E) weighted by
//      exp(-2W_s) * (2W_s)^n / n! where 2W_s(Q) = |Q|^2 U_iso,s(T).
//
//   4. Direct-Geometry Chopper Resolution ΔE(E; E_i):
//      Supports SEQUOIA, CNCS, ARCS, LET, or constant Gaussian σ via source-centered
//      ("push") energy broadening that preserves total spectral weight identically.
//
// Backward Compatibility:
//   If legacy callers (such as `test/ins_debug.mjs`) pass only `{ nq, nModes, freqs,
//   qpos, F2, recip }` without `eigRe`/`eigIm`, `computePowderSqE` transparently falls
//   back to the scalar `F2` path.

const KB = 0.08617333262;       // meV / K
const HBAR2_2MN = 2.0721248;    // meV * Å^2
const HBAR2_2MU = 2.0900795;    // meV * Å^2 * amu
const SQRT_8LN2 = Math.sqrt(8 * Math.LN2);

const CHOPPER_INSTRUMENTS = {
  SEQUOIA: { elasticFrac: 0.018, c0: 0.18, c1: 0.34, c2: 0.48 },
  CNCS:    { elasticFrac: 0.022, c0: 0.16, c1: 0.32, c2: 0.52 },
  ARCS:    { elasticFrac: 0.030, c0: 0.20, c1: 0.35, c2: 0.45 },
  LET:     { elasticFrac: 0.015, c0: 0.14, c1: 0.30, c2: 0.56 },
};

const LUT_N = 2048, LUT_RANGE = 5, LUT_INV_STEP = LUT_N / LUT_RANGE;
const GAUSS_LUT = new Float64Array(LUT_N + 2);
for (let i = 0; i <= LUT_N + 1; i++) {
  const x = i / LUT_INV_STEP;
  GAUSS_LUT[i] = Math.exp(-0.5 * x * x);
}

function bose(omega, T) {
  if (T <= 0 || omega <= 0) return 0;
  const x = omega / (KB * T);
  return x > 100 ? 0 : 1.0 / (Math.exp(x) - 1.0);
}

function sigmaProfile(Eaxis, Ei, sigmaConst, instrument) {
  const nE = Eaxis.length;
  const sigmas = new Float64Array(nE);
  const spec = instrument && instrument !== 'constant' ? CHOPPER_INSTRUMENTS[instrument] : null;
  if (!spec || !(Ei > 0)) {
    sigmas.fill(Math.max(0.03, sigmaConst));
    return sigmas;
  }
  const fwhmEl = Math.max(0.05, spec.elasticFrac * Ei);
  for (let i = 0; i < nE; i++) {
    const r = Math.max(0, Math.min(2.5, 1 - Eaxis[i] / Ei));
    const fwhm = fwhmEl * Math.sqrt(spec.c0 + spec.c1 * Math.pow(r, 1.5) + spec.c2 * Math.pow(r, 3.0));
    sigmas[i] = Math.max(0.03, fwhm / SQRT_8LN2);
  }
  return sigmas;
}

/**
 * Source-centered ("push") variable-width Gaussian broadening of a 2D column-major
 * spectrum `rawS[qi * nE + ei]` along the energy axis `ei`.
 */
function broadenColumnsAlongE(rawS, nQbins, nE, dE, sigmas) {
  const out = new Float64Array(nQbins * nE);
  // Precompute normalized kernel per source energy bin ei
  const kernels = new Array(nE);
  const offsets = new Int32Array(nE);

  for (let ei = 0; ei < nE; ei++) {
    const sig = sigmas[ei];
    if (sig < 0.25 * dE) {
      kernels[ei] = null;
      offsets[ei] = ei;
      continue;
    }
    const half = Math.min(nE - 1, Math.max(2, Math.ceil((4.0 * sig) / dE)));
    const jLo = Math.max(0, ei - half);
    const jHi = Math.min(nE - 1, ei + half);
    const k = new Float64Array(jHi - jLo + 1);
    let kSum = 0;
    for (let j = jLo; j <= jHi; j++) {
      const x = (j - ei) * dE;
      const w = Math.exp(-0.5 * (x / sig) ** 2);
      k[j - jLo] = w;
      kSum += w;
    }
    if (kSum > 0) for (let u = 0; u < k.length; u++) k[u] /= kSum;
    kernels[ei] = k;
    offsets[ei] = jLo;
  }

  for (let qi = 0; qi < nQbins; qi++) {
    const base = qi * nE;
    for (let ei = 0; ei < nE; ei++) {
      const v = rawS[base + ei];
      if (v === 0) continue;
      const k = kernels[ei];
      if (!k) {
        out[base + ei] += v;
      } else {
        const jLo = offsets[ei];
        for (let u = 0; u < k.length; u++) {
          out[base + jLo + u] += v * k[u];
        }
      }
    }
  }
  return out;
}

// Legacy fallback path when eigenvectors are absent and only scalar `F2` is provided.
function accumulateModesLegacy(F2arr, freqs, base, nModes, T, sigma, Emin, dE, nE, grid) {
  const cutoff = 4 * sigma, norm = 1 / (sigma * Math.sqrt(2 * Math.PI));
  const cutoff_dE = cutoff / dE, inv_step = LUT_INV_STEP, lut = GAUSS_LUT, inv_sigma = 1 / sigma;

  for (let mi = 0; mi < nModes; mi++) {
    const omega = freqs[base + mi];
    if (Math.abs(omega) < 1e-6) continue;
    const F2 = F2arr[base + mi];
    if (F2 === 0) continue;

    const absOmega = Math.abs(omega);
    const occ = omega > 0 ? bose(absOmega, T) + 1 : bose(absOmega, T);
    const amp = (F2 * occ / absOmega) * norm;
    const iCenter = (omega - Emin) / dE;
    const iLo = Math.max(0, Math.floor(iCenter - cutoff_dE));
    const iHi = Math.min(nE - 1, Math.ceil(iCenter + cutoff_dE));
    for (let Ei = iLo; Ei <= iHi; Ei++) {
      let d = (Emin + Ei * dE - omega) * inv_sigma;
      if (d < 0) d = -d;
      if (d >= LUT_RANGE) continue;
      const fi = d * inv_step, i0 = fi | 0;
      grid[Ei] += amp * (lut[i0] + (fi - i0) * (lut[i0 + 1] - lut[i0]));
    }
  }
}

function computePowderSqELegacy(D, T, sigma, Emin, Emax, nE, nQbins, Ei) {
  const { nq, nModes, freqs, qpos, F2, recip: rl } = D;
  const bv = rl.map(v => [2 * Math.PI * v[0], 2 * Math.PI * v[1], 2 * Math.PI * v[2]]);
  const bmag = bv.map(v => Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]));

  const ki_plot = Ei > 0 ? Math.sqrt(Ei / HBAR2_2MN) : 0;
  const Q_max_plot = Ei > 0 ? Math.max(1.5, 2 * ki_plot * 1.05)
    : Math.max(1.5, Math.sqrt(Math.max(Emax, 1) / HBAR2_2MN) * 1.2);
  const dQ = Q_max_plot / nQbins;

  const b_min = Math.min(...bmag) || 1;
  const sigQ = Math.max(1.5 * dQ, 0.4 * b_min), inv_sigQ = 1 / sigQ;

  const n_max = Math.min(20, Math.ceil(Q_max_plot / b_min) + 1);
  const G_reach = Q_max_plot + Math.max(...bmag);
  const G_buf = [];
  for (let n1 = -n_max; n1 <= n_max; n1++) {
    for (let n2 = -n_max; n2 <= n_max; n2++) {
      for (let n3 = -n_max; n3 <= n_max; n3++) {
        const Gx = n1 * bv[0][0] + n2 * bv[1][0] + n3 * bv[2][0];
        const Gy = n1 * bv[0][1] + n2 * bv[1][1] + n3 * bv[2][1];
        const Gz = n1 * bv[0][2] + n2 * bv[1][2] + n3 * bv[2][2];
        if (Math.sqrt(Gx * Gx + Gy * Gy + Gz * Gz) < G_reach) G_buf.push(Gx, Gy, Gz);
      }
    }
  }
  const nG = G_buf.length / 3;
  const Gflat = Float64Array.from(G_buf);

  const dE = (Emax - Emin) / (nE - 1);
  const Eaxis = new Float64Array(nE);
  for (let i = 0; i < nE; i++) Eaxis[i] = Emin + i * dE;
  const S = new Float64Array(nQbins * nE);
  const norm = new Float64Array(nQbins);
  const tmpS = new Float64Array(nE);
  const Qspan = Math.ceil(3.5 * sigQ / dQ);
  const wq_acc = new Float64Array(nQbins), wn_acc = new Float64Array(nQbins);
  const lut = GAUSS_LUT, inv_step = LUT_INV_STEP;

  for (let qi = 0; qi < nq; qi++) {
    tmpS.fill(0);
    accumulateModesLegacy(F2, freqs, qi * nModes, nModes, T, sigma, Emin, dE, nE, tmpS);
    const qx = 2 * Math.PI * (qpos[qi * 3] * rl[0][0] + qpos[qi * 3 + 1] * rl[1][0] + qpos[qi * 3 + 2] * rl[2][0]);
    const qy = 2 * Math.PI * (qpos[qi * 3] * rl[0][1] + qpos[qi * 3 + 1] * rl[1][1] + qpos[qi * 3 + 2] * rl[2][1]);
    const qz = 2 * Math.PI * (qpos[qi * 3] * rl[0][2] + qpos[qi * 3 + 1] * rl[1][2] + qpos[qi * 3 + 2] * rl[2][2]);

    wq_acc.fill(0); wn_acc.fill(0);
    for (let gi = 0; gi < nG; gi++) {
      const Qvx = qx + Gflat[3 * gi], Qvy = qy + Gflat[3 * gi + 1], Qvz = qz + Gflat[3 * gi + 2];
      const Q = Math.sqrt(Qvx * Qvx + Qvy * Qvy + Qvz * Qvz);
      if (Q <= 0 || Q > Q_max_plot) continue;
      const Q2 = Q * Q, Qi_f = Q / dQ;
      const Qi_lo = Math.max(0, Math.floor(Qi_f - Qspan));
      const Qi_hi = Math.min(nQbins - 1, Math.ceil(Qi_f + Qspan));
      for (let Qi = Qi_lo; Qi <= Qi_hi; Qi++) {
        let d = ((Qi + 0.5) * dQ - Q) * inv_sigQ;
        if (d < 0) d = -d;
        if (d >= LUT_RANGE) continue;
        const fi = d * inv_step, i0 = fi | 0;
        const wG = lut[i0] + (fi - i0) * (lut[i0 + 1] - lut[i0]);
        wq_acc[Qi] += Q2 * wG; wn_acc[Qi] += wG;
      }
    }
    for (let Qi = 0; Qi < nQbins; Qi++) {
      if (wq_acc[Qi] === 0) continue;
      norm[Qi] += wn_acc[Qi];
      const base = Qi * nE, w = wq_acc[Qi];
      for (let e = 0; e < nE; e++) S[base + e] += w * tmpS[e];
    }
  }

  for (let Qi = 0; Qi < nQbins; Qi++) {
    if (norm[Qi] > 0) {
      const base = Qi * nE, invN = 1 / norm[Qi];
      for (let e = 0; e < nE; e++) S[base + e] *= invN;
    }
  }
  let Smax = 0;
  for (let i = 0; i < S.length; i++) if (S[i] > Smax) Smax = S[i];
  return { S, nX: nQbins, nE, Eaxis, Smax, xMin: 0, xMax: Q_max_plot, Ei: Ei > 0 ? Ei : 0 };
}

/**
 * Compute full coherent + incoherent + multiphonon powder S(|Q|, E) with chopper ΔE(E; Ei).
 */
function computePowderSqE(D, T, sigma, Emin, Emax, nE, nQbins, Ei, opts = {}) {
  if (!D.eigRe || !D.eigIm || !D.siteBetaOverSqrtM) {
    return computePowderSqELegacy(D, T, sigma, Emin, Emax, nE, nQbins, Ei);
  }

  const {
    instrument = 'constant',
    scatteringMode = 'total', // 'total' | 'coherent' | 'incoherent_approx'
    includeMultiphonon = false,
    maxMultiphononOrder = 4,
  } = opts;

  const {
    nq, nModes, nSites, freqs, qpos, qWeights,
    eigRe, eigIm, siteFrac, siteMass,
    siteBetaOverSqrtM, siteSigmaInc, siteSigmaTot, siteUiso,
    recip: rl,
  } = D;

  const bv = rl.map(v => [2 * Math.PI * v[0], 2 * Math.PI * v[1], 2 * Math.PI * v[2]]);
  const bmag = bv.map(v => Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]));

  const ki_plot = Ei > 0 ? Math.sqrt(Ei / HBAR2_2MN) : 0;
  const Q_max_plot = Ei > 0 ? Math.max(1.5, 2 * ki_plot * 1.05)
    : Math.max(2.5, Math.sqrt(Math.max(Emax, 5) / HBAR2_2MN) * 2.2);
  const dQ = Q_max_plot / nQbins;

  let wqTotal = 0;
  for (let qi = 0; qi < nq; qi++) wqTotal += qWeights ? qWeights[qi] : 1.0;

  const b_min = Math.min(...bmag) || 1;
  // Scale radial Q-smoothing to mesh spacing (~0.65 * b_min / N_linear) so adjacent
  // Brillouin zones do not bleed into one another and acoustic Bragg cones stay sharp.
  const effLinear = Math.max(6, Math.cbrt(wqTotal));
  const sigQ = Math.max(1.25 * dQ, (0.65 * b_min) / effLinear);
  const inv_sigQ = 1 / sigQ;

  // Enumerate reciprocal lattice vectors G = (n1, n2, n3) within Q_max_plot + 3.5 * sigQ
  const n_max = Math.min(18, Math.ceil((Q_max_plot + 3.5 * sigQ) / b_min) + 1);
  const G_reach = Q_max_plot + Math.max(...bmag) + 3.5 * sigQ;
  const gHkl = [];
  const gCart = [];
  for (let n1 = -n_max; n1 <= n_max; n1++) {
    for (let n2 = -n_max; n2 <= n_max; n2++) {
      for (let n3 = -n_max; n3 <= n_max; n3++) {
        const Gx = n1 * bv[0][0] + n2 * bv[1][0] + n3 * bv[2][0];
        const Gy = n1 * bv[0][1] + n2 * bv[1][1] + n3 * bv[2][1];
        const Gz = n1 * bv[0][2] + n2 * bv[1][2] + n3 * bv[2][2];
        if (Math.sqrt(Gx * Gx + Gy * Gy + Gz * Gz) <= G_reach) {
          gHkl.push(n1, n2, n3);
          gCart.push(Gx, Gy, Gz);
        }
      }
    }
  }
  const nG = gCart.length / 3;

  // Precompute q-independent C-type reciprocal-lattice phase per (G, site):
  //   phaseRe/Im[gi * nSites + s] = (beta_coh,s / sqrt(M_s)) * exp(-i 2*pi G_hkl . x_s)
  const phaseRe = new Float64Array(nG * nSites);
  const phaseIm = new Float64Array(nG * nSites);
  for (let gi = 0; gi < nG; gi++) {
    const h = gHkl[gi * 3], k = gHkl[gi * 3 + 1], l = gHkl[gi * 3 + 2];
    for (let s = 0; s < nSites; s++) {
      const ph = -2 * Math.PI * (h * siteFrac[s * 3] + k * siteFrac[s * 3 + 1] + l * siteFrac[s * 3 + 2]);
      const amp = siteBetaOverSqrtM[s];
      phaseRe[gi * nSites + s] = amp * Math.cos(ph);
      phaseIm[gi * nSites + s] = amp * Math.sin(ph);
    }
  }

  const dE = (Emax - Emin) / (nE - 1);
  const Eaxis = new Float64Array(nE);
  for (let i = 0; i < nE; i++) Eaxis[i] = Emin + i * dE;

  const tEval = Math.max(0.5, T);

  // Raw unbroadened histograms in [barn / meV / atom] (column-major: Qi * nE + ei)
  const rawCoh = new Float64Array(nQbins * nE);
  const rawInc = new Float64Array(nQbins * nE);
  const shellNorm = new Float64Array(nQbins);

  // Per-site unbroadened 1-phonon spectral density on Eaxis (for incoherent & Sjolander multiphonon)
  const siteIncHist = Array.from({ length: nSites }, () => new Float64Array(nE));
  const siteUnitDosPos = Array.from({ length: nSites }, () => new Float64Array(nE));

  const Qspan = Math.ceil(3.5 * sigQ / dQ);
  const lut = GAUSS_LUT, inv_step = LUT_INV_STEP;
  const dwHalfSite = new Float64Array(nSites);
  const modeQ = new Float64Array(nModes * nQbins);
  const eCutLow = Math.max(0.25, 1.2 * dE);

  const computeCoherentInterference = scatteringMode !== 'incoherent_approx';

  for (let qi = 0; qi < nq; qi++) {
    const wq = qWeights ? qWeights[qi] : 1.0;
    const baseMode = qi * nModes;

    // 1. Accumulate per-site |e_{s,nu}|^2 onto energy bins for incoherent & multiphonon terms
    for (let mi = 0; mi < nModes; mi++) {
      const omega = freqs[baseMode + mi];
      const absE = Math.abs(omega);
      if (absE < eCutLow) continue;

      const eb = Math.round((omega - Emin) / dE);
      const ebPos = Math.round((absE - Emin) / dE);
      const eigBase = (baseMode + mi) * (3 * nSites);

      const occ = omega > 0 ? bose(absE, tEval) + 1 : bose(absE, tEval);
      const occOverE = occ / (absE * dE);

      for (let s = 0; s < nSites; s++) {
        const idx = eigBase + s * 3;
        const mag2 = eigRe[idx] * eigRe[idx] + eigIm[idx] * eigIm[idx]
          + eigRe[idx + 1] * eigRe[idx + 1] + eigIm[idx + 1] * eigIm[idx + 1]
          + eigRe[idx + 2] * eigRe[idx + 2] + eigIm[idx + 2] * eigIm[idx + 2];
        if (eb >= 0 && eb < nE) {
          siteIncHist[s][eb] += wq * mag2 * occOverE;
        }
        if (omega > 0 && ebPos >= 0 && ebPos < nE) {
          siteUnitDosPos[s][ebPos] += wq * mag2;
        }
      }
    }

    if (!computeCoherentInterference) continue;

    // 2. Coherent 1-phonon structure factor |F_coh(q + G, m)|^2 across all G in shell
    const qx = qpos[qi * 3] * bv[0][0] + qpos[qi * 3 + 1] * bv[1][0] + qpos[qi * 3 + 2] * bv[2][0];
    const qy = qpos[qi * 3] * bv[0][1] + qpos[qi * 3 + 1] * bv[1][1] + qpos[qi * 3 + 2] * bv[2][1];
    const qz = qpos[qi * 3] * bv[0][2] + qpos[qi * 3 + 1] * bv[1][2] + qpos[qi * 3 + 2] * bv[2][2];

    modeQ.fill(0);

    for (let gi = 0; gi < nG; gi++) {
      const Qx = qx + gCart[gi * 3];
      const Qy = qy + gCart[gi * 3 + 1];
      const Qz = qz + gCart[gi * 3 + 2];
      const Q2 = Qx * Qx + Qy * Qy + Qz * Qz;
      const Q = Math.sqrt(Q2);
      if (Q <= 1e-6 || Q > Q_max_plot + 3 * sigQ) continue;

      const Qi_f = Q / dQ;
      const Qi_lo = Math.max(0, Math.floor(Qi_f - Qspan));
      const Qi_hi = Math.min(nQbins - 1, Math.ceil(Qi_f + Qspan));
      if (Qi_lo > Qi_hi) continue;

      for (let s = 0; s < nSites; s++) {
        dwHalfSite[s] = Math.exp(-0.5 * Q2 * siteUiso[s]);
      }

      // Accumulate shell point-count denominator once per (q, G)
      for (let Qi = Qi_lo; Qi <= Qi_hi; Qi++) {
        let d = ((Qi + 0.5) * dQ - Q) * inv_sigQ;
        if (d < 0) d = -d;
        if (d >= LUT_RANGE) continue;
        const fi = d * inv_step, i0 = fi | 0;
        const wG = lut[i0] + (fi - i0) * (lut[i0 + 1] - lut[i0]);
        shellNorm[Qi] += wq * wG;
      }

      // Evaluate |F_coh(q + G, mi)|^2 for each active mode and deposit into modeQ
      const phBase = gi * nSites;
      for (let mi = 0; mi < nModes; mi++) {
        const omega = freqs[baseMode + mi];
        if (Math.abs(omega) < eCutLow) continue;
        const eb = Math.round((omega - Emin) / dE);
        if (eb < 0 || eb >= nE) continue;

        const eigBase = (baseMode + mi) * (3 * nSites);
        let fRe = 0, fIm = 0;
        for (let s = 0; s < nSites; s++) {
          const dw = dwHalfSite[s];
          const pRe = phaseRe[phBase + s] * dw;
          const pIm = phaseIm[phBase + s] * dw;
          const idx = eigBase + s * 3;
          // Dot product (Q . e_{s,mi}^C) = dRe + i dIm
          const dRe = Qx * eigRe[idx] + Qy * eigRe[idx + 1] + Qz * eigRe[idx + 2];
          const dIm = Qx * eigIm[idx] + Qy * eigIm[idx + 1] + Qz * eigIm[idx + 2];
          fRe += pRe * dRe - pIm * dIm;
          fIm += pRe * dIm + pIm * dRe;
        }
        const f2Val = fRe * fRe + fIm * fIm;
        if (f2Val <= 0) continue;

        const mOff = mi * nQbins;
        for (let Qi = Qi_lo; Qi <= Qi_hi; Qi++) {
          let d = ((Qi + 0.5) * dQ - Q) * inv_sigQ;
          if (d < 0) d = -d;
          if (d >= LUT_RANGE) continue;
          const fi = d * inv_step, i0 = fi | 0;
          const wG = lut[i0] + (fi - i0) * (lut[i0 + 1] - lut[i0]);
          modeQ[mOff + Qi] += wG * f2Val;
        }
      }
    }

    // Deposit this q-point's modeQ profiles into rawCoh[Qi * nE + eb]
    for (let mi = 0; mi < nModes; mi++) {
      const omega = freqs[baseMode + mi];
      const absE = Math.abs(omega);
      if (absE < eCutLow) continue;
      const eb = Math.round((omega - Emin) / dE);
      if (eb < 0 || eb >= nE) continue;

      const occ = omega > 0 ? bose(absE, tEval) + 1 : bose(absE, tEval);
      const pref = (wq * HBAR2_2MU * occ) / (nSites * absE * dE);
      const mOff = mi * nQbins;
      for (let Qi = 0; Qi < nQbins; Qi++) {
        const mq = modeQ[mOff + Qi];
        if (mq !== 0) rawCoh[Qi * nE + eb] += pref * mq;
      }
    }
  }

  // Normalize coherent shell accumulator by shell point count
  if (computeCoherentInterference) {
    for (let Qi = 0; Qi < nQbins; Qi++) {
      if (shellNorm[Qi] > 1e-12) {
        const invShell = 1 / shellNorm[Qi];
        const base = Qi * nE;
        for (let ei = 0; ei < nE; ei++) rawCoh[base + ei] *= invShell;
      }
    }
  }

  // Build incoherent 1-phonon (or full incoherent approximation when requested)
  const normSite1Ph = 1 / Math.max(3 * wqTotal, 1e-12);
  for (let s = 0; s < nSites; s++) {
    const sigWeight = scatteringMode === 'incoherent_approx'
      ? siteSigmaTot[s]
      : (scatteringMode === 'coherent' ? 0 : siteSigmaInc[s]);
    if (sigWeight <= 0 || !(siteMass[s] > 0)) continue;
    const hist = siteIncHist[s];
    const uIso = siteUiso[s];
    const invMass = HBAR2_2MU / (nSites * siteMass[s]);

    for (let Qi = 0; Qi < nQbins; Qi++) {
      const Q = (Qi + 0.5) * dQ;
      const Q2 = Q * Q;
      const qPref = sigWeight * invMass * Q2 * Math.exp(-Q2 * uIso) * normSite1Ph;
      const base = Qi * nE;
      for (let ei = 0; ei < nE; ei++) {
        rawInc[base + ei] += qPref * hist[ei];
      }
    }
  }

  // Optional Sjolander multiphonon (n = 2 .. maxMultiphononOrder)
  const rawMulti = new Float64Array(nQbins * nE);
  if (includeMultiphonon && maxMultiphononOrder >= 2 && Emax > eCutLow) {
    const nHalf = Math.min(400, Math.max(32, Math.ceil((1.15 * Emax) / dE)));
    const mLen = 2 * nHalf + 1;
    const f1 = new Float64Array(mLen);

    for (let s = 0; s < nSites; s++) {
      const sigTot = siteSigmaTot[s];
      const uIso = siteUiso[s];
      const mass = siteMass[s];
      if (!(sigTot > 0) || !(mass > 0)) continue;

      // Normalize site's 1-phonon DOS on Eaxis (> 0)
      const gRaw = siteUnitDosPos[s];
      let gArea = 0;
      for (let ei = 0; ei < nE; ei++) if (Eaxis[ei] > 0) gArea += gRaw[ei] * dE;
      if (!(gArea > 1e-15)) continue;

      f1.fill(0);
      let f1Area = 0;
      const pref = HBAR2_2MU / (mass * uIso);
      for (let k = 0; k < mLen; k++) {
        const es = (k - nHalf) * dE;
        const absE = Math.abs(es);
        if (absE < eCutLow || absE > Emax) continue;
        const idxE = Math.min(nE - 1, Math.max(0, Math.round((absE - Emin) / dE)));
        const gUnit = gRaw[idxE] / gArea;
        const occ = es > 0 ? bose(absE, tEval) + 1 : bose(absE, tEval);
        const val = pref * (gUnit / absE) * occ;
        f1[k] = val;
        f1Area += val * dE;
      }
      if (!(f1Area > 1e-15)) continue;
      for (let k = 0; k < mLen; k++) f1[k] /= f1Area;

      let fn = Float64Array.from(f1);
      let fact = 1;
      for (let order = 2; order <= maxMultiphononOrder; order++) {
        fact *= order;
        const nextFn = new Float64Array(mLen);
        for (let k = 0; k < mLen; k++) {
          let acc = 0;
          const jMin = Math.max(0, k - nHalf);
          const jMax = Math.min(mLen - 1, k + nHalf);
          for (let j = jMin; j <= jMax; j++) acc += fn[j] * f1[k - j + nHalf];
          nextFn[k] = acc * dE;
        }
        fn = nextFn;

        // Project fn onto Eaxis and deposit across Q bins with (Q^2 U_iso)^order / order! * exp(-Q^2 U_iso)
        for (let Qi = 0; Qi < nQbins; Qi++) {
          const Q = (Qi + 0.5) * dQ;
          const twoW = Q * Q * uIso;
          const wQ = (sigTot / nSites) * Math.exp(-twoW) * (Math.pow(twoW, order) / fact);
          const base = Qi * nE;
          for (let ei = 0; ei < nE; ei++) {
            const kSym = Math.round(Eaxis[ei] / dE) + nHalf;
            if (kSym >= 0 && kSym < mLen) {
              rawMulti[base + ei] += wQ * fn[kSym];
            }
          }
        }
      }
    }
  }

  const rawTotal = new Float64Array(nQbins * nE);
  for (let i = 0; i < rawTotal.length; i++) {
    rawTotal[i] = (computeCoherentInterference ? rawCoh[i] : 0) + rawInc[i] + rawMulti[i];
  }

  // Broaden along E with either constant σ or direct-geometry chopper ΔE(E; Ei)
  const sigmas = sigmaProfile(Eaxis, Ei, sigma, instrument);
  const S = broadenColumnsAlongE(rawTotal, nQbins, nE, dE, sigmas);

  let Smax = 0;
  for (let i = 0; i < S.length; i++) if (S[i] > Smax) Smax = S[i];
  return {
    S,
    nX: nQbins,
    nE,
    Eaxis,
    Smax,
    xMin: 0,
    xMax: Q_max_plot,
    Ei: Ei > 0 ? Ei : 0,
    instrument,
    scatteringMode,
    includeMultiphonon,
  };
}

function computePhononDOS(D, sigma, Emin, Emax, nE) {
  const { nq, nModes, freqs, qWeights } = D;
  const dE = (Emax - Emin) / (nE - 1);
  const halfWidth = Math.ceil(4 * sigma / dE), kLen = 2 * halfWidth + 1, pad = halfWidth;
  const histLen = nE + 2 * pad, hist = new Float64Array(histLen);
  let weightSum = 0;
  for (let qi = 0; qi < nq; qi++) {
    const wq = qWeights ? qWeights[qi] : 1.0;
    for (let m = 0; m < nModes; m++) {
      const omega = freqs[qi * nModes + m];
      if (Math.abs(omega) < 1e-6) continue;
      weightSum += wq;
      const fpos = (omega - Emin) / dE + pad, i0 = Math.floor(fpos), fr = fpos - i0;
      if (i0 >= 0 && i0 < histLen) hist[i0] += wq * (1 - fr);
      if (i0 + 1 >= 0 && i0 + 1 < histLen) hist[i0 + 1] += wq * fr;
    }
  }
  const kernel = new Float64Array(kLen), knorm = 1 / (sigma * Math.sqrt(2 * Math.PI)), inv2s2 = 0.5 / (sigma * sigma);
  for (let i = 0; i < kLen; i++) { const x = (i - halfWidth) * dE; kernel[i] = knorm * Math.exp(-x * x * inv2s2); }
  const dos = new Float64Array(nE);
  for (let i = 0; i < histLen; i++) {
    const hi = hist[i]; if (hi === 0) continue;
    const jBase = i - pad - halfWidth, kLo = Math.max(0, -jBase), kHi = Math.min(kLen - 1, nE - 1 - jBase);
    for (let k = kLo; k <= kHi; k++) dos[jBase + k] += hi * kernel[k];
  }
  if (weightSum > 0) { const inv = 1 / weightSum; for (let i = 0; i < nE; i++) dos[i] *= inv; }
  let dosMax = 0;
  for (let i = 0; i < nE; i++) if (dos[i] > dosMax) dosMax = dos[i];
  return { dos, nE, Emin, Emax, dosMax };
}

// Exported for testing/debugging (test/ins_debug.mjs, test/ins_physics_test.mjs).
export { computePowderSqE, computePhononDOS };

if (typeof self !== 'undefined' && typeof self.postMessage === 'function') {
  self.onmessage = (ev) => {
    const { data, params } = ev.data;
    try {
      const t0 = performance.now();
      const powResult = computePowderSqE(
        data,
        params.T,
        params.sigma,
        params.Emin,
        params.Emax,
        params.nE,
        params.nQbins,
        params.Ei,
        {
          instrument: params.instrument || 'constant',
          scatteringMode: params.scatteringMode || 'total',
          includeMultiphonon: Boolean(params.includeMultiphonon),
          maxMultiphononOrder: params.maxMultiphononOrder || 4,
        }
      );
      const t1 = performance.now();
      const dosResult = computePhononDOS(data, params.sigma, params.Emin, params.Emax, params.nE);
      const t2 = performance.now();
      const transfer = [];
      if (powResult) transfer.push(powResult.S.buffer, powResult.Eaxis.buffer);
      if (dosResult) transfer.push(dosResult.dos.buffer);
      self.postMessage({ success: true, powResult, dosResult, timings: { pow: t1 - t0, dos: t2 - t0 } }, transfer);
    } catch (err) {
      self.postMessage({ success: false, error: (err && err.message) || String(err) });
    }
  };
}
