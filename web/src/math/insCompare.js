// web/src/math/insCompare.js
//
// Quantitative 1D GDOS / S(E) and 2D S(|Q|, E) comparison & spectrometer physics:
//
// 1. Direct-Geometry Chopper Energy Resolution ΔE(E; E_i)
//    On direct-geometry time-of-flight (TOF) chopper spectrometers (SEQUOIA, CNCS,
//    ARCS at SNS; LET at ISIS), scattered neutrons fly from the sample to the detector
//    with final energy E_f = E_i - E. Because the TOF Jacobian scales as
//    |∂E_f / ∂t_2| ∝ E_f^(3/2) = E_i^(3/2) (1 - E / E_i)^(3/2), energy resolution
//    sharpens monotonically with increasing energy transfer E:
//
//      FWHM(E; E_i) = FWHM_el(E_i) * sqrt( c0 + c1*(1 - E/E_i)^(3/2) + c2*(1 - E/E_i)^3 )
//
//    where c0 + c1 + c2 = 1 and FWHM_el(E_i) = elasticFrac * E_i.
//    When broadening a spectrum that already has an intrinsic histogram smoothing
//    width W_base, only the quadrature difference sqrt(max(0, FWHM(E)^2 - W_base^2))
//    is applied so peaks match the exact spectrometer width.
//
// 2. Parameter-Free Sjolander Multiphonon (n >= 2) Expansion
//    From a unit-normalized 1-phonon GDOS g(E), we construct the two-sided thermal
//    single-phonon spectrum on [-E_max, +E_max]:
//      f_1(+E) ∝ (g(E) / E) * (n(E, T) + 1)   [phonon creation, E > 0]
//      f_1(-E) ∝ (g(E) / E) * n(E, T)          [phonon annihilation, E > 0]
//    normalized to ∫_{-∞}^{+∞} f_1(E) dE = 1 (Placzek 0th-moment sum rule).
//    On a Q-integrated 1D S(E)/GDOS curve at characteristic momentum transfer <Q>,
//    the n-phonon overtone/combination background relative to the n=1 spectrum is:
//      g_multi(E) / g_1(E)  with  2W = <Q>^2 * U_iso(T),
//      where the n-th order carries exact Poisson weight (2W)^n / n! relative to 2W * f_1(E).

import { KB_MEV_PER_K, HBAR2_OVER_2MU } from '../constants.js';

const SQRT_8LN2 = Math.sqrt(8 * Math.LN2);

/**
 * Published direct-geometry TOF chopper resolution presets.
 * Parameters match `CHOPPER_INSTRUMENTS` in `mlip/rmcph_mlip/neutron.py`.
 */
export const CHOPPER_PRESETS = {
  constant: {
    key: 'constant',
    label: 'Constant Gaussian FWHM',
    shortLabel: 'Constant',
    elasticFrac: 0,
    c0: 1,
    c1: 0,
    c2: 0,
  },
  SEQUOIA: {
    key: 'SEQUOIA',
    label: 'SEQUOIA (SNS · 1.8% Eᵢ)',
    shortLabel: 'SEQUOIA',
    elasticFrac: 0.018,
    c0: 0.18,
    c1: 0.34,
    c2: 0.48,
  },
  CNCS: {
    key: 'CNCS',
    label: 'CNCS (SNS · 2.2% Eᵢ)',
    shortLabel: 'CNCS',
    elasticFrac: 0.022,
    c0: 0.16,
    c1: 0.32,
    c2: 0.52,
  },
  ARCS: {
    key: 'ARCS',
    label: 'ARCS (SNS · 3.0% Eᵢ)',
    shortLabel: 'ARCS',
    elasticFrac: 0.030,
    c0: 0.20,
    c1: 0.35,
    c2: 0.45,
  },
  LET: {
    key: 'LET',
    label: 'LET (ISIS · 1.5% Eᵢ)',
    shortLabel: 'LET',
    elasticFrac: 0.015,
    c0: 0.14,
    c1: 0.30,
    c2: 0.56,
  },
};

/**
 * Evaluate direct-geometry chopper FWHM(E; Ei) [meV] at a single energy transfer E [meV].
 */
export function chopperResolutionFwhmAtE(E, eiMev, instrument = 'SEQUOIA', fallbackFwhm = 0.8) {
  const preset = CHOPPER_PRESETS[instrument];
  if (!preset || instrument === 'constant' || !(eiMev > 0)) {
    return Math.max(0.05, Number(fallbackFwhm) || 0.8);
  }
  const fwhmElastic = Math.max(0.05, preset.elasticFrac * eiMev);
  const efOverEi = Math.max(0, Math.min(2.5, 1 - E / eiMev));
  const shape = Math.sqrt(
    preset.c0
      + preset.c1 * Math.pow(efOverEi, 1.5)
      + preset.c2 * Math.pow(efOverEi, 3.0)
  );
  return fwhmElastic * shape;
}

/**
 * Evaluate chopper resolution FWHM(E; Ei) [meV] across an entire energy array `E`.
 */
export function chopperResolutionProfile(E, eiMev, instrument = 'SEQUOIA', fallbackFwhm = 0.8) {
  const out = new Float64Array(E.length);
  for (let i = 0; i < E.length; i++) {
    out[i] = chopperResolutionFwhmAtE(E[i], eiMev, instrument, fallbackFwhm);
  }
  return out;
}

/**
 * Convolve a uniformly spaced 1D spectrum y(E) with a constant Gaussian of `extraFwhm` [meV].
 */
export function gaussianBroaden1D(E, y, extraFwhm) {
  if (!extraFwhm || extraFwhm <= 0 || E.length < 3) return y.slice();
  const dE = Math.abs(E[1] - E[0]);
  if (!(dE > 0)) return y.slice();
  const sigma = extraFwhm / SQRT_8LN2;
  const half = Math.max(2, Math.min(200, Math.ceil((3.5 * sigma) / dE)));
  const kernel = new Float64Array(2 * half + 1);
  let kSum = 0;
  for (let k = -half; k <= half; k++) {
    const x = k * dE;
    const w = Math.exp(-0.5 * (x / sigma) ** 2);
    kernel[k + half] = w;
    kSum += w;
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= kSum;

  const n = y.length;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    let acc = 0;
    for (let k = -half; k <= half; k++) {
      const j = Math.max(0, Math.min(n - 1, i + k));
      acc += y[j] * kernel[k + half];
    }
    out[i] = acc;
  }
  return out;
}

/**
 * Broaden a uniformly spaced 1D spectrum y(E) with an energy-dependent FWHM array `fwhmVsE`.
 *
 * Uses source-centered ("push") Gaussian deposition: each source bin `i` distributes `y[i]`
 * across neighboring bins `j` using a unit-sum Gaussian kernel of width `sigma(E[i])`.
 * This models the physical spectrometer response to a delta-function excitation at `E[i]`
 * and preserves total integrated spectral area `∑_j out[j] === ∑_i y[i]` identically.
 *
 * If `baseFwhm > 0`, subtracts `baseFwhm` in quadrature so pre-broadened spectra are not
 * double-broadened: `fwhmAdd(E) = sqrt(max(0, targetFwhm(E)^2 - baseFwhm^2))`.
 */
export function variableGaussianBroaden1D(E, y, fwhmVsE, baseFwhm = 0) {
  const n = E.length;
  if (n < 3 || !fwhmVsE || fwhmVsE.length !== n) return Array.from(y);
  const dE = Math.abs(E[1] - E[0]);
  if (!(dE > 0)) return Array.from(y);

  const out = new Float64Array(n);
  const base2 = baseFwhm > 0 ? baseFwhm * baseFwhm : 0;

  for (let i = 0; i < n; i++) {
    const yi = y[i];
    if (yi === 0 || !Number.isFinite(yi)) continue;
    const target = Math.max(0, fwhmVsE[i]);
    const effFwhm = Math.sqrt(Math.max(0, target * target - base2));
    const sigma = effFwhm / SQRT_8LN2;
    if (sigma < 0.25 * dE) {
      out[i] += yi;
      continue;
    }
    const half = Math.min(n - 1, Math.max(2, Math.ceil((4.0 * sigma) / dE)));
    const jLo = Math.max(0, i - half);
    const jHi = Math.min(n - 1, i + half);
    const weights = new Float64Array(jHi - jLo + 1);
    let wSum = 0;
    for (let j = jLo; j <= jHi; j++) {
      const x = (j - i) * dE;
      const w = Math.exp(-0.5 * (x / sigma) ** 2);
      weights[j - jLo] = w;
      wSum += w;
    }
    if (!(wSum > 0)) {
      out[i] += yi;
      continue;
    }
    const scale = yi / wSum;
    for (let j = jLo; j <= jHi; j++) {
      out[j] += scale * weights[j - jLo];
    }
  }
  return Array.from(out);
}

/**
 * Linearly interpolate (xs, ys) at target x (assuming xs is sorted ascending).
 */
export function interp1D(xs, ys, x) {
  const n = xs.length;
  if (n === 0) return 0;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid;
    else hi = mid;
  }
  const span = xs[hi] - xs[lo];
  if (!(span > 0)) return ys[lo];
  const t = (x - xs[lo]) / span;
  return ys[lo] * (1 - t) + ys[hi] * t;
}

/**
 * Compute the parameter-free Sjolander multiphonon (n = 2 .. maxOrder) contribution
 * corresponding to a 1D neutron-weighted GDOS `gdos1D(E)` on energy grid `E` [meV].
 *
 * How 1D GDOS and multiphonon S_multi(E) relate physically:
 *   Experimental "GDOS" extracted from powder INS in the incoherent approximation divides
 *   the measured S(Q, E) by `(Q^2 / E) * (n(E, T) + 1) * exp(-2W)`. Any unsubtracted
 *   n >= 2 multiphonon scattering S_n(Q, E) = exp(-2W) * [(2W)^n / n!] * f_n(E) therefore
 *   appears in the apparent experimental GDOS with exact relative weight:
 *
 *     g_multi^apparent(E) = Area(g_1) * [ E / ((n(E, T) + 1) * E_recoil_eff) ]
 *                           * sum_{n=2}^{maxOrder} [ (2W)^(n - 1) / n! ] * U_iso * E_recoil_per_Q2 * f_n(E)
 *
 *   Using the exact identity `f_1(+E) = (hbar^2 / (2 m_u M U_iso)) * (g_norm(E) / E) * (n(E)+1)`,
 *   the mass & constants cancel cleanly! Specifically, for `E > 0`:
 *     g_norm(E) = [ E / (n(E, T) + 1) ] * [ (2 m_u M U_iso) / hbar^2 ] * f_1(E),
 *   so converting `f_n(E)` into the SAME apparent-GDOS units as `gdos1D(E)` is simply:
 *     g_multi(E) = Area(gdos1D) * [ E / (n(E, T) + 1) ] * C_norm
 *                  * sum_{n=2}^{maxOrder} [ (2W)^(n - 1) / n! ] * f_n(E),
 *   where `C_norm` is the exact normalization constant that made `∫_{-∞}^{+∞} f_1(E) dE = 1`!
 *
 * Returns `{ multiphononGdos, totalEffectiveGdos, twoW, uIsoEff }` where `multiphononGdos`
 * is directly additive to `gdos1D` on the exact same vertical scale!
 */
export function computeMultiphonon1D(E, gdos1D, opts = {}) {
  const {
    temperatureK = 5,
    meanQ = 4.0,
    uIso = null,
    meanMassAmu = 60.0,
    maxOrder = 4,
  } = opts;

  const n = E.length;
  if (n < 5 || maxOrder < 2) {
    return {
      multiphononGdos: new Array(n).fill(0),
      totalEffectiveGdos: Array.from(gdos1D),
      twoW: 0,
      uIsoEff: uIso || 0,
    };
  }

  const T = Math.max(1.0, Number(temperatureK) || 5);
  const kBT = KB_MEV_PER_K * T;

  // Compute area of input gdos1D over E > 0 so scale is preserved regardless of input normalization
  let areaIn = 0;
  for (let i = 1; i < n; i++) {
    if (E[i] > 0) {
      const dEi = E[i] - E[i - 1];
      areaIn += 0.5 * (Math.max(0, gdos1D[i - 1]) + Math.max(0, gdos1D[i])) * dEi;
    }
  }
  if (!(areaIn > 1e-15)) {
    return {
      multiphononGdos: new Array(n).fill(0),
      totalEffectiveGdos: Array.from(gdos1D),
      twoW: 0,
      uIsoEff: 0,
    };
  }

  const gNorm = gdos1D.map(v => Math.max(0, v) / areaIn);

  // Estimate harmonic U_iso(T) [Å^2] self-consistently from gNorm(E) if not supplied:
  //   U_iso(T) = (hbar^2 / (2 m_u M)) * ∫ (g_norm(E) / E) * coth(E / 2 k_B T) dE
  let uIsoEff = Number(uIso);
  if (!(uIsoEff > 0)) {
    let integral = 0;
    for (let i = 1; i < n; i++) {
      const eMid = 0.5 * (E[i - 1] + E[i]);
      if (eMid <= 0.5) continue;
      const gMid = 0.5 * (gNorm[i - 1] + gNorm[i]);
      const x = Math.min(500, eMid / kBT);
      const twoNPlus1 = x > 80 ? 1 : (Math.exp(x) + 1) / (Math.exp(x) - 1);
      integral += (gMid / eMid) * twoNPlus1 * (E[i] - E[i - 1]);
    }
    const mass = Math.max(1.0, Number(meanMassAmu) || 60.0);
    uIsoEff = Math.max(0.0005, Math.min(0.12, (HBAR2_OVER_2MU / mass) * integral));
  }

  const twoW = Math.max(0, meanQ * meanQ * uIsoEff);
  if (twoW <= 1e-8) {
    return {
      multiphononGdos: new Array(n).fill(0),
      totalEffectiveGdos: Array.from(gdos1D),
      twoW,
      uIsoEff,
    };
  }

  // Build symmetric grid [-E_max, +E_max] for two-sided creation + annihilation convolution
  const eMax = Math.max(5.0, E[n - 1]);
  const dE = Math.max(0.08, (E[n - 1] - E[0]) / Math.max(1, n - 1));
  const nHalf = Math.min(600, Math.max(32, Math.ceil((1.15 * eMax) / dE)));
  const mLen = 2 * nHalf + 1;
  const symE = new Float64Array(mLen);
  const f1Raw = new Float64Array(mLen);
  const eCut = Math.max(0.4, 1.5 * dE);

  for (let k = 0; k < mLen; k++) {
    const es = (k - nHalf) * dE;
    symE[k] = es;
    const absE = Math.abs(es);
    if (absE < eCut || absE > E[n - 1]) continue;
    const gVal = Math.max(0, interp1D(E, gNorm, absE));
    const x = Math.min(500, absE / kBT);
    if (es > 0) {
      // Phonon creation: (g(E) / E) * (n(E) + 1)
      const nPlus1 = 1 / (1 - Math.exp(-x));
      f1Raw[k] = (gVal / absE) * nPlus1;
    } else {
      // Phonon annihilation: (g(|E|) / |E|) * n(|E|)
      const nOcc = x > 100 ? 0 : 1 / (Math.exp(x) - 1);
      f1Raw[k] = (gVal / absE) * nOcc;
    }
  }

  // Normalize f1 = f1Raw / rawIntegral so ∫_{-∞}^{+∞} f1(E) dE == 1
  let rawIntegral = 0;
  for (let k = 0; k < mLen; k++) rawIntegral += f1Raw[k] * dE;
  if (!(rawIntegral > 1e-15)) {
    return {
      multiphononGdos: new Array(n).fill(0),
      totalEffectiveGdos: Array.from(gdos1D),
      twoW,
      uIsoEff,
    };
  }

  const f1 = new Float64Array(mLen);
  for (let k = 0; k < mLen; k++) f1[k] = f1Raw[k] / rawIntegral;

  // Accumulate sum_{n=2}^{maxOrder} [(2W)^(n - 1) / n!] * f_n(E) on symE
  let fn = Float64Array.from(f1);
  const multiSym = new Float64Array(mLen);
  let fact = 1;
  let pow2W = 1; // (2W)^(order - 1)

  for (let order = 2; order <= maxOrder; order++) {
    fact *= order;
    pow2W *= twoW;
    const nextFn = new Float64Array(mLen);
    // Direct discrete convolution cropped to [-nHalf .. +nHalf]:
    //   (fn * f1)[k] = dE * sum_j fn[j] * f1[k - j + nHalf]
    for (let k = 0; k < mLen; k++) {
      let acc = 0;
      const jMin = Math.max(0, k - nHalf);
      const jMax = Math.min(mLen - 1, k + nHalf);
      for (let j = jMin; j <= jMax; j++) {
        acc += fn[j] * f1[k - j + nHalf];
      }
      nextFn[k] = acc * dE;
    }
    fn = nextFn;
    const coeff = pow2W / fact;
    for (let k = 0; k < mLen; k++) multiSym[k] += coeff * fn[k];
  }

  // Convert back to apparent GDOS scale matching gdos1D(E):
  // Since f1(E>0) = (gNorm(E) / E) * (n(E)+1) / rawIntegral,
  // gNorm(E) corresponds to rawIntegral * (E / (n(E)+1)) * f1(E).
  const multiphononGdos = new Array(n).fill(0);
  const totalEffectiveGdos = new Array(n);
  const symEArr = Array.from(symE);
  const multiArr = Array.from(multiSym);

  for (let i = 0; i < n; i++) {
    const en = E[i];
    if (en <= eCut) {
      totalEffectiveGdos[i] = gdos1D[i];
      continue;
    }
    const x = Math.min(500, en / kBT);
    const nPlus1 = 1 / (1 - Math.exp(-x));
    const fMultiAtE = Math.max(0, interp1D(symEArr, multiArr, en));
    const gMulti = areaIn * rawIntegral * (en / nPlus1) * fMultiAtE;
    multiphononGdos[i] = gMulti;
    totalEffectiveGdos[i] = gdos1D[i] + gMulti;
  }

  return { multiphononGdos, totalEffectiveGdos, twoW, uIsoEff };
}

/**
 * Fit calc(E) to exp(E) over [eMin, eMax] via weighted least squares:
 *   y_fit = scale * y_calc + offset   (offset = 0 unless fitBackground = true)
 * Returns { scale, offset, rw, chi2, cosine, nPts, expE, fittedCalc, residual }.
 */
export function fitAndScore1D(expE, expI, expSigma, calcE, calcI, opts = {}) {
  const { eMin = -Infinity, eMax = Infinity, fitBackground = false } = opts;

  const xs = [];
  const yo = [];
  const yc = [];
  const wts = [];

  for (let i = 0; i < expE.length; i++) {
    const e = expE[i];
    if (e < eMin || e > eMax) continue;
    const o = expI[i];
    const c = interp1D(calcE, calcI, e);
    if (!Number.isFinite(o) || !Number.isFinite(c)) continue;
    const sig = expSigma?.[i];
    const w = (sig && sig > 0) ? 1 / (sig * sig) : 1;
    xs.push(e);
    yo.push(o);
    yc.push(c);
    wts.push(w);
  }

  const n = xs.length;
  if (n < 2) {
    return { scale: 1, offset: 0, rw: NaN, chi2: NaN, cosine: NaN, nPts: n, expE: xs, fittedCalc: [], residual: [] };
  }

  let scale = 1, offset = 0;
  if (!fitBackground) {
    let num = 0, den = 0;
    for (let i = 0; i < n; i++) {
      num += wts[i] * yo[i] * yc[i];
      den += wts[i] * yc[i] * yc[i];
    }
    scale = den > 1e-18 ? num / den : 1;
  } else {
    let sw = 0, sc = 0, so = 0, scc = 0, sco = 0;
    for (let i = 0; i < n; i++) {
      const w = wts[i];
      sw += w;
      sc += w * yc[i];
      so += w * yo[i];
      scc += w * yc[i] * yc[i];
      sco += w * yc[i] * yo[i];
    }
    const det = sw * scc - sc * sc;
    if (Math.abs(det) > 1e-18) {
      scale = (sw * sco - sc * so) / det;
      offset = (scc * so - sc * sco) / det;
    }
  }

  let rssW = 0, obsW = 0, dot = 0, no2 = 0, nf2 = 0;
  const fittedCalc = new Array(n);
  const residual = new Array(n);
  for (let i = 0; i < n; i++) {
    const f = scale * yc[i] + offset;
    const r = yo[i] - f;
    fittedCalc[i] = f;
    residual[i] = r;
    rssW += wts[i] * r * r;
    obsW += wts[i] * yo[i] * yo[i];
    dot += yo[i] * f;
    no2 += yo[i] * yo[i];
    nf2 += f * f;
  }

  const rw = obsW > 1e-18 ? Math.sqrt(rssW / obsW) : NaN;
  const dof = Math.max(1, n - (fitBackground ? 2 : 1));
  const chi2 = rssW / dof;
  const cosine = (no2 > 1e-18 && nf2 > 1e-18) ? dot / Math.sqrt(no2 * nf2) : NaN;

  return { scale, offset, rw, chi2, cosine, nPts: n, expE: xs, fittedCalc, residual };
}

/**
 * Bilinear interpolation of a 2D grid sqe[ei][qi] at (q, e).
 */
function interp2D(qGrid, eGrid, sqe, q, e) {
  const nQ = qGrid.length, nE = eGrid.length;
  if (q < qGrid[0] || q > qGrid[nQ - 1] || e < eGrid[0] || e > eGrid[nE - 1]) return 0;
  let qi = 0;
  while (qi < nQ - 2 && qGrid[qi + 1] <= q) qi++;
  let ei = 0;
  while (ei < nE - 2 && eGrid[ei + 1] <= e) ei++;
  const tq = (qGrid[qi + 1] > qGrid[qi]) ? (q - qGrid[qi]) / (qGrid[qi + 1] - qGrid[qi]) : 0;
  const te = (eGrid[ei + 1] > eGrid[ei]) ? (e - eGrid[ei]) / (eGrid[ei + 1] - eGrid[ei]) : 0;
  const v00 = sqe[ei][qi], v10 = sqe[ei][qi + 1];
  const v01 = sqe[ei + 1][qi], v11 = sqe[ei + 1][qi + 1];
  return (v00 * (1 - tq) + v10 * tq) * (1 - te) + (v01 * (1 - tq) + v11 * tq) * te;
}

/**
 * Select the requested physical scattering channel from a `powder_ins` payload:
 *   - 'total':      sqe (coherent + incoherent + multiphonon)
 *   - 'one_phonon': sqe_coherent + sqe_incoherent
 *   - 'coherent':   sqe_coherent
 *   - 'incoherent': sqe_incoherent
 *   - 'multiphonon': sqe_multiphonon
 */
export function selectPowderInsChannel(powderIns, channel = 'total') {
  if (!powderIns?.sqe) return null;
  if (channel === 'coherent' && powderIns.sqe_coherent) return powderIns.sqe_coherent;
  if (channel === 'incoherent' && powderIns.sqe_incoherent) return powderIns.sqe_incoherent;
  if (channel === 'multiphonon' && powderIns.sqe_multiphonon) return powderIns.sqe_multiphonon;
  if (channel === 'one_phonon' && powderIns.sqe_coherent && powderIns.sqe_incoherent) {
    return powderIns.sqe_coherent.map((row, ei) =>
      row.map((v, qi) => v + (powderIns.sqe_incoherent[ei]?.[qi] || 0))
    );
  }
  return powderIns.sqe;
}

/**
 * Resample candidate 2D S(|Q|, E) onto experimental grid and compute optimal scale,
 * difference map (calc - exp), Rw, and cosine similarity.
 */
export function compare2DSqe(candIns, expIns, channel = 'total') {
  const candGrid = selectPowderInsChannel(candIns, channel) || candIns.sqe;
  const { q_grid: qExp, e_grid: eExp, sqe: sExp } = expIns;
  const nE = eExp.length, nQ = qExp.length;
  const calcResampled = Array.from({ length: nE }, (_, ei) => {
    const row = new Array(nQ);
    for (let qi = 0; qi < nQ; qi++) {
      row[qi] = interp2D(candIns.q_grid, candIns.e_grid, candGrid, qExp[qi], eExp[ei]);
    }
    return row;
  });

  let num = 0, den = 0;
  for (let ei = 0; ei < nE; ei++) {
    for (let qi = 0; qi < nQ; qi++) {
      const o = sExp[ei][qi], c = calcResampled[ei][qi];
      if (!Number.isFinite(o) || !Number.isFinite(c)) continue;
      num += o * c;
      den += c * c;
    }
  }
  const scale = den > 1e-18 ? num / den : 1;

  let rss = 0, obs2 = 0, dot = 0, c2 = 0;
  const calcScaled = Array.from({ length: nE }, () => new Array(nQ));
  const diffGrid = Array.from({ length: nE }, () => new Array(nQ));
  for (let ei = 0; ei < nE; ei++) {
    for (let qi = 0; qi < nQ; qi++) {
      const o = sExp[ei][qi];
      const cRaw = calcResampled[ei][qi];
      const c = Number.isFinite(cRaw) ? cRaw * scale : 0;
      calcScaled[ei][qi] = c;
      if (!Number.isFinite(o)) {
        diffGrid[ei][qi] = 0;
        continue;
      }
      diffGrid[ei][qi] = c - o;
      rss += (o - c) ** 2;
      obs2 += o * o;
      dot += o * c;
      c2 += c * c;
    }
  }

  return {
    q_grid: qExp,
    e_grid: eExp,
    expGrid: sExp,
    calcGrid: calcScaled,
    diffGrid,
    scale,
    rw: obs2 > 1e-18 ? Math.sqrt(rss / obs2) : NaN,
    cosine: (obs2 > 1e-18 && c2 > 1e-18) ? dot / Math.sqrt(obs2 * c2) : NaN,
  };
}

/** Integrate 2D S(Q, E) over Q in [qLo, qHi] -> 1D S(E) curve (skipping NaN kinematic masks). */
export function integrateOverQ(sqe, qGrid, qLo, qHi) {
  const nE = sqe.length;
  const out = new Array(nE).fill(0);
  const counts = new Int32Array(nE);
  for (let qi = 0; qi < qGrid.length; qi++) {
    if (qGrid[qi] < qLo || qGrid[qi] > qHi) continue;
    for (let ei = 0; ei < nE; ei++) {
      const v = sqe[ei]?.[qi];
      if (Number.isFinite(v)) {
        out[ei] += v;
        counts[ei] += 1;
      }
    }
  }
  for (let ei = 0; ei < nE; ei++) {
    if (counts[ei] > 0) out[ei] /= counts[ei];
  }
  return out;
}

/** Integrate 2D S(Q, E) over E in [eLo, eHi] -> 1D S(|Q|) curve (skipping NaN kinematic masks). */
export function integrateOverE(sqe, eGrid, eLo, eHi) {
  const nQ = sqe[0]?.length || 0;
  const out = new Array(nQ).fill(0);
  const counts = new Int32Array(nQ);
  for (let ei = 0; ei < eGrid.length; ei++) {
    if (eGrid[ei] < eLo || eGrid[ei] > eHi) continue;
    for (let qi = 0; qi < nQ; qi++) {
      const v = sqe[ei]?.[qi];
      if (Number.isFinite(v)) {
        out[qi] += v;
        counts[qi] += 1;
      }
    }
  }
  for (let qi = 0; qi < nQ; qi++) {
    if (counts[qi] > 0) out[qi] /= counts[qi];
  }
  return out;
}
