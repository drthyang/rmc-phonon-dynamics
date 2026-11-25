// web/src/math/insCompare.js
//
// Quantitative 1D GDOS / S(E) and 2D S(|Q|, E) comparison utilities:
//   - Live Gaussian resolution broadening
//   - Linear interpolation + least-squares scale/background fit
//   - Rw, reduced chi^2, cosine similarity, residual curves
//   - 2D resampling, difference map, and 1D Q/E slice integrations

const SQRT_8LN2 = Math.sqrt(8 * Math.LN2);

/**
 * Convolve a uniformly spaced 1D spectrum y(E) with a Gaussian of given FWHM (meV).
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
 * Resample candidate 2D S(|Q|, E) onto experimental grid and compute optimal scale,
 * difference map (calc - exp), Rw, and cosine similarity.
 */
export function compare2DSqe(candIns, expIns) {
  const { q_grid: qExp, e_grid: eExp, sqe: sExp } = expIns;
  const nE = eExp.length, nQ = qExp.length;
  const calcResampled = Array.from({ length: nE }, (_, ei) => {
    const row = new Array(nQ);
    for (let qi = 0; qi < nQ; qi++) {
      row[qi] = interp2D(candIns.q_grid, candIns.e_grid, candIns.sqe, qExp[qi], eExp[ei]);
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
      const c = calcResampled[ei][qi] * scale;
      calcScaled[ei][qi] = c;
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

/** Integrate 2D S(Q, E) over Q in [qLo, qHi] -> 1D S(E) curve. */
export function integrateOverQ(sqe, qGrid, qLo, qHi) {
  const nE = sqe.length;
  const out = new Array(nE).fill(0);
  let count = 0;
  for (let qi = 0; qi < qGrid.length; qi++) {
    if (qGrid[qi] < qLo || qGrid[qi] > qHi) continue;
    count++;
    for (let ei = 0; ei < nE; ei++) out[ei] += sqe[ei][qi] || 0;
  }
  if (count > 0) for (let ei = 0; ei < nE; ei++) out[ei] /= count;
  return out;
}

/** Integrate 2D S(Q, E) over E in [eLo, eHi] -> 1D S(|Q|) curve. */
export function integrateOverE(sqe, eGrid, eLo, eHi) {
  const nQ = sqe[0]?.length || 0;
  const out = new Array(nQ).fill(0);
  let count = 0;
  for (let ei = 0; ei < eGrid.length; ei++) {
    if (eGrid[ei] < eLo || eGrid[ei] > eHi) continue;
    count++;
    for (let qi = 0; qi < nQ; qi++) out[qi] += sqe[ei][qi] || 0;
  }
  if (count > 0) for (let qi = 0; qi < nQ; qi++) out[qi] /= count;
  return out;
}
