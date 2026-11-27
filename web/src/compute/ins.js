// web/src/compute/ins.js
//
// Prepare transferable phonon + crystallographic scattering arrays for the
// off-main-thread powder INS S(|Q|, E) worker (`io/sqeworker.js`).
//
// Physics & Bloch-Gauge Normalization
// -----------------------------------
// Coherent one-phonon neutron scattering depends on:
//   1. Signed coherent scattering length beta_coh,s = sqrt(4*pi) * (b_coh,s [fm] / 10) [sqrt(barn)],
//      preserving negative scattering lengths (H, Li, Ti, V, Mn) and inter-sublattice interference.
//   2. Nuclear incoherent cross-section sigma_inc,s = max(0, sigma_tot,s - beta_coh,s^2) [barn].
//   3. Basis fractional coordinates x_s in the computation cell and isotropic Debye-Waller
//      mean-square displacements U_iso,s(T) [Å^2].
//   4. Complex polarization eigenvectors e_{s,nu}^C(q) in Phonopy's C-type (atomic-position)
//      Bloch gauge.
//
// Gauge Subtlety Fixed Here:
//   * Phonopy (`phonopyDM.js` and `band.yaml`) uses the C-type gauge (shortest vectors R + x_j - x_i),
//     where e_{s,nu}^C(q) already includes exp(-i 2*pi q . x_s).
//   * The RMC ensemble S(k) solver (`results.source === 'runner'`) uses the L-type (integer-cell R)
//     Bloch gauge, which does NOT include exp(-i 2*pi q . x_s).
//   * We multiply L-type eigenvectors once here by exp(-i 2*pi q . x_s) so `sqeworker.js` receives
//     uniform C-type eigenvectors regardless of whether the phonons came from RMC, band.yaml,
//     or a phonopy dynamical matrix!
//
// Full 3D Brillouin-Zone Mesh Option:
//   When `results.dynamicalMatrixPayload` is available (`_mlip_dm.json` or phonopy DM JSON),
//   k-path q-points only sample 1D lines. Setting `useFullMesh = true` evaluates D(q) on a
//   time-reversal-folded 3D Monkhorst-Pack mesh (e.g. 9x9x9 -> 365 unique q-points) so powder
//   S(|Q|, E) represents a true 3D Brillouin-zone spherical average.

import {
  ATOMIC_MASS,
  B_COH,
  NEUTRON_SCATT_SIGMA,
  KB_MEV_PER_K,
  HBAR2_OVER_2MU,
} from '../constants.js';
import { conventionalLattice, reciprocalLattice } from '../math/reciprocal.js';
import { solveDynamicalMatrixAtQ } from '../io/phonopyDM.js';

const SQRT_4PI_OVER_10 = Math.sqrt(4 * Math.PI) / 10;
// Cap zero-copy Float32 eigenvector buffer at ~64 MB (8M complex components)
const MAX_EIGVEC_FLOATS = 8_000_000;

function crossSectionsOf(el) {
  const sigmaTot = NEUTRON_SCATT_SIGMA[el] ?? 5.0;
  if (el in B_COH) {
    const betaCoh = SQRT_4PI_OVER_10 * B_COH[el];
    const sigmaCoh = betaCoh * betaCoh;
    const sigmaInc = Math.max(0, Math.max(sigmaTot, sigmaCoh) - sigmaCoh);
    return { betaCoh, sigmaCoh, sigmaInc, sigmaTot: Math.max(sigmaTot, sigmaCoh) };
  }
  return { betaCoh: Math.sqrt(sigmaTot), sigmaCoh: sigmaTot, sigmaInc: 0, sigmaTot };
}

/**
 * Generate a time-reversal-folded uniform 3D Gamma-centered mesh `N x N x N` in [-0.5, 0.5).
 * Because D(-q) = D*(q), modes at +q and -q have identical frequencies and |F(-q - G)|^2 = |F(q + G)|^2,
 * halving the number of diagonalizations while preserving exact 3D BZ integration weights!
 */
function buildTimeReversalMesh(nGrid) {
  const qPoints = [];
  const qWeights = [];
  const visited = new Uint8Array(nGrid * nGrid * nGrid);
  const idxOf = (i, j, k) => (i * nGrid + j) * nGrid + k;

  for (let i = 0; i < nGrid; i++) {
    for (let j = 0; j < nGrid; j++) {
      for (let k = 0; k < nGrid; k++) {
        const id = idxOf(i, j, k);
        if (visited[id]) continue;
        const ni = (nGrid - i) % nGrid;
        const nj = (nGrid - j) % nGrid;
        const nk = (nGrid - k) % nGrid;
        const invId = idxOf(ni, nj, nk);
        visited[id] = 1;
        visited[invId] = 1;

        const wrap = (u) => {
          const f = u / nGrid;
          return f >= 0.5 ? f - 1.0 : f;
        };
        qPoints.push([wrap(i), wrap(j), wrap(k)]);
        qWeights.push(id === invId ? 1.0 : 2.0);
      }
    }
  }
  return { qPoints, qWeights };
}

export function buildInsData(results, opts = {}) {
  const { temperatureK = results.temperature ?? 5, useFullMesh = true } = opts;
  const bs = results.baseStructure;
  const { atomDic, dim, v1, v2, v3, uniqueRN, hsym_xyz, masses: bsMasses } = bs;
  const recip = bs.bandRecip || reciprocalLattice(conventionalLattice(v1, v2, v3, dim));

  // Extract basis sites in eigenvector-row order with element, mass, fractional coords, and cross-sections
  const sites = [];
  if (bs.segSymbols && bs.segSymbols.length) {
    bs.segSymbols.forEach((el, r) => {
      const m = bsMasses?.[r] || ATOMIC_MASS[el] || 50;
      const fx = hsym_xyz ? hsym_xyz[r * 3] : 0;
      const fy = hsym_xyz ? hsym_xyz[r * 3 + 1] : 0;
      const fz = hsym_xyz ? hsym_xyz[r * 3 + 2] : 0;
      sites.push({ row: r, el, mass: m, frac: [fx, fy, fz], ...crossSectionsOf(el) });
    });
  } else {
    const rnToEl = new Map();
    for (const el of Object.keys(atomDic || {})) {
      for (const rn of atomDic[el]) rnToEl.set(rn, el);
    }
    const orderedRNs = uniqueRN && uniqueRN.length ? Array.from(uniqueRN) : [...rnToEl.keys()];
    orderedRNs.forEach((rn, r) => {
      const el = rnToEl.get(rn) || 'X';
      const m = bsMasses?.[r] || ATOMIC_MASS[el] || 50;
      const fx = hsym_xyz ? hsym_xyz[r * 3] : 0;
      const fy = hsym_xyz ? hsym_xyz[r * 3 + 1] : 0;
      const fz = hsym_xyz ? hsym_xyz[r * 3 + 2] : 0;
      sites.push({ row: r, el, mass: m, frac: [fx, fy, fz], ...crossSectionsOf(el) });
    });
  }
  const nSites = sites.length;

  // Choose q-point set: full 3D BZ mesh if dynamicalMatrixPayload is available, else loaded qPoints
  let qList = results.qPoints;
  let qWeightsArr = null;
  let bandsList = results.bands;
  let eigvecsList = results.eigvecs;
  let isLTypeGauge = results.source === 'runner';

  if (useFullMesh && results.dynamicalMatrixPayload && nSites > 0 && nSites <= 24) {
    const nGrid = nSites <= 6 ? 9 : nSites <= 12 ? 7 : 5;
    const mesh = buildTimeReversalMesh(nGrid);
    qList = mesh.qPoints;
    qWeightsArr = Float64Array.from(mesh.qWeights);
    bandsList = new Array(qList.length);
    eigvecsList = new Array(qList.length);
    for (let qi = 0; qi < qList.length; qi++) {
      const solved = solveDynamicalMatrixAtQ(results.dynamicalMatrixPayload, qList[qi]);
      bandsList[qi] = solved.energiesMev;
      eigvecsList[qi] = solved.eigvecs;
    }
    isLTypeGauge = false; // solveDynamicalMatrixAtQ uses Phonopy's C-type gauge
  }

  const nq = qList.length;
  const nModes = bandsList[0].length;
  if (!qWeightsArr) {
    qWeightsArr = new Float64Array(nq).fill(1.0);
  }

  const freqs = new Float64Array(nq * nModes);
  const qpos = new Float64Array(nq * 3);
  const F2 = new Float64Array(nq * nModes); // Kept for legacy / incoherent fallback compatibility

  const hasEig = Boolean(eigvecsList?.[0]?.[0]?.real);
  const totalEigFloats = nq * nModes * 3 * nSites;
  const packEigvecs = hasEig && nSites > 0 && totalEigFloats <= MAX_EIGVEC_FLOATS;
  const eigRe = packEigvecs ? new Float32Array(totalEigFloats) : null;
  const eigIm = packEigvecs ? new Float32Array(totalEigFloats) : null;

  // Accumulate harmonic isotropic mean-square displacement U_iso,s(T) [Å^2] per site
  const T = Math.max(1.0, Number(temperatureK) || 5);
  const kBT = KB_MEV_PER_K * T;
  const uIsoAcc = new Float64Array(nSites);
  let wTotal = 0;

  for (let qi = 0; qi < nq; qi++) {
    const q = qList[qi];
    const wq = qWeightsArr[qi];
    wTotal += wq;
    qpos[qi * 3] = q[0];
    qpos[qi * 3 + 1] = q[1];
    qpos[qi * 3 + 2] = q[2];

    for (let m = 0; m < nModes; m++) {
      const omega = bandsList[qi][m];
      freqs[qi * nModes + m] = omega;
      const ev = eigvecsList?.[qi]?.[m];
      if (!ev) continue;
      const re = ev.real, im = ev.imag;

      const absE = Math.abs(omega);
      const xB = absE > 0.4 ? Math.min(500, absE / kBT) : 0;
      const twoNPlus1 = absE > 0.4 ? (xB > 80 ? 1 : (Math.exp(xB) + 1) / (Math.exp(xB) - 1)) : 0;

      let f2Legacy = 0;
      const baseEig = (qi * nModes + m) * (3 * nSites);

      for (let s = 0; s < nSites; s++) {
        const site = sites[s];
        const r = site.row * 3;
        let rx0 = re[r], ix0 = im[r];
        let rx1 = re[r + 1], ix1 = im[r + 1];
        let rx2 = re[r + 2], ix2 = im[r + 2];

        // Convert RMC L-type eigenvector to Phonopy C-type gauge via exp(-i 2*pi q . x_s)
        if (isLTypeGauge) {
          const ph = -2 * Math.PI * (q[0] * site.frac[0] + q[1] * site.frac[1] + q[2] * site.frac[2]);
          const c = Math.cos(ph), sn = Math.sin(ph);
          const nrx0 = rx0 * c - ix0 * sn, nix0 = rx0 * sn + ix0 * c;
          const nrx1 = rx1 * c - ix1 * sn, nix1 = rx1 * sn + ix1 * c;
          const nrx2 = rx2 * c - ix2 * sn, nix2 = rx2 * sn + ix2 * c;
          rx0 = nrx0; ix0 = nix0;
          rx1 = nrx1; ix1 = nix1;
          rx2 = nrx2; ix2 = nix2;
        }

        if (packEigvecs) {
          const dst = baseEig + s * 3;
          eigRe[dst] = rx0;     eigIm[dst] = ix0;
          eigRe[dst + 1] = rx1; eigIm[dst + 1] = ix1;
          eigRe[dst + 2] = rx2; eigIm[dst + 2] = ix2;
        }

        const mag2 = rx0 * rx0 + ix0 * ix0 + rx1 * rx1 + ix1 * ix1 + rx2 * rx2 + ix2 * ix2;
        const bFm = B_COH[site.el] || 0;
        if (site.mass > 0) {
          f2Legacy += ((bFm * bFm) / site.mass) * mag2;
          if (absE > 0.4) {
            uIsoAcc[s] += wq * (HBAR2_OVER_2MU / (3 * site.mass * absE)) * twoNPlus1 * mag2;
          }
        }
      }
      F2[qi * nModes + m] = f2Legacy;
    }
  }

  const siteFrac = new Float64Array(nSites * 3);
  const siteMass = new Float64Array(nSites);
  const siteBetaOverSqrtM = new Float64Array(nSites);
  const siteSigmaCoh = new Float64Array(nSites);
  const siteSigmaInc = new Float64Array(nSites);
  const siteSigmaTot = new Float64Array(nSites);
  const siteUiso = new Float64Array(nSites);

  const precompUiso = results.powderIns?.u_iso_by_element || null;
  const precompT = results.powderIns?.temperature_k || T;

  for (let s = 0; s < nSites; s++) {
    const site = sites[s];
    siteFrac[s * 3] = site.frac[0];
    siteFrac[s * 3 + 1] = site.frac[1];
    siteFrac[s * 3 + 2] = site.frac[2];
    siteMass[s] = site.mass;
    siteBetaOverSqrtM[s] = site.mass > 0 ? site.betaCoh / Math.sqrt(site.mass) : 0;
    siteSigmaCoh[s] = site.sigmaCoh;
    siteSigmaInc[s] = site.sigmaInc;
    siteSigmaTot[s] = site.sigmaTot;

    let uVal = wTotal > 0 ? uIsoAcc[s] / wTotal : 0.004;
    if (precompUiso && precompUiso[site.el] > 0 && precompT > 0) {
      // Scale Phonopy's exact U_iso smoothly if user adjusted T
      uVal = precompUiso[site.el] * Math.max(0.25, T / precompT);
    }
    siteUiso[s] = Math.max(0.0004, Math.min(0.15, uVal));
  }

  const transfer = [
    freqs.buffer,
    qpos.buffer,
    qWeightsArr.buffer,
    F2.buffer,
    siteFrac.buffer,
    siteMass.buffer,
    siteBetaOverSqrtM.buffer,
    siteSigmaCoh.buffer,
    siteSigmaInc.buffer,
    siteSigmaTot.buffer,
    siteUiso.buffer,
  ];
  if (eigRe && eigIm) {
    transfer.push(eigRe.buffer, eigIm.buffer);
  }

  return {
    data: {
      nq,
      nModes,
      nSites,
      freqs,
      qpos,
      qWeights: qWeightsArr,
      F2,
      eigRe,
      eigIm,
      siteFrac,
      siteMass,
      siteBetaOverSqrtM,
      siteSigmaCoh,
      siteSigmaInc,
      siteSigmaTot,
      siteUiso,
      recip,
    },
    transfer,
  };
}
