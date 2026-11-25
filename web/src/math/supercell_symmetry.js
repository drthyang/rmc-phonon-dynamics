// web/src/math/supercell_symmetry.js
//
// Direct supercell symmetry analysis for RMC / MD ensembles.
// Solves the fundamental bottleneck of folding a supercell into a single
// averaged unit cell before running FINDSYM:
//   - Zone-boundary / superlattice distortions (k != 0) cancel in unit-cell folding.
//   - Multi-domain local distortions (k = 0, twin variants) cancel in unit-cell folding.
//   - Raw supercell coordinates always yield P1 under deterministic tolerance finders.
//
// Four complementary supercell-native analyses (zero external dependencies):
//   1. kSpaceModeSpectrum     — 3D Fourier mode power P(k), SNR over background,
//                               active k-stars, and spatial correlation length ξ.
//   2. localSubgroupAnalysis  — Per-cell orthogonal projection onto every closed
//                               subgroup H <= G0 (distinguishing conjugate domain
//                               orientations), variance explained R^2, DoF-normalized
//                               enrichment / F-ratio, domain populations & neighbor
//                               correlation.
//   3. statisticalSymmetryTest — Continuous symmetry-breaking residual ε(g) for each
//                               operation g ∈ G0 on the k-space displacement covariance
//                               C(k) = <u(k) u(k)†> vs. on the folded mean <u>.
//   4. filteredMinimalSupercell — Reconstruct a denoised minimal commensurate supercell
//                               from dominant k-modes and run findSpaceGroupOps /
//                               symmetryLadder directly on it.

import { findSpaceGroupOps, classifyOperations, symmetryLadder } from './symmetry.js';
import { operationReps } from './symmetrize.js';

const TWO_PI = 2 * Math.PI;
const wrapHalf = (x) => x - Math.round(x);
const wrap01 = (x) => ((x % 1) + 1) % 1;

// ── Subgroup enumeration from parent space-group operations ──────────────────

function mat3Mul(A, B) {
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      C[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j];
  return C;
}

function opKey(R, t) {
  const r = R.flat().join(',');
  const q = t.map(x => Math.round(wrap01(x) * 24) % 24).join(',');
  return `${r}|${q}`;
}

function composeOp(a, b) {
  const R = mat3Mul(a.R, b.R);
  const t = [
    wrap01(a.R[0][0] * b.t[0] + a.R[0][1] * b.t[1] + a.R[0][2] * b.t[2] + a.t[0]),
    wrap01(a.R[1][0] * b.t[0] + a.R[1][1] * b.t[1] + a.R[1][2] * b.t[2] + a.t[1]),
    wrap01(a.R[2][0] * b.t[0] + a.R[2][1] * b.t[1] + a.R[2][2] * b.t[2] + a.t[2]),
  ];
  return { R, t };
}

/**
 * Enumerate closed subgroups of the parent space group (keeping all centering
 * translations so we classify proper crystal subgroups of the conventional cell).
 * Groups conjugate orientations of the same space-group class together.
 *
 * @param {{R:number[][], t:number[], perm:number[], shift:number[][], Rcart:number[][]}[]} reps
 * @returns {{ id:string, spaceGroup:string, spaceGroupNumber:number|null, pointGroup:string,
 *             centering:string, order:number, nPoint:number, variants:{ opIndices:number[], ops:object[] }[] }[]}
 */
export function enumerateSubgroups(reps) {
  const nOps = reps.length;
  const keyToIdx = new Map();
  for (let i = 0; i < nOps; i++) keyToIdx.set(opKey(reps[i].R, reps[i].t), i);

  // Build multiplication table (indices into reps)
  const mul = Array.from({ length: nOps }, () => new Int32Array(nOps).fill(-1));
  for (let i = 0; i < nOps; i++) {
    for (let j = 0; j < nOps; j++) {
      const c = composeOp(reps[i], reps[j]);
      const k = keyToIdx.get(opKey(c.R, c.t));
      if (k !== undefined) mul[i][j] = k;
    }
  }

  // Identify pure centering translations (R = I)
  const transIndices = [];
  for (let i = 0; i < nOps; i++) {
    const R = reps[i].R;
    if (R[0][0] === 1 && R[1][1] === 1 && R[2][2] === 1 &&
        R[0][1] === 0 && R[0][2] === 0 && R[1][0] === 0 &&
        R[1][2] === 0 && R[2][0] === 0 && R[2][1] === 0) {
      transIndices.push(i);
    }
  }

  // Closure of a seed set of op indices (always including pure translations for cell-preserving subgroups)
  function closeSet(seeds) {
    const inSet = new Uint8Array(nOps);
    const queue = [];
    for (const s of seeds) {
      if (!inSet[s]) { inSet[s] = 1; queue.push(s); }
    }
    let head = 0;
    while (head < queue.length) {
      const a = queue[head++];
      for (let b = 0; b < nOps; b++) {
        if (!inSet[b]) continue;
        const ab = mul[a][b];
        if (ab >= 0 && !inSet[ab]) { inSet[ab] = 1; queue.push(ab); }
        const ba = mul[b][a];
        if (ba >= 0 && !inSet[ba]) { inSet[ba] = 1; queue.push(ba); }
      }
    }
    const sorted = [];
    for (let i = 0; i < nOps; i++) if (inSet[i]) sorted.push(i);
    return sorted;
  }

  // Pick one representative per coset of pure translations to keep BFS fast
  const cosetReps = [];
  const seenR = new Set();
  for (let i = 0; i < nOps; i++) {
    const rk = reps[i].R.flat().join(',');
    if (!seenR.has(rk)) { seenR.add(rk); cosetReps.push(i); }
  }

  // BFS over closed subgroups generated by adding coset generators
  const found = new Map(); // bit-key -> opIndices[]
  const base = closeSet(transIndices);
  found.set(base.join(','), base);
  const bfs = [base];
  let idx = 0;
  while (idx < bfs.length) {
    const cur = bfs[idx++];
    const curSet = new Set(cur);
    for (const g of cosetReps) {
      if (curSet.has(g)) continue;
      const next = closeSet([...cur, g]);
      const key = next.join(',');
      if (!found.has(key)) {
        found.set(key, next);
        bfs.push(next);
      }
    }
  }

  // Build inverse table for conjugacy class grouping: g H g^-1
  const idIdx = transIndices.find(i =>
    reps[i].t[0] < 1e-3 && reps[i].t[1] < 1e-3 && reps[i].t[2] < 1e-3
  );
  const inv = new Int32Array(nOps).fill(-1);
  for (let i = 0; i < nOps; i++) {
    for (let j = 0; j < nOps; j++) {
      if (mul[i][j] === idIdx) { inv[i] = j; break; }
    }
  }

  // Also include the pure P1 subgroup (just identity) if centering > 1
  if (idIdx !== undefined && transIndices.length > 1) {
    const p1 = [idIdx];
    const k = p1.join(',');
    if (!found.has(k)) found.set(k, p1);
  }

  // Group into true conjugacy classes under G0: H' = g H g^-1
  // Every variant in a conjugacy class is strictly symmetry-equivalent (identical DoF & physics).
  // Also record `alignOpIdx` = g^-1 mapping each variant H_v back onto canonical H_0:
  //   c_v H_v c_v^-1 = H_0, plus normalizer coset representatives N(H_0)/H_0.
  const visited = new Set();
  const families = [];
  for (const [key, opIndices] of found.entries()) {
    if (visited.has(key)) continue;
    const conjToAlign = new Map(); // conjKey -> alignOpIdx (g^-1 such that g^-1 H_v g = H_0)
    conjToAlign.set(key, idIdx);   // guarantee variants[0] is canonical H_0 with identity alignment
    const normalizerOps = [];
    for (let g = 0; g < nOps; g++) {
      const gi = inv[g];
      if (gi < 0) continue;
      const conj = opIndices.map(h => mul[mul[g][h]][gi]).sort((a, b) => a - b);
      const ck = conj.join(',');
      if (ck === key) normalizerOps.push(g);
      if (!conjToAlign.has(ck)) conjToAlign.set(ck, gi);
    }
    const variants = [];
    for (const [ck, alignOpIdx] of conjToAlign.entries()) {
      visited.add(ck);
      const arr = found.get(ck);
      if (arr) variants.push({ opIndices: arr, ops: arr.map(i => reps[i]), alignOpIdx });
    }
    if (!variants.length) continue;

    // Extract one coset representative per coset of H_0 in N(H_0)
    const h0Set = new Set(variants[0].opIndices);
    const normCosetReps = [];
    const seenNorm = new Set();
    for (const n of normalizerOps) {
      if (seenNorm.has(n)) continue;
      normCosetReps.push(n);
      for (const h of h0Set) seenNorm.add(mul[n][h]);
    }

    const cls = classifyOperations(variants[0].ops, 0.02);
    families.push({
      id: `${cls.spaceGroup}|${cls.pointGroup}|${variants[0].ops.length}|${variants.length}|${key}`,
      spaceGroup: cls.spaceGroup,
      spaceGroupNumber: cls.spaceGroupNumber,
      pointGroup: cls.pointGroup,
      centering: cls.centering,
      order: variants[0].ops.length,
      nPoint: cls.nPoint,
      normCosetReps,
      variants,
    });
  }

  return families.sort((a, b) => b.order - a.order);
}

// ── Helper: apply Γ(g) to a single unit-cell displacement vector u (3*nBasis) ─

export function applyGammaCell(g, u, nBasis, out) {
  out.fill(0);
  const { perm, Rcart } = g;
  for (let s = 0; s < nBasis; s++) {
    const ps = perm[s] * 3;
    const rs = s * 3;
    const ux = u[rs], uy = u[rs + 1], uz = u[rs + 2];
    out[ps]     = Rcart[0][0] * ux + Rcart[0][1] * uy + Rcart[0][2] * uz;
    out[ps + 1] = Rcart[1][0] * ux + Rcart[1][1] * uy + Rcart[1][2] * uz;
    out[ps + 2] = Rcart[2][0] * ux + Rcart[2][1] * uy + Rcart[2][2] * uz;
  }
}

/**
 * Exact degrees of freedom of the H-invariant subspace in a unit cell:
 *   DoF(H) = Tr(P_H) = (1/|H|) Σ_{g ∈ H} Σ_{s: perm_g(s) = s} Tr(Rcart_g)
 * Optionally subtracts the H-invariant acoustic translation DoF (3 for P1, 1 for 3m/mm2, 0 for -43m).
 */
export function subgroupDoF(ops, nBasis, subtractAcoustic = true) {
  let trTotal = 0;
  let trAcoustic = 0;
  for (const g of ops) {
    const trR = g.Rcart[0][0] + g.Rcart[1][1] + g.Rcart[2][2];
    trAcoustic += trR;
    let fixedSites = 0;
    for (let s = 0; s < nBasis; s++) if (g.perm[s] === s) fixedSites++;
    trTotal += fixedSites * trR;
  }
  const dofAll = Math.round(trTotal / ops.length);
  const dofAc = Math.round(trAcoustic / ops.length);
  return {
    total: dofAll,
    acoustic: dofAc,
    internal: subtractAcoustic ? Math.max(0, dofAll - dofAc) : dofAll,
  };
}

// Subtract center-of-mass (uniform) translation from a single cell vector u (3*nBasis)
export function removeCellDrift(u, nBasis, out) {
  let mx = 0, my = 0, mz = 0;
  for (let s = 0; s < nBasis; s++) {
    mx += u[s * 3]; my += u[s * 3 + 1]; mz += u[s * 3 + 2];
  }
  mx /= nBasis; my /= nBasis; mz /= nBasis;
  for (let s = 0; s < nBasis; s++) {
    out[s * 3]     = u[s * 3]     - mx;
    out[s * 3 + 1] = u[s * 3 + 1] - my;
    out[s * 3 + 2] = u[s * 3 + 2] - mz;
  }
}

// ── Method 1: Supercell k-Space Mode Power Spectrum ──────────────────────────

// Simple radix-arbitrary 1D DFT (N is typically 4..16 for RMC supercells, so exact DFT is fast & exact for any N)
function dft1d(inRe, inIm, stride, count, N, outRe, outIm, sign = -1) {
  const cosT = new Float64Array(N * N);
  const sinT = new Float64Array(N * N);
  for (let k = 0; k < N; k++) {
    for (let n = 0; n < N; n++) {
      const ang = sign * TWO_PI * k * n / N;
      cosT[k * N + n] = Math.cos(ang);
      sinT[k * N + n] = Math.sin(ang);
    }
  }
  const scale = 1 / Math.sqrt(N);
  for (let c = 0; c < count; c++) {
    const base = c * stride;
    for (let k = 0; k < N; k++) {
      let sr = 0, si = 0;
      const row = k * N;
      for (let n = 0; n < N; n++) {
        const idx = base + n;
        const cr = cosT[row + n], ci = sinT[row + n];
        const xr = inRe[idx], xi = inIm[idx];
        sr += xr * cr - xi * ci;
        si += xr * ci + xi * cr;
      }
      outRe[base + k] = sr * scale;
      outIm[base + k] = si * scale;
    }
  }
}

/**
 * 3D unitary DFT of a single supercell displacement field u[nx, ny, nz, d]
 * where d = 0 .. 3*nBasis-1.
 * Returns { re: Float64Array, im: Float64Array } of same layout [kx, ky, kz, d].
 */
export function supercellFFT3D(u, dim, nBasis) {
  const [N1, N2, N3] = dim;
  const D = 3 * nBasis;
  const nCells = N1 * N2 * N3;
  const total = nCells * D;

  // Precompute 1D DFT tables for N1, N2, N3
  const makeTable = (N) => {
    const c = new Float64Array(N * N), s = new Float64Array(N * N);
    const sc = 1 / Math.sqrt(N);
    for (let k = 0; k < N; k++) {
      for (let n = 0; n < N; n++) {
        const ang = -TWO_PI * k * n / N;
        c[k * N + n] = Math.cos(ang) * sc;
        s[k * N + n] = Math.sin(ang) * sc;
      }
    }
    return { c, s };
  };
  const t1 = makeTable(N1), t2 = makeTable(N2), t3 = makeTable(N3);

  // Pass 1: transform along z (real input -> complex)
  const aRe = new Float64Array(total), aIm = new Float64Array(total);
  for (let x = 0; x < N1; x++) {
    for (let y = 0; y < N2; y++) {
      const xyBase = (x * N2 + y) * N3;
      for (let kz = 0; kz < N3; kz++) {
        const dstBase = (xyBase + kz) * D;
        const row = kz * N3;
        for (let z = 0; z < N3; z++) {
          const cr = t3.c[row + z], ci = t3.s[row + z];
          const srcBase = (xyBase + z) * D;
          for (let d = 0; d < D; d++) {
            const v = u[srcBase + d];
            aRe[dstBase + d] += v * cr;
            aIm[dstBase + d] += v * ci;
          }
        }
      }
    }
  }

  // Pass 2: transform along y
  const bRe = new Float64Array(total), bIm = new Float64Array(total);
  for (let x = 0; x < N1; x++) {
    for (let kz = 0; kz < N3; kz++) {
      for (let ky = 0; ky < N2; ky++) {
        const dstBase = ((x * N2 + ky) * N3 + kz) * D;
        const row = ky * N2;
        for (let y = 0; y < N2; y++) {
          const cr = t2.c[row + y], ci = t2.s[row + y];
          const srcBase = ((x * N2 + y) * N3 + kz) * D;
          for (let d = 0; d < D; d++) {
            const xr = aRe[srcBase + d], xi = aIm[srcBase + d];
            bRe[dstBase + d] += xr * cr - xi * ci;
            bIm[dstBase + d] += xr * ci + xi * cr;
          }
        }
      }
    }
  }

  // Pass 3: transform along x
  const outRe = new Float64Array(total), outIm = new Float64Array(total);
  for (let ky = 0; ky < N2; ky++) {
    for (let kz = 0; kz < N3; kz++) {
      for (let kx = 0; kx < N1; kx++) {
        const dstBase = ((kx * N2 + ky) * N3 + kz) * D;
        const row = kx * N1;
        for (let x = 0; x < N1; x++) {
          const cr = t1.c[row + x], ci = t1.s[row + x];
          const srcBase = ((x * N2 + ky) * N3 + kz) * D;
          for (let d = 0; d < D; d++) {
            const xr = bRe[srcBase + d], xi = bIm[srcBase + d];
            outRe[dstBase + d] += xr * cr - xi * ci;
            outIm[dstBase + d] += xr * ci + xi * cr;
          }
        }
      }
    }
  }

  return { re: outRe, im: outIm };
}

/**
 * Method 1: Compute the full commensurate k-space mode power spectrum P(k)
 * averaged over an ensemble of supercell displacement configs, plus per-element
 * power, SNR over background, and dominant k-vectors / correlation length.
 *
 * @param {Float64Array[]} configs  Array of displacement arrays [N1*N2*N3*nBasis*3]
 * @param {number[]} dim            [N1, N2, N3]
 * @param {{el:string, frac:number[]}[]} basis
 */
export function kSpaceModeSpectrum(configs, dim, basis) {
  const [N1, N2, N3] = dim;
  const nCells = N1 * N2 * N3;
  const nBasis = basis.length;
  const D = 3 * nBasis;
  const nCfg = configs.length;

  const elements = [...new Set(basis.map(b => b.el))];
  const elIdxOfSite = basis.map(b => elements.indexOf(b.el));

  const power = new Float64Array(nCells);
  const powerByEl = elements.map(() => new Float64Array(nCells));
  // Coherent (mean-squared) power: |<u(k)>|^2 vs total <|u(k)|^2>
  const meanRe = new Float64Array(nCells * D);
  const meanIm = new Float64Array(nCells * D);

  for (const u of configs) {
    const { re, im } = supercellFFT3D(u, dim, nBasis);
    for (let c = 0; c < nCells; c++) {
      const base = c * D;
      let pTot = 0;
      for (let s = 0; s < nBasis; s++) {
        const ei = elIdxOfSite[s];
        const o = base + s * 3;
        let ps = 0;
        for (let a = 0; a < 3; a++) {
          const r = re[o + a], m = im[o + a];
          ps += r * r + m * m;
          meanRe[o + a] += r;
          meanIm[o + a] += m;
        }
        pTot += ps;
        powerByEl[ei][c] += ps;
      }
      power[c] += pTot;
    }
  }

  const invCfg = 1 / nCfg;
  const coherentPower = new Float64Array(nCells);
  for (let c = 0; c < nCells; c++) {
    power[c] *= invCfg;
    for (let e = 0; e < elements.length; e++) powerByEl[e][c] *= invCfg;
    const base = c * D;
    let cp = 0;
    for (let d = 0; d < D; d++) {
      const r = meanRe[base + d] * invCfg;
      const m = meanIm[base + d] * invCfg;
      meanRe[base + d] = r;
      meanIm[base + d] = m;
      cp += r * r + m * m;
    }
    coherentPower[c] = cp;
  }

  // Background baseline = median P(k) across all k-points
  const sortedP = Array.from(power).sort((a, b) => a - b);
  const medianPower = sortedP[Math.floor(sortedP.length / 2)] || 1e-12;
  const meanPower = sortedP.reduce((a, b) => a + b, 0) / sortedP.length;

  // Build k-point list with fractional coords in (-0.5, 0.5]
  const kPoints = [];
  for (let kx = 0; kx < N1; kx++) {
    for (let ky = 0; ky < N2; ky++) {
      for (let kz = 0; kz < N3; kz++) {
        const c = (kx * N2 + ky) * N3 + kz;
        const fx = kx > N1 / 2 ? kx / N1 - 1 : kx / N1;
        const fy = ky > N2 / 2 ? ky / N2 - 1 : ky / N2;
        const fz = kz > N3 / 2 ? kz / N3 - 1 : kz / N3;
        const byEl = {};
        for (let e = 0; e < elements.length; e++) byEl[elements[e]] = powerByEl[e][c];
        kPoints.push({
          idx: c,
          grid: [kx, ky, kz],
          kFrac: [fx, fy, fz],
          power: power[c],
          coherentPower: coherentPower[c],
          incoherentPower: Math.max(0, power[c] - coherentPower[c]),
          snr: power[c] / medianPower,
          byEl,
        });
      }
    }
  }

  const topK = [...kPoints].sort((a, b) => b.power - a.power).slice(0, 15);

  // Estimate correlation length around Gamma (in unit cells) from P(Gamma) vs nearest-neighbor q-shells
  const pGamma = power[0];
  let pNN = 0, nNN = 0;
  const nnOffsets = [
    [1, 0, 0], [N1 - 1, 0, 0],
    [0, 1, 0], [0, N2 - 1, 0],
    [0, 0, 1], [0, 0, N3 - 1],
  ];
  for (const [dx, dy, dz] of nnOffsets) {
    const c = ((dx % N1) * N2 + (dy % N2)) * N3 + (dz % N3);
    pNN += power[c]; nNN++;
  }
  pNN /= nNN;
  // Lorentzian fit: P(q) = P(0) / (1 + (q ξ)^2) with q_min = 2π / N (in unit-cell units)
  const qMin = TWO_PI / ((N1 + N2 + N3) / 3);
  const ratio = pGamma > pNN && pNN > medianPower * 0.5
    ? Math.max(0, (pGamma - medianPower) / Math.max(1e-12, pNN - medianPower) - 1)
    : 0;
  const gammaCorrelationCells = ratio > 0 ? Math.sqrt(ratio) / qMin : 0;

  return {
    elements,
    medianPower,
    meanPower,
    gammaPower: pGamma,
    gammaSnr: pGamma / medianPower,
    gammaCorrelationCells,
    topK,
    kPoints,
    meanFourierRe: meanRe,
    meanFourierIm: meanIm,
  };
}

// ── Method 2: Per-Cell / Local Subgroup Projection & Continuous Symmetry ─────

/**
 * Method 2: Project every individual unit cell's displacement vector u_l onto
 * every closed subgroup H <= G0 (tracking each conjugate domain orientation).
 *
 * Returns a ranked table of subgroups with:
 *   - meanR2:        average fraction of local distortion variance explained by H
 *   - bestDomainR2:  average R^2 using each cell's best-matching conjugate domain
 *   - nullR2:        expected R^2 for isotropic random thermal noise (DoF / D)
 *   - enrichment:    meanR2 / nullR2 (values >> 1 indicate true symmetry preference)
 *   - bestEnrichment: bestDomainR2 / nullR2
 *   - fStat:         variance-per-DoF ratio (signal vs residual)
 *   - domainFractions: fraction of cells choosing each conjugate domain orientation
 *   - neighborMatchRate: fraction of nearest-neighbor cell pairs sharing the same domain
 *                        (vs 1/nVariants for random uncorrelated domains)
 *
 * @param {Float64Array[]} configs
 * @param {number[]} dim
 * @param {number[][]} A            3x3 conventional unit-cell lattice rows
 * @param {{el:string, frac:number[]}[]} basis
 * @param {object} [opts]
 */
export function localSubgroupAnalysis(configs, dim, A, basis, opts = {}) {
  const {
    parentTol = 0.25,
    subtractDrift = true,
    maxCellsPerConfig = null, // optional subsampling if needed
  } = opts;

  const [N1, N2, N3] = dim;
  const nCells = N1 * N2 * N3;
  const nBasis = basis.length;
  const D = 3 * nBasis;
  const Dint = subtractDrift ? D - 3 : D;

  const parentSG = findSpaceGroupOps(A, basis, parentTol);
  const reps = operationReps(A, basis, parentSG.ops, Math.max(0.05, parentTol / 5));
  const subgroups = enumerateSubgroups(reps);

  // Precompute DoF for each subgroup family
  const famInfo = subgroups.map(fam => {
    const dof = subgroupDoF(fam.variants[0].ops, nBasis, subtractDrift);
    return {
      ...fam,
      dof: dof.internal,
      dofTotal: dof.total,
      nullR2: Dint > 0 ? dof.internal / Dint : 0,
      nVariants: fam.variants.length,
      sumMeanR2: 0,
      sumBestR2: 0,
      domainCounts: new Float64Array(fam.variants.length),
      neighborMatches: 0,
      neighborPairs: 0,
      nSamples: 0,
    };
  });

  // Reusable buffers
  const uCell = new Float64Array(D);
  const tmp = new Float64Array(D);
  // Pre-allocate per-operation transformed vectors: gU[opIdx] = Γ(g) u_l
  const gU = Array.from({ length: reps.length }, () => new Float64Array(D));
  const proj = new Float64Array(D);

  // Track best domain per cell in each config for nearest-neighbor domain correlation
  const cellDomain = famInfo.map(() => new Int32Array(nCells));

  const step = maxCellsPerConfig && maxCellsPerConfig < nCells
    ? Math.max(1, Math.floor(nCells / maxCellsPerConfig))
    : 1;

  for (const cfg of configs) {
    for (let c = 0; c < nCells; c += step) {
      const base = c * D;
      if (subtractDrift) {
        removeCellDrift(cfg.subarray(base, base + D), nBasis, uCell);
      } else {
        uCell.set(cfg.subarray(base, base + D));
      }

      let norm2 = 0;
      for (let d = 0; d < D; d++) norm2 += uCell[d] * uCell[d];
      if (norm2 < 1e-18) continue;

      // Compute Γ(g) u_l ONCE for all g ∈ G0 (huge speedup: shared across all subgroups!)
      for (let gi = 0; gi < reps.length; gi++) {
        applyGammaCell(reps[gi], uCell, nBasis, gU[gi]);
      }

      // Evaluate every subgroup family & conjugate variant
      for (let fi = 0; fi < famInfo.length; fi++) {
        const fam = famInfo[fi];
        let sumVarR2 = 0;
        let bestR2 = -1;
        let bestV = 0;

        for (let vi = 0; vi < fam.variants.length; vi++) {
          const opIdx = fam.variants[vi].opIndices;
          const invH = 1 / opIdx.length;
          proj.fill(0);
          for (let k = 0; k < opIdx.length; k++) {
            const gu = gU[opIdx[k]];
            for (let d = 0; d < D; d++) proj[d] += gu[d];
          }
          let pNorm2 = 0;
          for (let d = 0; d < D; d++) {
            const v = proj[d] * invH;
            pNorm2 += v * v;
          }
          const r2 = Math.min(1, pNorm2 / norm2);
          sumVarR2 += r2;
          if (r2 > bestR2) { bestR2 = r2; bestV = vi; }
        }

        fam.sumMeanR2 += sumVarR2 / fam.variants.length;
        fam.sumBestR2 += bestR2;
        fam.domainCounts[bestV] += 1;
        fam.nSamples += 1;
        cellDomain[fi][c] = bestV;
      }
    }

    // Nearest-neighbor domain correlation (only if full grid was evaluated)
    if (step === 1) {
      for (let fi = 0; fi < famInfo.length; fi++) {
        const fam = famInfo[fi];
        if (fam.nVariants <= 1) continue;
        const dom = cellDomain[fi];
        for (let x = 0; x < N1; x++) {
          const xp = (x + 1) % N1;
          for (let y = 0; y < N2; y++) {
            const yp = (y + 1) % N2;
            for (let z = 0; z < N3; z++) {
              const zp = (z + 1) % N3;
              const c = (x * N2 + y) * N3 + z;
              const d0 = dom[c];
              if (dom[(xp * N2 + y) * N3 + z] === d0) fam.neighborMatches++;
              if (dom[(x * N2 + yp) * N3 + z] === d0) fam.neighborMatches++;
              if (dom[(x * N2 + y) * N3 + zp] === d0) fam.neighborMatches++;
              fam.neighborPairs += 3;
            }
          }
        }
      }
    }
  }

  // Identify the parent (full G0) family index so we can also compute pure
  // symmetry-breaking variance orthogonal to G0: (P_H - P_G0) u_l.
  const parentFamIdx = famInfo.findIndex(f => f.order === reps.length);
  const parentDoF = parentFamIdx >= 0 ? famInfo[parentFamIdx].dof : 0;

  // Finalize statistics
  const parentMeanR2 = parentFamIdx >= 0
    ? famInfo[parentFamIdx].sumBestR2 / Math.max(1, famInfo[parentFamIdx].nSamples)
    : 0;

  const results = famInfo.map((fam, fi) => {
    const n = Math.max(1, fam.nSamples);
    const meanR2 = fam.sumMeanR2 / n;
    const bestDomainR2 = fam.sumBestR2 / n;
    const enrichment = fam.nullR2 > 0 ? meanR2 / fam.nullR2 : 0;
    const bestEnrichment = fam.nullR2 > 0 ? bestDomainR2 / fam.nullR2 : 0;
    const remDoF = Math.max(1, Dint - fam.dof);
    const fStat = fam.dof > 0 && fam.dof < Dint
      ? (bestDomainR2 / fam.dof) / Math.max(1e-12, (1 - bestDomainR2) / remDoF)
      : (fam.dof === Dint ? 1 : 0);

    // Pure symmetry-breaking component orthogonal to parent G0:
    // ΔR² = R²(H) - R²(G0), ΔDoF = DoF(H) - DoF(G0).
    const breakingDoF = Math.max(0, fam.dof - parentDoF);
    const breakingR2 = Math.max(0, bestDomainR2 - parentMeanR2);
    const breakingFStat = breakingDoF > 0 && fam.dof < Dint
      ? (breakingR2 / breakingDoF) / Math.max(1e-12, (1 - bestDomainR2) / remDoF)
      : 0;

    const domainFractions = Array.from(fam.domainCounts).map(x => x / n);
    const neighborMatchRate = fam.neighborPairs > 0
      ? fam.neighborMatches / fam.neighborPairs
      : null;
    const randomMatchRate = 1 / fam.nVariants;

    return {
      spaceGroup: fam.spaceGroup,
      spaceGroupNumber: fam.spaceGroupNumber,
      pointGroup: fam.pointGroup,
      centering: fam.centering,
      order: fam.order,
      nVariants: fam.nVariants,
      dof: fam.dof,
      nullR2: fam.nullR2,
      meanR2,
      bestDomainR2,
      csmResidual: 1 - bestDomainR2,
      enrichment,
      bestEnrichment,
      fStat,
      breakingDoF,
      breakingR2,
      breakingFStat,
      domainFractions,
      neighborMatchRate,
      neighborExcess: neighborMatchRate !== null ? neighborMatchRate - randomMatchRate : null,
      domainGrid: step === 1 && fam.nVariants > 1 ? Array.from(cellDomain[fi]) : null,
    };
  });

  // Sort by F-statistic (highest signal-per-DoF symmetry first)
  results.sort((a, b) => b.fStat - a.fStat);

  return {
    parentSpaceGroup: parentSG.spaceGroup,
    parentOrder: parentSG.order,
    internalDoF: Dint,
    nConfigs: configs.length,
    subgroups: results,
  };
}

// ── Method 3: Statistical Symmetry Testing on Covariance C(k) vs Mean <u> ────

/**
 * Method 3: Test every parent space-group operation g ∈ G0 directly on:
 *   (a) The Gamma-point displacement covariance C(0) = <u(0) u(0)†> (or local cell covariance)
 *   (b) The folded mean displacement <u> (the traditional FINDSYM input, for comparison)
 *
 * Returns per-operation breaking scores ε_cov(g) and ε_mean(g) in [0, 2],
 * plus the implied statistical subgroup at a given threshold.
 */
export function statisticalSymmetryTest(configs, dim, A, basis, opts = {}) {
  const { parentTol = 0.25, subtractDrift = true, covThreshold = 0.15 } = opts;
  const [N1, N2, N3] = dim;
  const nCells = N1 * N2 * N3;
  const nBasis = basis.length;
  const D = 3 * nBasis;

  const parentSG = findSpaceGroupOps(A, basis, parentTol);
  const reps = operationReps(A, basis, parentSG.ops, Math.max(0.05, parentTol / 5));

  // Compute folded mean <u> and local single-cell covariance C = <u_l u_l^T>
  const meanU = new Float64Array(D);
  const cov = new Float64Array(D * D);
  const uCell = new Float64Array(D);
  let nTotal = 0;

  for (const cfg of configs) {
    for (let c = 0; c < nCells; c++) {
      const base = c * D;
      if (subtractDrift) {
        removeCellDrift(cfg.subarray(base, base + D), nBasis, uCell);
      } else {
        uCell.set(cfg.subarray(base, base + D));
      }
      for (let i = 0; i < D; i++) {
        const ui = uCell[i];
        meanU[i] += ui;
        const row = i * D;
        for (let j = i; j < D; j++) cov[row + j] += ui * uCell[j];
      }
      nTotal++;
    }
  }

  for (let i = 0; i < D; i++) {
    meanU[i] /= nTotal;
    for (let j = i; j < D; j++) {
      const v = cov[i * D + j] / nTotal;
      cov[i * D + j] = v;
      cov[j * D + i] = v;
    }
  }

  let meanNorm = 0;
  for (let i = 0; i < D; i++) meanNorm += meanU[i] * meanU[i];
  meanNorm = Math.sqrt(meanNorm);

  let covNorm = 0;
  for (let i = 0; i < D * D; i++) covNorm += cov[i] * cov[i];
  covNorm = Math.sqrt(covNorm);

  // Test each operation g ∈ G0
  const gMean = new Float64Array(D);
  const opScores = [];
  const preservedOps = [];

  for (const g of reps) {
    // (a) Mean residual: ||Γ(g) <u> - <u>|| / ||<u>||
    applyGammaCell(g, meanU, nBasis, gMean);
    let dMean2 = 0;
    for (let i = 0; i < D; i++) {
      const d = gMean[i] - meanU[i];
      dMean2 += d * d;
    }
    const epsMean = meanNorm > 1e-12 ? Math.sqrt(dMean2) / meanNorm : 0;

    // (b) Covariance residual: ||Γ(g) C Γ(g)† - C||_F / ||C||_F
    // Since Γ(g) is block-monomial (perm + 3x3 Rcart), compute (Γ C Γᵀ) directly
    let dCov2 = 0;
    const { perm, Rcart } = g;
    for (let s1 = 0; s1 < nBasis; s1++) {
      const p1 = perm[s1] * 3, r1 = s1 * 3;
      for (let s2 = 0; s2 < nBasis; s2++) {
        const p2 = perm[s2] * 3, r2 = s2 * 3;
        for (let a = 0; a < 3; a++) {
          for (let b = 0; b < 3; b++) {
            let val = 0;
            for (let i = 0; i < 3; i++) {
              const wa = Rcart[a][i];
              const row = (r1 + i) * D + r2;
              for (let j = 0; j < 3; j++) {
                val += wa * cov[row + j] * Rcart[b][j];
              }
            }
            const diff = val - cov[(p1 + a) * D + (p2 + b)];
            dCov2 += diff * diff;
          }
        }
      }
    }
    const epsCov = covNorm > 1e-12 ? Math.sqrt(dCov2) / covNorm : 0;
    opScores.push({ R: g.R, t: g.t, epsCov, epsMean });
    if (epsCov <= covThreshold) preservedOps.push(g);
  }

  const statSG = classifyOperations(preservedOps, 0.02);
  const maxEpsCov = Math.max(...opScores.map(o => o.epsCov));
  const rmsMeanDisp = Math.sqrt(meanNorm * meanNorm / nBasis);
  const rmsLocalDisp = Math.sqrt(
    Array.from({ length: D }, (_, i) => cov[i * D + i]).reduce((a, b) => a + b, 0) / nBasis
  );

  return {
    parentSpaceGroup: parentSG.spaceGroup,
    statisticalSpaceGroup: statSG.spaceGroup,
    statisticalOrder: preservedOps.length,
    parentOrder: reps.length,
    maxEpsCov,
    rmsMeanDisp,
    rmsLocalDisp,
    meanToLocalRatio: rmsLocalDisp > 1e-12 ? rmsMeanDisp / rmsLocalDisp : 0,
    opScores,
  };
}

// ── Method 4: Mode-Filtered Minimal Supercell → Discrete Space Group ─────────

function gcd(a, b) { return b === 0 ? Math.abs(a) : gcd(b, a % b); }
function lcm(a, b) { return (a && b) ? Math.abs(a * b) / gcd(a, b) : Math.max(a, b); }

function inv3(m) {
  const d = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
          - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
          + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const id = 1 / d;
  return [
    [(m[1][1] * m[2][2] - m[1][2] * m[2][1]) * id, (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * id, (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * id],
    [(m[1][2] * m[2][0] - m[1][0] * m[2][2]) * id, (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * id, (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * id],
    [(m[1][0] * m[2][1] - m[1][1] * m[2][0]) * id, (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * id, (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * id],
  ];
}

/**
 * Method 4: Given a k-space spectrum (from kSpaceModeSpectrum), select all
 * coherent k-modes with SNR >= snrThreshold (and coherent fraction >= minCoherentFrac),
 * determine the minimal commensurate supercell [M1, M2, M3], reconstruct the
 * denoised atomic positions in that minimal supercell, and run symmetryLadder!
 */
export function filteredMinimalSupercell(spectrum, dim, A, basis, opts = {}) {
  const {
    snrThreshold = 3.0,
    minCoherentFrac = 0.25,
    maxMiniDim = 2,
    tolMax = 0.5,
  } = opts;
  const [N1, N2, N3] = dim;
  const nBasis = basis.length;
  const D = 3 * nBasis;
  const nCells = N1 * N2 * N3;
  const nCfgScale = 1 / Math.sqrt(nCells);

  // Select significant coherent k-points
  const activeK = spectrum.kPoints.filter(kp => {
    const cohFrac = kp.power > 1e-18 ? kp.coherentPower / kp.power : 0;
    return kp.snr >= snrThreshold && cohFrac >= minCoherentFrac;
  });

  // Determine minimal supercell period M = [M1, M2, M3] commensurate with activeK
  let M1 = 1, M2 = 1, M3 = 1;
  for (const kp of activeK) {
    const [kx, ky, kz] = kp.grid;
    const p1 = kx === 0 ? 1 : N1 / gcd(kx, N1);
    const p2 = ky === 0 ? 1 : N2 / gcd(ky, N2);
    const p3 = kz === 0 ? 1 : N3 / gcd(kz, N3);
    if (p1 <= maxMiniDim) M1 = lcm(M1, p1);
    if (p2 <= maxMiniDim) M2 = lcm(M2, p2);
    if (p3 <= maxMiniDim) M3 = lcm(M3, p3);
  }
  const miniDim = [M1, M2, M3];
  const keptK = activeK.filter(kp => {
    const [kx, ky, kz] = kp.grid;
    return (kx * M1) % N1 === 0 && (ky * M2) % N2 === 0 && (kz * M3) % N3 === 0;
  });

  // Reconstruct denoised displacements in the minimal supercell [M1, M2, M3]
  const Amini = [
    A[0].map(v => v * M1),
    A[1].map(v => v * M2),
    A[2].map(v => v * M3),
  ];
  const AminiInv = inv3(Amini);
  const miniBasis = [];

  for (let mx = 0; mx < M1; mx++) {
    for (let my = 0; my < M2; my++) {
      for (let mz = 0; mz < M3; mz++) {
        for (let s = 0; s < nBasis; s++) {
          let ux = 0, uy = 0, uz = 0;
          for (const kp of keptK) {
            const ang = TWO_PI * (kp.kFrac[0] * mx + kp.kFrac[1] * my + kp.kFrac[2] * mz);
            const cph = Math.cos(ang), sph = Math.sin(ang);
            const base = kp.idx * D + s * 3;
            // meanFourierRe/Im is summed over configs; normalize by nCells^(1/2) * nCfg
            // Note: spectrum stores summed meanFourierRe, so scale by 1/(sqrt(nCells)*nCfg)
            // We reconstruct from coherent amplitude directly:
            const rx = spectrum.meanFourierRe[base], ix = spectrum.meanFourierIm[base];
            const ry = spectrum.meanFourierRe[base + 1], iy = spectrum.meanFourierIm[base + 1];
            const rz = spectrum.meanFourierRe[base + 2], iz = spectrum.meanFourierIm[base + 2];
            ux += (rx * cph - ix * sph) * nCfgScale;
            uy += (ry * cph - iy * sph) * nCfgScale;
            uz += (rz * cph - iz * sph) * nCfgScale;
          }
          const baseFrac = [
            (mx + basis[s].frac[0]) / M1,
            (my + basis[s].frac[1]) / M2,
            (mz + basis[s].frac[2]) / M3,
          ];
          // Convert cartesian displacement u to fractional in Amini: df = u · Amini^{-1}
          const df = [
            ux * AminiInv[0][0] + uy * AminiInv[1][0] + uz * AminiInv[2][0],
            ux * AminiInv[0][1] + uy * AminiInv[1][1] + uz * AminiInv[2][1],
            ux * AminiInv[0][2] + uy * AminiInv[1][2] + uz * AminiInv[2][2],
          ];
          miniBasis.push({
            el: basis[s].el,
            frac: [wrap01(baseFrac[0] + df[0]), wrap01(baseFrac[1] + df[1]), wrap01(baseFrac[2] + df[2])],
          });
        }
      }
    }
  }

  const ladder = symmetryLadder(Amini, miniBasis, tolMax);

  return {
    miniDim,
    nActiveK: keptK.length,
    activeK: keptK.map(k => ({ kFrac: k.kFrac, snr: k.snr, power: k.power, coherentPower: k.coherentPower })),
    Amini,
    miniBasis,
    ladder,
  };
}

// ── Method 5: Domain-Aligned Averaging & Local CIF Export ────────────────────

function projectSubgroupVec(opIndices, reps, u, nBasis, out, tmp) {
  out.fill(0);
  const invH = 1 / opIndices.length;
  for (let k = 0; k < opIndices.length; k++) {
    applyGammaCell(reps[opIndices[k]], u, nBasis, tmp);
    for (let d = 0; d < out.length; d++) out[d] += tmp[d];
  }
  for (let d = 0; d < out.length; d++) out[d] *= invH;
}

/**
 * Domain-aligned averaging ("symmetry unfolding") for a chosen subgroup H <= G0:
 *   1. For every unit cell l in every configuration, find its winning twin domain H_{v*}.
 *   2. Rotate cell l onto the canonical domain H_0 via c_{v*} ∈ G0 (c_{v*} H_{v*} c_{v*}^{-1} = H_0).
 *   3. Resolve normalizer cosets n ∈ N(H_0)/H_0 (e.g. ±180° polar / antiphase domains)
 *      by aligning the symmetry-breaking component (P_{H_0} - P_{G_0}) w_l to a common sign.
 *   4. Average all aligned cells constructively — recovering the local distorted unit cell
 *      that standard unit-cell folding cancels to zero!
 *
 * @param {Float64Array[]} configs
 * @param {number[]} dim
 * @param {number[][]} A
 * @param {{rn?:number, el:string, frac:number[]}[]} basis
 * @param {string|{spaceGroup:string, order?:number, dof?:number}} subgroupSpec
 * @param {object} [opts]
 */
export function domainAlignedStructure(configs, dim, A, basis, subgroupSpec, opts = {}) {
  const { parentTol = 0.25, subtractDrift = true } = opts;
  const [N1, N2, N3] = dim;
  const nCells = N1 * N2 * N3;
  const nBasis = basis.length;
  const D = 3 * nBasis;

  const parentSG = findSpaceGroupOps(A, basis, parentTol);
  const reps = operationReps(A, basis, parentSG.ops, Math.max(0.05, parentTol / 5));
  const subgroups = enumerateSubgroups(reps);

  // Locate target subgroup family
  const targetName = typeof subgroupSpec === 'string' ? subgroupSpec : subgroupSpec?.spaceGroup;
  const targetOrder = typeof subgroupSpec === 'object' ? subgroupSpec?.order : null;
  const fam = subgroups.find(f =>
    f.spaceGroup === targetName && (targetOrder == null || f.order === targetOrder)
  ) || subgroups.find(f => f.spaceGroup === targetName);
  if (!fam) throw new Error(`Subgroup "${targetName}" not found in parent ${parentSG.spaceGroup}.`);

  const allOpIndices = Array.from({ length: reps.length }, (_, i) => i);
  const h0Ops = fam.variants[0].opIndices;
  const normCosets = fam.normCosetReps && fam.normCosetReps.length ? fam.normCosetReps : [0];

  const uCell = new Float64Array(D);
  const gU = Array.from({ length: reps.length }, () => new Float64Array(D));
  const proj = new Float64Array(D);
  const alignedCell = new Float64Array(D);
  const projH0 = new Float64Array(D);
  const projG0 = new Float64Array(D);
  const brkVec = new Float64Array(D);
  const candBrk = new Float64Array(D);
  const tmp = new Float64Array(D);

  // Pass 1: find the strongest symmetry-breaking cell as the canonical sign/coset reference
  const refBrk = new Float64Array(D);
  let bestBrkNorm2 = -1;
  const cellWinners = []; // store { cfgIdx, c, bestV, margin } for Pass 2

  for (let ci = 0; ci < configs.length; ci++) {
    const cfg = configs[ci];
    for (let c = 0; c < nCells; c++) {
      const base = c * D;
      if (subtractDrift) removeCellDrift(cfg.subarray(base, base + D), nBasis, uCell);
      else uCell.set(cfg.subarray(base, base + D));

      let norm2 = 0;
      for (let d = 0; d < D; d++) norm2 += uCell[d] * uCell[d];
      if (norm2 < 1e-18) continue;

      for (let gi = 0; gi < reps.length; gi++) applyGammaCell(reps[gi], uCell, nBasis, gU[gi]);

      let bestR2 = -1, secondR2 = -1, bestV = 0;
      for (let vi = 0; vi < fam.variants.length; vi++) {
        const opIdx = fam.variants[vi].opIndices;
        const invH = 1 / opIdx.length;
        proj.fill(0);
        for (let k = 0; k < opIdx.length; k++) {
          const gu = gU[opIdx[k]];
          for (let d = 0; d < D; d++) proj[d] += gu[d];
        }
        let pNorm2 = 0;
        for (let d = 0; d < D; d++) { const v = proj[d] * invH; pNorm2 += v * v; }
        const r2 = pNorm2 / norm2;
        if (r2 > bestR2) { secondR2 = bestR2; bestR2 = r2; bestV = vi; }
        else if (r2 > secondR2) { secondR2 = r2; }
      }

      const margin = fam.variants.length > 1 ? Math.max(0, bestR2 - Math.max(0, secondR2)) : bestR2;
      cellWinners.push({ ci, c, bestV, bestR2, margin });

      // Check symmetry-breaking strength for canonical coset reference
      const alignOp = fam.variants[bestV].alignOpIdx;
      const w = gU[alignOp];
      projectSubgroupVec(h0Ops, reps, w, nBasis, projH0, tmp);
      projectSubgroupVec(allOpIndices, reps, w, nBasis, projG0, tmp);
      let b2 = 0;
      for (let d = 0; d < D; d++) {
        const diff = projH0[d] - projG0[d];
        brkVec[d] = diff;
        b2 += diff * diff;
      }
      if (b2 > bestBrkNorm2) {
        bestBrkNorm2 = b2;
        refBrk.set(brkVec);
      }
    }
  }

  // Pass 2: align every cell (twin variant + normalizer polarity coset) and accumulate
  const sumAligned = new Float64Array(D);
  const sumFolded = new Float64Array(D);
  const cosetCounts = new Int32Array(normCosets.length);
  let nSamples = 0;

  for (const item of cellWinners) {
    const cfg = configs[item.ci];
    const base = item.c * D;
    if (subtractDrift) removeCellDrift(cfg.subarray(base, base + D), nBasis, uCell);
    else uCell.set(cfg.subarray(base, base + D));

    for (let d = 0; d < D; d++) sumFolded[d] += uCell[d];

    // Rotate winning variant onto canonical H_0
    applyGammaCell(reps[fam.variants[item.bestV].alignOpIdx], uCell, nBasis, alignedCell);

    // Resolve normalizer coset (e.g. +[111] vs -[111] polarity)
    let bestCosetIdx = 0;
    if (normCosets.length > 1 && bestBrkNorm2 > 1e-18) {
      projectSubgroupVec(h0Ops, reps, alignedCell, nBasis, projH0, tmp);
      projectSubgroupVec(allOpIndices, reps, alignedCell, nBasis, projG0, tmp);
      for (let d = 0; d < D; d++) brkVec[d] = projH0[d] - projG0[d];

      let bestDot = -Infinity;
      for (let ni = 0; ni < normCosets.length; ni++) {
        applyGammaCell(reps[normCosets[ni]], brkVec, nBasis, candBrk);
        let dot = 0;
        for (let d = 0; d < D; d++) dot += candBrk[d] * refBrk[d];
        if (dot > bestDot) { bestDot = dot; bestCosetIdx = ni; }
      }
      applyGammaCell(reps[normCosets[bestCosetIdx]], alignedCell, nBasis, tmp);
      alignedCell.set(tmp);
    }
    cosetCounts[bestCosetIdx]++;
    for (let d = 0; d < D; d++) sumAligned[d] += alignedCell[d];
    nSamples++;
  }

  const invN = 1 / Math.max(1, nSamples);
  const meanAligned = new Float64Array(D);
  const meanFolded = new Float64Array(D);
  for (let d = 0; d < D; d++) {
    meanAligned[d] = sumAligned[d] * invN;
    meanFolded[d] = sumFolded[d] * invN;
  }

  // Exact H_0-symmetrized projection of the aligned mean
  const symAligned = new Float64Array(D);
  projectSubgroupVec(h0Ops, reps, meanAligned, nBasis, symAligned, tmp);

  // Idealize the parent basis onto exact G_0 Wyckoff positions before adding
  // the H_0-projected displacement, so symmetrizedBasis has exact H_0 symmetry
  // even when the input refBasis has single-config RMC statistical noise.
  const idealParentBasis = basis.map((b, s) => ({ rn: b.rn, el: b.el, frac: [...b.frac] }));
  {
    const sumDf = Array.from({ length: nBasis }, () => [0, 0, 0]);
    for (let gi = 0; gi < reps.length; gi++) {
      const { R, t, perm } = reps[gi];
      for (let j = 0; j < nBasis; j++) {
        const s = perm[j];
        const xj = basis[j].frac;
        const xs = basis[s].frac;
        const gx0 = R[0][0] * xj[0] + R[0][1] * xj[1] + R[0][2] * xj[2] + t[0];
        const gx1 = R[1][0] * xj[0] + R[1][1] * xj[1] + R[1][2] * xj[2] + t[1];
        const gx2 = R[2][0] * xj[0] + R[2][1] * xj[1] + R[2][2] * xj[2] + t[2];
        sumDf[s][0] += wrapHalf(gx0 - xs[0]);
        sumDf[s][1] += wrapHalf(gx1 - xs[1]);
        sumDf[s][2] += wrapHalf(gx2 - xs[2]);
      }
    }
    const invG = 1 / Math.max(1, reps.length);
    for (let s = 0; s < nBasis; s++) {
      idealParentBasis[s].frac = [
        wrap01(basis[s].frac[0] + sumDf[s][0] * invG),
        wrap01(basis[s].frac[1] + sumDf[s][1] * invG),
        wrap01(basis[s].frac[2] + sumDf[s][2] * invG),
      ];
    }
  }

  // Build distorted fractional coordinates (A rows = lattice vectors -> df = u · A^-1)
  const Ainv = inv3(A);
  const toBasis = (uVec, baseSites) => baseSites.map((b, s) => {
    const ux = uVec[s * 3], uy = uVec[s * 3 + 1], uz = uVec[s * 3 + 2];
    const df = [
      ux * Ainv[0][0] + uy * Ainv[1][0] + uz * Ainv[2][0],
      ux * Ainv[0][1] + uy * Ainv[1][1] + uz * Ainv[2][1],
      ux * Ainv[0][2] + uy * Ainv[1][2] + uz * Ainv[2][2],
    ];
    return {
      rn: b.rn,
      el: b.el,
      frac: [wrap01(b.frac[0] + df[0]), wrap01(b.frac[1] + df[1]), wrap01(b.frac[2] + df[2])],
      dispCart: [ux, uy, uz],
      dispNorm: Math.hypot(ux, uy, uz),
    };
  });

  const alignedBasis = toBasis(meanAligned, basis);
  const symmetrizedBasis = toBasis(symAligned, idealParentBasis);
  const foldedBasis = toBasis(meanFolded, basis);

  // Per-element RMS displacements (aligned vs folded)
  const elements = [...new Set(basis.map(b => b.el))];
  const elementStats = elements.map(el => {
    const idxs = basis.map((b, i) => (b.el === el ? i : -1)).filter(i => i >= 0);
    const rmsOf = (arr) => Math.sqrt(idxs.reduce((s, i) => s + arr[i].dispNorm ** 2, 0) / Math.max(1, idxs.length));
    const maxOf = (arr) => Math.max(...idxs.map(i => arr[i].dispNorm));
    return {
      el,
      count: idxs.length,
      alignedRms: rmsOf(alignedBasis),
      symRms: rmsOf(symmetrizedBasis),
      foldedRms: rmsOf(foldedBasis),
      maxShift: maxOf(symmetrizedBasis),
    };
  });

  const rmsVec = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0) / Math.max(1, nBasis));
  const alignedRms = rmsVec(meanAligned);
  const symRms = rmsVec(symAligned);
  const foldedRms = rmsVec(meanFolded);
  const maxShift = Math.max(...symmetrizedBasis.map(s => s.dispNorm));

  // Verify the recovered space group on the symmetrized aligned unit cell at tight tolerance
  const verifiedSG = findSpaceGroupOps(A, symmetrizedBasis, 1e-3);

  return {
    targetSpaceGroup: fam.spaceGroup,
    targetSpaceGroupNumber: fam.spaceGroupNumber,
    targetPointGroup: fam.pointGroup,
    parentSpaceGroup: parentSG.spaceGroup,
    verifiedSpaceGroup: verifiedSG.spaceGroup,
    verifiedSpaceGroupNumber: verifiedSG.spaceGroupNumber,
    verifiedOrder: verifiedSG.order,
    nVariants: fam.variants.length,
    nNormalizerCosets: normCosets.length,
    cosetFractions: Array.from(cosetCounts).map(x => x / Math.max(1, nSamples)),
    nSamples,
    A,
    alignedRms,
    symRms,
    foldedRms,
    maxShift,
    recoveryGain: foldedRms > 1e-9 ? symRms / foldedRms : Infinity,
    elementStats,
    alignedBasis,
    symmetrizedBasis,
    foldedBasis,
  };
}

/**
 * Export a domain-aligned local structure as a standard CIF 1.1 file string
 * (compatible with VESTA, FINDSYM, GSAS-II, FullProf, pymatgen, ASE).
 */
export function generateAlignedCIF(aligned, opts = {}) {
  const { symmetrized = true, title = 'domain_aligned_local_structure' } = opts;
  const A = aligned.A;
  const norm = (v) => Math.hypot(v[0], v[1], v[2]);
  const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  const a = norm(A[0]), b = norm(A[1]), c = norm(A[2]);
  const alpha = Math.acos(Math.max(-1, Math.min(1, dot(A[1], A[2]) / (b * c)))) * 180 / Math.PI;
  const beta  = Math.acos(Math.max(-1, Math.min(1, dot(A[0], A[2]) / (a * c)))) * 180 / Math.PI;
  const gamma = Math.acos(Math.max(-1, Math.min(1, dot(A[0], A[1]) / (a * b)))) * 180 / Math.PI;

  const sites = symmetrized ? aligned.symmetrizedBasis : aligned.alignedBasis;
  const sgLabel = aligned.verifiedSpaceGroup || aligned.targetSpaceGroup || 'P1';

  const lines = [
    `data_${title.replace(/[^A-Za-z0-9_]/g, '_')}`,
    `_audit_creation_method  'rmc-phonon-dynamics domain-aligned supercell averaging'`,
    `_publ_section_comment`,
    `;`,
    `  Domain-aligned local structure from RMC supercell (${aligned.nSamples} unit cells).`,
    `  Parent space group: ${aligned.parentSpaceGroup}`,
    `  Target local subgroup: ${aligned.targetSpaceGroup} (${aligned.nVariants} twin variants x ${aligned.nNormalizerCosets} polarity cosets)`,
    `  Verified space group (tol=0.001 A): ${sgLabel}`,
    `  RMS local shift: ${aligned.symRms.toFixed(5)} A (vs standard folded mean: ${aligned.foldedRms.toFixed(5)} A)`,
    `;`,
    ``,
    `_cell_length_a                     ${a.toFixed(6)}`,
    `_cell_length_b                     ${b.toFixed(6)}`,
    `_cell_length_c                     ${c.toFixed(6)}`,
    `_cell_angle_alpha                  ${alpha.toFixed(4)}`,
    `_cell_angle_beta                   ${beta.toFixed(4)}`,
    `_cell_angle_gamma                  ${gamma.toFixed(4)}`,
    ``,
    `_symmetry_space_group_name_H-M    'P 1'`,
    `_symmetry_Int_Tables_number         1`,
    `loop_`,
    `_symmetry_equiv_pos_as_xyz`,
    `'x, y, z'`,
    ``,
    `loop_`,
    `_atom_site_label`,
    `_atom_site_type_symbol`,
    `_atom_site_occupancy`,
    `_atom_site_fract_x`,
    `_atom_site_fract_y`,
    `_atom_site_fract_z`,
  ];

  sites.forEach((s, i) => {
    const label = `${s.el}${s.rn ?? (i + 1)}`;
    lines.push(
      `  ${label.padEnd(8)} ${s.el.padEnd(4)} 1.0000  ${s.frac[0].toFixed(6).padStart(10)}  ${s.frac[1].toFixed(6).padStart(10)}  ${s.frac[2].toFixed(6).padStart(10)}`
    );
  });

  return lines.join('\n') + '\n';
}

