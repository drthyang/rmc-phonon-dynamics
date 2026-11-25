// web/src/math/ensemble_symmetry.js
//
// Streaming multi-configuration supercell symmetry & domain analysis architecture.
// Scales to 1..500+ RMC configurations in O(1) memory per config while adding:
//   1. Exact Monte Carlo isotropic-noise null calibration (eliminates multi-variant
//      selection bias in ΔF when comparing subgroups with different variant counts V).
//   2. Spatial nearest-neighbor cross-validated projection ΔR²_CV / ΔF_CV (zero
//      within-cell selection bias; directly tests whether neighbors share a domain).
//   3. Radial domain autocorrelation C_dom(r), fitted correlation length ξ (cells & Å),
//      and 3D periodic BFS connected-component domain cluster sizes.
//   4. Ensemble uncertainty across M configs (mean ± SE, z-score, rank-1 win rate %,
//      and convergence history vs. number of configs).
//   5. Streaming domain-aligned structures (mean ± SE atomic shifts) precomputed for
//      ALL symmetry-breaking subgroups in a single pass so UI switching is instant.

import { findSpaceGroupOps } from './symmetry.js';
import { operationReps } from './symmetrize.js';
import {
  enumerateSubgroups,
  subgroupDoF,
  supercellFFT3D,
  filteredMinimalSupercell,
  applyGammaCell,
  removeCellDrift,
} from './supercell_symmetry.js';

const TWO_PI = 2 * Math.PI;
const wrap01 = (x) => ((x % 1) + 1) % 1;
const wrapHalf = (x) => x - Math.round(x);

function inv3(A) {
  const [[a, b, c], [d, e, f], [g, h, i]] = A;
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  const inv = 1 / det;
  return [
    [(e * i - f * h) * inv, (c * h - b * i) * inv, (b * f - c * e) * inv],
    [(f * g - d * i) * inv, (a * i - c * g) * inv, (c * d - a * f) * inv],
    [(d * h - e * g) * inv, (b * g - a * h) * inv, (a * e - b * d) * inv],
  ];
}

// Deterministic fast PRNG (Mulberry32) for reproducible null-baseline calibration
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeGaussian(rng) {
  let spare = null;
  return () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    let u = 0, v = 0;
    while (u <= 1e-12) u = rng();
    while (v <= 1e-12) v = rng();
    const mag = Math.sqrt(-2 * Math.log(u));
    spare = mag * Math.sin(TWO_PI * v);
    return mag * Math.cos(TWO_PI * v);
  };
}

function projectOps(opIndices, reps, gU, D, out) {
  out.fill(0);
  const invH = 1 / opIndices.length;
  for (let k = 0; k < opIndices.length; k++) {
    const gu = gU[opIndices[k]];
    for (let d = 0; d < D; d++) out[d] += gu[d];
  }
  for (let d = 0; d < D; d++) out[d] *= invH;
}

function idealizeBasis(basis, reps) {
  const nBasis = basis.length;
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
  return basis.map((b, s) => ({
    rn: b.rn,
    el: b.el,
    frac: [
      wrap01(b.frac[0] + sumDf[s][0] * invG),
      wrap01(b.frac[1] + sumDf[s][1] * invG),
      wrap01(b.frac[2] + sumDf[s][2] * invG),
    ],
  }));
}

/**
 * 3D periodic BFS connected-component stats for a domain grid Int32Array(N1*N2*N3).
 */
function domainClusterStats(domGrid, N1, N2, N3) {
  const nCells = N1 * N2 * N3;
  const visited = new Uint8Array(nCells);
  const queue = new Int32Array(nCells);
  let nClusters = 0;
  let sumSize = 0;
  let sumSize2 = 0;
  let maxCluster = 0;

  for (let start = 0; start < nCells; start++) {
    if (visited[start]) continue;
    const d0 = domGrid[start];
    visited[start] = 1;
    let head = 0, tail = 0;
    queue[tail++] = start;
    while (head < tail) {
      const cur = queue[head++];
      const z = cur % N3;
      const rem = (cur - z) / N3;
      const y = rem % N2;
      const x = (rem - y) / N2;

      const nbs = [
        (((x + 1) % N1) * N2 + y) * N3 + z,
        (((x + N1 - 1) % N1) * N2 + y) * N3 + z,
        (x * N2 + ((y + 1) % N2)) * N3 + z,
        (x * N2 + ((y + N2 - 1) % N2)) * N3 + z,
        (x * N2 + y) * N3 + ((z + 1) % N3),
        (x * N2 + y) * N3 + ((z + N3 - 1) % N3),
      ];
      for (let k = 0; k < 6; k++) {
        const nb = nbs[k];
        if (!visited[nb] && domGrid[nb] === d0) {
          visited[nb] = 1;
          queue[tail++] = nb;
        }
      }
    }
    const sz = tail;
    nClusters++;
    sumSize += sz;
    sumSize2 += sz * sz;
    if (sz > maxCluster) maxCluster = sz;
  }

  return {
    nClusters,
    meanClusterCells: sumSize / Math.max(1, nClusters),
    weightMeanClusterCells: sumSize2 / Math.max(1, sumSize),
    maxClusterCells: maxCluster,
  };
}

/**
 * Fit exponential domain correlation length ξ (in unit cells) from radial C_dom(r) for r=1..Rmax.
 */
function fitCorrelationLength(radialCorr) {
  if (!radialCorr.length) return 0;
  const c1 = radialCorr[0];
  if (!(c1 > 0.015)) return 0;
  // Weighted log-linear fit on positive lags r=1..k while C(r) > 0.015
  const pts = [];
  for (let i = 0; i < radialCorr.length; i++) {
    const c = radialCorr[i];
    if (c > 0.015) pts.push([i + 1, Math.log(c), c]);
    else break;
  }
  if (pts.length === 1) {
    // Single-point estimate assuming C(0)=1 -> C(1) = exp(-1/xi)
    return c1 < 0.999 ? -1 / Math.log(c1) : radialCorr.length;
  }
  // Include (r=0, ln C=0) weakly or fit slope between r=1..k
  let sw = 0, sr = 0, sy = 0, srr = 0, sry = 0;
  for (const [r, y, w] of pts) {
    sw += w; sr += w * r; sy += w * y; srr += w * r * r; sry += w * r * y;
  }
  const det = sw * srr - sr * sr;
  if (Math.abs(det) < 1e-12) return c1 < 0.999 ? -1 / Math.log(c1) : 0;
  const slope = (sw * sry - sr * sy) / det;
  if (slope >= -1e-4) return radialCorr.length;
  return Math.min(radialCorr.length * 2, -1 / slope);
}

/**
 * Streaming multi-config supercell symmetry & domain accumulator.
 * Processes configurations one at a time in O(1) memory in M.
 */
export class StreamingSymmetryAccumulator {
  constructor(dim, A, refBasis, opts = {}) {
    const {
      parentTol = 0.25,
      subtractDrift = true,
      nNullSamples = 128,
      maxSavedDomainGrids = 12,
      minConfigRms = 1e-3,
    } = opts;

    this.dim = dim;
    this.N1 = dim[0]; this.N2 = dim[1]; this.N3 = dim[2];
    this.nCells = this.N1 * this.N2 * this.N3;
    this.A = A;
    this.Ainv = inv3(A);
    this.refBasis = refBasis;
    this.nBasis = refBasis.length;
    this.D = 3 * this.nBasis;
    this.subtractDrift = subtractDrift;
    this.Dint = subtractDrift ? this.D - 3 : this.D;
    this.parentTol = parentTol;
    this.maxSavedDomainGrids = maxSavedDomainGrids;
    this.minConfigRms = minConfigRms;
    this.skippedConfigs = [];

    // Average unit-cell edge length (Å) for reporting ξ in Å
    this.aAvg = (
      Math.hypot(A[0][0], A[0][1], A[0][2]) +
      Math.hypot(A[1][0], A[1][1], A[1][2]) +
      Math.hypot(A[2][0], A[2][1], A[2][2])
    ) / 3;

    // Parent symmetry & subgroup enumeration (done ONCE)
    this.parentSG = findSpaceGroupOps(A, refBasis, parentTol);
    this.reps = operationReps(A, refBasis, this.parentSG.ops, Math.max(0.05, parentTol / 5));
    this.idealParentBasis = idealizeBasis(refBasis, this.reps);
    this.subgroups = enumerateSubgroups(this.reps);
    this.allOpIndices = Array.from({ length: this.reps.length }, (_, i) => i);
    this.parentFamIdx = this.subgroups.findIndex(f => f.order === this.reps.length);

    this.maxLag = Math.max(1, Math.min(6, Math.floor(Math.min(this.N1, this.N2, this.N3) / 2)));

    // Per-subgroup accumulators
    this.parentDoF = 0;
    this.fams = this.subgroups.map((fam) => {
      const dof = subgroupDoF(fam.variants[0].ops, this.nBasis, subtractDrift);
      return {
        ...fam,
        dof: dof.internal,
        dofTotal: dof.total,
        nullR2: this.Dint > 0 ? dof.internal / this.Dint : 0,
        nVariants: fam.variants.length,
        // Null calibration targets (filled below)
        nullBestR2: 0,
        nullBreakingFStat: 0,
        nullBreakingFStd: 0,
        // Global cell-level accumulators
        sumMeanR2: 0,
        sumBestR2: 0,
        sumCvR2: 0,
        domainCounts: new Float64Array(fam.variants.length),
        nCellSamples: 0,
        // Radial domain correlation (lags 1..maxLag)
        lagMatches: new Float64Array(this.maxLag),
        lagPairs: new Float64Array(this.maxLag),
        // 3D cluster accumulators (per config)
        sumMeanCluster: 0,
        sumWeightCluster: 0,
        sumMaxCluster: 0,
        // Per-config online Welford stats for SE & z-score
        nCfgStats: 0,
        meanExcessDF: 0, m2ExcessDF: 0,
        meanCvDF: 0, m2CvDF: 0,
        meanRawDF: 0, m2RawDF: 0,
        meanNnMatch: 0, m2NnMatch: 0,
        winCount: 0,
        excessDFValues: [],
        // Per-variant domain fraction Welford stats across configs
        varFracMean: new Float64Array(fam.variants.length),
        varFracM2: new Float64Array(fam.variants.length),
        // Streaming domain-aligned structure accumulators (O(D) memory!)
        refBrk: new Float64Array(this.D),
        hasRefBrk: false,
        sumAligned: new Float64Array(this.D),
        sumFolded: new Float64Array(this.D),
        cosetCounts: new Float64Array(Math.max(1, fam.normCosetReps?.length || 1)),
        nAlignedCells: 0,
        // Per-config aligned mean Welford stats (for per-atom shift uncertainty)
        cfgAlignMean: new Float64Array(this.D),
        cfgAlignM2: new Float64Array(this.D),
        nCfgAlign: 0,
        // Saved representative domain grids (up to maxSavedDomainGrids configs)
        savedGrids: [],
      };
    });

    if (this.parentFamIdx >= 0) this.parentDoF = this.fams[this.parentFamIdx].dof;

    // Reusable work buffers (zero allocations per config)
    this._uCell = new Float64Array(this.D);
    this._gU = Array.from({ length: this.reps.length }, () => new Float64Array(this.D));
    this._proj = new Float64Array(this.D);
    this._projG0 = new Float64Array(this.D);
    this._alignedCell = new Float64Array(this.D);
    this._brkVec = new Float64Array(this.D);
    this._candBrk = new Float64Array(this.D);
    this._tmp = new Float64Array(this.D);

    // Per-config scratch arrays for spatial CV and domain grids
    this._cellDomain = this.fams.map(() => new Int32Array(this.nCells));
    this._cellVarR2 = this.fams.map(f => new Float32Array(this.nCells * f.nVariants));
    this._cellValid = new Uint8Array(this.nCells);
    this._cfgAlignSum = this.fams.map(() => new Float64Array(this.D));

    // Streaming k-space accumulators
    this.elements = [...new Set(refBasis.map(b => b.el))];
    this.elIdxOfSite = refBasis.map(b => this.elements.indexOf(b.el));
    this.kPower = new Float64Array(this.nCells);
    this.kPowerByEl = this.elements.map(() => new Float64Array(this.nCells));
    this.kMeanRe = new Float64Array(this.nCells * this.D);
    this.kMeanIm = new Float64Array(this.nCells * this.D);

    // Streaming convergence trace & config labels
    this.nConfigs = 0;
    this.configNames = [];
    this.convergence = [];

    // Run exact isotropic-noise null calibration once
    this._calibrateNull(nNullSamples);
  }

  _calibrateNull(nSamples) {
    if (nSamples <= 0 || this.Dint <= 0) return;
    const rng = mulberry32(20261125 ^ (this.nBasis * 2654435761));
    const randn = makeGaussian(rng);
    const D = this.D;
    const uRaw = new Float64Array(D);
    const uCell = this._uCell;
    const gU = this._gU;
    const proj = this._proj;

    const sumBest = new Float64Array(this.fams.length);
    const sumDF = new Float64Array(this.fams.length);
    const sumDF2 = new Float64Array(this.fams.length);

    for (let s = 0; s < nSamples; s++) {
      for (let d = 0; d < D; d++) uRaw[d] = randn();
      if (this.subtractDrift) removeCellDrift(uRaw, this.nBasis, uCell);
      else uCell.set(uRaw);
      let norm2 = 0;
      for (let d = 0; d < D; d++) norm2 += uCell[d] * uCell[d];
      if (norm2 < 1e-18) continue;

      for (let gi = 0; gi < this.reps.length; gi++) applyGammaCell(this.reps[gi], uCell, this.nBasis, gU[gi]);

      // Parent G0 R² on this null cell
      let parentR2 = 0;
      if (this.parentFamIdx >= 0) {
        const opIdx = this.fams[this.parentFamIdx].variants[0].opIndices;
        projectOps(opIdx, this.reps, gU, D, proj);
        let p2 = 0;
        for (let d = 0; d < D; d++) p2 += proj[d] * proj[d];
        parentR2 = Math.min(1, p2 / norm2);
      }

      for (let fi = 0; fi < this.fams.length; fi++) {
        const fam = this.fams[fi];
        let bestR2 = 0;
        for (let vi = 0; vi < fam.nVariants; vi++) {
          projectOps(fam.variants[vi].opIndices, this.reps, gU, D, proj);
          let p2 = 0;
          for (let d = 0; d < D; d++) p2 += proj[d] * proj[d];
          const r2 = Math.min(1, p2 / norm2);
          if (r2 > bestR2) bestR2 = r2;
        }
        sumBest[fi] += bestR2;
        const brkDoF = Math.max(0, fam.dof - this.parentDoF);
        const remDoF = Math.max(1, this.Dint - fam.dof);
        const df = brkDoF > 0 && fam.dof < this.Dint
          ? (Math.max(0, bestR2 - parentR2) / brkDoF) / Math.max(1e-12, (1 - bestR2) / remDoF)
          : 0;
        sumDF[fi] += df;
        sumDF2[fi] += df * df;
      }
    }

    const inv = 1 / nSamples;
    const nullParentR2 = this.parentFamIdx >= 0 ? sumBest[this.parentFamIdx] * inv : 0;
    for (let fi = 0; fi < this.fams.length; fi++) {
      const fam = this.fams[fi];
      fam.nullBestR2 = sumBest[fi] * inv;
      const brkDoF = Math.max(0, fam.dof - this.parentDoF);
      const remDoF = Math.max(1, this.Dint - fam.dof);
      // Use the exact same ratio-of-means formula as addConfig / finalize (avoids Jensen's inequality bias)
      fam.nullBreakingFStat = brkDoF > 0 && fam.dof < this.Dint
        ? (Math.max(0, fam.nullBestR2 - nullParentR2) / brkDoF) / Math.max(1e-12, (1 - fam.nullBestR2) / remDoF)
        : 0;
      const meanPerCellDF = sumDF[fi] * inv;
      const v = Math.max(0, sumDF2[fi] * inv - meanPerCellDF ** 2);
      fam.nullBreakingFStd = Math.sqrt(v);
    }
  }

  /**
   * Stream one supercell displacement configuration u: Float64Array(nCells * D).
   * After this call returns, the caller may immediately discard `u`.
   */
  addConfig(u, cfgName = null) {
    const { N1, N2, N3, nCells, nBasis, D, Dint } = this;
    const cfgLabel = cfgName || `Config ${this.nConfigs + 1}`;

    // Guard against unconverged / zero-displacement initial configurations (e.g. *_0.rmc6f)
    if (this.minConfigRms > 0) {
      let totSq = 0;
      for (let i = 0; i < u.length; i++) totSq += u[i] * u[i];
      const cfgRms = Math.sqrt(totSq / Math.max(1, nCells * nBasis));
      if (cfgRms < this.minConfigRms) {
        this.skippedConfigs.push({ name: cfgLabel, rms: cfgRms, reason: 'near-zero displacement (unconverged)' });
        return false;
      }
    }

    this.nConfigs++;
    this.configNames.push(cfgLabel);

    // 1. Streaming 3D FFT for k-space mode spectrum
    {
      const { re, im } = supercellFFT3D(u, this.dim, nBasis);
      for (let c = 0; c < nCells; c++) {
        const base = c * D;
        let pTot = 0;
        for (let s = 0; s < nBasis; s++) {
          const ei = this.elIdxOfSite[s];
          const o = base + s * 3;
          let ps = 0;
          for (let a = 0; a < 3; a++) {
            const r = re[o + a], m = im[o + a];
            ps += r * r + m * m;
            this.kMeanRe[o + a] += r;
            this.kMeanIm[o + a] += m;
          }
          pTot += ps;
          this.kPowerByEl[ei][c] += ps;
        }
        this.kPower[c] += pTot;
      }
    }

    // 2. Per-cell subgroup projections & streaming domain alignment
    const uCell = this._uCell;
    const gU = this._gU;
    const proj = this._proj;
    const projG0 = this._projG0;
    const alignedCell = this._alignedCell;
    const brkVec = this._brkVec;
    const candBrk = this._candBrk;
    const tmp = this._tmp;

    const cfgSumBest = new Float64Array(this.fams.length);
    const cfgVarCounts = this.fams.map(f => new Float64Array(f.nVariants));
    for (let fi = 0; fi < this.fams.length; fi++) this._cfgAlignSum[fi].fill(0);
    this._cellValid.fill(0);
    let nValidCells = 0;

    // On config #1, first find strongest breaking cell per subgroup to lock canonical polarity
    const lockPolarity = this.nConfigs === 1;
    const bestBrkNorm2 = lockPolarity ? new Float64Array(this.fams.length).fill(-1) : null;

    for (let c = 0; c < nCells; c++) {
      const base = c * D;
      if (this.subtractDrift) removeCellDrift(u.subarray(base, base + D), nBasis, uCell);
      else uCell.set(u.subarray(base, base + D));

      let norm2 = 0;
      for (let d = 0; d < D; d++) norm2 += uCell[d] * uCell[d];
      if (norm2 < 1e-18) continue;

      this._cellValid[c] = 1;
      nValidCells++;

      // Compute Γ(g) u_l ONCE per cell for all g ∈ G0
      for (let gi = 0; gi < this.reps.length; gi++) {
        applyGammaCell(this.reps[gi], uCell, nBasis, gU[gi]);
      }

      // Compute parent G0 projection once for symmetry-breaking polarity alignment
      if (this.parentFamIdx >= 0) {
        projectOps(this.allOpIndices, this.reps, gU, D, projG0);
      } else {
        projG0.fill(0);
      }

      for (let fi = 0; fi < this.fams.length; fi++) {
        const fam = this.fams[fi];
        const varR2Buf = this._cellVarR2[fi];
        const vOff = c * fam.nVariants;
        let sumVarR2 = 0;
        let bestR2 = -1;
        let bestV = 0;

        for (let vi = 0; vi < fam.nVariants; vi++) {
          projectOps(fam.variants[vi].opIndices, this.reps, gU, D, proj);
          let pNorm2 = 0;
          for (let d = 0; d < D; d++) pNorm2 += proj[d] * proj[d];
          const r2 = Math.min(1, pNorm2 / norm2);
          varR2Buf[vOff + vi] = r2;
          sumVarR2 += r2;
          if (r2 > bestR2) { bestR2 = r2; bestV = vi; }
        }

        fam.sumMeanR2 += sumVarR2 / fam.nVariants;
        fam.sumBestR2 += bestR2;
        fam.domainCounts[bestV] += 1;
        fam.nCellSamples += 1;
        cfgSumBest[fi] += bestR2;
        cfgVarCounts[fi][bestV] += 1;
        this._cellDomain[fi][c] = bestV;

        // Stream domain-aligned vector for breaking subgroups (dof > parentDoF and dof < Dint)
        if (fam.dof > this.parentDoF && fam.dof < Dint) {
          const alignOp = fam.variants[bestV].alignOpIdx;
          alignedCell.set(gU[alignOp]);

          const normCosets = fam.normCosetReps?.length ? fam.normCosetReps : [0];
          const h0Ops = fam.variants[0].opIndices;

          // Compute H_0 - G_0 breaking component of alignedCell
          // Apply H_0 projection on alignedCell
          proj.fill(0);
          for (let k = 0; k < h0Ops.length; k++) {
            applyGammaCell(this.reps[h0Ops[k]], alignedCell, nBasis, tmp);
            for (let d = 0; d < D; d++) proj[d] += tmp[d];
          }
          const invH0 = 1 / h0Ops.length;
          let b2 = 0;
          for (let d = 0; d < D; d++) {
            const diff = proj[d] * invH0 - projG0[d];
            brkVec[d] = diff;
            b2 += diff * diff;
          }

          if (!fam.hasRefBrk && b2 > 1e-18) {
            fam.refBrk.set(brkVec);
            fam.hasRefBrk = true;
            if (bestBrkNorm2) bestBrkNorm2[fi] = b2;
          } else if (lockPolarity && bestBrkNorm2 && b2 > bestBrkNorm2[fi] * 2.5) {
            // Keep polarity aligned with existing refBrk sign before strengthening reference
            let sgnDot = 0;
            for (let d = 0; d < D; d++) sgnDot += brkVec[d] * fam.refBrk[d];
            if (sgnDot >= 0) {
              fam.refBrk.set(brkVec);
              bestBrkNorm2[fi] = b2;
            }
          }

          let bestCosetIdx = 0;
          if (normCosets.length > 1 && fam.hasRefBrk) {
            let bestDot = -Infinity;
            for (let ni = 0; ni < normCosets.length; ni++) {
              applyGammaCell(this.reps[normCosets[ni]], brkVec, nBasis, candBrk);
              let dot = 0;
              for (let d = 0; d < D; d++) dot += candBrk[d] * fam.refBrk[d];
              if (dot > bestDot) { bestDot = dot; bestCosetIdx = ni; }
            }
            applyGammaCell(this.reps[normCosets[bestCosetIdx]], alignedCell, nBasis, tmp);
            alignedCell.set(tmp);
          }

          fam.cosetCounts[bestCosetIdx] += 1;
          for (let d = 0; d < D; d++) {
            fam.sumAligned[d] += alignedCell[d];
            fam.sumFolded[d] += uCell[d];
            this._cfgAlignSum[fi][d] += alignedCell[d];
          }
          fam.nAlignedCells += 1;
        }
      }
    }

    if (nValidCells === 0) return;

    // 3. Spatial neighbor cross-validation (bias-free ΔR²_CV) + Radial C_dom(r) + 3D BFS clusters
    const parentCfgR2 = this.parentFamIdx >= 0 ? cfgSumBest[this.parentFamIdx] / nValidCells : 0;
    let bestCfgFamIdx = -1;
    let bestCfgExcessDF = -Infinity;

    for (let fi = 0; fi < this.fams.length; fi++) {
      const fam = this.fams[fi];
      const dom = this._cellDomain[fi];
      const varR2Buf = this._cellVarR2[fi];
      const nV = fam.nVariants;

      // (a) Out-of-sample neighbor-voted R²_CV
      let cfgSumCvR2 = 0;
      if (nV === 1) {
        cfgSumCvR2 = cfgSumBest[fi];
      } else {
        const voteBuf = new Float64Array(nV);
        for (let x = 0; x < N1; x++) {
          const xp = (x + 1) % N1, xm = (x + N1 - 1) % N1;
          for (let y = 0; y < N2; y++) {
            const yp = (y + 1) % N2, ym = (y + N2 - 1) % N2;
            for (let z = 0; z < N3; z++) {
              const c = (x * N2 + y) * N3 + z;
              if (!this._cellValid[c]) continue;
              const zp = (z + 1) % N3, zm = (z + N3 - 1) % N3;
              const nbs = [
                (xp * N2 + y) * N3 + z, (xm * N2 + y) * N3 + z,
                (x * N2 + yp) * N3 + z, (x * N2 + ym) * N3 + z,
                (x * N2 + y) * N3 + zp, (x * N2 + y) * N3 + zm,
              ];
              voteBuf.fill(0);
              for (let k = 0; k < 6; k++) {
                const nb = nbs[k];
                if (!this._cellValid[nb]) continue;
                voteBuf[dom[nb]] += 1;
              }
              let bestNbV = 0, bestVote = -1;
              for (let vi = 0; vi < nV; vi++) {
                if (voteBuf[vi] > bestVote) { bestVote = voteBuf[vi]; bestNbV = vi; }
              }
              cfgSumCvR2 += varR2Buf[c * nV + bestNbV];
            }
          }
        }
      }
      fam.sumCvR2 += cfgSumCvR2;

      // (b) Radial autocorrelation for lags r = 1..maxLag + 3D BFS cluster sizes
      let cfgNnMatch = null;
      if (nV > 1) {
        for (let r = 1; r <= this.maxLag; r++) {
          let matches = 0, pairs = 0;
          for (let x = 0; x < N1; x++) {
            const xr = (x + r) % N1;
            for (let y = 0; y < N2; y++) {
              const yr = (y + r) % N2;
              for (let z = 0; z < N3; z++) {
                const zr = (z + r) % N3;
                const c = (x * N2 + y) * N3 + z;
                const d0 = dom[c];
                if (dom[(xr * N2 + y) * N3 + z] === d0) matches++;
                if (dom[(x * N2 + yr) * N3 + z] === d0) matches++;
                if (dom[(x * N2 + y) * N3 + zr] === d0) matches++;
                pairs += 3;
              }
            }
          }
          fam.lagMatches[r - 1] += matches;
          fam.lagPairs[r - 1] += pairs;
          if (r === 1) cfgNnMatch = matches / Math.max(1, pairs);
        }

        const cl = domainClusterStats(dom, N1, N2, N3);
        fam.sumMeanCluster += cl.meanClusterCells;
        fam.sumWeightCluster += cl.weightMeanClusterCells;
        fam.sumMaxCluster += cl.maxClusterCells;

        if (fam.savedGrids.length < this.maxSavedDomainGrids) {
          fam.savedGrids.push({ configName: cfgLabel, grid: Int32Array.from(dom) });
        }
      }

      // (c) Per-config ΔF stats via Welford online variance
      const cfgBestR2 = cfgSumBest[fi] / nValidCells;
      const cfgCvR2 = cfgSumCvR2 / nValidCells;
      const brkDoF = Math.max(0, fam.dof - this.parentDoF);
      const remDoF = Math.max(1, Dint - fam.dof);
      const rawDF = brkDoF > 0 && fam.dof < Dint
        ? (Math.max(0, cfgBestR2 - parentCfgR2) / brkDoF) / Math.max(1e-12, (1 - cfgBestR2) / remDoF)
        : 0;
      const excessDF = brkDoF > 0 && fam.dof < Dint ? rawDF - fam.nullBreakingFStat : 0;
      const cvDF = brkDoF > 0 && fam.dof < Dint
        ? ((cfgCvR2 - parentCfgR2) / brkDoF) / Math.max(1e-12, (1 - cfgCvR2) / remDoF)
        : 0;

      fam.nCfgStats++;
      fam.excessDFValues.push(excessDF);
      const k = fam.nCfgStats;
      {
        const d1 = rawDF - fam.meanRawDF; fam.meanRawDF += d1 / k; fam.m2RawDF += d1 * (rawDF - fam.meanRawDF);
        const d2 = excessDF - fam.meanExcessDF; fam.meanExcessDF += d2 / k; fam.m2ExcessDF += d2 * (excessDF - fam.meanExcessDF);
        const d3 = cvDF - fam.meanCvDF; fam.meanCvDF += d3 / k; fam.m2CvDF += d3 * (cvDF - fam.meanCvDF);
        if (cfgNnMatch !== null) {
          const d4 = cfgNnMatch - fam.meanNnMatch; fam.meanNnMatch += d4 / k; fam.m2NnMatch += d4 * (cfgNnMatch - fam.meanNnMatch);
        }
        for (let vi = 0; vi < nV; vi++) {
          const vf = cfgVarCounts[fi][vi] / nValidCells;
          const dv = vf - fam.varFracMean[vi];
          fam.varFracMean[vi] += dv / k;
          fam.varFracM2[vi] += dv * (vf - fam.varFracMean[vi]);
        }
      }

      if (brkDoF > 0 && fam.dof < Dint && excessDF > bestCfgExcessDF) {
        bestCfgExcessDF = excessDF;
        bestCfgFamIdx = fi;
      }

      // Per-config aligned mean Welford stats (for SE of atomic shifts across configs)
      if (fam.dof > this.parentDoF && fam.dof < Dint) {
        fam.nCfgAlign++;
        const ka = fam.nCfgAlign;
        const invC = 1 / nValidCells;
        const cSum = this._cfgAlignSum[fi];
        for (let d = 0; d < D; d++) {
          const x = cSum[d] * invC;
          const delta = x - fam.cfgAlignMean[d];
          fam.cfgAlignMean[d] += delta / ka;
          fam.cfgAlignM2[d] += delta * (x - fam.cfgAlignMean[d]);
        }
      }
    }

    if (bestCfgFamIdx >= 0) this.fams[bestCfgFamIdx].winCount += 1;

    // 4. Record ensemble convergence checkpoint
    let leadIdx = -1, leadScore = -Infinity;
    for (let fi = 0; fi < this.fams.length; fi++) {
      const f = this.fams[fi];
      if (f.dof > this.parentDoF && f.dof < Dint && f.meanExcessDF > leadScore) {
        leadScore = f.meanExcessDF;
        leadIdx = fi;
      }
    }
    if (leadIdx >= 0) {
      const lf = this.fams[leadIdx];
      const invA = 1 / Math.max(1, lf.nAlignedCells);
      let s2 = 0;
      for (let d = 0; d < D; d++) { const v = lf.sumAligned[d] * invA; s2 += v * v; }
      this.convergence.push({
        n: this.nConfigs,
        topSg: lf.spaceGroup,
        excessDeltaF: lf.meanExcessDF,
        rawDeltaF: lf.meanRawDF,
        cvDeltaF: lf.meanCvDF,
        topRms: Math.sqrt(s2 / Math.max(1, nBasis)),
      });
    }
  }

  /**
   * Build the finalized aligned structure object for subgroup family `fam`.
   */
  _buildAlignedForFamily(fam) {
    const { D, nBasis, A, Ainv } = this;
    const invN = 1 / Math.max(1, fam.nAlignedCells);
    const meanAligned = new Float64Array(D);
    const meanFolded = new Float64Array(D);
    for (let d = 0; d < D; d++) {
      meanAligned[d] = fam.sumAligned[d] * invN;
      meanFolded[d] = fam.sumFolded[d] * invN;
    }

    // Exact H_0-symmetrized projection of the ensemble aligned mean
    const h0Ops = fam.variants[0].opIndices;
    const symAligned = new Float64Array(D);
    const tmp = this._tmp;
    for (let k = 0; k < h0Ops.length; k++) {
      applyGammaCell(this.reps[h0Ops[k]], meanAligned, nBasis, tmp);
      for (let d = 0; d < D; d++) symAligned[d] += tmp[d];
    }
    const invH = 1 / h0Ops.length;
    for (let d = 0; d < D; d++) symAligned[d] *= invH;

    // Per-atom standard error across configs (when M >= 2)
    const nCfg = fam.nCfgAlign;
    const seVec = new Float64Array(D);
    if (nCfg >= 2) {
      for (let d = 0; d < D; d++) {
        const varCfg = fam.cfgAlignM2[d] / (nCfg - 1);
        seVec[d] = Math.sqrt(Math.max(0, varCfg) / nCfg);
      }
    }

    const toBasis = (uVec, baseSites) => baseSites.map((b, s) => {
      const ux = uVec[s * 3], uy = uVec[s * 3 + 1], uz = uVec[s * 3 + 2];
      const df = [
        ux * Ainv[0][0] + uy * Ainv[1][0] + uz * Ainv[2][0],
        ux * Ainv[0][1] + uy * Ainv[1][1] + uz * Ainv[2][1],
        ux * Ainv[0][2] + uy * Ainv[1][2] + uz * Ainv[2][2],
      ];
      const seNorm = Math.hypot(seVec[s * 3], seVec[s * 3 + 1], seVec[s * 3 + 2]);
      return {
        rn: b.rn,
        el: b.el,
        frac: [wrap01(b.frac[0] + df[0]), wrap01(b.frac[1] + df[1]), wrap01(b.frac[2] + df[2])],
        dispCart: [ux, uy, uz],
        dispNorm: Math.hypot(ux, uy, uz),
        dispSE: seNorm,
      };
    });

    const alignedBasis = toBasis(meanAligned, this.refBasis);
    const symmetrizedBasis = toBasis(symAligned, this.idealParentBasis);
    const foldedBasis = toBasis(meanFolded, this.refBasis);

    const rmsVec = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0) / Math.max(1, nBasis));
    const alignedRms = rmsVec(meanAligned);
    const symRms = rmsVec(symAligned);
    const foldedRms = rmsVec(meanFolded);
    const rmsSE = rmsVec(seVec);
    const maxShift = Math.max(0, ...symmetrizedBasis.map(s => s.dispNorm));

    const verifiedSG = findSpaceGroupOps(A, symmetrizedBasis, 1e-3);

    const elements = [...new Set(this.refBasis.map(b => b.el))];
    const elementStats = elements.map(el => {
      const idxs = this.refBasis.map((b, i) => (b.el === el ? i : -1)).filter(i => i >= 0);
      const rmsOf = (arr) => Math.sqrt(idxs.reduce((s, i) => s + arr[i].dispNorm ** 2, 0) / Math.max(1, idxs.length));
      const maxOf = (arr) => Math.max(0, ...idxs.map(i => arr[i].dispNorm));
      return {
        el,
        count: idxs.length,
        alignedRms: rmsOf(alignedBasis),
        symRms: rmsOf(symmetrizedBasis),
        foldedRms: rmsOf(foldedBasis),
        maxShift: maxOf(symmetrizedBasis),
      };
    });

    return {
      targetSpaceGroup: fam.spaceGroup,
      targetSpaceGroupNumber: fam.spaceGroupNumber,
      targetPointGroup: fam.pointGroup,
      parentSpaceGroup: this.parentSG.spaceGroup,
      verifiedSpaceGroup: verifiedSG.spaceGroup,
      verifiedSpaceGroupNumber: verifiedSG.spaceGroupNumber,
      verifiedOrder: verifiedSG.order,
      nVariants: fam.nVariants,
      nNormalizerCosets: Math.max(1, fam.normCosetReps?.length || 1),
      cosetFractions: Array.from(fam.cosetCounts).map(x => x / Math.max(1, fam.nAlignedCells)),
      nSamples: fam.nAlignedCells,
      nConfigs: this.nConfigs,
      A,
      alignedRms,
      symRms,
      rmsSE,
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
   * Finalize ensemble statistics across all streamed configurations.
   */
  finalize() {
    if (this.nConfigs === 0) throw new Error('No configurations were added to the ensemble analyzer.');

    const { N1, N2, N3, nCells, D, Dint, nConfigs } = this;
    const invCfg = 1 / nConfigs;

    // 1. Finalize k-space mode spectrum (coherent vs fluctuating/multi-domain diffuse power)
    const power = new Float64Array(nCells);
    const coherentPower = new Float64Array(nCells);
    const meanRe = new Float64Array(nCells * D);
    const meanIm = new Float64Array(nCells * D);
    for (let c = 0; c < nCells; c++) {
      power[c] = this.kPower[c] * invCfg;
      const base = c * D;
      let cp = 0;
      for (let d = 0; d < D; d++) {
        const r = this.kMeanRe[base + d] * invCfg;
        const m = this.kMeanIm[base + d] * invCfg;
        meanRe[base + d] = r;
        meanIm[base + d] = m;
        cp += r * r + m * m;
      }
      coherentPower[c] = cp;
    }

    const sortedP = Array.from(power).sort((a, b) => a - b);
    const medianPower = sortedP[Math.floor(sortedP.length / 2)] || 1e-12;
    const meanPower = sortedP.reduce((a, b) => a + b, 0) / sortedP.length;

    const kPoints = [];
    for (let kx = 0; kx < N1; kx++) {
      for (let ky = 0; ky < N2; ky++) {
        for (let kz = 0; kz < N3; kz++) {
          const c = (kx * N2 + ky) * N3 + kz;
          const fx = kx > N1 / 2 ? kx / N1 - 1 : kx / N1;
          const fy = ky > N2 / 2 ? ky / N2 - 1 : ky / N2;
          const fz = kz > N3 / 2 ? kz / N3 - 1 : kz / N3;
          const byEl = {};
          for (let e = 0; e < this.elements.length; e++) byEl[this.elements[e]] = this.kPowerByEl[e][c] * invCfg;
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

    // High-symmetry zone-boundary summary (Γ, X, M, R, low-q)
    const nearHalf = (x) => Math.abs(Math.abs(x) - 0.5) < 1e-6;
    const nearZero = (x) => Math.abs(x) < 1e-6;
    const avgSnr = (arr) => arr.length ? arr.reduce((s, p) => s + p.snr, 0) / arr.length : null;
    const highSymStars = {
      gamma: power[0] / medianPower,
      X: avgSnr(kPoints.filter(p => {
        const nh = [p.kFrac[0], p.kFrac[1], p.kFrac[2]].filter(nearHalf).length;
        const nz = [p.kFrac[0], p.kFrac[1], p.kFrac[2]].filter(nearZero).length;
        return nh === 1 && nz === 2;
      })),
      M: avgSnr(kPoints.filter(p => {
        const nh = [p.kFrac[0], p.kFrac[1], p.kFrac[2]].filter(nearHalf).length;
        const nz = [p.kFrac[0], p.kFrac[1], p.kFrac[2]].filter(nearZero).length;
        return nh === 2 && nz === 1;
      })),
      R: avgSnr(kPoints.filter(p => p.kFrac.every(nearHalf))),
    };

    const spectrum = {
      elements: this.elements,
      medianPower,
      meanPower,
      gammaPower: power[0],
      gammaSnr: power[0] / medianPower,
      highSymStars,
      topK,
      kPoints,
      meanFourierRe: meanRe,
      meanFourierIm: meanIm,
    };

    // 2. Finalize subgroup rankings & domain physics
    const parentMeanR2 = this.parentFamIdx >= 0
      ? this.fams[this.parentFamIdx].sumBestR2 / Math.max(1, this.fams[this.parentFamIdx].nCellSamples)
      : 0;

    const alignedBySubgroup = {};
    const subgroups = this.fams.map((fam) => {
      const n = Math.max(1, fam.nCellSamples);
      const meanR2 = fam.sumMeanR2 / n;
      const bestDomainR2 = fam.sumBestR2 / n;
      const cvDomainR2 = fam.sumCvR2 / n;
      const remDoF = Math.max(1, Dint - fam.dof);
      const breakingDoF = Math.max(0, fam.dof - this.parentDoF);
      const breakingR2 = Math.max(0, bestDomainR2 - parentMeanR2);
      const cvBreakingR2 = cvDomainR2 - parentMeanR2;

      const fStat = fam.dof > 0 && fam.dof < Dint
        ? (bestDomainR2 / fam.dof) / Math.max(1e-12, (1 - bestDomainR2) / remDoF)
        : (fam.dof === Dint ? 1 : 0);
      const breakingFStat = breakingDoF > 0 && fam.dof < Dint
        ? (breakingR2 / breakingDoF) / Math.max(1e-12, (1 - bestDomainR2) / remDoF)
        : 0;
      const pooledExcessDF = breakingDoF > 0 && fam.dof < Dint
        ? breakingFStat - fam.nullBreakingFStat
        : 0;
      const cvBreakingFStat = breakingDoF > 0 && fam.dof < Dint
        ? (cvBreakingR2 / breakingDoF) / Math.max(1e-12, (1 - cvDomainR2) / remDoF)
        : 0;

      // Ensemble standard error, median (outlier-robust), & z-score across configs
      const k = fam.nCfgStats;
      const excessDeltaF = k >= 2 ? fam.meanExcessDF : pooledExcessDF;
      const excessDeltaFSE = k >= 2 ? Math.sqrt(Math.max(0, fam.m2ExcessDF / (k - 1)) / k) : null;
      const cvDeltaFSE = k >= 2 ? Math.sqrt(Math.max(0, fam.m2CvDF / (k - 1)) / k) : null;
      const zScore = excessDeltaFSE && excessDeltaFSE > 1e-9 ? fam.meanExcessDF / excessDeltaFSE : null;
      const winRate = k > 0 ? fam.winCount / k : 0;
      let medianExcessDeltaF = excessDeltaF;
      if (fam.excessDFValues.length > 0) {
        const sArr = [...fam.excessDFValues].sort((a, b) => a - b);
        const mid = sArr.length >> 1;
        medianExcessDeltaF = sArr.length % 2 ? sArr[mid] : 0.5 * (sArr[mid - 1] + sArr[mid]);
      }

      // Domain fractions + SE
      const domainFractions = Array.from(fam.domainCounts).map(x => x / n);
      const domainFractionSE = k >= 2
        ? Array.from(fam.varFracM2).map(m2 => Math.sqrt(Math.max(0, m2 / (k - 1)) / k))
        : null;

      // Radial correlation C_dom(r) & correlation length ξ
      const randomMatchRate = 1 / fam.nVariants;
      const radialMatchRate = [];
      const radialCorr = [];
      if (fam.nVariants > 1) {
        for (let r = 0; r < this.maxLag; r++) {
          const pMatch = fam.lagPairs[r] > 0 ? fam.lagMatches[r] / fam.lagPairs[r] : randomMatchRate;
          radialMatchRate.push(pMatch);
          radialCorr.push((pMatch - randomMatchRate) / Math.max(1e-12, 1 - randomMatchRate));
        }
      }
      const neighborMatchRate = radialMatchRate.length ? radialMatchRate[0] : null;
      const neighborMatchSE = k >= 2 && fam.nVariants > 1
        ? Math.sqrt(Math.max(0, fam.m2NnMatch / (k - 1)) / k)
        : null;
      const corrLengthCells = fitCorrelationLength(radialCorr);
      const corrLengthAng = corrLengthCells * this.aAvg;

      // Prebuild ensemble domain-aligned structure for every breaking subgroup
      if (breakingDoF > 0 && fam.dof < Dint) {
        alignedBySubgroup[fam.spaceGroup] = this._buildAlignedForFamily(fam);
      }

      return {
        id: fam.id,
        spaceGroup: fam.spaceGroup,
        spaceGroupNumber: fam.spaceGroupNumber,
        pointGroup: fam.pointGroup,
        centering: fam.centering,
        order: fam.order,
        nVariants: fam.nVariants,
        dof: fam.dof,
        nullR2: fam.nullR2,
        nullBestR2: fam.nullBestR2,
        nullBreakingFStat: fam.nullBreakingFStat,
        meanR2,
        bestDomainR2,
        cvDomainR2,
        cvBreakingR2,
        fStat,
        breakingDoF,
        breakingR2,
        breakingFStat,
        excessDeltaF,
        medianExcessDeltaF,
        excessDeltaFSE,
        cvBreakingFStat,
        cvDeltaFSE,
        zScore,
        winRate,
        domainFractions,
        domainFractionSE,
        neighborMatchRate,
        neighborMatchSE,
        neighborExcess: neighborMatchRate !== null ? neighborMatchRate - randomMatchRate : null,
        radialCorr,
        corrLengthCells,
        corrLengthAng,
        meanClusterCells: fam.nVariants > 1 ? fam.sumMeanCluster / nConfigs : null,
        weightMeanClusterCells: fam.nVariants > 1 ? fam.sumWeightCluster / nConfigs : null,
        maxClusterCells: fam.nVariants > 1 ? fam.sumMaxCluster / nConfigs : null,
        domainGrid: fam.savedGrids.length ? Array.from(fam.savedGrids[0].grid) : null,
        savedGrids: fam.savedGrids.map(sg => ({ configName: sg.configName, grid: Array.from(sg.grid) })),
      };
    });

    // Rank breaking subgroups by outlier-robust median excess ΔF (for M >= 3) or excess ΔF
    const breakingSorted = subgroups
      .filter(s => s.breakingDoF > 0 && s.dof < Dint)
      .sort((a, b) => {
        const sa = nConfigs >= 3 ? a.medianExcessDeltaF : a.excessDeltaF;
        const sb = nConfigs >= 3 ? b.medianExcessDeltaF : b.excessDeltaF;
        return (sb - sa) || (b.cvBreakingFStat - a.cvBreakingFStat);
      });

    const topSubgroup = breakingSorted[0] || null;
    const aligned = topSubgroup ? alignedBySubgroup[topSubgroup.spaceGroup] || null : null;

    const minimal = filteredMinimalSupercell(spectrum, this.dim, this.A, this.refBasis, {
      snrThreshold: 2.5,
      minCoherentFrac: 0.1,
    });

    return {
      dim: this.dim,
      nConfigs,
      skippedConfigs: this.skippedConfigs,
      configNames: this.configNames,
      nBasis: this.nBasis,
      Aconv: this.A,
      refBasis: this.refBasis,
      idealParentBasis: this.idealParentBasis,
      spectrum,
      local: {
        parentSpaceGroup: this.parentSG.spaceGroup,
        parentOrder: this.parentSG.order,
        internalDoF: Dint,
        nConfigs,
        subgroups,
      },
      minimal,
      aligned,
      alignedBySubgroup,
      convergence: this.convergence,
    };
  }
}
