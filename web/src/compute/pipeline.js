import { ComputeEngine } from './engine';
import { eigh, eigenvaluesToMev } from '../math/diagonalize';
import { connectBands } from '../math/band_connection';
import { buildCellLabeling, vecMat3, IDENT } from '../math/cells';
import { conventionalLattice } from '../math/reciprocal';
import { ATOMIC_MASS, ENERGY_CONV, TWO_PI_PHASE } from '../constants';

/** Uniform Γ-centered q-grid over the conventional reciprocal cell [-0.5,0.5)^3
 *  (n^3 points) — the legacy Calculators.gen_grid, for phonon-DOS sampling. */
export function genGrid(n) {
  const ax = [];
  for (let i = 0; i < n; i++) ax.push(-0.5 + i / n);
  const pts = [];
  for (const x of ax) for (const y of ax) for (const z of ax) pts.push([x, y, z]);
  return pts;
}

export class PhononPipeline {
  constructor(onProgress) {
    this.engine = new ComputeEngine();
    this.onProgress = onProgress || (() => {});
    this.workerPool = [];
    this._cancel = false;
    this._cache = null;   // { key, parsedFrames } — reuse parse across runs
  }

  /** Request cancellation of the in-flight run (checked between k-points). */
  cancel() { this._cancel = true; }

  async initWorkers(count = 4) {
    for (let i = 0; i < count; i++) {
      this.workerPool.push(new Worker(new URL('../io/worker.js', import.meta.url), { type: 'module' }));
    }
    await this.engine.init();
  }

  parseFile(fileHandle, family, atomDic, dim, atype) {
    return new Promise((resolve, reject) => {
      const worker = this.workerPool.shift();
      if (!worker) { setTimeout(() => { this.parseFile(fileHandle, family, atomDic, dim, atype).then(resolve).catch(reject); }, 50); return; }
      worker.onmessage = (e) => {
        this.workerPool.push(worker);
        if (e.data.success) resolve(e.data); else reject(new Error(e.data.error));
      };
      worker.postMessage({ fileHandle, family, atomDic, dim, atype });
    });
  }

  // ── Shared setup: parse (cached) + displacement reference + masses/segments ──
  async _prepare(files, family, baseStructure, options = {}, batchSize = 50) {
    const { referenceHandle = null, referenceMode = 'per-atom' } = options;
    const numFiles = files.length;

    const els = Object.keys(baseStructure.atomDic).join(',');
    const cacheKey = `${family}|${numFiles}|${files[0]?.name}|${files[numFiles - 1]?.name}|${baseStructure.dim?.join('x')}|${els}`;
    let parsedFrames;
    if (this._cache && this._cache.key === cacheKey) {
      this.onProgress(30, `Reusing ${numFiles} parsed configs (cached)...`);
      parsedFrames = this._cache.parsedFrames;
    } else {
      this.onProgress(5, 'Parsing configurations...');
      let parsedCount = 0;
      const promises = files.map(async (file) => {
        const data = await this.parseFile(file, family, baseStructure.atomDic, baseStructure.dim, 0);
        parsedCount++;
        if (parsedCount % 20 === 0) this.onProgress(5 + (parsedCount / numFiles) * 25, `Parsed ${parsedCount} / ${numFiles} frames...`);
        return data;
      });
      parsedFrames = await Promise.all(promises);
      this._cache = { key: cacheKey, parsedFrames };
    }
    if (parsedFrames.length === 0) throw new Error('No valid frames parsed.');

    const numAtoms = parsedFrames[0].xyz.length / 3;
    const hsym_xyz = new Float64Array(numAtoms * 3);
    if (referenceHandle) {
      this.onProgress(30, 'Reading equilibrium reference file...');
      const ref = await this.parseFile(referenceHandle, 'rmc6f', baseStructure.atomDic, baseStructure.dim, 0);
      if (ref.xyz.length !== numAtoms * 3) throw new Error('Reference file atom count does not match the configurations.');
      const a0 = parsedFrames[0].atomType, c0 = parsedFrames[0].cellIdx;
      for (let i = 0; i < numAtoms; i++) {
        if (ref.atomType[i] !== a0[i] || ref.cellIdx[i * 3] !== c0[i * 3] || ref.cellIdx[i * 3 + 1] !== c0[i * 3 + 1] || ref.cellIdx[i * 3 + 2] !== c0[i * 3 + 2])
          throw new Error('Reference structure atom layout (RN / cell indices) does not match the configurations.');
      }
      hsym_xyz.set(ref.xyz);
    } else {
      for (let i = 0; i < parsedFrames.length; i++) { const xyz = parsedFrames[i].xyz; for (let j = 0; j < xyz.length; j++) hsym_xyz[j] += xyz[j]; }
      for (let j = 0; j < hsym_xyz.length; j++) hsym_xyz[j] /= parsedFrames.length;
    }

    // Per-atom element / mass (reference-number → element map).
    const firstFrameIds = parsedFrames[0].atomType;
    const reverseAtomDic = {};
    for (const [symbol, idxs] of Object.entries(baseStructure.atomDic)) idxs.forEach(idx => { reverseAtomDic[idx] = symbol; });
    const elements = new Array(numAtoms);
    const masses = new Float32Array(numAtoms);
    for (let i = 0; i < numAtoms; i++) {
      elements[i] = reverseAtomDic[firstFrameIds[i]];
      masses[i] = ATOMIC_MASS[elements[i]] || 0.0;
    }
    const v_super = [
      baseStructure.v1[0], baseStructure.v1[1], baseStructure.v1[2],
      baseStructure.v2[0], baseStructure.v2[1], baseStructure.v2[2],
      baseStructure.v3[0], baseStructure.v3[1], baseStructure.v3[2],
    ];

    // ── Computation cell (Phase 1) ──────────────────────────────────────────
    // Group the S(k) covariance by basis site τ of cell L = P·A_conv (default
    // P = I → conventional cell, identical to the previous per-RN grouping) and
    // index the Bloch phase by the cell index n in L units. See cells.js /
    // docs/cell-framework-plan.md.
    const P = options.computationCell?.P || IDENT;
    const Aconv = conventionalLattice(baseStructure.v1, baseStructure.v2, baseStructure.v3, baseStructure.dim);
    const cell0 = parsedFrames[0].cellIdx;
    const avgPos = new Array(numAtoms);
    for (let i = 0; i < numAtoms; i++) {
      // average position in conventional-cell fractional units = cellIdx + within-cell mean frac
      const fx = cell0[i * 3] + hsym_xyz[i * 3];
      const fy = cell0[i * 3 + 1] + hsym_xyz[i * 3 + 1];
      const fz = cell0[i * 3 + 2] + hsym_xyz[i * 3 + 2];
      avgPos[i] = [
        fx * Aconv[0][0] + fy * Aconv[1][0] + fz * Aconv[2][0],
        fx * Aconv[0][1] + fy * Aconv[1][1] + fz * Aconv[2][1],
        fx * Aconv[0][2] + fy * Aconv[1][2] + fz * Aconv[2][2],
      ];
    }
    const lab = buildCellLabeling(avgPos, elements, masses, Aconv, P, { tol: options.cellTol ?? 0.08 });
    if (lab.error) throw new Error(`Computation-cell labeling failed: ${lab.error}`);
    if (lab.issues.length) console.warn(
      `[cells] ${lab.issues.length} labeling issue(s) — the computation cell does not tile this ensemble average cleanly ` +
      `(e.g. a primitive cell on a structure whose average has broken the ideal centering). First few: ` +
      lab.issues.slice(0, 3).join('; ') + (lab.issues.length > 3 ? ' …' : ''));

    const numTypes = lab.nBasis;
    const typeIndices = lab.tau;
    const counts = lab.counts;
    const cellN = lab.cellN;            // per-atom cell index n (L units) for the phase

    // τ-ordered site identifiers (representative RN per basis site) for the
    // viewer model / exporters, plus the τ-ordered basis (frac + element).
    const tauRN = new Array(numTypes).fill(null);
    for (let i = 0; i < numAtoms; i++) { const t = typeIndices[i]; if (tauRN[t] == null) tauRN[t] = firstFrameIds[i]; }
    const segSymbols = lab.tauElement.slice();
    const basis = lab.tauFrac.map((frac, t) => ({ frac, element: lab.tauElement[t] }));

    // ── Displacement reference mode ─────────────────────────────────────────
    // per-atom (default): u = r − r̄_atom — each atom about its own ensemble mean
    //   (the validated behaviour; hsym_xyz is that per-atom mean).
    // symmetrized: u = r − (R_n + bf_τ) — about the cell's symmetrized basis site
    //   (shared across symmetry-equivalent sites); imposes the cell's symmetry on
    //   the equilibrium. bf_τ is in L units → convert to conventional within-cell
    //   fractions; the per-frame delta is wrapped to [-½,½) to handle the cell
    //   origin. See docs/cell-framework-plan.md.
    let refFrac = hsym_xyz, wrapRef = false;
    if (referenceMode === 'symmetrized') {
      const wrap01 = (x) => x - Math.floor(x);
      const bfConv = lab.tauFrac.map(f => vecMat3(f, P).map(wrap01));
      refFrac = new Float64Array(numAtoms * 3);
      for (let i = 0; i < numAtoms; i++) {
        const f = bfConv[typeIndices[i]];
        refFrac[i * 3] = f[0]; refFrac[i * 3 + 1] = f[1]; refFrac[i * 3 + 2] = f[2];
      }
      wrapRef = true;
    }

    return {
      parsedFrames, numFiles, numAtoms, hsym_xyz, firstFrameIds, reverseAtomDic,
      uniqueRN: tauRN, numTypes, typeIndices, masses, counts, segSymbols, basis, cellN,
      refFrac, wrapRef, P, cellL: lab.L, v_super, dim: baseStructure.dim, activeBatchSize: Math.min(batchSize, numFiles),
    };
  }

  // ── Ensemble-averaged S(k) at one kvec (radians/cell) ──────────────────────
  async _skAtKvec(kvec, prep) {
    const { parsedFrames, numFiles, numAtoms, refFrac, wrapRef, masses, typeIndices, counts, numTypes, cellN, v_super, dim, activeBatchSize } = prep;
    const D = 3 * numTypes;
    const Sk_real = new Float64Array(D * D);
    const Sk_imag = new Float64Array(D * D);
    for (let batchStart = 0; batchStart < numFiles; batchStart += activeBatchSize) {
      const currentBatchSize = Math.min(batchStart + activeBatchSize, numFiles) - batchStart;
      const dispBatch = new Float32Array(numAtoms * currentBatchSize * 3);
      const cellBatch = new Float32Array(numAtoms * currentBatchSize * 3);
      for (let b = 0; b < currentBatchSize; b++) {
        const frame = parsedFrames[batchStart + b];
        const offset = b * numAtoms * 3;
        for (let i = 0; i < numAtoms; i++) {
          const ix = i * 3, iy = i * 3 + 1, iz = i * 3 + 2;
          let rx = frame.xyz[ix] - refFrac[ix], ry = frame.xyz[iy] - refFrac[iy], rz = frame.xyz[iz] - refFrac[iz];
          if (wrapRef) { rx -= Math.round(rx); ry -= Math.round(ry); rz -= Math.round(rz); } // nearest-image (period 1, within-cell frac)
          const dx = rx / dim[0];
          const dy = ry / dim[1];
          const dz = rz / dim[2];
          dispBatch[offset + ix] = dx * v_super[0] + dy * v_super[3] + dz * v_super[6];
          dispBatch[offset + iy] = dx * v_super[1] + dy * v_super[4] + dz * v_super[7];
          dispBatch[offset + iz] = dx * v_super[2] + dy * v_super[5] + dz * v_super[8];
          // Bloch phase indexes the computation cell: n (in L units) from the
          // average structure — constant across frames (cells.buildCellLabeling).
          cellBatch[offset + ix] = cellN[ix];
          cellBatch[offset + iy] = cellN[iy];
          cellBatch[offset + iz] = cellN[iz];
        }
      }
      const r = await this.engine.computeBatch(kvec, dispBatch, cellBatch, masses, typeIndices, numTypes, currentBatchSize, counts);
      for (let i = 0; i < Sk_real.length; i++) { Sk_real[i] += r.Sk_real[i]; Sk_imag[i] += r.Sk_imag[i]; }
    }
    for (let i = 0; i < Sk_real.length; i++) { Sk_real[i] /= numFiles; Sk_imag[i] /= numFiles; }
    return { Sk_real, Sk_imag };
  }

  // ── Phonon band structure along a k-path ───────────────────────────────────
  async runCalculation(files, family, baseStructure, kPathPoints, temperature, batchSize = 50, options = {}) {
    const { degenerateTol = 5e-3 } = options;
    this._cancel = false;
    const prep = await this._prepare(files, family, baseStructure, options, batchSize);
    this.onProgress(35, 'Preparing WebGPU buffers...');

    const D = 3 * prep.numTypes;
    if (D > 600) console.warn(`[cells] large computation cell: ${prep.numTypes} basis sites → ${D}×${D} S(k); eigh is O(N³) and may be slow.`);
    const phononBands = [], phononEigvecs = [];
    const N = kPathPoints.length;
    // The user always picks the path on the CONVENTIONAL BZ. Map each point into
    // the computation cell's reciprocal before the Bloch phase: q_cell = P·q_conv
    // (so the phase q_cell·n matches the cell index n in L units). For P = I this
    // is the identity; for a custom supercell it folds, for the primitive cell it
    // unfolds. qPoints stored in the result stay conventional (for the plot axis).
    const P = prep.P;
    const toCell = (q) => [
      P[0][0] * q[0] + P[0][1] * q[1] + P[0][2] * q[2],
      P[1][0] * q[0] + P[1][1] * q[1] + P[1][2] * q[2],
      P[2][0] * q[0] + P[2][1] * q[1] + P[2][2] * q[2],
    ];
    for (let k = 0; k < N; k++) {
      if (this._cancel) throw new Error('cancelled');
      const q = toCell(kPathPoints[k]);
      this.onProgress(35 + (k / N) * 50, `Computing S(k) for k-point ${k + 1}/${N}...`);
      const { Sk_real, Sk_imag } = await this._skAtKvec([q[0] * TWO_PI_PHASE, q[1] * TWO_PI_PHASE, q[2] * TWO_PI_PHASE], prep);
      const { eigenvalues, eigenvectors } = eigh(Sk_real, Sk_imag, D);
      phononBands.push(eigenvaluesToMev(eigenvalues, temperature, ENERGY_CONV));
      phononEigvecs.push(eigenvectors);
    }

    this.onProgress(95, 'Connecting bands...');
    const { connectedBands, connectedEigvecs } = connectBands(phononBands, phononEigvecs, 2, 0.0, degenerateTol);
    this.onProgress(100, 'Done!');

    return {
      bands: connectedBands, eigvecs: connectedEigvecs, qPoints: kPathPoints, temperature,
      baseStructure: {
        ...baseStructure, hsym_xyz: prep.hsym_xyz, atomType: prep.firstFrameIds,
        cellIdx: prep.parsedFrames[0].cellIdx, uniqueRN: prep.uniqueRN, segSymbols: prep.segSymbols, counts: prep.counts,
        siteBasis: prep.basis, compCell: { P: prep.P, L: prep.cellL },
      },
    };
  }

  // ── Phonon DOS over a uniform q-grid (all eigenvalues) ─────────────────────
  async computeDOSGrid(files, family, baseStructure, gridN, temperature, batchSize = 50, options = {}) {
    this._cancel = false;
    const prep = await this._prepare(files, family, baseStructure, options, batchSize);
    const grid = genGrid(gridN);
    const nq = grid.length;
    const D = 3 * prep.numTypes;
    const energies = new Float64Array(nq * D);
    for (let g = 0; g < nq; g++) {
      if (this._cancel) throw new Error('cancelled');
      if (g % 4 === 0) this.onProgress(35 + (g / nq) * 60, `DOS grid point ${g + 1}/${nq} (${gridN}³)...`);
      const q = grid[g];
      const { Sk_real, Sk_imag } = await this._skAtKvec([q[0] * TWO_PI_PHASE, q[1] * TWO_PI_PHASE, q[2] * TWO_PI_PHASE], prep);
      const { eigenvalues } = eigh(Sk_real, Sk_imag, D);
      energies.set(eigenvaluesToMev(eigenvalues, temperature, ENERGY_CONV), g * D);
    }
    this.onProgress(100, 'DOS done');
    return { energies, gridN, nq, nModes: D };
  }

  /**
   * Ensemble-average basis for symmetry detection: one representative site per
   * reference number, circular-averaged over ALL cells and a sample of configs
   * (within-cell fractional). Much cleaner than the single-config
   * baseStructure.basis, so the detected symmetry shows at a tight tolerance.
   * @returns {Promise<{rn:number, el:string, frac:number[]}[]>} sorted by rn.
   */
  async computeAverageBasis(files, family, baseStructure, sampleSize = 60) {
    const sample = files.slice(0, Math.min(sampleSize, files.length));
    const frames = await Promise.all(sample.map(f => this.parseFile(f, family, baseStructure.atomDic, baseStructure.dim, 0)));
    if (!frames.length) return [];
    const numAtoms = frames[0].xyz.length / 3;
    const rnIds = frames[0].atomType;
    const reverseAtomDic = {};
    for (const [sym, idxs] of Object.entries(baseStructure.atomDic)) idxs.forEach(idx => { reverseAtomDic[idx] = sym; });
    // Circular mean of the within-cell fraction, pooled over every atom of each
    // reference number × every frame (handles the 0/1 boundary).
    const TAU = 2 * Math.PI;
    const acc = new Map();   // rn -> { sc:[3], ss:[3], el }
    for (const fr of frames) {
      for (let i = 0; i < numAtoms; i++) {
        const rn = rnIds[i];
        let a = acc.get(rn);
        if (!a) { a = { sc: [0, 0, 0], ss: [0, 0, 0], el: reverseAtomDic[rn] }; acc.set(rn, a); }
        for (let c = 0; c < 3; c++) { const ph = TAU * fr.xyz[i * 3 + c]; a.sc[c] += Math.cos(ph); a.ss[c] += Math.sin(ph); }
      }
    }
    const wrap01 = (x) => x - Math.floor(x);
    return [...acc.entries()].sort((p, q) => p[0] - q[0]).map(([rn, a]) => ({
      rn, el: a.el, frac: [0, 1, 2].map(c => wrap01(Math.atan2(a.ss[c], a.sc[c]) / TAU)),
    }));
  }
}
