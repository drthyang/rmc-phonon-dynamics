import { ComputeEngine } from './engine';
import { eigh, eigenvaluesToMev } from '../math/diagonalize';
import { connectBands } from '../math/band_connection';
import { ATOMIC_MASS, ENERGY_CONV, TWO_PI_PHASE } from '../constants';

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
      this.workerPool.push(
        new Worker(new URL('../io/worker.js', import.meta.url), { type: 'module' })
      );
    }
    await this.engine.init();
  }

  parseFile(fileHandle, family, atomDic, dim, atype) {
    return new Promise((resolve, reject) => {
      const worker = this.workerPool.shift(); // take available
      if (!worker) {
        // queue logic or just wait
        setTimeout(() => {
          this.parseFile(fileHandle, family, atomDic, dim, atype).then(resolve).catch(reject);
        }, 50);
        return;
      }
      worker.onmessage = (e) => {
        this.workerPool.push(worker); // return to pool
        if (e.data.success) {
          resolve(e.data);
        } else {
          reject(new Error(e.data.error));
        }
      };
      worker.postMessage({ fileHandle, family, atomDic, dim, atype });
    });
  }

  async runCalculation(files, family, baseStructure, kPathPoints, temperature, batchSize = 50, options = {}) {
    const { referenceHandle = null, degenerateTol = 5e-3 } = options;
    this._cancel = false;
    const numFiles = files.length;

    // 1. Parse configs — reuse the cache if the same dataset was just parsed, so
    // changing the k-path / temperature / reference only re-runs S(k), not the
    // slow file parse (browser equivalent of the legacy Sk cache benefit).
    const els = Object.keys(baseStructure.atomDic).join(',');
    const cacheKey = `${family}|${numFiles}|${files[0]?.name}|${files[numFiles - 1]?.name}|${baseStructure.dim?.join('x')}|${els}`;
    let parsedFrames;
    if (this._cache && this._cache.key === cacheKey) {
      this.onProgress(30, `Reusing ${numFiles} parsed configs (cached)...`);
      parsedFrames = this._cache.parsedFrames;
    } else {
      this.onProgress(5, "Parsing configurations...");
      let parsedCount = 0;
      const promises = files.map(async (file) => {
        const data = await this.parseFile(file, family, baseStructure.atomDic, baseStructure.dim, 0);
        parsedCount++;
        if (parsedCount % 20 === 0) {
          this.onProgress(5 + (parsedCount / numFiles) * 25, `Parsed ${parsedCount} / ${numFiles} frames...`);
        }
        return data;
      });
      parsedFrames = await Promise.all(promises);
      this._cache = { key: cacheKey, parsedFrames }; // keep only the latest dataset
    }
    this.onProgress(30, "Computing average structure (hsym)...");

    if (parsedFrames.length === 0) throw new Error("No valid frames parsed.");

    const numAtoms = parsedFrames[0].xyz.length / 3;
    const hsym_xyz = new Float64Array(numAtoms * 3);

    if (referenceHandle) {
      // Displacement reference from a chosen equilibrium .rmc6f, verified to
      // share the configs' atom (RN) + cell-index layout (legacy _hsym_from_file).
      this.onProgress(30, "Reading equilibrium reference file...");
      const ref = await this.parseFile(referenceHandle, 'rmc6f', baseStructure.atomDic, baseStructure.dim, 0);
      if (ref.xyz.length !== numAtoms * 3) throw new Error("Reference file atom count does not match the configurations.");
      const a0 = parsedFrames[0].atomType, c0 = parsedFrames[0].cellIdx;
      for (let i = 0; i < numAtoms; i++) {
        if (ref.atomType[i] !== a0[i] || ref.cellIdx[i * 3] !== c0[i * 3] || ref.cellIdx[i * 3 + 1] !== c0[i * 3 + 1] || ref.cellIdx[i * 3 + 2] !== c0[i * 3 + 2]) {
          throw new Error("Reference structure atom layout (RN / cell indices) does not match the configurations.");
        }
      }
      for (let j = 0; j < hsym_xyz.length; j++) hsym_xyz[j] = ref.xyz[j];
    } else {
      for (let i = 0; i < parsedFrames.length; i++) {
        const xyz = parsedFrames[i].xyz;
        for (let j = 0; j < xyz.length; j++) hsym_xyz[j] += xyz[j];
      }
      for (let j = 0; j < hsym_xyz.length; j++) hsym_xyz[j] /= parsedFrames.length;
    }

    this.onProgress(35, "Preparing WebGPU buffers...");

    // ── Group atoms by crystallographic BASIS SITE (RMC reference number),
    //    NOT by chemical element. This mirrors the legacy
    //    `np.unique(atype_static, return_inverse=True)` in src_gpu/Calculators.Sk_avg:
    //    each unique reference number is one phonon "site", giving
    //    3 * N_basis_sites modes (e.g. 156 for GTS_5K, not 9). Segment order is
    //    sorted-by-reference-number so eigenvector rows match the band.yaml
    //    `rank` (sorted reference numbers).
    const firstFrameIds = parsedFrames[0].atomType; // [N] reference numbers, one per atom

    // RN -> element symbol (for masses)
    const reverseAtomDic = {};
    for (const [symbol, idxs] of Object.entries(baseStructure.atomDic)) {
      idxs.forEach(idx => { reverseAtomDic[idx] = symbol; });
    }

    // Sorted unique reference numbers == basis sites (np.unique semantics).
    const uniqueRN = Array.from(new Set(firstFrameIds)).sort((a, b) => a - b);
    const numTypes = uniqueRN.length;
    const rnToSeg = new Map();
    uniqueRN.forEach((rn, seg) => rnToSeg.set(rn, seg));

    const typeIndices = new Uint32Array(numAtoms);
    const masses = new Float32Array(numAtoms);
    const counts = new Float32Array(numTypes);

    for (let i = 0; i < numAtoms; i++) {
      const rn = firstFrameIds[i];
      const seg = rnToSeg.get(rn);
      typeIndices[i] = seg;
      counts[seg] += 1;
      masses[i] = ATOMIC_MASS[reverseAtomDic[rn]] || 0.0;
    }

    // Per-segment element symbol + reference number, in eigenvector-row order
    // (used by band.yaml / 3D viewer / INS to interpret eigenvector rows).
    const segSymbols = uniqueRN.map(rn => reverseAtomDic[rn]);

    // Prepare batched displacement buffers (N * BatchSize * 3)
    const activeBatchSize = Math.min(batchSize, numFiles);
    
    // We need the v_super matrix flattened: [v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z]
    const v_super = [
      baseStructure.v1[0], baseStructure.v1[1], baseStructure.v1[2],
      baseStructure.v2[0], baseStructure.v2[1], baseStructure.v2[2],
      baseStructure.v3[0], baseStructure.v3[1], baseStructure.v3[2],
    ];
    const dim = baseStructure.dim;

    const phononBands = [];
    const phononEigvecs = [];
    const totalKPoints = kPathPoints.length;

    for (let k = 0; k < totalKPoints; k++) {
      if (this._cancel) throw new Error('cancelled');
      // kPathPoints are CONVENTIONAL-cell fractional reciprocal coords (q_frac).
      // src_gpu's Bloch phase needs radians per cell: kvec = 2*pi * q_frac
      // (TWO_PI_PHASE). This factor was missing before — without it S(G) != S(Gamma).
      const qfrac = kPathPoints[k];
      const kvec = [qfrac[0] * TWO_PI_PHASE, qfrac[1] * TWO_PI_PHASE, qfrac[2] * TWO_PI_PHASE];
      this.onProgress(35 + (k / totalKPoints) * 50, `Computing S(k) for k-point ${k + 1}/${totalKPoints}...`);

      let global_Sk_real = new Float64Array(3 * numTypes * 3 * numTypes);
      let global_Sk_imag = new Float64Array(3 * numTypes * 3 * numTypes);

      for (let batchStart = 0; batchStart < numFiles; batchStart += activeBatchSize) {
        const batchEnd = Math.min(batchStart + activeBatchSize, numFiles);
        const currentBatchSize = batchEnd - batchStart;

        const dispBatch = new Float32Array(numAtoms * currentBatchSize * 3);
        const cellBatch = new Float32Array(numAtoms * currentBatchSize * 3);

        for (let b = 0; b < currentBatchSize; b++) {
          const frame = parsedFrames[batchStart + b];
          const offset = b * numAtoms * 3;
          
          for (let i = 0; i < numAtoms; i++) {
            const idxX = i * 3 + 0;
            const idxY = i * 3 + 1;
            const idxZ = i * 3 + 2;

            // disp = (config - hsym) / dim @ v_super
            const dx = (frame.xyz[idxX] - hsym_xyz[idxX]) / dim[0];
            const dy = (frame.xyz[idxY] - hsym_xyz[idxY]) / dim[1];
            const dz = (frame.xyz[idxZ] - hsym_xyz[idxZ]) / dim[2];

            const cartX = dx * v_super[0] + dy * v_super[3] + dz * v_super[6];
            const cartY = dx * v_super[1] + dy * v_super[4] + dz * v_super[7];
            const cartZ = dx * v_super[2] + dy * v_super[5] + dz * v_super[8];

            dispBatch[offset + idxX] = cartX;
            dispBatch[offset + idxY] = cartY;
            dispBatch[offset + idxZ] = cartZ;

            cellBatch[offset + idxX] = frame.cellIdx[idxX];
            cellBatch[offset + idxY] = frame.cellIdx[idxY];
            cellBatch[offset + idxZ] = frame.cellIdx[idxZ];
          }
        }

        const { Sk_real, Sk_imag } = await this.engine.computeBatch(
          kvec, dispBatch, cellBatch, masses, typeIndices, numTypes, currentBatchSize, counts
        );

        for (let i = 0; i < Sk_real.length; i++) {
          global_Sk_real[i] += Sk_real[i];
          global_Sk_imag[i] += Sk_imag[i];
        }
      }

      for (let i = 0; i < global_Sk_real.length; i++) {
        global_Sk_real[i] /= numFiles;
        global_Sk_imag[i] /= numFiles;
      }

      const { eigenvalues, eigenvectors } = eigh(global_Sk_real, global_Sk_imag, 3 * numTypes);
      const energies = eigenvaluesToMev(eigenvalues, temperature, ENERGY_CONV);
      
      phononBands.push(energies);
      phononEigvecs.push(eigenvectors);
    }

    this.onProgress(95, "Connecting bands...");
    const { connectedBands, connectedEigvecs } = connectBands(phononBands, phononEigvecs, 2, 0.0, degenerateTol);

    this.onProgress(100, "Done!");
    // Attach derived structure metadata so the 3D viewer / band.yaml / INS can
    // interpret eigenvector rows (which are ordered by sorted reference number).
    const enrichedStructure = {
      ...baseStructure,
      hsym_xyz,                       // within-cell xyz*dim, atom order of frame 0
      atomType: firstFrameIds,        // reference number per atom (frame 0 order)
      cellIdx: parsedFrames[0].cellIdx, // integer cell index per atom (frame 0 order)
      uniqueRN,                       // sorted unique reference numbers (= eigvec row order)
      segSymbols,                     // element symbol per eigenvector site
      counts,                         // atom count per site
    };
    return {
      bands: connectedBands,
      eigvecs: connectedEigvecs,
      baseStructure: enrichedStructure,
      qPoints: kPathPoints,           // fractional q (NOT scaled by 2*pi)
      temperature,
    };
  }
}
