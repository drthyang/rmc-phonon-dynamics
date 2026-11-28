// web/src/compute/supercellAnalyzer.js
//
// Browser-universal streaming supercell symmetry & domain analyzer.
//
// Architecture & Memory Backpressure
// ----------------------------------
// Scales from 1 `.rmc6f` supercell to 500+ RMC ensemble configurations in O(1) RAM:
//   * In browser contexts (`typeof Worker !== 'undefined'`), `analyzeSupercellFiles`
//     spawns a dedicated module worker (`io/symmetryWorker.js`) and streams one
//     configuration text at a time with explicit `ack` backpressure (so at most 1
//     raw file string resides in memory at any instant while all 3D FFT, Monte Carlo
//     null calibration, and subgroup/domain projections run off the main UI thread!).
//     Aborting terminates the worker immediately in <1 ms.
//   * In Node.js test environments (where `Worker` is absent), transparently executes
//     the identical `createStreamingSymmetrySession` state machine in-process.

import { conventionalLattice } from '../math/reciprocal.js';
import { generateAlignedCIF } from '../math/supercell_symmetry.js';
import { StreamingSymmetryAccumulator } from '../math/ensemble_symmetry.js';
import { parseRmc6fFrame, parseFracFrame } from '../io/readers.js';

const TAU = 2 * Math.PI;
const wrap01 = (x) => ((x % 1) + 1) % 1;
const wrapHalf = (x) => x - Math.round(x);

function buildCircularRefBasis(frame, baseStructure) {
  const numAtoms = frame.xyz.length / 3;
  const reverseAtomDic = {};
  for (const [sym, idxs] of Object.entries(baseStructure.atomDic)) {
    idxs.forEach(idx => { reverseAtomDic[idx] = sym; });
  }
  const acc = new Map();
  for (let i = 0; i < numAtoms; i++) {
    const rn = frame.atomType[i];
    let a = acc.get(rn);
    if (!a) { a = { sc: [0, 0, 0], ss: [0, 0, 0], el: reverseAtomDic[rn] }; acc.set(rn, a); }
    for (let c = 0; c < 3; c++) {
      const ph = TAU * frame.xyz[i * 3 + c];
      a.sc[c] += Math.cos(ph);
      a.ss[c] += Math.sin(ph);
    }
  }
  return [...acc.entries()]
    .sort((p, q) => p[0] - q[0])
    .map(([rn, a]) => ({
      rn,
      el: a.el,
      frac: [0, 1, 2].map(c => wrap01(Math.atan2(a.ss[c], a.sc[c]) / TAU)),
    }));
}

function frameToDisplacements(frame, dim, Aconv, refBasis, rnToSite) {
  const [N1, N2, N3] = dim;
  const nCells = N1 * N2 * N3;
  const D = 3 * refBasis.length;
  const numAtoms = frame.xyz.length / 3;
  const u = new Float64Array(nCells * D);
  for (let i = 0; i < numAtoms; i++) {
    const s = rnToSite.get(frame.atomType[i]);
    if (s === undefined) continue;
    const nx = ((Math.round(frame.cellIdx[i * 3]) % N1) + N1) % N1;
    const ny = ((Math.round(frame.cellIdx[i * 3 + 1]) % N2) + N2) % N2;
    const nz = ((Math.round(frame.cellIdx[i * 3 + 2]) % N3) + N3) % N3;
    const rf = refBasis[s].frac;
    const dfx = wrapHalf(frame.xyz[i * 3] - rf[0]);
    const dfy = wrapHalf(frame.xyz[i * 3 + 1] - rf[1]);
    const dfz = wrapHalf(frame.xyz[i * 3 + 2] - rf[2]);
    const base = ((nx * N2 + ny) * N3 + nz) * D + s * 3;
    u[base]     = dfx * Aconv[0][0] + dfy * Aconv[1][0] + dfz * Aconv[2][0];
    u[base + 1] = dfx * Aconv[0][1] + dfy * Aconv[1][1] + dfz * Aconv[2][1];
    u[base + 2] = dfx * Aconv[0][2] + dfy * Aconv[1][2] + dfz * Aconv[2][2];
  }
  return u;
}

function isZeroMoveRmc6f(text) {
  const m = text.match(/Number of moves accepted:\s+(\d+)/i);
  return m ? parseInt(m[1], 10) === 0 : false;
}

/**
 * Shared O(1)-memory streaming session used identically by `symmetryWorker.js` and
 * in-thread Node.js execution.
 */
export function createStreamingSymmetrySession(family, baseStructure, options = {}) {
  const { parentTol = 0.25, nNullSamples = 128, multiMode = false } = options;
  const dim = baseStructure.dim;
  const Aconv = conventionalLattice(baseStructure.v1, baseStructure.v2, baseStructure.v3, dim);

  let acc = null;
  let rnToSite = null;
  let firstText = null;
  let firstName = null;
  const skippedHeader = [];

  return {
    pushText(text, name) {
      if (firstText === null) {
        firstText = text;
        firstName = name;
      }
      if (multiMode && family !== 'frac' && isZeroMoveRmc6f(text)) {
        skippedHeader.push({ name, rms: 0, reason: '0 accepted moves (unconverged)' });
        return { accepted: false };
      }

      const frame = family === 'frac' ? parseFracFrame(text, dim) : parseRmc6fFrame(text, dim);
      if (!acc) {
        const refBasis = buildCircularRefBasis(frame, baseStructure);
        rnToSite = new Map(refBasis.map((b, idx) => [b.rn, idx]));
        acc = new StreamingSymmetryAccumulator(dim, Aconv, refBasis, {
          parentTol,
          nNullSamples,
          minConfigRms: multiMode ? 1e-3 : 0,
        });
      }
      const u = frameToDisplacements(frame, dim, Aconv, acc.idealParentBasis, rnToSite);
      acc.addConfig(u, name);
      return { accepted: true, nConfigs: acc.nConfigs };
    },

    finalize(totalAvailableConfigs = 1) {
      if (!acc || acc.nConfigs === 0) {
        if (firstText === null) throw new Error('No configuration files streamed.');
        const fr0 = family === 'frac' ? parseFracFrame(firstText, dim) : parseRmc6fFrame(firstText, dim);
        const refBasis = buildCircularRefBasis(fr0, baseStructure);
        const map0 = new Map(refBasis.map((b, idx) => [b.rn, idx]));
        const acc0 = new StreamingSymmetryAccumulator(dim, Aconv, refBasis, {
          parentTol,
          nNullSamples,
          minConfigRms: 0,
        });
        acc0.addConfig(frameToDisplacements(fr0, dim, Aconv, acc0.idealParentBasis, map0), firstName || 'Config 1');
        const out0 = acc0.finalize();
        out0.skippedConfigs = [...skippedHeader, ...(out0.skippedConfigs || [])];
        out0.totalAvailableConfigs = totalAvailableConfigs;
        return out0;
      }

      const res = acc.finalize();
      res.skippedConfigs = [...skippedHeader, ...(res.skippedConfigs || [])];
      res.totalAvailableConfigs = totalAvailableConfigs;
      return res;
    },
  };
}

/**
 * Analyze an array of already-parsed frames synchronously via StreamingSymmetryAccumulator.
 */
export function analyzeSupercellFrames(frames, baseStructure, options = {}) {
  const { parentTol = 0.25, nNullSamples = 128 } = options;
  if (!frames?.length) throw new Error('No configurations to analyze.');

  const dim = baseStructure.dim;
  const Aconv = conventionalLattice(baseStructure.v1, baseStructure.v2, baseStructure.v3, dim);
  const refBasis = buildCircularRefBasis(frames[0], baseStructure);
  const rnToSite = new Map(refBasis.map((b, idx) => [b.rn, idx]));

  const minConfigRms = frames.length > 1 ? 1e-3 : 0;
  const acc = new StreamingSymmetryAccumulator(dim, Aconv, refBasis, {
    parentTol,
    nNullSamples,
    minConfigRms,
  });

  for (let i = 0; i < frames.length; i++) {
    const u = frameToDisplacements(frames[i], dim, Aconv, acc.idealParentBasis, rnToSite);
    acc.addConfig(u, `Config ${i + 1}`);
  }

  if (acc.nConfigs === 0) {
    const acc0 = new StreamingSymmetryAccumulator(dim, Aconv, refBasis, {
      parentTol,
      nNullSamples,
      minConfigRms: 0,
    });
    for (let i = 0; i < frames.length; i++) {
      const u = frameToDisplacements(frames[i], dim, Aconv, acc0.idealParentBasis, rnToSite);
      acc0.addConfig(u, `Config ${i + 1}`);
    }
    return acc0.finalize();
  }

  return acc.finalize();
}

async function analyzeSupercellFilesInWorker(sample, totalFilesCount, family, baseStructure, options) {
  const { parentTol = 0.25, nNullSamples = 128, onProgress = null, signal = null } = options;
  const multiMode = sample.length > 1;
  const worker = new Worker(new URL('../io/symmetryWorker.js', import.meta.url), { type: 'module' });

  let settled = false;
  let pendingResolve = null;
  let pendingReject = null;

  const cleanup = () => {
    settled = true;
    try { worker.terminate(); } catch { /* ignore */ }
  };

  const waitNextMessage = () => new Promise((resolve, reject) => {
    pendingResolve = resolve;
    pendingReject = reject;
  });

  worker.onmessage = (ev) => {
    const msg = ev.data;
    if (msg.type === 'error') {
      if (pendingReject) pendingReject(new Error(msg.error || 'Symmetry worker failed'));
      pendingResolve = null;
      pendingReject = null;
      return;
    }
    if (pendingResolve) {
      const res = pendingResolve;
      pendingResolve = null;
      pendingReject = null;
      res(msg);
    }
  };

  worker.onerror = (err) => {
    if (pendingReject) pendingReject(new Error(err?.message || 'Symmetry worker crashed'));
    pendingResolve = null;
    pendingReject = null;
  };

  const onAbort = () => {
    cleanup();
    if (pendingReject) pendingReject(new Error('Cancelled'));
  };
  if (signal) {
    if (signal.aborted) { cleanup(); throw new Error('Cancelled'); }
    signal.addEventListener('abort', onAbort, { once: true });
  }

  try {
    const slimBase = {
      v1: baseStructure.v1,
      v2: baseStructure.v2,
      v3: baseStructure.v3,
      dim: baseStructure.dim,
      atomDic: baseStructure.atomDic,
    };
    const initWait = waitNextMessage();
    worker.postMessage({
      type: 'init',
      family,
      baseStructure: slimBase,
      options: { parentTol, nNullSamples, multiMode },
    });
    await initWait;

    for (let i = 0; i < sample.length; i++) {
      if (signal?.aborted || settled) throw new Error('Cancelled');
      const item = sample[i];
      const file = typeof item.getFile === 'function' ? await item.getFile() : item;
      const name = item.name || file.name || `Config ${i + 1}`;
      if (onProgress) onProgress(i, sample.length, `Streaming ${i + 1}/${sample.length} · ${name}`);

      const text = await file.text();
      if (signal?.aborted || settled) throw new Error('Cancelled');

      const ackWait = waitNextMessage();
      worker.postMessage({ type: 'frame', text, name });
      await ackWait;
    }

    if (onProgress) onProgress(sample.length, sample.length, 'Finalizing ensemble statistics…');
    const finWait = waitNextMessage();
    worker.postMessage({ type: 'finalize', totalAvailableConfigs: totalFilesCount });
    const doneMsg = await finWait;
    return doneMsg.result;
  } finally {
    if (signal) signal.removeEventListener('abort', onAbort);
    cleanup();
  }
}

/**
 * Stream 1..N File or FileSystemFileHandle objects in O(1) memory with progress & cancellation.
 * Uses a dedicated background Web Worker in browsers and falls back to in-thread execution in Node.
 */
export async function analyzeSupercellFiles(files, family, baseStructure, options = {}) {
  const {
    sampleSize = 25,
    parentTol = 0.25,
    nNullSamples = 128,
    onProgress = null,
    signal = null,
  } = options;

  const limit = Number.isFinite(sampleSize) && sampleSize > 0
    ? Math.min(sampleSize, files.length)
    : files.length;
  const sample = files.slice(0, limit);
  if (!sample.length) throw new Error('No configuration files selected.');

  if (typeof Worker !== 'undefined') {
    return analyzeSupercellFilesInWorker(sample, files.length, family, baseStructure, options);
  }

  const multiMode = sample.length > 1;
  const session = createStreamingSymmetrySession(family, baseStructure, {
    parentTol,
    nNullSamples,
    multiMode,
  });

  for (let i = 0; i < sample.length; i++) {
    if (signal?.aborted) throw new Error('Cancelled');
    const item = sample[i];
    const file = typeof item.getFile === 'function' ? await item.getFile() : item;
    const name = item.name || file.name || `Config ${i + 1}`;
    if (onProgress) onProgress(i, sample.length, `Streaming ${i + 1}/${sample.length} · ${name}`);
    session.pushText(await file.text(), name);
    if (multiMode) await new Promise(r => setTimeout(r, 0));
  }

  if (onProgress) onProgress(sample.length, sample.length, 'Finalizing ensemble statistics…');
  return session.finalize(files.length);
}

/**
 * Instant subgroup selection: returns the pre-streamed ensemble-aligned structure
 * directly from `result.alignedBySubgroup` (0 ms, works for 500+ configs!).
 */
export function alignSubgroupFromResult(result, subgroupSpec) {
  if (!result?.alignedBySubgroup) return result?.aligned || null;
  if (typeof subgroupSpec === 'object' && subgroupSpec?.id && result.alignedBySubgroup[subgroupSpec.id]) {
    return result.alignedBySubgroup[subgroupSpec.id];
  }
  const sgName = typeof subgroupSpec === 'string' ? subgroupSpec : subgroupSpec?.spaceGroup;
  if (sgName && result.alignedBySubgroup[sgName]) {
    return result.alignedBySubgroup[sgName];
  }
  return result?.aligned || null;
}

export function buildAlignedCifText(aligned, { projected = true, title } = {}) {
  return generateAlignedCIF(aligned, {
    symmetrized: projected,
    title: title || `aligned_${aligned.targetSpaceGroup || 'local'}`,
  });
}
