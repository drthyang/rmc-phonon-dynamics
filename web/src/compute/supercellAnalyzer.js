// web/src/compute/supercellAnalyzer.js
//
// Browser-universal streaming supercell symmetry & domain analyzer.
// Scales smoothly from 1 dropped .rmc6f file to 500+ RMC ensemble configs
// in O(1) memory per config via StreamingSymmetryAccumulator, yielding between
// configs so the UI remains responsive with live progress & cancellation.

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
 * Analyze an array of already-parsed frames synchronously via StreamingSymmetryAccumulator.
 */
export function analyzeSupercellFrames(frames, baseStructure, options = {}) {
  const { parentTol = 0.25, nNullSamples = 128 } = options;
  if (!frames?.length) throw new Error('No configurations to analyze.');

  const dim = baseStructure.dim;
  const Aconv = conventionalLattice(baseStructure.v1, baseStructure.v2, baseStructure.v3, dim);
  const refBasis = buildCircularRefBasis(frames[0], baseStructure);
  const rnToSite = new Map(refBasis.map((b, idx) => [b.rn, idx]));

  // Allow zero-displacement single frame so initial .rmc6f files still preview gracefully
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

  // Fallback if every frame was zero-displacement
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

/**
 * Stream 1..N File or FileSystemFileHandle objects in O(1) memory with progress & cancellation.
 * Automatically skips unconverged (0 accepted moves) configs when multiple files are provided.
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

  const dim = baseStructure.dim;
  const Aconv = conventionalLattice(baseStructure.v1, baseStructure.v2, baseStructure.v3, dim);

  let acc = null;
  let rnToSite = null;
  const skippedHeader = [];
  const multiMode = sample.length > 1;

  for (let i = 0; i < sample.length; i++) {
    if (signal?.aborted) throw new Error('Cancelled');
    const item = sample[i];
    const file = typeof item.getFile === 'function' ? await item.getFile() : item;
    const name = item.name || file.name || `Config ${i + 1}`;

    if (onProgress) onProgress(i, sample.length, `Streaming ${i + 1}/${sample.length} · ${name}`);

    const text = await file.text();
    if (multiMode && family !== 'frac' && isZeroMoveRmc6f(text)) {
      skippedHeader.push({ name, rms: 0, reason: '0 accepted moves (unconverged)' });
      continue;
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

    // Yield to browser UI every config so progress bar & Cancel stay responsive
    if (multiMode) await new Promise(r => setTimeout(r, 0));
  }

  // If every single config in a multi-file selection was unconverged, fall back to analyzing the first file
  if (!acc || acc.nConfigs === 0) {
    const item0 = sample[0];
    const f0 = typeof item0.getFile === 'function' ? await item0.getFile() : item0;
    const t0 = await f0.text();
    const fr0 = family === 'frac' ? parseFracFrame(t0, dim) : parseRmc6fFrame(t0, dim);
    const refBasis = buildCircularRefBasis(fr0, baseStructure);
    const map0 = new Map(refBasis.map((b, idx) => [b.rn, idx]));
    const acc0 = new StreamingSymmetryAccumulator(dim, Aconv, refBasis, {
      parentTol,
      nNullSamples,
      minConfigRms: 0,
    });
    acc0.addConfig(frameToDisplacements(fr0, dim, Aconv, acc0.idealParentBasis, map0), f0.name || 'Config 1');
    const out0 = acc0.finalize();
    out0.skippedConfigs = [...skippedHeader, ...(out0.skippedConfigs || [])];
    if (onProgress) onProgress(sample.length, sample.length, 'Done');
    return out0;
  }

  if (onProgress) onProgress(sample.length, sample.length, 'Finalizing ensemble statistics…');
  const res = acc.finalize();
  res.skippedConfigs = [...skippedHeader, ...(res.skippedConfigs || [])];
  res.totalAvailableConfigs = files.length;
  return res;
}

/**
 * Instant subgroup selection: returns the pre-streamed ensemble-aligned structure
 * directly from `result.alignedBySubgroup` (0 ms, works for 500+ configs!).
 */
export function alignSubgroupFromResult(result, subgroupSpec) {
  const sgName = typeof subgroupSpec === 'string' ? subgroupSpec : subgroupSpec?.spaceGroup;
  if (result?.alignedBySubgroup?.[sgName]) {
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
