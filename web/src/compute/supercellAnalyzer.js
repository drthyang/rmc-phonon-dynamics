// web/src/compute/supercellAnalyzer.js
//
// Browser-universal supercell symmetry analyzer (no WebGPU or worker pool needed).
// Works directly on 1 dropped .rmc6f file OR a sample of files from a folder.

import { conventionalLattice } from '../math/reciprocal.js';
import {
  kSpaceModeSpectrum,
  localSubgroupAnalysis,
  statisticalSymmetryTest,
  filteredMinimalSupercell,
  domainAlignedStructure,
  generateAlignedCIF,
} from '../math/supercell_symmetry.js';
import { parseRmc6fFrame, parseFracFrame } from '../io/readers.js';

/**
 * Run direct supercell symmetry analysis on an array of already-parsed frames
 * ({ xyz, atomType, cellIdx }).
 */
export function analyzeSupercellFrames(frames, baseStructure, options = {}) {
  const { parentTol = 0.25 } = options;
  if (!frames?.length) throw new Error('No configurations to analyze.');

  const dim = baseStructure.dim;
  const [N1, N2, N3] = dim;
  const nCells = N1 * N2 * N3;
  const Aconv = conventionalLattice(baseStructure.v1, baseStructure.v2, baseStructure.v3, dim);

  const numAtoms = frames[0].xyz.length / 3;
  const reverseAtomDic = {};
  for (const [sym, idxs] of Object.entries(baseStructure.atomDic)) {
    idxs.forEach(idx => { reverseAtomDic[idx] = sym; });
  }

  // Circular-mean parent basis across all unit cells & frames
  const TAU = 2 * Math.PI;
  const acc = new Map();
  for (const fr of frames) {
    for (let i = 0; i < numAtoms; i++) {
      const rn = fr.atomType[i];
      let a = acc.get(rn);
      if (!a) { a = { sc: [0, 0, 0], ss: [0, 0, 0], el: reverseAtomDic[rn] }; acc.set(rn, a); }
      for (let c = 0; c < 3; c++) {
        const ph = TAU * fr.xyz[i * 3 + c];
        a.sc[c] += Math.cos(ph);
        a.ss[c] += Math.sin(ph);
      }
    }
  }
  const wrap01 = (x) => x - Math.floor(x);
  const wrapHalf = (x) => x - Math.round(x);
  const refBasis = [...acc.entries()]
    .sort((p, q) => p[0] - q[0])
    .map(([rn, a]) => ({
      rn,
      el: a.el,
      frac: [0, 1, 2].map(c => wrap01(Math.atan2(a.ss[c], a.sc[c]) / TAU)),
    }));

  const nBasis = refBasis.length;
  const D = 3 * nBasis;
  const rnToSite = new Map(refBasis.map((b, idx) => [b.rn, idx]));

  const configs = frames.map(fr => {
    const u = new Float64Array(nCells * D);
    for (let i = 0; i < numAtoms; i++) {
      const s = rnToSite.get(fr.atomType[i]);
      if (s === undefined) continue;
      const nx = ((Math.round(fr.cellIdx[i * 3]) % N1) + N1) % N1;
      const ny = ((Math.round(fr.cellIdx[i * 3 + 1]) % N2) + N2) % N2;
      const nz = ((Math.round(fr.cellIdx[i * 3 + 2]) % N3) + N3) % N3;
      const rf = refBasis[s].frac;
      const dfx = wrapHalf(fr.xyz[i * 3] - rf[0]);
      const dfy = wrapHalf(fr.xyz[i * 3 + 1] - rf[1]);
      const dfz = wrapHalf(fr.xyz[i * 3 + 2] - rf[2]);
      const base = ((nx * N2 + ny) * N3 + nz) * D + s * 3;
      u[base]     = dfx * Aconv[0][0] + dfy * Aconv[1][0] + dfz * Aconv[2][0];
      u[base + 1] = dfx * Aconv[0][1] + dfy * Aconv[1][1] + dfz * Aconv[2][1];
      u[base + 2] = dfx * Aconv[0][2] + dfy * Aconv[1][2] + dfz * Aconv[2][2];
    }
    return u;
  });

  const spectrum = kSpaceModeSpectrum(configs, dim, refBasis);
  const local = localSubgroupAnalysis(configs, dim, Aconv, refBasis, { parentTol });
  const statistical = statisticalSymmetryTest(configs, dim, Aconv, refBasis, { parentTol });
  const minimal = filteredMinimalSupercell(spectrum, dim, Aconv, refBasis, {
    snrThreshold: 2.5,
    minCoherentFrac: 0.1,
  });

  const ctx = { configs, dim, Aconv, refBasis, parentTol };

  const topBrk = local.subgroups
    .filter(s => s.breakingDoF > 0 && s.dof < local.internalDoF)
    .sort((a, b) => b.breakingFStat - a.breakingFStat)[0];
  const aligned = topBrk
    ? domainAlignedStructure(configs, dim, Aconv, refBasis, topBrk, { parentTol })
    : null;

  return {
    dim,
    nConfigs: configs.length,
    nBasis,
    Aconv,
    refBasis,
    spectrum,
    local,
    statistical,
    minimal,
    aligned,
    ctx,
  };
}

/**
 * Convenience wrapper: read 1..N File or FileSystemFileHandle objects and run analysis.
 */
export async function analyzeSupercellFiles(files, family, baseStructure, options = {}) {
  const { sampleSize = 10, parentTol = 0.25 } = options;
  const sample = files.slice(0, Math.min(sampleSize, files.length));
  const frames = [];
  for (const item of sample) {
    const file = typeof item.getFile === 'function' ? await item.getFile() : item;
    const text = await file.text();
    frames.push(family === 'frac'
      ? parseFracFrame(text, baseStructure.dim)
      : parseRmc6fFrame(text, baseStructure.dim));
  }
  return analyzeSupercellFrames(frames, baseStructure, { parentTol });
}

export function alignSubgroupFromContext(ctx, subgroupSpec) {
  if (!ctx) throw new Error('Run supercell symmetry analysis first.');
  return domainAlignedStructure(ctx.configs, ctx.dim, ctx.Aconv, ctx.refBasis, subgroupSpec, {
    parentTol: ctx.parentTol,
  });
}

export function buildAlignedCifText(aligned, { projected = true, title } = {}) {
  return generateAlignedCIF(aligned, {
    symmetrized: projected,
    title: title || `aligned_${aligned.targetSpaceGroup || 'local'}`,
  });
}
