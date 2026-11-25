// web/test/ensemble_symmetry_test.mjs
// Deterministic tests for StreamingSymmetryAccumulator:
//   1. Multi-config isotropic thermal noise null calibration (excessDeltaF ~ 0, CV ΔF ~ 1)
//   2. Multi-config spatially correlated P4mm domains + automatic unconverged-config skipping
//   3. Radial C_dom(r), correlation length ξ, 3D cluster sizes, winRate, SE, and instant alignedBySubgroup

import assert from 'node:assert/strict';
import { StreamingSymmetryAccumulator } from '../src/math/ensemble_symmetry.js';

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeGauss(seed) {
  const rng = mulberry32(seed);
  return () => {
    const u = Math.max(1e-12, rng());
    const v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}

// Perovskite BaTiO3 parent cell (Pm-3m)
const A = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]];
const basis = [
  { rn: 1, el: 'Ba', frac: [0.0, 0.0, 0.0] },
  { rn: 2, el: 'Ti', frac: [0.5, 0.5, 0.5] },
  { rn: 3, el: 'O',  frac: [0.5, 0.5, 0.0] },
  { rn: 4, el: 'O',  frac: [0.5, 0.0, 0.5] },
  { rn: 5, el: 'O',  frac: [0.0, 0.5, 0.5] },
];
const dim = [4, 4, 4];
const nCells = 64, D = 15;

// Test 1: Pure isotropic Gaussian thermal noise across 8 configs
{
  const randn = makeGauss(101);
  const acc = new StreamingSymmetryAccumulator(dim, A, basis, { parentTol: 0.1, nNullSamples: 200 });
  for (let m = 0; m < 8; m++) {
    const u = new Float64Array(nCells * D);
    for (let i = 0; i < u.length; i++) u[i] = 0.08 * randn();
    acc.addConfig(u, `noise_${m + 1}`);
  }
  const res = acc.finalize();
  assert.equal(res.nConfigs, 8);
  const breaking = res.local.subgroups.filter(s => s.breakingDoF > 0 && s.dof < res.local.internalDoF);
  for (const s of breaking) {
    assert.ok(Math.abs(s.excessDeltaF) < 0.35, `Expected null excessDeltaF ~ 0 for ${s.spaceGroup}, got ${s.excessDeltaF}`);
    assert.ok(s.cvBreakingFStat < 1.35, `Expected bias-free CV ΔF ~ 1 for ${s.spaceGroup}, got ${s.cvBreakingFStat}`);
  }
  console.log('PASS Test 1: Multi-config isotropic noise null calibration & bias-free CV ΔF');
}

// Test 2: Spatially correlated P4mm twin domains (2x4x4 slabs of x/y/z polar domains) across 8 configs + 1 unconverged config
{
  const randn = makeGauss(202);
  const acc = new StreamingSymmetryAccumulator(dim, A, basis, { parentTol: 0.1, nNullSamples: 128 });

  // Inject an unconverged zero-displacement config first — must be skipped automatically!
  acc.addConfig(new Float64Array(nCells * D), 'unconverged_0.rmc6f');

  for (let m = 0; m < 8; m++) {
    const u = new Float64Array(nCells * D);
    for (let x = 0; x < 4; x++) {
      for (let y = 0; y < 4; y++) {
        for (let z = 0; z < 4; z++) {
          const c = (x * 4 + y) * 4 + z;
          const off = c * D;
          // Small thermal noise
          for (let d = 0; d < D; d++) u[off + d] = 0.025 * randn();
          // Two large coherent P4mm domains: x < 2 -> +z Ti shift, x >= 2 -> +x Ti shift
          const axis = x < 2 ? 2 : 0;
          const apicalO = axis === 2 ? 2 : 4; // O at [0.5,0.5,0] for z, [0,0.5,0.5] for x
          const sign = (m % 2 === 0) ? 1 : -1;
          u[off + 1 * 3 + axis] += sign * 0.16;       // Ti off-centering
          u[off + apicalO * 3 + axis] -= sign * 0.04; // apical O counter-shift
        }
      }
    }
    acc.addConfig(u, `p4mm_cfg_${m + 1}`);
  }

  const res = acc.finalize();
  assert.equal(res.nConfigs, 8);
  assert.equal(res.skippedConfigs.length, 1);
  assert.equal(res.skippedConfigs[0].name, 'unconverged_0.rmc6f');
  assert.equal(res.convergence.length, 8);

  const p4mm = res.local.subgroups.find(s => s.spaceGroup === 'P4mm');
  assert.ok(p4mm, 'P4mm subgroup must be present');
  assert.ok(p4mm.excessDeltaF > 2.0, `Expected strong excessDeltaF for P4mm, got ${p4mm.excessDeltaF}`);
  assert.ok(p4mm.cvBreakingFStat > 2.0, `Expected strong bias-free CV ΔF for P4mm, got ${p4mm.cvBreakingFStat}`);
  assert.ok(p4mm.winRate >= 0.85, `Expected P4mm to win >= 85% of configs, got ${p4mm.winRate}`);
  assert.ok(p4mm.radialCorr[0] > 0.4, `Expected strong radial C_dom(1) > 0.4, got ${p4mm.radialCorr[0]}`);
  assert.ok(p4mm.corrLengthCells > 1.0, `Expected ξ > 1 cell, got ${p4mm.corrLengthCells}`);
  assert.ok(p4mm.meanClusterCells > 5.0, `Expected large 3D clusters, got ${p4mm.meanClusterCells}`);
  assert.ok(res.alignedBySubgroup.P4mm, 'Pre-streamed alignedBySubgroup.P4mm must exist');
  assert.equal(res.alignedBySubgroup.P4mm.verifiedSpaceGroup, 'P4mm');
  assert.ok(res.alignedBySubgroup.P4mm.symRms > 0.03);
  assert.ok(Number.isFinite(res.alignedBySubgroup.P4mm.rmsSE));

  console.log(
    `PASS Test 2: P4mm domains detected across 8 configs (excessΔF=${p4mm.excessDeltaF.toFixed(2)}±${p4mm.excessDeltaFSE.toFixed(2)}, CV ΔF=${p4mm.cvBreakingFStat.toFixed(2)}, win=${(100*p4mm.winRate).toFixed(0)}%, ξ=${p4mm.corrLengthCells.toFixed(2)} cells / ${p4mm.corrLengthAng.toFixed(1)} Å, verified=${res.alignedBySubgroup.P4mm.verifiedSpaceGroup})`
  );
}
