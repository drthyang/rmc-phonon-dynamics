// web/test/bonds_mantid_snapshot_test.mjs
//
// Regression suite for Items 1, 2, & 4:
//   1. First-shell coordination bond splitting, Baur distortion index, & 3D exaggeration
//   2. Direct Mantid `.spe` (`SaveSPE`), `.xye`, 4-col `SaveAscii`, & NCNR DAVE 1D/2D readers
//      with `-1e30` kinematic mask (`NaN`) preservation in `compare2DSqe` & `integrateOverQ`
//   3. Portable `.rmcph.json` workspace snapshot TypedArray + NaN round-trip
//   4. Streaming worker session (`createStreamingSymmetrySession`) parity with `analyzeSupercellFrames`

import assert from 'node:assert/strict';
import {
  computeBondSplitting,
  buildExaggeratedAlignedStructure,
} from '../src/math/bondSplitting.js';
import { parseExpIns1D, parseExpIns2D, parseMantidSpe } from '../src/io/insReaders.js';
import { compare2DSqe, integrateOverQ } from '../src/math/insCompare.js';
import {
  exportWorkspaceSnapshot,
  importWorkspaceSnapshot,
} from '../src/io/sessionSnapshot.js';
import {
  analyzeSupercellFrames,
  createStreamingSymmetrySession,
} from '../src/compute/supercellAnalyzer.js';
import { parseRmc6fFrame } from '../src/io/readers.js';

console.log('Running bond-splitting, Mantid/DAVE reader, snapshot, and worker-session tests...');

// ── 1. Coordination bond splitting & 3D exaggeration (`bondSplitting.js`) ──
{
  const a = 4.0;
  // Polar P4mm BaTiO3: Ti at (0.5, 0.5, 0.5) shifted +0.10 Å (+0.025 frac) along +z;
  // apical O_z at (0.5, 0.5, 0.0) shifted -0.04 Å (-0.010 frac);
  // equatorial O_x, O_y shifted -0.02 Å (-0.005 frac) along z.
  // Parent Ti-O = 6 × 2.000 Å.
  // Local Ti-O:
  //   short apical (z=1):  (4.0 - 0.04) - (2.0 + 0.10) = 1.860 Å  (×1)
  //   long apical  (z=0):  (2.0 + 0.10) - (-0.04)      = 2.140 Å  (×1)
  //   equatorial   (x,y):  sqrt(2.0^2 + 0.12^2)        = 2.0036 Å (×4)
  const aligned = {
    A: [[a, 0, 0], [0, a, 0], [0, 0, a]],
    symmetrizedBasis: [
      { rn: 1, el: 'Ba', frac: [0, 0, 0], dispCart: [0, 0, 0], dispNorm: 0 },
      { rn: 2, el: 'Ti', frac: [0.5, 0.5, 0.525], dispCart: [0, 0, 0.10], dispNorm: 0.10 },
      { rn: 3, el: 'O',  frac: [0.5, 0.5, 0.990], dispCart: [0, 0, -0.04], dispNorm: 0.04 },
      { rn: 4, el: 'O',  frac: [0.5, 0.0, 0.495], dispCart: [0, 0, -0.02], dispNorm: 0.02 },
      { rn: 5, el: 'O',  frac: [0.0, 0.5, 0.495], dispCart: [0, 0, -0.02], dispNorm: 0.02 },
    ],
  };

  const res = computeBondSplitting(aligned);
  const tiO = res.bondRows.find(r => r.canonicalPair === 'O–Ti' && r.coordNum === 6);
  assert.ok(tiO, 'Expected 6-coordinate Ti–O octahedral shell');
  assert.equal(tiO.parentSummary, '6×2.000');
  assert.equal(tiO.localSummary, '1×1.860 + 4×2.004 + 1×2.140');
  assert.ok(Math.abs(tiO.splittingAng - 0.28) < 1e-4, `Expected Δd = 0.280 Å, got ${tiO.splittingAng}`);
  assert.ok(tiO.baurIndex > 0.02, `Expected positive Baur distortion index, got ${tiO.baurIndex}`);

  // Exaggerated 3D structure at ×10 moves Ti from z=0.500 to z=0.500 + 10*0.025 = 0.750
  const exag10 = buildExaggeratedAlignedStructure(aligned, 10);
  assert.ok(Math.abs(exag10.baseStructure.hsym_xyz[1 * 3 + 2] - 0.75) < 1e-6);
  assert.ok(Math.abs(exag10.staticVectors[1 * 3 + 2] - 0.10) < 1e-9);
  console.log('  ok   Ti–O 6×2.000 -> 1×1.860 + 4×2.004 + 1×2.140 bond splitting & ×10 exaggeration verified');
}

// ── 2. Mantid `.spe`, `.xye`, 4-col `SaveAscii`, & DAVE 1D/2D readers ──────
{
  // 2a. Mantid SaveAscii 4-column + -1e30 masked point + DAVE ueV header conversion
  const mantid1DText = `# Mantid SaveAscii
# X: Energy Transfer (ueV)
1, 1000.0, -1.0e+30, 0.0
1, 2000.0, 12.5, 0.4
1, 3500.0, 25.0, 0.6
1, 5000.0, 8.0, 0.3
`;
  const p1 = parseExpIns1D(mantid1DText, 'let_slice.xye');
  assert.equal(p1.energy.length, 3, 'Masked -1e30 point must be filtered out');
  assert.ok(Math.abs(p1.energy[0] - 2.0) < 1e-6, 'ueV header must convert 2000 ueV -> 2.0 meV');
  assert.ok(Math.abs(p1.sigma[1] - 0.6) < 1e-6);

  // 2b. Mantid / MSlice `.spe` 2D with bin edges and `-1.000E+30` kinematic masks
  const speText = `2 3
### Q Grid
 1.000 2.000 3.000
### Energy Grid
 0.000 5.000 10.000 15.000
### S(Q,w)
 -1.000E+30  4.000E+00  8.000E+00
### Errors
  0.000E+00  0.200E+00  0.300E+00
### S(Q,w)
  2.000E+00  6.000E+00 -1.000E+30
### Errors
  0.100E+00  0.250E+00  0.000E+00
`;
  const spe = parseMantidSpe(speText, 'sequoia_run.spe');
  assert.deepEqual(spe.q_grid, [1.5, 2.5]);
  assert.deepEqual(spe.e_grid, [2.5, 7.5, 12.5]);
  assert.ok(Number.isNaN(spe.sqe[0][0]), 'Masked -1e30 bin (e=0, q=0) must be NaN');
  assert.ok(Number.isNaN(spe.sqe[2][1]), 'Masked -1e30 bin (e=2, q=1) must be NaN');
  assert.equal(spe.sqe[1][0], 4.0);
  assert.equal(spe.sqe[0][1], 2.0);

  // Auto-detection via parseExpIns2D + unbiased compare2DSqe when NaN masks are present
  const speAuto = parseExpIns2D(speText, 'sequoia_run.spe');
  assert.equal(speAuto.format, 'spe');
  const candIns = {
    q_grid: [1.5, 2.5],
    e_grid: [2.5, 7.5, 12.5],
    sqe: [
      [999.0, 1.0], // (0,0) is masked in experiment -> 999 must NOT bias Rw!
      [2.0,   3.0],
      [4.0, 999.0], // (2,1) is masked in experiment -> 999 must NOT bias Rw!
    ],
  };
  const cmp = compare2DSqe(candIns, speAuto);
  assert.ok(Math.abs(cmp.scale - 2.0) < 1e-9, `Expected scale=2.0 ignoring NaN masks, got ${cmp.scale}`);
  assert.ok(cmp.rw < 1e-10, `Expected Rw=0 on unmasked detector bins, got ${cmp.rw}`);

  const qInt = integrateOverQ(speAuto.sqe, speAuto.q_grid, 1.0, 3.0);
  assert.deepEqual(qInt, [2.0, 5.0, 8.0]);

  // 2c. NCNR DAVE 2D ASCII reader
  const daveText = `# Dimensions: 2 2
# X-axis: Energy (meV)
5.0 10.0
# Y-axis: |Q| (A^-1)
1.5 3.0
# Data
1.2 3.4
5.6 -1e30
`;
  const dave2d = parseExpIns2D(daveText, 'dcs_slice.dat');
  assert.equal(dave2d.format, 'dave-2d');
  assert.deepEqual(dave2d.q_grid, [1.5, 3.0]);
  assert.deepEqual(dave2d.e_grid, [5.0, 10.0]);
  assert.equal(dave2d.sqe[0][1], 5.6);
  assert.ok(Number.isNaN(dave2d.sqe[1][1]));
  console.log('  ok   Mantid .spe, .xye, 4-col SaveAscii, DAVE 2D, and NaN kinematic masks verified');
}

// ── 3. Portable `.rmcph.json` workspace snapshot round-trip ─────────────────
{
  const orig = {
    page: 'ins',
    activeCandidateId: 'cand-1',
    candidates: [
      {
        id: 'cand-1',
        label: 'Local · P4mm',
        color: '#c95d3a',
        kind: 'local-subgroup',
        model: {
          bands: [[0, 12.5]],
          baseStructure: {
            v1: [4, 0, 0], v2: [0, 4, 0], v3: [0, 0, 4],
            dim: [1, 1, 1],
            atomDic: { Ba: [1] },
            uniqueRN: [1],
            hsym_xyz: new Float64Array([0.0, 0.0, 0.025]),
            masses: new Float64Array([137.33]),
          },
        },
      },
    ],
    exp1D: { name: 'test.xye', energy: [2, 4, 6], intensity: [10, 20, 5], sigma: [0.5, 0.5, 0.5] },
    exp2D: { name: 'test.spe', q_grid: [1, 2], e_grid: [5, 10], sqe: [[1.5, NaN], [2.5, 3.5]] },
  };

  const json = exportWorkspaceSnapshot(orig);
  const restored = importWorkspaceSnapshot(json);
  assert.equal(restored.page, 'ins');
  assert.equal(restored.candidates.length, 1);
  assert.ok(restored.candidates[0].model.baseStructure.hsym_xyz instanceof Float64Array);
  assert.ok(Math.abs(restored.candidates[0].model.baseStructure.hsym_xyz[2] - 0.025) < 1e-6);
  assert.ok(Number.isNaN(restored.exp2D.sqe[0][1]), 'NaN mask in exp2D must survive JSON round-trip');
  assert.equal(restored.exp2D.sqe[1][1], 3.5);
  console.log('  ok   .rmcph.json workspace snapshot TypedArray & NaN round-trip verified');
}

// ── 4. Streaming worker session parity (`createStreamingSymmetrySession`) ──
{
  const nx = 4, ny = 4, nz = 4, a = 4.0;
  const baseStructure = {
    v1: [nx * a, 0, 0],
    v2: [0, ny * a, 0],
    v3: [0, 0, nz * a],
    dim: [nx, ny, nz],
    atomDic: { Ba: [1], Ti: [2] },
  };
  const lines = [
    '(Version 6f format configuration file)',
    `Supercell dimensions: ${nx} ${ny} ${nz}`,
    'Number of moves accepted: 500',
    'Lattice vectors (Ang):',
    `${nx * a} 0 0`,
    `0 ${ny * a} 0`,
    `0 0 ${nz * a}`,
    'Atoms:',
  ];
  const xyz = [];
  const atomType = [];
  const cellIdx = [];
  let id = 1;
  for (const [el, rn, f0, dz] of [['Ba', 1, [0, 0, 0], 0], ['Ti', 2, [0.5, 0.5, 0.5], 0.04]]) {
    for (let ix = 0; ix < nx; ix++) {
      for (let iy = 0; iy < ny; iy++) {
        for (let iz = 0; iz < nz; iz++) {
          const sgn = ix < 2 ? 1 : -1;
          const fx = (ix + f0[0]) / nx;
          const fy = (iy + f0[1]) / ny;
          const fz = (iz + f0[2] + sgn * dz) / nz;
          lines.push(`${id++} ${el} [${rn}] ${fx.toFixed(6)} ${fy.toFixed(6)} ${fz.toFixed(6)} ${rn} ${ix} ${iy} ${iz}`);
          xyz.push(fx, fy, fz);
          atomType.push(rn);
          cellIdx.push(ix, iy, iz);
        }
      }
    }
  }
  const rmcText = lines.join('\n') + '\n';
  const session = createStreamingSymmetrySession('rmc6f', baseStructure, { parentTol: 0.25, nNullSamples: 64 });
  session.pushText(rmcText, 'demo.rmc6f');
  const workerRes = session.finalize(1);

  const directRes = analyzeSupercellFrames(
    [parseRmc6fFrame(rmcText, [nx, ny, nz])],
    baseStructure,
    { parentTol: 0.25, nNullSamples: 64 }
  );

  assert.equal(workerRes.aligned.verifiedSpaceGroup, directRes.aligned.verifiedSpaceGroup);
  assert.ok(Math.abs(workerRes.aligned.symRms - directRes.aligned.symRms) < 1e-6);
  console.log('  ok   streaming worker session matches in-thread supercell analyzer identically');
}

console.log('✅ ALL BOND-SPLITTING, MANTID/DAVE, SNAPSHOT, & WORKER CHECKS PASSED');
