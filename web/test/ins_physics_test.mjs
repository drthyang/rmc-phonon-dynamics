// web/test/ins_physics_test.mjs
//
// Physics regression suite for Item 3:
//   1. Direct-geometry chopper energy resolution ΔE(E; E_i) (SEQUOIA, CNCS, ARCS, LET)
//   2. Area-preserving variable-width Gaussian broadening & quadrature baseFwhm subtraction
//   3. Parameter-free Sjolander multiphonon (n >= 2) overtones & scaling
//   4. Coherent + incoherent powder S(|Q|, E), L-type vs C-type Bloch-gauge equivalence,
//      Brillouin-zone structure-factor selection rules, and Debye–Waller damping.

import assert from 'node:assert/strict';
import {
  CHOPPER_PRESETS,
  chopperResolutionFwhmAtE,
  chopperResolutionProfile,
  variableGaussianBroaden1D,
  computeMultiphonon1D,
  selectPowderInsChannel,
} from '../src/math/insCompare.js';
import { buildInsData } from '../src/compute/ins.js';
import { computePowderSqE } from '../src/io/sqeworker.js';

console.log('Running INS physics tests (coherent S(Q,E), chopper resolution, Sjolander multiphonon)...');

// ── 1. Direct-geometry chopper resolution ΔE(E; Ei) ─────────────────────────
{
  const Ei = 60;
  for (const [key, frac] of [['SEQUOIA', 0.018], ['CNCS', 0.022], ['ARCS', 0.030], ['LET', 0.015]]) {
    assert.ok(CHOPPER_PRESETS[key], `Missing preset ${key}`);
    const f0 = chopperResolutionFwhmAtE(0, Ei, key);
    assert.ok(Math.abs(f0 - frac * Ei) < 1e-10, `${key} elastic FWHM mismatch: ${f0} vs ${frac * Ei}`);

    const E = Array.from({ length: 60 }, (_, i) => i);
    const prof = chopperResolutionProfile(E, Ei, key);
    for (let i = 1; i < prof.length; i++) {
      assert.ok(prof[i] < prof[i - 1], `${key} chopper FWHM must sharpen monotonically at E=${E[i]}`);
    }
    assert.ok(prof[55] < 0.55 * prof[0], `${key} high-E FWHM should be <55% of elastic FWHM`);
  }
  console.log('  ok   direct-geometry chopper resolution ΔE(E; Ei) sharpens monotonically');
}

// ── 2. Variable-width Gaussian broadening conserves spectral area ───────────
{
  const n = 301;
  const E = Array.from({ length: n }, (_, i) => i * 0.2); // 0 .. 60 meV
  const y = new Array(n).fill(0);
  y[50] = 4.0;   // E = 10 meV (broader)
  y[240] = 4.0;  // E = 48 meV (sharper)

  const prof = chopperResolutionProfile(E, 55, 'SEQUOIA');
  const broad = variableGaussianBroaden1D(E, y, prof, 0);

  const sumIn = y.reduce((a, b) => a + b, 0);
  const sumOut = broad.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sumOut - sumIn) < 1e-12, `Spectral sum not conserved: ${sumOut} vs ${sumIn}`);
  assert.ok(broad[240] > broad[50] * 1.45, `Peak at high E (${broad[240]}) must be taller/sharper than at low E (${broad[50]})`);

  // Quadrature subtraction: when baseFwhm >= target FWHM everywhere, spectrum is unchanged
  const unchanged = variableGaussianBroaden1D(E, y, prof, 5.0);
  assert.ok(Math.abs(unchanged[50] - 4.0) < 1e-12);
  console.log('  ok   variableGaussianBroaden1D conserves area (<1e-12) and sharpens high-E modes');
}

// ── 3. Sjolander multiphonon (n >= 2) overtones & scaling ───────────────────
{
  const E = Array.from({ length: 301 }, (_, i) => i * 0.2); // 0 .. 60 meV
  // Single narrow 1-phonon optic band at E0 = 14 meV
  const gdos = E.map(e => Math.exp(-0.5 * ((e - 14.0) / 0.65) ** 2));

  const resLowQ = computeMultiphonon1D(E, gdos, { temperatureK: 10, meanQ: 1.5, uIso: 0.01, maxOrder: 4 });
  const resHiQ = computeMultiphonon1D(E, gdos, { temperatureK: 10, meanQ: 4.5, uIso: 0.01, maxOrder: 4 });

  const i28 = 140; // 28.0 meV (2 * E0 overtone)
  const i20 = 100; // 20.0 meV (valley between E0 and 2*E0)
  assert.ok(
    resHiQ.multiphononGdos[i28] > 12 * resHiQ.multiphononGdos[i20],
    `Expected sharp n=2 Sjolander overtone at 2*E0=28 meV (got ${resHiQ.multiphononGdos[i28]} vs valley ${resHiQ.multiphononGdos[i20]})`
  );
  // Relative to n=1 GDOS, n=2 apparent GDOS scales with 2W = <Q>^2 * U_iso -> (4.5 / 1.5)^2 = 9x
  const ratio = resHiQ.multiphononGdos[i28] / resLowQ.multiphononGdos[i28];
  assert.ok(ratio > 8.5 && ratio < 11.0, `Expected ~9x+ multiphonon growth from Q=1.5 to Q=4.5, got ${ratio}`);
  console.log('  ok   Sjolander multiphonon n>=2 produces exact 2*E0 overtone and 2W scaling');
}

// ── 4. Coherent structure-factor interference & L-type <-> C-type gauge ─────
{
  // Construct a 2-atom BCC-like cell (Carbon at (0,0,0) and (0.5,0.5,0.5), a = 4.0 Å):
  //   Reciprocal lattice b1 = 2*pi/a = 1.5708 Å^-1.
  //   At G = (0,0,0) (first BZ, |Q| < 0.8 Å^-1), phase is (+1, +1) -> out-of-phase optical mode
  //   e_0 = +1/sqrt(2) x̂, e_1 = -1/sqrt(2) x̂ CANCELS destructively!
  //   At G = (1,0,0) (|Q| ≈ 1.57 Å^-1), exp(-i 2*pi G.x_1) = -1 -> out-of-phase optical mode
  //   interferes CONSTRUCTIVELY (amplitude (-1)*(-1) + (+1)*(+1) = +2)!
  const a = 4.0;
  const qPoints = [
    [0.08, 0, 0],
    [-0.08, 0, 0],
    [0, 0.08, 0],
    [0, 0, 0.08],
  ];
  const s2 = Math.SQRT1_2;
  const bands = qPoints.map(() => [20.0, 20.0, 20.0, 10.0, 10.0, 10.0]);

  // C-type eigenvectors: modes 0..2 = out-of-phase optical at 20 meV, modes 3..5 = in-phase at 10 meV
  const makeEigvecs = (isLType) => qPoints.map((q) => {
    const ph1 = isLType ? +2 * Math.PI * (q[0] * 0.5 + q[1] * 0.5 + q[2] * 0.5) : 0;
    const c1 = Math.cos(ph1), sn1 = Math.sin(ph1);
    const modes = [];
    for (let pol = 0; pol < 3; pol++) {
      const re = new Float64Array(6), im = new Float64Array(6);
      re[pol] = s2;
      re[3 + pol] = -s2 * c1;
      im[3 + pol] = -s2 * sn1;
      modes.push({ real: re, imag: im });
    }
    for (let pol = 0; pol < 3; pol++) {
      const re = new Float64Array(6), im = new Float64Array(6);
      re[pol] = s2;
      re[3 + pol] = s2 * c1;
      im[3 + pol] = s2 * sn1;
      modes.push({ real: re, imag: im });
    }
    return modes;
  });

  const baseStructure = {
    atomDic: { C: [1, 2] },
    dim: [1, 1, 1],
    v1: [a, 0, 0],
    v2: [0, a, 0],
    v3: [0, 0, a],
    uniqueRN: [1, 2],
    hsym_xyz: new Float64Array([0, 0, 0, 0.5, 0.5, 0.5]),
    masses: new Float64Array([12.011, 12.011]),
  };

  const cModel = { qPoints, bands, eigvecs: makeEigvecs(false), source: 'file', temperature: 5, baseStructure };
  const lModel = { qPoints, bands, eigvecs: makeEigvecs(true), source: 'runner', temperature: 5, baseStructure };

  const insC = buildInsData(cModel, { useFullMesh: false });
  const insL = buildInsData(lModel, { useFullMesh: false });

  const powCohC = computePowderSqE(insC.data, 5, 0.6, 0, 30, 121, 100, 0, {
    instrument: 'constant',
    scatteringMode: 'coherent',
    includeMultiphonon: false,
  });
  const powCohL = computePowderSqE(insL.data, 5, 0.6, 0, 30, 121, 100, 0, {
    instrument: 'constant',
    scatteringMode: 'coherent',
    includeMultiphonon: false,
  });

  // 4a. L-type (RMC runner) and C-type (Phonopy) gauges must agree identically!
  let maxGaugeDiff = 0;
  for (let i = 0; i < powCohC.S.length; i++) {
    maxGaugeDiff = Math.max(maxGaugeDiff, Math.abs(powCohC.S[i] - powCohL.S[i]));
  }
  assert.ok(maxGaugeDiff < 1e-6, `L-type vs C-type gauge mismatch: ${maxGaugeDiff}`);

  // 4b. Destructive interference of out-of-phase 20 meV optical mode near Q -> 0 vs strong peak at G=(1,0,0) ≈ 1.57 Å^-1
  const dQ = powCohC.xMax / powCohC.nX;
  // First-BZ q = 0.08 * b1 = 0.1257 Å^-1; G=(1,0,0) ± q is around 1.5708 Å^-1
  const qiFirstBz = Math.max(0, Math.floor(0.1257 / dQ));
  const qiG100 = Math.min(powCohC.nX - 1, Math.round((2 * Math.PI / a) / dQ));
  const ei20 = Math.round((20.0 / 30.0) * 120);
  const ei10 = Math.round((10.0 / 30.0) * 120);

  const optFirstBz = powCohC.S[qiFirstBz * powCohC.nE + ei20];
  const inPhaseFirstBz = powCohC.S[qiFirstBz * powCohC.nE + ei10];
  const optAtG100 = powCohC.S[qiG100 * powCohC.nE + ei20];

  assert.ok(optFirstBz < 0.02 * inPhaseFirstBz, `Out-of-phase optical mode must cancel in 1st BZ: ${optFirstBz} vs ${inPhaseFirstBz}`);
  assert.ok(optAtG100 > 20 * optFirstBz, `Out-of-phase optical mode must turn on strongly at G=(1,0,0): ${optAtG100} vs ${optFirstBz}`);

  // 4c. Debye-Waller high-|Q| damping check (high T = 1200 K -> strong damping above Q > 6 Å^-1)
  const insHighT = buildInsData({ ...cModel, temperature: 1200 }, { temperatureK: 1200, useFullMesh: false });
  const powIncHighT = computePowderSqE(insHighT.data, 1200, 0.8, 0, 30, 121, 120, 150, {
    instrument: 'SEQUOIA',
    scatteringMode: 'incoherent_approx',
    includeMultiphonon: false,
  });
  const dQHi = powIncHighT.xMax / powIncHighT.nX;
  const qiMid = Math.round(3.5 / dQHi);
  const qiHigh = Math.min(powIncHighT.nX - 1, Math.round(14.0 / dQHi));
  assert.ok(
    powIncHighT.S[qiHigh * powIncHighT.nE + ei10] < powIncHighT.S[qiMid * powIncHighT.nE + ei10],
    'Debye-Waller factor must damp S(|Q|, E) at large |Q|'
  );

  // 4d. Channel selector utility check
  const mockIns = {
    sqe: [[10]],
    sqe_coherent: [[6]],
    sqe_incoherent: [[1]],
    sqe_multiphonon: [[3]],
  };
  assert.deepEqual(selectPowderInsChannel(mockIns, 'one_phonon'), [[7]]);
  assert.deepEqual(selectPowderInsChannel(mockIns, 'multiphonon'), [[3]]);

  console.log('  ok   coherent BZ selection rules, L/C Bloch-gauge parity, and Debye-Waller damping verified');
}

console.log('✅ ALL INS PHYSICS CHECKS PASSED');
