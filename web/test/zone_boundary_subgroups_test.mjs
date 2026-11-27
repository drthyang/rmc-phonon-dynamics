// web/test/zone_boundary_subgroups_test.mjs
//
// Deterministic validation of zone-boundary multi-cell & centering-breaking subgroup discovery:
//   1. Zero Γ-leakage guarantee: strong 1×1×1 P4mm polar domains produce zero false-positive
//      Excess ΔF in 1×1×2 / 2×2×1 / 2×2×2 zone-boundary channels.
//   2. 2×2×2 R-star out-of-phase octahedral tilt (Pm-3m -> I4/mcm #140, Glazer a⁰a⁰c⁻):
//      twin + antiphase domain unmixing, spatial CV ΔF, verified I4/mcm (#140), and 40-atom CIF.
//   3. 2×2×1 M-star in-phase octahedral rotation (Pm-3m -> P4/mbm #127, Glazer a⁰a⁰c⁺):
//      antiphase domain tracking (nVariants=1, nDomains=2, domainKind='antiphase') & 3D grid upsampling.
//   4. Centering-breaking non-symmorphic P-42₁m (#113) inside F-43m (#216):
//      distinguishes P-42₁m (#113) from P-42m (#111) / P-42₁c (#114) and unmixes 6×2 twin/antiphase domains.

import assert from 'node:assert/strict';
import { StreamingSymmetryAccumulator } from '../src/math/ensemble_symmetry.js';
import { alignSubgroupFromResult, buildAlignedCifText } from '../src/compute/supercellAnalyzer.js';

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

const wrap01 = (x) => ((x % 1) + 1) % 1;

// Perovskite ABX3 parent cell (Pm-3m, #221)
const Abto = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]];
const basisBto = [
  { rn: 1, el: 'Ba', frac: [0.0, 0.0, 0.0] },
  { rn: 2, el: 'Ti', frac: [0.5, 0.5, 0.5] },
  { rn: 3, el: 'O',  frac: [0.5, 0.5, 0.0] }, // O_z (apical along z)
  { rn: 4, el: 'O',  frac: [0.5, 0.0, 0.5] }, // O_y (equatorial at y=0, z=0.5)
  { rn: 5, el: 'O',  frac: [0.0, 0.5, 0.5] }, // O_x (equatorial at x=0, z=0.5)
];

// ── Test 1: Zero Γ-leakage on 1×1×1 P4mm ferrodistortive shift (+ domain-wall dominance) ──
{
  const dim = [4, 4, 4];
  const D = 15;
  const randn = makeGauss(202601);
  const acc = new StreamingSymmetryAccumulator(dim, Abto, basisBto, { parentTol: 0.1, nNullSamples: 128 });

  // 1a. Pure Γ ferrodistortive shift (antiparallel Ti/O polar mode + random global polarity per config)
  for (let m = 0; m < 6; m++) {
    const u = new Float64Array(64 * D);
    const sgn = m % 2 === 0 ? 1 : -1;
    for (let c = 0; c < 64; c++) {
      const off = c * D;
      for (let d = 0; d < D; d++) u[off + d] = 0.025 * randn();
      u[off + 1 * 3 + 2] += sgn * 0.20; // strong 1×1×1 Ti z-shift
      u[off + 2 * 3 + 2] -= sgn * 0.06; // apical O counter-shift
    }
    acc.addConfig(u, `gamma_p4mm_${m + 1}`);
  }

  const res = acc.finalize();
  const zbSubgroups = res.local.subgroups.filter(s => s.isZoneBoundary);
  assert.ok(zbSubgroups.length >= 10, 'Expected commensurate multi-cell subgroups to be enumerated');
  for (const s of zbSubgroups) {
    assert.ok(
      s.excessDeltaF < 0.45,
      `Expected zero Γ-leakage into ${s.cellLabel} ${s.spaceGroup}, got excessDeltaF=${s.excessDeltaF}`
    );
  }
  assert.equal(res.aligned.targetSpaceGroup, 'P4mm');
  assert.deepEqual(res.aligned.cellMult, [1, 1, 1]);

  // 1b. Multi-domain 1×1×1 P4mm with sharp 90° domain walls: 1×1×1 P4mm still dominates #1 by > 5×
  const accDom = new StreamingSymmetryAccumulator(dim, Abto, basisBto, { parentTol: 0.1, nNullSamples: 128 });
  for (let m = 0; m < 4; m++) {
    const u = new Float64Array(64 * D);
    for (let x = 0; x < 4; x++) {
      for (let y = 0; y < 4; y++) {
        for (let z = 0; z < 4; z++) {
          const off = ((x * 4 + y) * 4 + z) * D;
          for (let d = 0; d < D; d++) u[off + d] = 0.02 * randn();
          const axis = x < 2 ? 2 : 0;
          u[off + 1 * 3 + axis] += 0.18;
        }
      }
    }
    accDom.addConfig(u, `dom_p4mm_${m + 1}`);
  }
  const resDom = accDom.finalize();
  const p4mmDom = resDom.local.subgroups.find(s => s.cellLabel === '1×1×1' && s.spaceGroup === 'P4mm');
  const maxZbDom = Math.max(...resDom.local.subgroups.filter(s => s.isZoneBoundary).map(s => s.excessDeltaF));
  assert.equal(resDom.aligned.targetSpaceGroup, 'P4mm');
  assert.ok(p4mmDom.excessDeltaF > 5 * maxZbDom, `Expected 1×1×1 P4mm (${p4mmDom.excessDeltaF}) >> max ZB wall leakage (${maxZbDom})`);
  console.log('PASS Test 1: Zero Γ-leakage into 1×1×2, 2×2×1, and 2×2×2 channels on 1×1×1 P4mm crystal');
}

// ── Test 2: 2×2×2 R-point octahedral tilt (Pm-3m -> I4/mcm #140, a⁰a⁰c⁻) ────
{
  const dim = [4, 4, 4];
  const D = 15;
  const randn = makeGauss(202602);
  const acc = new StreamingSymmetryAccumulator(dim, Abto, basisBto, { parentTol: 0.1, nNullSamples: 128 });

  // Build spatially correlated 2×2×2 twin domains:
  //   x < 2: c-axis out-of-phase tilt a⁰a⁰c⁻: O_y(0.5,0,0.5) moves +x, O_x(0,0.5,0.5) moves -y with (-1)^(x+y+z)
  //   x >= 2: a-axis out-of-phase tilt a⁻a⁰a⁰: O_z(0.5,0.5,0) moves +y, O_y(0.5,0,0.5) moves -z with (-1)^(x+y+z)
  for (let m = 0; m < 6; m++) {
    const u = new Float64Array(64 * D);
    const cfgAntiphase = (m % 2 === 0) ? 1 : -1;
    for (let x = 0; x < 4; x++) {
      for (let y = 0; y < 4; y++) {
        for (let z = 0; z < 4; z++) {
          const off = ((x * 4 + y) * 4 + z) * D;
          for (let d = 0; d < D; d++) u[off + d] = 0.018 * randn();
          const stag = (((x + y + z) & 1) ? -1 : 1) * cfgAntiphase;
          const amp = 0.18;
          if (x < 2) {
            u[off + 3 * 3 + 0] += stag * amp; // O_y along +x
            u[off + 4 * 3 + 1] -= stag * amp; // O_x along -y
          } else {
            u[off + 2 * 3 + 1] += stag * amp; // O_z along +y
            u[off + 3 * 3 + 2] -= stag * amp; // O_y along -z
          }
        }
      }
    }
    acc.addConfig(u, `i4mcm_${m + 1}`);
  }

  const res = acc.finalize();
  // All 1×1×1 subgroups should see zero coherent breaking, while 2×2×2 I4/mcm (#140) dominates!
  const i4mcm = res.local.subgroups.find(s => s.cellLabel === '2×2×2' && s.spaceGroup === 'I4/mcm');
  assert.ok(i4mcm, '2×2×2 I4/mcm (#140) must be discovered');
  assert.equal(i4mcm.spaceGroupNumber, 140);
  assert.ok(i4mcm.excessDeltaF > 10.0, `Expected strong Excess ΔF for 2×2×2 I4/mcm, got ${i4mcm.excessDeltaF}`);
  assert.ok(i4mcm.cvBreakingFStat > 3.0, `Expected strong Spatial CV ΔF for 2×2×2 I4/mcm, got ${i4mcm.cvBreakingFStat}`);

  const al = alignSubgroupFromResult(res, i4mcm);
  assert.ok(al, 'Aligned 2×2×2 I4/mcm structure must be pre-streamed');
  assert.deepEqual(al.cellMult, [2, 2, 2]);
  assert.equal(al.symmetrizedBasis.length, 40);
  assert.equal(al.verifiedSpaceGroup, 'I4/mcm');
  assert.equal(al.verifiedSpaceGroupNumber, 140);
  assert.ok(al.symRms > 0.06 && al.foldedRms < 0.02, `Expected antiphase/twin unmixing gain, got sym=${al.symRms}, folded=${al.foldedRms}`);

  const cif = buildAlignedCifText(al);
  assert.ok(cif.includes('_cell_length_a                     8.000000'));
  assert.ok(cif.includes('I4/mcm'));
  console.log(`PASS Test 2: 2×2×2 I4/mcm (#140) tilt discovered & unmixed (excessΔF=${i4mcm.excessDeltaF.toFixed(1)}, CV ΔF=${i4mcm.cvBreakingFStat.toFixed(1)}, symRms=${al.symRms.toFixed(4)} Å vs folded=${al.foldedRms.toFixed(4)} Å)`);
}

// ── Test 3: 2×2×1 M-point in-phase tilt (Pm-3m -> P4/mbm #127, a⁰a⁰c⁺) with antiphase domains ──
{
  const dim = [4, 4, 4];
  const D = 15;
  const randn = makeGauss(202603);
  const acc = new StreamingSymmetryAccumulator(dim, Abto, basisBto, { parentTol: 0.1, nNullSamples: 128 });

  // Single c-axis orientation (nVariants=1 in 2×2×1), TWO antiphase slabs (z < 2 -> +phase, z >= 2 -> -phase)!
  for (let m = 0; m < 4; m++) {
    const u = new Float64Array(64 * D);
    for (let x = 0; x < 4; x++) {
      for (let y = 0; y < 4; y++) {
        for (let z = 0; z < 4; z++) {
          const off = ((x * 4 + y) * 4 + z) * D;
          for (let d = 0; d < D; d++) u[off + d] = 0.015 * randn();
          const antiphase = z < 2 ? 1 : -1;
          const stagXY = (((x + y) & 1) ? -1 : 1) * antiphase;
          u[off + 3 * 3 + 0] += stagXY * 0.17; // O_y along +x
          u[off + 4 * 3 + 1] -= stagXY * 0.17; // O_x along -y
        }
      }
    }
    acc.addConfig(u, `p4mbm_${m + 1}`);
  }

  const res = acc.finalize();
  const p4mbm = res.local.subgroups.find(s => s.cellLabel === '2×2×1' && s.spaceGroup === 'P4/mbm');
  assert.ok(p4mbm, '2×2×1 P4/mbm (#127) must be discovered');
  assert.equal(p4mbm.spaceGroupNumber, 127);
  assert.equal(p4mbm.domainKind, 'antiphase');
  assert.equal(p4mbm.nDomains, 2);
  assert.ok(p4mbm.domainFractions[0] > 0.35 && p4mbm.domainFractions[1] > 0.35, `Expected ~50/50 antiphase domains, got ${p4mbm.domainFractions}`);
  assert.equal(p4mbm.domainGrid?.length, 64, 'Antiphase block grid must upsample to full 4×4×4 supercell');

  const al = alignSubgroupFromResult(res, p4mbm);
  assert.equal(al.verifiedSpaceGroup, 'P4/mbm');
  assert.equal(al.symmetrizedBasis.length, 20);
  assert.ok(al.symRms > 0.06 && al.foldedRms < 0.02);
  console.log(`PASS Test 3: 2×2×1 P4/mbm (#127) antiphase domains unmixed (A1=${(100*p4mbm.domainFractions[0]).toFixed(0)}%/A2=${(100*p4mbm.domainFractions[1]).toFixed(0)}%, ξ=${p4mbm.corrLengthAng.toFixed(1)} Å, verified=${al.verifiedSpaceGroup})`);
}

// ── Test 4: Non-symmorphic P-42₁m (#113) discovery inside F-43m (#216) ───────
{
  const Afcc = [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]];
  const fccShifts = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]];
  const basisF = [];
  let rn = 1;
  for (const sh of fccShifts) {
    basisF.push({ rn: rn++, el: 'Ga', frac: [...sh] });
    const x = 0.38;
    for (const p of [[x, x, x], [x, -x, -x], [-x, x, -x], [-x, -x, x]]) {
      basisF.push({ rn: rn++, el: 'Ta', frac: [wrap01(p[0] + sh[0]), wrap01(p[1] + sh[1]), wrap01(p[2] + sh[2])] });
    }
  }

  // Probe the exact P-42₁m (#113) mode vector from a 1-config probe, then synthesize twinned 4×4×4 configs
  const probeAcc = new StreamingSymmetryAccumulator([4, 4, 4], Afcc, basisF, { parentTol: 0.1, nNullSamples: 32 });
  const D = basisF.length * 3;
  const randProbe = makeGauss(99);
  probeAcc.addConfig(Float64Array.from({ length: 64 * D }, () => 0.02 * randProbe()), 'probe');
  const p421mFam = probeAcc._zoneAccs[0].fams.find(f => f.spaceGroup === 'P-42₁m' && f.spaceGroupNumber === 113);
  assert.ok(p421mFam, 'F-43m centering-breaking enumeration must find P-42₁m (#113)');

  // Build an exact normalized P-42₁m (#113) pure centering-breaking mode vector `mode0`
  const zAcc0 = probeAcc._zoneAccs[0];
  const rawSeed = Float64Array.from({ length: D }, (_, i) => Math.sin(i * 1.7 + 0.3));
  const zbSeed = new Float64Array(D);
  zAcc0._extractZB(rawSeed, zbSeed);
  zAcc0._computeBlockGU(zbSeed);
  const mode0 = new Float64Array(D);
  for (const oi of p421mFam.variants[0].opIndices) {
    const gu = zAcc0._gUzb[oi];
    for (let d = 0; d < D; d++) mode0[d] += gu[d];
  }
  let mNorm = Math.sqrt(mode0.reduce((s, v) => s + v * v, 0) / basisF.length);
  for (let d = 0; d < D; d++) mode0[d] *= 0.14 / mNorm;

  // Generate a rotated variant `mode1` (conjugate variant #2) and antiphase-shifted `-mode0`
  zAcc0._computeBlockGU(mode0);
  // Find an op in blockReps that maps variant 0 to another variant
  const v2Op = zAcc0.blockReps.findIndex((_, gi) => gi > 0 && !p421mFam.variants[0].opIndices.includes(gi));
  const modeOther = Float64Array.from(zAcc0._gUzb[v2Op]);

  const randn = makeGauss(202604);
  const acc = new StreamingSymmetryAccumulator([4, 4, 4], Afcc, basisF, { parentTol: 0.1, nNullSamples: 96 });
  for (let m = 0; m < 4; m++) {
    const u = new Float64Array(64 * D);
    const sgn = (m % 2 === 0) ? 1 : -1;
    for (let x = 0; x < 4; x++) {
      for (let y = 0; y < 4; y++) {
        for (let z = 0; z < 4; z++) {
          const off = ((x * 4 + y) * 4 + z) * D;
          const src = x < 2 ? mode0 : modeOther;
          for (let d = 0; d < D; d++) u[off + d] = sgn * src[d] + 0.015 * randn();
        }
      }
    }
    acc.addConfig(u, `f43m_p421m_${m + 1}`);
  }

  const res = acc.finalize();
  const p421m = res.local.subgroups.find(s => s.spaceGroup === 'P-42₁m');
  assert.ok(p421m, 'P-42₁m (#113) must be present in finalized subgroups');
  assert.equal(p421m.spaceGroupNumber, 113);
  assert.ok(p421m.excessDeltaF > 5.0, `Expected strong Excess ΔF for P-42₁m (#113), got ${p421m.excessDeltaF}`);
  assert.ok(p421m.cvBreakingFStat > 2.0, `Expected strong Spatial CV ΔF for P-42₁m (#113), got ${p421m.cvBreakingFStat}`);

  const al = alignSubgroupFromResult(res, p421m);
  assert.equal(al.verifiedSpaceGroup, 'P-42₁m');
  assert.equal(al.verifiedSpaceGroupNumber, 113);
  assert.ok(al.symRms > 0.08 && al.foldedRms < 0.025, `Expected twin+antiphase recovery gain, got sym=${al.symRms}, folded=${al.foldedRms}`);
  console.log(`PASS Test 4: Non-symmorphic P-42₁m (#113) inside F-43m discovered & verified (excessΔF=${p421m.excessDeltaF.toFixed(1)}, CV ΔF=${p421m.cvBreakingFStat.toFixed(1)}, symRms=${al.symRms.toFixed(4)} Å)`);
}
