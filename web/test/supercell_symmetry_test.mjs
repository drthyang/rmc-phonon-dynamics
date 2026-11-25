// web/test/supercell_symmetry_test.mjs
// Synthetic ground-truth tests proving that direct supercell symmetry analysis
// recovers distortions that unit-cell folding + FINDSYM completely misses:
//   1. Pure isotropic thermal noise (null calibration: enrichment ~ 1, flat P(k))
//   2. Multi-domain local polar distortion along <111> (4 twin domains -> 0 average!)
//   3. Zone-boundary antiferrodistortive mode at X = (0.5, 0, 0) (-> 0 average!)

import assert from 'node:assert/strict';
import {
  kSpaceModeSpectrum,
  localSubgroupAnalysis,
  statisticalSymmetryTest,
  filteredMinimalSupercell,
  domainAlignedStructure,
  generateAlignedCIF,
} from '../src/math/supercell_symmetry.js';

// Deterministic PRNG (Mulberry32) + Box-Muller Gaussian
function makeRng(seed = 42) {
  let a = seed >>> 0;
  const u = () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return () => {
    const r = Math.sqrt(-2 * Math.log(Math.max(1e-12, u())));
    return r * Math.cos(2 * Math.PI * u());
  };
}

// Simple cubic perovskite ABO3 (Pm-3m, 5 atoms/cell)
const A = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]];
const basis = [
  { el: 'Ba', frac: [0.0, 0.0, 0.0] },
  { el: 'Ti', frac: [0.5, 0.5, 0.5] },
  { el: 'O',  frac: [0.5, 0.5, 0.0] },
  { el: 'O',  frac: [0.5, 0.0, 0.5] },
  { el: 'O',  frac: [0.0, 0.5, 0.5] },
];
const dim = [4, 4, 4];
const nCells = dim[0] * dim[1] * dim[2];
const D = 3 * basis.length;

// ── Test 1: Pure isotropic thermal noise (null calibration) ──────────────────
{
  const randn = makeRng(101);
  const configs = Array.from({ length: 10 }, () => {
    const u = new Float64Array(nCells * D);
    for (let i = 0; i < u.length; i++) u[i] = 0.05 * randn();
    return u;
  });

  const spec = kSpaceModeSpectrum(configs, dim, basis);
  assert.ok(spec.gammaSnr < 2.0, `Null Gamma SNR should be ~1, got ${spec.gammaSnr}`);

  const loc = localSubgroupAnalysis(configs, dim, A, basis);
  assert.equal(loc.parentSpaceGroup, 'Pm-3m');
  for (const sg of loc.subgroups) {
    if (sg.dof > 0 && sg.dof < loc.internalDoF) {
      assert.ok(Math.abs(sg.enrichment - 1.0) < 0.15,
        `Null enrichment for ${sg.spaceGroup} should be ~1.0, got ${sg.enrichment}`);
    }
  }
  console.log('PASS Test 1: Isotropic thermal noise null calibration');
}

// ── Test 2: Multi-domain local R3m (<111> Ti off-centering) ──────────────────
// Equal populations of the 8 <111> polar domains -> folded unit cell has ~0 displacement!
{
  const randn = makeRng(202);
  const dirs = [
    [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1],
    [-1, -1, -1], [-1, 1, 1], [1, -1, 1], [1, 1, -1],
  ].map(v => v.map(x => x / Math.sqrt(3)));

  const configs = Array.from({ length: 8 }, () => {
    const u = new Float64Array(nCells * D);
    for (let c = 0; c < nCells; c++) {
      // Assign balanced domains (zero global mean!)
      const dir = dirs[c % dirs.length];
      const base = c * D;
      for (let d = 0; d < D; d++) u[base + d] = 0.015 * randn();
      // Ti (site 1) shifts along dir, O (sites 2,3,4) shift opposite -> R3m polar mode
      for (let a = 0; a < 3; a++) {
        u[base + 1 * 3 + a] += 0.18 * dir[a];
        u[base + 2 * 3 + a] -= 0.06 * dir[a];
        u[base + 3 * 3 + a] -= 0.06 * dir[a];
        u[base + 4 * 3 + a] -= 0.06 * dir[a];
      }
    }
    return u;
  });

  const stat = statisticalSymmetryTest(configs, dim, A, basis);
  // Folded mean is nearly zero (meanToLocalRatio << 0.1) — traditional FINDSYM sees Pm-3m!
  assert.ok(stat.meanToLocalRatio < 0.1,
    `Multi-domain folded mean should cancel, got ratio=${stat.meanToLocalRatio}`);
  // Global statistical covariance still looks cubic (all 8 domains balanced)
  assert.equal(stat.statisticalSpaceGroup, 'Pm-3m');

  // Local cell-by-cell subgroup projection detects 3m (P3m / R3m) with huge F-stat!
  const loc = localSubgroupAnalysis(configs, dim, A, basis);
  const top = loc.subgroups[0];
  assert.equal(top.pointGroup, '3m', `Expected top local point group 3m, got ${top.pointGroup} (${top.spaceGroup})`);
  assert.ok(top.bestDomainR2 > 0.90, `Expected 3m R^2 > 0.90, got ${top.bestDomainR2}`);
  assert.ok(top.fStat > 20, `Expected large F-stat for 3m, got ${top.fStat}`);

  // Domain-aligned averaging must recover the 0.18 Å Ti shift & P3m space group
  // across ALL 8 polar <111> domains (4 twin axes x 2 polarity cosets)!
  const aligned = domainAlignedStructure(configs, dim, A, basis, top.spaceGroup);
  const tiStat = aligned.elementStats.find(e => e.el === 'Ti');
  assert.equal(aligned.nNormalizerCosets, 2, `Expected 2 polarity cosets (±[111]), got ${aligned.nNormalizerCosets}`);
  assert.ok(Math.abs(tiStat.symRms - 0.18) < 0.015, `Expected Ti shift ~0.18 Å, got ${tiStat.symRms.toFixed(4)} Å`);
  assert.ok(tiStat.foldedRms < 0.01, `Folded Ti shift should be ~0, got ${tiStat.foldedRms.toFixed(4)} Å`);
  assert.equal(aligned.verifiedSpaceGroup, 'P3m', `Expected verified SG P3m, got ${aligned.verifiedSpaceGroup}`);
  const cif = generateAlignedCIF(aligned);
  assert.ok(cif.includes('_atom_site_fract_x') && cif.includes('Ti'), 'Valid CIF output');

  console.log(`PASS Test 2: Multi-domain local 3m detected (R²=${top.bestDomainR2.toFixed(3)}, F=${top.fStat.toFixed(1)}, folded Ti=${tiStat.foldedRms.toFixed(4)} Å -> aligned Ti=${tiStat.symRms.toFixed(4)} Å, verified SG=${aligned.verifiedSpaceGroup})`);
}

// ── Test 3: Zone-boundary antiferrodistortive mode at X = (0.5, 0, 0) ────────
// Alternating +z / -z shifts along x -> exact 0 in 1x1x1 folded unit cell!
{
  const randn = makeRng(303);
  const configs = Array.from({ length: 6 }, () => {
    const u = new Float64Array(nCells * D);
    for (let x = 0; x < dim[0]; x++) {
      const phase = (x % 2 === 0) ? 1 : -1;
      for (let y = 0; y < dim[1]; y++) {
        for (let z = 0; z < dim[2]; z++) {
          const base = ((x * dim[1] + y) * dim[2] + z) * D;
          for (let d = 0; d < D; d++) u[base + d] = 0.015 * randn();
          // Antiferroelectric z-shift of Ti vs O at k = (0.5, 0, 0)
          u[base + 1 * 3 + 2] += 0.15 * phase;
          u[base + 2 * 3 + 2] -= 0.05 * phase;
          u[base + 3 * 3 + 2] -= 0.05 * phase;
          u[base + 4 * 3 + 2] -= 0.05 * phase;
        }
      }
    }
    return u;
  });

  const spec = kSpaceModeSpectrum(configs, dim, basis);
  const top = spec.topK[0];
  assert.deepEqual(top.kFrac, [0.5, 0, 0], `Expected X-point peak [0.5,0,0], got ${top.kFrac}`);
  assert.ok(top.snr > 50, `Expected strong X-point SNR, got ${top.snr}`);

  const mini = filteredMinimalSupercell(spec, dim, A, basis, { snrThreshold: 5.0 });
  assert.deepEqual(mini.miniDim, [2, 1, 1], `Expected 2x1x1 minimal supercell, got ${mini.miniDim}`);
  assert.ok(mini.ladder.length >= 2, 'Expected non-trivial symmetry ladder on 2x1x1 supercell');
  // Tight tolerance should find orthorhombic (Pmm2 / Pmmm), looser tolerance restores cubic/tetragonal
  console.log(`PASS Test 3: Zone-boundary X=(0.5,0,0) detected (SNR=${top.snr.toFixed(1)}, miniDim=${mini.miniDim.join('x')}, ladder=${mini.ladder.map(b => `${b.spaceGroup}[${b.from.toFixed(3)}..${b.to.toFixed(3)}]`).join(' -> ')})`);
}
