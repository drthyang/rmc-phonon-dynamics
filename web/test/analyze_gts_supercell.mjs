// Real-data supercell symmetry analysis on GaTa4Se8 (GTS) 250K vs 5K RMC ensembles.
// Compares the old folded-unit-cell approach against direct supercell methods.

import fs from 'node:fs';
import path from 'node:path';
import {
  kSpaceModeSpectrum,
  localSubgroupAnalysis,
  statisticalSymmetryTest,
  filteredMinimalSupercell,
  domainAlignedStructure,
  generateAlignedCIF,
} from '/Users/thyang/GitHub/rmc-phonon-dynamics/web/src/math/supercell_symmetry.js';
import { symmetryLadder } from '/Users/thyang/GitHub/rmc-phonon-dynamics/web/src/math/symmetry.js';

function readRmc6fBase(filePath) {
  const lines = fs.readFileSync(filePath, 'utf8').split('\n');
  let dim = null, v1, v2, v3;
  const rnInfo = new Map();
  let inAtoms = false;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const parts = line.split(/\s+/);
    if (parts[0] === 'Supercell') dim = parts.slice(-3).map(Number);
    else if (parts[0] === 'Lattice') {
      v1 = lines[++i].trim().split(/\s+/).map(Number);
      v2 = lines[++i].trim().split(/\s+/).map(Number);
      v3 = lines[++i].trim().split(/\s+/).map(Number);
    } else if (line.startsWith('Atoms:')) { inAtoms = true; continue; }
    if (inAtoms && parts.length >= 7) {
      const el = parts[1];
      const rn = parseInt(parts[parts.length - 4]);
      if (isNaN(rn)) continue;
      if (!rnInfo.has(rn)) {
        const gf = [
          parseFloat(parts[parts.length - 7]),
          parseFloat(parts[parts.length - 6]),
          parseFloat(parts[parts.length - 5]),
        ];
        const frac = gf.map((g, k) => { let w = (g * dim[k]) % 1; if (w < 0) w += 1; return w; });
        rnInfo.set(rn, { el, frac });
      }
    }
  }
  const basis = [...rnInfo.entries()].sort((a, b) => a[0] - b[0]).map(([rn, info]) => ({ rn, el: info.el, frac: info.frac }));
  const rnToSite = new Map(basis.map((b, idx) => [b.rn, idx]));
  const A = [
    v1.map(x => x / dim[0]),
    v2.map(x => x / dim[1]),
    v3.map(x => x / dim[2]),
  ];
  return { A, dim, basis, rnToSite };
}

function readCartCoordConfig(filePath, dim, rnToSite, nBasis) {
  const [N1, N2, N3] = dim;
  const D = 3 * nBasis;
  const u = new Float64Array(N1 * N2 * N3 * D);
  const lines = fs.readFileSync(filePath, 'utf8').split('\n');
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line || isNaN(line[0])) continue;
    const p = line.split(/\s+/);
    if (p.length < 10) continue;
    const rn = parseInt(p[0]);
    const s = rnToSite.get(rn);
    if (s === undefined) continue;
    const dx = parseFloat(p[4]), dy = parseFloat(p[5]), dz = parseFloat(p[6]);
    const nx = parseInt(p[7]), ny = parseInt(p[8]), nz = parseInt(p[9]);
    const base = ((nx * N2 + ny) * N3 + nz) * D + s * 3;
    u[base] = dx;
    u[base + 1] = dy;
    u[base + 2] = dz;
  }
  return u;
}

function analyzeDataset(label, rmc6fPath, ensembleDir, maxConfigs = 20) {
  console.log(`\n${'='.repeat(72)}\n${label}\n${'='.repeat(72)}`);
  const { A, dim, basis, rnToSite } = readRmc6fBase(rmc6fPath);
  const files = fs.readdirSync(ensembleDir)
    .filter(f => f.startsWith('Cart_coord_') && f.endsWith('.txt'))
    .sort()
    .slice(0, maxConfigs);

  console.log(`Supercell: ${dim.join('x')} (${dim[0]*dim[1]*dim[2]} cells), ${basis.length} sites/cell, loading ${files.length} configs...`);
  const configs = files.map(f => readCartCoordConfig(path.join(ensembleDir, f), dim, rnToSite, basis.length));

  // 1. k-space mode spectrum
  const t0 = Date.now();
  const spec = kSpaceModeSpectrum(configs, dim, basis);
  console.log(`\n[1] k-Space Mode Spectrum (${Date.now() - t0} ms)`);
  console.log(`  Median P(k): ${spec.medianPower.toFixed(4)} Å², Gamma P(0): ${spec.gammaPower.toFixed(4)} Å² (SNR=${spec.gammaSnr.toFixed(2)}x), Gamma ξ ≈ ${spec.gammaCorrelationCells.toFixed(2)} cells`);
  console.log('  Top 8 k-points:');
  for (const kp of spec.topK.slice(0, 8)) {
    const kStr = `[${kp.kFrac.map(x => x.toFixed(3)).join(', ')}]`;
    const cohPct = (100 * kp.coherentPower / Math.max(1e-12, kp.power)).toFixed(1);
    const elStr = Object.entries(kp.byEl).map(([e, v]) => `${e}:${v.toFixed(3)}`).join(' ');
    console.log(`    k=${kStr.padEnd(23)} P=${kp.power.toFixed(4)} SNR=${kp.snr.toFixed(2)}x coh=${cohPct}%  (${elStr})`);
  }

  // 2. Local subgroup projection (cell-by-cell)
  const t1 = Date.now();
  const loc = localSubgroupAnalysis(configs, dim, A, basis, { parentTol: 0.25 });
  console.log(`\n[2] Local Cell-by-Cell Subgroup Projection (${Date.now() - t1} ms)`);
  console.log(`  Parent SG: ${loc.parentSpaceGroup} (order ${loc.parentOrder}), internal DoF=${loc.internalDoF}`);
  console.log('  Rank | SpaceGroup | PG     | Order | Vars | DoF | bestR² | nullR² | F-stat | ΔF(brk) | NN-match (excess)');
  for (let i = 0; i < Math.min(10, loc.subgroups.length); i++) {
    const s = loc.subgroups[i];
    const nn = s.neighborMatchRate !== null
      ? `${(100 * s.neighborMatchRate).toFixed(1)}% (${s.neighborExcess >= 0 ? '+' : ''}${(100 * s.neighborExcess).toFixed(1)}%)`
      : '   n/a';
    console.log(
      `  ${String(i + 1).padStart(4)} | ${s.spaceGroup.padEnd(10)} | ${s.pointGroup.padEnd(6)} | ${String(s.order).padStart(5)} | ${String(s.nVariants).padStart(4)} | ${String(s.dof).padStart(3)} | ${s.bestDomainR2.toFixed(4)} | ${s.nullR2.toFixed(4)} | ${s.fStat.toFixed(3).padStart(6)} | ${s.breakingFStat.toFixed(3).padStart(7)} | ${nn}`
    );
  }
  const topSG = loc.subgroups[0];
  if (topSG && topSG.nVariants > 1) {
    console.log(`  Top local subgroup (${topSG.spaceGroup}) domain fractions: [${topSG.domainFractions.map(f => (100*f).toFixed(1)+'%').join(', ')}]`);
  }

  // 3. Statistical symmetry test on covariance vs folded mean
  const t2 = Date.now();
  const stat = statisticalSymmetryTest(configs, dim, A, basis, { parentTol: 0.25 });
  console.log(`\n[3] Statistical Symmetry Test (${Date.now() - t2} ms)`);
  console.log(`  RMS folded-mean disp: ${stat.rmsMeanDisp.toFixed(5)} Å  vs  RMS local disp: ${stat.rmsLocalDisp.toFixed(5)} Å  (ratio=${stat.meanToLocalRatio.toFixed(4)})`);
  console.log(`  Global statistical SG (cov tol=0.15): ${stat.statisticalSpaceGroup} (${stat.statisticalOrder}/${stat.parentOrder} ops), max ε_cov=${stat.maxEpsCov.toFixed(4)}`);

  // 4. Filtered minimal supercell
  const mini = filteredMinimalSupercell(spec, dim, A, basis, { snrThreshold: 2.5, minCoherentFrac: 0.1 });
  console.log(`\n[4] Mode-Filtered Minimal Supercell`);
  console.log(`  Active coherent k-modes: ${mini.nActiveK}, minimal supercell: ${mini.miniDim.join('x')}`);
  console.log(`  Ladder: ${mini.ladder.map(b => `${b.spaceGroup}[${b.from.toFixed(4)}..${b.to.toFixed(4)}]`).join(' -> ')}`);

  // 5. Domain-aligned averaging for top symmetry-breaking subgroup
  const topBrk = loc.subgroups
    .filter(s => s.breakingDoF > 0 && s.dof < loc.internalDoF)
    .sort((a, b) => b.breakingFStat - a.breakingFStat)[0];
  if (topBrk) {
    const t3 = Date.now();
    const aligned = domainAlignedStructure(configs, dim, A, basis, topBrk, { parentTol: 0.25 });
    console.log(`\n[5] Domain-Aligned Local Structure for ${topBrk.spaceGroup} (${Date.now() - t3} ms)`);
    console.log(`  Verified SG (tol=0.001 Å): ${aligned.verifiedSpaceGroup} (order ${aligned.verifiedOrder}), ${aligned.nVariants} twin variants × ${aligned.nNormalizerCosets} polarity coset(s)`);
    console.log(`  RMS shift: aligned=${aligned.symRms.toFixed(5)} Å  vs  standard folded=${aligned.foldedRms.toFixed(5)} Å  (max shift=${aligned.maxShift.toFixed(5)} Å)`);
    for (const es of aligned.elementStats) {
      console.log(`    ${es.el.padEnd(3)} (n=${String(es.count).padStart(2)}): aligned RMS=${es.symRms.toFixed(5)} Å, max=${es.maxShift.toFixed(5)} Å (folded=${es.foldedRms.toFixed(5)} Å)`);
    }
    const cif = generateAlignedCIF(aligned, { title: `GTS_${topBrk.spaceGroup}` });
    console.log(`  CIF generated: ${cif.split('\n').length} lines`);
  }
}

const root = '/Users/thyang/GitHub/rmc-phonon-dynamics';
analyzeDataset('GaTa4Se8 — 250 K (High-T cubic phase)', `${root}/data/250K_ini/GTS_250K.rmc6f`, `${root}/data/ensemble_20A_250K`, 15);
analyzeDataset('GaTa4Se8 — 5 K (Low-T distorted phase)', `${root}/data/5K_ini/GTS_5K.rmc6f`, `${root}/data/ensemble_20A_5K`, 15);
