// Phase 1 tests for the computation-cell framework wired into the S(k) pipeline.
// Run: node test/cells_pipeline_test.mjs   (also part of `npm run validate`)
//
// These run the pure-JS equivalent of the WebGPU S(k) assembly (engine.js does
// the same outer product on the GPU reduction) so the cell labeling + phase +
// grouping can be validated deterministically in Node, without a GPU.
//
//   A. P = I reproduces the per-reference-number grouping byte-for-byte on the
//      validated reference fixture (zero regression for the default cell).
//   B. A primitive cell collapses the FCC conventional basis (4 sites → 1).
//   C. S(k) at an integer conventional reciprocal vector folds onto Γ, while a
//      conventional zone-boundary point (X at ½) does not — i.e. why the default
//      path must use conventional coordinates (Γ→X de-symmetrises).

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { buildCellLabeling, IDENT } from '../src/math/cells.js';
import { TWO_PI_PHASE } from '../src/constants.js';

const __dir = dirname(fileURLToPath(import.meta.url));
let fails = 0;
const ok = (cond, msg) => { console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${msg}`); if (!cond) fails++; };
const approx = (a, b, t = 1e-12) => Math.abs(a - b) <= t;

const Aconv = (v_super, dim) => [
  [v_super[0][0] / dim[0], v_super[0][1] / dim[1], v_super[0][2] / dim[2]],
  [v_super[1][0] / dim[0], v_super[1][1] / dim[1], v_super[1][2] / dim[2]],
  [v_super[2][0] / dim[0], v_super[2][1] / dim[1], v_super[2][2] / dim[2]],
];

// Ensemble mean of per-frame within-cell xyz.
function meanFrac(frames, n) {
  const mean = Array.from({ length: n }, () => [0, 0, 0]);
  for (const fr of frames) for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) mean[i][c] += fr[i][c];
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) mean[i][c] /= frames.length;
  return mean;
}

// Average cartesian positions = (cellIdx + meanFrac) · A_conv.
function avgPositions(cells, mean, A) {
  return mean.map((m, i) => {
    const f = [cells[i][0] + m[0], cells[i][1] + m[1], cells[i][2] + m[2]];
    return [
      f[0] * A[0][0] + f[1] * A[1][0] + f[2] * A[2][0],
      f[0] * A[0][1] + f[1] * A[1][1] + f[2] * A[2][1],
      f[0] * A[0][2] + f[1] * A[1][2] + f[2] * A[2][2],
    ];
  });
}

// Pure-JS S(k) grouped by basis site τ, phase indexed by cell n (engine.js math).
function skFromLabeling({ dim, v_super, frames }, masses, qfrac, lab) {
  const n = masses.length;
  const D = 3 * lab.nBasis;
  const mean = meanFrac(frames, n);
  const kvec = qfrac.map(q => q * TWO_PI_PHASE);
  const Sre = new Float64Array(D * D), Sim = new Float64Array(D * D);
  for (const fr of frames) {
    const A = new Float64Array(D), B = new Float64Array(D);
    for (let i = 0; i < n; i++) {
      const df = [0, 1, 2].map(c => (fr[i][c] - mean[i][c]) / dim[c]);
      const dc = [0, 1, 2].map(r => df[0] * v_super[0][r] + df[1] * v_super[1][r] + df[2] * v_super[2][r]);
      const phase = lab.cellN[i * 3] * kvec[0] + lab.cellN[i * 3 + 1] * kvec[1] + lab.cellN[i * 3 + 2] * kvec[2];
      const sm = Math.sqrt(masses[i]);
      const cp = Math.cos(phase) * sm, sp = Math.sin(phase) * sm;
      const s = lab.tau[i];
      for (let c = 0; c < 3; c++) { A[s * 3 + c] += dc[c] * cp; B[s * 3 + c] += dc[c] * sp; }
    }
    for (let t = 0; t < lab.nBasis; t++) { const nf = 1 / Math.sqrt(Math.max(lab.counts[t], 1)); for (let c = 0; c < 3; c++) { A[t * 3 + c] *= nf; B[t * 3 + c] *= nf; } }
    for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) {
      Sre[i * D + j] += A[i] * A[j] + B[i] * B[j];
      Sim[i * D + j] += B[i] * A[j] - A[i] * B[j];
    }
  }
  const inv = 1 / frames.length;
  for (let i = 0; i < D * D; i++) { Sre[i] *= inv; Sim[i] *= inv; }
  return { Sre, Sim, D };
}

// Baseline: the validated per-reference-number S(k) (copy of validate.mjs).
function computeSkByRN(inputs, qfrac) {
  const { dim, v_super, masses_by_rn, atoms, frames } = inputs;
  const n = atoms.length;
  const uniq = [...new Set(atoms.map(a => a.rn))].sort((a, b) => a - b);
  const T = uniq.length;
  const seg = new Map(uniq.map((rn, i) => [rn, i]));
  const counts = uniq.map(rn => atoms.filter(a => a.rn === rn).length);
  const kvec = qfrac.map(q => q * TWO_PI_PHASE);
  const mean = meanFrac(frames, n);
  const D = 3 * T;
  const Sre = new Float64Array(D * D), Sim = new Float64Array(D * D);
  for (const fr of frames) {
    const A = new Float64Array(D), B = new Float64Array(D);
    for (let i = 0; i < n; i++) {
      const a = atoms[i];
      const df = [0, 1, 2].map(c => (fr[i][c] - mean[i][c]) / dim[c]);
      const dc = [0, 1, 2].map(r => df[0] * v_super[0][r] + df[1] * v_super[1][r] + df[2] * v_super[2][r]);
      const phase = a.cell[0] * kvec[0] + a.cell[1] * kvec[1] + a.cell[2] * kvec[2];
      const sm = Math.sqrt(masses_by_rn[a.rn]);
      const cp = Math.cos(phase) * sm, sp = Math.sin(phase) * sm;
      const s = seg.get(a.rn);
      for (let c = 0; c < 3; c++) { A[s * 3 + c] += dc[c] * cp; B[s * 3 + c] += dc[c] * sp; }
    }
    for (let t = 0; t < T; t++) { const nf = 1 / Math.sqrt(Math.max(counts[t], 1)); for (let c = 0; c < 3; c++) { A[t * 3 + c] *= nf; B[t * 3 + c] *= nf; } }
    for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) {
      Sre[i * D + j] += A[i] * A[j] + B[i] * B[j];
      Sim[i * D + j] += B[i] * A[j] - A[i] * B[j];
    }
  }
  const inv = 1 / frames.length;
  for (let i = 0; i < D * D; i++) { Sre[i] *= inv; Sim[i] *= inv; }
  return { Sre, Sim, D };
}

const maxAbsDiff = (a, b, n) => { let m = 0; for (let i = 0; i < n; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };

// ── A. P = I reproduces per-RN grouping on the validated fixture ─────────────
console.log('\n[A] P = I reproduces per-reference-number S(k) (zero regression)');
{
  const ref = JSON.parse(readFileSync(join(__dir, 'reference.json'), 'utf8'));
  const inputs = ref.inputs;
  const n = inputs.atoms.length;
  const A = Aconv(inputs.v_super, inputs.dim);
  const mean = meanFrac(inputs.frames, n);
  const cells = inputs.atoms.map(a => a.cell);
  const pos = avgPositions(cells, mean, A);
  const elements = inputs.atoms.map(a => String(a.rn));
  const masses = inputs.atoms.map(a => inputs.masses_by_rn[a.rn]);
  const lab = buildCellLabeling(pos, elements, masses, A, IDENT, { tol: 0.08 });

  ok(lab.nBasis === 2, `nBasis = 2 (got ${lab.nBasis})`);
  ok(lab.issues.length === 0, `no labeling issues${lab.issues.length ? ': ' + lab.issues.join('; ') : ''}`);
  // cellN matches the file cell indices; τ partition matches the RN partition.
  let cellMatch = true, tauMatch = true;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) if (lab.cellN[i * 3 + c] !== cells[i][c]) cellMatch = false;
  }
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    if ((inputs.atoms[i].rn === inputs.atoms[j].rn) !== (lab.tau[i] === lab.tau[j])) tauMatch = false;
  }
  ok(cellMatch, 'cellN equals the file cell indices');
  ok(tauMatch, 'τ partition equals the reference-number partition');

  for (const name of Object.keys(inputs.kpoints)) {
    const q = inputs.kpoints[name];
    const base = computeSkByRN(inputs, q);
    const lk = skFromLabeling(inputs, masses, q, lab);
    // τ order may differ from sorted-RN order; here they coincide (rn 5 < rn 9,
    // frac 0.1 < 0.6). Compare directly.
    const d = Math.max(maxAbsDiff(base.Sre, lk.Sre, base.D * base.D), maxAbsDiff(base.Sim, lk.Sim, base.D * base.D));
    ok(approx(d, 0), `S(k) [${name}] equals per-RN baseline (|Δ|=${d.toExponential(2)})`);
  }
}

// ── B. Primitive cell collapses the FCC conventional basis ───────────────────
console.log('\n[B] FCC: conventional 4 sites vs primitive 1 site');
{
  const a = 4.04, N = 2;
  const AconvFCC = [[a, 0, 0], [0, a, 0], [0, 0, a]];
  const fccSites = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]];
  const pos = [], elements = [], masses = [];
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) for (let k = 0; k < N; k++) for (const s of fccSites) {
    pos.push([(i + s[0]) * a, (j + s[1]) * a, (k + s[2]) * a]); elements.push('Al'); masses.push(26.98);
  }
  const conv = buildCellLabeling(pos, elements, masses, AconvFCC, IDENT, { tol: 0.06 });
  ok(conv.nBasis === 4, `conventional → 4 basis sites / 12 branches (got ${conv.nBasis})`);
  const Pprim = [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]];
  const prim = buildCellLabeling(pos, elements, masses, AconvFCC, Pprim, { tol: 0.06 });
  ok(prim.nBasis === 1, `primitive → 1 basis site / 3 branches (got ${prim.nBasis})`);
}

// ── C. Conventional zone boundary vs folded integer point ────────────────────
console.log('\n[C] S(k=integer recip vector) folds onto Γ; conventional X (½) does not');
{
  // 2×1×1 supercell of a simple cubic cell with one site → exercises the phase
  // along x. Random-but-fixed displacements per frame.
  const a = 4.0, dim = [2, 1, 1];
  const v_super = [[a * 2, 0, 0], [0, a, 0], [0, 0, a]];
  const A = Aconv(v_super, dim);
  const cells = [[0, 0, 0], [1, 0, 0]];
  let s = 7; const rnd = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff - 0.5;
  const base = [0.0, 0.0, 0.0];
  const frames = [];
  for (let f = 0; f < 8; f++) frames.push(cells.map(() => [base[0] + 0.05 * rnd(), base[1] + 0.05 * rnd(), base[2] + 0.05 * rnd()]));
  const mean = meanFrac(frames, 2);
  const pos = avgPositions(cells, mean, A);
  const lab = buildCellLabeling(pos, ['Si', 'Si'], [28, 28], A, IDENT, { tol: 0.2 });
  ok(lab.nBasis === 1 && lab.nCells === 2, `2×1×1 → 1 site, 2 cells (got ${lab.nBasis}/${lab.nCells})`);

  const inputs = { dim, v_super, frames };
  const masses = [28, 28];
  const gamma = skFromLabeling(inputs, masses, [0, 0, 0], lab);
  const folded = skFromLabeling(inputs, masses, [1, 0, 0], lab);   // integer conv. recip vector
  const convX = skFromLabeling(inputs, masses, [0.5, 0, 0], lab);  // conventional zone boundary

  const dFold = Math.max(maxAbsDiff(gamma.Sre, folded.Sre, gamma.D * gamma.D), maxAbsDiff(gamma.Sim, folded.Sim, gamma.D * gamma.D));
  ok(approx(dFold, 0, 1e-9), `S([1,0,0]) == S(Γ)  — 2π periodicity, the spurious fold (|Δ|=${dFold.toExponential(2)})`);
  const dX = maxAbsDiff(gamma.Sre, convX.Sre, gamma.D * gamma.D);
  ok(dX > 1e-6, `S([0.5,0,0]) ≠ S(Γ)  — conventional X is a genuine point (|Δ|=${dX.toExponential(2)})`);
}

if (fails) { console.error(`\n❌ cell-framework pipeline: ${fails} check(s) failed`); process.exit(1); }
console.log('\n✅ cell-framework pipeline (Phase 1) OK');
