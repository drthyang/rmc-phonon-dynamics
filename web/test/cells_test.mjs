// Phase 0 tests for the computation-cell framework (math/cells.js).
// Run: node test/cells_test.mjs   (also part of `npm run validate`)
import { cellVectors, tilesSupercell, relabelAtoms, det3, inv3, matMul3 } from '../src/math/cells.js';

let fails = 0;
const ok = (cond, msg) => { console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${msg}`); if (!cond) fails++; };
const approx = (a, b, t = 1e-9) => Math.abs(a - b) < t;

// ── matrix helpers ─────────────────────────────────────────────────────────
{
  const I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  ok(approx(det3(I), 1), 'det3(I) = 1');
  const M = [[2, 0, 0], [0, 3, 0], [0, 0, 4]];
  const Mi = inv3(M);
  ok(Mi && approx(Mi[0][0], 0.5) && approx(Mi[2][2], 0.25), 'inv3 diagonal');
  const P = matMul3(M, Mi);
  ok(approx(P[0][0], 1) && approx(P[1][1], 1) && approx(P[2][2], 1), 'M·M⁻¹ = I');
}

// ── FCC: conventional cubic (a) with 4 sites, in a 2×2×2 supercell ─────────
const a = 4.04;
const Aconv = [[a, 0, 0], [0, a, 0], [0, 0, a]];
const fccSites = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]];
const N = 2;                                  // 2×2×2 conventional cells
const Asuper = [[N * a, 0, 0], [0, N * a, 0], [0, 0, N * a]];

const buildAtoms = (jitter = 0) => {
  const atoms = [];
  const rnd = () => (Math.random() - 0.5) * 2 * jitter;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) for (let k = 0; k < N; k++)
    for (const s of fccSites)
      atoms.push({ pos: [(i + s[0]) * a + rnd(), (j + s[1]) * a + rnd(), (k + s[2]) * a + rnd()], element: 'Al', mass: 26.98 });
  return atoms;
};
const atoms = buildAtoms();
ok(atoms.length === 4 * N * N * N, `built ${4 * N * N * N} atoms`);

// Conventional cell P = I → 4 basis sites, 8 cells.
{
  const L = cellVectors(Aconv, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
  const t = tilesSupercell(L, Asuper);
  ok(t.ok && t.nCells === 8, `conventional tiles supercell (nCells=${t.nCells})`);
  const r = relabelAtoms(atoms, L);
  ok(r.nBasis === 4, `conventional → 4 basis sites (got ${r.nBasis})`);
  ok(r.nCells === 8, `conventional → 8 cells (got ${r.nCells})`);
  ok(r.issues.length === 0, `conventional → no issues${r.issues.length ? ': ' + r.issues.join('; ') : ''}`);
}

// Primitive FCC cell P = [[0,½,½],[½,0,½],[½,½,0]] → 1 basis site, 32 cells.
{
  const Pfcc = [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]];
  const L = cellVectors(Aconv, Pfcc);
  const t = tilesSupercell(L, Asuper);
  ok(t.ok && t.nCells === 32, `primitive tiles supercell (nCells=${t.nCells})`);
  const r = relabelAtoms(atoms, L);
  ok(r.nBasis === 1, `primitive → 1 basis site (got ${r.nBasis})`);
  ok(r.nCells === 32, `primitive → 32 cells (got ${r.nCells})`);
  ok(r.issues.length === 0, `primitive → no issues${r.issues.length ? ': ' + r.issues.join('; ') : ''}`);
}

// Custom 1×1×2 supercell of the conventional cell → 8 basis sites, 4 cells.
{
  const L = cellVectors(Aconv, [[1, 0, 0], [0, 1, 0], [0, 0, 2]]);
  const t = tilesSupercell(L, Asuper);
  ok(t.ok && t.nCells === 4, `1×1×2 tiles supercell (nCells=${t.nCells})`);
  const r = relabelAtoms(atoms, L);
  ok(r.nBasis === 8, `1×1×2 → 8 basis sites (got ${r.nBasis})`);
  ok(r.nCells === 4, `1×1×2 → 4 cells (got ${r.nCells})`);
}

// Robustness: small displacements (average ≈ ideal) still cluster correctly.
{
  const L = cellVectors(Aconv, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
  const r = relabelAtoms(buildAtoms(0.15), L, { tol: 0.08 });
  ok(r.nBasis === 4 && r.nCells === 8, `jittered conventional still 4 sites / 8 cells (got ${r.nBasis}/${r.nCells})`);
}

if (fails) { console.error(`\n❌ cells: ${fails} check(s) failed`); process.exit(1); }
console.log('\n✅ cells framework (Phase 0) OK');
