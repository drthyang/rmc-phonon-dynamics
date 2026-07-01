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

// ── Multi-site F primitive over a supercell: nCells must be the TRUE cell count,
// not the (inflated) number of distinct sheared cell-indices. A GaTa₄Se₈-like
// 13-site F cell folds cleanly; a naive distinct-n count reported ~62 for a 2×2×2
// box (32 cells × sheared offsets) and wrongly flagged every basis site. ──
{
  const Fv = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]];
  const prim = [
    ['Ga', [0, 0, 0]],
    ['Ta', [0.6, 0.6, 0.6]], ['Ta', [0.6, 0.9, 0.9]], ['Ta', [0.9, 0.6, 0.9]], ['Ta', [0.9, 0.9, 0.6]],
    ['Se', [0.36, 0.36, 0.36]], ['Se', [0.36, 0.14, 0.14]], ['Se', [0.14, 0.36, 0.14]], ['Se', [0.14, 0.14, 0.36]],
    ['Se', [0.86, 0.86, 0.86]], ['Se', [0.86, 0.64, 0.64]], ['Se', [0.64, 0.86, 0.64]], ['Se', [0.64, 0.64, 0.86]],
  ];
  const M = [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]; // P = M (FCC conv→prim)
  const L = cellVectors(Aconv, M);
  const buildF = (dispA = 0) => {
    let s = 7; const rnd = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff - 0.5;
    const out = [];
    const Nc = 2;
    for (let i = 0; i < Nc; i++) for (let j = 0; j < Nc; j++) for (let k = 0; k < Nc; k++)
      for (const [el, f] of prim) for (const t of Fv)
        out.push({ pos: [(i + (f[0] + t[0]) % 1) * a + rnd() * 2 * dispA, (j + (f[1] + t[1]) % 1) * a + rnd() * 2 * dispA, (k + (f[2] + t[2]) % 1) * a + rnd() * 2 * dispA], element: el, mass: 1 });
    return out;
  };
  const r = relabelAtoms(buildF(0), L, { tol: 0.08 });
  ok(r.nBasis === 13, `multi-site F: folds to 13 primitive sites (got ${r.nBasis})`);
  ok(r.nCells === 2 * 2 * 2 * 4, `multi-site F: nCells = 32 true cells, not inflated distinct-n (got ${r.nCells})`);
  ok(r.issues.length === 0, `multi-site F: no spurious issues${r.issues.length ? ': ' + r.issues.slice(0, 2).join('; ') : ''}`);
  ok(r.maxResidual < 1e-9, `multi-site F: ideal fold → ~0 symmetry residual (got ${r.maxResidual.toExponential(1)} Å)`);

  // Symmetry residual tracks the static offset the fold averages over: displace
  // each atom by up to ±0.05 Å → residual should land in that ballpark.
  const rd = relabelAtoms(buildF(0.05), L, { tol: 0.08 });
  ok(rd.nBasis === 13, `disordered F still folds to 13 (got ${rd.nBasis})`);
  ok(rd.maxResidual > 0.01 && rd.maxResidual < 0.1, `disordered F: residual reflects ~0.05 Å offset (got ${rd.maxResidual.toFixed(3)} Å)`);
}

if (fails) { console.error(`\n❌ cells: ${fails} check(s) failed`); process.exit(1); }
console.log('\n✅ cells framework (Phase 0) OK');
