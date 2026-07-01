// web/test/symmetry_test.mjs
//
// Verify the pure-JS symmetry-operation finder against known space groups.
// Run: node test/symmetry_test.mjs   (part of `npm run validate`)

import { latticePointOps, findSpaceGroupOps } from '../src/math/symmetry.js';

let fails = 0;
const ok = (c, m) => { console.log(`  ${c ? 'ok  ' : 'FAIL'} ${m}`); if (!c) fails++; };

const a = 4.0;
const cubic = [[a, 0, 0], [0, a, 0], [0, 0, a]];
const el = (frac) => frac.map(f => ({ el: 'A', frac: f }));

// ── Point group of the cubic lattice = m-3m, order 48 ───────────────────────
console.log('\nLattice point operations:');
ok(latticePointOps(cubic).length === 48, `cubic lattice → 48 point ops (got ${latticePointOps(cubic).length})`);
{
  const tet = [[a, 0, 0], [0, a, 0], [0, 0, 1.3 * a]];       // c ≠ a → 4/mmm, order 16
  ok(latticePointOps(tet).length === 16, `tetragonal lattice → 16 point ops (got ${latticePointOps(tet).length})`);
  const orc = [[a, 0, 0], [0, 1.2 * a, 0], [0, 0, 1.5 * a]]; // mmm, order 8
  ok(latticePointOps(orc).length === 8, `orthorhombic lattice → 8 point ops (got ${latticePointOps(orc).length})`);
}

// ── Full space groups (point group × centering cosets) ──────────────────────
console.log('\nSpace-group operations:');
{
  const P = findSpaceGroupOps(cubic, el([[0, 0, 0]]), 0.1);           // Pm-3m
  ok(P.nPoint === 48 && P.nSpace === 48, `simple cubic P → 48 ops, 48 point (got ${P.nSpace}/${P.nPoint})`);
  ok(P.maxResidual < 1e-9, `simple cubic P → 0 residual (got ${P.maxResidual.toExponential(1)})`);

  const fcc = el([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]);
  const F = findSpaceGroupOps(cubic, fcc, 0.1);                       // Fm-3m
  ok(F.nPoint === 48 && F.nSpace === 192, `FCC → 192 ops (48×4 centering), 48 point (got ${F.nSpace}/${F.nPoint})`);

  const bcc = el([[0, 0, 0], [0.5, 0.5, 0.5]]);
  const I = findSpaceGroupOps(cubic, bcc, 0.1);                       // Im-3m
  ok(I.nPoint === 48 && I.nSpace === 96, `BCC → 96 ops (48×2 centering), 48 point (got ${I.nSpace}/${I.nPoint})`);
}

// ── Symmetry lowers with a distortion / with a lower-symmetry basis ─────────
console.log('\nSymmetry vs structure:');
{
  // A cubic cell but a basis that breaks the 3-fold (atoms only along z) → tetragonal-ish.
  const brokenBasis = el([[0, 0, 0], [0, 0, 0.5]]);
  const low = findSpaceGroupOps(cubic, brokenBasis, 0.05);
  ok(low.nPoint === 16, `z-only basis in a cubic box → 4/mmm point group, 16 (got ${low.nPoint})`);

  // Tolerance traces symmetry: a small z-displacement is still "cubic" at loose tol,
  // lower at tight tol.
  const nudged = el([[0, 0, 0], [0.5, 0.5, 0.02], [0.5, 0, 0.5], [0, 0.5, 0.5]]);
  const loose = findSpaceGroupOps(cubic, nudged, 0.3);
  const tight = findSpaceGroupOps(cubic, nudged, 0.02);
  ok(loose.nSpace > tight.nSpace, `looser tol keeps more symmetry (${loose.nSpace} > ${tight.nSpace})`);
  ok(loose.maxResidual > tight.maxResidual - 1e-12, `looser tol admits a larger residual (${loose.maxResidual.toFixed(3)} ≥ ${tight.maxResidual.toFixed(3)} Å)`);
}

// ── Real multi-element structure: GaTa₄Se₈ is F-43m (No. 216) → Td (order 24) ─
console.log('\nGaTa₄Se₈ (F-43m):');
{
  const A = [[10.356, 0, 0], [0, 10.356, 0], [0, 0, 10.356]];
  const Fv = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]];
  const prim = [
    ['Ga', [0, 0, 0]],
    ['Ta', [0.6, 0.6, 0.6]], ['Ta', [0.6, 0.9, 0.9]], ['Ta', [0.9, 0.6, 0.9]], ['Ta', [0.9, 0.9, 0.6]],
    ['Se', [0.36, 0.36, 0.36]], ['Se', [0.36, 0.14, 0.14]], ['Se', [0.14, 0.36, 0.14]], ['Se', [0.14, 0.14, 0.36]],
    ['Se', [0.86, 0.86, 0.86]], ['Se', [0.86, 0.64, 0.64]], ['Se', [0.64, 0.86, 0.64]], ['Se', [0.64, 0.64, 0.86]],
  ];
  const basis = [];
  for (const [el, f] of prim) for (const t of Fv) basis.push({ el, frac: [(f[0] + t[0]) % 1, (f[1] + t[1]) % 1, (f[2] + t[2]) % 1] });
  const r = findSpaceGroupOps(A, basis, 0.1);
  ok(r.nPoint === 24, `GaTa₄Se₈ → point group -43m (Td), order 24 (got ${r.nPoint})`);
  ok(r.nSpace === 96, `GaTa₄Se₈ → 96 conventional ops = 24 × 4 F-centering (got ${r.nSpace})`);
  ok(r.maxResidual < 1e-9, `GaTa₄Se₈ ideal → 0 residual (got ${r.maxResidual.toExponential(1)})`);
}

console.log(`\n${fails === 0 ? '✅ symmetry finder OK' : `❌ ${fails} failed`}`);
process.exit(fails ? 1 : 0);
