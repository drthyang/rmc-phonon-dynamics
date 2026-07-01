// web/test/symmetry_test.mjs
//
// Verify the pure-JS symmetry-operation finder against known space groups.
// Run: node test/symmetry_test.mjs   (part of `npm run validate`)

import { latticePointOps, findSpaceGroupOps, siteOrbits, symmetryLadder, wyckoffLetter } from '../src/math/symmetry.js';

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

  // Wyckoff orbits: F-43m GaTa₄Se₈ = Ga 4a + Ta 16e + Se 16e + Se 16e (52 sites).
  const orbits = siteOrbits(A, basis, r.ops, 0.1);
  const sig = orbits.map(o => `${o.element}${o.size}`).sort().join(' ');
  ok(orbits.length === 4, `GaTa₄Se₈ → 4 Wyckoff orbits (got ${orbits.length})`);
  ok(sig === 'Ga4 Se16 Se16 Ta16', `orbits = Ga×4, Ta×16, Se×16, Se×16 (got ${sig})`);
  // Site symmetry + Wyckoff letters: Ga 4a (-43m), Ta/Se 16e (3m).
  const label = (o) => `${o.element} ${o.size}${wyckoffLetter(216, 'F', o.size, o.site, o.rep) || `(${o.site})`}`;
  const labels = orbits.map(label).sort().join(', ');
  ok(labels === 'Ga 4a, Se 16e, Se 16e, Ta 16e', `Wyckoff labels: Ga 4a, Ta 16e, Se 16e, Se 16e (got ${labels})`);
}

// ── Site orbits on simpler cells ────────────────────────────────────────────
console.log('\nSite orbits:');
{
  const fcc = el([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]);
  const rf = findSpaceGroupOps(cubic, fcc, 0.1);
  const of = siteOrbits(cubic, fcc, rf.ops, 0.1);
  ok(of.length === 1 && of[0].size === 4, `FCC (all equivalent) → 1 orbit of 4 (got ${of.length}×${of[0]?.size})`);

  // Rock-salt AB: two elements → two orbits, one per sublattice.
  const rs = [{ el: 'A', frac: [0, 0, 0] }, { el: 'B', frac: [0.5, 0.5, 0.5] }];
  const rrs = findSpaceGroupOps(cubic, rs, 0.1);
  const ors = siteOrbits(cubic, rs, rrs.ops, 0.1);
  ok(ors.length === 2 && ors.every(o => o.size === 1), `BCC-type AB → 2 orbits (A, B) (got ${ors.length})`);
}

// ── Space-group identification (H–M symbol + number) ────────────────────────
console.log('\nSpace-group identification:');
{
  const sg = (A, basis, tol = 0.1) => { const r = findSpaceGroupOps(A, basis, tol); return `${r.spaceGroup} #${r.spaceGroupNumber}`; };
  ok(sg(cubic, el([[0, 0, 0]])) === 'Pm-3m #221', `simple cubic → Pm-3m #221 (got ${sg(cubic, el([[0, 0, 0]]))})`);
  const fcc = el([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]);
  ok(sg(cubic, fcc) === 'Fm-3m #225', `FCC → Fm-3m #225 (got ${sg(cubic, fcc)})`);
  ok(sg(cubic, el([[0, 0, 0], [0.5, 0.5, 0.5]])) === 'Im-3m #229', `BCC → Im-3m #229 (got ${sg(cubic, el([[0, 0, 0], [0.5, 0.5, 0.5]]))})`);
  ok(sg([[4, 0, 0], [0, 4, 0], [0, 0, 5.5]], el([[0, 0, 0]])) === 'P4/mmm #123', `tetragonal P → P4/mmm #123`);
  // GaTa₄Se₈ is F-43m (No. 216) — the real target.
  {
    const A = [[10.356, 0, 0], [0, 10.356, 0], [0, 0, 10.356]];
    const Fv = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]];
    const prim = [['Ga', [0, 0, 0]], ['Ta', [0.6, 0.6, 0.6]], ['Ta', [0.6, 0.9, 0.9]], ['Ta', [0.9, 0.6, 0.9]], ['Ta', [0.9, 0.9, 0.6]],
      ['Se', [0.36, 0.36, 0.36]], ['Se', [0.36, 0.14, 0.14]], ['Se', [0.14, 0.36, 0.14]], ['Se', [0.14, 0.14, 0.36]],
      ['Se', [0.86, 0.86, 0.86]], ['Se', [0.86, 0.64, 0.64]], ['Se', [0.64, 0.86, 0.64]], ['Se', [0.64, 0.64, 0.86]]];
    const basis = [];
    for (const [e, f] of prim) for (const t of Fv) basis.push({ el: e, frac: [(f[0] + t[0]) % 1, (f[1] + t[1]) % 1, (f[2] + t[2]) % 1] });
    ok(sg(A, basis) === 'F-43m #216', `GaTa₄Se₈ → F-43m #216 (got ${sg(A, basis)})`);
  }
  // A proper subgroup of the lattice (what the tolerance ladder produces) must be
  // classified from the actual operations, not the cubic metric.
  ok(sg(cubic, el([[0, 0, 0], [0.13, 0.27, 0.41]])) === 'P-1 #2', `generic 2-atom pair → P-1 #2 (got ${sg(cubic, el([[0, 0, 0], [0.13, 0.27, 0.41]]))})`);
}

// ── Tolerance ladder (P1 → … → full group) ──────────────────────────────────
console.log('\nSymmetry-vs-tolerance ladder:');
{
  const A = [[10.356, 0, 0], [0, 10.356, 0], [0, 0, 10.356]];
  const Fv = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]];
  const prim = [['Ga', [0, 0, 0]], ['Ta', [0.6, 0.6, 0.6]], ['Ta', [0.6, 0.9, 0.9]], ['Ta', [0.9, 0.6, 0.9]], ['Ta', [0.9, 0.9, 0.6]],
    ['Se', [0.36, 0.36, 0.36]], ['Se', [0.36, 0.14, 0.14]], ['Se', [0.14, 0.36, 0.14]], ['Se', [0.14, 0.14, 0.36]],
    ['Se', [0.86, 0.86, 0.86]], ['Se', [0.86, 0.64, 0.64]], ['Se', [0.64, 0.86, 0.64]], ['Se', [0.64, 0.64, 0.86]]];
  const mk = (nz) => { let s = 1; const r = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff - 0.5; const b = []; for (const [e, f] of prim) for (const t of Fv) b.push({ el: e, frac: [(f[0] + t[0]) % 1 + r() * nz, (f[1] + t[1]) % 1 + r() * nz, (f[2] + t[2]) % 1 + r() * nz] }); return b; };

  const clean = symmetryLadder(A, mk(0), 1.0);
  ok(clean.length === 1 && clean[0].spaceGroup === 'F-43m', `clean GaTa₄Se₈ → single F-43m rung from tol 0 (got ${clean.map(b => b.spaceGroup).join('→')})`);

  const noisy = symmetryLadder(A, mk(0.03), 1.0);
  ok(noisy[0].spaceGroup === 'P1' && noisy[noisy.length - 1].spaceGroup === 'F-43m',
    `noisy → ladder starts P1, ends F-43m (got ${noisy.map(b => b.spaceGroup).join('→')})`);
  ok(noisy.every((b, i) => i === 0 || b.nSpace > noisy[i - 1].nSpace), 'ladder is monotonic in operation count');
  ok(noisy.every((b, i) => i === 0 || b.from === noisy[i - 1].to), 'ladder brick ranges are contiguous');
}

console.log(`\n${fails === 0 ? '✅ symmetry finder OK' : `❌ ${fails} failed`}`);
process.exit(fails ? 1 : 0);
