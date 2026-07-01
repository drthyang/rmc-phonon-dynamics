// web/test/symmetrize_test.mjs
//
// Validate the S(k) symmetrization machinery (math/symmetrize.js). These are
// convention-pinning checks: the little-group average must be a Hermitian-
// preserving idempotent projector that POOLS equivalent sites. Run: part of
// `npm run validate`.

import { operationReps, symmetrizeSk } from '../src/math/symmetrize.js';

let fails = 0;
const ok = (c, m) => { console.log(`  ${c ? 'ok  ' : 'FAIL'} ${m}`); if (!c) fails++; };
const I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

const L = [[4, 0, 0], [0, 4, 0], [0, 0, 4]];
// Two sites swapped by the body-centering translation {I | ½½½} (a group of order 2).
const basis = [{ frac: [0, 0, 0] }, { frac: [0.5, 0.5, 0.5] }];
const ops = [{ R: I3, t: [0, 0, 0] }, { R: I3, t: [0.5, 0.5, 0.5] }];
const reps = operationReps(L, basis, ops);

console.log('\nOperation representations:');
ok(reps.length === 2, `both ops map the basis onto itself (got ${reps.length})`);
ok(reps[1].perm[0] === 1 && reps[1].perm[1] === 0, `{I|½½½} swaps the two sites (perm ${reps[1].perm})`);

// Build S: site-0 block = diag(1), site-1 block = diag(2), zero off-diagonal.
const D = 6;
const mkS = () => {
  const re = new Float64Array(D * D), im = new Float64Array(D * D);
  for (let a = 0; a < 3; a++) { re[a * D + a] = 1; re[(a + 3) * D + (a + 3)] = 2; }
  return { re, im };
};
const isHerm = (re, im) => { for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) { if (Math.abs(re[i * D + j] - re[j * D + i]) > 1e-9) return false; if (Math.abs(im[i * D + j] + im[j * D + i]) > 1e-9) return false; } return true; };

console.log('\nPooling of equivalent sites (k = Γ):');
{
  const S = mkS();
  const r = symmetrizeSk(S.re, S.im, 2, [0, 0, 0], reps);
  // both diagonal blocks should become the average (1+2)/2 = 1.5
  ok(Math.abs(r.re[0] - 1.5) < 1e-9 && Math.abs(r.re[3 * D + 3] - 1.5) < 1e-9,
    `equivalent sites pooled → both diagonal blocks 1.5 (got ${r.re[0].toFixed(3)}, ${r.re[3 * D + 3].toFixed(3)})`);
  ok(isHerm(r.re, r.im), 'symmetrized S is Hermitian');
}

console.log('\nProjector properties (random Hermitian S):');
{
  let s = 7; const rnd = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff - 0.5;
  const re = new Float64Array(D * D), im = new Float64Array(D * D);
  for (let i = 0; i < D; i++) for (let j = i; j < D; j++) {
    const a = rnd(), b = i === j ? 0 : rnd();
    re[i * D + j] = a; re[j * D + i] = a; im[i * D + j] = b; im[j * D + i] = -b;
  }
  const k = [0.3, 0.1, 0]; // generic k — little group here is still the full order-2 group (R=I)
  const r1 = symmetrizeSk(re, im, 2, k, reps);
  ok(isHerm(r1.re, r1.im), 'random Hermitian → symmetrized still Hermitian');
  const r2 = symmetrizeSk(r1.re, r1.im, 2, k, reps);
  let maxd = 0; for (let i = 0; i < D * D; i++) maxd = Math.max(maxd, Math.abs(r1.re[i] - r2.re[i]), Math.abs(r1.im[i] - r2.im[i]));
  ok(maxd < 1e-9, `idempotent: symmetrize(symmetrize(S)) = symmetrize(S) (|Δ|=${maxd.toExponential(1)})`);
}

console.log('\nTrivial group is a no-op:');
{
  const S = mkS();
  const r = symmetrizeSk(S.re, S.im, 2, [0, 0, 0], [reps[0]]); // identity only
  ok(r.re === S.re, 'identity-only little group → S returned unchanged (byte-identical)');
}

console.log('\nRotation enforces degeneracy (C4z on one site):');
{
  // One site at the origin, full 4-fold about z. Symmetrizing its 3×3 covariance
  // must make it invariant under C4z: xx = yy, and xy = xz = yz = 0.
  const b1 = [{ frac: [0, 0, 0] }];
  const R1 = [[0, -1, 0], [1, 0, 0], [0, 0, 1]];
  const R2 = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]];
  const R3 = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]];
  const rot = operationReps(L, b1, [{ R: I3, t: [0, 0, 0] }, { R: R1, t: [0, 0, 0] }, { R: R2, t: [0, 0, 0] }, { R: R3, t: [0, 0, 0] }]);
  let s = 3; const rnd = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff - 0.5;
  const re = new Float64Array(9), im = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = i; j < 3; j++) { const a = rnd() + 1; re[i * 3 + j] = a; re[j * 3 + i] = a; }
  const r = symmetrizeSk(re, im, 1, [0, 0, 0], rot);
  ok(Math.abs(r.re[0] - r.re[4]) < 1e-9, `C4z → xx = yy (degenerate) (${r.re[0].toFixed(3)} = ${r.re[4].toFixed(3)})`);
  ok(Math.abs(r.re[1]) < 1e-9 && Math.abs(r.re[2]) < 1e-9 && Math.abs(r.re[5]) < 1e-9, 'C4z → off-diagonal xy, xz, yz = 0');
}

console.log(`\n${fails === 0 ? '✅ symmetrize OK' : `❌ ${fails} failed`}`);
process.exit(fails ? 1 : 0);
