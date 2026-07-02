// web/test/symmetrize_gauge_test.mjs
//
// Fixed-point test for the S(k) symmetrization phase conventions against the
// ACTUAL cell labeling (buildCellLabeling), including its boundary snap.
//
// Physics: if every frame of the ensemble is EXACTLY invariant under a space-
// group operation g, then v(k) built from the production labels satisfies
// v = Γ(g) v identically, so S(k) = ⟨vv†⟩ must be an exact fixed point of the
// little-group projection — at every k, to machine precision. Any change made
// by symmetrizeSk is a convention/gauge error, not statistics.
//
// The regression this guards: relabelAtoms' boundary snap assigns cell labels
// n = round(f) (not round(f − frac)) for sites with a frac component in
// (1−snap, 1), so label-space site fractions are frac−1 there. Feeding the
// wrapped tauFrac to operationReps instead of tauFracLabel corrupts S(k) by a
// one-lattice-vector phase on every block linking snapped and un-snapped orbit
// members — 100% distortion at zone-boundary k for the C4 orbit below (which
// mixes sites at 0.90/0.75-type fracs with 0.10/0.25 partners; that is exactly
// the GaTa4Se8 situation, where Se sites sit at within-cell frac ≈ 0.886).
//
// A 2-site orbit cannot catch this (the gauge unitary commutes through an
// order-2 projection), hence the 4-site C4 orbit.

import { buildCellLabeling, IDENT } from '../src/math/cells.js';
import { operationReps, symmetrizeSk } from '../src/math/symmetrize.js';

let fails = 0;
const ok = (c, m) => { console.log(`  ${c ? 'ok  ' : 'FAIL'} ${m}`); if (!c) fails++; };

const a = 4, L = 6;
const Aconv = [[a, 0, 0], [0, a, 0], [0, 0, a]];
// C4z orbit: two members in the boundary-snap zone (component > 0.88).
const f = [[0.9, 0.25, 0], [0.75, 0.9, 0], [0.1, 0.75, 0], [0.25, 0.1, 0]];
const R = [[0, -1, 0], [1, 0, 0], [0, 0, 1]];   // C4z (fractional = cartesian here)
const I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
const mod = (x, n) => ((x % n) + n) % n;
const applyR = (R, v) => [0, 1, 2].map(i => R[i][0] * v[0] + R[i][1] * v[1] + R[i][2] * v[2]);

// atoms on an L^3 grid (site-inner ordering so idxOf is simple)
const atoms = [];
for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) for (let k2 = 0; k2 < L; k2++)
  for (let s = 0; s < 4; s++) atoms.push({ cell: [i, j, k2], s });
const idxOf = (n, s) => ((n[0] * L + n[1]) * L + n[2]) * 4 + s;
const avgPos = atoms.map(({ cell, s }) => [(cell[0] + f[s][0]) * a, (cell[1] + f[s][1]) * a, (cell[2] + f[s][2]) * a]);
const lab = buildCellLabeling(avgPos, atoms.map(() => 'X'), atoms.map(() => 1), Aconv, IDENT, { tol: 0.08 });

console.log('\nLabeling of the C4 orbit (boundary snap):');
ok(lab.nBasis === 4 && lab.nCells === L * L * L, `4 basis sites × ${L * L * L} cells (got ${lab.nBasis}/${lab.nCells})`);
// The snap must shift the labels of the 0.9-component sites (that is the gauge
// this test exercises) and tauFracLabel must expose it as frac−1.
const snapped = lab.tauFrac.filter((fr, t) => fr.some((x, c) => x > 0.88 && Math.abs(lab.tauFracLabel[t][c] - (x - 1)) < 1e-12)).length;
ok(snapped === 2, `2 sites in the snap zone with label-space frac = frac − 1 (got ${snapped})`);

// C4 group action on atoms: (n, s) -> (R·n + Δfrac(s) mod L, perm(s))
const perm = [1, 2, 3, 0];
const Dfrac = f.map((fs, s) => {
  const img = applyR(R, fs);
  return [0, 1, 2].map(c => Math.round(img[c] - f[perm[s]][c]));
});

// Frames that are EXACTLY C4-invariant fields: draw site-0 displacements, then
// propagate u(g·atom) = R_cart · u(atom) around the orbit.
let seed = 12345;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff - 0.5; };
const NF = 25;
const D = 3 * lab.nBasis;
const k = [0.5, 0.5, 1 / 3];   // commensurate with L=6; zone-boundary components
const TWO_PI = 2 * Math.PI;
const Sre = new Float64Array(D * D), Sim = new Float64Array(D * D);
for (let fr = 0; fr < NF; fr++) {
  const u = new Float64Array(atoms.length * 3);
  for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) for (let k2 = 0; k2 < L; k2++) {
    let n = [i, j, k2], s = 0, vec = [rnd(), rnd(), rnd()];
    u.set(vec, idxOf(n, s) * 3);
    for (let step = 0; step < 3; step++) {
      const rn = applyR(R, n);
      n = [0, 1, 2].map(c => mod(rn[c] + Dfrac[s][c], L));
      vec = applyR(R, vec);
      s = perm[s];
      u.set(vec, idxOf(n, s) * 3);
    }
  }
  // v(k) exactly as the kernel: phase from the production labels lab.cellN
  const A = new Float64Array(D), B = new Float64Array(D);
  for (let idx = 0; idx < atoms.length; idx++) {
    const t = lab.tau[idx];
    const ph = TWO_PI * (k[0] * lab.cellN[idx * 3] + k[1] * lab.cellN[idx * 3 + 1] + k[2] * lab.cellN[idx * 3 + 2]);
    const cp = Math.cos(ph), sp = Math.sin(ph);
    for (let c = 0; c < 3; c++) { A[t * 3 + c] += u[idx * 3 + c] * cp; B[t * 3 + c] += u[idx * 3 + c] * sp; }
  }
  for (let i2 = 0; i2 < D; i2++) for (let j2 = 0; j2 < D; j2++) {
    Sre[i2 * D + j2] += A[i2] * A[j2] + B[i2] * B[j2];
    Sim[i2 * D + j2] += B[i2] * A[j2] - A[i2] * B[j2];
  }
}
for (let i2 = 0; i2 < D * D; i2++) { Sre[i2] /= NF; Sim[i2] /= NF; }

// Reps the way pipeline.js builds them: from the LABEL-space fracs.
const R2 = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]], R3 = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]];
const opsC4 = [{ R: I3, t: [0, 0, 0] }, { R, t: [0, 0, 0] }, { R: R2, t: [0, 0, 0] }, { R: R3, t: [0, 0, 0] }];
const tauBasis = lab.tauFracLabel.map((frac, t) => ({ frac, el: lab.tauElement[t] }));
const reps = operationReps(lab.L, tauBasis, opsC4);
ok(reps.length === 4, `all 4 C4 ops map the label-space basis onto itself (got ${reps.length})`);

console.log('\nFixed point of the little-group projection (exactly C4-symmetric frames):');
const out = symmetrizeSk(Sre, Sim, lab.nBasis, k, reps);
let maxd = 0;
for (let i2 = 0; i2 < D * D; i2++) maxd = Math.max(maxd, Math.abs(out.re[i2] - Sre[i2]), Math.abs(out.im[i2] - Sim[i2]));
const scale = Math.max(...Sre.map(Math.abs));
ok(maxd / scale < 1e-9, `symmetrizeSk leaves the symmetric S(k) unchanged at k=[½,½,⅓] (rel |Δ| = ${(maxd / scale).toExponential(2)})`);

// Regression direction: with the WRAPPED fracs (the old bug) the same projection
// must NOT be a fixed point — proves this test has teeth.
const repsWrapped = operationReps(lab.L, lab.tauFrac.map((frac, t) => ({ frac, el: lab.tauElement[t] })), opsC4);
const bad = symmetrizeSk(Sre, Sim, lab.nBasis, k, repsWrapped);
let maxdBad = 0;
for (let i2 = 0; i2 < D * D; i2++) maxdBad = Math.max(maxdBad, Math.abs(bad.re[i2] - Sre[i2]), Math.abs(bad.im[i2] - Sim[i2]));
ok(maxdBad / scale > 0.1, `wrapped-frac reps corrupt S(k) (rel |Δ| = ${(maxdBad / scale).toExponential(2)}) — gauge matters here`);

console.log(`\n${fails === 0 ? '✅ symmetrize gauge fixed-point OK' : `❌ ${fails} failed`}`);
process.exit(fails ? 1 : 0);
