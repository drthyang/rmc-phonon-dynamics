// web/test/synthetic_dispersion_test.mjs
//
// End-to-end physical benchmark (not just a numerical-regression check): build a
// synthetic thermal ensemble from a KNOWN analytic dispersion relation, run it
// through the actual production math (buildCellLabeling, the Sk covariance
// assembly matching engine.js/Sk_kernel.wgsl exactly, eigh, eigenvaluesToMev),
// and confirm the extracted phonon energies + eigenvector polarizations
// reproduce the input model. This is the same validation strategy Goodwin et
// al. (PRL 93, 075502 (2004)) used for MgO: generate MD-like configs from a
// known force-constant model, then check the RMC/Dove extraction method
// reproduces it (their Fig. 2a) — except here the target is exact, not
// experimental, so we can assert a numeric tolerance.
//
// ── The model ────────────────────────────────────────────────────────────
// Simple-cubic, 1 atom/cell, L^3 supercell, fully decoupled Cartesian
// polarizations (off-diagonal dynamical-matrix terms are exactly zero):
//   E_alpha(q) = Emax * |sin(pi * q_alpha)|,  alpha = x,y,z
// (q_alpha is the conventional-cell fractional coordinate; this is the
// textbook nearest-neighbor monatomic-chain dispersion, applied independently
// along each axis with no cross-coupling.)
//
// ── Why this is a real test, not a tautology ────────────────────────────
// The generation and extraction formulas ARE mathematical inverses of each
// other (target lambda(q) = T*(ENERGY_CONV/E(q))^2, extraction inverts
// E = ENERGY_CONV*sqrt(T/lambda)) - that's what "equipartition-consistent
// extraction" means, and it's exactly how Goodwin validated against MD. What's
// actually being tested is the REAL numerical work in between: building a
// genuinely spatially-correlated 3-D displacement field (independent 1-D
// sine/cosine mode synthesis per axis, replicated over the transverse L^2
// grid — derived below from the general harmonic relation S(k) = kB*T*D(k)^-1),
// summing the Bloch phase from the ACTUAL production cell-labeling + kernel
// code at arbitrary OFF-symmetry q (not just Gamma / integer folds, which is
// all the existing tests exercise), and checking the Hermitian eigh correctly
// separates 3 simultaneously-present, non-degenerate branches by both
// eigenvalue (frequency) AND eigenvector (polarization).
//
// ── Derivation of the per-chain generator (equal-time covariance) ──────────
// For a classical harmonic crystal, ⟨u_a(R) u_b(R')⟩ = (kT/N) Σ_k [D(k)^-1]_ab
// e^{ik.(R-R')}. Here D is diagonal (alpha decoupled) and D_aa(k) depends only
// on q_alpha, so for alpha=x the k-sum factorizes and the ky,kz sums each
// collapse to L * delta(Ry,Ry') * delta(Rz,Rz') (discrete-Fourier identity).
// Result: u_x is correlated only WITHIN a fixed (Ry,Rz) row — i.e. the crystal
// decomposes into L^2 independent 1-D chains along x (and likewise y,z) — and
// each chain's covariance is exactly the real sine/cosine (paired j / L-j,
// self-paired Nyquist) synthesis implemented in genChain() below. Verified by
// hand (see chat) that the sqrt(2/L) [paired] and sqrt(1/L) [Nyquist]
// coefficients reproduce the target (1/L) Sum_j lambda(j) cos(2*pi*j*Delta/L)
// exactly in expectation.

import { buildCellLabeling, IDENT } from '../src/math/cells.js';
import { eigh, eigenvaluesToMev } from '../src/math/diagonalize.js';
import { TWO_PI_PHASE, ENERGY_CONV } from '../src/constants.js';

let fails = 0;
const ok = (cond, msg) => { console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${msg}`); if (!cond) fails++; };

// ── deterministic RNG (mulberry32 + Box-Muller) ─────────────────────────────
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function makeGaussian(rng) {
  let spare = null;
  return () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    let u1 = 0; while (u1 <= 1e-12) u1 = rng();
    const u2 = rng();
    const r = Math.sqrt(-2 * Math.log(u1)), theta = 2 * Math.PI * u2;
    spare = r * Math.sin(theta);
    return r * Math.cos(theta);
  };
}

// ── model ────────────────────────────────────────────────────────────────
const L = 10;                 // supercell (matches Goodwin's 10x10x10 MgO example)
const a = 4.0;                // A, conventional cubic lattice parameter
const MASS = 28.0;             // amu
const T_KELVIN = 300;
const E_MAX = 30.0;            // meV, zone-boundary energy for each branch

const targetE = (qfrac) => E_MAX * Math.abs(Math.sin(Math.PI * qfrac));
// Invert E = ENERGY_CONV*sqrt(T/lambda)  =>  lambda = T*(ENERGY_CONV/E)^2
const targetLambda = (qfrac) => {
  const E = targetE(qfrac);
  return T_KELVIN * (ENERGY_CONV / E) ** 2;   // amu.A^2, this chain's S(k) eigenvalue target
};

// Per-chain "displacement variance needed at each discrete j" (j=1..L-1; j=0 excluded).
const varByJ = new Float64Array(L);
for (let j = 1; j < L; j++) varByJ[j] = targetLambda(j / L) / MASS;

// Generate ONE real 1-D chain frame (length L) with the exact target covariance
// in expectation: (1/L) sum_{j=1}^{L-1} lambda(j)/m * cos(2*pi*j*delta/L).
function genChain(rng, gauss, out /* Float64Array(L), overwritten */) {
  out.fill(0);
  const halfL = Math.floor((L - 1) / 2);
  for (let j = 1; j <= halfL; j++) {
    const amp = Math.sqrt(varByJ[j]) * Math.sqrt(2 / L);
    const g1 = gauss(), g2 = gauss();
    for (let n = 0; n < L; n++) {
      const th = (2 * Math.PI * j * n) / L;
      out[n] += amp * (g1 * Math.cos(th) + g2 * Math.sin(th));
    }
  }
  if (L % 2 === 0) {
    const jN = L / 2;
    const amp = Math.sqrt(varByJ[jN]) * Math.sqrt(1 / L);
    const gN = gauss();
    for (let n = 0; n < L; n++) out[n] += amp * gN * (n % 2 === 0 ? 1 : -1);
  }
}

// ── build the L^3 lattice (1 basis site) ────────────────────────────────────
const N = L * L * L;
const Aconv = [[a, 0, 0], [0, a, 0], [0, 0, a]];
const cellIdx = new Int32Array(N * 3);
const avgPos = new Array(N);
const elements = new Array(N).fill('Si');
const masses = new Array(N).fill(MASS);
{
  let idx = 0;
  for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) for (let k = 0; k < L; k++) {
    cellIdx[idx * 3] = i; cellIdx[idx * 3 + 1] = j; cellIdx[idx * 3 + 2] = k;
    avgPos[idx] = [i * a, j * a, k * a];
    idx++;
  }
}
const lab = buildCellLabeling(avgPos, elements, masses, Aconv, IDENT, { tol: 0.1 });
console.log(`\n[synthetic] ${L}^3 = ${N}-atom decoupled-polarization lattice`);
ok(lab.nBasis === 1 && lab.nCells === N, `single basis site, ${N} cells (got ${lab.nBasis}/${lab.nCells})`);

// atom -> (chain axis, chain id, position along chain) for the independent-chain generator
// x-chains indexed by (j,k); y-chains by (i,k); z-chains by (i,j).
const atomIdxOf = (i, j, k) => (i * L + j) * L + k;

// ── S(k) accumulation, matching engine.js exactly (Sk_kernel.wgsl math) ────
function accumulateSk(dispCart /* Float64Array(N*3) this frame's Cartesian A displacement */, qfrac) {
  const kvec = qfrac.map((q) => q * TWO_PI_PHASE);
  const A = new Float64Array(3), B = new Float64Array(3);
  for (let idx = 0; idx < N; idx++) {
    const phase = lab.cellN[idx * 3] * kvec[0] + lab.cellN[idx * 3 + 1] * kvec[1] + lab.cellN[idx * 3 + 2] * kvec[2];
    const sm = Math.sqrt(masses[idx]);
    const cp = Math.cos(phase) * sm, sp = Math.sin(phase) * sm;
    for (let c = 0; c < 3; c++) {
      A[c] += dispCart[idx * 3 + c] * cp;
      B[c] += dispCart[idx * 3 + c] * sp;
    }
  }
  const nf = 1 / Math.sqrt(N);
  for (let c = 0; c < 3; c++) { A[c] *= nf; B[c] *= nf; }
  const Sre = new Float64Array(9), Sim = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    Sre[i * 3 + j] = A[i] * A[j] + B[i] * B[j];
    Sim[i * 3 + j] = B[i] * A[j] - A[i] * B[j];
  }
  return { Sre, Sim };
}

const N_FRAMES = 2500;
const seed = 20260630;
const rng = mulberry32(seed);
const gauss = makeGaussian(rng);

// q-points: generic (all 3 components distinct & nonzero) so all 3 branches are
// simultaneously present and non-degenerate — a genuine 3-way separation test.
const testPoints = [
  [1, 2, 3], [2, 4, 1], [4, 1, 3], [3, 4, 2], [5, 2, 1],   // last one exercises the Nyquist point (0.5)
].map((v) => v.map((x) => x / L));

console.log(`\n[synthetic] accumulating S(k) over ${N_FRAMES} synthetic frames at ${testPoints.length} generic q-points...`);

const SkSums = testPoints.map(() => ({ Sre: new Float64Array(9), Sim: new Float64Array(9) }));
const chainBuf = new Float64Array(L);
const dispCart = new Float64Array(N * 3);

for (let f = 0; f < N_FRAMES; f++) {
  // x-polarization: L^2 independent chains along x, one per (j,k)
  for (let j = 0; j < L; j++) for (let k = 0; k < L; k++) {
    genChain(rng, gauss, chainBuf);
    for (let i = 0; i < L; i++) dispCart[atomIdxOf(i, j, k) * 3 + 0] = chainBuf[i];
  }
  // y-polarization: L^2 independent chains along y, one per (i,k)
  for (let i = 0; i < L; i++) for (let k = 0; k < L; k++) {
    genChain(rng, gauss, chainBuf);
    for (let j = 0; j < L; j++) dispCart[atomIdxOf(i, j, k) * 3 + 1] = chainBuf[j];
  }
  // z-polarization: L^2 independent chains along z, one per (i,j)
  for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) {
    genChain(rng, gauss, chainBuf);
    for (let k = 0; k < L; k++) dispCart[atomIdxOf(i, j, k) * 3 + 2] = chainBuf[k];
  }

  for (let p = 0; p < testPoints.length; p++) {
    const { Sre, Sim } = accumulateSk(dispCart, testPoints[p]);
    for (let i = 0; i < 9; i++) { SkSums[p].Sre[i] += Sre[i]; SkSums[p].Sim[i] += Sim[i]; }
  }
}
for (let p = 0; p < testPoints.length; p++) for (let i = 0; i < 9; i++) { SkSums[p].Sre[i] /= N_FRAMES; SkSums[p].Sim[i] /= N_FRAMES; }

// ── extract + compare ───────────────────────────────────────────────────────
console.log('\n[synthetic] extracted vs analytic dispersion (E in meV, tol 20% — Monte Carlo, N=2500 frames)');
let worstRel = 0;
for (let p = 0; p < testPoints.length; p++) {
  const q = testPoints[p];
  const { Sre, Sim } = SkSums[p];
  const { eigenvalues, eigenvectors } = eigh(Sre, Sim, 3);
  const energies = eigenvaluesToMev(eigenvalues, T_KELVIN, ENERGY_CONV);

  // eigh sorts EIGENVALUES ascending; E ~ 1/sqrt(lambda), so ascending lambda
  // means DESCENDING energy. Sort analytic targets descending to match.
  const targets = [
    { axis: 0, E: targetE(q[0]) },
    { axis: 1, E: targetE(q[1]) },
    { axis: 2, E: targetE(q[2]) },
  ].sort((a, b) => b.E - a.E);

  console.log(`  q=[${q.map((x) => x.toFixed(2)).join(',')}]`);
  for (let m = 0; m < 3; m++) {
    const got = Math.abs(energies[m]);
    const want = targets[m].E;
    const rel = Math.abs(got - want) / want;
    worstRel = Math.max(worstRel, rel);
    ok(rel < 0.20, `  mode ${m}: E=${got.toFixed(2)} meV vs analytic ${want.toFixed(2)} meV (axis ${'xyz'[targets[m].axis]}, rel err ${(rel * 100).toFixed(1)}%)`);

    // Eigenvector polarization: dominant component should be the target axis.
    const ev = eigenvectors[m];
    const weight = [0, 1, 2].map((c) => ev.real[c] ** 2 + ev.imag[c] ** 2);
    const dominant = weight[0] > weight[1] && weight[0] > weight[2] ? 0 : weight[1] > weight[2] ? 1 : 2;
    ok(dominant === targets[m].axis, `  mode ${m}: eigenvector dominant axis = ${'xyz'[dominant]} (expected ${'xyz'[targets[m].axis]}, weights=[${weight.map((w) => w.toFixed(3)).join(',')}])`);
  }
}
console.log(`\n  worst relative energy error across all points/branches: ${(worstRel * 100).toFixed(1)}%`);

if (fails) { console.error(`\n❌ synthetic dispersion benchmark: ${fails} check(s) failed`); process.exit(1); }
console.log('\n✅ synthetic dispersion benchmark OK — extracted E(q) and eigenvector polarizations match the known input model');
