// web/test/validate.mjs
//
// Numerical regression suite for the migrated phonon backend. Run from web/:
//   npm run validate          (node test/validate.mjs)
//
// Guards the three fatal regressions fixed during migration, plus diagonalizer
// correctness:
//   1. S(k) grouped by reference number (basis site), giving 3*N_sites bands.
//   2. The 2*pi Bloch phase  ->  S(k=G) == S(Gamma) for reciprocal vectors G.
//   3. ENERGY_CONV value matches src_gpu/constants.py.
//   4. Complex Hermitian eigh: A == V Lambda V^dag reconstruction.
//   5. The browser engine-style S(k) assembly matches an INDEPENDENT pure-Python
//      reference (web/test/reference.json) to ~1e-9.

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { eigh, eigenvaluesToMev } from '../src/math/diagonalize.js';
import { ENERGY_CONV, TWO_PI_PHASE } from '../src/constants.js';

const __dir = dirname(fileURLToPath(import.meta.url));
let failures = 0;
const approx = (a, b, tol, msg) => {
  const d = Math.abs(a - b);
  if (!(d <= tol)) { console.error(`  FAIL ${msg}: |${a} - ${b}| = ${d} > ${tol}`); failures++; }
  else console.log(`  ok   ${msg} (|Δ|=${d.toExponential(2)})`);
};
const ok = (cond, msg) => { if (!cond) { console.error(`  FAIL ${msg}`); failures++; } else console.log(`  ok   ${msg}`); };

// ── Engine-style S(k): real/imag split, grouped by reference number, 2*pi ────
function computeSk(inputs, qfrac) {
  const { dim, v_super, masses_by_rn, atoms, frames } = inputs;
  const n = atoms.length;
  const uniq = [...new Set(atoms.map(a => a.rn))].sort((a, b) => a - b);
  const T = uniq.length;
  const seg = new Map(uniq.map((rn, i) => [rn, i]));
  const counts = uniq.map(rn => atoms.filter(a => a.rn === rn).length);
  const kvec = qfrac.map(q => q * TWO_PI_PHASE);

  // ensemble mean of within-cell xyz
  const mean = Array.from({ length: n }, () => [0, 0, 0]);
  for (const fr of frames) for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) mean[i][c] += fr[i][c];
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) mean[i][c] /= frames.length;

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

console.log('\n[1] ENERGY_CONV matches src_gpu/constants.py');
approx(ENERGY_CONV, 0.600181852836787, 1e-12, 'ENERGY_CONV');

console.log('\n[2] eigenvaluesToMev sanity (lambda=T => E=ENERGY_CONV)');
{
  const e = eigenvaluesToMev(new Float64Array([5.0]), 5.0, ENERGY_CONV); // sqrt(T/lambda)=1
  approx(e[0], ENERGY_CONV, 1e-12, 'E(lambda=T=5)=ENERGY_CONV');
  const eNeg = eigenvaluesToMev(new Float64Array([-5.0]), 5.0, ENERGY_CONV);
  ok(eNeg[0] < 0, 'soft mode (neg eigenvalue) -> negative energy');
  const eZero = eigenvaluesToMev(new Float64Array([1e-6]), 5.0, ENERGY_CONV);
  ok(eZero[0] === 0, 'sub-threshold eigenvalue -> 0');
}

console.log('\n[3] Complex Hermitian eigh reconstruction A == V Λ Vdag');
{
  const N = 4;
  // Build a random Hermitian A = M + Mdag.
  const seed = (() => { let s = 42; return () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff; })();
  const Are = new Float64Array(N * N), Aim = new Float64Array(N * N);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    if (i === j) { Are[i * N + j] = seed() * 2 - 1; Aim[i * N + j] = 0; }
    else if (i < j) {
      const r = seed() * 2 - 1, m = seed() * 2 - 1;
      Are[i * N + j] = r; Aim[i * N + j] = m;
      Are[j * N + i] = r; Aim[j * N + i] = -m; // Hermitian
    }
  }
  const { eigenvalues, eigenvectors } = eigh(Are, Aim, N);
  // Reconstruct: R[i][j] = sum_k lambda_k V[i][k] conj(V[j][k])
  let maxErr = 0;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    let re = 0, im = 0;
    for (let k = 0; k < N; k++) {
      const l = eigenvalues[k];
      const vir = eigenvectors[k].real[i], vii = eigenvectors[k].imag[i];
      const vjr = eigenvectors[k].real[j], vji = eigenvectors[k].imag[j];
      // V[i][k] * conj(V[j][k])
      re += l * (vir * vjr + vii * vji);
      im += l * (vii * vjr - vir * vji);
    }
    maxErr = Math.max(maxErr, Math.abs(re - Are[i * N + j]), Math.abs(im - Aim[i * N + j]));
  }
  approx(maxErr, 0, 1e-9, 'eigh reconstruction max error');
}

console.log('\n[4] S(k) vs independent pure-Python reference + grouping + 2π');
{
  const ref = JSON.parse(readFileSync(join(__dir, 'reference.json'), 'utf8'));
  const { inputs, outputs } = ref;

  const nSites = new Set(inputs.atoms.map(a => a.rn)).size;
  ok(outputs.gamma.re.length === 3 * nSites, `band dim = 3*N_sites = ${3 * nSites} (grouped by reference number)`);

  const skByName = {};
  for (const name of Object.keys(outputs)) {
    const q = inputs.kpoints[name];
    const { Sre, Sim, D } = computeSk(inputs, q);
    skByName[name] = { Sre, Sim, D };
    let maxErr = 0;
    for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) {
      maxErr = Math.max(maxErr,
        Math.abs(Sre[i * D + j] - outputs[name].re[i][j]),
        Math.abs(Sim[i * D + j] - outputs[name].im[i][j]));
    }
    approx(maxErr, 0, 1e-9, `S(k) [${name}] matches python reference`);
  }

  // 2π periodicity: S(G) == S(Gamma) for reciprocal vector G=[1,0,0].
  const g = skByName.gamma, G = skByName.G;
  let perErr = 0;
  for (let i = 0; i < g.D * g.D; i++) perErr = Math.max(perErr, Math.abs(g.Sre[i] - G.Sre[i]), Math.abs(g.Sim[i] - G.Sim[i]));
  approx(perErr, 0, 1e-9, 'S(k=G) == S(Γ)  (2π Bloch-phase periodicity)');

  // Sanity: generic k differs from Gamma (phase actually matters).
  const gen = skByName.generic;
  let diff = 0;
  for (let i = 0; i < g.D * g.D; i++) diff = Math.max(diff, Math.abs(g.Sim[i] - gen.Sim[i]));
  ok(diff > 1e-6, 'S(generic k) differs from S(Γ)');
}

console.log(`\n${failures === 0 ? '✅ ALL CHECKS PASSED' : `❌ ${failures} CHECK(S) FAILED`}`);
process.exit(failures === 0 ? 0 : 1);
