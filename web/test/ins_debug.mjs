// Debug harness for the INS S(Q,E)/DOS math. Builds a realistic synthetic
// dispersive phonon model and prints the resulting heatmap as ASCII so we can
// eyeball whether the simulation produces sensible dispersive intensity.
//   node test/ins_debug.mjs
import { computePowderSqE, computePhononDOS } from '../src/io/sqeworker.js';
import { ATOMIC_MASS, B_COH } from '../src/constants.js';

// 2 basis sites (Ga, Se) -> 6 modes. Cubic conventional a=4 Å.
const a = 4.0;
const recip = [[1 / a, 0, 0], [0, 1 / a, 0], [0, 0, 1 / a]]; // inv(A).T, no 2π
const sites = [{ el: 'Ga' }, { el: 'Se' }];
const nSites = sites.length;
const nModes = 3 * nSites;

// q-path Γ(0,0,0) -> X(0.5,0,0) -> M(0.5,0.5,0)
const seg = [[[0, 0, 0], [0.5, 0, 0]], [[0.5, 0, 0], [0.5, 0.5, 0]]];
const perSeg = 40;
const qpts = [];
for (const [A, B] of seg) for (let j = 0; j < perSeg; j++) {
  const t = j / (perSeg - 1);
  qpts.push([A[0] + t * (B[0] - A[0]), A[1] + t * (B[1] - A[1]), A[2] + t * (B[2] - A[2])]);
}
const nq = qpts.length;

// Frequencies: 3 acoustic (sin dispersion 0..12) + 3 optical (~22 + small disp).
function freqsAt(q) {
  const s = Math.abs(Math.sin(Math.PI * q[0])) + Math.abs(Math.sin(Math.PI * q[1]));
  const ac = 12 * Math.min(1, s / 2);
  return [ac * 0.6, ac * 0.8, ac, 21 + 2 * s, 24 + 2 * s, 27 + s];
}

// Random normalized complex eigenvectors per mode (length 3*nSites).
let seed = 7;
const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff - 0.5;

const freqs = new Float64Array(nq * nModes);
const qpos = new Float64Array(nq * 3);
const F2 = new Float64Array(nq * nModes);
const b2 = sites.map(s => { const b = B_COH[s.el]; const m = ATOMIC_MASS[s.el]; return b * b / m; });

for (let qi = 0; qi < nq; qi++) {
  qpos[qi * 3] = qpts[qi][0]; qpos[qi * 3 + 1] = qpts[qi][1]; qpos[qi * 3 + 2] = qpts[qi][2];
  const fr = freqsAt(qpts[qi]);
  for (let m = 0; m < nModes; m++) {
    freqs[qi * nModes + m] = fr[m];
    const re = [], im = [];
    let nrm = 0;
    for (let i = 0; i < 3 * nSites; i++) { re[i] = rnd(); im[i] = rnd(); nrm += re[i] * re[i] + im[i] * im[i]; }
    nrm = 1 / Math.sqrt(nrm);
    let f2 = 0;
    for (let s = 0; s < nSites; s++) {
      let mag = 0;
      for (let c = 0; c < 3; c++) { const r = (s * 3 + c); mag += (re[r] * nrm) ** 2 + (im[r] * nrm) ** 2; }
      f2 += b2[s] * mag;
    }
    F2[qi * nModes + m] = f2;
  }
}

const data = { nq, nModes, freqs, qpos, F2, recip };
const params = { T: 5, sigma: 1.0, Emin: 0, Emax: 35, nE: 70, nQbins: 70, Ei: 0 };

const pow = computePowderSqE(data, params.T, params.sigma, params.Emin, params.Emax, params.nE, params.nQbins, params.Ei);
const dos = computePhononDOS(data, params.sigma, params.Emin, params.Emax, params.nE);

console.log(`Q range 0..${pow.xMax.toFixed(2)} Å⁻¹, E 0..${params.Emax} meV, Smax=${pow.Smax.toExponential(3)}`);
let nz = 0; for (const v of pow.S) if (v > 0) nz++;
console.log(`nonzero S bins: ${nz}/${pow.S.length} (${(100 * nz / pow.S.length).toFixed(0)}%)`);

// ASCII heatmap: E (rows, top=high) x Q (cols). Downsample to ~30x60.
const chars = ' .:-=+*#%@';
const RW = 28, CW = 60;
const { S, nX, nE, Smax } = pow;
const logK = 1 / Math.log1p(1000);
const HBAR2_2MN = 2.0723;
const EiTest = 40;                 // incident energy for the kinematic cutoff demo
const ki = Math.sqrt(EiTest / HBAR2_2MN);
const dQ = pow.xMax / nX;
const masked = (Q, E) => { if (E > EiTest) return true; const kf = Math.sqrt(Math.max(0, EiTest - E) / HBAR2_2MN); return Q < Math.abs(ki - kf) || Q > ki + kf; };

for (const [name, mask] of [['log, full', false], [`log, kinematic Eᵢ=${EiTest}`, true]]) {
  console.log(`\nS(|Q|,E)  [${name}]  (top=high E, left=low Q):`);
  for (let r = 0; r < RW; r++) {
    const ei = Math.round((1 - r / (RW - 1)) * (nE - 1));
    const E = params.Emin + ei / (nE - 1) * (params.Emax - params.Emin);
    let line = '';
    for (let c = 0; c < CW; c++) {
      const qi = Math.round(c / (CW - 1) * (nX - 1));
      const Q = (qi + 0.5) * dQ;
      if (mask && masked(Q, E)) { line += ' '; continue; }
      const v = Math.log1p(Math.max(0, S[qi * nE + ei] / Smax) * 1000) * logK;
      line += chars[Math.min(chars.length - 1, Math.floor(v * chars.length))];
    }
    console.log(`${E.toFixed(0).padStart(3)}|${line}`);
  }
}

// DOS profile
console.log('\nDOS (E up):  dosMax=' + dos.dosMax.toExponential(2));
for (let r = 0; r < RW; r++) {
  const ei = Math.round((1 - r / (RW - 1)) * (dos.nE - 1));
  const n = Math.round(dos.dos[ei] / dos.dosMax * 40);
  console.log('   |' + '#'.repeat(n));
}
