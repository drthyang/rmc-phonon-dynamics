// web/src/math/cells.js
//
// Computation-cell framework — Phase 0 (pure geometry; not yet wired into the
// pipeline or UI). See web/docs/cell-framework-plan.md.
//
// A "computation cell" is any sub-lattice of the RMC supercell, defined by a
// transform P from the conventional cell (P = I → conventional, P = diag(n) →
// custom supercell, P = primitive transform → primitive). Given the average
// (reference) atomic sites, relabelAtoms() assigns each atom to a (cell index n,
// basis site τ) of that cell — the labeling the displacement-covariance phase
// sum needs. Lattice vectors are rows: A = [a1, a2, a3], a position r = f·A.

/* ── 3×3 helpers (rows = lattice vectors) ────────────────────────────────── */
export function det3(m) {
  return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}
export function inv3(m) {
  const d = det3(m);
  if (Math.abs(d) < 1e-18) return null;
  const id = 1 / d;
  return [
    [(m[1][1] * m[2][2] - m[1][2] * m[2][1]) * id, (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * id, (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * id],
    [(m[1][2] * m[2][0] - m[1][0] * m[2][2]) * id, (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * id, (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * id],
    [(m[1][0] * m[2][1] - m[1][1] * m[2][0]) * id, (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * id, (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * id],
  ];
}
export function matMul3(A, B) {
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    C[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j];
  return C;
}
export function vecMat3(v, m) {
  return [
    v[0] * m[0][0] + v[1] * m[1][0] + v[2] * m[2][0],
    v[0] * m[0][1] + v[1] * m[1][1] + v[2] * m[2][1],
    v[0] * m[0][2] + v[1] * m[1][2] + v[2] * m[2][2],
  ];
}

/* ── cell construction ───────────────────────────────────────────────────── */
/** Computation-cell vectors (rows) L = P · A_conv. */
export function cellVectors(Aconv, P) { return matMul3(P, Aconv); }

/**
 * Does the computation cell L tile the supercell A_super exactly?
 * K = A_super · L⁻¹ must be an integer matrix (each supercell vector is an
 * integer combination of cell vectors). Returns { ok, K, nCells }.
 */
export function tilesSupercell(L, Asuper, tol = 1e-4) {
  const Linv = inv3(L);
  if (!Linv) return { ok: false, K: null, nCells: 0 };
  const K = matMul3(Asuper, Linv);
  let ok = true;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    if (Math.abs(K[i][j] - Math.round(K[i][j])) > tol) ok = false;
  return { ok, K, nCells: Math.round(Math.abs(det3(K))) };
}

/* ── re-labeling ─────────────────────────────────────────────────────────── */
const wrap01 = (x) => x - Math.floor(x);
const cyc = (a, b) => { const d = Math.abs(a - b) % 1; return Math.min(d, 1 - d); };

/**
 * Assign each (average) atom to a (cell index n, basis site τ) of cell L.
 *
 * @param {{pos:number[], element?:string, mass?:number}[]} atoms - cartesian
 *        AVERAGE positions (+ optional element/mass).
 * @param {number[][]} L - computation-cell vectors (rows).
 * @param {{tol?:number}} [opts] - fractional match tolerance (cyclic).
 * @returns {{ basis, assign, nBasis, nCells, issues, error? }}
 *   basis  : [{ frac:[3], element, mass, count }] sorted by fractional position
 *   assign : per atom { n:[i,j,k], tau }   (aligned to `atoms`)
 */
export function relabelAtoms(atoms, L, { tol = 0.05 } = {}) {
  const Linv = inv3(L);
  if (!Linv) return { error: 'singular cell', basis: [], assign: [], nBasis: 0, nCells: 0, issues: ['singular cell'] };

  // Fractional coords of every atom in the cell.
  const fr = atoms.map(a => vecMat3(a.pos, Linv));

  // Cluster the wrapped fractions into basis sites (cyclic, tolerance-based).
  // Match against each cluster's running representative; accumulate the circular
  // mean (sin/cos sums) so the final site fraction is centred on the true site
  // even when members wrap across 0/1 — that keeps the cell index n stable.
  const TAU = 2 * Math.PI;
  const basis = [];
  const findBasis = (wf) => {
    for (let i = 0; i < basis.length; i++) {
      const r = basis[i].rep;
      if (cyc(wf[0], r[0]) < tol && cyc(wf[1], r[1]) < tol && cyc(wf[2], r[2]) < tol) return i;
    }
    return -1;
  };
  const tau0 = new Array(atoms.length);
  for (let i = 0; i < atoms.length; i++) {
    const wf = [wrap01(fr[i][0]), wrap01(fr[i][1]), wrap01(fr[i][2])];
    let b = findBasis(wf);
    if (b < 0) { basis.push({ rep: wf, sc: [0, 0, 0], ss: [0, 0, 0], element: atoms[i].element, mass: atoms[i].mass, count: 0, _els: new Set() }); b = basis.length - 1; }
    const bb = basis[b];
    for (let d = 0; d < 3; d++) { bb.sc[d] += Math.cos(TAU * wf[d]); bb.ss[d] += Math.sin(TAU * wf[d]); }
    bb.count++;
    if (atoms[i].element != null) bb._els.add(atoms[i].element);
    tau0[i] = b;
  }
  // Circular mean → the site's representative fraction in [0,1).
  for (const b of basis) b.frac = [0, 1, 2].map(d => wrap01(Math.atan2(b.ss[d], b.sc[d]) / TAU));

  // Deterministic order: sort basis by fractional position; remap τ.
  const order = basis.map((_, i) => i).sort((a, b) => basis[a].frac[0] - basis[b].frac[0] || basis[a].frac[1] - basis[b].frac[1] || basis[a].frac[2] - basis[b].frac[2]);
  const remap = new Array(basis.length);
  order.forEach((oldIdx, newIdx) => { remap[oldIdx] = newIdx; });
  const sortedBasis = order.map(oldIdx => basis[oldIdx]);

  // Cell index n = round(f − siteFrac). A site sitting ON the cell boundary
  // (frac ≈ 0 ≡ 1) is canonically at 0, so snap those components to 0; that
  // keeps the cell origin shared across all basis sites (interior fractions
  // like 0.25/0.5/0.75 are kept as-is).
  const snap = Math.max(0.12, tol * 1.5);
  const onBoundary = (x) => (x < snap || x > 1 - snap) ? 0 : x;
  const assign = new Array(atoms.length);
  for (let i = 0; i < atoms.length; i++) {
    const tau = remap[tau0[i]], bf = sortedBasis[tau].frac, f = fr[i];
    assign[i] = { tau, n: [Math.round(f[0] - onBoundary(bf[0])), Math.round(f[1] - onBoundary(bf[1])), Math.round(f[2] - onBoundary(bf[2]))] };
  }

  // True cells-per-site = the MODAL basis count. A sheared sub-lattice (e.g. the
  // FCC primitive cell over a rectangular RMC supercell) gives each site a
  // different set of cell indices n, so a naive count of distinct n over-counts
  // (13 vs 4, 62 vs 32, …). The physics only needs n consistent WITHIN a site
  // (it is), and every site occurs once per cell, so the count they share is the
  // real cell total; sites off that mode are the genuine issues (partial
  // occupancy / a mislabel), not sheared-index bookkeeping.
  const freq = new Map();
  for (const b of sortedBasis) freq.set(b.count, (freq.get(b.count) || 0) + 1);
  let nCells = sortedBasis.length ? sortedBasis[0].count : 0, bestFreq = -1;
  for (const [c, f] of freq) if (f > bestFreq || (f === bestFreq && c > nCells)) { bestFreq = f; nCells = c; }

  // Validation: each basis appears once per cell; one element per basis site.
  const issues = [];
  for (const b of sortedBasis) {
    if (b.count !== nCells) issues.push(`basis @ [${b.frac.map(x => x.toFixed(3)).join(', ')}] appears ${b.count}× (cell count ${nCells})`);
    if (b._els.size > 1) issues.push(`basis @ [${b.frac.map(x => x.toFixed(3)).join(', ')}] mixes elements: ${[...b._els].join('/')}`);
  }

  return {
    basis: sortedBasis.map(b => ({ frac: b.frac, element: b.element, mass: b.mass, count: b.count })),
    assign, nBasis: sortedBasis.length, nCells, issues,
  };
}

/* ── pipeline glue ───────────────────────────────────────────────────────── */
export const IDENT = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

/**
 * From per-atom AVERAGE cartesian positions (+ element/mass) build the
 * (basis-site τ, cell-index n) labeling for computation cell L = P·A_conv, in
 * the flat-buffer form the S(k) kernel consumes.
 *
 * For P = I this reproduces the conventional-cell, per-reference-number grouping
 * the pipeline used before (τ ↔ a basis site, n ↔ the file cell index), so it is
 * a no-op for the default cell. For P ≠ I it regroups every atom onto the chosen
 * sub-/super-cell (primitive collapses equivalent sites; a custom supercell adds
 * sites). The basis is τ-ordered (sorted by fractional position).
 *
 * @param {number[][]} avgPos  - per-atom average cartesian positions [[x,y,z],…]
 * @param {string[]}   elements- per-atom element symbol (aligned to avgPos)
 * @param {number[]}   masses  - per-atom mass (aligned to avgPos)
 * @param {number[][]} Aconv   - conventional-cell vectors (rows)
 * @param {number[][]} [P]     - cell transform L = P·A_conv (default I)
 * @param {{tol?:number}} [opts]
 * @returns {{ L, nBasis, nCells, basis, tau, cellN, counts, tauElement,
 *             tauMass, tauFrac, issues }}
 *   tau      : Uint32Array(N)      basis-site index per atom
 *   cellN    : Float32Array(3N)    cell index n (L units) per atom, for the phase
 *   counts   : Float32Array(nBasis) atoms per basis site (kernel normalization)
 *   tauElement/tauMass/tauFrac : per-τ element / mass / site fraction
 */
export function buildCellLabeling(avgPos, elements, masses, Aconv, P = IDENT, opts = {}) {
  const L = cellVectors(Aconv, P);
  const atoms = avgPos.map((pos, i) => ({ pos, element: elements[i], mass: masses[i] }));
  const r = relabelAtoms(atoms, L, opts);
  if (r.error) return { error: r.error, L, nBasis: 0, nCells: 0, basis: [], tau: new Uint32Array(0), cellN: new Float32Array(0), counts: new Float32Array(0), tauElement: [], tauMass: [], tauFrac: [], issues: r.issues };

  const n = atoms.length;
  const tau = new Uint32Array(n);
  const cellN = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const a = r.assign[i];
    tau[i] = a.tau;
    cellN[i * 3] = a.n[0]; cellN[i * 3 + 1] = a.n[1]; cellN[i * 3 + 2] = a.n[2];
  }
  const counts = new Float32Array(r.nBasis);
  const tauElement = new Array(r.nBasis);
  const tauMass = new Array(r.nBasis);
  const tauFrac = new Array(r.nBasis);
  for (let t = 0; t < r.nBasis; t++) {
    counts[t] = r.basis[t].count;
    tauElement[t] = r.basis[t].element;
    tauMass[t] = r.basis[t].mass;
    tauFrac[t] = r.basis[t].frac;
  }
  return { L, nBasis: r.nBasis, nCells: r.nCells, basis: r.basis, tau, cellN, counts, tauElement, tauMass, tauFrac, issues: r.issues };
}
