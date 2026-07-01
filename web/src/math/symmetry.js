// web/src/math/symmetry.js
//
// Pure, fully-offline symmetry-operation finder for the (average) structure — the
// "predictable way to trace symmetry" the cell framework needs, without any
// external dependency or WASM. Given the conventional cell A_conv + basis + a
// tolerance it returns the space-group operations {R|t} that map the basis onto
// itself, plus the residual of that fit, so symmetry can be traced as a function
// of tolerance on a disordered RMC average.
//
// Method (spglib-lite, but table-free and bounded):
//   1. Point operations = integer matrices R (entries in {-1,0,1}) that preserve
//      the lattice metric G = A·Aᵀ, i.e. RᵀGR = G. These are the lattice
//      automorphisms in the CONVENTIONAL direct basis; |det R| = 1.
//   2. For each R, the space-group translations t are found from atom images:
//      t = frac(b) − R·frac(a0) for candidate partners b (same element), kept when
//      {R|t} maps every atom onto a same-element atom within `tol` (cartesian Å).
//
// Fractional coords are COLUMN vectors here: x' = R·x + t. Lattice rows: A = [a1,a2,a3].
// NOTE: entries in {-1,0,1} cover the standard conventional settings (cubic,
// tetragonal, orthorhombic, hexagonal, rhombohedral-in-hex, monoclinic, triclinic).
// Stage 1 is REPORT-ONLY; it does not change the S(k) folding (that is Stage 2).

import { det3 } from './cells.js';

const wrap01 = (x) => x - Math.floor(x);
const nearestInt = (x) => x - Math.round(x);

/** Metric tensor G = A·Aᵀ (A rows = lattice vectors, cartesian). */
export function metricTensor(A) {
  const G = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    G[i][j] = A[i][0] * A[j][0] + A[i][1] * A[j][1] + A[i][2] * A[j][2];
  return G;
}

// Rᵀ · G · R for a 3×3 integer R and symmetric G.
function conjugate(R, G) {
  // M = Rᵀ G R.  (Rᵀ)_{ij} = R_{ji}
  const GR = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    GR[i][j] = G[i][0] * R[0][j] + G[i][1] * R[1][j] + G[i][2] * R[2][j];
  const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    M[i][j] = R[0][i] * GR[0][j] + R[1][i] * GR[1][j] + R[2][i] * GR[2][j];
  return M;
}

/**
 * Lattice point operations: integer R (entries in {-1,0,1}, |det|=1) with RᵀGR=G.
 * `tol` is a RELATIVE metric tolerance (fraction of the mean squared edge).
 */
export function latticePointOps(A, tol = 1e-3) {
  const G = metricTensor(A);
  const scale = (G[0][0] + G[1][1] + G[2][2]) / 3 || 1;
  const eps = tol * scale;
  const ops = [];
  const v = [-1, 0, 1];
  const R = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  // Iterate all 3^9 sign/zero patterns; keep metric-preserving unimodular ones.
  for (let code = 0; code < 19683; code++) {
    let c = code;
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) { R[a][b] = v[c % 3]; c = (c / 3) | 0; }
    const d = det3(R);
    if (d !== 1 && d !== -1) continue;
    const M = conjugate(R, G);
    let ok = true;
    for (let i = 0; i < 3 && ok; i++) for (let j = 0; j < 3 && ok; j++)
      if (Math.abs(M[i][j] - G[i][j]) > eps) ok = false;
    if (ok) ops.push(R.map(r => r.slice()));
  }
  return ops;
}

// Apply R (integer) to a fractional column vector.
function applyR(R, x) {
  return [
    R[0][0] * x[0] + R[0][1] * x[1] + R[0][2] * x[2],
    R[1][0] * x[0] + R[1][1] * x[1] + R[1][2] * x[2],
    R[2][0] * x[0] + R[2][1] * x[1] + R[2][2] * x[2],
  ];
}

// Cartesian distance between two fractional points (minimal image), rows A.
function cartDist(fa, fb, A) {
  const d = [nearestInt(fa[0] - fb[0]), nearestInt(fa[1] - fb[1]), nearestInt(fa[2] - fb[2])];
  const c = [
    d[0] * A[0][0] + d[1] * A[1][0] + d[2] * A[2][0],
    d[0] * A[0][1] + d[1] * A[1][1] + d[2] * A[2][1],
    d[0] * A[0][2] + d[1] * A[1][2] + d[2] * A[2][2],
  ];
  return Math.hypot(c[0], c[1], c[2]);
}

// For op {R|t}, the worst cartesian mapping error over all atoms (∞ if some atom
// has no same-element image within `tol`).
function mappingResidual(R, t, basis, byEl, A, tol) {
  let worst = 0;
  for (const s of basis) {
    const img = applyR(R, s.frac);
    img[0] = wrap01(img[0] + t[0]); img[1] = wrap01(img[1] + t[1]); img[2] = wrap01(img[2] + t[2]);
    let best = Infinity;
    for (const o of byEl.get(s.el)) { const dd = cartDist(img, o.frac, A); if (dd < best) best = dd; }
    if (best > tol) return Infinity;
    if (best > worst) worst = best;
  }
  return worst;
}

/**
 * Space-group operations of (A, basis) within a cartesian tolerance `tol` (Å).
 * basis: [{ el, frac:[x,y,z] }]. Returns operations + summary counts + residual.
 *
 * @returns {{ ops:{R,t,residual}[], nSpace, nPoint, order, maxResidual }}
 *   nPoint : distinct rotation parts present in the space group (its point group).
 *   nSpace : total {R|t} (= point-group order × #centering-type cosets for the cell).
 */
export function findSpaceGroupOps(A, basis, tol = 0.1, metricTol = 1e-2) {
  if (!basis || basis.length === 0) return { ops: [], nSpace: 0, nPoint: 0, order: 0, maxResidual: 0 };
  const pointOps = latticePointOps(A, metricTol);
  const byEl = new Map();
  for (const s of basis) { if (!byEl.has(s.el)) byEl.set(s.el, []); byEl.get(s.el).push(s); }

  const ops = [];
  const rotSeen = new Set();
  let maxResidual = 0;
  // Use the rarest element for candidate translations (fewest partners → fastest).
  let refEl = basis[0].el;
  for (const [el, arr] of byEl) if (arr.length < byEl.get(refEl).length) refEl = el;
  const refAtom = byEl.get(refEl)[0];

  for (const R of pointOps) {
    const Ra0 = applyR(R, refAtom.frac);
    const tSeen = [];
    for (const cand of byEl.get(refEl)) {
      const t = [wrap01(cand.frac[0] - Ra0[0]), wrap01(cand.frac[1] - Ra0[1]), wrap01(cand.frac[2] - Ra0[2])];
      if (tSeen.some(u => cartDist(u, t, A) < tol)) continue;   // dedupe translations mod lattice
      const res = mappingResidual(R, t, basis, byEl, A, tol);
      if (res === Infinity) continue;
      tSeen.push(t);
      ops.push({ R, t, residual: res });
      if (res > maxResidual) maxResidual = res;
    }
    if (tSeen.length) rotSeen.add(R.flat().join(','));
  }
  return { ops, nSpace: ops.length, nPoint: rotSeen.size, order: ops.length, maxResidual };
}
