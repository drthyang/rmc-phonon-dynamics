// web/src/math/symmetrize.js
//
// Symmetrize the reciprocal-space displacement covariance S(k) by projecting it
// onto the subspace invariant under the little group of k:
//
//     S_sym(k) = (1/|G_k|) Σ_{g ∈ G_k} Γ(g) · S(k) · Γ(g)†
//
// This does two physical things at once: it POOLS the statistics of
// symmetry-equivalent basis sites (the site-permutation part of Γ), and it
// ENFORCES the symmetry-required degeneracies (S_sym commutes with the little-group
// representation). With the trivial group it is the identity, so it is opt-in and
// gated.
//
// Γ(g) for g = {R|t} is a generalized permutation with 3×3 rotation blocks:
//   (Γ)_{(τ',α),(σ,α')} = δ_{τ', perm_g(σ)} · e^{2πi k·Δ_g(σ)} · Rcart_{αα'}
// where perm_g(σ)=τ' is the basis site that σ maps to, Δ_g(σ) the integer cell
// shift, and Rcart the cartesian rotation of the fractional rotation R on cell L.
// Convention (verified by the "symmetric input is invariant" test): the little
// group is { g : Rᵀk ≡ k (mod reciprocal lattice) }, phase e^{2πi k·Δ}.

const wrapHalf = (x) => x - Math.round(x);

// Cartesian rotation R_cart = Lᵀ · R · L⁻ᵀ for a fractional rotation R on cell L
// (rows = lattice vectors): a fractional point x has cartesian r = Lᵀx, so
// r' = Lᵀ R x = (Lᵀ R L⁻ᵀ) r.
function cartRotation(R, L, Linv) {
  // Lᵀ R  (Lᵀ)_{ij} = L_{ji}
  const LtR = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    LtR[i][j] = L[0][i] * R[0][j] + L[1][i] * R[1][j] + L[2][i] * R[2][j];
  // (LtR) · L⁻ᵀ , (L⁻ᵀ)_{ij} = Linv_{ji}
  const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    M[i][j] = LtR[i][0] * Linv[j][0] + LtR[i][1] * Linv[j][1] + LtR[i][2] * Linv[j][2];
  return M;
}

function inv3(m) {
  const d = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const id = 1 / d;
  return [
    [(m[1][1] * m[2][2] - m[1][2] * m[2][1]) * id, (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * id, (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * id],
    [(m[1][2] * m[2][0] - m[1][0] * m[2][2]) * id, (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * id, (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * id],
    [(m[1][0] * m[2][1] - m[1][1] * m[2][0]) * id, (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * id, (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * id],
  ];
}

/**
 * For each space-group op {R,t} that maps `basis` onto itself, precompute its
 * representation on the basis: the site permutation, the integer cell shift per
 * site, and the cartesian rotation. Ops that don't map the basis onto itself
 * (shouldn't happen for detected ops) are dropped.
 *
 * @param {number[][]} L      computation-cell lattice rows
 * @param {{frac:number[]}[]} basis basis sites (fractional, in L)
 * @param {{R:number[][], t:number[]}[]} ops detected space-group operations
 * @returns {{R, t, perm:number[], shift:number[][], Rcart:number[][]}[]}
 */
export function operationReps(L, basis, ops, tolFrac = 0.02) {
  const Linv = inv3(L);
  const n = basis.length;
  const reps = [];
  for (const { R, t } of ops) {
    const perm = new Array(n), shift = new Array(n);
    let ok = true;
    for (let s = 0; s < n && ok; s++) {
      const f = basis[s].frac;
      const img = [
        R[0][0] * f[0] + R[0][1] * f[1] + R[0][2] * f[2] + t[0],
        R[1][0] * f[0] + R[1][1] * f[1] + R[1][2] * f[2] + t[1],
        R[2][0] * f[0] + R[2][1] * f[1] + R[2][2] * f[2] + t[2],
      ];
      let found = -1, dr = null;
      for (let s2 = 0; s2 < n; s2++) {
        const d = [img[0] - basis[s2].frac[0], img[1] - basis[s2].frac[1], img[2] - basis[s2].frac[2]];
        if (Math.abs(wrapHalf(d[0])) < tolFrac && Math.abs(wrapHalf(d[1])) < tolFrac && Math.abs(wrapHalf(d[2])) < tolFrac) {
          found = s2; dr = [Math.round(d[0]), Math.round(d[1]), Math.round(d[2])]; break;
        }
      }
      if (found < 0) { ok = false; break; }
      perm[s] = found; shift[s] = dr;
    }
    if (ok) reps.push({ R, t, perm, shift, Rcart: cartRotation(R, L, Linv) });
  }
  return reps;
}

// Little group of k (fractional reciprocal): ops with Rᵀk ≡ k (mod 1).
function inLittleGroup(R, k, tol = 1e-3) {
  const rk = [
    R[0][0] * k[0] + R[1][0] * k[1] + R[2][0] * k[2],   // (Rᵀk)_0
    R[0][1] * k[0] + R[1][1] * k[1] + R[2][1] * k[2],
    R[0][2] * k[0] + R[1][2] * k[1] + R[2][2] * k[2],
  ];
  return Math.abs(wrapHalf(rk[0] - k[0])) < tol && Math.abs(wrapHalf(rk[1] - k[1])) < tol && Math.abs(wrapHalf(rk[2] - k[2])) < tol;
}

/**
 * S_sym(k) = (1/|G_k|) Σ Γ(g) S Γ(g)†, in place-free form. S is 3n×3n complex,
 * stored as row-major Sre/Sim (length (3n)²). Returns { re, im } symmetrized.
 * `reps` from operationReps; `k` fractional reciprocal coords of the point.
 */
export function symmetrizeSk(Sre, Sim, nBasis, k, reps) {
  const D = 3 * nBasis;
  const little = reps.filter(g => inLittleGroup(g.R, k));
  if (little.length <= 1) return { re: Sre, im: Sim };   // trivial → unchanged
  const TWO_PI = 2 * Math.PI;
  const are = new Float64Array(D * D), aim = new Float64Array(D * D);
  for (const g of little) {
    const { perm, shift, Rcart } = g;
    for (let s = 0; s < nBasis; s++) for (let s2 = 0; s2 < nBasis; s2++) {
      // phase e^{2πi k·(Δ_s − Δ_{s2})}
      const ph = TWO_PI * (k[0] * (shift[s][0] - shift[s2][0]) + k[1] * (shift[s][1] - shift[s2][1]) + k[2] * (shift[s][2] - shift[s2][2]));
      const cph = Math.cos(ph), sph = Math.sin(ph);
      // block B = Rcart · S[s,s2] · Rcartᵀ (3×3 complex), then × phase, add at (perm[s],perm[s2]).
      const ps = perm[s] * 3, p2 = perm[s2] * 3, rs = s * 3, r2 = s2 * 3;
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
        // (Rcart S Rcartᵀ)_{ab} = Σ_{i,j} Rcart[a][i] S[i,j] Rcart[b][j]
        let bre = 0, bim = 0;
        for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
          const w = Rcart[a][i] * Rcart[b][j];
          const idx = (rs + i) * D + (r2 + j);
          bre += w * Sre[idx]; bim += w * Sim[idx];
        }
        // × phase (cph + i sph)
        const tre = bre * cph - bim * sph, tim = bre * sph + bim * cph;
        const o = (ps + a) * D + (p2 + b);
        are[o] += tre; aim[o] += tim;
      }
    }
  }
  const inv = 1 / little.length;
  for (let i = 0; i < D * D; i++) { are[i] *= inv; aim[i] *= inv; }
  return { re: are, im: aim, nLittle: little.length };
}
