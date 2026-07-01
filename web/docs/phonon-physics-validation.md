# Phonon-extraction physics validation

Date: 2026-07-01. Scope: audit the S(k) → eigenvalue/eigenvector → phonon-energy
pipeline (`web/src/compute/`, `web/src/math/diagonalize.js`,
`web/src/math/band_connection.js`) for correctness of units, the 2π Bloch-phase
convention, and the S(k)/eigenvector/eigenvalue math, against the literature
method it implements. Two reference papers were read in full:

- Goodwin, Tucker, Dove & Keen, *"Phonons from Powder Diffraction: A
  Quantitative Model-Independent Evaluation"*, PRL **93**, 075502 (2004) — the
  method this codebase implements (direct phonon extraction from RMC
  atomistic-configuration displacement covariance).
- Dimitrov, Louca & Röder, *"Phonons from neutron powder diffraction"*, PRB
  **60**, 6204 (1999) — a **different, complementary** method (forward
  shell/force-constant model → theoretical S(Q) → PDF, RMC-fit the model
  parameters). Useful background (Debye-Waller / mass-weighting conventions)
  but not the method to trace equations against; this repo does not do a
  forward S(Q) calculation.

## 1. Equation-by-equation trace (Goodwin et al., Eqs. 3–7)

| Paper (Eq.) | Formula | Code | Verdict |
|---|---|---|---|
| Eq. 3 | `U(k,t) = (1/√N) Σⱼ √mⱼ uⱼ(t) exp(ik·Rⱼ)`, per basis site | [`Sk_kernel.wgsl:64-74`](../src/compute/Sk_kernel.wgsl), normalized by `1/√counts[t]` in [`engine.js:142-159`](../src/compute/engine.js) | exact match |
| Eq. 5 | `S(k) = ⟨T(k)·Tᵀ(−k)⟩` | Since `T(−k)=conj(T(k))` for real `u`, this is `T(k)T(k)†`. Hand-verified [`engine.js:162-168`](../src/compute/engine.js): `Sk_real=AᵢAⱼ+BᵢBⱼ`, `Sk_imag=BᵢAⱼ−AᵢBⱼ` is exactly `Tᵢ·conj(Tⱼ)`; confirmed Hermitian (real part symmetric, imag part antisymmetric) | exact match |
| Eq. 6–7 | eigendecompose `S`, `ω²ᵢ = kBT/⟨Q·Qᵀ⟩ᵢᵢ` | [`diagonalize.js`](../src/math/diagonalize.js) embeds Hermitian `S=A+iB` into a real symmetric `2N×2N` matrix `[[A,−B],[B,A]]` (standard trick; verified `Mᵀ=M` algebraically given `A=Aᵀ`, `B=−Bᵀ`); [`eigenvaluesToMev`](../src/math/diagonalize.js) computes `E=ħω=ħ√(kBT/λ)` — exactly Eq. 7 with `E=ħω` | exact match |

## 2. Units / 2π convention

This is the highest-risk spot for a bug (mixing fractional-reciprocal and
Cartesian conventions), so it was checked three ways:

1. **Algebraic derivation.** The code phases by `k·Rₙ = 2π(q_frac·n)` —
   fractional reciprocal coordinate dotted with the *integer* cell-translation
   index, times 2π (`TWO_PI_PHASE` in [`constants.js`](../src/constants.js),
   applied in [`pipeline.js:236`](../src/compute/pipeline.js)). Derived from
   `bᵢ·aⱼ=2πδᵢⱼ`, this holds **regardless of lattice metric** — mathematically
   identical to Goodwin's fully-Cartesian phase (`k` in Å⁻¹ dotted with `R` in
   Å, no explicit 2π), just a different coordinate choice.
2. **Cell-transform consistency (Phase 1–3 work).** The computation-cell
   framework requires transforming the k-path under `P`; verified
   `q_cell = P·q_conv` ([`pipeline.js:227-231`](../src/compute/pipeline.js))
   is the *exact* dual-lattice consequence of `L = P·A_conv`, derived from the
   reciprocal-lattice relation (not just empirically fit).
3. **Cross-validated three independent ways**: the JS WebGPU kernel, the
   legacy JAX Python (`archive/src_gpu/Calculators.py`, which documents its
   own empirical validation: *"S(k=G)=S(Γ) to ~1e-4 WITH the 2π factor vs.
   residual 44–388 WITHOUT it"*), and a from-scratch pure-Python reference
   with no shared code (`web/test/reference_sk.py`) that
   `web/test/validate.mjs` checks the JS against numerically.

## 3. Eigenvectors / mode tracking

[`band_connection.js`](../src/math/band_connection.js) implements Hungarian
assignment on eigenvector overlap plus SVD-rotation within near-degenerate
subspaces for continuity — this is the "each traces data with similar
eigenvectors" method Goodwin's Fig. 1/2 captions describe for LO/TO
assignment. It only reorders/rotates, never touches frequencies.

## 4. Known non-bug: small-eigenvalue clipping

[`diagonalize.js`](../src/math/diagonalize.js) clips any `|eigenvalue| <
1e-4` to `E=0` rather than computing `E=ħ√(kBT/λ)`. Since `ω²∝1/λ`, a tiny λ
would otherwise blow up to a huge spurious energy from float noise — this is
a defensive numerical clip (matches the legacy Python's identical threshold),
not a physics shortcut.

## 5. New: synthetic-dispersion end-to-end benchmark

The equation trace above proves each formula in isolation, but nothing
exercised the pipeline at **generic (off-symmetry) k across a real
multi-branch dispersion** — every existing test used Γ, integer folds, or
single-atom uncorrelated noise. Added
[`web/test/synthetic_dispersion_test.mjs`](../test/synthetic_dispersion_test.mjs),
now part of `npm run validate`, using the same validation strategy Goodwin et
al. used for MgO (their Fig. 2a: generate MD configs from a *known*
force-constant model, check the extraction reproduces it) — except the
target is exact here, so it's asserted with a numeric tolerance instead of
compared by eye.

**Model.** 10×10×10 simple-cubic lattice, 1 atom/cell, a dynamical matrix
diagonal and fully decoupled in the Cartesian directions:
`E_α(q) = 30·|sin(πq_α)|` meV independently per axis α = x,y,z (textbook
nearest-neighbor monatomic-chain dispersion, applied per direction, no
cross-coupling).

**Synthetic ensemble.** From the general classical harmonic relation
`S(k) = kBT·D(k)⁻¹`, this model's real-space displacement covariance
factorizes into independent 1D chains along each axis (derived by hand — the
ky,kz sums collapse to Kronecker deltas since D is independent of them).
Built an exact real sine/cosine mode-synthesis generator per chain (paired
`j`/`L−j` coefficient `√(2/L)`, self-paired Nyquist coefficient `√(1/L)`,
algebraically verified to reproduce the target covariance in expectation) —
2500 synthetic thermal "frames," each an independent random draw.

**Extraction.** Fed straight through the production math — `buildCellLabeling`
([`cells.js`](../src/math/cells.js)), the same `T(k)T(k)†` covariance
assembly `Sk_kernel.wgsl`/`engine.js` use, `eigh`, `eigenvaluesToMev` — at 5
generic q-points where all 3 branches are simultaneously present and
non-degenerate (not along a symmetry line, where 2 of 3 would sit at Γ).

**Result.** All 30 checks (5 q-points × 3 branches × {energy, eigenvector})
pass; worst relative energy error **2.2%** (tolerance 20%, generous for
Monte Carlo noise at N=2500 frames — a real bug would show >>20% or
misassigned eigenvectors, not a few-percent statistical wobble). Eigenvectors
identify the correct polarization axis to >99% purity at every point.

One bug surfaced while building this — in the **test**, not the production
code: eigenvalues from `eigh` are sorted ascending, and since `E ∝ 1/√λ`,
ascending λ is *descending* E. The first run showed mode 1 (middle energy)
matching exactly while modes 0/2 were precisely swapped with correct
eigenvectors — an unambiguous signature of a comparison-order bug rather
than a numerical one. Fixed by sorting the analytic targets descending to
match `eigh`'s output order.

## Bottom line

No unit, 2π, S(k), eigenvector, or eigenvalue errors found. Every formula in
the extraction pipeline was traced to its equation in Goodwin et al. (2004),
the 2π convention was verified algebraically (not just empirically), and the
new synthetic-dispersion benchmark closes the remaining gap: proof that the
whole pipeline correctly resolves a genuine multi-branch dispersion shape at
arbitrary off-symmetry k, not just at Γ or trivial folds.
