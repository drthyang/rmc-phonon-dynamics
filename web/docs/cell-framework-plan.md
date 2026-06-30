# Computation-cell framework — implementation plan

Status: **Phase 0 complete**, **Phase 1 mostly landed** — relabel-driven
per-basis-site S(k) wired into `compute/pipeline.js` (default `P = I` = zero
regression) **and** the conventional default k-path (the Γ→X fix), verified
end-to-end in-browser. Tests in `npm run validate` (`cells_pipeline_test.mjs`,
`highsym_test.mjs`). What's left in Phase 1: the reference-mode knob and exposing
`P` in the UI — see the Phase 1 bullet below. This is the reference plan; build
piece by piece and check work against it.

## Problem

S(k) is computed by summing the displacement-covariance phase `exp(2πi k·n)`
over repeat cells `n` of a reference lattice. Today the RMC supercell is treated
as a tiling of the **conventional** cell and grouped by **element type**, while
the k-path comes from the **primitive** BZ. For centered lattices (FCC/BCC/…)
the primitive high-symmetry points fold onto conventional reciprocal-lattice
vectors (e.g. FCC X = conventional `(0,0,1)` ≡ Γ), so Γ→X comes out as a full
period — a spurious mirror-symmetric band. The `2π` is correct; the bug is the
**mismatch between the cell the covariance is summed over and the cell the
k-path is labeled in**.

## Core idea

A **computation cell** is any sub-lattice of the supercell, defined by a
transform `P` from the conventional cell:

- `P = I` → conventional cell (done correctly = self-consistent, folded);
- `P =` primitive transform → primitive cell (true, unfolded dispersion);
- `P = diag(n₁,n₂,n₃)` (or any integer matrix) → a custom supercell.

So **primitive and custom-supercell are the same feature** with different `P`.
Custom supercells need no symmetry detection; the primitive cell needs symmetry
only to *find* `P`. Smaller cell ⇒ more displacement samples per basis site but
more symmetry assumed; larger cell ⇒ fewer samples per site, fewer assumptions
(closer to P1). The cell choice is a **symmetry-vs-statistics knob**.

## Model

```
ReferenceStructure : conventional lattice (a1,a2,a3 + dim) + average frac
                     positions + element/mass per site (from the ensemble mean).
ComputationCell    : { P (3×3), L = P·A_conv, basis:[{frac,element,mass}],
                       cellsPerSupercell }.
relabel(ref, cell) → per atom { n:[i,j,k] in L units, τ: basisIndex }   (geometry,
                     tolerance-based, against AVERAGE positions).
S(k)               = ⟨W W†⟩,  Wτα(k)=Σ_{atoms∈τ} √m·u·e^{2πi k·n},  dim 3·N_basis.
k-path / BZ        = reciprocal & high-sym of L (consistent with the phase).
```

The one substantive science change: S(k) is indexed by **basis site τ**
(dimension `3·N_basis`), not by element type. That ripples into `eigh` (any N is
fine), band connection, and the viewer model (`baseStructure` must carry the
basis; eigvecs length `3·N_basis`).

## Reference construction is cell-aware

The displacement reference is **not** independent of the cell. RMC is solved in a
**P1 supercell**, so the assumption-free reference is the **per-site ensemble
mean** — each physical site is its own average, nothing symmetrized (this is
what the app does today, and it's correct for P1). Choosing a *smaller* cell
folds symmetry-equivalent sites onto one basis site τ, which introduces an
**additional averaging** — and that is exactly where the symmetry is imposed.
It is twofold:

1. **Reference positions** — `bf_τ` = mean of the equivalent sites' average
   positions → a *symmetrized* equilibrium (`relabelAtoms` already returns this).
2. **Statistics** — pooling those sites' displacement covariances into one per-τ
   block (more samples per site, on the assumption they're equivalent).

So **picking the cell sets how much the equilibrium is symmetrized.** Expose a
**reference mode**:

| Mode | u = | Meaning |
|---|---|---|
| symmetrized-site (default, cell-consistent) | `r − (R_n + bf_τ)` | equilibrium has the chosen cell's symmetry; `u` includes each atom's *static* offset from the symmetrized site. More samples per site. |
| per-atom | `r − r̄_atom` | each atom about its own ensemble mean; pure dynamic fluctuation; the cell only regroups (P1 extreme = the whole supercell). |

They converge for a truly symmetric, well-sampled crystal and diverge when the
RMC average is genuinely distorted (the "symmetry lower than FCC" case).
**Phase 0 only computes the labels + `bf_τ` (no displacement change); applying
the reference and pooling the statistics is Phase 1.**

## Phases

- **Phase 0 — data model + re-labeling (pure, no behavior change). ✅ DONE.**
  `math/cells.js` (`det3/inv3/matMul3/vecMat3`, `cellVectors`, `tilesSupercell`,
  `relabelAtoms`) + Node tests (`test/cells_test.mjs`, in `npm run validate`).
  Every atom → exactly one (n,τ); basis consistent across cells; circular-mean
  `bf_τ` with boundary-snap so cell origins stay shared.
- **Phase 1 — generalize S(k) to per-basis-site over arbitrary P. ◑ IN PROGRESS.**
  Reuse the WGSL phase kernel; group by basis site instead of element; feed `n`
  in L units. Build the **cell-aware reference** (per-basis-site mean; expose the
  symmetrized-site vs per-atom **reference mode** above) and compute `u` from it.
  Default `P = I` with the **conventional BZ path** (X at ½) → fixes the Γ→X
  symmetry self-consistently. *This is the first real behavior change* (S(k) dim
  → `3·N_basis`; viewer-model `baseStructure` grows a basis).
  - **Done (compute core):** `cells.buildCellLabeling()` (τ + `n` in L units +
    per-τ masses/counts/basis) wired into `compute/pipeline.js`; the Bloch phase
    now indexes the relabel `n` (constant across frames), grouping by basis site
    τ. Default `P = I` reproduces the per-reference-number grouping **byte-for-
    byte** (asserted vs the validated fixture, |Δ|=0). `io/viewermodel.js` carries
    the τ-ordered `siteBasis` so eigvec row r ↔ site r. Node coverage:
    `test/cells_pipeline_test.mjs` (in `npm run validate`) — P=I regression,
    FCC primitive 4→1 collapse, and the fold demo (`S([1,0,0])==S(Γ)` vs
    `S([0.5,0,0])≠S(Γ)`) that motivates the conventional path.
  - **Done (conventional default path):** `highsym.buildConventionalBZModel()`
    (conventional reciprocal cell + `reciprocal.HIGH_SYM` points, X at ½, WS box
    BZ); `RunnerPage` defaults to it. FCC X is now `(½,0,0)` instead of the
    folded `(0,1,0) ≡ Γ`, so Γ→X de-symmetrises. Covered in `highsym_test.mjs`.
    Verified end-to-end in-browser on the GaTaSe ensemble: conventional cubic BZ
    renders, a GPU run produces **156 = 3·52** branches over 41 conventional
    k-points, viewer + per-site band character work, no console errors.
  - **Done (reference mode):** `options.referenceMode` = `per-atom` (default,
    `u = r − r̄_atom`, the validated behaviour) | `symmetrized` (`u = r − (R_n+bf_τ)`,
    shared per-τ site with nearest-image wrap). Exposed as the "Reference site"
    control on the runner. Node coverage in `cells_pipeline_test.mjs` [D]
    (symmetrized ≠ per-atom under static disorder; identical for statistically
    equal sites).
  - **Remaining:** surface `P` via the UI (Phase 2). The pipeline already accepts
    `options.computationCell.P`. (NB: the primitive seekpath BZ picker —
    `buildBZModel`, with W/K/U/L etc. — is no longer the runner default; it returns
    as the *unfolded* primitive option in Phase 3.)
- **Phase 2 — UI cell selector + custom supercell** (`Conventional | Custom
  n₁×n₂×n₃`).
- **Phase 3 — primitive cell via symmetry.** Derive primitive `P` from
  `analyzeBravais` (manual override fallback); add `Primitive` option → unfolded
  dispersion.
- **Phase 4 — polish.** High-sym path per cell, genuinely-lower-symmetry
  averages, mixed-occupancy/alloy site policy.

## Validation

- **Internal:** conventional = folding of primitive on the same data → identical
  eigenvalues, integer-multiple branch count (assert in tests).
- **Physics:** Γ acoustic → 0; branch count = `3·N_basis`; primitive FCC Γ→X
  **asymmetric**.
- **External:** compare against the loadable phonopy reference (`mp-1000` BaTe).
- Extend `npm run validate` with cell-framework fixtures.

## Risks / open questions

1. Symmetry robustness (Phase 3): trust `analyzeBravais.T` vs. a Niggli/Delaunay
   reduction; provide a manual override.
2. Mixed-occupancy basis sites (alloys): policy = error / average-mass /
   per-element fallback. Ordered crystals are fine.
3. Performance: `N_basis` grows with custom supercells; `eigh` is O(N³) → cap
   and warn.
4. Boundary: keep the WGSL kernel + `eigh` + band-connection; only reshape inputs
   and the viewer-model `baseStructure`.
