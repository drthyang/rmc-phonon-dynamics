# Computation-cell framework — implementation plan

Status: **Phase 0 in progress.** This is the reference plan; build piece by piece
and check work against it.

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

## Phases

- **Phase 0 — data model + re-labeling (pure, no behavior change).**
  `math/cells.js` (`det3/inv3/matMul3/vecMat3`, `cellVectors`, `tilesSupercell`,
  `relabelAtoms`) + Node tests. Every atom → exactly one (n,τ); basis consistent
  across cells; positions reconstruct. *← current.*
- **Phase 1 — generalize S(k) to per-basis-site over arbitrary P.** Reuse the
  WGSL phase kernel; group by basis site instead of element; feed `n` in L
  units. Default `P = I` with the **conventional BZ path** (X at ½) → fixes the
  Γ→X symmetry self-consistently.
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
