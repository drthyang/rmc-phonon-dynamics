# Computation-cell framework — implementation plan

Status: **Phases 0–3 complete.** Relabel-driven per-basis-site S(k) (default
`P = I` = zero regression), conventional default k-path (the Γ→X fix), reference
mode, and a UI computation-cell selector (`Conventional | Primitive | Custom
n₁×n₂×n₃`, the path mapped `q_cell = P·q_conv`). Verified in-browser + `npm run
validate` (`cells_pipeline_test.mjs` [A–F], `highsym_test.mjs`). **Next: Phase 4**
— robust primitive folding on disorder-broken averages (symmetrized-reference
relabel / tolerance / Niggli), per-cell high-sym paths, alloy site policy. This is
the reference plan; build piece by piece.

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
  - **Done:** `P` is surfaced via the UI (see Phase 2). NB: the primitive seekpath
    BZ picker (`buildBZModel`, W/K/U/L) is no longer the runner default; it returns
    as the *unfolded* primitive option in Phase 3.
- **Phase 2 — UI cell selector + custom supercell** (`Conventional | Custom
  n₁×n₂×n₃`). **✅ DONE.** Key design: the user always picks the path on the
  CONVENTIONAL BZ; the cell choice only sets `P`, and `runCalculation` maps each
  point `q_cell = P·q_conv` before the Bloch phase (DOS `genGrid` is already in
  cell-frac). So no per-cell BZ rebuild. `RunnerPage` "Computation cell" selector
  (`Conventional | Custom n₁×n₂×n₃` → `P = diag(n)`) passes `computationCell.P` to
  `runCalculation`/`computeDOSGrid`; results carry `compCell:{P,L}`. Cell-aware
  consumers: `viewermodel` shows the computation cell `L` and attaches `bandRecip`
  (conventional reciprocal) so band-path distances stay physical; `BandStructurePlot`
  prefers `bandRecip`. Large-`N` guard warns when `3·N_basis > 600`. Node coverage:
  `cells_pipeline_test.mjs` [E] (`q_cell = P·q_conv`; custom 2×1×1 folds conventional
  X onto the supercell Γ; genuine ½ point stays distinct).
- **Phase 3 — primitive cell via symmetry. ✅ DONE.** `analyzeBravais` now returns
  the centering matrix `M` (`A_prim = M·A_conv`); the runner's `Primitive` option
  (shown only for centered lattices) sets `P = M`, and the generic `q_cell = P·q_conv`
  machinery unfolds. Verified on ideal FCC (`cells_pipeline_test.mjs` [F]): `P = M`
  collapses 4→1 sites, `q_cell = M·X(½,0,0) = (0,¼,¼)`, `S(X) ≠ S(Γ)`.
  **Verified end-to-end on real data:** the GaTaSe ensemble (Ga₄Ta₁₆Se₃₂, 8×8×8
  supercell) folds correctly to **13 primitive sites → 39 branches**; the viewer
  shows the primitive (rhombohedral) cell and a visibly unfolded, sparser
  dispersion. (An earlier "stays at 52 sites" observation was a UI-testing artifact
  — clicking Run before the Primitive state settled ran the conventional cell, `P=I`,
  52 sites × 8³; the "512×" in that stale warning is the conventional signature.)
  The runner still relabels the reference basis to show the TRUE site/branch count
  and flags "avg not centered" *if* a dataset's average genuinely breaks the
  centering — a real safety, just not triggered by GaTaSe.
- **Phase 4 — polish (in progress).**
  - **Done — `nCells` / labeling diagnostics fix.** `relabelAtoms` counted the
    *distinct* cell-index tuples for `nCells`, which a sheared sub-lattice (the FCC
    primitive cell over a rectangular RMC supercell) inflates — each basis site
    gets its own offset set of `n`, so distinct-`n` ≫ the true cell count (13 vs 4,
    62 vs 32). That made the validation flag *every* basis site even on a perfect
    fold (the earlier warning flood — which fires even on a CORRECT 13-site fold,
    since that fold is itself a sheared sub-lattice). Now `nCells` = the **modal
    per-site count**; the physics is unchanged (`n` only needs to be consistent
    within a site, which it is). Diagnosis (scratchpad): a synthetic GaTa₄Se₈-like
    13-site F cell folds cleanly to 13 at every supercell size and disorder ≤ σ0.02,
    and the real GaTaSe ensemble folds to 13 too — the **folding code is correct**.
    Regression: `cells_test.mjs` multi-site-F case.
  - **Done — symmetry-residual readout.** `relabelAtoms` now returns a per-site
    residual + `maxResidual`: the RMS cartesian distance (Å, minimal-image) of a
    folded site's members from their shared symmetrized position `R_n + bf_τ`. ≈ 0
    for a clean fold, and it grows with the static offset the cell averages over —
    a direct measure of *how much symmetry the cell choice imposes* (the plan's
    symmetry-vs-statistics knob, quantified). The runner shows it in the cell hint
    (`… · ⌀0.NN Å`), warn-colored above 0.3 Å. Test: `cells_test.mjs` (ideal → ~0,
    ±0.05 Å disorder → ~0.05 Å).
  - **Done — symmetry finder (Stage 1, report-only).** `math/symmetry.js`: a pure,
    fully-offline space-group operation finder (no WASM / no server). Point ops =
    integer matrices (entries {-1,0,1}, |det|=1) preserving the metric `RᵀGR=G`;
    space ops add the translations `t` that map the basis onto itself within a
    cartesian tolerance. Returns op counts + the residual (Å) of the fit, so
    symmetry is traceable **as a function of tolerance** — which pairs with the
    per-site residual readout. Verified against canonical groups (`symmetry_test.mjs`,
    in `npm run validate`): Pm-3m→48, Fm-3m→192, Im-3m→96, and the real GaTa₄Se₈
    → point group -43m (Td, order 24) × 4 F-centering = 96 (F-43m, No. 216), ~0
    residual, 18 ms on 52 sites. Shown next to the Bravais label as `N sym-ops · ⌀Å`.
    This is the intended replacement for FINDSYM/spglib in a static app (chosen for
    the offline constraint). **Stage 2 (next): drive the folding from these ops** —
    derive `P`/the orbits from the detected operations instead of the centering-only
    heuristic, and symmetrize accordingly.
  - **Remaining:** Stage 2 (operation-driven folding); optional fold-tolerance knob;
    per-cell high-sym path; mixed-occupancy/alloy site policy.

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
