# RMC Phonon Dynamics — Development Roadmap

_Last updated: 2026-06-29. The application is the browser app in [`web/`](web/),
hosted on GitHub Pages._

---

## Done

- [x] **Browser/WebGPU migration** — the full runner + viewer now run client-side
      in `web/`; the Python engines and GUI are retired to `archive/`.
- [x] **Cobalt redesign** of the Runner and Viewer.
- [x] **GitHub Pages hosting** via a CI build of `web/`.
- [x] **band.yaml / band.json** export and load.

## In progress — computation-cell framework

Choose the unit cell S(k) is computed over (conventional / primitive / custom
supercell). Fixes the spurious Γ→X mirror symmetry (a cell-convention mismatch,
not the `2π`). Full plan + reference-mode discussion in
[`web/docs/cell-framework-plan.md`](web/docs/cell-framework-plan.md).

- [x] **Phase 0** — `math/cells.js` re-labeling (atom → cell n, basis τ) + tests.
- [ ] **Phase 1** — per-basis-site S(k) over an arbitrary cell + cell-aware
      reference (symmetrized-site vs per-atom mode); default conventional cell
      with a self-consistent conventional BZ path. *(first behavior change)*
- [ ] **Phase 2** — UI cell selector + custom supercell.
- [ ] **Phase 3** — primitive cell via symmetry (the unfolded dispersion).
- [ ] **Phase 4** — polish (per-cell high-sym path, lower-symmetry averages,
      mixed-occupancy policy).

## Near-term

- [ ] **CIF equilibrium-reference** — accept a `.cif` for the displacement
      reference (currently `.rmc6f`/ensemble-average only).
- [ ] **Large-ensemble `.rmc6f` parsing** — vectorize the reader for 500+ configs.
- [ ] **Large `band.yaml` guardrails** — warn before writing eigenvector-heavy
      YAML; steer toward the JSON fast-path.
- [ ] **Wider validation** — UI-mapping/regression fixtures beyond the science
      `npm run validate` suite.

## Analysis depth

- [ ] **Partial PDF / partial DOS** — per-pair G(r) and per-element DOS overlays.
- [ ] **Temperature comparison** — overlay two ensembles (e.g. 5 K vs 250 K).
- [ ] **Mode extraction** — pick a ridge in S(|Q|,E) to export frequency vs Q.
- [ ] **Figure / data export** — PNG/SVG of plots and CSV of plotted curves.

## Future / research

- [ ] **Broader browser support** — fallbacks where WebGPU / File System Access
      are unavailable (e.g. file-input load path for non-Chromium).
- [ ] **Neutron-weighted DOS fitting** against measured inelastic data.

> The pre-migration roadmap (legacy `rmcph_gui`/`src_gpu` milestones) is part of
> the git history; those components now live in `archive/`.
