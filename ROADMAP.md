# RMC Phonon Dynamics — Development Roadmap

_Last updated: 2026-05-26. Current stable release: v0.1.0._

---

## v0.2 — Stability & Reliability
_Make the app safe to use in real research sessions without losing work._

- [ ] **Job-state persistence** — Backend stores running/completed job state to disk (`rmcph_gui/backend/core/jobs.py`); frontend reconnects and restores the Run panel after a page reload. Currently a crash or reload loses all job context.
- [ ] **`src/` CPU engine 2π fix** — `src/Calculators.py` still omits the 2π factor in the Bloch phase that was validated and corrected in `src_gpu/`. Align both engines.
- [ ] **CIF equilibrium-reference support** — The "Equilibrium file" picker in Step 1 already accepts `.cif` but raises `NotImplementedError`. Wire it through the reader.
- [ ] **Large `band.yaml` guardrails** — Warn before writing eigenvector-included YAML files that can reach hundreds of MB; suggest the JSON fast-path.

---

## v0.3 — Analysis Depth
_Extend what can be seen from a completed RMC run._

- [ ] **Neutron S(Q,ω) fit quality** — Add fit panels for neutron S(Q,ω) alongside the current X-ray F(Q)/G(r), so the same Rw-overview + Rietveld layout covers the neutron constraint too.
- [ ] **Partial PDF panel** — Optionally show G(r) partial pairs (Ga–Ga, Ga–Ta, …) with color-coded curves, toggled from the Fit Quality card. Data already exists in `*_PDFpartials.csv`; backend already returns it.
- [ ] **Temperature comparison** — Side-by-side overlay of Rw trends and S(Q,ω) between two ensembles (e.g. 5 K vs 250 K), using a second folder picker.
- [ ] **Peak / mode analysis** — Click on a ridge in the S(Q,ω) viewer to extract mode frequency vs. Q; export as a dispersion curve table.

---

## v0.4 — Performance
_Handle larger ensembles without stalling._

- [ ] **`.rmc6f` parser optimization** — The current Python-loop reader is slow for large ensembles (500+ configs, 52 atoms/cell). Replace with a vectorized NumPy reader or a compiled parser; benchmark against the Frac path.
- [ ] **Cached Sk intermediate** — The per-config S(k) computation is re-run on every job even if the ensemble hasn't changed. Add a content-hash–keyed cache in `results/skcache/` (partial plumbing already exists).
- [ ] **Streaming job progress** — Replace polling with a Server-Sent Events stream from the runner so the progress bar updates in real time without repeated HTTP round-trips.

---

## v0.5 — Export & Sharing
_Make outputs usable outside the app._

- [ ] **Figure export** — Add PNG/SVG download buttons to each Fit Quality chart and the viewer's S(Q,ω) panel. Chart.js has a built-in `toBase64Image()` path.
- [ ] **Data export** — Download the plotted F(Q), G(r), and Diff curves as CSV directly from the panel.
- [ ] **Viewer permalink** — Encode the current viewer state (dataset path, color scale, k-path slice) in the URL hash so a view can be shared or bookmarked.
- [ ] **`band.yaml` → JSON pre-conversion CLI** — A one-command script to pre-convert large YAML outputs to the JSON fast-path format for use with the standalone viewer.

---

## v0.6 — Testing & Infrastructure
_Make it safe to extend without regressions._

- [ ] **Backend API tests** — pytest coverage for all routes: `/api/data`, `/api/structure`, `/api/reciprocal`, `/api/jobs`, `/api/sqgr`.
- [ ] **Runner smoke tests** — Deterministic output-shape and metadata checks for `src_gpu/runner.py` on a small synthetic ensemble.
- [ ] **Viewer regression fixtures** — YAML vs JSON parity tests for the S(Q,ω) compute worker.
- [ ] **LICENSE file** — Add MIT or BSD-3-Clause to the repo root to formally enable collaborator use and modification.
- [ ] **CI pipeline** — GitHub Actions: lint + backend tests on every push to `main`.

---

## Future / Research Features
_Longer-horizon ideas that depend on the above being stable._

- [ ] **Neutron-weighted DOS fitting** — Compare computed phonon DOS against measured inelastic neutron data.
- [ ] **Multi-Q-point animation** — Animate atomic displacements for a selected phonon mode in the 3D viewer.
- [ ] **Cloud/HPC runner** — Submit jobs to a SLURM cluster via SSH rather than running locally; poll remote job state through the same API contract.
- [ ] **Shared design system** — Unify `rmcph_gui` and `viz/rmcph.html` onto a common CSS token set and component library (currently styled independently).

---

## Recommended starting point

1. **Job-state persistence** (`rmcph_gui/backend/core/jobs.py` + `rmcph_gui/frontend/js/views/run.js`) — correctness issue that affects every real session.
2. **`src/` CPU 2π fix** (`src/Calculators.py`) — silent numerical error, one-line change once the validated GPU path is used as reference.

Both are correctness issues that affect real use and require no new UI surface.
