# Project Status & Handoff (May 26, 2026)

This document is a quick handoff to help contributors pick up work.

## Current state

- Core physics pipelines exist in both `src/` (CPU) and `src_gpu/` (JAX/GPU).
- The local runner app (`rmcph_gui/`) can browse datasets, build k-paths, run jobs, and hand results to the viewer.
- The `rmcph_gui` Fit Quality panel now renders an Rw-vs-configuration overview for F(Q) and G(r); clicking an overview point selects that configuration and expands the detailed observed/RMC/difference figures.
- The viewer (`viz/`) has major load/perf optimizations, including Web Worker compute and optional JSON fast-path.

## What is stable

- End-to-end `src_gpu` band workflow through GUI runner.
- `.rmc6f` and `Frac*.txt` config-family detection in GUI flow.
- Brillouin-zone / high-symmetry point workflow with seekpath mapping to conventional-cell k vectors.
- Fit Quality preview for RMCProfile output CSVs (`*_XFQ1.csv`, `*_FT_XFQ1.csv`) in `rmcph_gui`, including clickable Rw trend summaries, residual overlays, and Chart.js zoom/reset controls.
- Viewer-side S(Q,E) + DOS computation pipeline and delayed heavy 3D initialization.

## Known gaps / follow-ups

1. **Large `band.yaml` UX warnings** (especially when eigenvectors make files huge).
2. **CIF equilibrium-reference support** for displacement reference selection.
3. **`.rmc6f` parser optimization** for large ensembles (currently Python-loop heavy).
4. **Automated tests expansion** across GUI backend APIs + runner integration + viewer parsing paths.

## Recent development notes

- May 26, 2026: Refined `rmcph_gui` Fit Quality figures. Changes include a clickable Rw summary plot over all configurations with percent ticks and a selected-config guide, folded detailed S(Q)/F(Q) and G(r) plots that expand on summary click, external figure-header legends, solid residual baseline styling, hover tooltips, x-axis zoom/reset controls, Rw-only metric display, and tightened axis bounds to remove unnecessary empty plot regions.
- May 27, 2026: Added `PHYSICS_ALGORITHM_AUDIT.md`, a physics/algorithm handoff covering the active GPU runner flow, displacement and k-vector conventions, energy conversion, band output, risks, and prioritized implementation/debug TODOs.
- June 8, 2026: Added JSON-backed job-state persistence plus a `/api/jobs/latest` reconnect path. The Run view now restores the latest active or completed job after a page reload and resumes polling active jobs.

## Suggested next milestones

### Milestone A — Reliability
- Add restart-safe cancellation reconciliation.

### Milestone B — Performance
- Profile `.rmc6f` parse/read path and batch conversion.
- Add cached structured intermediate for repeated runs on same dataset.

### Milestone C — UX and Guardrails
- Add proactive size warnings for large `band.yaml` and suggest JSON export path.
- Improve validation messages for missing files / incompatible folder layouts.

### Milestone D — Testing and CI
- Add backend API tests for data/structure/reciprocal/jobs routes.
- Add deterministic smoke tests for `src_gpu/runner.py` output shape and metadata.
- Add viewer parser regression test fixtures for YAML vs JSON parity.

## Where to start (recommended)

1. Implement backend job-state persistence in `rmcph_gui/backend/core/jobs.py` (+ minimal API support in `rmcph_gui/backend/api/jobs.py`).
2. Add tests around state transitions (queued → running → done/cancelled/error).
3. Expose reconnect semantics in frontend `rmcph_gui/frontend/js/views/run.js`.

This sequence reduces user-facing failure modes before deeper optimization work.
