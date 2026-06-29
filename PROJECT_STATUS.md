# Project Status & Handoff (June 29, 2026)

A quick orientation for contributors.

## Current state

- **The application is the browser app in [`web/`](web/)** (React + Vite +
  WebGPU). It loads an RMC ensemble (`.rmc6f` or `Frac*.txt` + companion
  `.rmc6f`), computes the displacement-covariance phonon bands on WebGPU,
  animates 3D modes, simulates powder INS S(|Q|,E) + DOS, and shows fit quality
  — all client-side. It is **hosted on GitHub Pages** (built from `web/` by CI).
- The Runner and Viewer use the **Cobalt light theme** (see
  [`web/DESIGN_NOTES.md`](web/DESIGN_NOTES.md) and the design handoffs in
  `archive/design_handoffs/`).
- The legacy Python engines (`src/`, `src_gpu/`), the FastAPI GUI (`rmcph_gui/`),
  and the standalone viewer (`viz/`) are **retired to [`archive/`](archive/)**
  for reference only.

## Stable / working

- WebGPU S(k) → `eigh` → band-connection → bands/DOS pipeline
  (`web/src/compute`, `web/src/math`), guarded by `npm run validate`.
- Crystal-system / high-symmetry / Brillouin-zone k-path selection.
- 3D mode animation, simulated INS heatmap (kinematic cutoff), phonon DOS.
- Fit-quality Rw overview + F(Q)/G(r) overlays from RMCProfile output.
- `band.yaml` / `band.json` export; `band.yaml`/`.json` load.

## Known gaps / follow-ups

1. **Browser support** — Chromium-only (WebGPU + File System Access API).
2. **CIF equilibrium-reference** support for the displacement reference.
3. **Large-ensemble parsing** of `.rmc6f` (vectorize the reader).
4. **Large `band.yaml` guardrails** — warn / prefer the JSON fast-path.
5. **Automated UI/regression coverage** beyond the science validate suite.

## Where to start

The boundary contract for the compute/io/math layers and the component prop
shapes is in [`web/DESIGN_NOTES.md`](web/DESIGN_NOTES.md) — read it first. UI
work lives in `web/src/pages` and `web/src/components`; keep
`web/src/{compute,io,math}` and `constants.js` behaviour unchanged (the validate
suite guards them). See [`ROADMAP.md`](ROADMAP.md) for prioritized next work.
