# RMC Phonon Dynamics

[![Deploy](https://github.com/drthyang/rmc-phonon-dynamics/actions/workflows/deploy-pages.yml/badge.svg)](https://github.com/drthyang/rmc-phonon-dynamics/actions/workflows/deploy-pages.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **browser-based phonon calculator** for [RMCProfile](https://rmcprofile.ornl.gov)
ensembles — no install, no setup, no server. Open the page, pick your run folder,
and get phonon band structures, animated 3D modes, and simulated neutron spectra
in seconds.

> **What makes it different:** the phonons come straight from your
> **RMC ensemble — atomic configurations fitted to experimental total-scattering
> and diffraction data** (real-space PDF + reciprocal-space S(Q)/Bragg). They are
> extracted from the *displacement covariance* of those configurations, so the
> dynamics are a direct consequence of the measured structure — **not** taken
> from a spectroscopy measurement (INS/Raman/IR) or a lattice-dynamics/DFT model.

### ▶️ Open the app — [drthyang.github.io/rmc-phonon-dynamics](https://drthyang.github.io/rmc-phonon-dynamics/)

1. Visit the link above.
2. Click **Select directory** and choose your RMC ensemble folder (a numbered
   `.rmc6f` ensemble, or `Frac*.txt` configs with a companion `.rmc6f`).
3. Build a **k-path** on the Brillouin zone, set the temperature, and **Run** —
   the band structure, 3D modes, INS S(|Q|,E), and phonon DOS render right in
   your browser.

🔒 **Your data never leaves your device.** Run files are read and *every*
calculation happens locally in your browser — nothing is ever uploaded to any
server. It's a private, secure way to analyze unpublished data. (Your browser's
folder picker may say “Upload”, but nothing is sent anywhere.)

⚡ **Computed on your GPU.** The phonon displacement-covariance S(k) →
diagonalization runs on **WebGPU**, directly on your machine — fast, with no
backend to install or wait on.

🖥️ **Use a Chromium browser** (Chrome or Edge): the app needs WebGPU and the
File System Access API, which Firefox/Safari don't yet provide.

## What it does

- **Phonon band structure** — dispersion E(k) along a high-symmetry path, with a
  hover readout, drag-to-zoom, and soft-mode (imaginary-frequency) highlighting.
- **3D mode viewer** — click a band point to animate that mode; ball-and-stick /
  spacefill / wireframe, bonds, displacement vectors, and per-element colours.
- **Simulated INS** — powder-averaged S(|Q|,E) heatmap with a kinematic cutoff,
  plus the phonon density of states.
- **Fit quality** — per-configuration Rw overview and F(Q)/G(r) overlays from
  RMCProfile output.
- **Export** — phonopy-compatible `band.yaml` / `band.json`.

Everything above runs **100% client-side** — the hosted link is static files;
there is no server doing the work.

## Requirements

Just a **Chromium-based browser** (Chrome or Edge) — the app needs both
**WebGPU** and the **File System Access API** (`showDirectoryPicker`). A secure
context (the hosted HTTPS site, or `http://localhost`) is required. There is
nothing to install to use the hosted app.

## Run locally (optional)

Only needed for development — to *use* the app, just open the
[hosted link](https://drthyang.github.io/rmc-phonon-dynamics/).

```bash
cd web
npm install
npm run dev          # open the printed http://localhost:5173
npm run validate     # science + UI-mapping tests
npm run build        # static build → web/dist/
```

See [`web/README.md`](web/README.md) for the in-app workflow and
[`web/FEATURE_PARITY_REPORT.md`](web/FEATURE_PARITY_REPORT.md) for how the
in-browser results map to the original Python implementation.

## Repository layout

| Path | Purpose |
| --- | --- |
| [`web/`](web/) | The application — React + Vite + WebGPU (the hosted link above). |
| [`archive/`](archive/) | Retired Python engines (`src/`, `src_gpu/`), the FastAPI GUI (`rmcph_gui/`), the standalone viewer (`viz/`), design handoffs, and historical notes. |
| `.github/workflows/` | CI that builds `web/` and deploys it to GitHub Pages on every push to `main`. |

`data/` and `results/` are local-only (git-ignored).

## References and acknowledgments

The data-processing pipeline (RMCProfile → phonopy YAML), INS simulation, DOS
calculations, and the WebGPU 3D mode viewer were developed for this project by
Tsung-Han Yang. The viewer follows the phonon-mode display convention of the
**phononwebsite** project [4], which also underpins the original viewer kept in
`archive/viz/`.

1. Dove, M. T. (1993). *Introduction to Lattice Dynamics*. Cambridge University Press.
2. Goodwin, A. L., Tucker, M. G., Dove, M. T., & Keen, D. A. (2004). Phonons from powder diffraction: A quantitative model-independent evaluation. *Physical Review Letters*, **93**, 075502. https://doi.org/10.1103/PhysRevLett.93.075502
3. Tucker, M. G., Keen, D. A., Dove, M. T., Goodwin, A. L., & Hui, Q. (2007). RMCProfile: Reverse Monte Carlo for polycrystalline materials. *Journal of Physics: Condensed Matter*, **19**, 335218.
4. Miranda, H. P. C. *phononwebsite*. https://github.com/henriquemiranda/phononwebsite

## License

Released under the [MIT License](LICENSE) © 2026 Tsung-Han Yang.
