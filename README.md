# RMC Phonon Dynamics

[![Deploy](https://github.com/drthyang/rmc-phonon-dynamics/actions/workflows/deploy-pages.yml/badge.svg)](https://github.com/drthyang/rmc-phonon-dynamics/actions/workflows/deploy-pages.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)

A **browser-based symmetry, phonon, and inelastic neutron scattering (INS) studio** for [RMCProfile](https://rmcprofile.ornl.gov) supercell ensembles — plus an optional universal MLIP companion (`rmcph-mlip`).

### ▶️ Open the app — [drthyang.github.io/rmc-phonon-dynamics](https://drthyang.github.io/rmc-phonon-dynamics/)

## Workflow

1. **Step 1 · Supercell Symmetry & Domain Studio** — Stream 1 to 500+ `.rmc6f` supercells in `O(1)` memory with Monte Carlo null calibration, spatial cross-validation (`CV ΔF`), domain correlation length `ξ(r)`, and automatic `Γ` (`1×1×1`), centering-breaking (`1×1×1·P`), and zone-boundary (`1×1×2`, `2×2×1`, `2×2×2`) subgroup discovery.Works in **any modern browser** (Chrome, Safari, Firefox, Edge).
2. **Step 2 · Phonons & Unfolded Parent Brillouin Zone** — Inspect harmonic MLIP (`_mlip_dm.json` / `band.yaml`) or RMC covariance dispersions, unfold lower-symmetry supercell modes onto the parent Brillouin zone (`A(Q, E)`), animate 3D eigenvectors, inspect `FixSymmetry` energetics (`ΔE/atom`), and compute coherent + incoherent powder `S(|Q|, E)`.
3. **Step 3 · Experimental INS Compare** — Quantitatively rank candidate local symmetries (`Rw`, `cos`, `χ²`) against experimental 1D GDOS / `S(E)` and 2D `S(|Q|, E)` with direct-geometry TOF chopper resolution (`SEQUOIA`, `CNCS`, `ARCS`, `LET`) and parameter-free Sjolander multiphonon (`n ≥ 2`) backgrounds.
4. **Legacy WebGPU RMC Runner** — Direct displacement-covariance `S(k)` diagonalization across 500+ RMC configurations on WebGPU (Chrome / Edge).

🔒 **Your data stays on your device.** All browser analysis runs 100% locally on your machine — nothing is uploaded to any external server.

## What it does

- **Cell & symmetry** — the space group is auto-detected with a tolerance ladder
  (`P1 → … → F-43m`) and Wyckoff labels; fold to the **conventional or primitive**
  cell or build a **custom supercell**, and optionally **impose symmetry** to pool
  symmetry-equivalent sites and enforce the required branch degeneracies.
- **Interactive Brillouin zone** — the true Wigner–Seitz zone for the chosen cell;
  click high-symmetry points to lay out a **k-path**.
- **Phonon band structure** — dispersion E(k) from the RMC displacement covariance,
  with a hover readout, drag-to-zoom, and soft-mode (imaginary-frequency) highlighting.
- **3D mode viewer** — click a band point to animate that mode; ball-and-stick /
  spacefill / wireframe, bonds, displacement vectors, and per-element colours.
- **Simulated INS + DOS** — powder-averaged S(|Q|,E) heatmap with a kinematic
  cutoff, plus the phonon density of states.
- **Fit quality** — per-configuration Rw overview across **X-ray and neutron
  S(Q) & G(r) plus Bragg**, each as measured-vs-model overlays; flag and exclude
  poorly-fit configurations from the run.
- **Export** — phonopy-compatible `band.yaml` / `band.json` and S(Q,E) CSV.

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
| [`web/`](web/) | React + Vite web application (Steps 1–3 work in all modern browsers; Legacy RMC uses WebGPU). |
| [`mlip/`](mlip/) | Optional Python `rmcph-mlip` CLI & local companion server (`uv run rmcph-mlip serve`) for `FixSymmetry` relaxation, multi-uMLIP (`MACE-MP-0`, `MACE-MPA-0`, `MACE-OMAT-0`, `MACE-r²SCAN`, `MatterSim`, `ORB-v3`, `SevenNet`, `CHGNet`) phonons, and coherent powder INS `S(|Q|, E)`. |
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

Released under the [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE) © 2026 Tsung-Han Yang.

*This project is personal work, developed and maintained in my personal capacity.*
