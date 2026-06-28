# RMC Phonon Dynamics — Browser / WebGPU App

A static, browser-based migration of the legacy `rmcphonon` runner + phonon
viewer. It loads an RMC ensemble (`.rmc6f` or `Frac*.txt`), computes the
mass-weighted displacement-covariance phonon band structure on **WebGPU**,
connects bands, animates 3D vibrational modes, and simulates powder INS
S(|Q|,E) and the phonon DOS — all client-side, with no backend server.

## Requirements

- A **Chromium-based browser** (Chrome/Edge) — needs both **WebGPU** and the
  **File System Access API** (`showDirectoryPicker`). Firefox/Safari currently
  cannot select a folder. A secure context (http://localhost or HTTPS) is
  required.
- Node 18+ and Python 3 (Python only for regenerating the validation reference).

## Run locally

```bash
cd web
npm install
npm run dev        # open the printed http://localhost:5173
```

Then in the app:

1. **Select Directory** → pick a folder with a numbered `.rmc6f` ensemble, or
   `Frac*.txt` configs **plus** a companion `.rmc6f` (needed for the lattice and
   reference-number → element map). The repo's `data/5K_ini/` and
   `data/ensemble_20A_5K/` work.
2. The detected crystal system and high-symmetry points appear; build a
   **k-path** by clicking spheres (or use **Default**).
3. Set **Temperature** and **points per segment**, then **Run Calculation**.
4. Inspect the **band structure**; **click a band point** to drive the **3D
   mode** viewer. **Export band.yaml** for phonopy-compatible output.
5. In the **Simulated INS** panel, set the energy window and **Run INS** to get
   S(|Q|,E) + DOS; **Export S(Q,E) CSV**.

## Validate the science

```bash
npm run validate      # node test/validate.mjs
```

This checks, against an **independent pure-Python reference**
(`test/reference.json`, regenerate with `npm run gen-reference`):

- `ENERGY_CONV` matches `src_gpu/constants.py` (the old web value was wrong by ~3.5e5×);
- S(k) is grouped by **reference number** → `3 × N_basis_sites` bands;
- the **2π Bloch phase** holds: `S(k=G) == S(Γ)` for reciprocal vectors G;
- the complex Hermitian eigensolver reconstructs `A = V Λ V†`.

## Build & deploy (GitHub Pages)

```bash
npm run build                                   # -> web/dist (base = '/')
# For a PROJECT Pages site served at /<repo>/:
VITE_BASE=/rmc-phonon-dynamics/ npm run build
```

`dist/` is fully static. Publish it to Pages (e.g. via an Actions workflow that
builds `web/` and uploads `web/dist`). No server is required at runtime. Note
the browser caveat above — Pages is HTTPS (secure context, OK), but folder
selection still needs Chromium.

## Architecture

| Layer | File |
|---|---|
| Data loading / parsing | `src/io/readers.js`, `src/io/worker.js` |
| WebGPU S(k) kernel | `src/compute/Sk_kernel.wgsl`, `src/compute/engine.js` |
| Pipeline (S(k) → eigh → connect) | `src/compute/pipeline.js` |
| Diagonalization / band connection | `src/math/diagonalize.js`, `src/math/band_connection.js` |
| k-path / reciprocal (lattice-aware) | `src/math/reciprocal.js` |
| band.yaml export | `src/io/writers.js` |
| INS S(Q,E) + DOS | `src/compute/ins.js`, `src/io/sqeworker.js`, `src/components/InsPanel.jsx` |
| Viewers | `src/components/BandChart.jsx`, `CrystalViewer.jsx`, `BrillouinZoneViewer.jsx` |

See `FEATURE_PARITY_REPORT.md` for the legacy-vs-new parity matrix, the science
preserved, and the documented limitations carried into the next (UI) stage.
