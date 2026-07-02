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
2. In **Cell & symmetry**: pick the **base cell** (unit cell or a custom
   n₁×n₂×n₃ supercell); its **space group** is auto-detected and shown with a
   tolerance ladder (`P1 → … → F-43m`) and Wyckoff labels — press **avg** to
   detect on the ensemble mean. Choose the **fold** (Conventional | Primitive),
   and optionally **Impose symmetry** to pool symmetry-equivalent sites and
   enforce branch degeneracies.
3. The **Brillouin zone** follows the chosen cell; build a **k-path** by clicking
   high-symmetry points (or use **Default**).
4. Set **Temperature** and **points per segment**, then **Run Calculation**.
5. Inspect the **band structure**; **click a band point** to drive the **3D
   mode** viewer. **Export band.yaml** for phonopy-compatible output.
6. In the **Simulated INS** panel, set the energy window and **Run INS** to get
   S(|Q|,E) + DOS; **Export S(Q,E) CSV**.

## Validate the science

```bash
npm run validate      # deterministic science + UI-mapping test suite
```

The suite pins the physics against an **independent pure-Python reference**
(`test/reference.json`, regenerate with `npm run gen-reference`) and guards the
core math end to end:

- `ENERGY_CONV`, the **2π Bloch phase** (`S(k=G) == S(Γ)`), and S(k) grouped by
  basis site → `3 × N_basis` bands, all matching the reference;
- the complex Hermitian eigensolver reconstructs `A = V Λ V†` **and** returns
  orthonormal eigenvectors under degeneracy;
- symmetry detection + S(k) symmetrization is an exact fixed point for a
  symmetric ensemble (little-group projection convention);
- the fit-quality reader classifies X-ray/neutron/Bragg CSVs correctly; and
- a **synthetic-dispersion benchmark** recovers a known analytic E(k) and its
  eigenvector polarizations from a generated thermal ensemble.

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
| Cell, space-group & symmetrization | `src/math/cells.js`, `src/math/symmetry.js`, `src/math/symmetrize.js` |
| k-path / reciprocal / Brillouin zone | `src/math/reciprocal.js`, `src/math/bravais.js`, `src/math/highsym.js`, `src/math/brillouin.js` |
| Fit quality (X-ray/neutron/Bragg) | `src/io/sqgr.js`, `src/components/FitQuality.jsx` |
| band.yaml export | `src/io/writers.js` |
| INS S(Q,E) + DOS | `src/compute/ins.js`, `src/io/sqeworker.js`, `src/components/InsPanel.jsx` |
| Viewers | `src/components/BandStructurePlot.jsx`, `CrystalViewer.jsx`, `BrillouinZoneViewer.jsx` |

See `FEATURE_PARITY_REPORT.md` for the legacy-vs-new parity matrix, the science
preserved, and the documented limitations carried into the next (UI) stage.
