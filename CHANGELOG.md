# Changelog

## 2026-06-29 — Browser/WebGPU app becomes the application; Cobalt redesign; hosted

- **The app is now fully in the browser** (`web/`, React + Vite + WebGPU): load an
  RMC ensemble, compute phonon bands on WebGPU, animate 3D modes, simulate INS
  S(|Q|,E) + DOS — no backend server.
- **Legacy retired to `archive/`**: the Python engines (`src/`, `src_gpu/`), the
  FastAPI GUI (`rmcph_gui/`), the standalone viewer (`viz/`), the physics audit,
  and the design handoffs. History preserved via `git mv`.
- **Runner & Viewer redesigned** to the Cobalt light theme. Viewer highlights:
  red band structure with drag-box (data-domain) zoom; 3D mode viewer with a
  palette colour picker, bond thickness, absolute Å bond cutoff, adjustable
  shading strength, vector colour and on-atom vector origin, and a
  container-responsive canvas; live meV↔THz; `band.yaml`/`band.json` export.
- **Hosted on GitHub Pages** via a CI workflow that builds `web/` and deploys it
  on every push to `main`.

## 2026-05-22 — `viz/` S(Q,E) viewer: 141× faster file load

Cut time-from-file-selection to S(Q,E) heatmap shown from **~37.9 s to ~0.27 s**
(JSON path) on a representative 8×8×8 / 52-atom-cell `band.yaml`.

| Stage | Total file→S(Q,E) | vs original |
|-------|------------------:|------------:|
| Original (main-thread parse + blocking 3D viewer) | 37 853 ms | 1× |
| Defer phononwebsite 3D load until after S(Q,E) renders | 1 648 ms | 23× |
| Move YAML parse + THz→meV into the Web Worker | 1 338 ms | 28× |
| Accept `band.json` (≈10× faster to parse than YAML) | **268 ms** | **141×** |

### How

- **Web Worker compute** (`viz/sqeworker.js`): S(Q,E) and DOS run off the main
  thread. DOS rewritten as histogram + single Gaussian convolution (~30–60×);
  Gaussian lookup table replaces `Math.exp` in the E- and Q-smear inner loops
  (~2–4× on S(Q,E)); zero-scattering atoms skipped.
- **Deferred 3D viewer**: phononwebsite's synchronous supercell-mesh build
  (~35 s on a large cell) was the dominant cost. It's now intercepted in the
  capture phase and re-dispatched only after the S(Q,E) heatmap has painted, so
  the user sees results in ~1.5 s instead of staring at a frozen tab.
- **Parse in the worker**: ship the raw file text (cheap string transfer)
  instead of a structured-cloned `ydata` tree; parse with `js-yaml` on the
  worker thread.
- **`band.json` support + "Save JSON"**: after a YAML load, one click saves a
  THz-encoded JSON copy. Re-loading it uses `JSON.parse` (~115 ms vs ~1134 ms
  for `js-yaml`). The saved JSON is valid YAML 1.2 with THz frequencies, so the
  same file (renamed `.yaml`) still feeds phononwebsite's 3D structure and band
  plot. Round-trips losslessly.
- **Render-only inputs** (colormap, log scale) redraw without recomputing.

All numerical output (S(Q,E), DOS, band energies) is unchanged.
