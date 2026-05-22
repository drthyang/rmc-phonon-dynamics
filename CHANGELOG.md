# Changelog

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
