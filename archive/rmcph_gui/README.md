# rmcph_gui — RMC Phonon Runner

A local web app to drive the `src_gpu` phonon-band calculation end to end: pick a
data folder, inspect the crystal structure and Brillouin zone, build a k-path,
run the calculation with live progress, then open the results in the **RMC Phonon
Viewer** (`../viz/rmcph.html`). Single-user, runs on your machine, serves its own
UI.

## What it does

1. **Select data** — browse to an ensemble folder. Configs are auto-detected as
   either `Frac*.txt` **or** a numbered `.rmc6f` ensemble (see *Config sources*
   below). The structure `.rmc6f` (atom types + lattice) is found automatically.
2. **Displacement reference (hsym)** — default is the ensemble average; you can
   override it with an equilibrium `.rmc6f` file (`u = config − reference`).
3. **Structure** — 3D view of the unit cell (three.js: atoms + bonds).
4. **Brillouin zone & k-path** — seekpath-standardized BZ and high-symmetry
   points; build a path by clicking points, with per-segment k-point density and
   discontinuous "breaks".
5. **Run** — submit a phonon-bands job; watch a live progress bar (per k-point,
   then *connecting bands* / *writing band.yaml*); cancel mid-run.
6. **Results** — one click opens the band.yaml in the RMC Phonon Viewer
   (band structure + S(Q,E) + 3D modes).

## Config sources (auto-detected)

`Readers.list_configs(folder)` decides the family:

- **`.rmc6f` ensemble** (preferred when present): numbered `<stem>_<N>.rmc6f`
  with `N ≥ 1`, **excluding** `*AVERAGE.rmc6f` (running-average dumps), the
  un-numbered base (the structure file), and `_0` (initial structure). For
  `data/ensemble_20A_5K` this is the 500 files `GTS_5K_1…500.rmc6f`.
- **`Frac*.txt`**: checked in `folder/configs/` first, then `folder` (so a lone
  orphan `Frac_coord_1.txt` in an ensemble parent isn't mistaken for the set).

Both formats are read into the identical within-cell layout and give matching
results (verified to ~4e-5; `.rmc6f` is full precision vs Frac's 5 decimals).
**`.rmc6f` files are read-only source** — the Sk cache is written to
`results/skcache/`, never into the data folder.

## Conventions

- **k-vectors**: `kvec = 2π × (conventional-cell fractional coord)`. seekpath
  works in the primitive cell, so high-symmetry points carry both `frac`
  (primitive, for the BZ display) and `frac_conv` (conventional, fed to
  `src_gpu`, whose `cell_idx · kvec` phase tiles the conventional cell).
- **Engine**: the GUI uses the **GPU path (`../src_gpu`)**, validated by a
  reciprocal-lattice-periodicity test (`src_gpu/validate_kpath_2pi.py`).

## Architecture

```
Browser (frontend/)  ──HTTP──►  Backend (backend/)  ──imports──►  ../src_gpu
  vanilla JS + three.js          FastAPI + uvicorn                 (jax / numpy / pymatgen)
```

Runs in the **`jax-metal` conda env** so one process serves HTTP and runs the GPU
compute. The backend also serves the viewer and results for the hand-off:
`/` → frontend, `/api/*`, `/viz/*` → the viewer, `/results/*` → computed
`band.yaml` files.

## Layout

```
backend/
  app.py                 FastAPI app; mounts /api, /viz, /results, frontend
  config.py              repo-relative paths (DATA_ROOT, SRC_GPU_DIR, RESULTS_DIR, …)
  api/                   ping, data, structure, reciprocal, jobs
  core/
    data_access.py       folder browse + dataset inspection (config-family detect)
    reciprocal.py        seekpath BZ + high-symmetry points (frac / frac_conv)
    structure.py         unit-cell build for the 3D view
    session.py           single-user in-memory active dataset
    jobs.py              threaded JobManager (submit / progress / cancel)
    runners/
      base.py            Runner plugin interface + registry
      phonon_bands.py    PhononBandsRunner (wraps src_gpu/runner.run_bands_segments)
frontend/
  index.html             shell + three.js import map
  js/  api.js  state.js  main.js  views/{folder,structure3d,bz,run,filepicker}.js
  css/style.css
```

## Run

```bash
# one-time: install backend deps into the env that runs src_gpu
conda activate jax-metal
pip install -r rmcph_gui/requirements.txt

# launch (auto-uses the jax-metal env if present)
bash rmcph_gui/run.sh            # → http://localhost:7236
```

## Adding a calculation type

Implement a `Runner` subclass in `backend/core/runners/`, declare its
`param_schema()`, and `register()` it. The frontend renders the settings form
from that schema — no view code changes needed. `phonon_bands.py` is the
reference implementation.

## Status

Phases 0–6 complete (data selection, structure view, BZ/k-path, importable
runner + job manager, submission UI with live progress + cancel, results
hand-off) plus the unified Frac/`.rmc6f` reader and the equilibrium-file
displacement reference.

Known follow-ups: reconnect to a running job after a page reload; surface a
clearer warning for very large `band.yaml` (eigenvectors); CIF displacement
reference (only `.rmc6f` is supported); optimize the `.rmc6f` parser (currently
a Python loop) before routinely running full 500-config rmc6f ensembles.
