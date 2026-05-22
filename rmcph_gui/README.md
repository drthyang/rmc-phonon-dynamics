# rmcph_gui — web interface for running `src_gpu` phonon calculations

A local web app to: select a data folder, view the crystal structure and
reciprocal/Brillouin zone, pick a k-path, and launch the `src_gpu` phonon-band
calculation — then view results in the optimized `viz/rmcph.html` viewer.

## Status

Built incrementally, phase by phase (tracked in the session task list):

| Phase | What | State |
|-------|------|-------|
| 0 | Scaffold: FastAPI backend + vanilla-JS frontend + `/api/ping` | ✅ |
| 1 | Data folder selection & parsing | ⬜ |
| 2 | Crystal structure 3D view (three.js) | ⬜ |
| 3 | Reciprocal cell / Brillouin zone + k-path selection | ⬜ |
| 4 | Refactor `test_run.py` → importable runner + job manager | ⬜ |
| 5 | Job submission UI, live progress, cancel/resume | ⬜ |
| 6 | Results → hand `band.yaml` to `viz/rmcph.html` | ⬜ |

## Architecture

```
Browser (frontend/)  ──HTTP/WebSocket──►  Backend (backend/)  ──imports──►  ../src_gpu
  vanilla JS + three.js                    FastAPI + uvicorn                  (jax/numpy/pymatgen)
```

- **Local single-user tool.** The backend runs on your machine; it needs
  filesystem + GPU access and serves the UI itself.
- Runs in the **`jax-metal` conda env** so one process can both serve HTTP and
  run the GPU compute (same env as `src_gpu`).
- **Results reuse `viz/rmcph.html`** — no rebuilding band/structure/S(Q,E)
  rendering.

## Layout

```
backend/
  app.py              FastAPI app; serves frontend + mounts /api
  config.py           repo-relative paths (DATA_ROOT, SRC_GPU_DIR, …)
  api/                route modules (ping; data/structure/jobs land later)
  core/
    runners/base.py   Runner plugin interface + registry (extensibility seam)
  models/             pydantic request/response schemas
frontend/
  index.html          shell + three.js import map
  js/ api.js state.js main.js  views/  lib/
  css/style.css
```

## Adding a calculation type (future)

Implement a `Runner` subclass in `backend/core/runners/`, declare its
`param_schema()`, and `register()` it. The frontend renders the settings form
from that schema — no view code changes needed.

## Run

```bash
# one-time: install backend deps into the env that runs src_gpu
conda activate jax-metal
pip install -r rmcph_gui/requirements.txt

# launch
bash rmcph_gui/run.sh          # → http://localhost:7236
```
