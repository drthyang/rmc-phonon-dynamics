"""rmcph_gui backend — FastAPI app.

Serves the vanilla-JS frontend and exposes the /api/* routes. Runs inside the
`jax-metal` conda env so the same process can import src_gpu (jax/numpy/
pymatgen) for the compute phases.

Launch:
    bash rmcph_gui/run.sh
or:
    uvicorn backend.app:app --reload --port 7236   # from inside rmcph_gui/
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .config import APP_NAME, APP_VERSION, FRONTEND_DIR, VIZ_DIR, RESULTS_DIR
from .api import ping, data, structure, reciprocal, jobs, sqgr

app = FastAPI(title=APP_NAME, version=APP_VERSION)

# API routes first; they take precedence over the catch-all static mount below.
app.include_router(ping.router, prefix="/api")
app.include_router(data.router, prefix="/api")
app.include_router(structure.router, prefix="/api")
app.include_router(reciprocal.router, prefix="/api")
app.include_router(jobs.router, prefix="/api")
app.include_router(sqgr.router, prefix="/api")

# Results hand-off (Phase 6): serve the S(Q,E) viewer and the computed band.yaml
# files at the same origin, so a finished job can deep-link the viewer to its
# output (/viz/rmcph.html?band=/results/<file>). Mounted before the "/"
# catch-all so these prefixes win.
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/viz", StaticFiles(directory=str(VIZ_DIR), html=True), name="viz")
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

# Serve the frontend at the root. html=True makes "/" return index.html.
# Registered last so /api/* and the static prefixes above match first.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
