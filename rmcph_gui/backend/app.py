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

from .config import APP_NAME, APP_VERSION, FRONTEND_DIR
from .api import ping, data

app = FastAPI(title=APP_NAME, version=APP_VERSION)

# API routes first; they take precedence over the catch-all static mount below.
app.include_router(ping.router, prefix="/api")
app.include_router(data.router, prefix="/api")

# Serve the frontend at the root. html=True makes "/" return index.html.
# Registered last so /api/* is matched before the static handler.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
