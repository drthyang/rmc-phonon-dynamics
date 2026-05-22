#!/bin/bash
# Launch the rmcph_gui backend (which also serves the frontend).
#
# Runs uvicorn from the rmcph_gui/ directory so `backend` is importable as a
# package. Uses the jax-metal env's python if available (it has src_gpu's deps
# + fastapi); otherwise falls back to whatever `python` is on PATH.
set -e

cd "$(dirname "$0")"

PORT="${PORT:-7236}"
JAX_METAL_PY="$HOME/miniforge3/envs/jax-metal/bin/python"

if [ -x "$JAX_METAL_PY" ]; then
    PY="$JAX_METAL_PY"
else
    PY="$(command -v python3 || command -v python)"
    echo "WARN: jax-metal env not found; using $PY (src_gpu compute may be unavailable)."
fi

echo "Serving rmcph_gui at http://localhost:${PORT}  (Ctrl-C to stop)"
exec "$PY" -m uvicorn backend.app:app --reload --port "${PORT}"
