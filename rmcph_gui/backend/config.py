"""Backend configuration: filesystem roots and shared paths.

Paths are derived relative to the repository so the tool works regardless of
where it is checked out.  DATA_ROOT is the default starting point for the
server-side directory browser (Phase 1).
"""
from pathlib import Path

# rmcph_gui/backend/config.py  →  parents[2] == repo root
REPO_ROOT   = Path(__file__).resolve().parents[2]
SRC_GPU_DIR = REPO_ROOT / "src_gpu"
VIZ_DIR     = REPO_ROOT / "viz"
DATA_ROOT   = REPO_ROOT / "data"
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# Where computed band.yaml / results are written (Phase 4+).
RESULTS_DIR = REPO_ROOT / "results"

# Server-side directory browser (Phase 1):
#   - browsing starts here
#   - navigation is clamped to BROWSE_ROOT to avoid wandering the whole disk
DATA_BROWSE_START = DATA_ROOT if DATA_ROOT.is_dir() else REPO_ROOT
BROWSE_ROOT       = Path.home()

APP_NAME    = "rmcph_gui"
APP_VERSION = "0.1.0"
