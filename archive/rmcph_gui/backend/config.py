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
JOBS_STATE_FILE = RESULTS_DIR / "jobs_state.json"

# Server-side directory browser (Phase 1):
#   - browsing starts here
#   - navigation is clamped to BROWSE_ROOT so requests can't escape the volume.
#     This is the filesystem anchor of the checkout ("/" on POSIX, "C:\\" on
#     Windows) — not Path.home(), because the repo/data may live outside the
#     home dir (e.g. an external volume), which would otherwise wall off the
#     folder picker.
DATA_BROWSE_START = DATA_ROOT if DATA_ROOT.is_dir() else REPO_ROOT
BROWSE_ROOT       = Path(REPO_ROOT.anchor)

APP_NAME    = "rmcph_gui"
APP_VERSION = "0.1.0"
