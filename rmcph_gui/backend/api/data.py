"""Data-folder API: server-side directory browser + dataset inspection."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from ..config import BROWSE_ROOT, DATA_BROWSE_START
from ..core import data_access, session
from ..models.data import OpenFolderRequest

router = APIRouter()


def _clamp(path: Path) -> Path:
    """Keep navigation within BROWSE_ROOT; otherwise fall back to the start dir."""
    path = path.resolve()
    try:
        path.relative_to(BROWSE_ROOT.resolve())
    except ValueError:
        return DATA_BROWSE_START.resolve()
    return path


@router.get("/data/browse")
def browse(
    path: str | None = Query(default=None),
    files: str | None = Query(default=None),   # comma-separated exts, e.g. "rmc6f,cif"
):
    target = _clamp(Path(path)) if path else DATA_BROWSE_START.resolve()
    globs = None
    if files:
        globs = [f"*.{ext.strip().lstrip('.')}" for ext in files.split(",") if ext.strip()]
    try:
        return data_access.list_directory(target, file_globs=globs)
    except (NotADirectoryError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/data/open")
def open_folder(req: OpenFolderRequest):
    try:
        result = data_access.inspect_folder(req.path, structure_file=req.structure_file)
    except (NotADirectoryError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:  # parsing errors → 422 with the message
        raise HTTPException(status_code=422, detail=f"{type(e).__name__}: {e}")
    session.set_dataset(result)
    return result
