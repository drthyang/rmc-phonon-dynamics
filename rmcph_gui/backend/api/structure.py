"""Structure API: unit-cell geometry for the 3D view (Phase 2)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..core import session, structure

router = APIRouter()


@router.get("/structure")
def get_structure():
    ds = session.get_dataset()
    if not ds:
        raise HTTPException(status_code=409, detail="No dataset open. Open a folder first.")
    if not ds.get("structure_file") or not ds.get("cell") or not ds.get("dim"):
        raise HTTPException(status_code=409, detail="Dataset missing structure file / cell / dim.")
    try:
        return structure.build_unit_cell(ds["structure_file"], ds["cell"], ds["dim"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
