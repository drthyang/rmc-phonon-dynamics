"""Reciprocal-cell API: BZ + high-symmetry points for the k-path UI (Phase 3a)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..core import reciprocal, session, structure

router = APIRouter()


@router.get("/reciprocal")
def get_reciprocal():
    ds = session.get_dataset()
    if not ds:
        raise HTTPException(status_code=409, detail="No dataset open. Open a folder first.")
    if not ds.get("structure_file") or not ds.get("cell") or not ds.get("dim"):
        raise HTTPException(status_code=409, detail="Dataset missing structure file / cell / dim.")
    try:
        uc = structure.build_unit_cell(ds["structure_file"], ds["cell"], ds["dim"])
        return reciprocal.build_reciprocal(uc["lattice"], uc["atoms"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
