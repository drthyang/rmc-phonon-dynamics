"""S(Q) / G(r) fit-quality API.

Serves per-config X-ray F(Q) (XFQ1.csv), X-ray PDF G(r) (FT_XFQ1.csv),
and G(r) partial pair correlations (PDFpartials.csv) from an RMC ensemble.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from ..config import BROWSE_ROOT

router = APIRouter()

# Matches e.g. "GTS_5K_100_XFQ1.csv" → stem="GTS_5K", config=100
_XFQ1_RE = re.compile(r"^(.+)_(\d+)_XFQ1\.csv$", re.IGNORECASE)


def _safe_dir(folder: str) -> Path:
    p = Path(folder).resolve()
    try:
        p.relative_to(BROWSE_ROOT.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Path outside allowed root")
    if not p.is_dir():
        raise HTTPException(status_code=404, detail="Folder not found")
    return p


def _read_csv(path: Path) -> dict[str, list[float]]:
    """Return {stripped_header: [float, ...]} for each column."""
    with path.open() as f:
        reader = csv.reader(f)
        headers = [h.strip() for h in next(reader)]
        cols: dict[str, list[float]] = {h: [] for h in headers}
        for row in reader:
            for h, v in zip(headers, row):
                try:
                    cols[h].append(float(v.strip()))
                except (ValueError, AttributeError):
                    pass
    return cols


@router.get("/sqgr/configs")
def list_sqgr_configs(folder: str = Query(...)):
    """List config numbers that have XFQ1 data in the given folder."""
    p = _safe_dir(folder)
    configs: list[int] = []
    for f in p.glob("*_XFQ1.csv"):
        if "FT_XFQ1" in f.name:
            continue
        m = _XFQ1_RE.match(f.name)
        if m:
            configs.append(int(m.group(2)))
    configs.sort()
    return {"configs": configs, "count": len(configs)}


@router.get("/sqgr/data")
def get_sqgr_data(folder: str = Query(...), config: int = Query(...)):
    """Return X-ray F(Q), X-ray PDF G(r), and G(r) partials for one config."""
    p = _safe_dir(folder)

    # Discover file stem from the XFQ1 file for this config
    candidates = [f for f in p.glob(f"*_{config}_XFQ1.csv") if "FT_XFQ1" not in f.name]
    if not candidates:
        raise HTTPException(status_code=404, detail=f"No XFQ1 data for config {config}")
    stem = _XFQ1_RE.match(candidates[0].name).group(1)

    result: dict = {}

    # Panel 1: X-ray F(Q) — columns: Q, F(Q)_RMC, F(Q)_Expt
    xfq_path = p / f"{stem}_{config}_XFQ1.csv"
    if xfq_path.exists():
        cols = _read_csv(xfq_path)
        keys = list(cols)
        result["xfq"] = {"q": cols[keys[0]], "rmc": cols[keys[1]], "expt": cols[keys[2]]}

    # Panel 2: X-ray PDF G(r) — columns: r(A), X_ray-calc, X_ray_exp_renorm
    xpdf_path = p / f"{stem}_{config}_FT_XFQ1.csv"
    if xpdf_path.exists():
        cols = _read_csv(xpdf_path)
        keys = list(cols)
        result["xpdf"] = {"r": cols[keys[0]], "rmc": cols[keys[1]], "expt": cols[keys[2]]}

    # Panel 3: G(r) partial pairs — columns: r(Ang), Ga-Ga, Ga-Ta, ...
    partials_path = p / f"{stem}_{config}_PDFpartials.csv"
    if partials_path.exists():
        cols = _read_csv(partials_path)
        keys = list(cols)
        result["partials"] = {
            "r": cols[keys[0]],
            "pairs": {k: cols[k] for k in keys[1:]},
        }

    return result
