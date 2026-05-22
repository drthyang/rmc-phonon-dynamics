"""Filesystem browsing + dataset inspection, wrapping src_gpu/Readers.

src_gpu uses implicit top-level imports (``import Readers``) that only resolve
when src_gpu is on sys.path. We add it lazily (inside the function that needs
it) so importing this module — and starting the server — never pays the jax
import cost. Folder browsing needs no jax at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

from .. import config


def _readers():
    """Import src_gpu/Readers on first use (triggers the jax import)."""
    src = str(config.SRC_GPU_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)
    import Readers  # noqa: E402  — provided by src_gpu
    return Readers


# ── Directory browser ─────────────────────────────────────────────────────────

def list_directory(path: Path) -> dict:
    """List immediate subdirectories of `path` (+ data-presence hints)."""
    path = Path(path).resolve()
    if not path.is_dir():
        raise NotADirectoryError(str(path))

    subdirs: list[str] = []
    try:
        for p in sorted(path.iterdir(), key=lambda x: x.name.lower()):
            try:
                if p.is_dir() and not p.name.startswith("."):
                    subdirs.append(p.name)
            except OSError:
                continue
    except PermissionError:
        pass

    parent = path.parent
    return {
        "path": str(path),
        "parent": str(parent) if parent != path else None,
        "subdirs": subdirs,
        "has_frac": _has_frac(path),
        "has_rmc6f": bool(list(path.glob("*.rmc6f"))),
    }


# ── Dataset inspection ──────────────────────────────────────────────────────────

def _has_frac(path: Path) -> bool:
    return any(path.glob("Frac*.txt"))


def _find_frac_dir(path: Path) -> Path | None:
    """Frac*.txt either live directly in `path` or in a `configs/` subfolder."""
    if _has_frac(path):
        return path
    cfg = path / "configs"
    if cfg.is_dir() and _has_frac(cfg):
        return cfg
    return None


def _find_rmc6f(path: Path) -> list[Path]:
    """Look for *.rmc6f in the folder, its ancestors, and parent's siblings.

    Covers the data/<T>K_ini/GTS_<T>K.rmc6f layout where the equilibrium file
    sits in a sibling of the ensemble folder.
    """
    candidates: list[Path] = []
    search: list[Path] = [path, path.parent, path.parent.parent]
    grandparent = path.parent.parent
    if grandparent.is_dir():
        try:
            search += [d for d in grandparent.iterdir() if d.is_dir()]
        except (PermissionError, OSError):
            pass
    seen: set[Path] = set()
    for d in search:
        if d and d not in seen and d.is_dir():
            seen.add(d)
            for f in sorted(d.glob("*.rmc6f")):
                if f not in candidates:
                    candidates.append(f)
    return candidates


def inspect_folder(path_str: str, eq_file: str | None = None) -> dict:
    """Validate a chosen folder and parse atoms/cell from its .rmc6f.

    Returns a JSON-serialisable dataset descriptor.
    """
    path = Path(path_str).resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    frac_dir = _find_frac_dir(path)
    n_configs = len(list(frac_dir.glob("Frac*.txt"))) if frac_dir else 0
    candidates = _find_rmc6f(path)

    chosen = None
    if eq_file:
        ef = Path(eq_file).resolve()
        if ef.is_file():
            chosen = ef
            if ef not in candidates:
                candidates.insert(0, ef)
    if chosen is None and candidates:
        chosen = candidates[0]

    result: dict = {
        "path": str(path),
        "configs_dir": str(frac_dir) if frac_dir else None,
        "n_configs": n_configs,
        "rmc6f_candidates": [str(p) for p in candidates],
        "eq_file": str(chosen) if chosen else None,
        "atoms": None,
        "cell": None,
        "dim": None,
        "warnings": [],
    }
    if not frac_dir:
        result["warnings"].append(
            "No Frac*.txt files found here or in a configs/ subfolder."
        )
    if chosen is None:
        result["warnings"].append(
            "No .rmc6f equilibrium file found nearby — cannot parse atoms/cell."
        )
        return result

    Readers = _readers()
    atom_dic = Readers.get_atom_idx(str(chosen), verbose=0)
    v1, v2, v3, dim = Readers.read_cell_vec(str(chosen), verbose=0)

    result["atoms"] = [
        {"symbol": el, "count": len(ids)} for el, ids in atom_dic.items()
    ]
    result["cell"] = [
        [float(x) for x in v1],
        [float(x) for x in v2],
        [float(x) for x in v3],
    ]
    result["dim"] = [int(x) for x in dim]
    result["natom"] = sum(len(ids) for ids in atom_dic.values())
    return result
