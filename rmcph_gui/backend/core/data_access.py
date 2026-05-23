"""Filesystem browsing + dataset inspection, wrapping src_gpu/Readers.

src_gpu uses implicit top-level imports (``import Readers``) that only resolve
when src_gpu is on sys.path. We add it lazily (inside the function that needs
it) so importing this module — and starting the server — never pays the jax
import cost. Folder browsing needs no jax at all.
"""
from __future__ import annotations

import re
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

def list_directory(path: Path, file_globs: list[str] | None = None) -> dict:
    """List immediate subdirectories of `path` (+ data-presence hints).

    If `file_globs` is given (e.g. ["*.rmc6f", "*.cif"]), also list matching
    files — used by the file picker for selecting an equilibrium structure.
    """
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

    files: list[str] = []
    if file_globs:
        names: set[str] = set()
        for g in file_globs:
            for f in path.glob(g):
                try:
                    if f.is_file():
                        names.add(f.name)
                except OSError:
                    continue
        files = sorted(names, key=str.lower)

    parent = path.parent
    return {
        "path": str(path),
        "parent": str(parent) if parent != path else None,
        "subdirs": subdirs,
        "files": files,
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


# A *.rmc6f is a structure-file candidate (atom types + lattice) unless it is an
# ensemble member: a numbered snapshot (<stem>_<N>.rmc6f) or a running-average
# dump (*AVERAGE.rmc6f). Filtering these keeps the structure-file picker from
# listing the hundreds of config files that live alongside the equilibrium one.
_RMC6F_NUMBERED_RE = re.compile(r"_\d+\.rmc6f$", re.IGNORECASE)
_RMC6F_AVERAGE_RE = re.compile(r"AVERAGE\.rmc6f$", re.IGNORECASE)


def _is_structure_rmc6f(name: str) -> bool:
    return (name.lower().endswith(".rmc6f")
            and not _RMC6F_NUMBERED_RE.search(name)
            and not _RMC6F_AVERAGE_RE.search(name))


def _find_rmc6f(path: Path) -> list[Path]:
    """Look for structure-file *.rmc6f in the folder, ancestors, parent's siblings.

    Covers the data/<T>K_ini/GTS_<T>K.rmc6f layout where the equilibrium file
    sits in a sibling of the ensemble folder. Numbered config snapshots and
    *AVERAGE.rmc6f are excluded — they are ensemble members, not structure files.
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
                if _is_structure_rmc6f(f.name) and f not in candidates:
                    candidates.append(f)
    return candidates


def inspect_folder(path_str: str, structure_file: str | None = None) -> dict:
    """Validate a chosen folder and parse atoms/cell from its .rmc6f.

    The .rmc6f is the *structure file*: it supplies atom elements + lattice,
    which the Frac*.txt files do not contain, so it is always required. It is
    distinct from the *displacement reference* (hsym), which defaults to the
    average of all configurations and is chosen separately by the user.

    Returns a JSON-serialisable dataset descriptor.
    """
    path = Path(path_str).resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    Readers = _readers()
    cfg_files, config_family = Readers.list_configs(str(path))
    configs_dir = str(Path(cfg_files[0]).parent) if cfg_files else None
    n_configs = len(cfg_files)
    candidates = _find_rmc6f(path)

    chosen = None
    if structure_file:
        sf = Path(structure_file).resolve()
        if sf.is_file():
            chosen = sf
            if sf not in candidates:
                candidates.insert(0, sf)
    if chosen is None and candidates:
        chosen = candidates[0]

    result: dict = {
        "path": str(path),
        "configs_dir": configs_dir,
        "n_configs": n_configs,
        # "frac" (Frac*.txt) or "rmc6f" (numbered .rmc6f ensemble) or "none".
        "config_family": config_family,
        "structure_candidates": [str(p) for p in candidates],
        "structure_file": str(chosen) if chosen else None,
        # Displacement reference (hsym): default = average of all configs.
        # mode "file" + a path overrides it (chosen via the file picker).
        "reference": {"mode": "average", "file": None},
        "atoms": None,
        "cell": None,
        "dim": None,
        "warnings": [],
    }
    if not cfg_files:
        result["warnings"].append(
            "No Frac*.txt or numbered .rmc6f configurations found here or in a "
            "configs/ subfolder."
        )
    if chosen is None:
        result["warnings"].append(
            "No .rmc6f structure file found nearby — needed for atom types + lattice."
        )
        return result

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
