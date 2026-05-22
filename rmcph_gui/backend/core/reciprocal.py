"""Reciprocal cell, Brillouin zone, and high-symmetry points (Phase 3a).

Convention note (important): src_gpu computes the Bloch phase as
``cell_idx · kvec`` where kvec is fractional w.r.t. the UNIT cell's reciprocal
lattice — the conventional cell that is tiled in the supercell. So we work
entirely in the *given* unit cell's reciprocal basis (NOT a primitive
standardization). For a conventional cubic cell that yields the simple-cubic
BZ (a cube) with points Γ, X, M, R — matching the hand-written k-paths in
src_gpu/test_run.py.

We use spglib (via pymatgen) only to *classify* the crystal family, then return
the standard P-lattice high-symmetry points for that family in the unit cell's
reciprocal fractional basis, plus the BZ geometry for 3D rendering.

Cartesian coordinates (high-sym points + BZ facets) all use pymatgen's
``reciprocal_lattice`` (the 2π convention) so they share one scale in the view.
The fractional coordinates are what the calculation consumes and are
2π-independent.
"""
from __future__ import annotations

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

_T = 1.0 / 3.0

# High-symmetry points for the simple (P) Bravais lattice of each crystal
# family, in conventional reciprocal fractional coords, with a default path.
# 'GM' = Γ (kept as in test_run.py; the frontend renders it as Γ).
HS_TABLE: dict[str, dict] = {
    "cubic": {
        "points": {"GM": [0, 0, 0], "X": [0, 0.5, 0], "M": [0.5, 0.5, 0], "R": [0.5, 0.5, 0.5]},
        "path": [["GM", "X"], ["X", "M"], ["M", "GM"], ["GM", "R"], ["R", "X"], ["M", "R"]],
    },
    "tetragonal": {
        "points": {"GM": [0, 0, 0], "X": [0, 0.5, 0], "M": [0.5, 0.5, 0],
                   "Z": [0, 0, 0.5], "R": [0, 0.5, 0.5], "A": [0.5, 0.5, 0.5]},
        "path": [["GM", "X"], ["X", "M"], ["M", "GM"], ["GM", "Z"], ["Z", "R"],
                 ["R", "A"], ["A", "Z"], ["X", "R"], ["M", "A"]],
    },
    "orthorhombic": {
        "points": {"GM": [0, 0, 0], "X": [0.5, 0, 0], "Y": [0, 0.5, 0], "Z": [0, 0, 0.5],
                   "S": [0.5, 0.5, 0], "U": [0.5, 0, 0.5], "T": [0, 0.5, 0.5], "R": [0.5, 0.5, 0.5]},
        "path": [["GM", "X"], ["X", "S"], ["S", "Y"], ["Y", "GM"], ["GM", "Z"],
                 ["Z", "U"], ["U", "R"], ["R", "T"], ["T", "Z"], ["Y", "T"], ["U", "X"], ["S", "R"]],
    },
    "hexagonal": {
        "points": {"GM": [0, 0, 0], "M": [0.5, 0, 0], "K": [_T, _T, 0],
                   "A": [0, 0, 0.5], "L": [0.5, 0, 0.5], "H": [_T, _T, 0.5]},
        "path": [["GM", "M"], ["M", "K"], ["K", "GM"], ["GM", "A"], ["A", "L"],
                 ["L", "H"], ["H", "A"], ["L", "M"], ["K", "H"]],
    },
}
# trigonal / monoclinic / triclinic: Γ only (user adds custom points in 3c).
_FALLBACK = {"points": {"GM": [0, 0, 0]}, "path": []}


def _classify(lattice: Lattice, atoms: list) -> dict:
    info = {"spacegroup": "?", "spacegroup_number": 0,
            "crystal_system": "triclinic", "lattice_type": "?"}
    try:
        st = Structure(lattice, [a["symbol"] for a in atoms], [a["frac"] for a in atoms])
        for sp in (1e-3, 1e-2, 5e-2):
            try:
                sga = SpacegroupAnalyzer(st, symprec=sp)
                info.update(
                    spacegroup=sga.get_space_group_symbol(),
                    spacegroup_number=int(sga.get_space_group_number()),
                    crystal_system=sga.get_crystal_system(),
                    lattice_type=sga.get_lattice_type(),
                )
                break
            except Exception:
                continue
    except Exception:
        pass
    return info


def build_reciprocal(lattice, atoms) -> dict:
    L = Lattice(lattice)
    info = _classify(L, atoms)

    # 2π reciprocal — shared scale for BZ facets + high-sym cartesian.
    R = np.array(L.reciprocal_lattice.matrix)

    bz = L.get_brillouin_zone()  # list of facets, each a list of 3-vectors
    bz_facets = [[[float(c) for c in v] for v in facet] for facet in bz]

    table = HS_TABLE.get(info["crystal_system"], _FALLBACK)
    high_sym = []
    for label, frac in table["points"].items():
        cart = (np.array(frac, dtype=float) @ R).tolist()
        high_sym.append({
            "label": label,
            "frac": [float(x) for x in frac],
            "cart": [float(x) for x in cart],
        })

    return {
        "spacegroup": info["spacegroup"],
        "spacegroup_number": info["spacegroup_number"],
        "crystal_system": info["crystal_system"],
        "lattice_type": info["lattice_type"],
        "recip_lattice": [[float(x) for x in row] for row in R],
        "bz_facets": bz_facets,
        "high_sym_points": high_sym,
        "suggested_path": table["path"],
    }
