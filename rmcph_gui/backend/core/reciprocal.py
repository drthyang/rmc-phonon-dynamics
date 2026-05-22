"""Reciprocal cell, Brillouin zone & high-symmetry points via seekpath.

We use seekpath (Hinuma et al.) for standardized, publication-grade band paths
and high-symmetry labels for ANY Bravais lattice. seekpath works in the
standardized PRIMITIVE cell, so each high-symmetry point carries TWO sets of
coordinates:

  - ``frac`` (primitive reciprocal basis)  — used for the 3D BZ display, which
    renders the primitive Brillouin zone.
  - ``frac_conv`` (conventional reciprocal basis) — used to FEED src_gpu, whose
    Bloch phase ``cell_idx · kvec`` indexes the conventional cell tiled in the
    supercell.

Transform: k_conv = k_prim · (B_prim · B_conv⁻¹), with B_* the reciprocal
lattices of seekpath's primitive and conventional cells (same frame, so
rotation-safe). Validated to preserve the cartesian k-vector exactly.

NOTE (validation pending): feeding the transformed conventional coords through
src_gpu's no-2π convention must be checked against a known-good run before the
results are trusted — see Phase 4/5.
"""
from __future__ import annotations

import numpy as np
from pymatgen.core import Element, Lattice


def _bz_facets(lattice: Lattice):
    return [[[float(c) for c in v] for v in facet]
            for facet in lattice.get_brillouin_zone()]


def build_reciprocal(lattice, atoms) -> dict:
    import seekpath  # local import; only needed here

    A = np.array(lattice, dtype=float)
    frac = np.array([a["frac"] for a in atoms], dtype=float)
    numbers = [Element(a["symbol"]).Z for a in atoms]
    cell = (A, frac, numbers)

    res = seekpath.get_path(cell, with_time_reversal=True, symprec=1e-2)

    A_prim = np.array(res["primitive_lattice"])
    A_conv = np.array(res["conv_lattice"])
    B_prim = Lattice(A_prim).reciprocal_lattice.matrix     # 2π, primitive frame
    B_conv = Lattice(A_conv).reciprocal_lattice.matrix
    T = B_prim @ np.linalg.inv(B_conv)                     # k_conv = k_prim @ T

    high_sym = []
    for label, kp in res["point_coords"].items():
        kp = np.array(kp, dtype=float)
        cart = kp @ B_prim
        kconv = kp @ T
        high_sym.append({
            "label": label,
            "frac": [float(x) for x in kp],          # primitive (display)
            "frac_conv": [float(x) for x in kconv],  # conventional (calc)
            "cart": [float(x) for x in cart],        # primitive cartesian (render)
        })

    suggested_path = [[a, b] for (a, b) in res["path"]]

    return {
        "engine": "seekpath",
        "spacegroup": res.get("spacegroup_international", "?"),
        "spacegroup_number": int(res.get("spacegroup_number", 0)),
        "crystal_system": res.get("bravais_lattice", "?"),
        "lattice_type": res.get("bravais_lattice_extended", "?"),
        "recip_lattice": [[float(x) for x in row] for row in B_prim],
        "bz_facets": _bz_facets(Lattice(A_prim)),
        "high_sym_points": high_sym,
        "suggested_path": suggested_path,
        "basis_note": "frac = primitive reciprocal (display); frac_conv = conventional reciprocal (calc)",
    }
