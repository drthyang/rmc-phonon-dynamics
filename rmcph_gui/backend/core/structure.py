"""Build the unit cell for the 3D structure view.

The .rmc6f holds a *supercell* (e.g. 8×8×8). We extract the unit cell as the
atoms in cell index (0,0,0), with:
    unit-cell lattice  = supercell vectors / dim
    unit-cell frac     = supercell frac * dim   (mod 1)

This reads positions directly from the .rmc6f Atoms section (src_gpu/Readers
only parses the element↔ID map and the lattice, not per-atom positions).
"""
from __future__ import annotations

from pathlib import Path

# Cordero et al. (2008) covalent radii [Å], used for bond detection.
COVALENT_RADII = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39,
    "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20,
    "As": 1.19, "Se": 1.20, "Br": 1.20, "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75,
    "Nb": 1.64, "Mo": 1.54, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44,
    "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Cs": 2.44, "Ba": 2.15,
    "La": 2.07, "Hf": 1.75, "Ta": 1.70, "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41,
    "Pt": 1.36, "Au": 1.36, "Hg": 1.32, "Tl": 1.45, "Pb": 1.46, "Bi": 1.48,
}
DEFAULT_RCOV = 1.50
BOND_TOL = 1.20   # bond if dist <= (rcov_i + rcov_j) * BOND_TOL


def _read_rmc6f_atoms(path: Path):
    """Yield (element, [x,y,z] frac-in-supercell, [nx,ny,nz]) for each atom.

    Mirrors the column convention used by src_gpu Readers.get_atom_idx:
      <idx> <element> [..] <x> <y> <z> <RN> <nx> <ny> <nz>
    i.e. element = parts[1]; the last 7 tokens are x y z RN nx ny nz.
    """
    in_atoms = False
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if not in_atoms:
                if s.startswith("Atoms:"):
                    in_atoms = True
                continue
            parts = s.split()
            if len(parts) < 8:
                continue
            try:
                x, y, z = (float(parts[-7]), float(parts[-6]), float(parts[-5]))
                nx, ny, nz = (int(parts[-3]), int(parts[-2]), int(parts[-1]))
            except ValueError:
                continue
            element = parts[1]
            yield element, [x, y, z], [nx, ny, nz]


def build_unit_cell(structure_file: str, cell_vectors, dim) -> dict:
    """Return {lattice, atoms, natom, source} for the unit cell.

    cell_vectors : 3×3 supercell lattice (rows = a,b,c in Å)
    dim          : [d0, d1, d2] supercell repeats
    """
    path = Path(structure_file)
    d = [int(x) for x in dim]
    if any(v <= 0 for v in d):
        d = [1, 1, 1]

    # Unit-cell lattice = supercell row i / dim[i]
    lattice = [[cell_vectors[i][j] / d[i] for j in range(3)] for i in range(3)]

    atoms = []
    for element, frac, cell in _read_rmc6f_atoms(path):
        if cell == [0, 0, 0]:
            frac_uc = [(frac[i] * d[i]) % 1.0 for i in range(3)]
            atoms.append({"symbol": element, "frac": frac_uc})

    bonds = _compute_bonds(lattice, atoms)

    return {
        "lattice": lattice,
        "atoms": atoms,
        "bonds": bonds,
        "natom": len(atoms),
        "nbond": len(bonds),
        "source": "rmc6f",
    }


def _compute_bonds(lattice, atoms, tol: float = BOND_TOL) -> list:
    """Bonds within (rcov_i + rcov_j)·tol, including periodic images.

    Each home-cell atom is connected to every qualifying neighbour (some in
    adjacent cells, i.e. fractional coords outside [0,1) — drawn as short
    stubs at the cell boundary). Interior bonds are emitted from both ends;
    that double-draw is harmless and keeps edge bonds symmetric.

    Returns [{"a": [fa,fb,fc], "b": [fa,fb,fc]}] in fractional coords.
    """
    n = len(atoms)
    if n == 0:
        return []

    def cart(f):
        return (
            f[0]*lattice[0][0] + f[1]*lattice[1][0] + f[2]*lattice[2][0],
            f[0]*lattice[0][1] + f[1]*lattice[1][1] + f[2]*lattice[2][1],
            f[0]*lattice[0][2] + f[1]*lattice[1][2] + f[2]*lattice[2][2],
        )

    carts = [cart(a["frac"]) for a in atoms]
    rcov = [COVALENT_RADII.get(a["symbol"], DEFAULT_RCOV) for a in atoms]
    shifts = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]

    bonds = []
    for i in range(n):
        ci = carts[i]
        for j in range(n):
            cut = (rcov[i] + rcov[j]) * tol
            cut2 = cut * cut
            fj = atoms[j]["frac"]
            for s in shifts:
                if i == j and s == (0, 0, 0):
                    continue
                nf = [fj[0] + s[0], fj[1] + s[1], fj[2] + s[2]]
                nc = cart(nf)
                d2 = (ci[0]-nc[0])**2 + (ci[1]-nc[1])**2 + (ci[2]-nc[2])**2
                if 0.0 < d2 <= cut2:
                    bonds.append({"a": list(atoms[i]["frac"]), "b": nf})
    return bonds
