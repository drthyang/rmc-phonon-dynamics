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

    return {
        "lattice": lattice,
        "atoms": atoms,
        "natom": len(atoms),
        "source": "rmc6f",
    }
