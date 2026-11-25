"""RMCProfile .rmc6f reader, supercell-to-unit-cell folding, and spglib symmetry ladder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import spglib
from ase import Atoms


@dataclass
class SymmetryCandidate:
    symprec: float
    number: int
    international: str
    n_prim_atoms: int
    n_ops: int


@dataclass
class FoldedSupercell:
    unit_cell: Atoms
    supercell_dim: tuple[int, int, int]
    n_super_atoms: int
    u_iso_per_site: np.ndarray  # <u^2>/3 in A^2 for each folded unit-cell site
    source_path: str


def read_rmc6f_folded(path: str | Path) -> FoldedSupercell:
    """Read a single RMCProfile .rmc6f supercell and fold it into the average unit cell.

    Uses the toroidal (circular) mean of fractional coordinates grouped by
    (element, ref_num) so periodic boundary crossings are handled seamlessly.
    """
    p = Path(path)
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()

    supercell_dim = None
    lattice_rows: list[list[float]] = []
    atoms_start = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Supercell dimensions:"):
            parts = line.split(":", 1)[1].split()
            supercell_dim = (int(parts[0]), int(parts[1]), int(parts[2]))
        elif line.startswith("Lattice vectors"):
            for k in range(1, 4):
                lattice_rows.append([float(x) for x in lines[i + k].split()[:3]])
            i += 3
        elif line.startswith("Atoms:"):
            atoms_start = i + 1
            break
        i += 1

    if supercell_dim is None or len(lattice_rows) != 3 or atoms_start is None:
        raise ValueError(f"Could not parse .rmc6f header in {path}")

    super_cell = np.asarray(lattice_rows, dtype=float)
    nx, ny, nz = supercell_dim
    dim_arr = np.array([nx, ny, nz], dtype=float)
    unit_lattice = super_cell / dim_arr[:, None]

    # Parse atom lines: idx  El  [charge]  fx  fy  fz  ref_num  ix  iy  iz
    # Using circular mean sin/cos accumulators per (element, ref_num)
    accum_cos: dict[tuple[str, int], np.ndarray] = {}
    accum_sin: dict[tuple[str, int], np.ndarray] = {}
    counts: dict[tuple[str, int], int] = {}
    order: list[tuple[str, int]] = []
    site_fracs: dict[tuple[str, int], list[list[float]]] = {}

    n_super_atoms = 0
    for line in lines[atoms_start:]:
        s = line.strip()
        if not s:
            continue
        toks = s.split()
        if len(toks) < 7:
            continue
        el = toks[1]
        # Count from right: ... fx fy fz ref_num ix iy iz
        # Works with [charge], numeric charge, or no charge column.
        fx, fy, fz = float(toks[-7]), float(toks[-6]), float(toks[-5])
        ref_num = int(toks[-4])

        # Fold fractional coords from supercell into unit cell [0, 1)
        uf = np.array([(fx * nx) % 1.0, (fy * ny) % 1.0, (fz * nz) % 1.0], dtype=float)
        key = (el, ref_num)
        if key not in counts:
            counts[key] = 0
            accum_cos[key] = np.zeros(3, dtype=float)
            accum_sin[key] = np.zeros(3, dtype=float)
            site_fracs[key] = []
            order.append(key)

        ang = 2.0 * np.pi * uf
        accum_cos[key] += np.cos(ang)
        accum_sin[key] += np.sin(ang)
        counts[key] += 1
        site_fracs[key].append(uf.tolist())
        n_super_atoms += 1

    # Compute mean fractional position and isotropic thermal displacement U_iso
    symbols: list[str] = []
    mean_fracs: list[np.ndarray] = []
    u_iso_list: list[float] = []

    for key in order:
        el, _ = key
        c = accum_cos[key] / counts[key]
        s = accum_sin[key] / counts[key]
        mean_f = (np.arctan2(s, c) / (2.0 * np.pi)) % 1.0
        symbols.append(el)
        mean_fracs.append(mean_f)

        # Minimum-image displacement from mean_f in Cartesian Angstroms
        arr = np.asarray(site_fracs[key], dtype=float)
        df = arr - mean_f[None, :]
        df -= np.round(df)
        dcart = df @ unit_lattice
        u_iso = float(np.mean(np.sum(dcart * dcart, axis=1)) / 3.0)
        u_iso_list.append(u_iso)

    unit_atoms = Atoms(
        symbols=symbols,
        scaled_positions=np.asarray(mean_fracs, dtype=float),
        cell=unit_lattice,
        pbc=True,
    )

    return FoldedSupercell(
        unit_cell=unit_atoms,
        supercell_dim=supercell_dim,
        n_super_atoms=n_super_atoms,
        u_iso_per_site=np.asarray(u_iso_list, dtype=float),
        source_path=str(path),
    )


def detect_symmetry_ladder(
    atoms: Atoms,
    tolerances: tuple[float, ...] = (1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2),
) -> list[SymmetryCandidate]:
    """Scan spglib across a tolerance ladder to reveal the crystal's symmetry hierarchy."""
    cell_tuple = (
        atoms.cell.array.copy(),
        atoms.get_scaled_positions().copy(),
        atoms.get_atomic_numbers().copy(),
    )
    results: list[SymmetryCandidate] = []
    for tol in tolerances:
        ds = spglib.get_symmetry_dataset(cell_tuple, symprec=tol)
        if ds is None:
            continue
        prim = spglib.standardize_cell(cell_tuple, to_primitive=True, no_idealize=False, symprec=tol)
        n_prim = len(prim[2]) if prim is not None else len(atoms)
        results.append(
            SymmetryCandidate(
                symprec=tol,
                number=int(ds.number),
                international=str(ds.international),
                n_prim_atoms=int(n_prim),
                n_ops=int(len(ds.rotations)),
            )
        )
    return results


def choose_best_symprec(
    ladder: list[SymmetryCandidate],
    max_auto_symprec: float = 0.1,
) -> SymmetryCandidate | None:
    """Pick the tightest tolerance that achieves the highest space-group symmetry <= max_auto_symprec."""
    eligible = [c for c in ladder if c.symprec <= max_auto_symprec]
    if not eligible:
        return ladder[-1] if ladder else None
    max_ops = max(c.n_ops for c in eligible)
    # Among candidates with the highest number of symmetry ops, pick the tightest tolerance
    best = [c for c in eligible if c.n_ops == max_ops]
    return min(best, key=lambda c: c.symprec)
