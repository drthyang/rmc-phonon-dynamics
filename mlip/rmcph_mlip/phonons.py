"""Harmonic phonon calculation workflow using ASE + spglib + Phonopy + MLIP.

Supports single RMCProfile .rmc6f supercell configurations (folded to the
average unit cell with automatic symmetry-ladder detection) as well as .cif
and other ASE-readable crystal structure files.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import seekpath
import spglib
from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.io import read as ase_read
from ase.optimize import BFGS
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from .neutron import (
    NeutronDosResult,
    PowderInsResult,
    compute_neutron_dos,
    compute_powder_ins,
)
from .rmc6f import (
    SymmetryCandidate,
    choose_best_symprec,
    detect_symmetry_ladder,
    read_rmc6f_folded,
)


@dataclass
class PhononResult:
    """Container for a completed MLIP phonon + neutron DOS/INS calculation."""

    phonopy: Phonopy
    primitive_atoms: Atoms
    spacegroup: str
    symmetry_ladder: list[SymmetryCandidate]
    chosen_symprec: float
    qpoints: np.ndarray
    highsym_qpts: list[tuple[int, str]]
    segment_sizes: list[int]
    bands_thz: np.ndarray
    neutron_dos: NeutronDosResult
    powder_ins: PowderInsResult
    metadata: dict


def atoms_to_spglib(atoms: Atoms):
    return (
        atoms.cell.array.copy(),
        atoms.get_scaled_positions().copy(),
        atoms.get_atomic_numbers().copy(),
    )


def load_unit_cell(structure_path: str | Path) -> tuple[Atoms, dict]:
    """Load a crystal structure from .rmc6f (single supercell folded to unit cell) or .cif/etc."""
    p = Path(structure_path)
    if p.suffix.lower() == ".rmc6f":
        folded = read_rmc6f_folded(p)
        print(
            f"[mlip] Read single .rmc6f supercell {p.name}: "
            f"{folded.n_super_atoms} atoms ({'×'.join(map(str, folded.supercell_dim))}) "
            f"→ folded unit cell: {folded.unit_cell.get_chemical_formula()} "
            f"({len(folded.unit_cell)} atoms), mean U_iso={folded.u_iso_per_site.mean():.4f} Å²"
        )
        return folded.unit_cell, {
            "input_type": "rmc6f_single_supercell",
            "rmc_supercell_dim": list(folded.supercell_dim),
            "rmc_n_super_atoms": folded.n_super_atoms,
            "rmc_mean_u_iso_A2": round(float(folded.u_iso_per_site.mean()), 5),
        }

    raw_atoms = ase_read(str(p))
    print(f"[mlip] Read {p.name}: {raw_atoms.get_chemical_formula()} ({len(raw_atoms)} atoms)")
    return raw_atoms, {"input_type": p.suffix.lstrip(".").lower() or "ase"}


def standardize_primitive(atoms: Atoms, symprec: float) -> tuple[Atoms, str]:
    """Detect space group and reduce to the standardized primitive cell."""
    cell_tuple = atoms_to_spglib(atoms)
    dataset = spglib.get_symmetry_dataset(cell_tuple, symprec=symprec)
    if dataset is None:
        return atoms.copy(), "P1 (1)"

    sg_symbol = f"{dataset.international} ({dataset.number})"
    prim = spglib.standardize_cell(cell_tuple, to_primitive=True, no_idealize=False, symprec=symprec)
    if prim is None:
        return atoms.copy(), sg_symbol

    lattice, scaled_pos, numbers = prim
    prim_atoms = Atoms(
        numbers=numbers,
        scaled_positions=scaled_pos,
        cell=lattice,
        pbc=True,
    )
    return prim_atoms, sg_symbol


def relax_structure(
    atoms: Atoms,
    calculator,
    fmax: float = 1e-3,
    max_steps: int = 200,
    relax_cell: bool = False,
) -> tuple[Atoms, float, int]:
    """Relax atomic positions (and optionally lattice vectors) with BFGS."""
    work = atoms.copy()
    work.calc = calculator
    target = FrechetCellFilter(work) if relax_cell else work
    opt = BFGS(target, logfile=None)
    opt.run(fmax=fmax, steps=max_steps)
    forces = work.get_forces()
    final_fmax = float(np.linalg.norm(forces, axis=1).max())
    return work, final_fmax, opt.nsteps


def build_kpath(prim_atoms: Atoms, npoints_per_segment: int = 51):
    """Build a standard crystallographic high-symmetry k-path via seekpath."""
    res = seekpath.get_path(atoms_to_spglib(prim_atoms))
    point_coords = res["point_coords"]
    path = res["path"]

    qpoints: list[list[float]] = []
    highsym: list[tuple[int, str]] = []
    seg_sizes: list[int] = []

    for seg_idx, (start_lbl, end_lbl) in enumerate(path):
        p0 = np.asarray(point_coords[start_lbl], dtype=float)
        p1 = np.asarray(point_coords[end_lbl], dtype=float)
        pts = np.linspace(p0, p1, npoints_per_segment)

        start_i = len(qpoints)
        if seg_idx > 0 and path[seg_idx - 1][1] == start_lbl:
            pts = pts[1:]
            seg_sizes.append(len(pts))
            for pt in pts:
                qpoints.append(pt.tolist())
            highsym.append((len(qpoints) - 1, _clean_label(end_lbl)))
        else:
            highsym.append((start_i, _clean_label(start_lbl)))
            seg_sizes.append(len(pts))
            for pt in pts:
                qpoints.append(pt.tolist())
            highsym.append((len(qpoints) - 1, _clean_label(end_lbl)))

    return np.asarray(qpoints, dtype=float), highsym, seg_sizes, res


def _clean_label(label: str) -> str:
    return "Γ" if label.upper() in ("GAMMA", "G") else label


def compute_harmonic_phonons(
    structure_path: str | Path,
    calculator,
    supercell: Sequence[int] = (2, 2, 2),
    symprec: float | None = None,
    displacement: float = 0.01,
    relax: bool = True,
    relax_cell: bool = False,
    fmax: float = 1e-3,
    npoints: int = 41,
    dos_mesh: Sequence[int] = (15, 15, 15),
    temperature_k: float = 5.0,
    fwhm_mev: float = 0.8,
    q_max: float = 8.0,
    ei_mev: float | None = 50.0,
    model_name: str = "mace-mp-medium",
) -> PhononResult:
    """Run end-to-end harmonic phonon + neutron-weighted DOS + INS calculation."""
    t0 = time.perf_counter()
    unit_atoms, input_info = load_unit_cell(structure_path)

    # Scan symmetry tolerance ladder
    ladder = detect_symmetry_ladder(unit_atoms)
    print("[mlip] Symmetry tolerance ladder:")
    for cand in ladder:
        print(
            f"  symprec={cand.symprec:6.3f} Å → {cand.international:10s} (#{cand.number:3d}), "
            f"ops={cand.n_ops:3d}, n_prim={cand.n_prim_atoms}"
        )

    if symprec is None:
        best = choose_best_symprec(ladder, max_auto_symprec=0.1)
        chosen_symprec = best.symprec if best is not None else 0.05
    else:
        chosen_symprec = float(symprec)

    prim_atoms, sg_symbol = standardize_primitive(unit_atoms, symprec=chosen_symprec)
    print(f"[mlip] Selected symprec={chosen_symprec} Å → {sg_symbol} ({len(prim_atoms)} atoms in primitive cell)")

    # Align primitive cell to seekpath's standardized orientation
    sp_out = seekpath.get_path(atoms_to_spglib(prim_atoms))
    prim_atoms = Atoms(
        numbers=sp_out["primitive_types"],
        scaled_positions=sp_out["primitive_positions"],
        cell=sp_out["primitive_lattice"],
        pbc=True,
    )

    relax_info = {"performed": False}
    if relax:
        t_rel = time.perf_counter()
        prim_atoms, final_fmax, nsteps = relax_structure(
            prim_atoms,
            calculator,
            fmax=fmax,
            relax_cell=relax_cell,
        )
        dt_rel = time.perf_counter() - t_rel
        print(
            f"[mlip] Relaxed ({'cell+pos' if relax_cell else 'pos-only'}) in {nsteps} steps "
            f"({dt_rel:.1f}s), fmax={final_fmax:.2e} eV/Å"
        )
        relax_info = {
            "performed": True,
            "relax_cell": relax_cell,
            "steps": int(nsteps),
            "final_fmax_eV_per_A": final_fmax,
        }

    # Build Phonopy on the exact seekpath primitive cell (primitive_matrix = I)
    unitcell = PhonopyAtoms(
        symbols=prim_atoms.get_chemical_symbols(),
        cell=prim_atoms.cell.array,
        scaled_positions=prim_atoms.get_scaled_positions(),
    )
    sc_matrix = np.diag([int(x) for x in supercell])
    ph = Phonopy(unitcell, supercell_matrix=sc_matrix, primitive_matrix=np.eye(3), symprec=1e-3)
    ph.generate_displacements(distance=displacement)

    supercells = ph.supercells_with_displacements
    print(
        f"[mlip] Evaluating MLIP forces on {len(supercells)} displaced supercell(s) "
        f"({len(supercells[0])} atoms each, supercell={list(supercell)})..."
    )

    t_fc = time.perf_counter()
    forces_list = []
    for idx, sc in enumerate(supercells):
        sc_atoms = Atoms(
            symbols=sc.symbols,
            cell=sc.cell,
            scaled_positions=sc.scaled_positions,
            pbc=True,
        )
        sc_atoms.calc = calculator
        forces = sc_atoms.get_forces()
        forces = forces - forces.mean(axis=0, keepdims=True)
        forces_list.append(forces)
        print(f"  displacement {idx + 1}/{len(supercells)}: max |F| = {np.linalg.norm(forces, axis=1).max():.4f} eV/Å")

    ph.forces = forces_list
    ph.produce_force_constants()
    ph.symmetrize_force_constants()
    dt_fc = time.perf_counter() - t_fc
    print(f"[mlip] Force constants computed & symmetrized in {dt_fc:.1f}s")

    # Band structure along standardized k-path
    qpoints, highsym, seg_sizes, _ = build_kpath(prim_atoms, npoints_per_segment=npoints)
    ph.run_band_structure([qpoints.tolist()], with_eigenvectors=True)
    band_dict = ph.get_band_structure_dict()
    bands_thz = np.asarray(band_dict["frequencies"][0], dtype=float)

    # Neutron-weighted DOS (GDOS) + simulated powder INS S(|Q|, E)
    t_ins = time.perf_counter()
    mesh_tuple = (int(dos_mesh[0]), int(dos_mesh[1]), int(dos_mesh[2]))
    ndos = compute_neutron_dos(ph, mesh=mesh_tuple, fwhm_mev=fwhm_mev)
    ins = compute_powder_ins(
        ph,
        ndos,
        temperature_k=temperature_k,
        q_max=q_max,
        ei_mev=ei_mev,
    )
    dt_ins = time.perf_counter() - t_ins
    weights_str = ", ".join(f"{el}:{w:.3f}" for el, w in ndos.element_weights.items())
    print(
        f"[mlip] Neutron-weighted DOS & INS S(|Q|,E) computed in {dt_ins:.1f}s "
        f"(T={temperature_k} K, FWHM={fwhm_mev} meV, σ/M [barn/amu]: {weights_str})"
    )

    total_time = time.perf_counter() - t0
    print(
        f"[mlip] Done in {total_time:.1f}s — freq range: "
        f"{bands_thz.min():.2f} .. {bands_thz.max():.2f} THz "
        f"({bands_thz.min() * 4.1356677:.2f} .. {bands_thz.max() * 4.1356677:.2f} meV)"
    )

    clean_model = Path(model_name).stem if str(model_name).endswith(".model") else str(model_name)

    return PhononResult(
        phonopy=ph,
        primitive_atoms=prim_atoms,
        spacegroup=sg_symbol,
        symmetry_ladder=ladder,
        chosen_symprec=chosen_symprec,
        qpoints=qpoints,
        highsym_qpts=highsym,
        segment_sizes=seg_sizes,
        bands_thz=bands_thz,
        neutron_dos=ndos,
        powder_ins=ins,
        metadata={
            "source_structure": str(structure_path),
            **input_info,
            "formula": prim_atoms.get_chemical_formula(),
            "spacegroup": sg_symbol,
            "chosen_symprec_A": chosen_symprec,
            "model": clean_model,
            "supercell": [int(x) for x in supercell],
            "displacement_A": displacement,
            "n_displacements": len(supercells),
            "temperature_K": temperature_k,
            "fwhm_meV": fwhm_mev,
            "ei_meV": ei_mev,
            "relax": relax_info,
            "elapsed_seconds": round(total_time, 2),
        },
    )
