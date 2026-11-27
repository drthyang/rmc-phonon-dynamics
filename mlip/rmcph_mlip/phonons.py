"""Harmonic phonon workflow with symmetry-preserving subgroup relaxation (FixSymmetry).

Why Symmetry-Constrained Relaxation Matters for RMC Local Subgroups
===================================================================
When Reverse Monte Carlo (RMC) supercell symmetry analysis identifies both a high-symmetry
Parent average structure G (e.g. cubic Pm-3m) and a lower-symmetry Local motif H < G
(e.g. polar R3m, Amm2, P4mm, or octahedral-tilt I4/mcm, P4/mbm), evaluating harmonic
MLIP phonons requires careful treatment of structural relaxation:

1. `relax_mode = "fix_symmetry"` (Default — Symmetry-Constrained Wyckoff Relaxation):
   * Attaches `ase.constraints.FixSymmetry(atoms, symprec=...)` during BFGS optimization.
   * At every step, atomic forces F_i and cell stresses are projected onto the totally
     symmetric A_1g subspace of space group H.
   * Physical benefits:
     (a) High-frequency bond-length noise from finite RMC statistics relaxes cleanly to
         ||P_H F||_inf < fmax, eliminating spurious stiff bond-stretch artifacts.
     (b) Space group H cannot drift into P1 from floating-point roundoff, AND rank-k
         saddle-point subgroups (such as orthorhombic Amm2 or tetragonal P4mm between
         cubic Pm-3m and rhombohedral R3m) stay locked on their symmetry manifold so
         each candidate hypothesis can be tested independently against experimental INS!
     (c) If an anharmonic high-temperature RMC distortion has zero restoring barrier at
         T = 0 K and its order-parameter amplitude collapses to zero inside H (promoting
         the relaxed structure back into supergroup G), we explicitly detect this via
         post-relaxation `spglib` space-group verification (`collapsed_to_supergroup = True`)
         so the user knows to also inspect `relax_mode = "frozen"`.

2. `relax_mode = "frozen"` (Instantaneous RMC Local Geometry, 0 BFGS Steps):
   * Evaluates the single-point energy E_0/atom and finite-displacement Hessian directly
     at the symmetrized RMC coordinates without moving atoms. Useful when local distortions
     are dynamically/entropically stabilized at finite temperature.

3. `relax_mode = "unconstrained"` (Free BFGS Optimization):
   * Allows atoms to relax freely without symmetry projection to locate the nearest
     T = 0 K local minimum (even if it lowers symmetry further to H' < H).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import seekpath
import spglib
from ase import Atoms
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.io import read as ase_read
from ase.optimize import BFGS
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from .neutron import (
    THZ_TO_MEV,
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
    metadata: dict[str, Any]


def atoms_to_spglib(atoms: Atoms):
    return (
        atoms.cell.array.copy(),
        atoms.get_scaled_positions().copy(),
        atoms.get_atomic_numbers().copy(),
    )


def detect_spacegroup_info(atoms: Atoms, symprec: float) -> tuple[str, int]:
    """Return (`'Symbol (Number)'`, `number`) at tolerance `symprec`."""
    ds = spglib.get_symmetry_dataset(atoms_to_spglib(atoms), symprec=max(float(symprec), 1e-5))
    if ds is None:
        return "P1 (1)", 1
    return f"{ds.international} ({ds.number})", int(ds.number)


def _minimum_image_displacements(before: Atoms, after: Atoms) -> np.ndarray:
    """Compute per-atom Cartesian displacement magnitudes [Å] under periodic minimum image."""
    df = after.get_scaled_positions() - before.get_scaled_positions()
    df -= np.round(df)
    d_cart = df @ after.cell.array
    return np.linalg.norm(d_cart, axis=1)


def load_unit_cell(structure_path: str | Path) -> tuple[Atoms, dict[str, Any]]:
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
    relax_mode: str = "fix_symmetry",
    symprec: float = 1e-2,
) -> tuple[Atoms, float, int, dict[str, Any]]:
    """Relax atomic positions (and optionally cell) while tracking energetics & space-group invariants.

    Parameters
    ----------
    relax_mode : {'fix_symmetry', 'unconstrained', 'frozen'}
        * `'fix_symmetry'` (default): applies `ase.constraints.FixSymmetry` at `symprec` so
          forces and strains remain strictly inside the initial space group's Wyckoff manifold.
        * `'unconstrained'`: standard unconstrained BFGS relaxation.
        * `'frozen'`: 0 optimization steps; evaluates initial energy & forces only.

    Returns
    -------
    (clean_atoms, final_fmax, nsteps, diagnostics_dict)
        `clean_atoms` is returned WITHOUT active ASE constraints so subsequent Phonopy
        finite-displacement supercells evaluate unrestricted restoring forces.
    """
    mode = (relax_mode or "fix_symmetry").strip().lower()
    if mode not in ("fix_symmetry", "unconstrained", "frozen"):
        raise ValueError(f"Unknown relax_mode={relax_mode!r}; expected 'fix_symmetry', 'unconstrained', or 'frozen'.")

    sp_check = max(min(float(symprec), 0.05), 1e-3)
    init_sg_label, init_sg_num = detect_spacegroup_info(atoms, symprec=sp_check)

    work = atoms.copy()
    work.calc = calculator

    # Measure pre-relaxation potential energy & unprojected forces at the input geometry
    e_init_total = float(work.get_potential_energy())
    e_init_per_atom = e_init_total / len(work)
    raw_forces_init = work.get_forces(apply_constraint=False)
    init_fmax = float(np.linalg.norm(raw_forces_init, axis=1).max())

    if mode == "frozen":
        clean = atoms.copy()
        diagnostics = {
            "performed": False,
            "mode": "frozen",
            "relax_cell": False,
            "steps": 0,
            "initial_fmax_eV_per_A": round(init_fmax, 6),
            "final_fmax_eV_per_A": round(init_fmax, 6),
            "raw_fmax_eV_per_A": round(init_fmax, 6),
            "initial_energy_per_atom_eV": round(e_init_per_atom, 7),
            "relaxed_energy_per_atom_eV": round(e_init_per_atom, 7),
            "energy_drop_meV_per_atom": 0.0,
            "max_atom_disp_A": 0.0,
            "rms_atom_disp_A": 0.0,
            "initial_spacegroup": init_sg_label,
            "relaxed_spacegroup": init_sg_label,
            "symmetry_preserved": True,
            "collapsed_to_supergroup": False,
        }
        return clean, init_fmax, 0, diagnostics

    if mode == "fix_symmetry":
        work.set_constraint(
            FixSymmetry(
                work,
                symprec=max(float(symprec), 1e-4),
                adjust_positions=True,
                adjust_cell=bool(relax_cell),
            )
        )

    target = FrechetCellFilter(work) if relax_cell else work
    opt = BFGS(target, logfile=None)
    opt.run(fmax=fmax, steps=max_steps)

    e_final_total = float(work.get_potential_energy())
    e_final_per_atom = e_final_total / len(work)
    proj_forces = work.get_forces(apply_constraint=True)
    raw_forces = work.get_forces(apply_constraint=False)
    final_fmax = float(np.linalg.norm(proj_forces, axis=1).max())
    raw_fmax = float(np.linalg.norm(raw_forces, axis=1).max())

    # Detach FixSymmetry before returning so Phonopy displaced supercells are unconstrained
    clean = Atoms(
        numbers=work.get_atomic_numbers(),
        scaled_positions=work.get_scaled_positions(),
        cell=work.cell.array.copy(),
        pbc=True,
    )

    disp_mags = _minimum_image_displacements(atoms, clean)
    max_disp = float(disp_mags.max()) if len(disp_mags) else 0.0
    rms_disp = float(np.sqrt(np.mean(disp_mags ** 2))) if len(disp_mags) else 0.0

    # Check relaxed space group at tight tolerance (1e-3 Å) to detect supergroup collapse or drift
    relaxed_sg_label, relaxed_sg_num = detect_spacegroup_info(clean, symprec=sp_check)
    _, tight_relaxed_num = detect_spacegroup_info(clean, symprec=1e-3)
    _, tight_init_num = detect_spacegroup_info(atoms, symprec=1e-3)

    collapsed_to_supergroup = bool(tight_relaxed_num > max(init_sg_num, tight_init_num))
    symmetry_preserved = bool(relaxed_sg_num == init_sg_num and not collapsed_to_supergroup)

    diagnostics = {
        "performed": True,
        "mode": mode,
        "relax_cell": bool(relax_cell),
        "steps": int(opt.nsteps),
        "initial_fmax_eV_per_A": round(init_fmax, 6),
        "final_fmax_eV_per_A": round(final_fmax, 6),
        "raw_fmax_eV_per_A": round(raw_fmax, 6),
        "initial_energy_per_atom_eV": round(e_init_per_atom, 7),
        "relaxed_energy_per_atom_eV": round(e_final_per_atom, 7),
        "energy_drop_meV_per_atom": round((e_final_per_atom - e_init_per_atom) * 1000.0, 4),
        "max_atom_disp_A": round(max_disp, 5),
        "rms_atom_disp_A": round(rms_disp, 5),
        "initial_spacegroup": init_sg_label,
        "relaxed_spacegroup": relaxed_sg_label,
        "symmetry_preserved": symmetry_preserved,
        "collapsed_to_supergroup": collapsed_to_supergroup,
    }
    return clean, final_fmax, int(opt.nsteps), diagnostics


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
    relax_mode: str | None = None,
    fmax: float = 1e-3,
    npoints: int = 41,
    dos_mesh: Sequence[int] = (15, 15, 15),
    temperature_k: float = 5.0,
    fwhm_mev: float = 0.8,
    q_max: float = 8.0,
    ei_mev: float | None = 50.0,
    instrument: str = "SEQUOIA",
    model_name: str = "mace-mp-medium",
) -> PhononResult:
    """Run end-to-end harmonic phonon + FixSymmetry relaxation + neutron GDOS/INS calculation."""
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

    eff_relax_mode = relax_mode if relax_mode is not None else ("fix_symmetry" if relax else "frozen")
    t_rel = time.perf_counter()
    prim_atoms, final_fmax, nsteps, relax_info = relax_structure(
        prim_atoms,
        calculator,
        fmax=fmax,
        relax_cell=relax_cell,
        relax_mode=eff_relax_mode,
        symprec=max(min(chosen_symprec, 0.02), 1e-3),
    )
    dt_rel = time.perf_counter() - t_rel
    if relax_info["performed"]:
        collapse_note = " [NOTE: amplitude relaxed into higher-symmetry supergroup!]" if relax_info["collapsed_to_supergroup"] else ""
        print(
            f"[mlip] Relaxed ({eff_relax_mode}, {'cell+pos' if relax_cell else 'pos-only'}) in {nsteps} steps "
            f"({dt_rel:.1f}s): fmax={final_fmax:.2e} eV/Å, ΔE={relax_info['energy_drop_meV_per_atom']:+.2f} meV/atom, "
            f"max Δr={relax_info['max_atom_disp_A']:.4f} Å, SG={relax_info['relaxed_spacegroup']}{collapse_note}"
        )
    else:
        print(
            f"[mlip] Frozen geometry (0 steps): E0={relax_info['relaxed_energy_per_atom_eV']:.5f} eV/atom, "
            f"initial fmax={final_fmax:.2e} eV/Å"
        )

    # Build Phonopy on the standardized primitive cell (primitive_matrix = I)
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
    if getattr(ph, "band_structure", None) is not None:
        bands_thz = np.asarray(ph.band_structure.frequencies[0], dtype=float)
    else:
        bands_thz = np.asarray(ph.get_band_structure_dict()["frequencies"][0], dtype=float)

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
        instrument=instrument,
    )
    dt_ins = time.perf_counter() - t_ins
    weights_str = ", ".join(f"{el}:{w:.3f}" for el, w in ndos.element_weights.items())
    print(
        f"[mlip] Neutron-weighted DOS & INS S(|Q|,E) computed in {dt_ins:.1f}s "
        f"(T={temperature_k} K, inst={instrument}, σ/M [barn/amu]: {weights_str})"
    )

    total_time = time.perf_counter() - t0
    bands_mev = bands_thz * THZ_TO_MEV
    min_freq_mev = float(bands_mev.min())
    max_freq_mev = float(bands_mev.max())
    n_imag_branches = int(np.sum(np.min(bands_mev, axis=0) < -0.5))
    print(
        f"[mlip] Done in {total_time:.1f}s — freq range: "
        f"{bands_thz.min():.2f} .. {bands_thz.max():.2f} THz "
        f"({min_freq_mev:.2f} .. {max_freq_mev:.2f} meV, imaginary branches={n_imag_branches})"
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
            "temperature_k": temperature_k,
            "fwhm_meV": fwhm_mev,
            "ei_meV": ei_mev,
            "instrument": instrument,
            "energy_per_atom_eV": relax_info["relaxed_energy_per_atom_eV"],
            "min_freq_meV": round(min_freq_mev, 3),
            "max_freq_meV": round(max_freq_mev, 3),
            "n_imaginary_branches": n_imag_branches,
            "relax": relax_info,
            "elapsed_seconds": round(total_time, 2),
        },
    )
