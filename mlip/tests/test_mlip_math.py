"""Unit tests for rmcph_mlip math, physics invariants, and web-export parity."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from rmcph_mlip.calculators import list_available_models
from rmcph_mlip.cli import main as cli_main
from rmcph_mlip.neutron import (
    HBAR2_OVER_2MN,
    NEUTRON_SCATT_SIGMA,
    chopper_resolution_fwhm,
    coherent_and_incoherent_cross_sections,
    compute_neutron_dos,
    compute_powder_ins,
    compute_sjolander_multiphonon,
    gaussian_broaden,
    variable_gaussian_broaden,
)
from rmcph_mlip.phonons import relax_structure
from rmcph_mlip.rmc6f import read_rmc6f_folded


class _DoubleWellSubgroupCalc(Calculator):
    """Analytic Landau-like potential for a 2-atom AB cubic cell (A fixed near 0, B near (0.5, 0.5, 0.5 + dz)).

    If `favor_collapse=False`:
      * Has a genuine P4mm minimum along z at delta_z = +0.20 Å, WHILE being unstable along x/y
        (a rank-2 saddle point toward R3m if unconstrained with tiny transverse noise!).
    If `favor_collapse=True`:
      * Purely harmonic quadratic well centered at ideal centrosymmetric (0.5, 0.5, 0.5), so any
        initial P4mm polar offset collapses back to Pm-3m (#221) during relaxation.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, favor_collapse: bool = False):
        super().__init__()
        self.favor_collapse = favor_collapse

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos = self.atoms.get_positions()
        cell = self.atoms.cell.array
        center = 0.5 * (cell[0] + cell[1] + cell[2])
        # Relative displacement of atom 1 from body center minus atom 0
        u = (pos[1] - center) - pos[0]
        ux, uy, uz = float(u[0]), float(u[1]), float(u[2])

        k_anchor = 8.0
        r_cm = 0.5 * (pos[0] + (pos[1] - center))
        e_cm = 0.5 * k_anchor * float(np.dot(r_cm, r_cm))
        f_cm = -k_anchor * r_cm

        if self.favor_collapse:
            k_harm = 5.0
            e_rel = 0.5 * k_harm * (ux * ux + uy * uy + uz * uz)
            grad_u = np.array([k_harm * ux, k_harm * uy, k_harm * uz], dtype=float)
        else:
            # Double well: along z, E(uz) = -2*(uz^2) + 25*(uz^4) -> min at uz = 0.20 Å!
            # Along x, y: steeper negative curvature -4*(ux^2 + uy^2) + 25*(ux^4 + uy^4) so P4mm is a saddle!
            e_rel = (
                -2.0 * uz**2 + 25.0 * uz**4
                - 4.0 * (ux**2 + uy**2) + 25.0 * (ux**4 + uy**4)
            )
            grad_u = np.array(
                [
                    -8.0 * ux + 100.0 * ux**3,
                    -8.0 * uy + 100.0 * uy**3,
                    -4.0 * uz + 100.0 * uz**3,
                ],
                dtype=float,
            )

        forces = np.zeros((2, 3), dtype=float)
        forces[0] = 0.5 * f_cm + grad_u
        forces[1] = 0.5 * f_cm - grad_u
        self.results["energy"] = float(e_cm + e_rel)
        self.results["forces"] = forces


class TestMlipMath(unittest.TestCase):
    def test_rmc6f_circular_mean_and_u_iso(self):
        # 2x1x1 supercell with periodic boundary crossing (0.99 and 0.01 around 0.00)
        sample = """(Version 6f format configuration file)
Supercell dimensions: 2 1 1
Lattice vectors (Ang):
10.0 0.0 0.0
0.0 5.0 0.0
0.0 0.0 5.0
Atoms:
1 Ga [1] 0.495 0.500 0.500 1 0 0 0
2 Ga [1] 0.505 0.500 0.500 1 1 0 0
"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test.rmc6f"
            p.write_text(sample, encoding="utf-8")
            folded = read_rmc6f_folded(p)
            self.assertEqual(folded.supercell_dim, (2, 1, 1))
            self.assertEqual(len(folded.unit_cell), 1)
            fx = folded.unit_cell.get_scaled_positions()[0, 0]
            dist_to_zero = min(fx, 1.0 - fx)
            self.assertLess(dist_to_zero, 1e-6)
            expected_u_iso = (0.05**2) / 3.0
            self.assertAlmostEqual(folded.u_iso_per_site[0], expected_u_iso, places=7)

    def test_gaussian_broaden_preserves_area(self):
        e = np.linspace(0.0, 40.0, 401)
        y = np.zeros_like(e)
        y[200] = 1.0 / (e[1] - e[0])
        b = gaussian_broaden(e, y, fwhm=2.0)
        area = float(np.trapezoid(b, e))
        self.assertAlmostEqual(area, 1.0, places=3)

    def test_chopper_resolution_and_variable_broaden_invariants(self):
        e = np.linspace(0.0, 48.0, 241)
        ei = 50.0
        for inst, frac in [("SEQUOIA", 0.018), ("CNCS", 0.022), ("ARCS", 0.030), ("LET", 0.015)]:
            fwhm = chopper_resolution_fwhm(e, ei_mev=ei, instrument=inst)
            # Elastic line matches elastic_frac * Ei
            self.assertAlmostEqual(float(fwhm[0]), frac * ei, places=6)
            # Resolution sharpens strictly monotonically with increasing energy transfer E < Ei
            diffs = np.diff(fwhm)
            self.assertTrue(np.all(diffs < 0.0), f"{inst} FWHM must sharpen monotonically with E")
            self.assertLess(float(fwhm[-1]), 0.6 * float(fwhm[0]))

        # Source-centered variable-width Gaussian broadening conserves spectral sum to machine precision
        y = np.zeros_like(e)
        y[40] = 2.5   # E = 8 meV (broader chopper width)
        y[200] = 2.5  # E = 40 meV (sharper chopper width)
        fwhm_seq = chopper_resolution_fwhm(e, ei_mev=ei, instrument="SEQUOIA")
        b = variable_gaussian_broaden(e, y, fwhm_seq)
        self.assertAlmostEqual(float(np.sum(b)), float(np.sum(y)), places=12)
        # Higher-E peak must be taller because its chopper resolution width is narrower
        self.assertGreater(float(b[200]), float(b[40]) * 1.4)

    def test_cross_sections_and_sjolander_multiphonon_overtones(self):
        # Vanadium is >99.5% nuclear incoherent; Carbon is ~100% coherent
        beta_v, scoh_v, sinc_v = coherent_and_incoherent_cross_sections("V")
        self.assertLess(beta_v, 0.0)  # negative coherent scattering length preserved!
        self.assertLess(scoh_v, 0.03)
        self.assertGreater(sinc_v, 5.0)

        beta_c, scoh_c, sinc_c = coherent_and_incoherent_cross_sections("C")
        self.assertGreater(beta_c, 0.0)
        self.assertAlmostEqual(scoh_c, 5.551, places=2)
        self.assertLess(sinc_c, 0.01)

        # Single Einstein mode at E0 = 15 meV -> n=2 overtone peaks cleanly at 30 meV, n=3 at 45 meV
        e_ndos = np.linspace(0.0, 60.0, 601)
        g_unit = np.exp(-0.5 * ((e_ndos - 15.0) / 0.6) ** 2)
        g_unit /= float(np.trapezoid(g_unit, e_ndos))

        e_grid = e_ndos[e_ndos > 0.5]
        q_grid = np.array([1.0, 4.0, 7.0])
        sqe_multi = compute_sjolander_multiphonon(
            e_grid=e_grid,
            unit_dos_by_el={"Ga": (e_ndos, g_unit)},
            q_grid=q_grid,
            u_iso_by_el={"Ga": 0.01},
            mass_by_el={"Ga": 69.723},
            count_by_el={"Ga": 1},
            temperature_k=10.0,
            max_order=4,
        )
        idx_30 = int(np.argmin(np.abs(e_grid - 30.0)))
        idx_22 = int(np.argmin(np.abs(e_grid - 22.0)))
        # Clear n=2 overtone peak at 2*E0 = 30 meV
        self.assertGreater(float(sqe_multi[idx_30, 1]), 15.0 * float(sqe_multi[idx_22, 1]))
        # Quadratic growth of n=2 with Q^2 at small 2W: (4/1)^4 = 256 * exp(-0.15)
        ratio_q = float(sqe_multi[idx_30, 1] / sqe_multi[idx_30, 0])
        expected_ratio = (16.0**2) * np.exp(-(16.0 - 1.0) * 0.01)
        self.assertAlmostEqual(ratio_q / expected_ratio, 1.0, places=2)

    def test_neutron_dos_coherent_ins_and_kinematics(self):
        # Analytic 2-atom harmonic model (Ga + Se) with exact force constants
        a = 5.0
        unit = PhonopyAtoms(
            symbols=["Ga", "Se"],
            cell=np.eye(3) * a,
            scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        )
        ph = Phonopy(unit, supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        n_sup = len(ph.supercell)
        fc = np.zeros((n_sup, n_sup, 3, 3), dtype=float)
        pos = ph.supercell.positions
        cell = ph.supercell.cell
        inv_cell = np.linalg.inv(cell)
        k_spring = 1.5  # eV / A^2
        for i in range(n_sup):
            for j in range(n_sup):
                if i == j:
                    continue
                d = pos[j] - pos[i]
                df = d @ inv_cell
                df -= np.round(df)
                dc = df @ cell
                if np.linalg.norm(dc) < 4.5:
                    fc[i, j] = -k_spring * np.eye(3)
            fc[i, i] = -np.sum(fc[i], axis=0)
        ph.force_constants = fc

        ndos = compute_neutron_dos(ph, mesh=(7, 7, 7), fwhm_mev=1.0, n_energy=250)
        self.assertAlmostEqual(float(np.trapezoid(ndos.total_dos, ndos.energy_mev)), 1.0, places=2)
        self.assertAlmostEqual(float(np.trapezoid(ndos.neutron_gdos, ndos.energy_mev)), 1.0, places=2)
        el_sum = ndos.element_gdos["Ga"] + ndos.element_gdos["Se"]
        self.assertLess(float(np.max(np.abs(el_sum - ndos.neutron_gdos))), 1e-10)
        self.assertAlmostEqual(
            ndos.element_weights["Ga"],
            NEUTRON_SCATT_SIGMA["Ga"] / ph.primitive.masses[0],
            places=5,
        )

        # Full coherent + incoherent + multiphonon Powder INS check (Ei = 30 meV, SEQUOIA)
        ei = 30.0
        ins = compute_powder_ins(
            ph,
            ndos,
            temperature_k=50.0,
            q_max=8.0,
            n_q=80,
            ei_mev=ei,
            instrument="SEQUOIA",
            include_coherent=True,
            include_multiphonon=True,
        )
        self.assertIsNotNone(ins.sqe_coherent)
        self.assertIsNotNone(ins.sqe_incoherent)
        self.assertIsNotNone(ins.sqe_multiphonon)
        # Exact additive channel decomposition: sqe == sqe_coherent + sqe_incoherent + sqe_multiphonon
        sum_channels = ins.sqe_coherent + ins.sqe_incoherent + ins.sqe_multiphonon
        self.assertLess(float(np.max(np.abs(ins.sqe - sum_channels))), 1e-12)
        self.assertAlmostEqual(float(ins.sqe.max()), 1.0, places=6)
        self.assertGreater(float(ins.sqe_coherent.max()), float(ins.sqe_incoherent.max()))

        # Kinematic cutoff check
        ki = np.sqrt(ei / HBAR2_OVER_2MN)
        for ie, en in enumerate(ins.e_grid):
            if en >= ei:
                self.assertTrue(np.allclose(ins.sqe[ie], 0.0))
            else:
                kf = np.sqrt((ei - en) / HBAR2_OVER_2MN)
                q_lo, q_hi = abs(ki - kf), ki + kf
                outside = (ins.q_grid < q_lo) | (ins.q_grid > q_hi)
                self.assertTrue(np.allclose(ins.sqe[ie, outside], 0.0))

    def test_fix_symmetry_relaxation_and_model_registry(self):
        # 1. Registry check: MACE-MP-0, MPA-0, OMAT-0, r2SCAN, MatterSim, ORB-v3, SevenNet, CHGNet
        models = list_available_models()
        ids = {m["id"] for m in models}
        for expected in (
            "mace-mp-medium",
            "mace-mpa-0",
            "mace-omat-0",
            "mace-matpes-r2scan-0",
            "mattersim-5m",
            "orb-v3",
            "sevennet-0",
            "chgnet",
        ):
            self.assertIn(expected, ids)
        mace_med = next(m for m in models if m["id"] == "mace-mp-medium")
        self.assertTrue(mace_med["installed"])
        self.assertEqual(cli_main(["models"]), 0)

        # 2. P4mm (#99) polar subgroup with tiny 2e-4 Å transverse noise:
        #    FixSymmetry locks structure onto P4mm manifold (uz -> 0.20 Å, ux=uy=0),
        #    whereas unconstrained BFGS slides off the P4mm saddle into Cm / P1!
        a = 4.0
        p4mm_noisy = Atoms(
            symbols=["Ba", "Ti"],
            cell=np.eye(3) * a,
            positions=[[0.0, 0.0, 0.0], [2.0002, 1.9999, 2.08]],  # dz = +0.08 Å + tiny transverse noise
            pbc=True,
        )
        calc_saddle = _DoubleWellSubgroupCalc(favor_collapse=False)

        rel_fix, fmax_fix, steps_fix, diag_fix = relax_structure(
            p4mm_noisy,
            calc_saddle,
            fmax=1e-4,
            relax_mode="fix_symmetry",
            symprec=1e-2,
        )
        self.assertGreater(steps_fix, 0)
        self.assertLess(fmax_fix, 1e-4)
        self.assertEqual(diag_fix["initial_spacegroup"], "P4mm (99)")
        self.assertEqual(diag_fix["relaxed_spacegroup"], "P4mm (99)")
        self.assertTrue(diag_fix["symmetry_preserved"])
        self.assertFalse(diag_fix["collapsed_to_supergroup"])
        self.assertLess(diag_fix["energy_drop_meV_per_atom"], -5.0)
        # Returned Atoms must have constraints stripped so Phonopy displacements are unrestricted
        self.assertEqual(len(rel_fix.constraints), 0)

        # Unconstrained BFGS on the same noisy saddle rolls off P4mm into lower symmetry!
        _, _, _, diag_free = relax_structure(
            p4mm_noisy,
            calc_saddle,
            fmax=1e-4,
            relax_mode="unconstrained",
            symprec=1e-2,
        )
        self.assertNotEqual(diag_free["relaxed_spacegroup"], "P4mm (99)")

        # 3. Supergroup collapse detection: when potential restores dz -> 0 (Pm-3m #221)
        calc_collapse = _DoubleWellSubgroupCalc(favor_collapse=True)
        _, _, _, diag_col = relax_structure(
            p4mm_noisy,
            calc_collapse,
            fmax=1e-4,
            relax_mode="fix_symmetry",
            symprec=1e-2,
        )
        self.assertTrue(diag_col["collapsed_to_supergroup"])
        self.assertEqual(diag_col["relaxed_spacegroup"], "Pm-3m (221)")

        # 4. Frozen RMC mode: 0 steps, exact coordinates preserved, energy evaluated
        _, _, steps_frz, diag_frz = relax_structure(
            p4mm_noisy,
            calc_saddle,
            relax_mode="frozen",
            symprec=1e-2,
        )
        self.assertEqual(steps_frz, 0)
        self.assertEqual(diag_frz["max_atom_disp_A"], 0.0)


if __name__ == "__main__":
    unittest.main()
