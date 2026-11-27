"""Unit tests for rmcph_mlip math, physics invariants, and web-export parity."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

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
from rmcph_mlip.rmc6f import read_rmc6f_folded


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


if __name__ == "__main__":
    unittest.main()
