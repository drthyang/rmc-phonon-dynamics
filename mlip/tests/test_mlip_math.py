"""Unit tests for rmcph_mlip math, physics invariants, and web-export parity."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from rmcph_mlip.neutron import (
    HBAR2_OVER_2MN,
    NEUTRON_SCATT_SIGMA,
    compute_neutron_dos,
    compute_powder_ins,
    gaussian_broaden,
)
from rmcph_mlip.rmc6f import read_rmc6f_folded


class TestMlipMath(unittest.TestCase):
    def test_rmc6f_circular_mean_and_u_iso(self):
        # 2x1x1 supercell with periodic boundary crossing (0.99 and 0.01 around 0.00)
        # In 2x1x1 supercell: cell 0 has fx = 0.495 -> uf = 0.99; cell 1 has fx = 0.505 -> uf = 0.01
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
            # Circular mean of 0.99 and 0.01 must be 0.0 (mod 1), NOT 0.5
            fx = folded.unit_cell.get_scaled_positions()[0, 0]
            dist_to_zero = min(fx, 1.0 - fx)
            self.assertLess(dist_to_zero, 1e-6)
            # Displacement in x is +/- 0.01 * 5.0 A = +/- 0.05 A -> <u^2>/3 = 0.05^2 / 3
            expected_u_iso = (0.05**2) / 3.0
            self.assertAlmostEqual(folded.u_iso_per_site[0], expected_u_iso, places=7)

    def test_gaussian_broaden_preserves_area(self):
        e = np.linspace(0.0, 40.0, 401)
        y = np.zeros_like(e)
        y[200] = 1.0 / (e[1] - e[0])
        b = gaussian_broaden(e, y, fwhm=2.0)
        area = float(np.trapezoid(b, e))
        self.assertAlmostEqual(area, 1.0, places=3)

    def test_neutron_dos_and_ins_kinematics(self):
        # Construct an analytic 2-atom harmonic model (Ga + Se) with exact force constants
        a = 5.0
        unit = PhonopyAtoms(
            symbols=["Ga", "Se"],
            cell=np.eye(3) * a,
            scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        )
        ph = Phonopy(unit, supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        n_sup = len(ph.supercell)
        # Simple spring constant k between Ga and nearest Se + diagonal acoustic sum rule
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
                dist = np.linalg.norm(dc)
                if dist < 4.5:
                    fc[i, j] = -k_spring * np.eye(3)
            fc[i, i] = -np.sum(fc[i], axis=0)
        ph.force_constants = fc

        ndos = compute_neutron_dos(ph, mesh=(7, 7, 7), fwhm_mev=1.0, n_energy=250)
        # Normalization & sum rule
        self.assertAlmostEqual(float(np.trapezoid(ndos.total_dos, ndos.energy_mev)), 1.0, places=2)
        self.assertAlmostEqual(float(np.trapezoid(ndos.neutron_gdos, ndos.energy_mev)), 1.0, places=2)
        el_sum = ndos.element_gdos["Ga"] + ndos.element_gdos["Se"]
        self.assertLess(float(np.max(np.abs(el_sum - ndos.neutron_gdos))), 1e-10)
        # Check sigma/M weights
        self.assertAlmostEqual(
            ndos.element_weights["Ga"],
            NEUTRON_SCATT_SIGMA["Ga"] / ph.primitive.masses[0],
            places=5,
        )

        # Powder INS kinematic cutoff check (Ei = 30 meV)
        ei = 30.0
        ins = compute_powder_ins(ph, ndos, temperature_k=50.0, q_max=8.0, n_q=80, ei_mev=ei)
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
