"""Compare F-43m vs 1x1x2 P-421m (No. 113) phonons in GaTa4Se8 using MACE-MP-0-medium.

1. Builds exact F-43m (52-atom conv / 13-atom prim) from data/GTS_5K.cif.
2. Projects 5K RMC supercell configurations (data/ensemble_20A_5K/GTS_5K_*.rmc6f)
   onto the 1x1x2 P-421m (No. 113) symmetry subspace (104 atoms, a x a x 2a).
3. Also tests a symmetry-pure P-421m Ta4-cluster distortion (amplified to check
   sensitivity of the ~13 meV region).
4. Computes harmonic phonons, neutron-weighted GDOS, per-element partial GDOS,
   and analyzes the exact atomic eigenvector character of all modes near 13 meV.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import spglib
from ase import Atoms
from ase.io import read, write

from rmcph_mlip.calculators import build_calculator
from rmcph_mlip.export import export_neutron_dos_csv, export_phonopy_dm_json
from rmcph_mlip.neutron import THZ_TO_MEV
from rmcph_mlip.phonons import compute_harmonic_phonons

# Exact 8 symmetry operations of P-421m (Space Group No. 113)
OPS_P421M = [
    (np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float), np.array([0.0, 0.0, 0.0])),
    (np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float), np.array([0.0, 0.0, 0.0])),
    (np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1]], dtype=float), np.array([0.0, 0.0, 0.0])),
    (np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]], dtype=float), np.array([0.0, 0.0, 0.0])),
    (np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float), np.array([0.5, 0.5, 0.0])),
    (np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float), np.array([0.5, 0.5, 0.0])),
    (np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float), np.array([0.5, 0.5, 0.0])),
    (np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float), np.array([0.5, 0.5, 0.0])),
]


def build_p421m_perms(ideal_112: Atoms) -> list[np.ndarray]:
    """Precompute exact atom-index permutations for the 8 P-421m operations on ideal_112."""
    pos = ideal_112.get_scaled_positions() % 1.0
    nums = ideal_112.get_atomic_numbers()
    n = len(pos)
    perms = []
    for R, t in OPS_P421M:
        p2 = (pos @ R.T + t) % 1.0
        perm = np.empty(n, dtype=int)
        for i in range(n):
            d = pos - p2[i]
            d -= np.round(d)
            dists = np.linalg.norm(d, axis=1)
            j = int(np.argmin(dists))
            assert dists[j] < 1e-4 and nums[j] == nums[i], f"Ideal P-421m mismatch: d={dists[j]}, {nums[i]} vs {nums[j]}"
            perm[i] = j
        perms.append(perm)
    return perms


def symmetrize_to_p421m(atoms: Atoms, ideal_112: Atoms, perms: list[np.ndarray]) -> Atoms:
    """Project displacements (atoms - ideal_112) onto exact P-421m (#113) using exact permutations."""
    ideal_frac = ideal_112.get_scaled_positions() % 1.0
    df_in = atoms.get_scaled_positions() - ideal_frac
    df_in -= np.round(df_in)
    n = len(ideal_frac)
    sum_df = np.zeros((n, 3), dtype=float)
    for (R, _t), perm in zip(OPS_P421M, perms):
        # Under g = (R, t), displacement df_i on atom i maps to R @ df_i on atom perm[i]
        rot_df = df_in @ R.T
        sum_df[perm] += rot_df
    new_pos = (ideal_frac + sum_df / len(OPS_P421M)) % 1.0
    out = ideal_112.copy()
    out.set_scaled_positions(new_pos)
    return out


def extract_rmc_1x1x2_p421m(f43m_conv: Atoms, rmc6f_paths: list[Path]) -> tuple[Atoms, np.ndarray]:
    """Fold 8x8x8 RMC supercells into 1x1x2 blocks, domain-align onto P-421m, and project."""
    ideal_112 = f43m_conv.repeat((1, 1, 2))
    perms = build_p421m_perms(ideal_112)
    ideal_frac = ideal_112.get_scaled_positions() % 1.0
    ideal_syms = list(ideal_112.symbols)
    cell_112 = ideal_112.cell.array
    inv_cell_112 = np.linalg.inv(cell_112)
    n_112 = len(ideal_112)

    # Precompute exact origin shift (and inversion) from RMC6F unit-cell frame -> spglib F-43m frame
    # by minimizing max same-species distance over ALL 52 atoms (handles non-centrosymmetric F-43m cleanly).
    from rmcph_mlip.rmc6f import read_rmc6f_folded
    folded0 = read_rmc6f_folded(rmc6f_paths[0])
    p_uc = folded0.unit_cell.get_scaled_positions() % 1.0
    n_uc = folded0.unit_cell.get_atomic_numbers()
    p_f = f43m_conv.get_scaled_positions() % 1.0
    n_f = f43m_conv.get_atomic_numbers()
    best_align = (1e9, 1, np.zeros(3))
    for inv in (1, -1):
        for sx in (0.0, 0.25, 0.5, 0.75):
            for sy in (0.0, 0.25, 0.5, 0.75):
                for sz in (0.0, 0.25, 0.5, 0.75):
                    sh = np.array([sx, sy, sz])
                    p2 = (inv * p_uc + sh) % 1.0
                    maxd = 0.0
                    for k_a in range(len(p2)):
                        mask = (n_f == n_uc[k_a])
                        d = p_f[mask] - p2[k_a]
                        d -= np.round(d)
                        md = float(np.min(np.linalg.norm(d, axis=1)))
                        if md > maxd:
                            maxd = md
                    if maxd < best_align[0]:
                        best_align = (maxd, inv, sh)
    _, align_inv, origin_shift_unit = best_align
    print(f"[p421m] Exact RMC6F -> spglib F-43m alignment: inv={align_inv}, shift={origin_shift_unit}, max_err={best_align[0]*f43m_conv.cell.lengths()[0]:.4f} Å")

    disp_accum = np.zeros((n_112, 3), dtype=float)
    n_blocks = 0
    ref_mode = None

    for path in rmc6f_paths:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        dim = None
        start = None
        i = 0
        while i < len(lines):
            s = lines[i].strip()
            if s.startswith("Supercell dimensions:"):
                dim = [int(x) for x in s.split(":", 1)[1].split()]
            elif s.startswith("Atoms:"):
                start = i + 1
                break
            i += 1
        nx, ny, nz = dim
        assert nz % 2 == 0

        # Group atoms by (bx, by, bz_pair) where bz_pair = iz // 2
        blocks: dict[tuple[int, int, int], list[tuple[str, np.ndarray]]] = {}
        for line in lines[start:]:
            toks = line.split()
            if len(toks) < 7:
                continue
            el = toks[1]
            fx, fy, fz = float(toks[-7]), float(toks[-6]), float(toks[-5])
            ix, iy, iz = int(toks[-3]), int(toks[-2]), int(toks[-1])
            key = (ix, iy, iz // 2)
            # Fractional coordinates inside the 1x1x2 block (with exact origin shift applied)
            bfx = (align_inv * fx * nx + origin_shift_unit[0]) % 1.0
            bfy = (align_inv * fy * ny + origin_shift_unit[1]) % 1.0
            bfz = (align_inv * fz * (nz // 2) + origin_shift_unit[2] * 0.5) % 1.0
            blocks.setdefault(key, []).append((el, np.array([bfx, bfy, bfz])))

        # For each 1x1x2 block, match each atom to nearest SAME-SPECIES ideal_112 site
        idx_by_el = {el: np.array([k for k, s in enumerate(ideal_syms) if s == el]) for el in ("Ga", "Ta", "Se")}
        for key, atom_list in blocks.items():
            if len(atom_list) != n_112:
                continue
            u_block = np.zeros((n_112, 3), dtype=float)
            for el, bf in atom_list:
                cand_idxs = idx_by_el[el]
                df = bf[None, :] - ideal_frac[cand_idxs]
                df -= np.round(df)
                dcart = df @ cell_112
                dists = np.linalg.norm(dcart, axis=1)
                best_local = int(np.argmin(dists))
                j = int(cand_idxs[best_local])
                u_block[j] = dcart[best_local]
            # Remove block COM drift
            u_block -= u_block.mean(axis=0, keepdims=True)
            # Project onto P-421m
            cand = ideal_112.copy()
            cand.set_scaled_positions((ideal_frac + u_block @ inv_cell_112) % 1.0)
            sym_cand = symmetrize_to_p421m(cand, ideal_112, perms)
            df_sym = sym_cand.get_scaled_positions() - ideal_frac
            df_sym -= np.round(df_sym)
            u_sym = df_sym @ cell_112

            # Align z-half-shift polarity (t_z = 0.5 swaps the two z-layers)
            if ref_mode is None:
                ref_mode = u_sym.copy()
                disp_accum += u_sym
            else:
                # Compare dot product directly vs shifted by z+0.5
                cand_sh = cand.copy()
                p_sh = cand.get_scaled_positions()
                p_sh[:, 2] = (p_sh[:, 2] + 0.5) % 1.0
                cand_sh.set_scaled_positions(p_sh)
                sym_sh = symmetrize_to_p421m(cand_sh, ideal_112, perms)
                df_sh = sym_sh.get_scaled_positions() - ideal_frac
                df_sh -= np.round(df_sh)
                u_sh = df_sh @ cell_112
                if float(np.sum(u_sh * ref_mode)) > float(np.sum(u_sym * ref_mode)):
                    disp_accum += u_sh
                else:
                    disp_accum += u_sym
            n_blocks += 1

    mean_u = disp_accum / max(1, n_blocks)
    # Remove any residual COM drift and re-symmetrize
    mean_u -= mean_u.mean(axis=0, keepdims=True)
    out = ideal_112.copy()
    out.set_scaled_positions((ideal_frac + mean_u @ inv_cell_112) % 1.0)
    out = symmetrize_to_p421m(out, ideal_112, perms)
    df_final = out.get_scaled_positions() - ideal_frac
    df_final -= np.round(df_final)
    u_final = df_final @ cell_112
    print(f"[p421m] Projected {n_blocks} 1x1x2 blocks from {len(rmc6f_paths)} RMC configs.")
    print(f"[p421m] RMS distortion = {np.sqrt(np.mean(np.sum(u_final**2, axis=1))):.5f} Å, max = {np.max(np.linalg.norm(u_final, axis=1)):.5f} Å")
    return out, u_final


def analyze_modes_near(res, label: str, e_lo: float = 11.0, e_hi: float = 15.0):
    """Inspect Gamma-point and BZ-mesh modes in [e_lo, e_hi] meV and characterize eigenvectors."""
    ph = res.phonopy
    prim = ph.primitive
    symbols = list(prim.symbols)
    masses = np.asarray(prim.masses, dtype=float)
    pos = prim.positions
    cell = prim.cell

    # Compute Gamma-point frequencies and eigenvectors
    freqs_thz, eigvecs = ph.get_frequencies_with_eigenvectors([0.0, 0.0, 0.0])
    freqs_mev = np.asarray(freqs_thz, dtype=float) * THZ_TO_MEV

    print(f"\n=== {label}: Gamma-point modes in [{e_lo:.1f}, {e_hi:.1f}] meV ===")
    in_win = [m for m, e in enumerate(freqs_mev) if e_lo <= e <= e_hi]
    print(f"Count at Gamma: {len(in_win)} modes (total branches = {len(freqs_mev)})")
    for m in in_win:
        ev = eigvecs[:, m].reshape(len(symbols), 3)
        w_atom = np.sum(np.abs(ev) ** 2, axis=1)
        by_el = {}
        for el in ("Ga", "Ta", "Se"):
            by_el[el] = float(sum(w_atom[i] for i, s in enumerate(symbols) if s == el))
        # Cartesian displacement amplitude direction (x, y, z)
        w_dir = np.sum(np.abs(ev) ** 2, axis=0)
        print(
            f"  mode {m+1:3d}: E = {freqs_mev[m]:6.3f} meV ({freqs_thz[m]:5.3f} THz) | "
            f"Ga={by_el['Ga']*100:4.1f}% Ta={by_el['Ta']*100:4.1f}% Se={by_el['Se']*100:4.1f}% | "
            f"pol (x,y,z)=({w_dir[0]:.2f},{w_dir[1]:.2f},{w_dir[2]:.2f})"
        )

    # Also inspect Ta4 cluster bond-length modulation for the mode(s) closest to 13 meV
    ta_idx = [i for i, s in enumerate(symbols) if s == "Ta"]
    inv_c = np.linalg.inv(cell)
    # Find short Ta-Ta bonds (< 3.2 A = intra-cluster Ta4 bonds)
    ta_pairs = []
    for a in range(len(ta_idx)):
        for b in range(a + 1, len(ta_idx)):
            i, j = ta_idx[a], ta_idx[b]
            d = pos[j] - pos[i]
            df = d @ inv_c
            df -= np.round(df)
            dc = df @ cell
            dist = float(np.linalg.norm(dc))
            if dist < 3.3:
                ta_pairs.append((i, j, dc / dist, dist))
    if in_win and ta_pairs:
        print(f"  (Found {len(ta_pairs)} intra-cluster Ta-Ta bonds, mean d={np.mean([p[3] for p in ta_pairs]):.3f} Å)")
        for m in in_win:
            ev = np.real(eigvecs[:, m].reshape(len(symbols), 3)) / np.sqrt(masses[:, None])
            # Relative bond-length changes dr_ij = (u_j - u_i) · r_hat_ij
            drs = [float(np.dot(ev[j] - ev[i], rhat)) for i, j, rhat, _ in ta_pairs]
            drs_arr = np.asarray(drs)
            # Breathing = mean(dr) (all bonds expand/contract in phase); Shear/JT = std(dr)
            breath = abs(float(np.mean(drs_arr)))
            shear = float(np.std(drs_arr))
            # Check z-component vs xy-component of Ta motion
            ta_u = ev[ta_idx]
            ta_norm = np.linalg.norm(ta_u) + 1e-12
            if shear > 0.005 or breath > 0.005:
                kind = "Ta4 breathing" if breath > 1.5 * shear else "Ta4 JT/shear" if shear > 1.5 * breath else "mixed Ta4"
            else:
                kind = "Se/Ga-dominated or Ta4 rigid"
            if abs(freqs_mev[m] - 13.0) < 1.5:
                print(f"    -> mode {m+1:3d} ({freqs_mev[m]:.2f} meV): {kind} (|mean Δd|={breath:.4f}, std Δd={shear:.4f})")


def main():
    repo = Path(__file__).resolve().parents[2]
    out_dir = repo / "results" / "gts_f43m_vs_p421m"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Standardize F-43m conventional cell from data/GTS_5K.cif
    raw = read(str(repo / "data" / "GTS_5K.cif"))
    std = spglib.standardize_cell(
        (raw.cell.array, raw.get_scaled_positions(), raw.get_atomic_numbers()),
        to_primitive=False, no_idealize=False, symprec=0.1,
    )
    f43m_conv = Atoms(numbers=std[2], scaled_positions=std[1], cell=std[0], pbc=True)
    f43m_cif = out_dir / "GTS_F43m.cif"
    write(str(f43m_cif), f43m_conv)

    # 2. Extract 1x1x2 P-421m from 5K RMC configurations
    rmc_files = sorted((repo / "data" / "ensemble_20A_5K").glob("GTS_5K_*.rmc6f"))
    rmc_files = [p for p in rmc_files if "AVERAGE" not in p.name][:5]
    p421m_rmc, u_rmc = extract_rmc_1x1x2_p421m(f43m_conv, rmc_files)

    # Check spglib symmetry of RMC-projected 1x1x2 P-421m
    ds_rmc = spglib.get_symmetry_dataset(
        (p421m_rmc.cell.array, p421m_rmc.get_scaled_positions(), p421m_rmc.get_atomic_numbers()),
        symprec=1e-4,
    )
    print(f"[p421m] RMC-projected spglib (1e-4 Å): {ds_rmc.international} (#{ds_rmc.number}), ops={len(ds_rmc.rotations)}")

    # Also create a 5x-amplified P-421m distortion (~0.06 A RMS, typical of static low-T JT/quadrupolar order)
    # so we can cleanly separate pure zone-folding from finite-amplitude P-421m distortion effects!
    ideal_112 = f43m_conv.repeat((1, 1, 2))
    perms_112 = build_p421m_perms(ideal_112)
    inv_c = np.linalg.inv(ideal_112.cell.array)
    rms_rmc = float(np.sqrt(np.mean(np.sum(u_rmc**2, axis=1))))
    scale_amp = 0.06 / max(rms_rmc, 1e-6)
    p421m_amp = ideal_112.copy()
    p421m_amp.set_scaled_positions((ideal_112.get_scaled_positions() + (u_rmc * scale_amp) @ inv_c) % 1.0)
    p421m_amp = symmetrize_to_p421m(p421m_amp, ideal_112, perms_112)
    ds_amp = spglib.get_symmetry_dataset(
        (p421m_amp.cell.array, p421m_amp.get_scaled_positions(), p421m_amp.get_atomic_numbers()),
        symprec=1e-3,
    )
    print(f"[p421m] Amplified (0.06 Å RMS) spglib (1e-3 Å): {ds_amp.international} (#{ds_amp.number}), ops={len(ds_amp.rotations)}")

    p421m_rmc_cif = out_dir / "GTS_1x1x2_P421m_rmc.cif"
    p421m_amp_cif = out_dir / "GTS_1x1x2_P421m_0p06A.cif"
    write(str(p421m_rmc_cif), p421m_rmc)
    write(str(p421m_amp_cif), p421m_amp)

    # 3. Compute MACE-MP-0-medium phonons for F-43m and 1x1x2 P-421m
    calc = build_calculator(model="mace-mp-medium", device="cpu", dtype="float64")

    print("\n--- Running MACE-MP on F-43m (relaxed, 2x2x2 supercell = 104 atoms) ---")
    res_f43m = compute_harmonic_phonons(
        structure_path=f43m_cif,
        calculator=calc,
        supercell=(2, 2, 2),
        symprec=0.05,
        relax=True,
        temperature_k=5.0,
        fwhm_mev=0.6,
        dos_mesh=(11, 11, 11),
    )
    export_phonopy_dm_json(res_f43m, out_dir / "GTS_F43m_mlip_dm.json")
    export_neutron_dos_csv(res_f43m, out_dir / "GTS_F43m_mlip_ndos.csv")

    # Check F-43m X-point (0, 0, 1/2) in conventional / (0.5, 0.5, 0) in primitive!
    # In FCC primitive cell, conventional (0, 0, 1/2) is q_prim = (1/2, 1/2, 0) = X!
    x_freqs_thz, x_eigvecs = res_f43m.phonopy.get_frequencies_with_eigenvectors([0.5, 0.5, 0.0])
    x_freqs_mev = np.asarray(x_freqs_thz) * THZ_TO_MEV
    print("\n=== F-43m X-point q_conv=(0,0,1/2) modes in [11.0, 15.0] meV (fold to Gamma in 1x1x2!) ===")
    prim_f = res_f43m.phonopy.primitive
    for m, e in enumerate(x_freqs_mev):
        if 11.0 <= e <= 15.0:
            ev = x_eigvecs[:, m].reshape(len(prim_f.symbols), 3)
            w = np.sum(np.abs(ev) ** 2, axis=1)
            by_el = {el: float(sum(w[i] for i, s in enumerate(prim_f.symbols) if s == el)) for el in ("Ga", "Ta", "Se")}
            print(f"  X-mode {m+1:2d}: E = {e:6.3f} meV | Ga={by_el['Ga']*100:4.1f}% Ta={by_el['Ta']*100:4.1f}% Se={by_el['Se']*100:4.1f}%")

    print("\n--- Running MACE-MP on 1x1x2 P-421m (RMC-projected, no-relax, 1x1x1 = 104 atoms) ---")
    res_p421m_rmc = compute_harmonic_phonons(
        structure_path=p421m_rmc_cif,
        calculator=calc,
        supercell=(1, 1, 1),
        symprec=1e-4,
        relax=False,
        temperature_k=5.0,
        fwhm_mev=0.6,
        dos_mesh=(5, 5, 3),
    )
    export_phonopy_dm_json(res_p421m_rmc, out_dir / "GTS_1x1x2_P421m_rmc_mlip_dm.json")
    export_neutron_dos_csv(res_p421m_rmc, out_dir / "GTS_1x1x2_P421m_rmc_mlip_ndos.csv")

    print("\n--- Running MACE-MP on 1x1x2 P-421m (0.06 Å distorted, no-relax, 1x1x1 = 104 atoms) ---")
    res_p421m_amp = compute_harmonic_phonons(
        structure_path=p421m_amp_cif,
        calculator=calc,
        supercell=(1, 1, 1),
        symprec=1e-3,
        relax=False,
        temperature_k=5.0,
        fwhm_mev=0.6,
        dos_mesh=(5, 5, 3),
    )
    export_phonopy_dm_json(res_p421m_amp, out_dir / "GTS_1x1x2_P421m_0p06A_mlip_dm.json")
    export_neutron_dos_csv(res_p421m_amp, out_dir / "GTS_1x1x2_P421m_0p06A_mlip_ndos.csv")

    # 4. Detailed mode analysis around 13 meV
    analyze_modes_near(res_f43m, "F-43m (relaxed)", 11.0, 15.0)
    analyze_modes_near(res_p421m_rmc, "1x1x2 P-421m (RMC-projected)", 11.0, 15.0)
    analyze_modes_near(res_p421m_amp, "1x1x2 P-421m (0.06 Å RMS distortion)", 11.0, 15.0)

    # 5. Quantitative GDOS difference around 13 meV
    e_common = np.linspace(0.0, 34.0, 341)
    g_f = np.interp(e_common, res_f43m.neutron_dos.energy_mev, res_f43m.neutron_dos.neutron_gdos)
    g_rmc = np.interp(e_common, res_p421m_rmc.neutron_dos.energy_mev, res_p421m_rmc.neutron_dos.neutron_gdos)
    g_amp = np.interp(e_common, res_p421m_amp.neutron_dos.energy_mev, res_p421m_amp.neutron_dos.neutron_gdos)

    print("\n=== Neutron-weighted GDOS difference ΔG(E) = G(P-421m) - G(F-43m) ===")
    for name, g_cmp in [("RMC-projected P-421m", g_rmc), ("0.06 Å distorted P-421m", g_amp)]:
        diff = g_cmp - g_f
        win = (e_common >= 11.0) & (e_common <= 15.5)
        idx_max = int(np.argmax(diff[win]))
        e_peak = float(e_common[win][idx_max])
        d_peak = float(diff[win][idx_max])
        print(f"  {name}: peak gain in [11, 15.5] meV at E = {e_peak:.2f} meV (ΔG = {d_peak:+.5f}, G_F43m={g_f[win][idx_max]:.5f}, G_P421m={g_cmp[win][idx_max]:.5f})")
        # Print top 3 positive ΔG peaks across full spectrum
        pos_peaks = []
        for i in range(1, len(diff) - 1):
            if diff[i] > diff[i - 1] and diff[i] > diff[i + 1] and diff[i] > 0.002:
                pos_peaks.append((diff[i], e_common[i]))
        pos_peaks.sort(reverse=True)
        print("  Top positive ΔG peaks (full range):", [(round(e, 2), round(d, 5)) for d, e in pos_peaks[:5]])


    print("\n=== Energy-by-energy GDOS comparison in [10.0, 17.0] meV ===")
    print("  E(meV) | G(F-43m) | G(P-421m RMC) | ΔG(RMC)  | G(P-421m 0.06Å) | ΔG(0.06Å)")
    for e_val in np.arange(10.0, 17.1, 0.5):
        i = int(np.argmin(np.abs(e_common - e_val)))
        print(
            f"  {e_common[i]:5.1f}  | {g_f[i]:8.5f} | {g_rmc[i]:13.5f} | {g_rmc[i]-g_f[i]:+8.5f} | {g_amp[i]:15.5f} | {g_amp[i]-g_f[i]:+8.5f}"
        )


if __name__ == "__main__":
    main()
