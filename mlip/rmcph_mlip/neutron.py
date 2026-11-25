"""Neutron-weighted phonon DOS (GDOS) and simulated powder-averaged INS S(|Q|, E)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from phonopy import Phonopy

THZ_TO_MEV = 4.135667696
KB_MEV_PER_K = 0.08617333262
# hbar^2 / (2 m_n) in meV * A^2
HBAR2_OVER_2MN = 2.0721248

# NIST natural-abundance total neutron scattering cross-sections [barn]
# Matches web/src/constants.js NEUTRON_SCATT_SIGMA
NEUTRON_SCATT_SIGMA: dict[str, float] = {
    "H": 82.02, "D": 7.64, "He": 1.34, "Li": 1.37, "Be": 7.63, "B": 5.24, "C": 5.551, "N": 11.51, "O": 4.232,
    "F": 4.018, "Ne": 2.62, "Na": 3.28, "Mg": 3.71, "Al": 1.503, "Si": 2.167, "P": 3.31, "S": 1.026, "Cl": 16.8,
    "K": 1.96, "Ar": 0.68, "Ca": 2.83, "Sc": 23.4, "Ti": 4.35, "V": 5.10, "Cr": 3.49, "Mn": 2.15, "Fe": 11.62,
    "Co": 6.07, "Ni": 18.5, "Cu": 8.03, "Zn": 4.054, "Ga": 6.83, "Ge": 8.42, "As": 5.48, "Se": 8.30, "Br": 5.9,
    "Kr": 7.66, "Rb": 6.32, "Sr": 6.25, "Y": 7.76, "Zr": 6.46, "Nb": 6.253, "Mo": 5.71, "Tc": 6.0, "Ru": 5.1,
    "Rh": 4.81, "Pd": 4.5, "Ag": 4.99, "Cd": 6.5, "In": 2.61, "Sn": 4.89, "Sb": 3.9, "Te": 4.25, "I": 3.55,
    "Xe": 4.3, "Cs": 4.23, "Ba": 3.38, "La": 9.81, "Ce": 2.94, "Pr": 3.58, "Nd": 16.5, "Pm": 16.0, "Sm": 39.0,
    "Eu": 9.2, "Gd": 175.0, "Tb": 23.0, "Dy": 34.4, "Ho": 8.8, "Er": 8.0, "Tm": 8.5, "Yb": 23.4, "Lu": 5.9,
    "Hf": 5.88, "Ta": 6.01, "W": 4.87, "Re": 13.5, "Os": 12.6, "Ir": 14.0, "Pt": 11.71, "Au": 7.63, "Hg": 26.5,
    "Tl": 9.89, "Pb": 11.11, "Bi": 9.16, "Th": 12.63, "U": 14.16,
}


@dataclass
class NeutronDosResult:
    energy_mev: np.ndarray
    total_dos: np.ndarray                 # unweighted normalized DOS
    neutron_gdos: np.ndarray              # neutron-weighted GDOS (sum_k (sigma_k/M_k) g_k(E)), normalized
    element_gdos: dict[str, np.ndarray]   # per-element contribution to neutron_gdos
    element_weights: dict[str, float]     # sigma / M [barn/amu] per element
    fwhm_mev: float


@dataclass
class PowderInsResult:
    q_grid: np.ndarray        # |Q| in A^-1, shape (nq,)
    e_grid: np.ndarray        # Energy transfer in meV, shape (ne,)
    sqe: np.ndarray           # S(|Q|, E), shape (ne, nq)
    temperature_k: float
    ei_mev: float | None
    fwhm_mev: float
    u_iso_by_element: dict[str, float]


def gaussian_broaden(x: np.ndarray, y: np.ndarray, fwhm: float) -> np.ndarray:
    """Convolve a uniformly spaced 1D spectrum y(x) with a Gaussian of given FWHM."""
    if fwhm <= 0 or len(x) < 3:
        return y.copy()
    dx = float(x[1] - x[0])
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    half = max(3, int(np.ceil(4.0 * sigma / dx)))
    kx = np.arange(-half, half + 1) * dx
    kernel = np.exp(-0.5 * (kx / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(y, kernel, mode="same")


def compute_neutron_dos(
    ph: Phonopy,
    mesh: tuple[int, int, int] = (15, 15, 15),
    fwhm_mev: float = 0.8,
    n_energy: int = 400,
) -> NeutronDosResult:
    """Compute total DOS and neutron-weighted generalized DOS (GDOS) from Phonopy mesh."""
    ph.run_mesh(list(mesh), with_eigenvectors=True, is_mesh_symmetry=False)
    if getattr(ph, "mesh", None) is not None:
        freqs_mev = np.asarray(ph.mesh.frequencies, dtype=float) * THZ_TO_MEV  # (nq, 3N)
        eigvecs = np.asarray(ph.mesh.eigenvectors)                              # (nq, 3N, 3N), col=mode
        weights = np.asarray(ph.mesh.weights, dtype=float)                      # (nq,)
    else:
        mesh_dict = ph.get_mesh_dict()
        freqs_mev = np.asarray(mesh_dict["frequencies"], dtype=float) * THZ_TO_MEV
        eigvecs = np.asarray(mesh_dict["eigenvectors"])
        weights = np.asarray(mesh_dict["weights"], dtype=float)

    prim = ph.primitive
    symbols = list(prim.symbols)
    masses = np.asarray(prim.masses, dtype=float)
    n_atoms = len(symbols)

    # Energy grid
    e_min = min(0.0, float(freqs_mev.min()) - 2.0 * fwhm_mev)
    e_max = float(freqs_mev.max()) + max(2.0, 3.0 * fwhm_mev)
    e_grid = np.linspace(e_min, e_max, n_energy)
    de = e_grid[1] - e_grid[0]

    # Per-atom projected weight |e_{q,mode,atom}|^2, shape (nq, n_atoms, 3N)
    # In phonopy get_mesh_dict(): eigvecs[q, 3*atom + alpha, mode]
    ev_sq = np.abs(eigvecs) ** 2
    atom_proj = ev_sq.reshape(len(weights), n_atoms, 3, -1).sum(axis=2)  # (nq, n_atoms, n_modes)

    # Histogram into energy bins
    unique_els = list(dict.fromkeys(symbols))
    el_partial: dict[str, np.ndarray] = {el: np.zeros(n_energy, dtype=float) for el in unique_els}
    total_hist = np.zeros(n_energy, dtype=float)

    bin_idx = np.floor((freqs_mev - e_min) / de).astype(int)
    valid = (bin_idx >= 0) & (bin_idx < n_energy)

    for qi in range(len(weights)):
        w_q = weights[qi]
        b_q = bin_idx[qi]
        v_q = valid[qi]
        if not np.any(v_q):
            continue
        np.add.at(total_hist, b_q[v_q], w_q)
        for a_idx, el in enumerate(symbols):
            np.add.at(el_partial[el], b_q[v_q], w_q * atom_proj[qi, a_idx, v_q])

    # Apply Gaussian resolution broadening
    total_dos = gaussian_broaden(e_grid, total_hist, fwhm_mev)
    area_tot = np.trapezoid(total_dos, e_grid)
    if area_tot > 0:
        total_dos /= area_tot

    el_weights: dict[str, float] = {}
    el_gdos: dict[str, np.ndarray] = {}
    gdos_sum = np.zeros(n_energy, dtype=float)

    for el in unique_els:
        sigma = NEUTRON_SCATT_SIGMA.get(el, 5.0)
        # Average mass for this element
        m_el = float(np.mean([masses[i] for i, s in enumerate(symbols) if s == el]))
        w_n = sigma / m_el
        el_weights[el] = w_n
        broad = gaussian_broaden(e_grid, el_partial[el], fwhm_mev) * w_n
        el_gdos[el] = broad
        gdos_sum += broad

    area_g = np.trapezoid(gdos_sum, e_grid)
    if area_g > 0:
        gdos_sum /= area_g
        for el in unique_els:
            el_gdos[el] /= area_g

    return NeutronDosResult(
        energy_mev=e_grid,
        total_dos=total_dos,
        neutron_gdos=gdos_sum,
        element_gdos=el_gdos,
        element_weights=el_weights,
        fwhm_mev=fwhm_mev,
    )


def compute_powder_ins(
    ph: Phonopy,
    ndos: NeutronDosResult,
    temperature_k: float = 5.0,
    q_max: float = 8.0,
    n_q: int = 160,
    ei_mev: float | None = 50.0,
) -> PowderInsResult:
    """Compute incoherent powder-averaged one-phonon INS S(|Q|, E) with thermal Debye-Waller factor.

    S_inc(Q, E) = sum_el [ (sigma_el / M_el) * g_el(E) * (Q^2 / E) * (n(E, T) + 1) * exp(-Q^2 * U_iso_el(T)) ]
    """
    # Compute mean-square thermal displacements U_iso(T) from Phonopy (excluding Gamma acoustic modes)
    t_eval = max(float(temperature_k), 1.0)
    ph.run_thermal_displacements(
        t_min=t_eval,
        t_max=t_eval,
        t_step=10.0,
        freq_min=0.05,
    )
    if getattr(ph, "thermal_displacements", None) is not None:
        msd = np.asarray(ph.thermal_displacements.thermal_displacements[0], dtype=float).reshape(-1, 3)
    else:
        td = ph.get_thermal_displacements_dict()
        msd = np.asarray(td["thermal_displacements"][0], dtype=float).reshape(-1, 3)
    u_iso_atoms = msd.mean(axis=1)

    symbols = list(ph.primitive.symbols)
    u_iso_by_el: dict[str, float] = {}
    for el in ndos.element_gdos:
        vals = [u_iso_atoms[i] for i, s in enumerate(symbols) if s == el]
        u_iso_by_el[el] = float(np.mean(vals)) if vals else 0.005

    q_grid = np.linspace(0.05, q_max, n_q)
    e_mask = ndos.energy_mev > 0.3  # exclude elastic line E ~ 0
    e_grid = ndos.energy_mev[e_mask]

    # Bose-Einstein factor n(E, T) + 1 for neutron energy loss (E > 0)
    x_bose = np.clip(e_grid / (KB_MEV_PER_K * t_eval), 1e-6, 500.0)
    bose_p1 = 1.0 / (1.0 - np.exp(-x_bose))

    # Build S(Q, E)
    sqe = np.zeros((len(e_grid), len(q_grid)), dtype=float)
    q2 = q_grid[None, :] ** 2
    pref_e = (bose_p1 / e_grid)[:, None]

    for el, g_el in ndos.element_gdos.items():
        u_iso = u_iso_by_el[el]
        dw = np.exp(-q2 * u_iso)
        sqe += g_el[e_mask, None] * pref_e * q2 * dw

    # Apply kinematic boundary if incident energy Ei is given
    if ei_mev is not None and ei_mev > 0:
        ki = np.sqrt(ei_mev / HBAR2_OVER_2MN)
        for ie, en in enumerate(e_grid):
            ef = ei_mev - en
            if ef <= 0:
                sqe[ie, :] = 0.0
                continue
            kf = np.sqrt(ef / HBAR2_OVER_2MN)
            q_lo = abs(ki - kf)
            q_hi = ki + kf
            sqe[ie, (q_grid < q_lo) | (q_grid > q_hi)] = 0.0

    # Normalize to max = 1 for display
    mx = float(sqe.max())
    if mx > 0:
        sqe /= mx

    return PowderInsResult(
        q_grid=q_grid,
        e_grid=e_grid,
        sqe=sqe,
        temperature_k=temperature_k,
        ei_mev=ei_mev,
        fwhm_mev=ndos.fwhm_mev,
        u_iso_by_element=u_iso_by_el,
    )
