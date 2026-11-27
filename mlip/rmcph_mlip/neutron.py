"""Inelastic neutron scattering (INS) physics for crystalline & disordered phonons.

Physical Overview & Equations
=============================
This module computes both 1D neutron-weighted phonon densities of states (GDOS) and
2D powder-averaged dynamical structure factors S(|Q|, E) with three physical ingredients
required to compare directly against direct-geometry time-of-flight (TOF) chopper data:

1. Coherent + Incoherent One-Phonon Powder S_1(|Q|, E)
------------------------------------------------------
For total momentum transfer Q = q_cart + G_cart (where q is in the first Brillouin zone
and G is a reciprocal lattice vector) and phonon branch nu with energy E_{q,nu} > 0:

  * Phonopy constructs the mass-weighted dynamical matrix in the C-type (atomic-position)
    Bloch gauge:
        D_{ij}(q) = (M_i M_j)^(-1/2) * sum_R Phi_{0i, Rj} * exp(+i 2*pi q . (R + x_j - x_i))
    Because C-type eigenvectors e_{j,nu}^C(q) already absorb exp(-i 2*pi q . x_j), the
    q . x_j phase cancels in exp(-i Q . r_j), leaving ONLY the reciprocal-lattice phase
    exp(-i 2*pi G . x_j) across different Brillouin zones G!

  * Coherent one-phonon cross-section weight at Q = q_cart + G_cart [barn * meV^-1]:
        w_coh(Q, nu) = (hbar^2 / 2 m_u) * ((n(E, T) + 1) / E)
                       * | sum_j (beta_coh,j / sqrt(M_j)) * exp(-0.5 * |Q|^2 * U_iso,j(T))
                           * (Q . e_{j,nu}^C(q)) * exp(-i 2*pi G . x_j) |^2
    where beta_coh,j = sqrt(4*pi) * (b_coh,j [fm] / 10) [sqrt(barn)] preserves the sign
    of negative scattering lengths (e.g. H, Li, Ti, V, Mn), and beta_coh,j^2 = sigma_coh,j.

  * Spherical |Q|-shell powder averaging over the uniform q + G reciprocal grid divides
    the shell-summed coherent intensity by the shell point count and by N_atoms, yielding
    S_coh,1(|Q|, E) in [barn / meV / atom]. At high |Q| (many Brillouin zones), off-diagonal
    j != j' interference averages out and S_coh,1(|Q|, E) smoothly approaches the coherent
    part of the incoherent approximation.

  * Nuclear spin/isotope incoherent one-phonon term [barn / meV / atom]:
        S_inc,1(|Q|, E) = (1 / N_atoms) * sum_j sigma_inc,j * exp(-|Q|^2 * U_iso,j(T))
                          * (hbar^2 |Q|^2 / (2 m_u M_j)) * (g_j(E) / E) * (n(E, T) + 1)
    where sigma_inc,j = max(0, sigma_tot,j - sigma_coh,j) and int_0^inf g_j(E) dE = 1.

2. Parameter-Free Sjolander Multiphonon (n >= 2) Expansion
----------------------------------------------------------
In the harmonic Gaussian (Sjolander 1958; Sears 1984) expansion, the intermediate
scattering function I_j(Q, t) = sigma_tot,j * exp(-Q^2 U_iso,j) * exp(Q^2 gamma_j(t))
has time-zero autocorrelation gamma_j(0) = U_iso,j(T). Defining the unit-normalized
two-sided single-phonon thermal spectrum on (-inf, +inf):

    f_{1,j}(+E) = (hbar^2 / (2 m_u M_j U_iso,j(T))) * (g_j(E) / E) * (n(E, T) + 1)   (E > 0)
    f_{1,j}(-E) = (hbar^2 / (2 m_u M_j U_iso,j(T))) * (g_j(E) / E) * n(E, T)          (E > 0)

guarantees int_{-inf}^{+inf} f_{1,j}(E) dE = 1 identically (satisfying Placzek's 0th and
1st moment sum rules). Higher phonon orders n = 2, 3, ..., N_max are obtained without
free parameters by repeated convolution f_{n,j} = f_{1,j} * f_{n-1,j}:

    S_multi(|Q|, E) = (1 / N_atoms) * sum_j sigma_tot,j * exp(-|Q|^2 U_iso,j(T))
                      * sum_{n=2}^{N_max} [ (|Q|^2 U_iso,j(T))^n / n! ] * f_{n,j}(E).

3. Direct-Geometry Chopper Instrument Energy Resolution DeltaE(E; Ei)
---------------------------------------------------------------------
On direct-geometry TOF chopper spectrometers (SEQUOIA, CNCS, ARCS at SNS; LET at ISIS),
scattered neutrons have final energy E_f = E_i - E, so secondary flight-time uncertainty
scales via the TOF Jacobian |dE_f / dt_2| propto E_f^(3/2) = E_i^(3/2) * (1 - E / E_i)^(3/2):

    FWHM(E; E_i) = FWHM_el(E_i) * sqrt( c_0 + c_1 (1 - E/E_i)^(3/2) + c_2 (1 - E/E_i)^3 )

with c_0 + c_1 + c_2 = 1. Resolution is broadest near the elastic line (E = 0) and
sharpens substantially as E -> E_i (E_f -> 0). Source-centered ("push") variable-width
Gaussian broadening preserves spectral weight int S(Q, E) dE to machine precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from phonopy import Phonopy

THZ_TO_MEV = 4.135667696
KB_MEV_PER_K = 0.08617333262
# hbar^2 / (2 m_n) in meV * A^2 (neutron kinematics k -> E)
HBAR2_OVER_2MN = 2.0721248
# hbar^2 / (2 m_u) in meV * A^2 * amu (atomic recoil energy hbar^2 Q^2 / 2 M)
HBAR2_OVER_2MU = 2.0900795

# NIST natural-abundance total bound neutron scattering cross-sections [barn]
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

# NIST bound coherent neutron scattering lengths b_coh [fm] (1 fm = 10^-15 m; 100 fm^2 = 1 barn)
# Matches web/src/constants.js B_COH (retaining negative signs for H, Li, Ti, V, Mn)
NEUTRON_COH_B_FM: dict[str, float] = {
    "H": -3.739, "D": 6.671, "He": 3.26, "Li": -1.90, "Be": 7.79, "B": 5.30, "C": 6.646, "N": 9.36, "O": 5.803,
    "F": 5.654, "Ne": 4.566, "Na": 3.63, "Mg": 5.375, "Al": 3.449, "Si": 4.1491, "P": 5.13, "S": 2.847, "Cl": 9.577,
    "Ar": 1.909, "K": 3.67, "Ca": 4.70, "Sc": 12.29, "Ti": -3.370, "V": -0.3824, "Cr": 3.635, "Mn": -3.73,
    "Fe": 9.45, "Co": 2.49, "Ni": 10.3, "Cu": 7.718, "Zn": 5.680, "Ga": 7.288, "Ge": 8.185,
    "As": 6.58, "Se": 7.970, "Br": 6.795, "Kr": 7.81, "Rb": 7.09, "Sr": 7.02, "Y": 7.75, "Zr": 7.16, "Nb": 7.054,
    "Mo": 6.715, "Tc": 6.80, "Ru": 7.03, "Rh": 5.88, "Pd": 5.91, "Ag": 5.922, "Cd": 4.87, "In": 4.065,
    "Sn": 6.225, "Sb": 5.57, "Te": 5.80, "I": 5.28, "Xe": 4.92, "Cs": 5.42, "Ba": 5.07, "La": 8.24, "Ce": 4.84,
    "Pr": 4.58, "Nd": 7.69, "Pm": 12.6, "Sm": 0.80, "Eu": 7.22, "Gd": 6.5, "Tb": 7.38, "Dy": 16.9, "Ho": 8.01,
    "Er": 7.79, "Tm": 7.07, "Yb": 12.43, "Lu": 7.21, "Hf": 7.77, "Ta": 6.91, "W": 4.86, "Re": 9.2,
    "Os": 10.7, "Ir": 10.6, "Pt": 9.60, "Au": 7.63, "Hg": 12.692, "Tl": 8.776, "Pb": 9.405, "Bi": 8.532,
    "Th": 10.31, "U": 8.417,
}

# Analytic direct-geometry TOF chopper energy-resolution presets:
#   FWHM(E; Ei) = (elastic_frac * Ei) * sqrt(c0 + c1*(1 - E/Ei)^1.5 + c2*(1 - E/Ei)^3)
# References:
#   - SEQUOIA: Granroth et al., J. Phys.: Conf. Ser. 251, 012058 (2010)
#   - CNCS:    Ehlers et al., Rev. Sci. Instrum. 82, 085108 (2011)
#   - ARCS:    Abernathy et al., Nucl. Instrum. Methods A 680, 5 (2012)
#   - LET:     Bewley et al., Nucl. Instrum. Methods A 637, 128 (2011)
CHOPPER_INSTRUMENTS: dict[str, dict[str, float]] = {
    "SEQUOIA": {"elastic_frac": 0.018, "c0": 0.18, "c1": 0.34, "c2": 0.48},
    "CNCS":    {"elastic_frac": 0.022, "c0": 0.16, "c1": 0.32, "c2": 0.52},
    "ARCS":    {"elastic_frac": 0.030, "c0": 0.20, "c1": 0.35, "c2": 0.45},
    "LET":     {"elastic_frac": 0.015, "c0": 0.14, "c1": 0.30, "c2": 0.56},
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
    q_grid: np.ndarray                        # |Q| in A^-1, shape (nq,)
    e_grid: np.ndarray                        # Energy transfer in meV, shape (ne,)
    sqe: np.ndarray                           # Total S(|Q|, E) normalized to max=1, shape (ne, nq)
    temperature_k: float
    ei_mev: float | None
    fwhm_mev: float
    u_iso_by_element: dict[str, float]
    sqe_coherent: np.ndarray | None = None    # Coherent 1-phonon component (same scale as sqe)
    sqe_incoherent: np.ndarray | None = None  # Nuclear incoherent 1-phonon component
    sqe_multiphonon: np.ndarray | None = None # Sjolander multiphonon (n >= 2) component
    instrument: str = "SEQUOIA"
    fwhm_vs_e: np.ndarray | None = None       # Instrument FWHM(E; Ei) in meV along e_grid


def coherent_and_incoherent_cross_sections(symbol: str) -> tuple[float, float, float]:
    """Return (beta_coh_sqrt_barn, sigma_coh_barn, sigma_inc_barn) for an element symbol.

    * beta_coh = sqrt(4*pi) * (b_coh [fm] / 10)  [sqrt(barn), signed!]
    * sigma_coh = beta_coh^2                     [barn]
    * sigma_inc = max(0, sigma_tot - sigma_coh)  [barn]
    """
    sigma_tot = NEUTRON_SCATT_SIGMA.get(symbol, 5.0)
    if symbol in NEUTRON_COH_B_FM:
        b_fm = NEUTRON_COH_B_FM[symbol]
        beta_coh = float(np.sqrt(4.0 * np.pi) * (b_fm / 10.0))
        sigma_coh = beta_coh * beta_coh
        # Guard tiny roundoff differences in tabulated natural-abundance data
        if sigma_coh > sigma_tot:
            sigma_tot = sigma_coh
        sigma_inc = max(0.0, sigma_tot - sigma_coh)
    else:
        sigma_coh = sigma_tot
        beta_coh = float(np.sqrt(sigma_coh))
        sigma_inc = 0.0
    return beta_coh, sigma_coh, sigma_inc


def chopper_resolution_fwhm(
    e_grid_mev: np.ndarray,
    ei_mev: float | None,
    instrument: str = "SEQUOIA",
    fallback_fwhm_mev: float = 0.8,
) -> np.ndarray:
    """Compute energy-dependent FWHM(E; Ei) [meV] for a direct-geometry chopper spectrometer.

    When `instrument == 'constant'` or `ei_mev` is None / <= 0, returns a constant array
    filled with `fallback_fwhm_mev`.
    """
    e_arr = np.asarray(e_grid_mev, dtype=float)
    key = (instrument or "constant").upper()
    if key == "CONSTANT" or ei_mev is None or ei_mev <= 0 or key not in CHOPPER_INSTRUMENTS:
        return np.full_like(e_arr, max(float(fallback_fwhm_mev), 0.05), dtype=float)

    spec = CHOPPER_INSTRUMENTS[key]
    fwhm_elastic = max(0.05, spec["elastic_frac"] * float(ei_mev))
    # Reduced final energy E_f / E_i = 1 - E / E_i, clipped >= 0 above kinematic cutoff
    ef_over_ei = np.clip(1.0 - e_arr / float(ei_mev), 0.0, 2.5)
    shape = np.sqrt(
        spec["c0"]
        + spec["c1"] * np.power(ef_over_ei, 1.5)
        + spec["c2"] * np.power(ef_over_ei, 3.0)
    )
    return fwhm_elastic * shape


def gaussian_broaden(x: np.ndarray, y: np.ndarray, fwhm: float) -> np.ndarray:
    """Convolve a uniformly spaced 1D spectrum y(x) with a constant-FWHM Gaussian."""
    if fwhm <= 0 or len(x) < 3:
        return y.copy()
    dx = float(x[1] - x[0])
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    half = max(3, int(np.ceil(4.0 * sigma / dx)))
    kx = np.arange(-half, half + 1) * dx
    kernel = np.exp(-0.5 * (kx / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(y, kernel, mode="same")


def variable_gaussian_broaden(
    x: np.ndarray,
    y: np.ndarray,
    fwhm_vs_x: np.ndarray,
) -> np.ndarray:
    """Broaden 1D (ne,) or 2D (ne, nq) array along axis 0 with energy-dependent FWHM(x).

    Uses source-centered ("push") Gaussian deposition: each source energy bin i emits a
    unit-normalized Gaussian instrument response of width sigma(x[i]). This matches the
    physical spectrometer response to a delta-function mode at x[i] and conserves total
    spectral weight sum_j y_out[j] == sum_i y[i] to machine precision.
    """
    n = len(x)
    if n < 3:
        return y.copy()
    dx = abs(float(x[1] - x[0]))
    if dx <= 0:
        return y.copy()

    sigmas = np.asarray(fwhm_vs_x, dtype=float) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    out = np.zeros_like(y, dtype=float)
    sqrt_2ln2 = 1.0  # unused; sigmas already converted from FWHM

    for i in range(n):
        sig = float(sigmas[i]) * sqrt_2ln2
        if sig < 0.25 * dx:
            out[i] += y[i]
            continue
        half = min(n - 1, max(2, int(np.ceil(4.0 * sig / dx))))
        j_lo = max(0, i - half)
        j_hi = min(n - 1, i + half)
        jx = (np.arange(j_lo, j_hi + 1) - i) * dx
        k = np.exp(-0.5 * (jx / sig) ** 2)
        k_sum = float(k.sum())
        if k_sum <= 0:
            out[i] += y[i]
            continue
        k /= k_sum
        if y.ndim == 1:
            out[j_lo : j_hi + 1] += float(y[i]) * k
        else:
            out[j_lo : j_hi + 1, :] += k[:, None] * y[i][None, :]
    return out


def compute_sjolander_multiphonon(
    e_grid: np.ndarray,
    unit_dos_by_el: dict[str, np.ndarray],
    q_grid: np.ndarray,
    u_iso_by_el: dict[str, float],
    mass_by_el: dict[str, float],
    count_by_el: dict[str, int],
    temperature_k: float,
    max_order: int = 4,
) -> np.ndarray:
    """Compute Sjolander multiphonon S_multi(|Q|, E) [barn / meV / atom] for orders n = 2..max_order.

    Constructs the exact two-sided thermal single-phonon profile f_1(E) on a symmetric
    energy grid [-E_max, +E_max] covering both phonon creation (E > 0) and annihilation
    (E < 0), normalized so int_{-inf}^{+inf} f_1(E) dE = 1, and evaluates n-fold
    convolutions f_n = f_1 * f_{n-1} weighted by (Q^2 U_iso)^n / n! * exp(-Q^2 U_iso).
    """
    if max_order < 2 or len(e_grid) < 3:
        return np.zeros((len(e_grid), len(q_grid)), dtype=float)

    t_eval = max(float(temperature_k), 1.0)
    e_max_pos = float(e_grid[-1])
    # Uniform positive/negative grid with spacing matched to e_grid for clean FFT/direct conv
    de = max(float(e_grid[1] - e_grid[0]), 0.05)
    n_half = max(32, int(np.ceil((1.25 * e_max_pos) / de)))
    sym_e = np.arange(-n_half, n_half + 1, dtype=float) * de  # length 2*n_half + 1, centered at 0

    n_atoms_total = max(1, sum(count_by_el.values()))
    q2 = np.asarray(q_grid, dtype=float)[None, :] ** 2
    sqe_multi = np.zeros((len(e_grid), len(q_grid)), dtype=float)

    pos_mask = sym_e > max(0.35, 1.5 * de)
    neg_mask = sym_e < -max(0.35, 1.5 * de)
    e_pos = sym_e[pos_mask]
    e_neg_abs = -sym_e[neg_mask]

    x_pos = np.clip(e_pos / (KB_MEV_PER_K * t_eval), 1e-6, 500.0)
    n_pos_p1 = 1.0 / (1.0 - np.exp(-x_pos))

    x_neg = np.clip(e_neg_abs / (KB_MEV_PER_K * t_eval), 1e-6, 500.0)
    n_neg = 1.0 / np.expm1(x_neg)

    for el, g_unit_on_ndos in unit_dos_by_el.items():
        u_iso = max(float(u_iso_by_el.get(el, 0.004)), 1e-5)
        mass = max(float(mass_by_el.get(el, 50.0)), 1.0)
        sigma_tot = NEUTRON_SCATT_SIGMA.get(el, 5.0)
        frac_atoms = float(count_by_el.get(el, 1)) / float(n_atoms_total)

        # Interpolate element's unit-normalized 1-phonon DOS onto sym_e (> 0)
        # Caller passes (ndos_energy, g_unit) tuple or array aligned with e_grid
        f1 = np.zeros_like(sym_e)
        g_pos = np.interp(e_pos, g_unit_on_ndos[0], g_unit_on_ndos[1], left=0.0, right=0.0)
        g_neg = np.interp(e_neg_abs, g_unit_on_ndos[0], g_unit_on_ndos[1], left=0.0, right=0.0)

        pref = HBAR2_OVER_2MU / (mass * u_iso)
        f1[pos_mask] = pref * (np.maximum(g_pos, 0.0) / e_pos) * n_pos_p1
        f1[neg_mask] = pref * (np.maximum(g_neg, 0.0) / e_neg_abs) * n_neg

        area_f1 = float(np.trapezoid(f1, sym_e))
        if area_f1 <= 1e-12:
            continue
        # Enforce exact Placzek 0th-moment normalization int_{-inf}^{+inf} f_1(E) dE == 1
        f1 /= area_f1

        dw = np.exp(-q2 * u_iso)
        two_w = q2 * u_iso  # 2W_j(Q) = Q^2 * U_iso,j(T)

        fn = f1.copy()
        fact = 1.0
        for order in range(2, max_order + 1):
            fact *= float(order)
            # Continuous convolution (f_1 * f_{n-1})(E) = sum f_1 * f_{n-1} * dE, cropped to sym_e
            fn_full = np.convolve(fn, f1, mode="full") * de
            fn = fn_full[n_half : n_half + len(sym_e)]
            fn_on_e = np.interp(e_grid, sym_e, fn, left=0.0, right=0.0)
            weight_q = dw * (np.power(two_w, order) / fact)  # (1, nq)
            sqe_multi += frac_atoms * sigma_tot * fn_on_e[:, None] * weight_q

    return sqe_multi


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


def _extract_mesh_arrays(ph: Phonopy) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Retrieve (qpoints_frac, freqs_mev, eigvecs, weights) from Phonopy's active mesh."""
    if getattr(ph, "mesh", None) is None:
        ph.run_mesh([11, 11, 11], with_eigenvectors=True, is_mesh_symmetry=False)
    if getattr(ph, "mesh", None) is not None and getattr(ph.mesh, "qpoints", None) is not None:
        qpts = np.asarray(ph.mesh.qpoints, dtype=float)
        freqs_mev = np.asarray(ph.mesh.frequencies, dtype=float) * THZ_TO_MEV
        eigvecs = np.asarray(ph.mesh.eigenvectors, dtype=np.complex128)
        weights = np.asarray(ph.mesh.weights, dtype=float)
    else:
        md = ph.get_mesh_dict()
        qpts = np.asarray(md["qpoints"], dtype=float)
        freqs_mev = np.asarray(md["frequencies"], dtype=float) * THZ_TO_MEV
        eigvecs = np.asarray(md["eigenvectors"], dtype=np.complex128)
        weights = np.asarray(md["weights"], dtype=float)
    return qpts, freqs_mev, eigvecs, weights


def _enumerate_reciprocal_lattice_vectors(
    b_cart: np.ndarray,
    q_reach: float,
    max_shell: int = 18,
) -> tuple[np.ndarray, np.ndarray]:
    """Return integer Miller indices G_hkl (nG, 3) and Cartesian vectors G_cart (nG, 3) within q_reach."""
    b_norms = np.linalg.norm(b_cart, axis=1)
    n_lim = [min(max_shell, int(np.ceil(q_reach / max(bn, 1e-6))) + 1) for bn in b_norms]
    r0 = np.arange(-n_lim[0], n_lim[0] + 1, dtype=int)
    r1 = np.arange(-n_lim[1], n_lim[1] + 1, dtype=int)
    r2 = np.arange(-n_lim[2], n_lim[2] + 1, dtype=int)
    g0, g1, g2 = np.meshgrid(r0, r1, r2, indexing="ij")
    g_hkl = np.stack([g0.ravel(), g1.ravel(), g2.ravel()], axis=1)
    g_cart = g_hkl @ b_cart
    keep = np.linalg.norm(g_cart, axis=1) <= q_reach
    return g_hkl[keep], g_cart[keep]


def compute_powder_ins(
    ph: Phonopy,
    ndos: NeutronDosResult,
    temperature_k: float = 5.0,
    q_max: float = 8.0,
    n_q: int = 160,
    ei_mev: float | None = 50.0,
    instrument: str = "SEQUOIA",
    include_coherent: bool = True,
    include_multiphonon: bool = True,
    max_multiphonon_order: int = 4,
) -> PowderInsResult:
    """Compute powder-averaged INS S(|Q|, E) combining coherent 1-phonon, incoherent 1-phonon,
    Sjolander multiphonon (n >= 2), and direct-geometry chopper resolution DeltaE(E; Ei).
    """
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
    u_iso_atoms = np.clip(msd.mean(axis=1), 1e-5, 0.25)

    prim = ph.primitive
    symbols = list(prim.symbols)
    masses = np.asarray(prim.masses, dtype=float)
    pos_frac = np.asarray(prim.scaled_positions, dtype=float)
    n_atoms = len(symbols)

    unique_els = list(dict.fromkeys(symbols))
    u_iso_by_el: dict[str, float] = {}
    mass_by_el: dict[str, float] = {}
    count_by_el: dict[str, int] = {}
    for el in unique_els:
        idx_el = [i for i, s in enumerate(symbols) if s == el]
        u_iso_by_el[el] = float(np.mean(u_iso_atoms[idx_el])) if idx_el else 0.005
        mass_by_el[el] = float(np.mean(masses[idx_el])) if idx_el else 50.0
        count_by_el[el] = len(idx_el)

    q_grid = np.linspace(0.05, q_max, n_q)
    dq = float(q_grid[1] - q_grid[0]) if n_q > 1 else q_max
    e_mask = ndos.energy_mev > 0.3  # exclude elastic line E ~ 0
    e_grid = ndos.energy_mev[e_mask]
    n_e = len(e_grid)
    de = float(e_grid[1] - e_grid[0]) if n_e > 1 else 0.1
    e_lo = float(e_grid[0])

    # Instrument resolution FWHM(E; Ei) along e_grid
    fwhm_vs_e = chopper_resolution_fwhm(
        e_grid,
        ei_mev=ei_mev,
        instrument=instrument,
        fallback_fwhm_mev=ndos.fwhm_mev,
    )

    # ── 1. Coherent + Incoherent One-Phonon Scattering from Phonopy Mesh ──────
    qpts_frac, freqs_mev, eigvecs, q_weights = _extract_mesh_arrays(ph)

    beta_atoms = np.zeros(n_atoms, dtype=float)
    sigma_coh_atoms = np.zeros(n_atoms, dtype=float)
    sigma_inc_atoms = np.zeros(n_atoms, dtype=float)
    for j, sym in enumerate(symbols):
        beta_j, scoh_j, sinc_j = coherent_and_incoherent_cross_sections(sym)
        beta_atoms[j] = beta_j
        sigma_coh_atoms[j] = scoh_j
        sigma_inc_atoms[j] = sinc_j

    # Extract unbroadened unit-area partial DOS g_j(E) from mesh for exact incoherent & multiphonon
    ev_sq_all = np.abs(eigvecs) ** 2
    atom_proj_all = ev_sq_all.reshape(len(q_weights), n_atoms, 3, -1).sum(axis=2)  # (nq_mesh, n_atoms, 3N)

    # Histogram raw (unbroadened) incoherent 1-phonon energy profile per atom onto e_grid
    inc_atom_hist = np.zeros((n_atoms, n_e), dtype=float)
    el_raw_dos: dict[str, np.ndarray] = {el: np.zeros(len(ndos.energy_mev), dtype=float) for el in unique_els}
    de_ndos = float(ndos.energy_mev[1] - ndos.energy_mev[0])
    e0_ndos = float(ndos.energy_mev[0])

    for qi in range(len(q_weights)):
        wq = float(q_weights[qi])
        fr = freqs_mev[qi]
        # Full ndos grid histogram for Sjolander multiphonon unit DOS
        b_nd = np.floor((fr - e0_ndos) / de_ndos).astype(int)
        v_nd = (b_nd >= 0) & (b_nd < len(ndos.energy_mev))
        if np.any(v_nd):
            for j, sym in enumerate(symbols):
                np.add.at(el_raw_dos[sym], b_nd[v_nd], wq * atom_proj_all[qi, j, v_nd])

        # Positive e_grid histogram for 1-phonon scattering
        b_e = np.floor((fr - e_lo) / de).astype(int)
        v_e = (b_e >= 0) & (b_e < n_e) & (fr > 0.3)
        if np.any(v_e):
            x_b = np.clip(fr[v_e] / (KB_MEV_PER_K * t_eval), 1e-6, 500.0)
            occ_over_e = (1.0 / (1.0 - np.exp(-x_b))) / fr[v_e]
            for j in range(n_atoms):
                np.add.at(inc_atom_hist[j], b_e[v_e], wq * atom_proj_all[qi, j, v_e] * occ_over_e)

    # Normalize raw 1-phonon DOS integrals: sum_{q,nu} w_q |e_j|^2 = 3 * sum_q w_q
    w_q_total = float(np.sum(q_weights))
    # Divide by (3 * w_q_total * de) so int g_j(E) dE == 1
    inc_atom_hist /= max(3.0 * w_q_total * de, 1e-12)

    q2 = q_grid[None, :] ** 2
    sqe_inc_unbroad = np.zeros((n_e, n_q), dtype=float)
    sqe_coh_inc_approx_unbroad = np.zeros((n_e, n_q), dtype=float)

    for j in range(n_atoms):
        dw_j = np.exp(-q2 * u_iso_atoms[j])
        recoil_j = HBAR2_OVER_2MU * q2 / masses[j]
        base_j = (inc_atom_hist[j][:, None] * recoil_j * dw_j) / float(n_atoms)
        sqe_inc_unbroad += sigma_inc_atoms[j] * base_j
        sqe_coh_inc_approx_unbroad += sigma_coh_atoms[j] * base_j

    if include_coherent:
        # Reciprocal basis rows B_cart = 2*pi * inv(A_prim)^T  [A^-1]
        b_cart = 2.0 * np.pi * np.linalg.inv(np.asarray(prim.cell, dtype=float)).T
        b_min = float(np.min(np.linalg.norm(b_cart, axis=1)))
        # Match radial Q-smoothing to the 3D q-mesh spacing b_min / N_mesh (keeps Bragg cones crisp!)
        eff_n_linear = max(4.0, float(w_q_total) ** (1.0 / 3.0))
        sig_q = max(1.25 * dq, 0.65 * b_min / eff_n_linear)
        g_hkl, g_cart = _enumerate_reciprocal_lattice_vectors(b_cart, q_reach=q_max + 3.5 * sig_q)
        n_g = len(g_hkl)

        # Precompute q-independent C-type reciprocal-lattice phase:
        #   phase_g_atom[g, j] = (beta_coh,j / sqrt(M_j)) * exp(-i 2*pi G . x_j)
        beta_over_sqrt_m = beta_atoms / np.sqrt(masses)
        phase_g_atom = beta_over_sqrt_m[None, :] * np.exp(-2j * np.pi * (g_hkl @ pos_frac.T))  # (nG, n_atoms)

        # Subsample q-points smoothly only if (nq_mesh * n_g) is very large (dense G-grid in big cells)
        nq_mesh = len(q_weights)
        stride = max(1, int(np.ceil((nq_mesh * max(n_g, 1)) / 1_600_000)))
        q_indices = np.arange(0, nq_mesh, stride)

        sqe_coh_num = np.zeros((n_e, n_q), dtype=float)
        shell_norm = np.zeros(n_q, dtype=float)
        inv_2sig_q2 = 0.5 / (sig_q * sig_q)
        u_half = -0.5 * u_iso_atoms[None, :]  # (1, n_atoms)

        for qi in q_indices:
            wq = float(q_weights[qi])
            fr = freqs_mev[qi]
            b_e = np.floor((fr - e_lo) / de).astype(int)
            v_e = (b_e >= 0) & (b_e < n_e) & (fr > 0.3)
            if not np.any(v_e):
                continue

            q_cart = qpts_frac[qi] @ b_cart
            big_q_vec = q_cart[None, :] + g_cart                 # (nG, 3)
            big_q_mag = np.linalg.norm(big_q_vec, axis=1)        # (nG,)
            in_shell = (big_q_mag > 1e-6) & (big_q_mag <= q_max + 3.0 * sig_q)
            if not np.any(in_shell):
                continue

            qv = big_q_vec[in_shell]                             # (nG_s, 3)
            qm = big_q_mag[in_shell]                             # (nG_s,)
            qm2 = qm * qm

            # Radial Gaussian shell kernel K_q[iq, g] = exp(-0.5 * ((q_grid[iq] - |Q_g|) / sig_q)^2)
            dq_mat = q_grid[:, None] - qm[None, :]               # (n_q, nG_s)
            k_q = np.exp(-(dq_mat * dq_mat) * inv_2sig_q2)       # (n_q, nG_s)
            shell_norm += wq * k_q.sum(axis=1)

            # Build W_g[g, 3*j + alpha] = phase_g_atom[g, j] * exp(-0.5 |Q|^2 U_iso,j) * Q_{g,alpha}
            dw_half = np.exp(qm2[:, None] * u_half)              # (nG_s, n_atoms)
            atom_amp = phase_g_atom[in_shell] * dw_half          # (nG_s, n_atoms) complex128
            w_g = (atom_amp[:, :, None] * qv[:, None, :]).reshape(len(qm), 3 * n_atoms)

            # Single BLAS zgemm for all active modes at q, then dgemm to bin onto q_grid!
            ev_active = eigvecs[qi][:, v_e]                      # (3*n_atoms, n_active)
            f_coh = w_g @ ev_active                              # (nG_s, n_active)
            f_coh_sq = f_coh.real ** 2 + f_coh.imag ** 2         # (nG_s, n_active)
            mode_q = k_q @ f_coh_sq                              # (n_q, n_active)

            fr_act = fr[v_e]
            x_b = np.clip(fr_act / (KB_MEV_PER_K * t_eval), 1e-6, 500.0)
            pref_mode = (wq * HBAR2_OVER_2MU / (float(n_atoms) * de)) * (
                1.0 / ((1.0 - np.exp(-x_b)) * fr_act)
            )
            weighted_modes = mode_q * pref_mode[None, :]         # (n_q, n_active)
            for idx_m, eb in enumerate(b_e[v_e]):
                sqe_coh_num[eb, :] += weighted_modes[:, idx_m]

        valid_shell = shell_norm > 1e-12
        sqe_coh_unbroad = np.zeros((n_e, n_q), dtype=float)
        sqe_coh_unbroad[:, valid_shell] = sqe_coh_num[:, valid_shell] / shell_norm[None, valid_shell]
    else:
        sqe_coh_unbroad = sqe_coh_inc_approx_unbroad

    # Broaden 1-phonon coherent and incoherent maps with chopper DeltaE(E; Ei)
    sqe_coh = variable_gaussian_broaden(e_grid, sqe_coh_unbroad, fwhm_vs_e)
    sqe_inc = variable_gaussian_broaden(e_grid, sqe_inc_unbroad, fwhm_vs_e)

    # ── 2. Parameter-Free Sjolander Multiphonon (n = 2 .. max_multiphonon_order) ──
    if include_multiphonon and max_multiphonon_order >= 2:
        unit_dos_by_el: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for el in unique_els:
            raw = el_raw_dos[el]
            area_el = float(np.trapezoid(raw, ndos.energy_mev))
            g_unit = raw / area_el if area_el > 1e-12 else np.zeros_like(raw)
            unit_dos_by_el[el] = (ndos.energy_mev, g_unit)

        sqe_multi_unbroad = compute_sjolander_multiphonon(
            e_grid=e_grid,
            unit_dos_by_el=unit_dos_by_el,
            q_grid=q_grid,
            u_iso_by_el=u_iso_by_el,
            mass_by_el=mass_by_el,
            count_by_el=count_by_el,
            temperature_k=t_eval,
            max_order=max_multiphonon_order,
        )
        sqe_multi = variable_gaussian_broaden(e_grid, sqe_multi_unbroad, fwhm_vs_e)
    else:
        sqe_multi = np.zeros((n_e, n_q), dtype=float)

    sqe_tot = sqe_coh + sqe_inc + sqe_multi

    # ── 3. Apply Direct-Geometry Kinematic Locus [|k_i - k_f|, k_i + k_f] ─────
    if ei_mev is not None and ei_mev > 0:
        ki = np.sqrt(ei_mev / HBAR2_OVER_2MN)
        for ie, en in enumerate(e_grid):
            ef = ei_mev - en
            if ef <= 0:
                sqe_tot[ie, :] = 0.0
                sqe_coh[ie, :] = 0.0
                sqe_inc[ie, :] = 0.0
                sqe_multi[ie, :] = 0.0
                continue
            kf = np.sqrt(ef / HBAR2_OVER_2MN)
            outside = (q_grid < abs(ki - kf)) | (q_grid > (ki + kf))
            sqe_tot[ie, outside] = 0.0
            sqe_coh[ie, outside] = 0.0
            sqe_inc[ie, outside] = 0.0
            sqe_multi[ie, outside] = 0.0

    # Normalize all channels by the same peak scalar so sqe == sqe_coh + sqe_inc + sqe_multi
    mx = float(sqe_tot.max())
    if mx > 0:
        sqe_tot /= mx
        sqe_coh /= mx
        sqe_inc /= mx
        sqe_multi /= mx

    return PowderInsResult(
        q_grid=q_grid,
        e_grid=e_grid,
        sqe=sqe_tot,
        temperature_k=temperature_k,
        ei_mev=ei_mev,
        fwhm_mev=float(fwhm_vs_e[0]) if len(fwhm_vs_e) else ndos.fwhm_mev,
        u_iso_by_element=u_iso_by_el,
        sqe_coherent=sqe_coh,
        sqe_incoherent=sqe_inc,
        sqe_multiphonon=sqe_multi,
        instrument=instrument,
        fwhm_vs_e=fwhm_vs_e,
    )
