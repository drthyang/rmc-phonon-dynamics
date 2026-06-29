import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
KB_J      = 1.380649e-23       # Boltzmann constant  [J/K]
KB_meV    = 8.617333262e-2     # Boltzmann constant  [meV/K]
HBAR_meVs = 6.582119569e-13    # Reduced Planck constant [meV·s]
AMU_TO_KG = 1.66053906660e-27  # Atomic mass unit    [kg/amu]
ANG2_TO_M2 = 1e-20             # Å² → m² conversion factor

# ── Derived energy conversion ─────────────────────────────────────────────────
# E [meV] = ENERGY_CONV * sqrt(T [K] / λ [amu·Å²])
# Derived from the classical equipartition theorem:
#   ω [rad/s] = sqrt(kb_J * T / (m_kg * <u_m²>))
#   E [meV]   = ℏ [meV·s] * ω
# where λ = eigenvalue of the mass-weighted displacement-correlation S(k) matrix,
# computed with mass in amu and displacements in Cartesian Å.
ENERGY_CONV = HBAR_meVs * np.sqrt(KB_J / (AMU_TO_KG * ANG2_TO_M2))
# ≈ 0.6001  meV · amu^(1/2) · Å / K^(1/2)

# ── Atomic masses [amu / g·mol⁻¹] ────────────────────────────────────────────
# Source: IUPAC 2021 standard atomic weights
ATOMIC_MASS = {
    'H':   1.008,   'He':  4.0026,  'Li':  6.94,    'Be':  9.0122,
    'B':  10.81,    'C':  12.011,   'N':  14.007,   'O':  15.999,
    'F':  18.998,   'Ne': 20.180,   'Na': 22.990,   'Mg': 24.305,
    'Al': 26.982,   'Si': 28.085,   'P':  30.974,   'S':  32.06,
    'Cl': 35.45,    'Ar': 39.948,   'K':  39.098,   'Ca': 40.078,
    'Sc': 44.956,   'Ti': 47.867,   'V':  50.942,   'Cr': 51.996,
    'Mn': 54.938,   'Fe': 55.845,   'Co': 58.933,   'Ni': 58.693,
    'Cu': 63.546,   'Zn': 65.38,    'Ga': 69.723,   'Ge': 72.63,
    'As': 74.922,   'Se': 78.96,    'Br': 79.904,   'Kr': 83.798,
    'Rb': 85.468,   'Sr': 87.62,    'Y':  88.906,   'Zr': 91.224,
    'Nb': 92.906,   'Mo': 95.96,    'Tc': 98.0,     'Ru':101.07,
    'Rh':102.91,    'Pd':106.42,    'Ag':107.87,    'Cd':112.41,
    'In':114.82,    'Sn':118.71,    'Sb':121.76,    'Te':127.60,
    'I': 126.90,    'Xe':131.29,    'Cs':132.91,    'Ba':137.33,
    'La':138.91,    'Ce':140.12,    'Pr':140.91,    'Nd':144.24,
    'Pm':145.0,     'Sm':150.36,    'Eu':151.96,    'Gd':157.25,
    'Tb':158.93,    'Dy':162.50,    'Ho':164.93,    'Er':167.26,
    'Tm':168.93,    'Yb':173.05,    'Lu':174.97,    'Hf':178.49,
    'Ta':180.95,    'W': 183.84,    'Re':186.21,    'Os':190.23,
    'Ir':192.22,    'Pt':195.08,    'Au':196.97,    'Hg':200.59,
    'Tl':204.38,    'Pb':207.2,     'Bi':208.98,    'Po':209.0,
    'At':210.0,     'Rn':222.0,     'Fr':223.0,     'Ra':226.0,
    'Ac':227.0,     'Th':232.04,    'Pa':231.04,    'U': 238.03,
    'Np':237.0,     'Pu':244.0,     'Am':243.0,     'Cm':247.0,
    'Bk':247.0,     'Cf':251.0,     'Es':252.0,     'Fm':257.0,
    'Md':258.0,     'No':259.0,     'Lr':262.0,
}

# ── Neutron total scattering cross-sections [barn] ────────────────────────────
# Source: NIST Center for Neutron Research (natural abundance)
NEUTRON_SCATT_SIGMA = {
    'H':  82.02,  'D':   7.64,  'He':  1.34,  'Li':  1.37,  'Be':  7.63,
    'B':   5.24,  'C':   5.551, 'N':  11.51,  'O':   4.232, 'F':   4.018,
    'Ne':  2.62,  'Na':  3.28,  'Mg':  3.71,  'Al':  1.503, 'Si':  2.167,
    'P':   3.31,  'S':   1.026, 'Cl': 16.8,   'K':   1.96,  'Ar':  0.68,
    'Ca':  2.83,  'Sc': 23.4,   'Ti':  4.35,  'V':   5.10,  'Cr':  3.49,
    'Mn':  2.15,  'Fe': 11.62,  'Co':  6.07,  'Ni': 18.5,   'Cu':  8.03,
    'Zn':  4.054, 'Ga':  6.83,  'Ge':  8.42,  'As':  5.48,  'Se':  8.30,
    'Br':  5.9,   'Kr':  7.66,  'Rb':  6.32,  'Sr':  6.25,  'Y':   7.76,
    'Zr':  6.46,  'Nb':  6.253, 'Mo':  5.71,  'Tc':  6.0,   'Ru':  5.1,
    'Rh':  4.81,  'Pd':  4.5,   'Ag':  4.99,  'Cd':  6.5,   'In':  2.61,
    'Sn':  4.89,  'Sb':  3.9,   'Te':  4.25,  'I':   3.55,  'Xe':  4.3,
    'Cs':  4.23,  'Ba':  3.38,  'La':  9.81,  'Ce':  2.94,  'Pr':  3.58,
    'Nd': 16.5,   'Pm': 16.0,   'Sm': 39.0,   'Eu':  9.2,   'Gd':175.0,
    'Tb': 23.0,   'Dy': 34.4,   'Ho':  8.8,   'Er':  8.0,   'Tm':  8.5,
    'Yb': 23.4,   'Lu':  5.9,   'Hf':  5.88,  'Ta':  6.01,  'W':   4.87,
    'Re': 13.5,   'Os': 12.6,   'Ir': 14.0,   'Pt': 11.71,  'Au':  7.63,
    'Hg': 26.5,   'Tl':  9.89,  'Pb': 11.11,  'Bi':  9.16,  'Th': 12.63,
    'U':  14.16,
}
