#!/usr/bin/env python3
"""Test whether the 2π factor in the Bloch phase exp(2πi k·R) is correct.

Runs the band calculation along Γ→h00 with both conventions using a small
config subset, then plots them side-by-side and prints diagnostics.

Physical expectations for the CORRECT convention:
  1. Acoustic modes at Γ (k=[0,0,0]) should have the LOWEST energies
     (large eigenvalues → small E = ENERGY_CONV * sqrt(T/λ)).
  2. The dispersion should be smooth with no sharp oscillations between
     adjacent k-points.
  3. Zone-boundary modes at h00=[0.5,0,0] should be distinct from Γ,
     with optical modes above acoustic modes throughout the path.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import Readers
from constants import ENERGY_CONV, ATOMIC_MASS, NEUTRON_SCATT_SIGMA

# ── Parameters ────────────────────────────────────────────────────────────────
T          = 5       # K
N_CONFIGS  = 50      # use a subset for speed (full run takes much longer)
KSTEP      = 24      # k-points per segment
DATA       = '../data/'
FPATH_EQ   = DATA + '5K_ini/GTS_5K.rmc6f'
FPATH_CONFIGS = DATA + 'ensemble_20A_5K/configs/'

sym_pnts = {
    'GM':  np.array([0.0, 0.0, 0.0]),
    'h00': np.array([0.5, 0.0, 0.0]),
}
K_PATH = ['GM', 'h00']

# ── Shared helpers (duplicated here to test both phase conventions) ───────────

def get_mass_per_type(atom_dic):
    rev = {v: k for k, vals in atom_dic.items() for v in vals}
    return rev  # type_id -> element

def select_type(tag, atype, config, cell_idx):
    sel = [i for i, t in enumerate(atype) if t == tag]
    return config[sel], cell_idx[sel]

def build_Uk(kvec, atype, displacements_A, cell_idx, atom_dic, use_2pi):
    """Build mass-weighted U_k with or without 2π in the phase."""
    rev = {v: k for k, vals in atom_dic.items() for v in vals}
    unique_types = sorted(set(atype))
    phase_factor = 2 * np.pi if use_2pi else 1.0

    U_k_t = []
    for type_id in unique_types:
        disp, cells = select_type(type_id, atype, displacements_A, cell_idx)
        mass = ATOMIC_MASS[rev[type_id]]
        sqrt_m = np.sqrt(mass)
        phases = np.exp(1j * phase_factor * (cells @ kvec))
        weighted = sqrt_m * disp * phases[:, np.newaxis]
        U_k_t.append(weighted.sum(axis=0) / np.sqrt(len(cells)))

    return np.array(U_k_t).reshape(1, -1)

def calc_Sk(U):
    return U.T @ U.conj()

def ev_to_meV(eigenvalues, T):
    safe = np.where(eigenvalues != 0, np.abs(eigenvalues), np.nan)
    E = ENERGY_CONV * np.sqrt(T / safe)
    E = np.where(np.isnan(E), 0.0, E)
    return np.where(eigenvalues >= 0, E, -E)

# ── Load data ─────────────────────────────────────────────────────────────────
atom_dic = Readers.get_atom_idx(FPATH_EQ)
v1, v2, v3, dim = Readers.read_cell_vec(FPATH_EQ)
v_super = np.array([v1, v2, v3])

rmcfiles = sorted(glob.glob(FPATH_CONFIGS + 'Frac*.txt'))[:N_CONFIGS]
print(f'Using {len(rmcfiles)} / {len(glob.glob(FPATH_CONFIGS+"Frac*.txt"))} configs')

hsym = Readers.avg_frac_atom_ph(rmcfiles, atom_dic, dim)
hsym_xyz = hsym[1]  # (N, 3) average positions

# ── Compute bands for both conventions ────────────────────────────────────────
bands = {True: [], False: []}   # True = with 2π, False = without

for use_2pi in [True, False]:
    label = 'with 2π' if use_2pi else 'without 2π'
    print(f'\nComputing bands {label} ...')

    for ii in range(len(K_PATH) - 1):
        k_vec = sym_pnts[K_PATH[ii + 1]] - sym_pnts[K_PATH[ii]]
        for jj in range(KSTEP):
            kpnt = sym_pnts[K_PATH[ii]] + jj * k_vec / KSTEP

            Sk_sum = None
            for fname in rmcfiles:
                test = Readers.read_frac_atom_ph(fname, atom_dic, dim)
                atype, cfg, cell_idx = test
                displacements_A = ((cfg - hsym_xyz) / dim) @ v_super
                U_k = build_Uk(kpnt, atype, displacements_A, cell_idx, atom_dic, use_2pi)
                Sk = calc_Sk(U_k)
                Sk_sum = Sk if Sk_sum is None else Sk_sum + Sk

            eigenvalues, _ = np.linalg.eigh(Sk_sum / len(rmcfiles))
            bands[use_2pi].append(ev_to_meV(eigenvalues, T))

# ── Raw eigenvalue diagnostics ────────────────────────────────────────────────
# Recompute raw eigenvalues at Γ and h00 for direct inspection
print('\n── Raw S(k) eigenvalues at Γ (should be all ≥ 0 for PSD matrix) ──')
for use_2pi in [True, False]:
    label = 'with 2π   ' if use_2pi else 'without 2π'
    # Γ eigenvalues come from the first k-point (jj=0, ii=0)
    kpnt = sym_pnts['GM']
    Sk_sum = None
    for fname in rmcfiles:
        test = Readers.read_frac_atom_ph(fname, atom_dic, dim)
        atype, cfg, cell_idx = test
        displacements_A = ((cfg - hsym_xyz) / dim) @ v_super
        U_k = build_Uk(kpnt, atype, displacements_A, cell_idx, atom_dic, use_2pi)
        Sk = calc_Sk(U_k)
        Sk_sum = Sk if Sk_sum is None else Sk_sum + Sk
    raw_ev = np.linalg.eigh(Sk_sum / len(rmcfiles))[0]
    print(f'  {label}: min={raw_ev.min():.4e}  max={raw_ev.max():.4e}  all={np.round(raw_ev, 4)}')

print('\n── Raw S(k) eigenvalues at h00=[0.5,0,0] ──')
for use_2pi in [True, False]:
    label = 'with 2π   ' if use_2pi else 'without 2π'
    kpnt = sym_pnts['h00']
    Sk_sum = None
    for fname in rmcfiles:
        test = Readers.read_frac_atom_ph(fname, atom_dic, dim)
        atype, cfg, cell_idx = test
        displacements_A = ((cfg - hsym_xyz) / dim) @ v_super
        U_k = build_Uk(kpnt, atype, displacements_A, cell_idx, atom_dic, use_2pi)
        Sk = calc_Sk(U_k)
        Sk_sum = Sk if Sk_sum is None else Sk_sum + Sk
    raw_ev = np.linalg.eigh(Sk_sum / len(rmcfiles))[0]
    print(f'  {label}: min={raw_ev.min():.4e}  max={raw_ev.max():.4e}  all={np.round(raw_ev, 4)}')

# Filter bands: drop modes with |E| > 1000 meV (near-zero eigenvalue artifacts)
# and apply for smoothness and plotting
E_MAX = 200  # meV — reasonable upper bound for phonons in this material

def filter_bands(band_list):
    return [[e for e in sorted(b) if abs(e) < E_MAX] for b in band_list]

# Smoothness: mean abs difference between adjacent k-points (lower = smoother)
def smoothness(band_list):
    filtered = filter_bands(band_list)
    min_len = min(len(b) for b in filtered)
    if min_len == 0:
        return float('inf')
    arr = np.array([b[:min_len] for b in filtered])
    return np.mean(np.abs(np.diff(arr, axis=0)))

s_2pi   = smoothness(bands[True])
s_no2pi = smoothness(bands[False])
print(f'\n── Dispersion smoothness for |E|<{E_MAX} meV modes (lower = smoother) ──')
print(f'  with 2π   : {s_2pi:.4f} meV/step')
print(f'  without 2π: {s_no2pi:.4f} meV/step')

print(f'\n── Physical modes (|E| < {E_MAX} meV) at Γ ──')
for use_2pi in [True, False]:
    label = 'with 2π   ' if use_2pi else 'without 2π'
    physical = sorted([e for e in bands[use_2pi][0] if abs(e) < E_MAX])
    print(f'  {label}: {[round(e,2) for e in physical]}')

print(f'\n── Physical modes (|E| < {E_MAX} meV) at h00 ──')
for use_2pi in [True, False]:
    label = 'with 2π   ' if use_2pi else 'without 2π'
    physical = sorted([e for e in bands[use_2pi][-1] if abs(e) < E_MAX])
    print(f'  {label}: {[round(e,2) for e in physical]}')

# ── Plot ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(9, 4))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

for col, use_2pi in enumerate([True, False]):
    ax = fig.add_subplot(gs[col])
    tmp = np.transpose(bands[use_2pi])
    for branch in tmp:
        clipped = np.clip(branch, 0, E_MAX)
        ax.plot(clipped, color='steelblue', lw=0.8)

    ax.axvline(x=0, color='gray', lw=0.5, ls='--')
    ax.axvline(x=KSTEP - 1, color='gray', lw=0.5, ls='--')
    ax.set_xticks([0, KSTEP - 1])
    ax.set_xticklabels([r'$\Gamma$', 'h00'])
    ax.set_ylabel('Energy (meV)')
    ax.set_ylim([0, E_MAX])
    title = 'With 2π in phase' if use_2pi else 'Without 2π in phase'
    ax.set_title(title, fontsize=10)
    ax.tick_params(direction='in', top=True, right=True)

fig.suptitle(f'Phase convention test  (T={T} K, {N_CONFIGS} configs)', fontsize=11)
plt.savefig('../results/test_2pi_phase.png', dpi=150, bbox_inches='tight')
print('\nPlot saved to ../results/test_2pi_phase.png')
plt.show()
