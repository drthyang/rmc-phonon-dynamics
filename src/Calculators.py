import numpy as np
import csv
import os
import glob
from tqdm.auto import trange

import Readers
from constants import ATOMIC_MASS, NEUTRON_SCATT_SIGMA, ENERGY_CONV


def get_mass_array(atom_idx, atom_dic):
    '''Return a mass [amu] for each element of atom_idx.

    atom_idx : iterable of atom type IDs (same integer IDs stored in atom_dic values)
    atom_dic : dict  element_symbol -> list[type_id]
    '''
    reverse_mapping = {v: k for k, values in atom_dic.items() for v in values}
    return [ATOMIC_MASS[reverse_mapping[tid]] for tid in atom_idx]


def get_nxs_array(atom_idx, atom_dic):
    '''Return the neutron total scattering cross-section [barn] for each type in atom_idx.'''
    reverse_mapping = {v: k for k, values in atom_dic.items() for v in values}
    return [NEUTRON_SCATT_SIGMA[reverse_mapping[tid]] for tid in atom_idx]


def select_atom_type(tag, atype, config, cell_idx):
    sel_idx = [ii for ii in range(len(atype)) if atype[ii] == tag]
    return config[sel_idx], cell_idx[sel_idx]


def calc_collect_var(kvec, atype, configuration, cell_idx, hsymconfig, atom_dic, v_super, dim):
    '''Build the mass-weighted, Fourier-transformed displacement vector U_k.

    Displacements are converted from dim-scaled fractional coordinates to
    Cartesian Å before mass-weighting so that eigenvalues of S(k) = U_k† U_k
    carry units of amu·Å², enabling a physically correct energy conversion.

    Phase convention: exp(2πi k_frac · n_cell), where k_frac are fractional
    reciprocal coordinates and n_cell are integer unit-cell indices.

    Parameters
    ----------
    kvec          : (3,) fractional reciprocal-space coordinate
    atype         : list of integer atom-type IDs, one per atom
    configuration : (N, 3) dim-scaled fractional coordinates from Readers
    cell_idx      : (N, 3) integer unit-cell indices
    hsymconfig    : (N, 3) average (equilibrium) dim-scaled fractional coordinates
    atom_dic      : dict  element -> list[type_id]
    v_super       : (3, 3) supercell lattice matrix, rows = vectors in Å
    dim           : (3,) supercell repeat dimensions
    '''
    kvec = np.array(kvec)

    # Displacements in dim-scaled fractional → Cartesian Å
    # u_Å[n] = (u_frac_scaled[n] / dim) @ v_super
    displacements_frac = configuration - hsymconfig
    displacements_Å = (displacements_frac / dim) @ v_super  # (N, 3) in Å

    unique_types = np.array(sorted(set(atype)))
    mass_per_type = get_mass_array(unique_types, atom_dic)  # one mass per unique type

    U_k_t = []
    for ii, type_id in enumerate(unique_types):
        tmp_disp, tmp_cell = select_atom_type(type_id, atype, displacements_Å, cell_idx)
        n_atoms = len(tmp_cell)
        sqrt_mass = np.sqrt(mass_per_type[ii])

        # U_k,t = (1/√N_t) Σ_n √m_t · u_n · exp(2πi k_frac · n_cell)
        # 2π converts the fractional reciprocal coord k_frac ∈ [-0.5, 0.5]
        # (units of b_i) into radians per integer cell index.
        phases = np.exp(2j * np.pi * (tmp_cell @ kvec))  # (N_t,)
        weighted = sqrt_mass * tmp_disp * phases[:, np.newaxis]  # (N_t, 3)
        U_k_t.append(weighted.sum(axis=0) / np.sqrt(n_atoms))   # (3,)

    U_k_t = np.array(U_k_t).reshape(1, -1)  # (1, 3*n_types)
    return U_k_t


def calc_Sk(U_k_t):
    '''S(k) = U_k† · U_k  (outer product, units amu·Å²).'''
    return U_k_t.T @ U_k_t.conj()


def eigenvalues_to_meV(eigenvalues, T):
    '''Convert S(k) eigenvalues [amu·Å²] to phonon energies [meV].

    Uses the classical equipartition relation:
        ω = sqrt(kb_J · T / (m_kg · <u_m²>))
        E = ℏ · ω

    Negative eigenvalues (dynamically unstable / soft modes) are returned as
    negative energies so that imaginary branches remain visually distinct.
    Zero eigenvalues (acoustic modes at Γ) are mapped to 0.

    Parameters
    ----------
    eigenvalues : array-like, eigenvalues of the averaged S(k) matrix [amu·Å²]
    T           : float, temperature [K]

    Returns
    -------
    energies : ndarray [meV], same shape as eigenvalues
    '''
    ev = np.asarray(eigenvalues, dtype=float)
    # Threshold: eigenvalues smaller than 1e-4 amu·Å² are acoustic-mode noise
    # (true acoustic eigenvalue at Γ is ~0; small positive values give huge energies)
    threshold = 1e-4
    valid = np.abs(ev) >= threshold
    safe = np.where(valid, np.abs(ev), np.nan)
    energies = ENERGY_CONV * np.sqrt(T / safe)
    energies = np.where(np.isnan(energies), 0.0, energies)
    return np.where(ev >= 0, energies, -energies)


def Sk_avg(fpath, hsym_config, atom_dic, dim, kpnt, v_super,
           loadfile=True, save=True):
    '''Average S(k) matrix over all configurations in fpath.

    Cached partial sums are stored per k-point so that interrupted runs can
    resume.  Delete the Sk_sum_kvec_*.csv files after changing any formula
    parameter (phase convention, displacement units, etc.).
    '''
    fnames = sorted(glob.glob(fpath + 'Frac*.txt'))
    saved_Sk = fpath + 'Sk_sum_kvec_{}_{}_{}.csv'.format(*kpnt)

    ini_idx = 0
    Sk_sum = None

    if loadfile and os.path.exists(saved_Sk):
        with open(saved_Sk, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            ini_idx = int(header[0]) + 1
            Sk_sum = np.array([list(map(complex, row)) for row in reader])

    for ii in trange(ini_idx, len(fnames),
                     desc='k={}'.format(kpnt), leave=None, position=0, disable=True):
        test = Readers.read_frac_atom_ph(fnames[ii], atom_dic, dim)
        U_k = calc_collect_var(kpnt, test[0], test[1], test[2],
                               hsym_config[1], atom_dic, v_super, dim)
        Sk = calc_Sk(U_k)
        Sk_sum = Sk if Sk_sum is None else Sk_sum + Sk

    if save and ini_idx < len(fnames) and Sk_sum is not None:
        with open(saved_Sk, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([len(fnames) - 1])
            writer.writerows(Sk_sum)

    return Sk_sum / len(fnames)


def Partial_Sk_avg(fpath, hsym_config, atom_dic, dim, kpnt, atype, v_super,
                   loadfile=True, save=True):
    '''Partial S(k) averaged over all configurations, for a single atom type.'''
    fnames = sorted(glob.glob(fpath + 'Frac*.txt'))
    saved_Sk = fpath + '{}_Sk_sum_kvec_{}_{}_{}.csv'.format(atype, *kpnt)

    ini_idx = 0
    Sk_sum = None

    if loadfile and os.path.exists(saved_Sk):
        with open(saved_Sk, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            ini_idx = int(header[0]) + 1
            Sk_sum = np.array([list(map(complex, row)) for row in reader])

    for ii in trange(ini_idx, len(fnames),
                     desc='{} k={}'.format(atype, kpnt), leave=None, position=0, disable=False):
        test = Readers.read_frac_atom_ph(fnames[ii], atom_dic, dim, atype)
        U_k = calc_collect_var(kpnt, test[0], test[1], test[2],
                               hsym_config[1], atom_dic, v_super, dim)
        Sk = calc_Sk(U_k)
        Sk_sum = Sk if Sk_sum is None else Sk_sum + Sk

    if save and ini_idx < len(fnames) and Sk_sum is not None:
        with open(saved_Sk, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([len(fnames) - 1])
            writer.writerows(Sk_sum)

    return Sk_sum / len(fnames)


def gen_grid(n_points=5):
    '''Uniform q-point mesh over the first Brillouin zone [-0.5, 0.5)^3.'''
    q = np.linspace(-0.5, 0.5, n_points, endpoint=False)
    return np.array(np.meshgrid(q, q, q)).T.reshape(-1, 3)


def get_ph_weights(atom_dic, IRs):
    atom_types = list(atom_dic.keys())
    print('Calculating the weights for {} ...'.format(atom_types))
    weights_all = []
    for ii in range(len(IRs)):
        eigvecs = np.real(IRs[ii].reshape(len(IRs[ii]) // 3, 3))
        weight_tmp = []
        for atom_type in atom_types:
            idx_tmp = np.array(atom_dic[atom_type]) - 1
            eigvecs_tmp = eigvecs[idx_tmp]
            accum_disp = sum(
                np.sqrt(np.real(np.sum(eigvecs_tmp[jj] ** 2)))
                for jj in range(len(eigvecs_tmp))
            )
            weight_tmp.append(accum_disp)
        weights_all.append(weight_tmp)
    return np.transpose(weights_all)
