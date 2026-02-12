import numpy as np
import csv
import os
import glob
from tqdm.auto import trange
import Readers  # Import your Readers module

import numpy as np

# def build_type_to_mass(atom_dic):
#     # Atomic mass in amu (g/mol)
#     atomic_mass = {
# 		'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'Be': 9.0122, 'B': 10.81, 'C': 12.011,
# 		'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180, 'Na': 22.990, 'Mg': 24.305,
# 		'Al': 26.982, 'Si': 28.085, 'P': 30.974, 'S': 32.06, 'Cl': 35.45, 'K': 39.098,
# 		'Ar': 39.948, 'Ca': 40.078, 'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996,
# 		'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38,
# 		'Ga': 69.723, 'Ge': 72.63, 'As': 74.922, 'Se': 78.96, 'Br': 79.904, 'Kr': 83.798,
# 		'Rb': 85.468, 'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224, 'Nb': 92.906, 'Mo': 95.96,
# 		'Tc': 98.0, 'Ru': 101.07, 'Rh': 102.91, 'Pd': 106.42, 'Ag': 107.87, 'Cd': 112.41,
# 		'In': 114.82, 'Sn': 118.71, 'Sb': 121.76, 'Te': 127.60, 'I': 126.90, 'Xe': 131.29,
# 		'Cs': 132.91, 'Ba': 137.33, 'La': 138.91, 'Ce': 140.12, 'Pr': 140.91, 'Nd': 144.24,
# 		'Pm': 145.0, 'Sm': 150.36, 'Eu': 151.96, 'Gd': 157.25, 'Tb': 158.93, 'Dy': 162.50,
# 		'Ho': 164.93, 'Er': 167.26, 'Tm': 168.93, 'Yb': 173.05, 'Lu': 174.97, 'Hf': 178.49,
# 		'Ta': 180.95, 'W': 183.84, 'Re': 186.21, 'Os': 190.23, 'Ir': 192.22, 'Pt': 195.08,
# 		'Au': 196.97, 'Hg': 200.59, 'Tl': 204.38, 'Pb': 207.2, 'Bi': 208.98, 'Po': 209.0,
# 		'At': 210.0, 'Rn': 222.0, 'Fr': 223.0, 'Ra': 226.0, 'Ac': 227.0, 'Th': 232.04,
# 		'Pa': 231.04, 'U': 238.03, 'Np': 237.0, 'Pu': 244.0, 'Am': 243.0, 'Cm': 247.0,
# 		'Bk': 247.0, 'Cf': 251.0, 'Es': 252.0, 'Fm': 257.0, 'Md': 258.0, 'No': 259.0,
# 		'Lr': 262.0
# 	}
#     # reverse_mapping: type_int -> element_symbol
#     type_to_elem = {v: k for k, values in atom_dic.items() for v in values}
#     # type_int -> mass
#     type_to_mass = {t: atomic_mass[type_to_elem[t]] for t in type_to_elem}
#     return type_to_mass

# def mass_per_atom_from_map(atype, type_to_mass):
#     atype = np.asarray(atype, dtype=np.int32)
#     return np.asarray([type_to_mass[int(t)] for t in atype], dtype=np.float32)

# import jax
# import jax.numpy as jnp
# from jax import jit

# @jit
# def calc_collect_var(kvec, displacements, cell_idx, type_ids, sqrt_mass_by_type, ntypes):
#     """
#     kvec: (3,) float32
#     displacements: (N,3) float32
#     cell_idx: (N,3) float32 or int32 (we'll cast)
#     type_ids: (N,) int32 in [0, ntypes-1]
#     sqrt_mass_by_type: (ntypes,) float32
#     ntypes: int
#     returns: (1, ntypes*3) complex64
#     """
#     kvec = kvec.astype(jnp.float32)
#     cell_idx = cell_idx.astype(jnp.float32)

#     phase = jnp.exp(1j * (cell_idx @ kvec))            # (N,) complex64
#     m = sqrt_mass_by_type[type_ids]                    # (N,) float32
#     weighted = (displacements * m[:, None]) * phase[:, None]  # (N,3) complex64

#     sums = jnp.zeros((ntypes, 3), dtype=weighted.dtype).at[type_ids].add(weighted)
#     counts = jnp.zeros((ntypes,), dtype=jnp.float32).at[type_ids].add(1.0)

#     U = sums / jnp.sqrt(jnp.maximum(counts, 1.0))[:, None]
#     return U.reshape(1, -1)

# @jit
# def calc_Sk(U_k_t):
#     return U_k_t.T @ jnp.conj(U_k_t)

# def prepare_type_ids_and_sqrt_mass(atype, type_to_mass):
#     atype = np.asarray(atype, dtype=np.int32)
#     unique_types = np.unique(atype)  # sorted
#     ntypes = int(unique_types.size)

#     # map actual type codes -> 0..ntypes-1
#     mapper = {int(t): i for i, t in enumerate(unique_types.tolist())}
#     type_ids = np.fromiter((mapper[int(t)] for t in atype), dtype=np.int32, count=atype.size)

#     # sqrt(mass) aligned with unique_types (NOT per atom)
#     sqrt_mass_by_type = np.sqrt(np.asarray([type_to_mass[int(t)] for t in unique_types], dtype=np.float32))
#     return unique_types, type_ids, sqrt_mass_by_type, ntypes

# import numpy as np
# import jax
# import jax.numpy as jnp

# def calc_collect_var(kvec, atype, configuration, cell_idx, hsymconfig, type_to_mass):
#     kvec_np = np.asarray(kvec, dtype=np.float32)
#     displacements_np = (np.asarray(configuration, dtype=np.float32) - np.asarray(hsymconfig, dtype=np.float32))
#     cell_np = np.asarray(cell_idx, dtype=np.float32)  # dot needs float

#     _, type_ids_np, sqrt_mass_by_type_np, ntypes = prepare_type_ids_and_sqrt_mass(atype, type_to_mass)

#     U = calc_collect_var(
#         jnp.asarray(kvec_np),
#         jnp.asarray(displacements_np),
#         jnp.asarray(cell_np),
#         jnp.asarray(type_ids_np),
#         jnp.asarray(sqrt_mass_by_type_np),
#         ntypes,
#     )

#     return np.asarray(jax.device_get(U))  # numpy return like original


# import glob, os, csv
# from tqdm import trange
# import jax
# import jax.numpy as jnp
# import numpy as np

# def Sk_avg(fpath, hsym_config, atom_dic, dim, kpnt, loadfile=True, save=True):
#     fnames = glob.glob(fpath + 'Frac*.txt')
#     saved_Sk = fpath + 'Sk_sum_kvec_{}_{}_{}.csv'.format(*kpnt)
    
#     ref_fname = fnames[0]  
#     atype_ref, config_ref, cell_ref = Readers.read_frac_atom_ph(ref_fname, atom_dic, dim)
#     hsym_ref = np.asarray(config_ref, dtype=np.float32)  # shape (N,3)

#     type_to_mass = build_type_to_mass(atom_dic)  # build once

#     ini_idx = 0
#     Sk_sum = None

#     if loadfile and os.path.exists(saved_Sk):
#         with open(saved_Sk, 'r') as file:
#             reader = csv.reader(file)
#             header = next(reader)
#             ini_idx = int(header[0]) + 1
#             Sk_sum = np.array([list(map(complex, row)) for row in reader])

#     hsym_ref = np.asarray(hsym_config[1], dtype=np.float32)

#     for ii in trange(ini_idx, len(fnames), desc=f'k={kpnt}', leave=None):
#         atype, config, cell_idx = Readers.read_frac_atom_ph(fnames[ii], atom_dic, dim)  # CPU I/O

#         U_k = calc_collect_var(kpnt, atype, config, cell_idx, hsym_ref, type_to_mass)  # GPU compute
#         Sk = np.asarray(jax.device_get(calc_Sk(jnp.asarray(U_k))))

#         if Sk_sum is None:
#             Sk_sum = Sk
#         else:
#             Sk_sum += Sk

#     if save and ini_idx < len(fnames):
#         with open(saved_Sk, 'w', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow([len(fnames)])
#             writer.writerows(Sk_sum)

#     return Sk_sum / len(fnames)


# import numpy as np

# def select_atom_type(tag, atype, config, cell_idx):
#     atype = np.asarray(atype)
#     sel = (atype == tag)
#     return config[sel], cell_idx[sel]


# # CPU version
# Generate a mass array for calculating U_k
def get_mass_array(atom_idx,atom_dic) :
	# Atomic mass in amu (g/mol)
	atomic_mass = {
		'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'Be': 9.0122, 'B': 10.81, 'C': 12.011,
		'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180, 'Na': 22.990, 'Mg': 24.305,
		'Al': 26.982, 'Si': 28.085, 'P': 30.974, 'S': 32.06, 'Cl': 35.45, 'K': 39.098,
		'Ar': 39.948, 'Ca': 40.078, 'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996,
		'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38,
		'Ga': 69.723, 'Ge': 72.63, 'As': 74.922, 'Se': 78.96, 'Br': 79.904, 'Kr': 83.798,
		'Rb': 85.468, 'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224, 'Nb': 92.906, 'Mo': 95.96,
		'Tc': 98.0, 'Ru': 101.07, 'Rh': 102.91, 'Pd': 106.42, 'Ag': 107.87, 'Cd': 112.41,
		'In': 114.82, 'Sn': 118.71, 'Sb': 121.76, 'Te': 127.60, 'I': 126.90, 'Xe': 131.29,
		'Cs': 132.91, 'Ba': 137.33, 'La': 138.91, 'Ce': 140.12, 'Pr': 140.91, 'Nd': 144.24,
		'Pm': 145.0, 'Sm': 150.36, 'Eu': 151.96, 'Gd': 157.25, 'Tb': 158.93, 'Dy': 162.50,
		'Ho': 164.93, 'Er': 167.26, 'Tm': 168.93, 'Yb': 173.05, 'Lu': 174.97, 'Hf': 178.49,
		'Ta': 180.95, 'W': 183.84, 'Re': 186.21, 'Os': 190.23, 'Ir': 192.22, 'Pt': 195.08,
		'Au': 196.97, 'Hg': 200.59, 'Tl': 204.38, 'Pb': 207.2, 'Bi': 208.98, 'Po': 209.0,
		'At': 210.0, 'Rn': 222.0, 'Fr': 223.0, 'Ra': 226.0, 'Ac': 227.0, 'Th': 232.04,
		'Pa': 231.04, 'U': 238.03, 'Np': 237.0, 'Pu': 244.0, 'Am': 243.0, 'Cm': 247.0,
		'Bk': 247.0, 'Cf': 251.0, 'Es': 252.0, 'Fm': 257.0, 'Md': 258.0, 'No': 259.0,
		'Lr': 262.0
	}
	# Step 1: Create reverse mapping dictionary
	reverse_mapping = {v: k for k, values in atom_dic.items() for v in values}
	# Step 2: Replace numbers in the list with corresponding keys
	replaced_list = [reverse_mapping[num] for num in atom_idx]
	# Step 3: Replace atom type with mass
	mass_array = [atomic_mass[atom] for atom in replaced_list]
	return mass_array

# Generate a neutron cross-section array for each atoms
def get_nxs_array(atom_idx,atom_dic) :
 	# Total Neutron Scattering Cross-Sections (sigma_scatt) in Barns
	# Source: NIST Center for Neutron Research (Natural Abundance)
	neutron_scatt_sigma = {
		'H': 82.02,  'D': 7.64,   'He': 1.34,  'Li': 1.37,  'Be': 7.63,
		'B': 5.24,   'C': 5.551,  'N': 11.51,  'O': 4.232,  'F': 4.018,
		'Ne': 2.62,  'Na': 3.28,  'Mg': 3.71,  'Al': 1.503, 'Si': 2.167,
		'P': 3.31,   'S': 1.026,  'Cl': 16.8,  'K': 1.96,   'Ar': 0.68,
		'Ca': 2.83,  'Sc': 23.4,  'Ti': 4.35,  'V': 5.10,   'Cr': 3.49,
		'Mn': 2.15,  'Fe': 11.62, 'Co': 6.07,  'Ni': 18.5,  'Cu': 8.03,
		'Zn': 4.054, 'Ga': 6.83,  'Ge': 8.42,  'As': 5.48,  'Se': 8.30,
		'Br': 5.9,   'Kr': 7.66,  'Rb': 6.32,  'Sr': 6.25,  'Y': 7.76,
		'Zr': 6.46,  'Nb': 6.253, 'Mo': 5.71,  'Tc': 6.0,   'Ru': 5.1,
		'Rh': 4.81,  'Pd': 4.5,   'Ag': 4.99,  'Cd': 6.5,   'In': 2.61,
		'Sn': 4.89,  'Sb': 3.9,   'Te': 4.25,  'I': 3.55,   'Xe': 4.3,
		'Cs': 4.23,  'Ba': 3.38,  'La': 9.81,  'Ce': 2.94,  'Pr': 3.58,
		'Nd': 16.5,  'Pm': 16.0,  'Sm': 39.0,  'Eu': 9.2,   'Gd': 175.0,
		'Tb': 23.0,  'Dy': 34.4,  'Ho': 8.8,   'Er': 8.0,   'Tm': 8.5,
		'Yb': 23.4,  'Lu': 5.9,   'Hf': 5.88,  'Ta': 6.01,  'W': 4.87,
		'Re': 13.5,  'Os': 12.6,  'Ir': 14.0,  'Pt': 11.71, 'Au': 7.63,
		'Hg': 26.5,  'Tl': 9.89,  'Pb': 11.11, 'Bi': 9.16,  'Th': 12.63,
		'U': 14.16
	}
	# Step 1: Create reverse mapping dictionary
	reverse_mapping = {v: k for k, values in atom_dic.items() for v in values}
	# Step 2: Replace numbers in the list with corresponding keys
	replaced_list = [reverse_mapping[num] for num in atom_idx]
	# Step 3: Replace atom type with neutron corss-section
	nxs_array = [neutron_scatt_sigma[atom] for atom in replaced_list]
	return nxs_array

def select_atom_type(tag, atype, config, cell_idx):
    sel_idx = []
    for ii in np.arange(len(atype)):
        if atype[ii] == tag:
            sel_idx.append(ii)
    return config[sel_idx], cell_idx[sel_idx]

def calc_collect_var(kvec, atype, configuration, cell_idx, hsymconfig, atom_dic):
    kvec = np.array(kvec)
    displacements = configuration - hsymconfig
    atom_idx = np.array(list(set(atype)))
    mass_array = get_mass_array(atype, atom_dic)
    
    U_k_t = []
    for ii in np.arange(len(atom_idx)):
        tmp_config, tmp_cell_idx = select_atom_type(atom_idx[ii], atype, displacements, cell_idx)
        tmp_data = 0
        tmp_cnt = 0
        for jj in np.arange(len(tmp_cell_idx)):
            # Calculating term
            tmp_data += np.sqrt(mass_array[ii]) * tmp_config[jj] * np.exp(1j * np.dot(kvec, tmp_cell_idx[jj]))
            tmp_cnt += 1
        U_k_t.append(tmp_data/np.sqrt(tmp_cnt))
    
    U_k_t = np.array(U_k_t)
    U_k_t = U_k_t.reshape(1, -1)
    return U_k_t

def calc_Sk(U_k_t):
    U_l = U_k_t.T
    U_r = U_k_t.conj()
    result = U_l @ U_r
    return result

def Sk_avg(fpath, hsym_config, atom_dic, dim, kpnt, loadfile=True, save=True):
    fnames = glob.glob(fpath + 'Frac*.txt')
    saved_Sk = fpath + 'Sk_sum_kvec_{}_{}_{}.csv'.format(*kpnt)
    
    ini_idx = 0
    Sk_sum = None
    show_prog = True # Default unless loading fails/starts fresh

    if loadfile and os.path.exists(saved_Sk):
        with open(saved_Sk, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)
            ini_idx = int(header[0]) + 1
            Sk_sum = np.array([list(map(complex, row)) for row in reader])
    
    # Calculate for remaining files
    for ii in trange(ini_idx, len(fnames), desc='k={}'.format(kpnt), leave=None, position=0, disable=show_prog):
        test = Readers.read_frac_atom_ph(fnames[ii], atom_dic, dim)
        U_k = calc_collect_var(kpnt, test[0], test[1], test[2], hsym_config[1], atom_dic)
        Sk = calc_Sk(U_k)
        if ii == 0 and Sk_sum is None:
            Sk_sum = Sk
        else:
            Sk_sum += Sk

    if save and ini_idx < len(fnames):
        with open(saved_Sk, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([len(fnames)])
            writer.writerows(Sk_sum)
            
    Sk_avg_val = Sk_sum/len(fnames)
    return Sk_avg_val

def Partial_Sk_avg(fpath, hsym_config, atom_dic, dim, kpnt, atype, loadfile=True, save=True):
    fnames = glob.glob(fpath + 'Frac*.txt')
    saved_Sk = fpath + '{}_Sk_sum_kvec_{}_{}_{}.csv'.format(atype, *kpnt)
    
    ini_idx = 0
    Sk_sum = None
    show_prog = False

    if loadfile and os.path.exists(saved_Sk):
        with open(saved_Sk, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)
            ini_idx = int(header[0]) + 1
            Sk_sum = np.array([list(map(complex, row)) for row in reader])
    
    for ii in trange(ini_idx, len(fnames), desc='#### {} #### k={}'.format(atype, kpnt), leave=None, position=0, disable=show_prog):
        test = Readers.read_frac_atom_ph(fnames[ii], atom_dic, dim, atype)
        U_k = calc_collect_var(kpnt, test[0], test[1], test[2], hsym_config[1], atom_dic)
        Sk = calc_Sk(U_k)
        if ii == 0 and Sk_sum is None:
            Sk_sum = Sk
        else:
            Sk_sum += Sk

    if save and ini_idx < len(fnames):
        with open(saved_Sk, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([len(fnames)])
            writer.writerows(Sk_sum)
            
    Sk_avg_val = Sk_sum/len(fnames)
    return Sk_avg_val

def gen_grid(n_points=5):
    q_min = -0.5
    q_max = 0.5
    qx = np.linspace(q_min, q_max, n_points)
    qy = np.linspace(q_min, q_max, n_points)
    qz = np.linspace(q_min, q_max, n_points)
    q_points = np.array(np.meshgrid(qx, qy, qz)).T.reshape(-1, 3)
    return q_points

def get_ph_weights(atom_dic, IRs):
    atom_types = list(atom_dic.keys())
    print('Calculating the weights for {} ...'.format(atom_types))
    weights_all = []
    for ii in np.arange(len(IRs)):
        eigvecs = np.real(IRs[ii].reshape(len(IRs[ii])//3, 3))
        weight_tmp = []
        for atom_type in atom_types:
            idx_tmp = np.array(atom_dic[atom_type]) - 1
            eigvecs_tmp = eigvecs[idx_tmp]
            accum_disp = 0
            for jj in np.arange(len(eigvecs_tmp)):
                accum_disp += np.sqrt(np.real(np.sum(eigvecs_tmp[jj]**2)))
            weight_tmp.append(accum_disp)
        weights_all.append(weight_tmp)
    return np.transpose(weights_all)