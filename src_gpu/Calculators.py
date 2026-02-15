import numpy as np
import csv
import os
import glob
from tqdm.auto import trange
import Readers  # Import your Readers module
from tqdm import tqdm

## GPU version
import jax.numpy as jnp
from jax import jit, vmap
import jax

os.environ["JAX_ENABLE_X64"] = "True"

# Global cache to store the mass table on the GPU so we don't rebuild it every frame.
_GPU_MASS_TABLE = None

def get_mass_array(atom_idx, atom_dic):
    """
    JAX/Metal optimized version.
    Automatically builds a lookup table on the first run, then uses 
    fast GPU indexing for all subsequent calls.
    """
    global _GPU_MASS_TABLE

    # Only build the mass table if it doesn't exist yet
    if _GPU_MASS_TABLE is None:
        print("Initializing GPU Mass Table (First Run Only)...")
        
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

        # Determine the size of the array (Max Index + 1)
        all_indices = [idx for indices in atom_dic.values() for idx in indices]
        if not all_indices:
            max_idx = 0
        else:
            max_idx = max(all_indices)

        # Create a dense array: index = atom_id, value = mass
        # We use float32 for Metal GPU efficiency
        lut_cpu = np.zeros(max_idx + 1, dtype=np.float64)

        for symbol, indices in atom_dic.items():
            mass = atomic_mass.get(symbol, 0.0)
            lut_cpu[indices] = mass
        
        # Move to GPU once
        _GPU_MASS_TABLE = jnp.array(lut_cpu)

    # --- THE FAST PART ---
    # Ensure indices are on JAX/GPU
    indices_gpu = jnp.array(atom_idx)
    
    # Vectorized Lookup (Gather)
    return _GPU_MASS_TABLE[indices_gpu]

@jit
def calc_Sk(U_k_t):
    """
    Computes U.T @ U.conj() using accelerated JAX kernels.
    
    Args:
        U_k_t (jnp.ndarray): Input matrix (likely complex).
    """
    # JAX handles the lazy evaluation of T and conj inside the matmul kernel
    return jnp.matmul(U_k_t.T, U_k_t.conj())

from functools import partial

# Import segment_sum (location varies by JAX version)
try:
    from jax.ops import segment_sum
except ImportError:
    from jax.numpy import segment_sum

# FIX: We use 'static_argnums=(5,)' to tell JAX that the 6th argument (num_types)
# is a constant integer, allowing it to compile the output shape correctly.
@partial(jit, static_argnums=(5,))
def _calc_collect_var_kernel(kvec, displacements, cell_idx, masses, type_indices, num_types):
    """
    GPU Kernel.
    num_types must be static so JAX knows the output array size is (num_types, 3).
    """
    # 1. Vectorized Phase Calculation: e^(i * k . n)
    dot_products = jnp.dot(cell_idx, kvec)
    phase = jnp.exp(1j * dot_products)
    
    # 2. Vectorized Term Calculation: sqrt(m) * u * phase
    # Reshape weights to (N, 1) to broadcast against (N, 3) displacements
    weights = jnp.sqrt(masses)[:, None] * phase[:, None]
    weighted_displacements = displacements * weights
    
    # 3. Summation by Group
    # Note: num_segments is now a static integer thanks to the decorator
    summed_terms = segment_sum(weighted_displacements, type_indices, num_segments=num_types)
    
    # 4. Normalization Count
    # Sum '1's grouped by type to get counts
    counts = segment_sum(jnp.ones_like(type_indices), type_indices, num_segments=num_types)
    
    # 5. Normalize
    norm_factors = 1.0 / jnp.sqrt(jnp.maximum(counts, 1.0))
    normalized_U = summed_terms * norm_factors[:, None]
    
    # 6. Flatten
    return normalized_U.reshape(1, -1)

def calc_collect_var(kvec, atype, configuration, cell_idx, hsymconfig, atom_dic):
    """
    Wrapper function. Prepares data on CPU, executes on GPU.
    """
    # --- Pre-processing (CPU side) ---
    kvec = jnp.array(kvec)
    displacements = jnp.array(configuration - hsymconfig)
    cell_idx = jnp.array(cell_idx)
    
    # 1. Get Masses (using your optimized get_mass_array)
    mass_array = get_mass_array(atype, atom_dic)
    masses = jnp.array(mass_array)
    
    # 2. Convert 'atype' list to Integer Indices
    # return_inverse gives indices (0, 1, 0...) mapping to unique types
    unique_types, type_indices = np.unique(atype, return_inverse=True)
    
    # CRITICAL: This must be a standard Python int, not a JAX array
    num_types = int(len(unique_types))
    
    type_indices = jnp.array(type_indices)
    
    # --- GPU Execution ---
    return _calc_collect_var_kernel(
        kvec, 
        displacements, 
        cell_idx, 
        masses, 
        type_indices, 
        num_types  # Passed as static int
    )


# Import segment_sum
try:
    from jax.ops import segment_sum
except ImportError:
    from jax.numpy import segment_sum

# --- 1. The Fixed GPU Kernel ---

# FIX: We tell JAX that argument #5 (num_types) is a static integer.
@partial(jit, static_argnums=(5,)) 
def process_batch_kernel(kvec, displacements_batch, cell_idx_batch, masses, type_indices, num_types):
    """
    Computes Sk for a batch.
    num_types (arg 5) MUST be static so JAX knows the output shape.
    """
    
    # Define the logic for a single frame
    def single_frame_calc(disp, cell):
        # 1. Phase & Term
        dot_products = jnp.dot(cell, kvec)
        # check if it has 2*pi
        phase = jnp.exp(1j * dot_products)
        #phase = jnp.exp(1j * 2 * jnp.pi * dot_products)
        weights = jnp.sqrt(masses)[:, None] * phase[:, None]
        weighted_disp = disp * weights
        
        # 2. Segment Sum (Group by atom type)
        # num_types is now a constant integer here, so this works!
        summed = segment_sum(weighted_disp, type_indices, num_segments=num_types)
        
        # 3. Normalize
        counts = segment_sum(jnp.ones_like(type_indices), type_indices, num_segments=num_types)
        norm_factors = 1.0 / jnp.sqrt(jnp.maximum(counts, 1.0))
        U_k = (summed * norm_factors[:, None]).reshape(1, -1)
        
        # 4. Calculate Sk (U.T @ U.conj)
        # Result shape: (3*num_types, 3*num_types)
        return jnp.matmul(U_k.T, U_k.conj())

    # Apply to batch using vmap
    # in_axes=(0, 0) -> First two args map over axis 0 (Batch dimension)
    batch_Sk = vmap(single_frame_calc, in_axes=(0, 0))(displacements_batch, cell_idx_batch)
    
    # Sum the results on the GPU before returning to Python
    return jnp.sum(batch_Sk, axis=0)


# --- 2. The Driver Function ---

def Sk_avg(fpath, hsym_config, atom_dic, dim, kpnt, loadfile=True, save=True, batch_size=50):
    fnames = sorted(glob.glob(fpath + 'Frac*.txt'))
    saved_Sk_path = fpath + f'Sk_sum_kvec_{kpnt[0]}_{kpnt[1]}_{kpnt[2]}.csv'
    
    Sk_sum = None
    start_idx = 0
    
    # Initialize static data
    # We assume atom types don't change between files, so we read the first file to set up JAX.
    if len(fnames) > 0:
        # Read first file to get system topology
        temp_data = Readers.read_frac_atom_ph(fnames[0], atom_dic, dim)
        atype_static = temp_data[0] 
        
        # Prepare Static JAX Arrays
        mass_array = get_mass_array(atype_static, atom_dic) # Use your new GPU function
        masses_gpu = jnp.array(mass_array)
        
        unique_types, type_indices_cpu = np.unique(atype_static, return_inverse=True)
        type_indices_gpu = jnp.array(type_indices_cpu)
        
        # CRITICAL: Ensure this is a standard Python int
        num_types = int(len(unique_types))
        
        kvec_gpu = jnp.array(kpnt)
        
        # Calculate matrix dimension size for initialization
        dim_size = num_types * 3

    # Load previous progress if exists
    if loadfile and os.path.exists(saved_Sk_path):
        print(f"Loading from {saved_Sk_path}...")
        try:
            with open(saved_Sk_path, 'r') as file:
                reader = csv.reader(file)
                header = next(reader)
                # Check if file matches expected dimensions
                if int(header[0]) <= len(fnames):
                    start_idx = int(header[0])
                    Sk_sum = np.array([[complex(x) for x in row] for row in reader])
        except Exception:
            print("Load failed, starting fresh.")
            Sk_sum = np.zeros((dim_size, dim_size), dtype=np.complex128)
    else:
        Sk_sum = np.zeros((dim_size, dim_size), dtype=np.complex128)

    # Main Batch Loop
    # Iterate through files in chunks of 'batch_size'
    for i in tqdm(range(start_idx, len(fnames), batch_size), desc=f'Processing Batches', disable=True):
        # Get list of files for this batch
        batch_files = fnames[i : i + batch_size]
        
        disp_list = []
        cell_list = []
        
        # Read batch from disk (CPU bottleneck)
        for fname in batch_files:
            _, config, cell_idx = Readers.read_frac_atom_ph(fname, atom_dic, dim)
            # Calculate displacement relative to reference structure
            disp = config - hsym_config[1]
            disp_list.append(disp)
            cell_list.append(cell_idx)
            
        # Stack and Move to GPU
        # Convert list of arrays -> single numpy array -> JAX array
        disp_batch_gpu = jnp.array(np.stack(disp_list))
        cell_batch_gpu = jnp.array(np.stack(cell_list))
        
        # Execute Kernel
        batch_result = process_batch_kernel(
            kvec_gpu, 
            disp_batch_gpu, 
            cell_batch_gpu, 
            masses_gpu, 
            type_indices_gpu, 
            num_types  # Passed as static int
        )
        
        # Accumulate result (Wait for GPU here)
        Sk_sum += np.array(batch_result)
        
        # Optional: Save every 500 frames or so to prevent data loss
        if (i + len(batch_files)) % 500 == 0 and save:
             with open(saved_Sk_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([i + len(batch_files)])
                writer.writerows(Sk_sum)

    # Final Save
    if save:
        with open(saved_Sk_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([len(fnames)])
            writer.writerows(Sk_sum)

    return Sk_sum / len(fnames)

def Partial_Sk_avg(fpath, hsym_config, atom_dic, dim, kpnt, atype, loadfile=True, save=True, batch_size=50):
    fnames = sorted(glob.glob(fpath + 'Frac*.txt'))
    # Filename includes atype now
    saved_Sk_path = fpath + f'{atype}_Sk_sum_kvec_{kpnt[0]}_{kpnt[1]}_{kpnt[2]}.csv'
    
    Sk_sum = None
    start_idx = 0
    
    # --- 1. Static Setup (Run Once) ---
    # We need to slice the reference structure (hsym_config) to match ONLY the atoms of 'atype'
    # Assuming atom_dic maps {'Type': [index1, index2...]}
    if atype not in atom_dic:
        raise ValueError(f"Atom type '{atype}' not found in atom_dic.")
        
    target_indices = atom_dic[atype]
    
    # 1a. Get Reference Positions for just this atom type
    # Shape: (N_subset, 3)
    hsym_ref_subset = hsym_config[1][target_indices]
    
    # 1b. Prepare JAX Static Arrays
    # Since we are filtering by type, every atom in our batch is the same type.
    # So type_indices is all zeros, and num_types is 1.
    num_target_atoms = len(target_indices)
    
    # Mass is constant for all atoms of this type
    mass_val = get_mass_array([atype], atom_dic)[0] # Get scalar mass
    masses_gpu = jnp.full((num_target_atoms,), mass_val, dtype=jnp.float64)
    
    # All atoms belong to group "0"
    type_indices_gpu = jnp.zeros(num_target_atoms, dtype=jnp.int64)
    num_types = 1
    
    kvec_gpu = jnp.array(kpnt)

    # --- 2. Load Previous Progress ---
    if loadfile and os.path.exists(saved_Sk_path):
        print(f"Loading partial progress from {saved_Sk_path}...")
        try:
            with open(saved_Sk_path, 'r') as file:
                reader = csv.reader(file)
                header = next(reader)
                if int(header[0]) <= len(fnames):
                    start_idx = int(header[0])
                    Sk_sum = np.array([[complex(x) for x in row] for row in reader])
        except Exception:
            print("Load failed, starting fresh.")
            Sk_sum = np.zeros((3, 3), dtype=np.complex128)
    else:
        # Partial Sk is usually 3x3 for a single species
        Sk_sum = np.zeros((3, 3), dtype=np.complex128)

    # --- 3. Batch Processing Loop ---
    for i in tqdm(range(start_idx, len(fnames), batch_size), desc=f'Partial Sk ({atype})'):
        batch_files = fnames[i : i + batch_size]
        
        disp_list = []
        cell_list = []
        
        for fname in batch_files:
            # Readers.read... returns ONLY the atoms of 'atype' because we passed it as an arg
            test = Readers.read_frac_atom_ph(fname, atom_dic, dim, atype)
            
            # test[1] is config (N_subset, 3)
            # test[2] is cell_idx (N_subset, 3)
            current_config = test[1]
            current_cell_idx = test[2]
            
            # Calculate displacement using the sliced reference
            # Note: We do this on CPU before stacking
            disp = current_config - hsym_ref_subset
            
            disp_list.append(disp)
            cell_list.append(current_cell_idx)
            
        # Stack into (Batch, N_subset, 3)
        disp_batch_gpu = jnp.array(np.stack(disp_list))
        cell_batch_gpu = jnp.array(np.stack(cell_list))
        
        # Execute GPU Kernel
        # We reuse the SAME kernel from the previous answer!
        batch_result = process_batch_kernel(
            kvec_gpu, 
            disp_batch_gpu, 
            cell_batch_gpu, 
            masses_gpu, 
            type_indices_gpu, 
            num_types  # Passed as static int (1)
        )
        
        # Accumulate
        Sk_sum += np.array(batch_result)
        
        # Periodic Save
        if (i + len(batch_files)) % 500 == 0 and save:
             with open(saved_Sk_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([i + len(batch_files)])
                writer.writerows(Sk_sum)

    # --- 4. Final Save ---
    if save:
        with open(saved_Sk_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([len(fnames)])
            writer.writerows(Sk_sum)

    return Sk_sum / len(fnames)

# # CPU version
# Generate a mass array for calculating U_k
# def get_mass_array(atom_idx,atom_dic) :
# 	# Atomic mass in amu (g/mol)
# 	atomic_mass = {
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
# 	# Step 1: Create reverse mapping dictionary
# 	reverse_mapping = {v: k for k, values in atom_dic.items() for v in values}
# 	# Step 2: Replace numbers in the list with corresponding keys
# 	replaced_list = [reverse_mapping[num] for num in atom_idx]
# 	# Step 3: Replace atom type with mass
# 	mass_array = [atomic_mass[atom] for atom in replaced_list]
# 	return mass_array

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

# def calc_collect_var(kvec, atype, configuration, cell_idx, hsymconfig, atom_dic):
#     kvec = np.array(kvec)
#     displacements = configuration - hsymconfig
#     atom_idx = np.array(list(set(atype)))
#     mass_array = get_mass_array(atype, atom_dic)
    
#     U_k_t = []
#     for ii in np.arange(len(atom_idx)):
#         tmp_config, tmp_cell_idx = select_atom_type(atom_idx[ii], atype, displacements, cell_idx)
#         tmp_data = 0
#         tmp_cnt = 0
#         for jj in np.arange(len(tmp_cell_idx)):
#             # Calculating term
#             tmp_data += np.sqrt(mass_array[ii]) * tmp_config[jj] * np.exp(1j * np.dot(kvec, tmp_cell_idx[jj]))
#             tmp_cnt += 1
#         U_k_t.append(tmp_data/np.sqrt(tmp_cnt))
    
#     U_k_t = np.array(U_k_t)
#     U_k_t = U_k_t.reshape(1, -1)
#     return U_k_t

# def calc_Sk(U_k_t):
#     U_l = U_k_t.T
#     U_r = U_k_t.conj()
#     result = U_l @ U_r
#     return result

# def Sk_avg(fpath, hsym_config, atom_dic, dim, kpnt, loadfile=True, save=True):
#     fnames = glob.glob(fpath + 'Frac*.txt')
#     saved_Sk = fpath + 'Sk_sum_kvec_{}_{}_{}.csv'.format(*kpnt)
    
#     ini_idx = 0
#     Sk_sum = None
#     show_prog = True # Default unless loading fails/starts fresh

#     if loadfile and os.path.exists(saved_Sk):
#         with open(saved_Sk, 'r') as file:
#             reader = csv.reader(file)
#             header = next(reader)
#             ini_idx = int(header[0]) + 1
#             Sk_sum = np.array([list(map(complex, row)) for row in reader])
    
#     # Calculate for remaining files
#     for ii in trange(ini_idx, len(fnames), desc='k={}'.format(kpnt), leave=None, position=0, disable=show_prog):
#         test = Readers.read_frac_atom_ph(fnames[ii], atom_dic, dim)
#         U_k = calc_collect_var(kpnt, test[0], test[1], test[2], hsym_config[1], atom_dic)
#         Sk = calc_Sk(U_k)
#         if ii == 0 and Sk_sum is None:
#             Sk_sum = Sk
#         else:
#             Sk_sum += Sk

#     if save and ini_idx < len(fnames):
#         with open(saved_Sk, 'w', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow([len(fnames)])
#             writer.writerows(Sk_sum)
            
#     Sk_avg_val = Sk_sum/len(fnames)
#     return Sk_avg_val

# def Partial_Sk_avg(fpath, hsym_config, atom_dic, dim, kpnt, atype, loadfile=True, save=True):
#     fnames = glob.glob(fpath + 'Frac*.txt')
#     saved_Sk = fpath + '{}_Sk_sum_kvec_{}_{}_{}.csv'.format(atype, *kpnt)
    
#     ini_idx = 0
#     Sk_sum = None
#     show_prog = False

#     if loadfile and os.path.exists(saved_Sk):
#         with open(saved_Sk, 'r') as file:
#             reader = csv.reader(file)
#             header = next(reader)
#             ini_idx = int(header[0]) + 1
#             Sk_sum = np.array([list(map(complex, row)) for row in reader])
    
#     for ii in trange(ini_idx, len(fnames), desc='#### {} #### k={}'.format(atype, kpnt), leave=None, position=0, disable=show_prog):
#         test = Readers.read_frac_atom_ph(fnames[ii], atom_dic, dim, atype)
#         U_k = calc_collect_var(kpnt, test[0], test[1], test[2], hsym_config[1], atom_dic)
#         Sk = calc_Sk(U_k)
#         if ii == 0 and Sk_sum is None:
#             Sk_sum = Sk
#         else:
#             Sk_sum += Sk

#     if save and ini_idx < len(fnames):
#         with open(saved_Sk, 'w', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow([len(fnames)])
#             writer.writerows(Sk_sum)
            
#     Sk_avg_val = Sk_sum/len(fnames)
#     return Sk_avg_val

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