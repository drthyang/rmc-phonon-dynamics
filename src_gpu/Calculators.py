import numpy as np
import csv
import os
import glob
from tqdm.auto import trange
import Readers  # Import your Readers module
from tqdm import tqdm
from constants import ATOMIC_MASS, NEUTRON_SCATT_SIGMA, ENERGY_CONV

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
        
        # Determine the size of the array (Max Index + 1)
        all_indices = [idx for indices in atom_dic.values() for idx in indices]
        if not all_indices:
            max_idx = 0
        else:
            max_idx = max(all_indices)

        # float32: Metal does not support float64
        lut_cpu = np.zeros(max_idx + 1, dtype=np.float32)

        for symbol, indices in atom_dic.items():
            mass = ATOMIC_MASS.get(symbol, 0.0)
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
    kvec = jnp.array(kvec, dtype=jnp.float32)
    displacements = jnp.array(configuration - hsymconfig, dtype=jnp.float32)
    cell_idx = jnp.array(cell_idx, dtype=jnp.float32)
    
    # 1. Get Masses (using your optimized get_mass_array)
    mass_array = get_mass_array(atype, atom_dic)
    masses = jnp.array(mass_array)
    
    # 2. Convert 'atype' list to Integer Indices
    # return_inverse gives indices (0, 1, 0...) mapping to unique types
    unique_types, type_indices = np.unique(atype, return_inverse=True)
    
    # CRITICAL: This must be a standard Python int, not a JAX array
    num_types = int(len(unique_types))
    
    type_indices = jnp.array(type_indices, dtype=jnp.int32)
    
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

# Metal does not support complex<f32> in GPU kernels, so we split U_k into
# real (A) and imaginary (B) parts and compute:
#   Sk_real = A^T A + B^T B
#   Sk_imag = B^T A - A^T B
# Both outputs are float32; the caller recombines them on CPU.
@partial(jit, static_argnums=(5,))
def process_batch_kernel(kvec, displacements_batch, cell_idx_batch, masses, type_indices, num_types):
    """Returns (Sk_real, Sk_imag) as float32 arrays of shape (3*num_types, 3*num_types)."""

    def single_frame_calc(disp, cell):
        # Phase: e^(i k·n) = cos(k·n) + i sin(k·n)
        dot_products = jnp.dot(cell, kvec)          # (N,)
        cos_p = jnp.cos(dot_products)               # (N,)
        sin_p = jnp.sin(dot_products)               # (N,)

        sqrt_m = jnp.sqrt(masses)                   # (N,)

        # Weighted displacements split into real and imaginary contributions
        wd_real = disp * (sqrt_m * cos_p)[:, None]  # (N, 3)
        wd_imag = disp * (sqrt_m * sin_p)[:, None]  # (N, 3)

        # Segment sum by atom type — use float32 ones to keep counts in float32
        ones_f32 = jnp.ones(type_indices.shape[0], dtype=jnp.float32)
        sum_real = segment_sum(wd_real, type_indices, num_segments=num_types)  # (T, 3)
        sum_imag = segment_sum(wd_imag, type_indices, num_segments=num_types)
        counts   = segment_sum(ones_f32, type_indices, num_segments=num_types) # (T,) float32

        norm = (jnp.float32(1.0) / jnp.sqrt(jnp.maximum(counts, jnp.float32(1.0))))[:, None]

        A = (sum_real * norm).reshape(-1)   # real part of U_k, shape (3T,)
        B = (sum_imag * norm).reshape(-1)   # imag part of U_k, shape (3T,)

        # Sk = U.T @ U.conj,  U = A + iB
        Sk_real = jnp.outer(A, A) + jnp.outer(B, B)
        Sk_imag = jnp.outer(B, A) - jnp.outer(A, B)
        return Sk_real, Sk_imag

    batch_real, batch_imag = vmap(single_frame_calc, in_axes=(0, 0))(
        displacements_batch, cell_idx_batch)
    return jnp.sum(batch_real, axis=0), jnp.sum(batch_imag, axis=0)


# --- 2. The Driver Function ---

def Sk_avg(fpath, hsym_config, atom_dic, dim, kpnt, v_super,
           loadfile=True, save=True, batch_size=50):
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
        type_indices_gpu = jnp.array(type_indices_cpu, dtype=jnp.int32)
        
        # CRITICAL: Ensure this is a standard Python int
        num_types = int(len(unique_types))
        
        kvec_gpu = jnp.array(kpnt, dtype=jnp.float32)
        
        # Calculate matrix dimension size for initialization
        dim_size = num_types * 3

    # Accumulate in float64 on CPU to avoid rounding error when summing many float32 batches
    Sk_sum_real = np.zeros((dim_size, dim_size), dtype=np.float64)
    Sk_sum_imag = np.zeros((dim_size, dim_size), dtype=np.float64)

    if loadfile and os.path.exists(saved_Sk_path):
        print(f"Loading from {saved_Sk_path}...")
        try:
            with open(saved_Sk_path, 'r') as file:
                reader = csv.reader(file)
                header = next(reader)
                if int(header[0]) <= len(fnames):
                    start_idx = int(header[0])
                    Sk_loaded = np.array([[complex(x) for x in row] for row in reader])
                    Sk_sum_real = np.real(Sk_loaded).astype(np.float64)
                    Sk_sum_imag = np.imag(Sk_loaded).astype(np.float64)
        except Exception:
            print("Load failed, starting fresh.")

    # Main Batch Loop
    for i in tqdm(range(start_idx, len(fnames), batch_size), desc=f'Processing Batches', disable=True):
        batch_files = fnames[i : i + batch_size]

        disp_list = []
        cell_list = []

        for fname in batch_files:
            _, config, cell_idx = Readers.read_frac_atom_ph(fname, atom_dic, dim)
            disp = (config - hsym_config[1]) / dim @ v_super  # Cartesian Å
            disp_list.append(disp)
            cell_list.append(cell_idx)

        disp_batch_gpu = jnp.array(np.stack(disp_list), dtype=jnp.float32)
        cell_batch_gpu = jnp.array(np.stack(cell_list), dtype=jnp.float32)

        batch_real, batch_imag = process_batch_kernel(
            kvec_gpu, disp_batch_gpu, cell_batch_gpu,
            masses_gpu, type_indices_gpu, num_types)

        Sk_sum_real += np.array(batch_real, dtype=np.float64)
        Sk_sum_imag += np.array(batch_imag, dtype=np.float64)

        if (i + len(batch_files)) % 500 == 0 and save:
            Sk_save = Sk_sum_real.astype(np.complex128) + 1j * Sk_sum_imag
            with open(saved_Sk_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([i + len(batch_files)])
                writer.writerows(Sk_save)

    Sk_sum = Sk_sum_real.astype(np.complex128) + 1j * Sk_sum_imag

    if save:
        with open(saved_Sk_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([len(fnames)])
            writer.writerows(Sk_sum)

    return Sk_sum / len(fnames)

def Partial_Sk_avg(fpath, hsym_config, atom_dic, dim, kpnt, atype, v_super,
                   loadfile=True, save=True, batch_size=50):
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
    masses_gpu = jnp.full((num_target_atoms,), mass_val, dtype=jnp.float32)
    
    # All atoms belong to group "0"
    type_indices_gpu = jnp.zeros(num_target_atoms, dtype=jnp.int32)
    num_types = 1
    
    kvec_gpu = jnp.array(kpnt, dtype=jnp.float32)

    # --- 2. Load Previous Progress ---
    Sk_sum_real = np.zeros((3, 3), dtype=np.float64)
    Sk_sum_imag = np.zeros((3, 3), dtype=np.float64)

    if loadfile and os.path.exists(saved_Sk_path):
        print(f"Loading partial progress from {saved_Sk_path}...")
        try:
            with open(saved_Sk_path, 'r') as file:
                reader = csv.reader(file)
                header = next(reader)
                if int(header[0]) <= len(fnames):
                    start_idx = int(header[0])
                    Sk_loaded = np.array([[complex(x) for x in row] for row in reader])
                    Sk_sum_real = np.real(Sk_loaded).astype(np.float64)
                    Sk_sum_imag = np.imag(Sk_loaded).astype(np.float64)
        except Exception:
            print("Load failed, starting fresh.")

    # --- 3. Batch Processing Loop ---
    for i in tqdm(range(start_idx, len(fnames), batch_size), desc=f'Partial Sk ({atype})'):
        batch_files = fnames[i : i + batch_size]

        disp_list = []
        cell_list = []

        for fname in batch_files:
            test = Readers.read_frac_atom_ph(fname, atom_dic, dim, atype)
            disp = (test[1] - hsym_ref_subset) / dim @ v_super  # Cartesian Å
            disp_list.append(disp)
            cell_list.append(test[2])

        disp_batch_gpu = jnp.array(np.stack(disp_list), dtype=jnp.float32)
        cell_batch_gpu = jnp.array(np.stack(cell_list), dtype=jnp.float32)

        batch_real, batch_imag = process_batch_kernel(
            kvec_gpu, disp_batch_gpu, cell_batch_gpu,
            masses_gpu, type_indices_gpu, num_types)

        Sk_sum_real += np.array(batch_real, dtype=np.float64)
        Sk_sum_imag += np.array(batch_imag, dtype=np.float64)

        if (i + len(batch_files)) % 500 == 0 and save:
            Sk_save = Sk_sum_real.astype(np.complex128) + 1j * Sk_sum_imag
            with open(saved_Sk_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([i + len(batch_files)])
                writer.writerows(Sk_save)

    # --- 4. Final Save ---
    Sk_sum = Sk_sum_real.astype(np.complex128) + 1j * Sk_sum_imag

    if save:
        with open(saved_Sk_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([len(fnames)])
            writer.writerows(Sk_sum)

    return Sk_sum / len(fnames)

def get_nxs_array(atom_idx, atom_dic):
    '''Return neutron total scattering cross-section [barn] for each type in atom_idx.'''
    reverse_mapping = {v: k for k, values in atom_dic.items() for v in values}
    return [NEUTRON_SCATT_SIGMA[reverse_mapping[tid]] for tid in atom_idx]

def select_atom_type(tag, atype, config, cell_idx):
    sel_idx = []
    for ii in np.arange(len(atype)):
        if atype[ii] == tag:
            sel_idx.append(ii)
    return config[sel_idx], cell_idx[sel_idx]

def eigenvalues_to_meV(eigenvalues, T):
    '''Convert S(k) eigenvalues [amu·Å²] to phonon energies [meV].

    Identical to the CPU version. Negative eigenvalues (soft modes) are
    returned as negative energies. Zero eigenvalues map to 0.
    '''
    ev = np.asarray(eigenvalues, dtype=float)
    threshold = 1e-4
    valid = np.abs(ev) >= threshold
    safe = np.where(valid, np.abs(ev), np.nan)
    energies = ENERGY_CONV * np.sqrt(T / safe)
    energies = np.where(np.isnan(energies), 0.0, energies)
    return np.where(ev >= 0, energies, -energies)


def gen_grid(n_points=5):
    q = np.linspace(-0.5, 0.5, n_points, endpoint=False)
    return np.array(np.meshgrid(q, q, q)).T.reshape(-1, 3)

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