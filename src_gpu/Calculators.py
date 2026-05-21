import os
os.environ["JAX_ENABLE_X64"] = "True"  # must be set before any JAX import

import numpy as np
import csv
import glob
from functools import partial
from tqdm import tqdm
import Readers
from constants import ATOMIC_MASS, NEUTRON_SCATT_SIGMA, ENERGY_CONV

import jax.numpy as jnp
from jax import jit, vmap
import jax

# Import segment_sum (location varies by JAX version)
try:
    from jax.ops import segment_sum
except ImportError:
    from jax.numpy import segment_sum


# ── GPU mass lookup table ─────────────────────────────────────────────────────

# Global cache: built once on first call, then reused for all subsequent frames.
_GPU_MASS_TABLE = None

def get_mass_array(atom_idx, atom_dic):
    """Return a JAX array of masses [amu] for each integer atom ID in atom_idx.

    Builds a GPU lookup table on the first call (indexed by integer atom ID),
    then uses fast vectorised indexing for all subsequent calls.
    """
    global _GPU_MASS_TABLE

    if _GPU_MASS_TABLE is None:
        print("Initializing GPU Mass Table (first run only)...")
        all_indices = [idx for indices in atom_dic.values() for idx in indices]
        max_idx = max(all_indices) if all_indices else 0
        # float32: Metal does not support float64 in GPU kernels
        lut_cpu = np.zeros(max_idx + 1, dtype=np.float32)
        for symbol, indices in atom_dic.items():
            lut_cpu[indices] = ATOMIC_MASS.get(symbol, 0.0)
        _GPU_MASS_TABLE = jnp.array(lut_cpu)

    return _GPU_MASS_TABLE[jnp.array(atom_idx)]


# ── GPU batch kernel ──────────────────────────────────────────────────────────

# Metal does not support complex<f32> in GPU kernels, so U_k is split into
# real (A) and imaginary (B) parts:
#   Sk_real = A^T A + B^T B
#   Sk_imag = B^T A - A^T B
# Both parts are float32; the caller recombines them on CPU.
@partial(jit, static_argnums=(5,))
def process_batch_kernel(kvec, displacements_batch, cell_idx_batch, masses, type_indices, num_types):
    """Accumulate S(k) contributions for a batch of frames.

    Returns (Sk_real, Sk_imag) as float32 arrays of shape (3*num_types, 3*num_types).
    num_types must be a static Python int so JAX can infer output shapes at compile time.
    """

    def single_frame_calc(disp, cell):
        dot_products = jnp.dot(cell, kvec)           # (N,)
        cos_p = jnp.cos(dot_products)                # (N,)
        sin_p = jnp.sin(dot_products)                # (N,)
        sqrt_m = jnp.sqrt(masses)                    # (N,)

        wd_real = disp * (sqrt_m * cos_p)[:, None]   # (N, 3)
        wd_imag = disp * (sqrt_m * sin_p)[:, None]   # (N, 3)

        ones_f32 = jnp.ones(type_indices.shape[0], dtype=jnp.float32)
        sum_real = segment_sum(wd_real, type_indices, num_segments=num_types)  # (T, 3)
        sum_imag = segment_sum(wd_imag, type_indices, num_segments=num_types)
        counts   = segment_sum(ones_f32, type_indices, num_segments=num_types) # (T,)

        norm = (jnp.float32(1.0) / jnp.sqrt(jnp.maximum(counts, jnp.float32(1.0))))[:, None]
        A = (sum_real * norm).reshape(-1)   # real part of U_k, shape (3T,)
        B = (sum_imag * norm).reshape(-1)   # imag part of U_k, shape (3T,)

        Sk_real = jnp.outer(A, A) + jnp.outer(B, B)
        Sk_imag = jnp.outer(B, A) - jnp.outer(A, B)
        return Sk_real, Sk_imag

    batch_real, batch_imag = vmap(single_frame_calc, in_axes=(0, 0))(
        displacements_batch, cell_idx_batch)
    return jnp.sum(batch_real, axis=0), jnp.sum(batch_imag, axis=0)


# ── Driver functions ──────────────────────────────────────────────────────────

def Sk_avg(fpath, hsym_config, atom_dic, dim, kpnt, v_super,
           loadfile=True, save=True, batch_size=50):
    """Compute the ensemble-averaged S(k) matrix for all atom types."""
    fnames = sorted(glob.glob(fpath + 'Frac*.txt'))
    saved_Sk_path = fpath + f'Sk_sum_kvec_{kpnt[0]}_{kpnt[1]}_{kpnt[2]}.csv'

    start_idx = 0

    # Initialise static JAX arrays from the first file (topology assumed fixed)
    if len(fnames) > 0:
        atype_static = Readers.read_frac_atom_ph(fnames[0], atom_dic, dim)[0]
        masses_gpu = jnp.array(get_mass_array(atype_static, atom_dic))
        unique_types, type_indices_cpu = np.unique(atype_static, return_inverse=True)
        type_indices_gpu = jnp.array(type_indices_cpu, dtype=jnp.int32)
        num_types = int(len(unique_types))   # must be a plain Python int for static_argnums
        kvec_gpu = jnp.array(kpnt, dtype=jnp.float32)
        dim_size = num_types * 3

    # Accumulate in float64 on CPU to limit rounding error across many float32 batches
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

    for i in tqdm(range(start_idx, len(fnames), batch_size), desc='Processing batches', disable=True):
        batch_files = fnames[i : i + batch_size]
        disp_list, cell_list = [], []
        for fname in batch_files:
            _, config, cell_idx = Readers.read_frac_atom_ph(fname, atom_dic, dim)
            disp_list.append((config - hsym_config[1]) / dim @ v_super)
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
    """Compute the ensemble-averaged S(k) matrix for a single atom type."""
    fnames = sorted(glob.glob(fpath + 'Frac*.txt'))
    saved_Sk_path = fpath + f'{atype}_Sk_sum_kvec_{kpnt[0]}_{kpnt[1]}_{kpnt[2]}.csv'

    start_idx = 0

    if atype not in atom_dic:
        raise ValueError(f"Atom type '{atype}' not found in atom_dic.")

    target_indices = atom_dic[atype]
    hsym_ref_subset = hsym_config[1][target_indices]    # (N_subset, 3)
    num_target_atoms = len(target_indices)

    # All atoms are the same type, so type_indices is all-zero and num_types=1
    mass_val = ATOMIC_MASS.get(atype, 0.0)
    masses_gpu = jnp.full((num_target_atoms,), mass_val, dtype=jnp.float32)
    type_indices_gpu = jnp.zeros(num_target_atoms, dtype=jnp.int32)
    num_types = 1

    kvec_gpu = jnp.array(kpnt, dtype=jnp.float32)

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

    for i in tqdm(range(start_idx, len(fnames), batch_size), desc=f'Partial Sk ({atype})'):
        batch_files = fnames[i : i + batch_size]
        disp_list, cell_list = [], []
        for fname in batch_files:
            frame = Readers.read_frac_atom_ph(fname, atom_dic, dim, atype)
            disp_list.append((frame[1] - hsym_ref_subset) / dim @ v_super)
            cell_list.append(frame[2])

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


# ── Utilities ─────────────────────────────────────────────────────────────────

def get_nxs_array(atom_idx, atom_dic):
    """Return neutron total scattering cross-section [barn] for each atom in atom_idx."""
    reverse_mapping = {v: k for k, values in atom_dic.items() for v in values}
    return [NEUTRON_SCATT_SIGMA[reverse_mapping[tid]] for tid in atom_idx]


def eigenvalues_to_meV(eigenvalues, T):
    """Convert S(k) eigenvalues [amu·Å²] to phonon energies [meV].

    Negative eigenvalues (soft modes) are returned as negative energies.
    Zero eigenvalues map to 0.
    """
    ev = np.asarray(eigenvalues, dtype=float)
    threshold = 1e-4
    valid = np.abs(ev) >= threshold
    safe = np.where(valid, np.abs(ev), np.nan)
    energies = ENERGY_CONV * np.sqrt(T / safe)
    energies = np.where(np.isnan(energies), 0.0, energies)
    return np.where(ev >= 0, energies, -energies)


def gen_grid(n_points=5):
    """Generate a uniform 3-D q-point grid over [-0.5, 0.5)^3."""
    q = np.linspace(-0.5, 0.5, n_points, endpoint=False)
    return np.array(np.meshgrid(q, q, q)).T.reshape(-1, 3)


# ── Archive (not used in current main workflow) ───────────────────────────────
# These functions are retained for reference and potential future use.

@jit
def _archive_calc_Sk(U_k_t):
    """Single-frame S(k) = U.T @ U.conj() via JAX matmul."""
    return jnp.matmul(U_k_t.T, U_k_t.conj())


@partial(jit, static_argnums=(5,))
def _archive_calc_collect_var_kernel(kvec, displacements, cell_idx, masses, type_indices, num_types):
    """Single-frame GPU kernel (complex path). Replaced by process_batch_kernel."""
    dot_products = jnp.dot(cell_idx, kvec)
    phase = jnp.exp(1j * dot_products)
    weights = jnp.sqrt(masses)[:, None] * phase[:, None]
    weighted_displacements = displacements * weights
    summed_terms = segment_sum(weighted_displacements, type_indices, num_segments=num_types)
    counts = segment_sum(jnp.ones_like(type_indices), type_indices, num_segments=num_types)
    norm_factors = 1.0 / jnp.sqrt(jnp.maximum(counts, 1.0))
    normalized_U = summed_terms * norm_factors[:, None]
    return normalized_U.reshape(1, -1)


def _archive_calc_collect_var(kvec, atype, configuration, cell_idx, hsymconfig, atom_dic):
    """Single-frame wrapper for _archive_calc_collect_var_kernel."""
    kvec = jnp.array(kvec, dtype=jnp.float32)
    displacements = jnp.array(configuration - hsymconfig, dtype=jnp.float32)
    cell_idx = jnp.array(cell_idx, dtype=jnp.float32)
    mass_array = get_mass_array(atype, atom_dic)
    masses = jnp.array(mass_array)
    unique_types, type_indices = np.unique(atype, return_inverse=True)
    num_types = int(len(unique_types))
    type_indices = jnp.array(type_indices, dtype=jnp.int32)
    return _archive_calc_collect_var_kernel(kvec, displacements, cell_idx, masses, type_indices, num_types)


def _archive_select_atom_type(tag, atype, config, cell_idx):
    """Return config and cell_idx rows where atype == tag."""
    sel_idx = [ii for ii in range(len(atype)) if atype[ii] == tag]
    return config[sel_idx], cell_idx[sel_idx]


def _archive_get_ph_weights(atom_dic, IRs):
    """Compute per-element phonon weights from eigenvectors IRs."""
    atom_types = list(atom_dic.keys())
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
