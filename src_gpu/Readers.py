import numpy as np
import glob
import jax
import jax.numpy as jnp
from tqdm import trange
from collections import defaultdict

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


def read_cell_vec(fname, verbose=1):
    """Read lattice vectors and supercell dimension from a *.rmc6f (streaming, robust)."""
    if verbose:
        print(f"📖 Reading cell information from {fname}")

    v1 = v2 = v3 = None
    dim = None

    with open(fname, "r") as f:
        lines_iter = iter(f)
        for line in lines_iter:
            s = line.strip()
            if not s:
                continue

            # Tokenize once per candidate line
            parts = s.split()
            key = parts[0]

            if key == "Supercell":
                # Expect last 3 tokens are dimensions
                try:
                    dim = np.array(parts[-3:], dtype=np.float64)
                    if verbose:
                        print(f"Supercell dimensions = {dim}")
                except ValueError:
                    raise ValueError(f"Failed to parse Supercell dimensions from line: {s}")

            elif key == "Lattice":
                # Next three lines contain the lattice vectors
                try:
                    v1 = np.fromstring(next(lines_iter), sep=" ", dtype=np.float64)
                    v2 = np.fromstring(next(lines_iter), sep=" ", dtype=np.float64)
                    v3 = np.fromstring(next(lines_iter), sep=" ", dtype=np.float64)
                except StopIteration:
                    raise ValueError("Unexpected end of file while reading Lattice vectors.")
                except ValueError:
                    raise ValueError("Failed to parse Lattice vectors as floats.")

                if verbose:
                    print("Lattice vectors:\n", v1, "\n", v2, "\n", v3)

            # Early exit if we have everything
            if (v1 is not None) and (v2 is not None) and (v3 is not None) and (dim is not None):
                break

    if dim is None:
        dim = np.array([], dtype=np.float64)  # or raise, depending on your expectation

    return v1, v2, v3, dim

def get_atom_idx(fname, verbose=1):
    """Get mapping of atom symbols to unique indices (streaming, set-based)."""
    if verbose:
        print(f"📖 Reading atom indices from {fname}")

    atom_sets = defaultdict(set)
    in_atoms_block = False

    with open(fname, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            if not in_atoms_block:
                # Avoid split unless needed
                if s.startswith("Atoms:"):
                    in_atoms_block = True
                continue

            # Now we're in the atoms block
            parts = s.split()
            # Need at least: ... atom_symbol ... and enough columns for ln[-4]
            if len(parts) < 5:
                continue

            atom = parts[1]
            try:
                atom_idx = int(parts[-4])
            except ValueError:
                continue

            atom_sets[atom].add(atom_idx)

    # Convert sets to sorted lists for deterministic output
    return {atom: sorted(idxs) for atom, idxs in atom_sets.items()}


def read_frac_atom_ph(fname, atom_dic, dim, atype=0, mode="Frac"):
    if mode != "Frac":
        raise NotImplementedError("Only mode='Frac' supported in this fast path.")

    # Skip header (first 5 lines)
    # Use float for everything; we'll cast columns as needed.
    arr = np.loadtxt(fname, skiprows=5)

    # Column assumptions based on your code:
    # col0 = atom_type
    # col1:4 = fractional coords
    # last 3 cols = cell indices
    atom_type = arr[:, 0].astype(np.int32)

    if atype == 0:
        mask = np.ones(atom_type.shape[0], dtype=bool)
    else:
        allowed = np.array(list(atom_dic[atype]), dtype=np.int32)
        mask = np.isin(atom_type, allowed)

    atom_type = atom_type[mask]
    xyz = arr[mask, 1:4].astype(np.float64) * np.asarray(dim, dtype=np.float64)
    cell_idx = arr[mask, -3:].astype(np.int32)

    # Your original wrap: x-dim[0] if x > 1 else x
    # This is odd because you compare to 1 after scaling by dim.
    # Keeping behavior identical:
    xyz = np.where(xyz > 1.0, xyz - dim[0], xyz)

    return atom_type.tolist(), xyz, cell_idx


def avg_frac_atom_ph(fnames, atom_dic, dim, atype=0, mode="Frac", dtype=jnp.float32):
    """Calculate average configuration from multiple files (GPU reduction with JAX).
    
    Notes:
      - File I/O stays on CPU (read_frac_atom_ph).
      - Numeric reduction (mean) runs on GPU via JAX/Metal.
    """
    data_list = []
    cell_tmp = None
    atmtype = None

    for fidx in trange(len(fnames), desc="📊 Calculating average configuration", disable=False):
        atmtype, data, cell_idx = read_frac_atom_ph(fnames[fidx], atom_dic, dim, atype, mode)

        data_np = np.asarray(data)
        cell_np = np.asarray(cell_idx)

        if fidx > 1 and cell_tmp is not None:
            if cell_tmp.shape != cell_np.shape or not (cell_tmp == cell_np).all():
                raise ValueError("The cell indices do not match ... Please check ...")

        cell_tmp = cell_np
        data_list.append(data_np)

    # Stack once on CPU, then move once to GPU
    data_stack_np = np.stack(data_list, axis=0)  # shape: (nfiles, ...)

    data_stack = jnp.asarray(data_stack_np, dtype=dtype)  # device put (GPU)
    data_avg = jnp.mean(data_stack, axis=0)

    # If you want a NumPy array returned (CPU), convert back:
    data_avg_np = np.asarray(jax.device_get(data_avg))

    return atmtype, data_avg_np, np.asarray(cell_tmp)



# def read_cell_vec(fname, verbose=1):
#     '''Read cell vectors and supercell dimension from a *.rmc6f'''
#     if verbose != 0:
#         print("📖 Reading cell information from {}".format(fname))
#     with open(fname, 'r') as f:
#         lines = f.readlines()
    
#     v1, v2, v3, dim = None, None, None, None
    
#     for ii in np.arange(len(lines)):
#         if lines[ii].split()[0] == 'Lattice':
#             v1 = np.array(np.float64(lines[ii+1].split()))
#             v2 = np.array(np.float64(lines[ii+2].split()))
#             v3 = np.array(np.float64(lines[ii+3].split()))
#             if verbose != 0:
#                 print('Lattice vectors:\n', v1, '\n', v2, '\n', v3)
#             break
#         elif lines[ii].split()[0] == 'Supercell':
#             dim = np.float64(lines[ii].split()[-3:])
#             if verbose != 0:
#                 print('Supercell dimensions = {}'.format(dim))
                
#     return v1, v2, v3, np.array(dim)

# def get_atom_idx(fname, verbose=1):
#     '''Get mapping of atom types to indices'''
#     if verbose != 0:
#         print("📖 Reading atom indices from {}".format(fname))
#     with open(fname, 'r') as f:
#         lines = f.readlines()
        
#     atom_dic = {}
#     idx_ini = 0
#     for ii in np.arange(len(lines)):
#         if lines[ii].split()[0] == 'Atoms:':
#             idx_ini = ii
#             break
            
#     for ii in np.arange(idx_ini+1, len(lines), 1):
#         ln = lines[ii].split()
#         atom = ln[1]
#         atom_idx = int(ln[-4])
#         if atom not in atom_dic:
#             atom_dic[atom] = [atom_idx]
#         else:
#             atom_dic[atom].append(atom_idx)
            
#     for key in atom_dic:
#         atom_dic[key] = list(set(atom_dic[key]))
#     return atom_dic

# CPU version
# def read_frac_atom_ph(fname, atom_dic, dim, atype=0, mode='Frac', v1_norm=None, v2_norm=None, v3_norm=None):
#     '''Read fractional coordinates from Frac*.txt'''
#     with open(fname, 'r') as f:
#         lines = f.readlines()
        
#     atmtype = []
#     data = []
#     cell_idx = []
    
#     for ii in np.arange(5, len(lines), 1):
#         ln = lines[ii].split()
#         current_atom_type = int(ln[0])
        
#         # Check if we should process this atom
#         process = False
#         if atype == 0:
#             process = True
#         elif current_atom_type in atom_dic[atype]:
#             process = True
            
#         if process:
#             atmtype.append(current_atom_type)
#             xyz = np.array(np.float64(ln[1:4])) * dim
#             xyz = np.array([x-dim[0] if x > 1 else x for x in xyz])
#             cell = np.array([int(ln[-3]), int(ln[-2]), int(ln[-1])])
            
#             if mode == 'Frac':
#                 data.append(xyz)
#                 cell_idx.append(cell)
#             else:
#                 # NOTE: cvt_pos was not defined in the original script provided.
#                 # You must define cvt_pos or ensure mode is always 'Frac'
#                 # xyz = np.array(cvt_pos(xyz, v1_norm, v2_norm, v3_norm))
#                 pass 
                
#     return atmtype, np.array(data), np.array(cell_idx)

# CPU version
# def avg_frac_atom_ph(fnames, atom_dic, dim, atype=0, mode='Frac'):
#     '''Calculate average configuration from multiple files'''
#     data_accum = None
#     cell_tmp = None
    
#     for fidx in trange(len(fnames), desc='📊 Calculating average configuration', disable=False):
#         atmtype, data, cell_idx = read_frac_atom_ph(fnames[fidx], atom_dic, dim, atype, mode)
        
#         if fidx == 0:
#             data_accum = np.array(data)
#         else:
#             data_accum += np.array(data)
            
#         if fidx > 1 and cell_tmp is not None:
#              if cell_tmp.shape != np.array(cell_idx).shape or not (cell_tmp == np.array(cell_idx)).all():
#                 print('The cell indices do not match ... Please check ...')
#         cell_tmp = np.array(cell_idx)
        
#     data_avg = np.array(data_accum) / len(fnames)
#     return atmtype, np.array(data_avg), np.array(cell_tmp)


