import numpy as np
import glob
import jax
import jax.numpy as jnp
from tqdm import trange
from collections import defaultdict

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
    atom_type = arr[:, 0].astype(np.int64)

    if atype == 0:
        mask = np.ones(atom_type.shape[0], dtype=bool)
    else:
        allowed = np.array(list(atom_dic[atype]), dtype=np.int64)
        mask = np.isin(atom_type, allowed)

    atom_type = atom_type[mask]
    xyz = arr[mask, 1:4].astype(np.float64) * np.asarray(dim, dtype=np.float64)
    cell_idx = arr[mask, -3:].astype(np.int64)

    # Your original wrap: x-dim[0] if x > 1 else x
    # This is odd because you compare to 1 after scaling by dim.
    # Keeping behavior identical:
    xyz = np.where(xyz > 1.0, xyz - dim[0], xyz)

    return atom_type.tolist(), xyz, cell_idx


def avg_frac_atom_ph(fnames, atom_dic, dim, atype=0, mode="Frac", dtype=np.float64):
    """Calculate average configuration from multiple files (CPU mean with numpy).

    Metal does not support float64, so the mean is computed on CPU with numpy.
    GPU acceleration is reserved for the heavier Sk_avg computation.
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

    data_stack_np = np.stack(data_list, axis=0)
    data_avg_np = np.mean(data_stack_np, axis=0, dtype=dtype)

    return atmtype, data_avg_np, np.asarray(cell_tmp)
