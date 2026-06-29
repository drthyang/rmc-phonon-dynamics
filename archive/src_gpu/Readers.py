import numpy as np
import glob
import os
import re
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

    xyz = np.mod(xyz, np.asarray(dim, dtype=np.float64))  # wrap each component to [0, dim[i]) independently

    return atom_type.tolist(), xyz, cell_idx


def read_rmc6f_atom_ph(fname, atom_dic, dim, atype=0):
    """Read equilibrium positions from a *.rmc6f into the SAME layout as
    read_frac_atom_ph — so a chosen equilibrium structure can serve as the
    displacement reference (hsym) instead of the ensemble average.

    rmc6f "Atoms:" line:  id element [type] fx fy fz RN Nx Ny Nz
      coords   = parts[-7:-4]  (GLOBAL supercell fractional)
      RN       = parts[-4]     (reference number; same column get_atom_idx uses)
      cell_idx = parts[-3:]    (Nx Ny Nz)
    End-anchored so it is robust to variation in the leading columns.

    Returns (atom_type_list, xyz, cell_idx) matching read_frac_atom_ph EXACTLY,
    including its WITHIN-CELL coordinate convention: read_frac_atom_ph's Frac
    X/Y/Z are within-unit-cell fractions (the integer cell offset lives only in
    Nx,Ny,Nz), so xyz = frac*dim stays in [0,1) per component. The rmc6f stores
    GLOBAL supercell fractions (cell offset folded in), so we strip it with
    mod 1.0 to land in the same within-cell frame. This MUST match, or the
    per-atom subtraction in Sk_avg ((config - hsym)) picks up spurious integer
    cell offsets. Verified equal to read_frac_atom_ph to ~4e-5 (Frac rounding)
    on paired configs — see validate_rmc6f_equiv.py.
    """
    rn_list, xyz_list, cell_list = [], [], []
    in_atoms = False
    with open(fname, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if not in_atoms:
                if s.startswith("Atoms:"):
                    in_atoms = True
                continue
            parts = s.split()
            if len(parts) < 7:
                continue
            try:
                rn = int(parts[-4])
                cell = [int(parts[-3]), int(parts[-2]), int(parts[-1])]
                frac = [float(parts[-7]), float(parts[-6]), float(parts[-5])]
            except (ValueError, IndexError):
                continue
            rn_list.append(rn)
            xyz_list.append(frac)
            cell_list.append(cell)

    if not rn_list:
        raise ValueError(f"No atoms parsed from {fname} (expected an 'Atoms:' block).")

    atom_type = np.asarray(rn_list, dtype=np.int64)
    # Global supercell coordinate in unit-cell units, then strip the integer
    # cell offset (mod 1.0) to match read_frac_atom_ph's within-cell frame.
    xyz = np.mod(np.asarray(xyz_list, dtype=np.float64) * np.asarray(dim, dtype=np.float64), 1.0)
    cell_idx = np.asarray(cell_list, dtype=np.int64)

    if atype != 0:
        allowed = np.array(list(atom_dic[atype]), dtype=np.int64)
        mask = np.isin(atom_type, allowed)
        atom_type, xyz, cell_idx = atom_type[mask], xyz[mask], cell_idx[mask]

    return atom_type.tolist(), xyz, cell_idx


def read_config_atom_ph(fname, atom_dic, dim, atype=0):
    """Read ONE configuration into (atom_type, within-cell xyz, cell_idx),
    dispatching on file type so callers don't care about the source format:
      *.rmc6f -> read_rmc6f_atom_ph   (read-only source files)
      else     -> read_frac_atom_ph   (Frac*.txt)
    Both return the identical within-cell layout (verified to 4e-5 on paired
    snapshots; see validate_rmc6f_equiv.py).
    """
    if fname.lower().endswith('.rmc6f'):
        return read_rmc6f_atom_ph(fname, atom_dic, dim, atype)
    return read_frac_atom_ph(fname, atom_dic, dim, atype)


# A configuration .rmc6f is <stem>_<N>.rmc6f with N>=1; the running-average
# dumps end in 'AVERAGE.rmc6f' and are NOT samples; <stem>.rmc6f (no index) is
# the base/structure file and <stem>_0.rmc6f is the initial structure (identical
# to the base) — all excluded from the ensemble.
_RMC6F_AVERAGE_RE = re.compile(r'AVERAGE\.rmc6f$', re.IGNORECASE)
_RMC6F_INDEX_RE = re.compile(r'_(\d+)\.rmc6f$', re.IGNORECASE)


def list_configs(path):
    """Return (sorted_config_files, family) for an ensemble folder, auto-detecting:

      'rmc6f' : numbered <stem>_<N>.rmc6f (N>=1) in `path`, EXCLUDING
                *AVERAGE.rmc6f, the un-numbered base, and _0 (initial structure).
      'frac'  : Frac*.txt in `path/configs/` (canonical) or directly in `path`.
      'none'  : neither found.

    Detection order matters:
      - A numbered .rmc6f ensemble wins (reading source .rmc6f directly is the
        point of this path), so pointing at the ensemble folder uses it even
        when a derived configs/ also exists.
      - For Frac, `configs/` is checked BEFORE `path`, because a lone orphan
        Frac*.txt can sit in an ensemble parent dir and must not be mistaken for
        the real (configs/) ensemble — averaging that one file gives garbage.

    These .rmc6f are read-only source files — this only lists them.
    """
    path = path.rstrip('/')

    numbered = []
    for p in glob.glob(os.path.join(path, '*.rmc6f')):
        base = os.path.basename(p)
        if _RMC6F_AVERAGE_RE.search(base):
            continue
        m = _RMC6F_INDEX_RE.search(base)
        if not m:                 # un-numbered base / structure file
            continue
        n = int(m.group(1))
        if n < 1:                 # _0 = initial structure (== base)
            continue
        numbered.append((n, p))
    if numbered:
        numbered.sort()
        return [p for _, p in numbered], 'rmc6f'

    for d in (os.path.join(path, 'configs'), path):
        frac = sorted(glob.glob(os.path.join(d, 'Frac*.txt')))
        if frac:
            return frac, 'frac'

    return [], 'none'


def avg_frac_atom_ph(fnames, atom_dic, dim, atype=0, mode="Frac", dtype=np.float64):
    """Calculate average configuration from multiple files (CPU mean with numpy).

    Metal does not support float64, so the mean is computed on CPU with numpy.
    GPU acceleration is reserved for the heavier Sk_avg computation.
    """
    data_list = []
    cell_tmp = None
    atmtype = None

    for fidx in trange(len(fnames), desc="📊 Calculating average configuration", disable=False):
        atmtype, data, cell_idx = read_config_atom_ph(fnames[fidx], atom_dic, dim, atype)

        data_np = np.asarray(data)
        cell_np = np.asarray(cell_idx)

        if fidx > 0 and cell_tmp is not None:
            if cell_tmp.shape != cell_np.shape or not (cell_tmp == cell_np).all():
                raise ValueError("The cell indices do not match ... Please check ...")

        cell_tmp = cell_np
        data_list.append(data_np)

    data_stack_np = np.stack(data_list, axis=0)
    data_avg_np = np.mean(data_stack_np, axis=0, dtype=dtype)

    return atmtype, data_avg_np, np.asarray(cell_tmp)
