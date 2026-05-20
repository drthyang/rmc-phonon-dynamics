import numpy as np
import os

from scipy.optimize import linear_sum_assignment
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from constants import ATOMIC_MASS


def _safe_path(path):
    """Return path unchanged if it doesn't exist, otherwise append _1, _2, … until free."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    idx = 1
    while True:
        candidate = f'{base}_{idx}{ext}'
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def _degenerate_groups(freqs, tol):
    """Return list of index-groups whose frequencies are within tol of each other."""
    n = len(freqs)
    visited = [False] * n
    groups = []
    for i in range(n):
        if visited[i]:
            continue
        grp = [i]
        visited[i] = True
        for j in range(i + 1, n):
            if not visited[j] and np.isfinite(freqs[i]) and np.isfinite(freqs[j]):
                if abs(freqs[i] - freqs[j]) <= tol:
                    grp.append(j)
                    visited[j] = True
        groups.append(grp)
    return groups


def connect_bands(ph_band, eigenvectors_all,
                  degenerate_tol=1e-3, n_passes=2, freq_weight=0.0):
    """Reorder bands by eigenvector continuity (BAND_CONNECTION=.TRUE. equivalent).

    Algorithm (per q-step, repeated n_passes times):
      1. Hungarian assignment on a combined score:
            score = |overlap| * exp(-freq_weight * |Δω| / ω_scale)
         Pure eigenvector overlap when freq_weight=0; add frequency continuity
         as a tie-breaker by increasing freq_weight.
      2. Within each group of near-degenerate modes, SVD-rotate the eigenvectors
         to best align with the previous q-point's basis before finalising.
      Multiple passes improve reliability: pass 1 builds a globally consistent
      ordering; pass 2+ refines it using the now-smoother eigenvectors as reference.

    Parameters
    ----------
    ph_band          : list of 1-D arrays (n_modes,), one per q-point
    eigenvectors_all : list of (n_modes x n_modes) arrays, columns = eigenvectors,
                       as returned by np.linalg.eigh
    degenerate_tol   : float, default 1e-3
        Relative frequency tolerance for degenerate-subspace SVD rotation.
        Increase to 0.05–0.1 for densely packed spectra or avoided crossings.
        Set to 0 to disable.
    n_passes         : int, default 2
        Number of forward passes over the full q-path.
        Pass 1 gives approximate global continuity; pass 2+ refines ambiguous
        assignments that were locally suboptimal in the previous pass.
        Values of 2–3 are usually sufficient; rarely need more than 4.
    freq_weight      : float, default 0.0
        Weight for frequency-continuity penalty in the assignment score.
        score[i,j] *= exp(-freq_weight * |ω_prev_i - ω_curr_j| / ω_scale)
        Try 1–5 to penalise large frequency jumps when eigenvector overlaps
        are ambiguous (dense spectrum, noisy RMC data).
        Too large a value can override genuine band crossings — use cautiously.

    Returns
    -------
    ph_band_conn, eigvecs_conn : same structure, with columns/entries reordered
    """
    # Work on copies so the originals are not mutated
    current_band   = [f.copy() for f in ph_band]
    current_eigvec = [e.copy() for e in eigenvectors_all]

    for _ in range(n_passes):
        new_band   = [current_band[0].copy()]
        new_eigvec = [current_eigvec[0].copy()]

        for qi in range(1, len(current_band)):
            ev_prev     = new_eigvec[qi - 1]           # already reordered this pass
            ev_curr     = current_eigvec[qi].copy()
            freqs_prev  = new_band[qi - 1]
            freqs_curr  = current_band[qi]

            # ── Step 1: combined assignment score ────────────────────────────
            overlap = np.abs(ev_prev.conj().T @ ev_curr)

            if freq_weight > 0 and np.any(np.isfinite(freqs_curr)):
                freq_scale  = np.nanmax(np.abs(freqs_curr[np.isfinite(freqs_curr)]))
                freq_scale  = max(freq_scale, 1e-12)
                # |ω_prev[i] - ω_curr[j]| for all (i,j) pairs
                delta_freq  = np.abs(freqs_prev[:, None] - freqs_curr[None, :])
                overlap     = overlap * np.exp(-freq_weight * delta_freq / freq_scale)

            _, col_ind = linear_sum_assignment(-overlap)
            ev_curr    = ev_curr[:, col_ind]
            freqs_curr = freqs_curr[col_ind]

            # ── Step 2: SVD rotation within degenerate subspaces ─────────────
            if degenerate_tol > 0 and np.any(np.isfinite(freqs_curr)):
                freq_scale = np.nanmax(np.abs(freqs_curr[np.isfinite(freqs_curr)]))
                tol = degenerate_tol * max(freq_scale, 1e-12)
                for grp in _degenerate_groups(freqs_curr, tol):
                    if len(grp) < 2:
                        continue
                    sub_p = ev_prev[:, grp]
                    sub_c = ev_curr[:, grp]
                    U, _, Vh = np.linalg.svd(sub_p.conj().T @ sub_c)
                    R = Vh.conj().T @ U.conj().T
                    ev_curr[:, grp] = sub_c @ R

            new_band.append(freqs_curr)
            new_eigvec.append(ev_curr)

        current_band   = new_band
        current_eigvec = new_eigvec

    return current_band, current_eigvec


def gen_vasp_phonon(atom_dic, hsym_test, v1, v2, v3, dim,
                    eigenvectors=None, mode_indices=None,
                    amplitude=0.03, out_dir='../results/VASP/'):
    '''Write VASP input files (POSCAR, INCAR, KPOINTS) for phonon calculations.

    Eigenvectors follow the row-indexing convention of gen_ev_mcif:
    eigenvectors[i] is the i-th mode displacement vector of shape (3*N_atoms,).
    Displaced POSCARs are written only when eigenvectors is provided.
    '''
    atom_type_list, xyz, _ = hsym_test
    os.makedirs(out_dir, exist_ok=True)

    # Supercell lattice matrix (rows = lattice vectors)
    v_super = np.array([v1 / dim[0], v2 / dim[1], v3 / dim[2]])

    # Element names and per-element counts, preserving atom_dic insertion order
    elements = list(atom_dic.keys())
    counts = [len(atom_dic[el]) for el in elements]

    # Map each atom index to its row in hsym_test arrays
    atom_type_arr = np.array(atom_type_list)
    file_order = {int(idx): int(np.where(atom_type_arr == idx)[0][0])
                  for el in elements for idx in atom_dic[el]}

    # Fractional coords in POSCAR element order, wrapped to [0, 1).
    # xyz is frac_in_file * dim = unit-cell fractional coord; do NOT divide by dim again.
    frac_eq = np.array([
        xyz[file_order[idx]] % 1.0
        for el in elements for idx in atom_dic[el]
    ])

    _write_poscar(out_dir + 'POSCAR', v_super, elements, counts, frac_eq)

    if eigenvectors is not None:
        # Sk eigenvector rows are ordered by np.unique(atom_type_list) — i.e. sorted atom indices
        all_sorted_indices = sorted(int(idx) for el in elements for idx in atom_dic[el])
        rank = {atom_idx: r for r, atom_idx in enumerate(all_sorted_indices)}

        if mode_indices is None:
            mode_indices = np.arange(eigenvectors.shape[0])

        v_inv = np.linalg.inv(v_super)

        for mode_idx in mode_indices:
            eigvec_flat = np.real(eigenvectors[mode_idx])   # shape (3*N,)

            # Displacement for each atom in POSCAR order, in fractional supercell coords
            disp_frac = np.array([
                eigvec_flat[rank[idx] * 3: rank[idx] * 3 + 3] @ v_inv
                for el in elements for idx in atom_dic[el]
            ])

            frac_disp = (frac_eq + amplitude * disp_frac) % 1.0
            fname = out_dir + f'POSCAR_mode_{mode_idx}'
            comment = f'Mode {mode_idx} displaced amplitude={amplitude} A'
            _write_poscar(fname, v_super, elements, counts, frac_disp, comment=comment)

    _write_incar_template(out_dir + 'INCAR')
    _write_kpoints_template(out_dir + 'KPOINTS')
    print(f'VASP files written to {out_dir}')


def _write_poscar(fname, v_super, elements, counts, frac_coords, comment=None):
    formula = ''.join(f'{el}{n}' for el, n in zip(elements, counts))
    header = comment if comment else formula
    lines = [header, '1.0']
    for v in v_super:
        lines.append(f'  {v[0]:18.12f}  {v[1]:18.12f}  {v[2]:18.12f}')
    lines.append('  ' + '  '.join(elements))
    lines.append('  ' + '  '.join(str(n) for n in counts))
    lines.append('Direct')
    for pos in frac_coords:
        lines.append(f'  {pos[0]:18.12f}  {pos[1]:18.12f}  {pos[2]:18.12f}')
    with open(fname, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def _write_incar_template(fname):
    content = (
        '# VASP INCAR template for phonon calculation\n'
        '# ============================================\n'
        'SYSTEM  = phonon\n'
        'ISTART  = 0\n'
        'ICHARG  = 2\n'
        'ENCUT   = 400       # EDIT: match your POTCAR recommendations\n'
        'PREC    = Accurate\n'
        'EDIFF   = 1E-8\n'
        'NSW     = 0\n'
        'IBRION  = 8         # EDIT: 8=DFPT, 6=finite differences\n'
        'NFREE   = 4         # only used if IBRION=6\n'
        'POTIM   = 0.015     # only used if IBRION=6\n'
        'ISMEAR  = 0\n'
        'SIGMA   = 0.05\n'
        'NCORE   = 4         # EDIT: match your cluster setup\n'
        '# NOTE: POTCAR must be provided separately (VASP license required)\n'
    )
    with open(fname, 'w') as f:
        f.write(content)


def _write_kpoints_template(fname):
    content = (
        'Automatic k-mesh\n'
        '0\n'
        'Gamma\n'
        '  2  2  2    # EDIT: increase for production runs\n'
        '  0  0  0\n'
    )
    with open(fname, 'w') as f:
        f.write(content)

def _hsym_label(name, sym_labels):
    """Return the display label for a high-symmetry point name."""
    if sym_labels and name in sym_labels:
        return sym_labels[name]
    if name in ('GM', 'Gamma', 'GAMMA', 'G'):
        return '$\\Gamma$'
    return name


def gen_phonopy_band_yaml(atom_dic, hsym_test, v1, v2, v3, dim,
                          ph_band, eigenvectors_all,
                          k_path, sym_pnts, kstep,
                          freq_factor=1.0,
                          sym_labels=None,
                          out_dir='../results/'):
    '''Write a phonopy-compatible band.yaml from RMC phonon results.

    Parameters
    ----------
    ph_band          : list of 1-D arrays, one per k-point (same loop order as main.py)
    eigenvectors_all : list of (3N x 3N) complex arrays from np.linalg.eigh, one per k-point
    k_path           : list of label strings, e.g. ['GM', 'X', 'M', 'GM']
    sym_pnts         : dict label -> fractional reciprocal coordinates
    kstep            : number of q-points per segment
    freq_factor      : multiply all frequencies by this to convert to THz
    sym_labels       : optional dict overriding display labels, e.g. {'GM': '$\\Gamma$', 'X': 'X'}
                       'GM' -> '$\\Gamma$' is applied automatically if not overridden.

    Collect eigenvectors in main.py by adding:
        eigenvectors_all = []
        # inside the k-loop, after np.linalg.eigh:
        eigenvectors_all.append(eigenvectors)
    '''
    atom_type_list, xyz, _ = hsym_test

    # Supercell lattice (rows = vectors, Å)
    v_super = np.array([v1 / dim[0], v2 / dim[1], v3 / dim[2]])

    # Reciprocal lattice without 2π: B = inv(A).T  (Å^-1)
    recip = np.linalg.inv(v_super).T

    elements  = list(atom_dic.keys())
    counts    = [len(atom_dic[el]) for el in elements]
    n_atoms   = sum(counts)

    # Equilibrium fractional coords in element-grouped order.
    # xyz from read_frac_atom_ph is frac_in_file * dim, i.e. already the unit-cell
    # fractional coordinate in [0, 1). Do NOT divide by dim again.
    atom_type_arr = np.array(atom_type_list)
    file_order = {int(idx): int(np.where(atom_type_arr == idx)[0][0])
                  for el in elements for idx in atom_dic[el]}
    frac_eq = np.array([
        xyz[file_order[idx]] % 1.0
        for el in elements for idx in atom_dic[el]
    ])

    # Flat atom list preserving element-grouped order
    atom_list = [(el, int(idx)) for el in elements for idx in atom_dic[el]]

    # Eigenvector index: atom_idx -> row in np.unique(atom_type_list) ordering
    all_sorted = sorted(int(idx) for el in elements for idx in atom_dic[el])
    rank = {atom_idx: r for r, atom_idx in enumerate(all_sorted)}

    # Reconstruct k-points in the same order as ph_band (matching main.py loop).
    # Also record which flat q-point index is a high-symmetry point and its label.
    k_points  = []
    hsym_qi   = {}   # qi -> display label string
    for ii in range(len(k_path) - 1):
        k_start = sym_pnts[k_path[ii]]
        k_vec   = sym_pnts[k_path[ii + 1]] - k_start
        hsym_qi[len(k_points)] = _hsym_label(k_path[ii], sym_labels)
        for jj in range(kstep + 1):
            k_points.append(k_start + jj * k_vec / kstep)
    # Label the very last q-point with the endpoint of the final segment
    hsym_qi[len(k_points) - 1] = _hsym_label(k_path[-1], sym_labels)

    n_qpoints  = len(k_points)
    n_segments = len(k_path) - 1
    n_modes    = len(ph_band[0]) if ph_band else 0

    # Cumulative path distance in Å^-1 (no 2π)
    distances = [0.0]
    for i in range(1, n_qpoints):
        dq = (k_points[i] - k_points[i - 1]) @ recip
        distances.append(distances[-1] + float(np.linalg.norm(dq)))

    os.makedirs(out_dir, exist_ok=True)
    out_path = _safe_path(out_dir + 'band_gpu.yaml')

    lines = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append(f'nqpoint: {n_qpoints}')
    lines.append(f'npath: {n_segments}')
    lines.append('segment_nqpoint:')
    for _ in range(n_segments):
        lines.append(f'- {kstep + 1}')

    lines.append('reciprocal_lattice:')
    for rv, lab in zip(recip, ['a*', 'b*', 'c*']):
        lines.append(f'- [ {rv[0]:15.10f}, {rv[1]:15.10f}, {rv[2]:15.10f} ] # {lab}')

    lines.append(f'natom: {n_atoms}')
    lines.append('lattice:')
    for rv, lab in zip(v_super, ['a', 'b', 'c']):
        lines.append(f'- [ {rv[0]:15.10f}, {rv[1]:15.10f}, {rv[2]:15.10f} ] # {lab}')

    lines.append('points:')
    for i, (el, _) in enumerate(atom_list):
        pos  = frac_eq[i]
        mass = ATOMIC_MASS.get(el, 0.0)
        lines.append(f'- symbol: {el}')
        lines.append(f'  coordinates: [ {pos[0]:14.10f}, {pos[1]:14.10f}, {pos[2]:14.10f} ]')
        lines.append(f'  mass: {mass:.5f}')

    # ── Phonon data ───────────────────────────────────────────────────────────
    lines.append('phonon:')
    for qi in range(n_qpoints):
        kpt    = k_points[qi]
        dist   = distances[qi]
        freqs  = ph_band[qi]
        eigvecs = eigenvectors_all[qi] if eigenvectors_all is not None else None

        lines.append(f'- q-position: [ {kpt[0]:12.8f}, {kpt[1]:12.8f}, {kpt[2]:12.8f} ]')
        lines.append(f'  distance:   {dist:14.8f}')
        if qi in hsym_qi:
            lines.append(f'  label: \'{hsym_qi[qi]}\'')
        lines.append('  band:')

        for mode_idx in range(n_modes):
            raw_freq = float(freqs[mode_idx])
            freq = (raw_freq if np.isfinite(raw_freq) else 0.0) * freq_factor
            lines.append(f'  - # {mode_idx + 1}')
            lines.append(f'    frequency:  {freq:15.10f}')

            if eigvecs is not None:
                lines.append('    eigenvector:')
                # Column mode_idx = the mode_idx-th eigenvector from np.linalg.eigh
                ev_col = eigvecs[:, mode_idx]
                for j, (_, atom_idx) in enumerate(atom_list):
                    r = rank[atom_idx]
                    ev3 = ev_col[r * 3: r * 3 + 3]
                    lines.append(f'    - # atom {j + 1}')
                    for comp in ev3:
                        lines.append(f'      - [ {comp.real:15.10f}, {comp.imag:15.10f} ]')

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'band_gpu.yaml written to {out_path}')


def gen_ev_mcif(cifpath, atom_dic, vectors, eigen_num=np.arange(1), name=None):
    '''Generate MCIF files for eigenvectors'''
    structure = Structure.from_file(cifpath)

    # Reverse mapping
    rev_atm_dict = {}
    for element, numbers in atom_dic.items():
        for number in numbers:
            rev_atm_dict[number] = element

    labels_list = [f"{element}{number}" for element, numbers in atom_dic.items() for number in numbers]

    for ii in eigen_num:
        eigvecs = np.real(vectors[ii].reshape(len(vectors[ii])//3, 3))

        for jj in np.arange(len(labels_list)):
            atm_tmp = rev_atm_dict[jj+1]
            # Assumes 1-based indexing in atom_dic
            label_tmp = atm_tmp + str(jj+1)

            # Find index in structure structure
            # Note: This requires the structure labels to match exactly
            try:
                idx_tmp = structure.labels.index(label_tmp)
                structure[idx_tmp].properties["magmom"] = eigvecs[jj]
            except ValueError:
                print(f"Warning: Label {label_tmp} not found in structure.")

        mcif_writer = CifWriter(structure, write_magmoms=True)

        out_dir = '../results/Eigenvectors/'
        # Create the directory if it doesn't exist
        os.makedirs(out_dir, exist_ok=True)
        if name is None:
            fname = f'{out_dir}Eigenvector_#{ii}.mcif'
        else:
            fname = f'{out_dir}Eigenvector_#{ii}_{name}.mcif'

        mcif_writer.write_file(fname)
