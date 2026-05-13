import numpy as np
import os

from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter


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

    _ATOMIC_MASS = {
        'H':1.008,'He':4.003,'Li':6.94,'Be':9.012,'B':10.81,'C':12.011,
        'N':14.007,'O':15.999,'F':18.998,'Ne':20.180,'Na':22.990,'Mg':24.305,
        'Al':26.982,'Si':28.085,'P':30.974,'S':32.06,'Cl':35.45,'Ar':39.948,
        'K':39.098,'Ca':40.078,'Sc':44.956,'Ti':47.867,'V':50.942,'Cr':51.996,
        'Mn':54.938,'Fe':55.845,'Co':58.933,'Ni':58.693,'Cu':63.546,'Zn':65.38,
        'Ga':69.723,'Ge':72.63,'As':74.922,'Se':78.96,'Br':79.904,'Kr':83.798,
        'Rb':85.468,'Sr':87.62,'Y':88.906,'Zr':91.224,'Nb':92.906,'Mo':95.96,
        'Tc':98.0,'Ru':101.07,'Rh':102.91,'Pd':106.42,'Ag':107.87,'Cd':112.41,
        'In':114.82,'Sn':118.71,'Sb':121.76,'Te':127.60,'I':126.90,'Xe':131.29,
        'Cs':132.91,'Ba':137.33,'La':138.91,'Ce':140.12,'Pr':140.91,'Nd':144.24,
        'Pm':145.0,'Sm':150.36,'Eu':151.96,'Gd':157.25,'Tb':158.93,'Dy':162.50,
        'Ho':164.93,'Er':167.26,'Tm':168.93,'Yb':173.05,'Lu':174.97,'Hf':178.49,
        'Ta':180.95,'W':183.84,'Re':186.21,'Os':190.23,'Ir':192.22,'Pt':195.08,
        'Au':196.97,'Hg':200.59,'Tl':204.38,'Pb':207.2,'Bi':208.98,
        'Th':232.04,'Pa':231.04,'U':238.03,
    }

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
    out_path = out_dir + 'band.yaml'

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
        mass = _ATOMIC_MASS.get(el, 0.0)
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

    print(f'band.yaml written to {out_path}')


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
