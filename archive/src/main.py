#!/usr/bin/env python3
import numpy as np
import glob
from tqdm.auto import tqdm

import Readers
import Calculators
import Writers
import Visualization

# ── Run parameters ────────────────────────────────────────────────────────────
T = 5  # temperature [K] — must match the RMC ensemble temperature

stempath = '../data/'
fpath_eq      = stempath + '5K_ini/GTS_5K.rmc6f'
fpath         = stempath + 'ensemble_20A_5K/configs/'

plot_PDOS       = True
plot_PartialDOS = True

sym_pnts = {
    'A':    np.array([ 0.5,  0.5,  0.5]),
    'GM':   np.array([ 0.0,  0.0,  0.0]),
    'M':    np.array([ 0.5,  0.5,  0.0]),
    'R':    np.array([ 0.0,  0.5,  0.5]),
    'X':    np.array([ 0.0,  0.5,  0.0]),
    'Z':    np.array([ 0.0,  0.0,  0.5]),
    'hh-h': np.array([ 0.5,  0.5, -0.5]),
    'h00':  np.array([ 0.5,  0.0,  0.0]),
}

# ── Main execution ────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # 1. Read structural information
    atom_dic = Readers.get_atom_idx(fpath_eq)
    v1, v2, v3, dim = Readers.read_cell_vec(fpath_eq)
    v_super = np.array([v1, v2, v3])  # (3, 3) supercell lattice matrix [Å]

    # 2. Load configurations and build average (equilibrium) structure
    rmcfiles = sorted(glob.glob(fpath + 'Frac*.txt'))
    print('🔎 Found ** {} ** configurations ...'.format(len(rmcfiles)))
    hsym_test = Readers.avg_frac_atom_ph(rmcfiles, atom_dic, dim)

    # 3. Define k-path and step count
    k_path = ['GM', 'h00']
    kstep  = 32
    print('📊 Calculating phonon bands along: {} ...'.format(k_path))

    ph_band          = []
    eigenvectors_all = []

    # 4. Loop over k-path segments
    total_steps = (len(k_path) - 1) * kstep
    with tqdm(total=total_steps, desc='⏩️ Total Progress') as pbar:
        for ii in range(len(k_path) - 1):
            k_plot_vec = sym_pnts[k_path[ii + 1]] - sym_pnts[k_path[ii]]

            for jj in range(kstep):
                current_k = sym_pnts[k_path[ii]] + jj * k_plot_vec / kstep

                Sk = Calculators.Sk_avg(fpath, hsym_test, atom_dic, dim,
                                        current_k, v_super)
                eigenvalues, eigenvectors = np.linalg.eigh(Sk)

                ph_band.append(Calculators.eigenvalues_to_meV(eigenvalues, T))
                eigenvectors_all.append(eigenvectors)

                # Write eigenvector CIF at the first point of each segment
                if jj == 0:
                    Writers.gen_ev_mcif('../data/GTS_5K.cif', atom_dic,
                                        eigenvectors, name=k_path[ii])
                pbar.update(1)

    # 5. Plot bands
    Visualization.plot_phonon_bands(ph_band, k_path, kstep)

    # 6. Total phonon DOS
    if plot_PDOS:
        qpnts = Calculators.gen_grid(5)
        wk = []
        for qpnt in qpnts:
            Sk = Calculators.Sk_avg(fpath, hsym_test, atom_dic, dim,
                                    qpnt, v_super)
            eigenvalues, _ = np.linalg.eigh(Sk)
            wk.append(Calculators.eigenvalues_to_meV(eigenvalues, T))
        Visualization.plot_ph_dos(wk)

    # 7. Partial phonon DOS (per element)
    if plot_PartialDOS:
        elements = list(atom_dic.keys())
        for partial_type in elements:
            print('Calculating partial phonon DOS for {} ...'.format(partial_type))
            hsym_partial = Readers.avg_frac_atom_ph(
                rmcfiles, atom_dic, dim, atype=partial_type)

            qpnts = Calculators.gen_grid(5)
            wk = []
            for qpnt in qpnts:
                Sk = Calculators.Partial_Sk_avg(fpath, hsym_partial, atom_dic,
                                                dim, qpnt, partial_type, v_super)
                eigenvalues, _ = np.linalg.eigh(Sk)
                wk.append(Calculators.eigenvalues_to_meV(eigenvalues, T))

            print('Plotting Partial DOS for {}'.format(partial_type))
            Visualization.plot_ph_dos(wk)
