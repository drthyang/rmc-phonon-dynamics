#!/usr/bin/env python3
import os
os.environ["JAX_ENABLE_X64"] = "True"

import numpy as np
import glob

import Readers
import Calculators
import Writers
import constants

T = 5  # integer — folder names use e.g. '5K', not '5.0K'

stempath = '../data/'
fpath_eq = stempath + f'{T}K_ini/GTS_{T}K.rmc6f'
fpath    = stempath + f'ensemble_20A_{T}K/configs/'  # 500 Frac_coord_*.txt files live here

atom_dic        = Readers.get_atom_idx(fpath_eq)
v1, v2, v3, dim = Readers.read_cell_vec(fpath_eq)
v_super         = np.array([v1, v2, v3])

rmcfiles = sorted(glob.glob(fpath + 'Frac*.txt'))
print(f'Found {len(rmcfiles)} configurations ...')
hsym_test = Readers.avg_frac_atom_ph(rmcfiles, atom_dic, dim)

# High-symmetry points of the SIMPLE-CUBIC BZ, in fractions of the conventional
# (cubic) cell that src_gpu tiles. These — not seekpath's FCC-primitive points —
# are the right family here: src_gpu indexes whole cubic cells, so its S(k) is
# periodic in the simple-cubic reciprocal lattice. (FCC X=[0,1,0] would fold onto Γ.)
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

#k_path = ['GM', 'X', 'M', 'GM', 'Z', 'R', 'A', 'Z']
k_path = ['GM', 'X', 'M', 'GM']
kstep  = 16

print(f'Running {(len(k_path)-1)*(kstep+1)} k-points ...')

ph_band          = []
eigenvectors_all = []

for ii in range(len(k_path) - 1):
    k_start = sym_pnts[k_path[ii]]
    k_vec   = sym_pnts[k_path[ii + 1]] - k_start
    for jj in range(kstep + 1):
        k_frac = k_start + jj * k_vec / kstep
        # Scale fractional k to src_gpu's phase convention. The 2π factor is the
        # single reversible switch constants.APPLY_2PI_PHASE (see constants.py).
        current_k = constants.TWO_PI_PHASE * k_frac
        print(f'  k = {k_frac} (x{constants.TWO_PI_PHASE:.4f})', end=' ... ', flush=True)

        Sk = Calculators.Sk_avg(fpath, hsym_test, atom_dic, dim, current_k, v_super,
                                loadfile=True, save=True)

        Sk = (Sk + Sk.conj().T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(Sk)

        ph_band.append(Calculators.eigenvalues_to_meV(eigenvalues, T))
        eigenvectors_all.append(eigenvectors)
        print('done')

print('Applying band connection ...')
ph_band, eigenvectors_all = Writers.connect_bands(ph_band, eigenvectors_all, degenerate_tol=5e-3)

print('Writing band.yaml ...')
Writers.gen_phonopy_band_yaml(
    atom_dic, hsym_test, v1, v2, v3, dim,
    ph_band, eigenvectors_all,
    k_path, sym_pnts, kstep,
    out_dir='../results/',
)

print('Done.')
