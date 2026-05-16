#!/usr/bin/env python3
import os
os.environ["JAX_ENABLE_X64"] = "True"

import numpy as np
import glob

import Readers
import Calculators
import Writers

T = 5.0

stempath = '../data/'
fpath_eq = stempath + f'{T}K_ini/GTS_{T}K.rmc6f'
fpath    = stempath + f'ensemble_20A_{T}K/configs/'

atom_dic        = Readers.get_atom_idx(fpath_eq)
v1, v2, v3, dim = Readers.read_cell_vec(fpath_eq)
v_super         = np.array([v1, v2, v3])

rmcfiles = sorted(glob.glob(fpath + 'Frac*.txt'))
print(f'Found {len(rmcfiles)} configurations ...')
hsym_test = Readers.avg_frac_atom_ph(rmcfiles, atom_dic, dim)

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
        current_k = k_start + jj * k_vec / kstep
        print(f'  k = {current_k}', end=' ... ', flush=True)

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
