#!/usr/bin/env python3
"""Example phonon-band run. Thin caller around runner.run_bands (Phase 4)."""
import os
os.environ["JAX_ENABLE_X64"] = "True"

import numpy as np

import runner

T = 5  # integer — folder names use e.g. '5K', not '5.0K'
stempath = '../data/'

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

result = runner.run_bands(
    structure_file=stempath + f'{T}K_ini/GTS_{T}K.rmc6f',
    configs_dir=stempath + f'ensemble_20A_{T}K/configs/',
    sym_pnts=sym_pnts,
    k_path=k_path,
    kstep=kstep,
    T=T,
    out_dir='../results/',
)
print(f'Done. {result["n_qpoints"]} k-points -> {result["band_yaml"]}')
