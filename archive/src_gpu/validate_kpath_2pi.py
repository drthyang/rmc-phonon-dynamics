#!/usr/bin/env python3
"""Validate the k-vector convention src_gpu expects.

src_gpu tiles the rmc6f's conventional cell with INTEGER cell indices, and
``Calculators.process_batch_kernel`` uses ``phase = exp(i · cell_idx · kvec)``
(no 2π). For a physically correct Bloch phase, ``kvec`` must therefore be
``2π × (conventional-cell fractional coordinate)``.

Two checks:

1. Reciprocal-lattice periodicity (the decisive 2π test). For a true reciprocal
   lattice vector G, S(k=G) must equal S(k=Γ). That holds only when kvec=2π·G;
   with raw fractions it does not.

2. Folding demo. The crystal is FCC, but src_gpu tiles the *conventional cubic*
   cell, so it samples the simple-cubic BZ. seekpath's FCC-primitive points
   (e.g. X=[0,1,0]) fold/alias onto other points (X→Γ) and are NOT the right
   thing to feed directly — the simple-cubic BZ points are.

Usage:  python validate_kpath_2pi.py [T] [n_configs]
        (defaults: T=5, n_configs=40 — a subset is plenty; checks are per-frame exact)
"""
import os
os.environ["JAX_ENABLE_X64"] = "True"
import sys, glob, tempfile
import numpy as np

import Readers
import Calculators

T = int(sys.argv[1]) if len(sys.argv) > 1 else 5
N_CFG = int(sys.argv[2]) if len(sys.argv) > 2 else 40
TWO_PI = 2 * np.pi

sf = f'../data/{T}K_ini/GTS_{T}K.rmc6f'
cfg = f'../data/ensemble_20A_{T}K/configs/'


def main():
    atom_dic = Readers.get_atom_idx(sf, verbose=0)
    v1, v2, v3, dim = Readers.read_cell_vec(sf, verbose=0)
    v_super = np.array([v1, v2, v3])

    src = sorted(glob.glob(cfg + 'Frac*.txt'))[:N_CFG]
    if not src:
        sys.exit(f'No Frac*.txt configs found under {cfg}')
    tmp = tempfile.mkdtemp()
    for f in src:
        os.symlink(os.path.abspath(f), os.path.join(tmp, os.path.basename(f)))
    fpath = tmp + '/'
    hsym = Readers.avg_frac_atom_ph(sorted(glob.glob(fpath + 'Frac*.txt')), atom_dic, dim)

    def S(kvec):
        M = Calculators.Sk_avg(fpath, hsym, atom_dic, dim, np.array(kvec, float),
                               v_super, loadfile=False, save=False)
        return (M + M.conj().T) / 2

    S_gamma = S([0, 0, 0])
    n_gamma = np.linalg.norm(S_gamma)
    tol = 1e-2 * max(n_gamma, 1.0)
    print(f'T={T}K | configs={len(src)} | dim={dim.astype(int).tolist()} | ||S(Γ)||={n_gamma:.4f}\n')

    print('[1] Reciprocal-lattice periodicity:  S(k=G) must equal S(Γ)')
    ok = True
    for G in ([1, 1, 1], [2, 0, 0], [2, 2, 0]):
        G = np.array(G, float)
        d_raw = np.linalg.norm(S(G) - S_gamma)              # raw fractions (wrong)
        d_2pi = np.linalg.norm(S(TWO_PI * G) - S_gamma)     # 2π-scaled (correct)
        ok &= (d_2pi < tol) and (d_raw > tol)
        print(f'    G={G.astype(int).tolist()}:  raw={d_raw:9.4f}  2π={d_2pi:9.4f}')
    print(f'    => {"PASS: kvec must be 2π·fraction" if ok else "UNEXPECTED — investigate"}\n')

    print('[2] Folding: seekpath FCC-primitive points vs simple-cubic BZ points (×2π)')
    cases = [
        ('cubic   X  [0,0.5,0]',   [0, 0.5, 0]),
        ('cubic   M  [0.5,0.5,0]', [0.5, 0.5, 0]),
        ('seekpath X  [0,1,0]',    [0, 1, 0]),
        ('seekpath W  [0.5,1,0]',  [0.5, 1, 0]),
    ]
    for label, k in cases:
        d = np.linalg.norm(S(TWO_PI * np.array(k, float)) - S_gamma)
        tag = '== Γ (aliases)' if d < tol else 'distinct'
        print(f'    {label:24s} ||S−S(Γ)||={d:9.4f}   {tag}')


if __name__ == '__main__':
    main()
