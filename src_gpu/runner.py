"""Importable phonon-band runner (Phase 4).

Wraps the Sk_avg -> eigh -> band-connection -> band.yaml flow that test_run.py
used inline, so the same path can be driven from a script or the GUI job
manager. k-points come from kpath.build_kpath, so the 2*pi convention is applied
once via constants.TWO_PI_PHASE.

Two entry points share one compute core (_compute_bands):
  - run_bands           : label-based, uniform points/segment, contiguous path.
  - run_bands_segments  : explicit per-segment k-path (varying npoints + breaks),
                          which is what the GUI produces.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "True")

import glob
import numpy as np

import Readers
import Calculators
import Writers
import constants
import kpath


def _compute_bands(structure_file, configs_dir, kp, T,
                   loadfile=True, save=True, degenerate_tol=5e-3,
                   verbose=True, on_step=None):
    """Run the Sk -> eigh -> band-connection core over a materialized k-path.

    structure_file : *.rmc6f equilibrium file (atom types + lattice).
    configs_dir    : folder of Frac*.txt ensemble configurations (all are used).
    kp             : dict from kpath.build_kpath ({q_frac, kvec, seg_sizes}).
    on_step        : optional callback(i, n_total, k_frac) for live progress.

    Returns the inputs the band.yaml writers need:
      {atom_dic, hsym, v1, v2, v3, dim, ph_band, eigenvectors_all, n_qpoints}.
    """
    if configs_dir and not configs_dir.endswith('/'):
        configs_dir += '/'

    atom_dic = Readers.get_atom_idx(structure_file, verbose=0)
    v1, v2, v3, dim = Readers.read_cell_vec(structure_file, verbose=0)
    v_super = np.array([v1, v2, v3])

    rmcfiles = sorted(glob.glob(configs_dir + 'Frac*.txt'))
    if not rmcfiles:
        raise FileNotFoundError(f'No Frac*.txt configurations in {configs_dir}')
    hsym = Readers.avg_frac_atom_ph(rmcfiles, atom_dic, dim)

    n = len(kp['kvec'])
    if verbose:
        print(f'Running {n} k-points over {len(rmcfiles)} configs ...')

    ph_band, eigenvectors_all = [], []
    for i, (k_frac, current_k) in enumerate(zip(kp['q_frac'], kp['kvec'])):
        if on_step:
            on_step(i, n, k_frac)
        if verbose:
            print(f'  [{i+1}/{n}] k = {k_frac} (x{constants.TWO_PI_PHASE:.4f})',
                  end=' ... ', flush=True)

        Sk = Calculators.Sk_avg(configs_dir, hsym, atom_dic, dim, current_k, v_super,
                                loadfile=loadfile, save=save)
        Sk = (Sk + Sk.conj().T) / 2
        eigvals, eigvecs = np.linalg.eigh(Sk)

        ph_band.append(Calculators.eigenvalues_to_meV(eigvals, T))
        eigenvectors_all.append(eigvecs)
        if verbose:
            print('done')

    if verbose:
        print('Applying band connection ...')
    ph_band, eigenvectors_all = Writers.connect_bands(
        ph_band, eigenvectors_all, degenerate_tol=degenerate_tol)

    return {'atom_dic': atom_dic, 'hsym': hsym, 'v1': v1, 'v2': v2, 'v3': v3,
            'dim': dim, 'ph_band': ph_band, 'eigenvectors_all': eigenvectors_all,
            'n_qpoints': n}


def run_bands(structure_file, configs_dir, sym_pnts, k_path, kstep, T,
              out_dir='../results/', loadfile=True, save=True,
              degenerate_tol=5e-3, verbose=True, on_step=None):
    """Compute a phonon band structure and write a phonopy band.yaml.

    Label-based, uniform-density, contiguous path (the test_run.py case).

    sym_pnts : {label: conventional-cell fractional reciprocal coord}.
    k_path   : list of labels, e.g. ['GM', 'X', 'M', 'GM'].
    kstep    : intervals per segment (kstep+1 points per segment).

    Returns: {band_yaml, n_qpoints, ph_band, q_frac}.
    """
    kp = kpath.build_kpath(kpath.segments_from_path(sym_pnts, k_path, kstep))
    res = _compute_bands(structure_file, configs_dir, kp, T,
                         loadfile=loadfile, save=save,
                         degenerate_tol=degenerate_tol, verbose=verbose,
                         on_step=on_step)

    if verbose:
        print('Writing band.yaml ...')
    band_yaml = Writers.gen_phonopy_band_yaml(
        res['atom_dic'], res['hsym'], res['v1'], res['v2'], res['v3'], res['dim'],
        res['ph_band'], res['eigenvectors_all'],
        k_path, sym_pnts, kstep,
        out_dir=out_dir,
    )
    return {'band_yaml': band_yaml, 'n_qpoints': res['n_qpoints'],
            'ph_band': res['ph_band'], 'q_frac': kp['q_frac']}


def run_bands_segments(structure_file, configs_dir, segments, T,
                       out_dir='../results/', out_name='band_gpu.yaml',
                       loadfile=True, save=True, degenerate_tol=5e-3,
                       verbose=True, on_step=None):
    """Compute a phonon band structure from an explicit per-segment k-path.

    Unlike run_bands, segments may have DIFFERENT point counts and the path may
    be DISCONTINUOUS (breaks). This is what the GUI runner feeds.

    segments : list of dicts, each:
        {'from_frac': (3,) conventional-cell fractional reciprocal coord,
         'to_frac':   (3,) ditto,
         'npoints':   int >= 2 (endpoints inclusive),
         'from_label': str, 'to_label': str}

    Returns: {band_yaml, n_qpoints, ph_band, q_frac}.
    """
    if not segments:
        raise ValueError('run_bands_segments: empty k-path (no segments)')

    kp = kpath.build_kpath(segments)
    res = _compute_bands(structure_file, configs_dir, kp, T,
                         loadfile=loadfile, save=save,
                         degenerate_tol=degenerate_tol, verbose=verbose,
                         on_step=on_step)

    seg_labels = [(s.get('from_label', ''), s.get('to_label', '')) for s in segments]
    if verbose:
        print('Writing band.yaml ...')
    band_yaml = Writers.gen_phonopy_band_yaml_segments(
        res['atom_dic'], res['hsym'], res['v1'], res['v2'], res['v3'], res['dim'],
        res['ph_band'], res['eigenvectors_all'],
        kp['q_frac'], kp['seg_sizes'], seg_labels,
        out_dir=out_dir, out_name=out_name,
    )
    return {'band_yaml': band_yaml, 'n_qpoints': res['n_qpoints'],
            'ph_band': res['ph_band'], 'q_frac': kp['q_frac']}
