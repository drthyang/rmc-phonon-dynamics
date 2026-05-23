"""K-path materialization for phonon band runs (Phase 4, piece 1).

Pure geometry/interpolation — no jax, no I/O. Turns a high-symmetry path into
the per-q-point arrays the runner feeds to src_gpu.

k-points are CONVENTIONAL-cell fractional reciprocal coordinates (the cell
src_gpu tiles). src_gpu's phase wants radians per cell = 2*pi * fraction, applied
here via the single reversible switch constants.TWO_PI_PHASE.
"""
import numpy as np

import constants


def segments_from_path(sym_pnts, k_path, kstep):
    """Convert a labelled path into segment dicts.

    sym_pnts : {label: fractional reciprocal coord}
    k_path   : list of labels, e.g. ['GM', 'X', 'M', 'GM']
    kstep    : intervals per segment -> kstep+1 points per segment (endpoints incl.)
    """
    segments = []
    for i in range(len(k_path) - 1):
        segments.append({
            'from_frac': np.asarray(sym_pnts[k_path[i]], float),
            'to_frac':   np.asarray(sym_pnts[k_path[i + 1]], float),
            'npoints':   kstep + 1,
        })
    return segments


def build_kpath(segments, two_pi_phase=None):
    """Materialize segments into q-points for a band run.

    segments : list of {from_frac, to_frac, npoints>=2}, fractions in the
               conventional cell's reciprocal basis.
    two_pi_phase : scale applied to get src_gpu's kvec (default constants.TWO_PI_PHASE).

    Returns dict:
      q_frac    : (N,3) fractional k-points (endpoints of each segment included,
                  so junction points repeat — matches the phonopy band convention)
      kvec      : (N,3) what src_gpu consumes = two_pi_phase * q_frac
      seg_sizes : per-segment point counts (for band.yaml segment_nqpoint)
    """
    if two_pi_phase is None:
        two_pi_phase = constants.TWO_PI_PHASE

    q, seg_sizes = [], []
    for s in segments:
        a = np.asarray(s['from_frac'], float)
        b = np.asarray(s['to_frac'], float)
        n = max(2, int(s['npoints']))
        for jj in range(n):
            q.append(a + jj * (b - a) / (n - 1))
        seg_sizes.append(n)

    q_frac = np.array(q, float) if q else np.zeros((0, 3))
    return {
        'q_frac': q_frac,
        'kvec': two_pi_phase * q_frac,
        'seg_sizes': seg_sizes,
    }


if __name__ == '__main__':
    # Self-test (no jax, no data needed).
    sym = {'GM': [0, 0, 0], 'X': [0, 0.5, 0], 'M': [0.5, 0.5, 0]}
    kstep = 16
    segs = segments_from_path(sym, ['GM', 'X', 'M'], kstep)
    kp = build_kpath(segs)

    assert len(segs) == 2
    assert kp['seg_sizes'] == [kstep + 1, kstep + 1]
    assert kp['q_frac'].shape == (2 * (kstep + 1), 3)
    # endpoints land exactly on the labelled points
    assert np.allclose(kp['q_frac'][0], sym['GM'])
    assert np.allclose(kp['q_frac'][kstep], sym['X'])
    assert np.allclose(kp['q_frac'][-1], sym['M'])
    # kvec is q_frac scaled by the (reversible) 2*pi switch
    assert np.allclose(kp['kvec'], constants.TWO_PI_PHASE * kp['q_frac'])
    # midpoint of GM->X is [0,0.25,0]
    assert np.allclose(kp['q_frac'][kstep // 2], [0, 0.25, 0])

    print('kpath self-test OK')
    print(f'  segments={len(segs)}  total q-points={len(kp["q_frac"])}'
          f'  TWO_PI_PHASE={constants.TWO_PI_PHASE:.5f}')
