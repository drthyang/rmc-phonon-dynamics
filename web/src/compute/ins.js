// web/src/compute/ins.js
//
// Build the in-memory `ydata` structure (band.yaml-equivalent, frequencies in
// meV) that the S(Q,E)/DOS worker consumes, directly from pipeline results.
// Avoids a YAML round-trip while keeping the exact data layout the legacy
// viewer worker expects: points = basis sites, eigenvector per site = 3x[re,im].

import { ATOMIC_MASS } from '../constants';
import { conventionalLattice, reciprocalLattice } from '../math/reciprocal';

export function buildYData(results) {
  const bs = results.baseStructure;
  const { atomDic, dim, v1, v2, v3, uniqueRN } = bs;
  const recip = reciprocalLattice(conventionalLattice(v1, v2, v3, dim));

  const rnToRow = new Map();
  (uniqueRN || []).forEach((rn, r) => rnToRow.set(rn, r));

  // Basis-site list, element-grouped (matches band.yaml points order).
  const atomList = [];
  for (const el of Object.keys(atomDic)) for (const rn of atomDic[el]) atomList.push({ el, rn });

  const points = atomList.map(a => ({ symbol: a.el, mass: ATOMIC_MASS[a.el] || 0 }));

  const phonon = [];
  for (let qi = 0; qi < results.qPoints.length; qi++) {
    const band = [];
    const nModes = results.bands[qi].length;
    for (let m = 0; m < nModes; m++) {
      const ev = results.eigvecs[qi][m];
      const eigenvector = atomList.map(a => {
        const r = rnToRow.get(a.rn);
        return [
          [ev.real[r * 3], ev.imag[r * 3]],
          [ev.real[r * 3 + 1], ev.imag[r * 3 + 1]],
          [ev.real[r * 3 + 2], ev.imag[r * 3 + 2]],
        ];
      });
      band.push({ frequency: results.bands[qi][m], eigenvector });
    }
    phonon.push({ 'q-position': results.qPoints[qi], band });
  }

  return { natom: points.length, nqpoint: phonon.length, points, reciprocal_lattice: recip, phonon };
}
