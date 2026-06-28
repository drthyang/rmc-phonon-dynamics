import { ATOMIC_MASS } from '../constants';
import { conventionalLattice, reciprocalLattice } from '../math/reciprocal';

/**
 * Full phonopy-style band.yaml writer (port of Writers._write_band_yaml).
 *
 * Atoms in band.yaml are crystallographic BASIS SITES (one per RMC reference
 * number), element-grouped. Eigenvector rows are ordered by sorted reference
 * number (uniqueRN) — the same ordering produced by the diagonalizer — so each
 * site's 3-vector is row `rank[rn]`.
 *
 * @param {Object} baseStructure - enriched: {atomDic, dim, v1,v2,v3, hsym_xyz, atomType, uniqueRN}
 * @param {Array}  qPoints       - fractional q (NOT scaled by 2*pi), ph_band order
 * @param {Array<Float64Array>} bands    - frequency [meV] per q per mode
 * @param {Array<Array<{real,imag}>>} eigvecs - eigenvectors per q per mode
 * @param {Object} kpathMeta     - {segSizes, hsymIndex}
 */
export function generatePhonopyBandYaml(baseStructure, qPoints, bands, eigvecs, kpathMeta) {
  const { atomDic, dim, v1, v2, v3, hsym_xyz, atomType, uniqueRN } = baseStructure;
  const A = conventionalLattice(v1, v2, v3, dim);   // unit-cell lattice (rows a,b,c)
  const recip = reciprocalLattice(A);                // inv(A).T, no 2*pi

  const elements = Object.keys(atomDic);
  // Basis-site list (element-grouped) and equilibrium fractional positions.
  const rnToRow = new Map();
  (uniqueRN || []).forEach((rn, r) => rnToRow.set(rn, r));
  const rnToFirstAtom = new Map();
  if (atomType) for (let i = 0; i < atomType.length; i++) {
    if (!rnToFirstAtom.has(atomType[i])) rnToFirstAtom.set(atomType[i], i);
  }
  const atomList = [];
  for (const el of elements) for (const rn of atomDic[el]) {
    let pos = [0, 0, 0];
    const ai = rnToFirstAtom.get(rn);
    if (ai !== undefined && hsym_xyz) {
      pos = [hsym_xyz[ai * 3] % 1.0, hsym_xyz[ai * 3 + 1] % 1.0, hsym_xyz[ai * 3 + 2] % 1.0];
    }
    atomList.push({ el, rn, pos });
  }

  const nQ = qPoints.length;
  const nModes = bands[0]?.length || 0;
  const segSizes = kpathMeta?.segSizes || [nQ];
  const hsymIndex = kpathMeta?.hsymIndex || {};

  // Segment-start flat indices (zero the distance increment there).
  const segStarts = new Set();
  { let off = 0; for (const sz of segSizes) { segStarts.add(off); off += sz; } }

  // Cumulative path distance in A^-1 (no 2*pi).
  const distances = [0];
  for (let i = 1; i < nQ; i++) {
    if (segStarts.has(i)) { distances.push(distances[i - 1]); continue; }
    const dq = [qPoints[i][0] - qPoints[i - 1][0], qPoints[i][1] - qPoints[i - 1][1], qPoints[i][2] - qPoints[i - 1][2]];
    const cx = dq[0] * recip[0][0] + dq[1] * recip[1][0] + dq[2] * recip[2][0];
    const cy = dq[0] * recip[0][1] + dq[1] * recip[1][1] + dq[2] * recip[2][1];
    const cz = dq[0] * recip[0][2] + dq[1] * recip[1][2] + dq[2] * recip[2][2];
    distances.push(distances[i - 1] + Math.sqrt(cx * cx + cy * cy + cz * cz));
  }

  const f = (x) => Number(x).toFixed(10);
  const lines = [];
  lines.push(`nqpoint: ${nQ}`);
  lines.push(`npath: ${segSizes.length}`);
  lines.push('segment_nqpoint:');
  for (const sz of segSizes) lines.push(`- ${sz}`);
  lines.push('reciprocal_lattice:');
  for (let r = 0; r < 3; r++) lines.push(`- [ ${f(recip[r][0])}, ${f(recip[r][1])}, ${f(recip[r][2])} ] # ${['a*', 'b*', 'c*'][r]}`);
  lines.push(`natom: ${atomList.length}`);
  lines.push('lattice:');
  for (let r = 0; r < 3; r++) lines.push(`- [ ${f(A[r][0])}, ${f(A[r][1])}, ${f(A[r][2])} ] # ${['a', 'b', 'c'][r]}`);
  lines.push('points:');
  for (const a of atomList) {
    lines.push(`- symbol: ${a.el}`);
    lines.push(`  coordinates: [ ${f(a.pos[0])}, ${f(a.pos[1])}, ${f(a.pos[2])} ]`);
    lines.push(`  mass: ${(ATOMIC_MASS[a.el] || 0).toFixed(5)}`);
  }
  lines.push('phonon:');
  for (let qi = 0; qi < nQ; qi++) {
    const q = qPoints[qi];
    lines.push(`- q-position: [ ${q[0].toFixed(8)}, ${q[1].toFixed(8)}, ${q[2].toFixed(8)} ]`);
    lines.push(`  distance: ${distances[qi].toFixed(8)}`);
    if (hsymIndex[qi] !== undefined) lines.push(`  label: '${hsymIndex[qi]}'`);
    lines.push('  band:');
    for (let m = 0; m < nModes; m++) {
      lines.push(`  - # ${m + 1}`);
      lines.push(`    frequency: ${Number(bands[qi][m]).toFixed(10)}`);
      const ev = eigvecs?.[qi]?.[m];
      if (ev) {
        lines.push('    eigenvector:');
        for (let j = 0; j < atomList.length; j++) {
          const r = rnToRow.get(atomList[j].rn);
          lines.push(`    - # atom ${j + 1}`);
          for (let c = 0; c < 3; c++) {
            const idx = r * 3 + c;
            lines.push(`      - [ ${f(ev.real[idx])}, ${f(ev.imag[idx])} ]`);
          }
        }
      }
    }
  }
  return lines.join('\n') + '\n';
}

export function downloadString(text, filename) {
  const blob = new Blob([text], { type: 'text/yaml' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
