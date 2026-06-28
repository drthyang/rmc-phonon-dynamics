// web/src/io/vaspexport.js
//
// VASP phonon export — port of Writers.gen_vasp_phonon. Writes the equilibrium
// POSCAR, a displaced-mode POSCAR for the selected band point, and INCAR/KPOINTS
// templates. (POTCAR must be supplied separately — VASP license.)

import { mat3Inverse } from '../math/reciprocal';
import { downloadString } from './writers';

function poscarText(vSuper, elements, counts, fracCoords, comment) {
  const header = comment || elements.map((el, i) => `${el}${counts[i]}`).join('');
  const L = [header, '1.0'];
  for (const v of vSuper) L.push(`  ${v[0].toFixed(12).padStart(18)}  ${v[1].toFixed(12).padStart(18)}  ${v[2].toFixed(12).padStart(18)}`);
  L.push('  ' + elements.join('  '));
  L.push('  ' + counts.join('  '));
  L.push('Direct');
  for (const p of fracCoords) L.push(`  ${p[0].toFixed(12).padStart(18)}  ${p[1].toFixed(12).padStart(18)}  ${p[2].toFixed(12).padStart(18)}`);
  return L.join('\n') + '\n';
}

const INCAR = `# VASP INCAR template for phonon calculation
SYSTEM  = phonon
ISTART  = 0
ICHARG  = 2
ENCUT   = 400       # EDIT: match your POTCAR recommendations
PREC    = Accurate
EDIFF   = 1E-8
NSW     = 0
IBRION  = 8         # EDIT: 8=DFPT, 6=finite differences
ISMEAR  = 0
SIGMA   = 0.05
NCORE   = 4         # EDIT: match your cluster setup
# NOTE: POTCAR must be provided separately (VASP license required)
`;
const KPOINTS = `Automatic k-mesh
0
Gamma
  2  2  2    # EDIT: increase for production runs
  0  0  0
`;

/**
 * Export VASP inputs for the selected mode (kIndex, modeIndex). amplitude in Å.
 */
export function exportVASP(model, kIndex, modeIndex, amplitude = 0.05) {
  const bs = model.baseStructure;
  const vSuper = [bs.v1, bs.v2, bs.v3];            // unit cell rows (Å)
  const vInv = mat3Inverse(vSuper);
  const rnToRow = new Map((bs.uniqueRN || []).map((rn, r) => [rn, r]));

  const elements = Object.keys(bs.atomDic);
  const counts = elements.map(el => bs.atomDic[el].length);

  // Equilibrium fractional coords, element-grouped order; same site→row mapping
  // used for the eigenvector.
  const order = [];
  for (const el of elements) for (const rn of bs.atomDic[el]) order.push(rn);
  const fracEq = order.map(rn => {
    const r = rnToRow.get(rn);
    return [((bs.hsym_xyz[r * 3] % 1) + 1) % 1, ((bs.hsym_xyz[r * 3 + 1] % 1) + 1) % 1, ((bs.hsym_xyz[r * 3 + 2] % 1) + 1) % 1];
  });

  downloadString(poscarText(vSuper, elements, counts, fracEq), 'POSCAR', 'text/plain');

  const ev = model.eigvecs?.[kIndex]?.[modeIndex];
  if (ev) {
    const fracDisp = order.map((rn, i) => {
      const r = rnToRow.get(rn);
      const d = [ev.real[r * 3], ev.real[r * 3 + 1], ev.real[r * 3 + 2]]; // Cartesian Å (real part)
      const df = [
        d[0] * vInv[0][0] + d[1] * vInv[1][0] + d[2] * vInv[2][0],
        d[0] * vInv[0][1] + d[1] * vInv[1][1] + d[2] * vInv[2][1],
        d[0] * vInv[0][2] + d[1] * vInv[1][2] + d[2] * vInv[2][2],
      ];
      return [((fracEq[i][0] + amplitude * df[0]) % 1 + 1) % 1, ((fracEq[i][1] + amplitude * df[1]) % 1 + 1) % 1, ((fracEq[i][2] + amplitude * df[2]) % 1 + 1) % 1];
    });
    const comment = `Mode ${modeIndex + 1} @ k${kIndex + 1} displaced amplitude=${amplitude} A`;
    downloadString(poscarText(vSuper, elements, counts, fracDisp, comment), `POSCAR_mode_${modeIndex + 1}`, 'text/plain');
  }

  downloadString(INCAR, 'INCAR', 'text/plain');
  downloadString(KPOINTS, 'KPOINTS', 'text/plain');
}
