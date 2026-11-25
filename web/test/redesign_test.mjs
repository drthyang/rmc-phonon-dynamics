// web/test/redesign_test.mjs
// Deterministic tests for single-file .rmc6f parsing, INS 1D/2D readers, and INS comparison metrics.

import assert from 'node:assert/strict';
import { parseBaseStructureText, parseRmc6fFrame, generateUnitCellCIF } from '../src/io/readers.js';
import { parseExpIns1D, parseExpIns2D, parseNeutronDosCsv } from '../src/io/insReaders.js';
import { gaussianBroaden1D, fitAndScore1D, compare2DSqe, integrateOverQ, integrateOverE } from '../src/math/insCompare.js';
import { domainAlignedStructure } from '../src/math/supercell_symmetry.js';

// 1. Single-file .rmc6f text parsing + CIF generation
const sampleRmc6f = `
Supercell dimensions: 2 2 2
Lattice vectors:
10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0
Atoms:
1 Ga 0 0.05 0.05 0.05 1 0 0 0
2 Ga 0 0.55 0.05 0.05 1 1 0 0
3 S  0 0.25 0.25 0.25 2 0 0 0
`;

const base = parseBaseStructureText(sampleRmc6f);
assert.deepEqual(base.dim, [2, 2, 2]);
assert.equal(base.basis.length, 2);
assert.deepEqual(base.atomDic.Ga, [1]);
assert.deepEqual(base.atomDic.S, [2]);

const frame = parseRmc6fFrame(sampleRmc6f, base.dim);
assert.equal(frame.atomType.length, 3);
assert.ok(Math.abs(frame.xyz[0] - 0.1) < 1e-6);

const cif = generateUnitCellCIF([[5, 0, 0], [0, 5, 0], [0, 0, 5]], base.basis, { title: 'test_parent', spaceGroup: 'F-43m' });
assert.ok(cif.includes('_cell_length_a'));
assert.ok(cif.includes('Ga1'));

// 2. Experimental 1D INS parser + 1D fit & scoring
const exp1dText = `# E_meV  I  sigma
0.0 0.0 0.1
5.0 2.0 0.1
10.0 4.0 0.1
15.0 2.0 0.1
20.0 0.0 0.1
`;
const exp1d = parseExpIns1D(exp1dText, 'test.dat');
assert.equal(exp1d.energy.length, 5);
assert.equal(exp1d.sigma.length, 5);

const calcE = [0, 5, 10, 15, 20];
const calcI = [0, 1, 2, 1, 0]; // exact 0.5x scale
const fit = fitAndScore1D(exp1d.energy, exp1d.intensity, exp1d.sigma, calcE, calcI);
assert.ok(Math.abs(fit.scale - 2.0) < 1e-6);
assert.ok(fit.rw < 1e-6);
assert.ok(Math.abs(fit.cosine - 1.0) < 1e-6);

// Gaussian broadening preserves total area approximately
const broadened = gaussianBroaden1D([0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 0, 0, 0, 1, 0, 0, 0, 0], 1.5);
const sumB = broadened.reduce((a, b) => a + b, 0);
assert.ok(Math.abs(sumB - 1.0) < 0.05);

// 3. Neutron DOS CSV parser
const ndosCsv = `energy_meV,total_dos,neutron_gdos,gdos_Ga,gdos_S
0.0,0.0,0.0,0.0,0.0
5.0,0.5,0.6,0.4,0.2
10.0,1.0,1.2,0.8,0.4
15.0,0.2,0.2,0.1,0.1
`;
const nd = parseNeutronDosCsv(ndosCsv, 'test_ndos.csv');
assert.equal(nd.energy_mev.length, 4);
assert.deepEqual(Object.keys(nd.element_gdos), ['Ga', 'S']);

// 4. 2D S(|Q|, E) parser (matrix + triplet) & 2D comparison
const exp2dMatrix = `E\\Q,1.0,2.0,3.0
0.0,0.0,0.1,0.2
5.0,0.5,1.0,0.8
10.0,0.2,0.4,0.3
`;
const exp2d = parseExpIns2D(exp2dMatrix, 'exp2d.csv');
assert.deepEqual(exp2d.q_grid, [1, 2, 3]);
assert.deepEqual(exp2d.e_grid, [0, 5, 10]);

const cand2d = {
  q_grid: [1, 2, 3],
  e_grid: [0, 5, 10],
  sqe: exp2d.sqe.map(row => row.map(v => v * 0.5)),
};
const cmp = compare2DSqe(cand2d, exp2d);
assert.ok(Math.abs(cmp.scale - 2.0) < 1e-6);
assert.ok(cmp.rw < 1e-6);
assert.equal(integrateOverQ(cmp.calcGrid, cmp.q_grid, 1, 3).length, 3);
assert.equal(integrateOverE(cmp.calcGrid, cmp.e_grid, 0, 10).length, 3);

// 5. domainAlignedStructure idealizes noisy parent refBasis so verifiedSpaceGroup matches target subgroup
{
  const A = [[4, 0, 0], [0, 4, 0], [0, 0, 4]];
  // Simple cubic BaTiO3-like basis with 0.004 A noise in refBasis (breaks Pm-3m at 1e-3 A if not idealized!)
  const noisyBasis = [
    { rn: 1, el: 'Ba', frac: [0.001, 0.999, 0.0005] },
    { rn: 2, el: 'Ti', frac: [0.4992, 0.5008, 0.5003] },
    { rn: 3, el: 'O',  frac: [0.5005, 0.5002, 0.0004] },
    { rn: 4, el: 'O',  frac: [0.5003, 0.0006, 0.4995] },
    { rn: 5, el: 'O',  frac: [0.0002, 0.4997, 0.5006] },
  ];
  const dim = [2, 2, 2];
  const nCells = 8, D = 15;
  const u = new Float64Array(nCells * D);
  // Apply coherent [001] Ti off-centering (P4mm)
  for (let c = 0; c < nCells; c++) {
    u[c * D + 1 * 3 + 2] = 0.12;
  }
  const aligned = domainAlignedStructure([u], dim, A, noisyBasis, 'P4mm', { parentTol: 0.1 });
  assert.equal(aligned.verifiedSpaceGroup, 'P4mm');
  assert.ok(aligned.symRms > 0.01);
}

console.log('redesign_test: all assertions passed');
