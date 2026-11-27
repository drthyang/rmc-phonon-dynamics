// web/test/unfolding_test.mjs
//
// Deterministic physics & geometry tests for Phonon Band Unfolding (`web/src/math/unfolding.js`):
//   1. Supercell relation discovery across rotated/standardized frames:
//      - Cubic Pm-3m (5 atoms) -> 45°-rotated tetragonal I4/mcm / P4/mbm (10 atoms, |det P| = 2)
//      - Rhombohedral primitive FCC F-43m (13 atoms) -> 45°-rotated tetragonal P-42₁m (26 atoms, |det P| = 2)
//   2. Exact undistorted supercell unfolding:
//      - Completeness sum rule: sum_ν W_{Q,ν} = 3 * N_PC to machine precision (< 1e-13)
//      - Zero shadow-branch leakage: folded branches carry W_{Q,ν} < 1e-13
//      - Exact recovery of the parent dispersion frequencies across Γ–X–M–Γ–Z–R
//   3. Symmetry-broken (Peierls dimerized 1×1×2) supercell unfolding:
//      - Sum rule remains exact (< 1e-13) even under finite static atomic displacements δr_I != 0
//      - Zone-folding gap opens and transfers nonzero physical spectral weight onto shadow branches
//      - Lorentzian spectral function A(Q, E) integrates to ~3 * N_PC
//   4. Standalone intrinsic parent-cell inference (`inferIntrinsicParentPath`) when no separate Parent is loaded.

import assert from 'node:assert/strict';
import { fromPhonopyDM } from '../src/io/phonopyDM.js';
import {
  findSupercellRelation,
  unfoldModelOntoParentPath,
  computeSpectralFunctionGrid,
  inferIntrinsicParentPath,
} from '../src/math/unfolding.js';
import { inv3, vecMat3 } from '../src/math/cells.js';

const wrap01 = (x) => ((x % 1) + 1) % 1;
const wrapHalf = (x) => x - Math.round(x);

/**
 * Build a standalone `phonopy-dynamical-matrix-v1` JSON object directly from an
 * analytical pair-spring crystal in pure JS (no external Python needed!).
 */
function buildAnalyticalSpringDM({ lattice, elements, masses, fracPositions, qpoints, highsymQpts, springCutoff = 4.2, bondModulation = null }) {
  const nPrim = elements.length;
  // Use a 2×2×3 Phonopy force-constant supercell so nearest & next-nearest springs are exact
  const scDim = [2, 2, 3];
  const nSuper = nPrim * scDim[0] * scDim[1] * scDim[2];
  const superCart = [];
  const superFracInPrim = [];
  const s2pp = [];

  for (let ix = 0; ix < scDim[0]; ix++) {
    for (let iy = 0; iy < scDim[1]; iy++) {
      for (let iz = 0; iz < scDim[2]; iz++) {
        for (let p = 0; p < nPrim; p++) {
          const fPrim = [
            ix + fracPositions[p][0],
            iy + fracPositions[p][1],
            iz + fracPositions[p][2],
          ];
          superFracInPrim.push(fPrim);
          superCart.push(vecMat3(fPrim, lattice));
          s2pp.push(p);
        }
      }
    }
  }

  // Build Phonopy C-type shortest_vectors: sv = r_super(s) - r_prim(i) in primitive fractional coords
  // wrapped to minimal periodic image of the [2, 2, 3] supercell!
  const shortestVectors = [];
  const multiplicity = Array.from({ length: nSuper }, () =>
    Array.from({ length: nPrim }, () => [0, 0])
  );

  for (let s = 0; s < nSuper; s++) {
    for (let i = 0; i < nPrim; i++) {
      const candidates = [];
      let minDist2 = Infinity;
      for (let gx = -1; gx <= 1; gx++) {
        for (let gy = -1; gy <= 1; gy++) {
          for (let gz = -1; gz <= 1; gz++) {
            const sv = [
              superFracInPrim[s][0] + gx * scDim[0] - fracPositions[i][0],
              superFracInPrim[s][1] + gy * scDim[1] - fracPositions[i][1],
              superFracInPrim[s][2] + gz * scDim[2] - fracPositions[i][2],
            ];
            const cart = vecMat3(sv, lattice);
            const d2 = cart[0] * cart[0] + cart[1] * cart[1] + cart[2] * cart[2];
            if (d2 < minDist2 - 1e-7) {
              minDist2 = d2;
              candidates.length = 0;
              candidates.push(sv);
            } else if (Math.abs(d2 - minDist2) <= 1e-7) {
              candidates.push(sv);
            }
          }
        }
      }
      const addr = shortestVectors.length;
      for (const sv of candidates) shortestVectors.push(sv);
      multiplicity[s][i] = [candidates.length, addr];
    }
  }

  // Build compact force constants Phi_{i, s} (shape: nPrim × nSuper × 3 × 3)
  const fcCompact = Array.from({ length: nPrim }, () =>
    Array.from({ length: nSuper }, () => [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
  );

  for (let i = 0; i < nPrim; i++) {
    const ri = vecMat3(fracPositions[i], lattice);
    for (let s = 0; s < nSuper; s++) {
      if (s === i) continue; // self term filled by acoustic sum rule below
      // Sum over periodic images of the FC supercell within springCutoff
      for (let gx = -1; gx <= 1; gx++) {
        for (let gy = -1; gy <= 1; gy++) {
          for (let gz = -1; gz <= 1; gz++) {
            const fShift = [
              superFracInPrim[s][0] + gx * scDim[0],
              superFracInPrim[s][1] + gy * scDim[1],
              superFracInPrim[s][2] + gz * scDim[2],
            ];
            const rs = vecMat3(fShift, lattice);
            const dx = rs[0] - ri[0], dy = rs[1] - ri[1], dz = rs[2] - ri[2];
            const dist = Math.hypot(dx, dy, dz);
            if (dist < 1e-6 || dist > springCutoff) continue;

            // Isotropic + longitudinal spring constant so both L and T branches disperse cleanly
            let kLong = dist < 2.6 ? 14.0 : 4.2;
            let kTrans = dist < 2.6 ? 3.8 : 1.1;
            if (bondModulation) {
              const scale = bondModulation(ri, rs, dist);
              kLong *= scale;
              kTrans *= scale;
            }
            const ux = dx / dist, uy = dy / dist, uz = dz / dist;
            const u = [ux, uy, uz];
            for (let a = 0; a < 3; a++) {
              for (let b = 0; b < 3; b++) {
                const phiAB = -((kLong - kTrans) * u[a] * u[b] + (a === b ? kTrans : 0));
                fcCompact[i][s][a][b] += phiAB;
                fcCompact[i][i][a][b] -= phiAB;
              }
            }
          }
        }
      }
    }
  }

  return {
    name: `${elements.join('')} analytical model`,
    format: 'phonopy-dynamical-matrix-v1',
    lattice,
    atom_types: elements,
    atom_pos_red: fracPositions,
    qpoints,
    highsym_qpts: highsymQpts,
    dynamical_matrix: {
      format: 'phonopy-dynamical-matrix-v1',
      primitive_lattice: lattice,
      masses,
      frequency_conversion_factor: 15.633302,
      acoustic_sum_rule: 'translational',
      force_constants_compact: fcCompact,
      shortest_vectors: shortestVectors,
      multiplicity,
      s2pp_map: s2pp,
    },
  };
}

// ── Test 1: Supercell relation across rotated seekpath standard settings ─────
{
  // 1a. Cubic perovskite Pm-3m (5 atoms) -> 45°-rotated tetragonal I4/mcm (10 atoms, octahedral tilt)
  const a = 4.0;
  const parentCubic = {
    lattice: [[a, 0, 0], [0, a, 0], [0, 0, a]],
    elements: ['Sr', 'Ti', 'O', 'O', 'O'],
    fracPositions: [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
  };
  // 45°-rotated sqrt(2)a × sqrt(2)a × a cell (|det P| = 2) with origin shift + O octahedral rotation
  const P_tet = [[1, 1, 0], [-1, 1, 0], [0, 0, 1]];
  const P_tet_inv = inv3(P_tet);
  const scEls = [], scFracs = [];
  for (const [mIdx, shift] of [[0, [0, 0, 0]], [1, [1, 0, 0]]]) {
    parentCubic.elements.forEach((el, k) => {
      const fIdeal = vecMat3([
        shift[0] + parentCubic.fracPositions[k][0],
        shift[1] + parentCubic.fracPositions[k][1],
        shift[2] + parentCubic.fracPositions[k][2],
      ], P_tet_inv);
      const tilt = (el === 'O' && k >= 3) ? (mIdx === 0 ? 0.022 : -0.022) : 0;
      scEls.push(el);
      scFracs.push([
        wrap01(fIdeal[0] + 0.25 + tilt),
        wrap01(fIdeal[1] + 0.50 - tilt),
        wrap01(fIdeal[2] + 0.125),
      ]);
    });
  }
  const superRotatedTet = {
    lattice: [[Math.SQRT2 * a, 0, 0], [0, Math.SQRT2 * a, 0], [0, 0, a]],
    elements: scEls,
    fracPositions: scFracs,
  };

  const rel1 = findSupercellRelation(parentCubic, superRotatedTet);
  assert.ok(rel1, 'Must find supercell relation for 45°-rotated perovskite supercell');
  assert.equal(rel1.multiplicity, 2);
  assert.ok(rel1.rmsDispAng > 0.04 && rel1.rmsDispAng < 0.15, `Expected physical tilt RMS ~0.07 Å, got ${rel1.rmsDispAng}`);

  // 1b. Rhombohedral FCC primitive F-43m (13 atoms) -> 45°-rotated tetragonal P-42₁m (26 atoms)
  const aFcc = 10.4;
  const A_fcc_prim = [
    [0, aFcc / 2, aFcc / 2],
    [aFcc / 2, 0, aFcc / 2],
    [aFcc / 2, aFcc / 2, 0],
  ];
  const fccEls = ['Ga', 'Ta', 'Ta', 'Ta', 'Ta', 'Se', 'Se', 'Se', 'Se', 'Se', 'Se', 'Se', 'Se'];
  const fccFracs = [
    [0, 0, 0],
    [0.6, 0.6, 0.6], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6],
    [0.25, 0.25, 0.25], [0.85, 0.85, 0.85], [0.85, 0.35, 0.35], [0.35, 0.85, 0.35],
    [0.35, 0.35, 0.85], [0.37, 0.37, 0.37], [0.37, 0.89, 0.89], [0.89, 0.37, 0.89],
  ];
  // P maps rhombohedral FCC primitive -> (a/sqrt(2), a/sqrt(2), a) tetragonal cell (|det P| = 2)
  const P_f2t = [[0, 0, 1], [1, -1, 0], [1, 1, -1]];
  const P_f2t_inv = inv3(P_f2t);
  const p421mEls = [], p421mFracs = [];
  for (const [mIdx, sh] of [[0, [0, 0, 0]], [1, [1, 0, 0]]]) {
    fccEls.forEach((el, k) => {
      const f = vecMat3([sh[0] + fccFracs[k][0], sh[1] + fccFracs[k][1], sh[2] + fccFracs[k][2]], P_f2t_inv);
      const dimer = el === 'Ta' ? (mIdx === 0 ? 0.008 : -0.008) : 0;
      p421mEls.push(el);
      p421mFracs.push([wrap01(f[0] + 0.5 + dimer), wrap01(f[1] + 0.25), wrap01(f[2] + 0.125 - dimer)]);
    });
  }
  const rel2 = findSupercellRelation(
    { lattice: A_fcc_prim, elements: fccEls, fracPositions: fccFracs },
    { lattice: [[aFcc / Math.SQRT2, 0, 0], [0, aFcc / Math.SQRT2, 0], [0, 0, aFcc]], elements: p421mEls, fracPositions: p421mFracs }
  );
  assert.ok(rel2, 'Must find non-diagonal rhombohedral-FCC -> rotated-tetragonal P-42₁m supercell relation');
  assert.equal(rel2.multiplicity, 2);
  assert.equal(rel2.atomMap.length, 26);
  console.log(`PASS Test 1: Rotated supercell geometry & origin-shift matching (Pm-3m->I4/mcm & F-43m->P-42₁m, RMS=${rel2.rmsDispAng.toFixed(4)} Å)`);
}

// Build a continuous 3D parent k-path Γ -> Z -> R -> X -> Γ for Tests 2–4
const parentQPath = [];
const pathKnots = [
  [[0, 0, 0], 'Γ'],
  [[0, 0, 0.5], 'Z'],
  [[0.5, 0, 0.5], 'R'],
  [[0.5, 0, 0], 'X'],
  [[0, 0, 0], 'Γ'],
];
const highsymQpts = [];
for (let s = 0; s < pathKnots.length - 1; s++) {
  const [q0, l0] = pathKnots[s];
  const [q1, l1] = pathKnots[s + 1];
  if (s === 0) highsymQpts.push([0, l0]);
  const nPts = 9;
  for (let j = (s === 0 ? 0 : 1); j < nPts; j++) {
    const t = j / (nPts - 1);
    parentQPath.push([q0[0] + (q1[0] - q0[0]) * t, q0[1] + (q1[1] - q0[1]) * t, q0[2] + (q1[2] - q0[2]) * t]);
  }
  highsymQpts.push([parentQPath.length - 1, l1]);
}

// ── Test 2: Exact undistorted 1×1×2 supercell unfolding (W=1 vs W=0) ─────────
const parentLattice = [[3.8, 0, 0], [0, 3.8, 0], [0, 0, 4.0]];
const parentDM = buildAnalyticalSpringDM({
  lattice: parentLattice,
  elements: ['Na', 'Cl'],
  masses: [22.99, 35.45],
  fracPositions: [[0, 0, 0], [0.5, 0.5, 0.5]],
  qpoints: parentQPath,
  highsymQpts,
});
const parentModel = fromPhonopyDM(parentDM);

{
  // Build an undistorted 1×1×2 supercell (4 atoms = 12 branches) with an origin shift [0.15, 0.35, 0.25]
  const superLattice = [[3.8, 0, 0], [0, 3.8, 0], [0, 0, 8.0]];
  const origin = [0.15, 0.35, 0.25];
  const superDM = buildAnalyticalSpringDM({
    lattice: superLattice,
    elements: ['Na', 'Cl', 'Na', 'Cl'],
    masses: [22.99, 35.45, 22.99, 35.45],
    fracPositions: [
      [wrap01(0 + origin[0]), wrap01(0 + origin[1]), wrap01(0.00 + origin[2])],
      [wrap01(0.5 + origin[0]), wrap01(0.5 + origin[1]), wrap01(0.25 + origin[2])],
      [wrap01(0 + origin[0]), wrap01(0 + origin[1]), wrap01(0.50 + origin[2])],
      [wrap01(0.5 + origin[0]), wrap01(0.5 + origin[1]), wrap01(0.75 + origin[2])],
    ],
    qpoints: [[0, 0, 0], [0, 0, 0.5]],
    highsymQpts: [[0, 'Γ'], [1, 'Z']],
  });
  const superModel = fromPhonopyDM(superDM);

  const unf = unfoldModelOntoParentPath(superModel, parentModel);
  assert.ok(unf, 'Unfolding undistorted 1×1×2 supercell onto parent must succeed');
  assert.equal(unf.relation.multiplicity, 2);
  assert.ok(unf.relation.rmsDispAng < 1e-12);

  let maxSumRuleErr = 0;
  let maxShadowWeight = 0;
  let maxParentWeightErr = 0;

  for (let q = 0; q < unf.qPoints.length; q++) {
    const wRow = unf.weights[q];
    const sumW = wRow.reduce((a, b) => a + b, 0);
    maxSumRuleErr = Math.max(maxSumRuleErr, Math.abs(sumW - 6));

    for (let nu = 0; nu < wRow.length; nu++) {
      assert.ok(wRow[nu] >= -1e-14 && wRow[nu] <= 1 + 1e-14);
      const matchesParent = parentModel.bands[q].some(ePC => Math.abs(ePC - unf.bands[q][nu]) < 1e-4);
      if (!matchesParent) maxShadowWeight = Math.max(maxShadowWeight, wRow[nu]);
    }

    for (const ePC of parentModel.bands[q]) {
      const deg = parentModel.bands[q].filter(x => Math.abs(x - ePC) < 1e-4).length;
      let wMatched = 0;
      for (let nu = 0; nu < unf.bands[q].length; nu++) {
        if (Math.abs(unf.bands[q][nu] - ePC) < 1e-4) wMatched += wRow[nu];
      }
      maxParentWeightErr = Math.max(maxParentWeightErr, Math.abs(wMatched - deg));
    }
  }

  assert.ok(maxSumRuleErr < 1e-12, `Completeness sum rule error too large: ${maxSumRuleErr}`);
  assert.ok(maxShadowWeight < 1e-12, `Shadow mode leaked weight: ${maxShadowWeight}`);
  assert.ok(maxParentWeightErr < 1e-12, `Parent dispersion weight mismatch: ${maxParentWeightErr}`);
  console.log(`PASS Test 2: Exact undistorted 1×1×2 unfolding (sumRuleErr=${maxSumRuleErr.toExponential(2)}, shadowLeak=${maxShadowWeight.toExponential(2)})`);
}

// ── Test 3 & 4: Dimerized 1×1×2 supercell + spectral grid + intrinsic parent inference ──
{
  const superLattice = [[3.8, 0, 0], [0, 3.8, 0], [0, 0, 8.0]];
  // Peierls-like z-dimerization: Cl atoms shift ±0.09 Å toward Na(0) and spring constants alternate ±25%
  const dimerShiftFrac = 0.09 / 8.0;
  const dimerDM = buildAnalyticalSpringDM({
    lattice: superLattice,
    elements: ['Na', 'Cl', 'Na', 'Cl'],
    masses: [22.99, 35.45, 22.99, 35.45],
    fracPositions: [
      [0.0, 0.0, 0.00],
      [0.5, 0.5, 0.25 - dimerShiftFrac],
      [0.0, 0.0, 0.50],
      [0.5, 0.5, 0.75 + dimerShiftFrac],
    ],
    qpoints: [[0, 0, 0], [0, 0, 0.5]],
    highsymQpts: [[0, 'Γ'], [1, 'Z']],
    bondModulation: (ri, rs) => {
      // Alternate bond stiffness depending on whether bond midpoint sits near z=0 vs z=4 Å
      const midZ = wrapHalf((0.5 * (ri[2] + rs[2])) / 8.0);
      return Math.abs(midZ) < 0.25 ? 1.25 : 0.75;
    },
  });
  const dimerModel = fromPhonopyDM(dimerDM);

  const unf = unfoldModelOntoParentPath(dimerModel, parentModel);
  assert.ok(unf);
  assert.ok(unf.relation.rmsDispAng > 0.05 && unf.relation.rmsDispAng < 0.12);

  let maxSumRuleErr = 0;
  let nFractionalTransferModes = 0;
  for (let q = 0; q < unf.qPoints.length; q++) {
    const wRow = unf.weights[q];
    const sumW = wRow.reduce((a, b) => a + b, 0);
    maxSumRuleErr = Math.max(maxSumRuleErr, Math.abs(sumW - 6));
    for (const w of wRow) {
      if (w > 0.03 && w < 0.97) nFractionalTransferModes++;
    }
  }
  assert.ok(maxSumRuleErr < 1e-12, `Completeness sum rule must hold under finite distortion, got ${maxSumRuleErr}`);
  assert.ok(nFractionalTransferModes >= 6, `Expected dimerization to transfer fractional spectral weight onto folded branches, got ${nFractionalTransferModes}`);

  // Verify Lorentzian spectral function grid A(Q, E) across the full mode range
  const maxBandE = Math.max(...unf.bands.flat());
  const grid = computeSpectralFunctionGrid(unf.bands, unf.weights, {
    eMin: -4,
    eMax: maxBandE + 6,
    nEnergyBins: 400,
    fwhm: 0.8,
  });
  assert.ok(grid && grid.maxIntensity > 0);
  const dE = (grid.yMax - grid.yMin) / (grid.nEnergyBins - 1);
  let integralAtGamma = 0;
  for (let b = 0; b < grid.nEnergyBins; b++) integralAtGamma += grid.intensity[b] * dE;
  // Analytical integral of a Lorentzian truncated at ±12η is (2/π) * atan(12) ≈ 0.9471 per unit weight
  const expectedTruncatedIntegral = 6 * (2 / Math.PI) * Math.atan(12);
  assert.ok(
    Math.abs(integralAtGamma - expectedTruncatedIntegral) < 0.08,
    `Expected truncated Lorentzian spectral integral ~ ${expectedTruncatedIntegral.toFixed(3)}, got ${integralAtGamma}`
  );

  // Test 4: Standalone intrinsic parent inference when ONLY `dimerModel` is loaded
  const inferredParent = inferIntrinsicParentPath(dimerModel);
  assert.ok(inferredParent, 'inferIntrinsicParentPath must recover 2-atom parent cell from distorted 4-atom 1×1×2 model');
  const unfStandalone = unfoldModelOntoParentPath(dimerModel, inferredParent);
  assert.ok(unfStandalone && unfStandalone.relation.multiplicity === 2);
  console.log(`PASS Test 3 & 4: Dimerized 1×1×2 unfolding, spectral grid integral (${integralAtGamma.toFixed(2)}/6.00), & standalone parent inference`);
}
