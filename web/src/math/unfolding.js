// web/src/math/unfolding.js
//
// ═══════════════════════════════════════════════════════════════════════════════
// PHONON BAND UNFOLDING (SUPERCELL → PARENT BRILLOUIN ZONE)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Physical Motivation
// -------------------
// When a crystal develops a local symmetry-breaking distortion that enlarges the
// primitive unit cell by an integer factor M = V_SC / V_PC (for example, a 1×1×2
// dimerization, a 2×2×1 or 2×2×2 perovskite octahedral tilt, or an F-centered ->
// P-centered transition), its Brillouin zone shrinks by the same factor M:
//
//   • The number of phonon branches multiplies from 3 * N_PC to 3 * M * N_PC.
//   • High-symmetry paths of the lower-symmetry supercell (e.g. tetragonal or
//     monoclinic) no longer match the parent crystal's Brillouin-zone path
//     (e.g. cubic Γ–X–M–Γ–R or FCC Γ–X–W–K–Γ–L).
//   • Many of the 3 * M * N_PC supercell branches are "zone-folded shadow modes"
//     that carry near-zero plane-wave character at a given parent wave vector Q,
//     whereas genuine physical splittings redistribute spectral weight W_{Q,ν}
//     across split branches.
//
// Band unfolding projects each supercell normal mode |q_0, ν> back onto the
// Bloch translation eigenstates of the undistorted parent primitive lattice T_PC
// along the parent wave-vector path Q (following Allen, Berlijn, Casavant &
// Soler, Phys. Rev. B 87, 085322 (2013); Boykin et al., J. Phys.: Condens.
// Matter 26, 395501 (2014); Ikeda, Carreras, Seko, Togo & Tanaka, Phys. Rev. B
// 95, 024305 (2017) — the Phonopy/`upho` convention).
//
//
// Exact Equations & Phase Conventions
// -----------------------------------
// 1. Lattice row-vector convention (used throughout `web/src/math/`):
//      A_PC = [a1_PC; a2_PC; a3_PC]   (3×3 matrix whose rows are lattice vectors)
//      A_SC_ideal = P * A_PC          (P is a 3×3 integer matrix, |det(P)| = M)
//
//    Note that spglib / seekpath may rotate a lower-symmetry supercell in 3D
//    Cartesian space by an orthogonal matrix R_orient in O(3):
//      A_SC * R_orient^T ≈ P * A_PC.
//    Because the metric (Gram) tensor G = A * A^T is invariant under O(3), we
//    find P directly from G_SC ≈ P * G_PC * P^T and verify 1-to-1 atomic site
//    matching across all elements.
//
// 2. Reciprocal wave-vector mapping (parent Q_PC -> supercell q_SC):
//    A plane wave exp(i k_cart . r) has fractional coordinates Q_i = (k_cart . a_i)/(2π).
//    Since A_SC_ideal = P * A_PC, a parent fractional wave vector Q_PC maps to
//    the unwrapped supercell fractional wave vector:
//
//      q_SC(Q_PC) = P * Q_PC  =  q_0 + G_SC,
//
//    where q_0 in [-1/2, 1/2)^3 is the reduced wave vector inside the first
//    supercell Brillouin zone and G_SC = q_SC - q_0 in Z^3 is a supercell
//    reciprocal-lattice vector.
//
// 3. Phonopy C-type Bloch gauge vs. physical intra-supercell polarization:
//    Phonopy constructs the mass-weighted dynamical matrix D_{IJ}(q_0) with the
//    atomic-position ("C-type") phase factor exp(+i 2π q_0 . (R_SC + x_J - x_I)).
//    Consequently, a normal mode eigenvector e^C_{I,α}(q_0, ν) describes real-space
//    mass-weighted displacements:
//
//      sqrt(m_I) * u_{L, I, α} = w_{I, α}(q_0, ν) * exp(+i 2π q_0 . L),
//      where   w_{I, α}(q_0, ν) = e^C_{I, α}(q_0, ν) * exp(+i 2π q_0 . x_I).
//
//    Notice that w_{I, α} is strictly gauge-invariant under wrapping q_0 -> q_0 + G_SC!
//
// 4. Parent-lattice translational projection & spectral weight W_{Q, ν}:
//    Each supercell atom I in {0 .. N_SC - 1} corresponds bijectively to a
//    parent basis site κ(I) in {0 .. N_PC - 1} located at ideal unwrapped
//    supercell fractional coordinate r^ideal_I (so P * Q_PC . r^ideal_I equals
//    Q_PC . (t_{m(I)} + τ_{κ(I)}) in parent coordinates).
//
//    Projecting the mode's displacement pattern onto the parent Bloch character
//    exp(+i 2π q_SC . r^ideal_I) for each parent site κ and Cartesian axis α gives:
//
//      c_{κ, α}(Q, ν) = (1 / sqrt(M)) * sum_{I : κ(I) = κ}
//                         exp(-i 2π q_SC . r^ideal_I) * w_{I, α}(q_0, ν)
//                     = (1 / sqrt(M)) * sum_{I : κ(I) = κ}
//                         exp(i 2π [ q_0 . x_I  -  q_SC . r^ideal_I ]) * e^C_{I, α}(q_0, ν)
//
//      W_{Q, ν} = sum_{κ = 0}^{N_PC - 1} sum_{α in {x, y, z}} |c_{κ, α}(Q, ν)|^2.
//
// 5. Three Exact Physical Guarantees:
//    (a) Bounds:        0 <= W_{Q, ν} <= 1  for every mode ν.
//    (b) Completeness:  sum_{ν = 1}^{3 N_SC} W_{Q, ν} = 3 * N_PC  at every Q,
//                       exact to machine precision (~1e-15) even when atoms have
//                       finite symmetry-breaking static shifts x_I != r^ideal_I.
//    (c) Undistorted:   When the supercell is an undistorted repetition of the
//                       parent cell, 3 * N_PC modes have W_{Q, ν} = 1 (recovering
//                       the exact parent dispersion) and all 3*(N_SC - N_PC)
//                       folded shadow modes have W_{Q, ν} = 0.
// ═══════════════════════════════════════════════════════════════════════════════

import { det3, inv3, matMul3, vecMat3, relabelAtoms } from './cells.js';
import { conventionalLattice, reciprocalLattice, pathDistances } from './reciprocal.js';
import { analyzeBravais } from './bravais.js';
import { buildBZModel } from './highsym.js';
import { solveDynamicalMatrixAtQ } from '../io/phonopyDM.js';

const TWO_PI = 2 * Math.PI;
const wrap01 = (x) => ((x % 1) + 1) % 1;
const wrapHalf = (x) => x - Math.round(x);

/** Gram (metric) tensor G = A * A^T of a 3×3 row-vector lattice matrix A. */
function metricTensor(A) {
  const G = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      G[i][j] = A[i][0] * A[j][0] + A[i][1] * A[j][1] + A[i][2] * A[j][2];
    }
  }
  return G;
}

/** Multiply 3×3 matrix M by column 3-vector v: out = M * v. */
function matVec3(M, v) {
  return [
    M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
    M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
    M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2],
  ];
}

/** Transpose a 3×3 matrix. */
function transpose3(M) {
  return [
    [M[0][0], M[1][0], M[2][0]],
    [M[0][1], M[1][1], M[2][1]],
    [M[0][2], M[1][2], M[2][2]],
  ];
}

/**
 * Compute the closest orthogonal 3×3 matrix R in O(3) to `F` via Newton–Schulz /
 * Higham polar iteration Q_{k+1} = 0.5 * (Q_k + Q_k^{-T}). Converges quadratically
 * to machine precision in ~6 iterations for lattice strain < 20%.
 */
function orthogonalPolarFactor(F) {
  let Q = F.map(row => row.slice());
  for (let iter = 0; iter < 8; iter++) {
    const Qinv = inv3(Q);
    if (!Qinv) break;
    const QinvT = transpose3(Qinv);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        Q[i][j] = 0.5 * (Q[i][j] + QinvT[i][j]);
      }
    }
  }
  return Q;
}

/**
 * Extract `{ lattice, elements, fracPositions }` from the app's `baseStructure` object.
 */
export function extractCellFromBaseStructure(baseStructure) {
  if (!baseStructure?.v1 || !baseStructure?.hsym_xyz) return null;
  const lattice = conventionalLattice(
    baseStructure.v1,
    baseStructure.v2,
    baseStructure.v3,
    baseStructure.dim || [1, 1, 1]
  );
  const reverseAtomDic = {};
  for (const [el, idxs] of Object.entries(baseStructure.atomDic || {})) {
    for (const idx of idxs) reverseAtomDic[idx] = el;
  }
  const nAtoms = baseStructure.hsym_xyz.length / 3;
  const elements = new Array(nAtoms);
  const fracPositions = new Array(nAtoms);
  for (let i = 0; i < nAtoms; i++) {
    const rn = baseStructure.atomType?.[i] ?? i;
    elements[i] = reverseAtomDic[rn] || 'X';
    fracPositions[i] = [
      wrap01(baseStructure.hsym_xyz[i * 3]),
      wrap01(baseStructure.hsym_xyz[i * 3 + 1]),
      wrap01(baseStructure.hsym_xyz[i * 3 + 2]),
    ];
  }
  return { lattice, elements, fracPositions };
}

/**
 * Enumerate the M integer lattice translations `t_m` in Z^3 (parent fractional units)
 * that lie inside one supercell `A_SC = P * A_PC`.
 */
function enumerateParentSubcellsInSupercell(P) {
  const Pinv = inv3(P);
  const expectedM = Math.round(Math.abs(det3(P)));
  if (!Pinv || expectedM < 1) return [];

  // Bound the integer box in parent coordinates from the 8 corners [0,1]^3 of the supercell
  let minI = 0, maxI = 0, minJ = 0, maxJ = 0, minK = 0, maxK = 0;
  for (let c0 = 0; c0 <= 1; c0++) {
    for (let c1 = 0; c1 <= 1; c1++) {
      for (let c2 = 0; c2 <= 1; c2++) {
        const cornerPC = vecMat3([c0, c1, c2], P);
        minI = Math.min(minI, Math.floor(cornerPC[0]) - 1);
        maxI = Math.max(maxI, Math.ceil(cornerPC[0]) + 1);
        minJ = Math.min(minJ, Math.floor(cornerPC[1]) - 1);
        maxJ = Math.max(maxJ, Math.ceil(cornerPC[1]) + 1);
        minK = Math.min(minK, Math.floor(cornerPC[2]) - 1);
        maxK = Math.max(maxK, Math.ceil(cornerPC[2]) + 1);
      }
    }
  }

  const seenCosets = new Set();
  const subcells = [];
  const keyOfFrac = (f) => f.map(x => Math.round(wrap01(x) * 1440) % 1440).join(',');

  for (let i = minI; i <= maxI; i++) {
    for (let j = minJ; j <= maxJ; j++) {
      for (let k = minK; k <= maxK; k++) {
        const fSC = vecMat3([i, j, k], Pinv);
        // Prefer canonical representatives inside [0, 1)
        if (fSC[0] < -1e-6 || fSC[0] >= 1 - 1e-6) continue;
        if (fSC[1] < -1e-6 || fSC[1] >= 1 - 1e-6) continue;
        if (fSC[2] < -1e-6 || fSC[2] >= 1 - 1e-6) continue;
        const key = keyOfFrac(fSC);
        if (!seenCosets.has(key)) {
          seenCosets.add(key);
          subcells.push({
            parentShift: [i, j, k],
            superFracShift: [wrap01(fSC[0]), wrap01(fSC[1]), wrap01(fSC[2])],
          });
        }
      }
    }
  }

  // Fallback sweep if boundary-rounding missed a coset on oblique cells
  if (subcells.length < expectedM) {
    for (let i = minI; i <= maxI && subcells.length < expectedM; i++) {
      for (let j = minJ; j <= maxJ && subcells.length < expectedM; j++) {
        for (let k = minK; k <= maxK && subcells.length < expectedM; k++) {
          const fSC = vecMat3([i, j, k], Pinv);
          const key = keyOfFrac(fSC);
          if (!seenCosets.has(key)) {
            seenCosets.add(key);
            subcells.push({
              parentShift: [i, j, k],
              superFracShift: [wrap01(fSC[0]), wrap01(fSC[1]), wrap01(fSC[2])],
            });
          }
        }
      }
    }
  }

  return subcells;
}

/**
 * Discover the exact crystallographic supercell transformation `P` (where
 * `A_SC * R_orient^T ≈ P * A_PC`) and the 1-to-1 atom mapping between a
 * potentially rotated/standardized lower-symmetry supercell and a parent cell.
 *
 * Works seamlessly even when `spglib` / `seekpath` standardized the two cells
 * into differently oriented 3D Cartesian frames (e.g. 45°-rotated tetragonal
 * P-42₁m or I4/mcm vs. rhombohedral primitive FCC F-43m or cubic Pm-3m).
 *
 * @param {{ lattice: number[][], elements: string[], fracPositions: number[][] }} parentCell
 * @param {{ lattice: number[][], elements: string[], fracPositions: number[][] }} superCell
 * @param {{ strainTol?: number, maxAtomicDispAng?: number }} [options]
 */
export function findSupercellRelation(parentCell, superCell, options = {}) {
  const { strainTol = 0.08, maxAtomicDispAng = 0.75 } = options;
  if (!parentCell || !superCell) return null;

  const nPC = parentCell.elements.length;
  const nSC = superCell.elements.length;
  if (nPC === 0 || nSC === 0 || nSC % nPC !== 0) return null;
  const multiplicity = nSC / nPC;

  // Step 1: Verify chemical stoichiometry matches with exact factor `multiplicity`.
  const countEl = (els) => {
    const m = new Map();
    for (const el of els) m.set(el, (m.get(el) || 0) + 1);
    return m;
  };
  const pcCounts = countEl(parentCell.elements);
  const scCounts = countEl(superCell.elements);
  if (pcCounts.size !== scCounts.size) return null;
  for (const [el, cPC] of pcCounts.entries()) {
    if (scCounts.get(el) !== cPC * multiplicity) return null;
  }

  // Step 2: Find candidate integer 3×3 matrices P (|det P| = multiplicity) whose
  //         ideal supercell Gram tensor P * G_PC * P^T matches G_SC = A_SC * A_SC^T.
  const A_PC = parentCell.lattice;
  const A_SC = superCell.lattice;
  const G_PC = metricTensor(A_PC);
  const G_SC = metricTensor(A_SC);
  const scLengths = [0, 1, 2].map(r => Math.sqrt(G_SC[r][r]));

  // Enumerate integer row vectors p in [-maxCoord, maxCoord]^3 matching each row length of A_SC
  const minPcLen = Math.min(Math.sqrt(G_PC[0][0]), Math.sqrt(G_PC[1][1]), Math.sqrt(G_PC[2][2]));
  const maxCoord = Math.min(6, Math.max(2, Math.ceil(Math.max(...scLengths) / Math.max(0.5, minPcLen)) + 1));

  const rowCandidates = [[], [], []];
  for (let i = -maxCoord; i <= maxCoord; i++) {
    for (let j = -maxCoord; j <= maxCoord; j++) {
      for (let k = -maxCoord; k <= maxCoord; k++) {
        if (i === 0 && j === 0 && k === 0) continue;
        const vCart = vecMat3([i, j, k], A_PC);
        const len = Math.hypot(vCart[0], vCart[1], vCart[2]);
        for (let r = 0; r < 3; r++) {
          if (Math.abs(len - scLengths[r]) / scLengths[r] <= strainTol) {
            rowCandidates[r].push({ p: [i, j, k], vCart, len });
          }
        }
      }
    }
  }
  if (rowCandidates.some(rc => rc.length === 0)) return null;

  const matrixCandidates = [];
  for (const r0 of rowCandidates[0]) {
    for (const r1 of rowCandidates[1]) {
      // Check (0, 1) dot-product compatibility before inner loop
      const g01 = r0.vCart[0] * r1.vCart[0] + r0.vCart[1] * r1.vCart[1] + r0.vCart[2] * r1.vCart[2];
      if (Math.abs(g01 - G_SC[0][1]) > strainTol * scLengths[0] * scLengths[1] * 1.5) continue;

      for (const r2 of rowCandidates[2]) {
        const P = [r0.p, r1.p, r2.p];
        const detP = det3(P);
        // Prefer proper (right-handed, det > 0) supercell embeddings first, allow improper if needed
        if (Math.round(Math.abs(detP)) !== multiplicity) continue;

        const g02 = r0.vCart[0] * r2.vCart[0] + r0.vCart[1] * r2.vCart[1] + r0.vCart[2] * r2.vCart[2];
        const g12 = r1.vCart[0] * r2.vCart[0] + r1.vCart[1] * r2.vCart[1] + r1.vCart[2] * r2.vCart[2];
        if (Math.abs(g02 - G_SC[0][2]) > strainTol * scLengths[0] * scLengths[2] * 1.5) continue;
        if (Math.abs(g12 - G_SC[1][2]) > strainTol * scLengths[1] * scLengths[2] * 1.5) continue;

        const metricErr = (
          Math.abs(r0.len - scLengths[0]) / scLengths[0] +
          Math.abs(r1.len - scLengths[1]) / scLengths[1] +
          Math.abs(r2.len - scLengths[2]) / scLengths[2] +
          Math.abs(g01 - G_SC[0][1]) / (scLengths[0] * scLengths[1]) +
          Math.abs(g02 - G_SC[0][2]) / (scLengths[0] * scLengths[2]) +
          Math.abs(g12 - G_SC[1][2]) / (scLengths[1] * scLengths[2])
        );
        matrixCandidates.push({ P, detP, metricErr });
      }
    }
  }

  if (!matrixCandidates.length) return null;
  matrixCandidates.sort((a, b) => (
    // Prefer right-handed (detP > 0) when metric error is tied
    (a.metricErr - b.metricErr) || ((b.detP > 0 ? 1 : 0) - (a.detP > 0 ? 1 : 0))
  ));

  // Step 3: Test candidate matrices P against actual atomic positions & elements,
  //         solving for any origin shift Δ introduced by spglib cell standardization.
  const anchorElement = [...pcCounts.entries()].sort((a, b) => a[1] - b[1])[0][0];
  const scAnchorIdx = superCell.elements.indexOf(anchorElement);

  let bestMatch = null;
  const maxToTest = Math.min(matrixCandidates.length, 32);

  for (let ci = 0; ci < maxToTest; ci++) {
    const { P, detP, metricErr } = matrixCandidates[ci];
    const Pinv = inv3(P);
    if (!Pinv) continue;

    const subcells = enumerateParentSubcellsInSupercell(P);
    if (subcells.length !== multiplicity) continue;

    // Build all N_SC ideal parent sites inside the supercell
    const idealSites = [];
    for (let m = 0; m < multiplicity; m++) {
      const tParent = subcells[m].parentShift;
      for (let kappa = 0; kappa < nPC; kappa++) {
        const tau = parentCell.fracPositions[kappa];
        const unwrappedParentPos = [tParent[0] + tau[0], tParent[1] + tau[1], tParent[2] + tau[2]];
        const idealSuperFrac = vecMat3(unwrappedParentPos, Pinv);
        idealSites.push({
          kappa,
          subcellIdx: m,
          element: parentCell.elements[kappa],
          idealSuperFrac, // exact unwrapped f_SC = (t_m + tau_kappa) * P^-1
        });
      }
    }

    // Candidate origin shifts Δ = x_anchor - y_ideal (for every ideal site of the anchor element)
    for (let j = 0; j < idealSites.length; j++) {
      if (idealSites[j].element !== anchorElement) continue;
      const originShift = [
        superCell.fracPositions[scAnchorIdx][0] - idealSites[j].idealSuperFrac[0],
        superCell.fracPositions[scAnchorIdx][1] - idealSites[j].idealSuperFrac[1],
        superCell.fracPositions[scAnchorIdx][2] - idealSites[j].idealSuperFrac[2],
      ];

      // Greedy/nearest 1-to-1 matching per element in Cartesian Å
      const usedIdeal = new Uint8Array(nSC);
      const atomToIdeal = new Array(nSC);
      let sumDispSq = 0;
      let ok = true;

      for (let i = 0; i < nSC; i++) {
        const el = superCell.elements[i];
        const xSC = superCell.fracPositions[i];
        let bestJ = -1;
        let bestDistSq = Infinity;
        let bestUnwrappedIdealFrac = null;

        for (let candJ = 0; candJ < nSC; candJ++) {
          if (usedIdeal[candJ] || idealSites[candJ].element !== el) continue;
          const y0 = idealSites[candJ].idealSuperFrac;
          // Shifted ideal coordinate in the supercell frame: y_shifted = y0 + originShift
          const df0 = wrapHalf(xSC[0] - (y0[0] + originShift[0]));
          const df1 = wrapHalf(xSC[1] - (y0[1] + originShift[1]));
          const df2 = wrapHalf(xSC[2] - (y0[2] + originShift[2]));
          const dCart = vecMat3([df0, df1, df2], A_SC);
          const d2 = dCart[0] * dCart[0] + dCart[1] * dCart[1] + dCart[2] * dCart[2];
          if (d2 < bestDistSq) {
            bestDistSq = d2;
            bestJ = candJ;
            // Exact periodic image of y0 closest to (xSC - originShift) so phase is continuous!
            bestUnwrappedIdealFrac = [
              (xSC[0] - originShift[0]) - df0,
              (xSC[1] - originShift[1]) - df1,
              (xSC[2] - originShift[2]) - df2,
            ];
          }
        }

        if (bestJ < 0 || bestDistSq > maxAtomicDispAng * maxAtomicDispAng) {
          ok = false;
          break;
        }
        usedIdeal[bestJ] = 1;
        sumDispSq += bestDistSq;
        atomToIdeal[i] = {
          parentAtomIdx: idealSites[bestJ].kappa,
          subcellIdx: idealSites[bestJ].subcellIdx,
          idealSuperFrac: bestUnwrappedIdealFrac,
          dispAng: Math.sqrt(bestDistSq),
        };
      }

      if (!ok) continue;
      const rmsDispAng = Math.sqrt(sumDispSq / nSC);

      if (!bestMatch || rmsDispAng < bestMatch.rmsDispAng - 1e-5) {
        // Refine originShift by subtracting the mean fractional displacement so ∑ δf_I = 0
        let meanDf0 = 0, meanDf1 = 0, meanDf2 = 0;
        for (let i = 0; i < nSC; i++) {
          const xSC = superCell.fracPositions[i];
          const yId = atomToIdeal[i].idealSuperFrac;
          meanDf0 += wrapHalf(xSC[0] - (yId[0] + originShift[0]));
          meanDf1 += wrapHalf(xSC[1] - (yId[1] + originShift[1]));
          meanDf2 += wrapHalf(xSC[2] - (yId[2] + originShift[2]));
        }
        const refinedOrigin = [
          originShift[0] + meanDf0 / nSC,
          originShift[1] + meanDf1 / nSC,
          originShift[2] + meanDf2 / nSC,
        ];
        let refinedSumSq = 0;
        for (let i = 0; i < nSC; i++) {
          const xSC = superCell.fracPositions[i];
          const yId = atomToIdeal[i].idealSuperFrac;
          const dCart = vecMat3([
            wrapHalf(xSC[0] - (yId[0] + refinedOrigin[0])),
            wrapHalf(xSC[1] - (yId[1] + refinedOrigin[1])),
            wrapHalf(xSC[2] - (yId[2] + refinedOrigin[2])),
          ], A_SC);
          const d2 = dCart[0] * dCart[0] + dCart[1] * dCart[1] + dCart[2] * dCart[2];
          atomToIdeal[i].dispAng = Math.sqrt(d2);
          refinedSumSq += d2;
        }

        // Cartesian frame alignment: A_SC * R_orient^T ≈ P * A_PC  =>  R_orient^T ≈ A_SC^-1 * (P * A_PC)
        const A_SC_inv = inv3(A_SC);
        const F_T = A_SC_inv ? matMul3(A_SC_inv, matMul3(P, A_PC)) : [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
        const R_orient = transpose3(orthogonalPolarFactor(F_T));

        bestMatch = {
          P,
          detP: Math.round(detP),
          multiplicity,
          nParentAtoms: nPC,
          nSuperAtoms: nSC,
          originShift: refinedOrigin,
          rmsDispAng: Math.sqrt(refinedSumSq / nSC),
          metricErr,
          cartesianRotationToParent: R_orient,
          atomMap: atomToIdeal,
        };
      }
    }

    // Stop early if we found a nearly exact match (< 0.08 Å RMS static distortion)
    if (bestMatch && bestMatch.rmsDispAng < 0.08) break;
  }

  return bestMatch;
}

/**
 * Compute the unfolded parent-Brillouin-zone spectral weights `W_{Q, ν}` for all
 * `3 * N_SC` phonon modes at a single parent wave vector `Q_PC`.
 *
 * Physics step-by-step:
 *   1. For each supercell atom `I`, restore the gauge-invariant displacement-wave
 *      phase `exp(+i 2π q_0 . x_I)` (undoing Phonopy's C-type intra-cell phase)
 *      and apply the parent Bloch translation character `exp(-i 2π q_SC . r^ideal_I)`.
 *   2. Sum the phased complex eigenvector components coherently over the `M`
 *      supercell atoms that belong to the same parent basis site `κ` (`0 .. N_PC-1`),
 *      divided by `sqrt(M)`.
 *   3. Sum `|c_{κ, α}|^2` over all parent basis sites `κ` and Cartesian axes `α`
 *      to obtain `W_{Q, ν}` in `[0, 1]`, satisfying `sum_ν W_{Q, ν} = 3 * N_PC`.
 *
 * @param {{ real: Float64Array|number[], imag: Float64Array|number[] }[]} eigvecsAtQ0
 * @param {number[]} qSuperUnwrapped - `q_SC = P * Q_PC` (unwrapped supercell reciprocal coords)
 * @param {number[]} qSuperWrapped   - `q_0 = wrapHalf(q_SC)` in `[-0.5, 0.5)^3`
 * @param {number[][]} superFracPositions - actual fractional coordinates `x_I` in `superCell`
 * @param {object} relation - output of `findSupercellRelation`
 * @returns {Float64Array} `weights[ν]` for `ν = 0 .. 3*N_SC - 1`
 */
export function computeModeSpectralWeights(
  eigvecsAtQ0,
  qSuperUnwrapped,
  qSuperWrapped,
  superFracPositions,
  relation
) {
  const { multiplicity, nParentAtoms, nSuperAtoms, atomMap } = relation;
  const nModes = eigvecsAtQ0.length;
  const invMultiplicity = 1 / multiplicity;

  // Step 1: Precompute the combined Bloch unfolding phase per supercell atom I:
  //         theta_I = 2π * ( q_0 . x_I  -  q_SC_unwrapped . r^ideal_I )
  //         Note: we also include `originShift` on r^ideal_I so x_I and r^ideal_I
  //         live in the exact same supercell coordinate origin (a global constant
  //         phase shift exp(-i 2π q_SC . originShift) cancels identically in |c|^2!).
  const phaseCos = new Float64Array(nSuperAtoms);
  const phaseSin = new Float64Array(nSuperAtoms);
  const origin = relation.originShift || [0, 0, 0];

  for (let i = 0; i < nSuperAtoms; i++) {
    const xActual = superFracPositions[i];
    const rIdealShifted = [
      atomMap[i].idealSuperFrac[0] + origin[0],
      atomMap[i].idealSuperFrac[1] + origin[1],
      atomMap[i].idealSuperFrac[2] + origin[2],
    ];
    const blochRestoreDot = (
      qSuperWrapped[0] * xActual[0] +
      qSuperWrapped[1] * xActual[1] +
      qSuperWrapped[2] * xActual[2]
    );
    const parentTargetDot = (
      qSuperUnwrapped[0] * rIdealShifted[0] +
      qSuperUnwrapped[1] * rIdealShifted[1] +
      qSuperUnwrapped[2] * rIdealShifted[2]
    );
    const angle = TWO_PI * (blochRestoreDot - parentTargetDot);
    phaseCos[i] = Math.cos(angle);
    phaseSin[i] = Math.sin(angle);
  }

  // Reusable accumulators for c_{κ, α} across parent sites κ = 0..N_PC-1, α = 0..2
  const projReal = new Float64Array(nParentAtoms * 3);
  const projImag = new Float64Array(nParentAtoms * 3);
  const weights = new Float64Array(nModes);

  // Step 2 & 3: Project every supercell mode ν onto the parent Bloch basis
  for (let nu = 0; nu < nModes; nu++) {
    const evRe = eigvecsAtQ0[nu].real;
    const evIm = eigvecsAtQ0[nu].imag;
    projReal.fill(0);
    projImag.fill(0);

    for (let i = 0; i < nSuperAtoms; i++) {
      const c = phaseCos[i];
      const s = phaseSin[i];
      const dstOff = atomMap[i].parentAtomIdx * 3;
      const srcOff = i * 3;

      for (let alpha = 0; alpha < 3; alpha++) {
        const re = evRe[srcOff + alpha];
        const im = evIm[srcOff + alpha];
        // Complex product: (c + i s) * (re + i im) = (c*re - s*im) + i*(c*im + s*re)
        projReal[dstOff + alpha] += c * re - s * im;
        projImag[dstOff + alpha] += c * im + s * re;
      }
    }

    let modeWeight = 0;
    for (let d = 0; d < projReal.length; d++) {
      modeWeight += projReal[d] * projReal[d] + projImag[d] * projImag[d];
    }
    weights[nu] = modeWeight * invMultiplicity;
  }

  return weights;
}

/**
 * Unfold a supercell Phonopy dynamical-matrix model onto a parent Brillouin-zone
 * wave-vector path `{ qPoints, kpathMeta, baseStructure }`.
 *
 * Returns a complete viewer-compatible dispersion object containing:
 *   - `bands[qi][nu]`     : supercell phonon energies (meV) at `q_0(Q_qi)`
 *   - `weights[qi][nu]`   : unfolded parent-BZ spectral weights `W_{Q_qi, ν}` in `[0, 1]`
 *   - `eigvecs[qi][nu]`   : supercell eigenvectors at `q_0(Q_qi)` (so clicking any
 *                           unfolded branch in Step 2 animates its 3D atomic motion!)
 *   - `qPointsSuper[qi]`  : wrapped supercell wave vectors `q_0` (for 3D animation phase)
 *   - `qPoints`           : parent reciprocal-path wave vectors `Q_qi`
 *   - `kpathMeta`         : parent high-symmetry tick labels & segment sizes
 *   - `parentStructure`   : parent `baseStructure` (for exact parent reciprocal-axis scaling)
 *   - `relation`          : `{ P, multiplicity, rmsDispAng, ... }`
 */
export function unfoldModelOntoParentPath(superModel, parentPathSource, options = {}) {
  const payload = superModel?.dynamicalMatrixPayload;
  if (!payload || !superModel?.baseStructure || !parentPathSource?.qPoints?.length) return null;

  const parentCell = extractCellFromBaseStructure(parentPathSource.baseStructure);
  const superCell = extractCellFromBaseStructure(superModel.baseStructure);
  const relation = options.relation || findSupercellRelation(parentCell, superCell, options);
  if (!relation) return null;

  const { P } = relation;
  const parentQPoints = parentPathSource.qPoints;
  const nQ = parentQPoints.length;

  const bands = new Array(nQ);
  const weights = new Array(nQ);
  const eigvecs = new Array(nQ);
  const qPointsSuper = new Array(nQ);

  for (let qi = 0; qi < nQ; qi++) {
    const Q_PC = parentQPoints[qi];
    // Map parent fractional wave vector Q_PC -> unwrapped & wrapped supercell wave vectors
    const qSuperUnwrapped = matVec3(P, Q_PC);
    const qSuperWrapped = [
      wrapHalf(qSuperUnwrapped[0]),
      wrapHalf(qSuperUnwrapped[1]),
      wrapHalf(qSuperUnwrapped[2]),
    ];

    // Non-analytic LO-TO direction when q_0 lands at Γ: transform path travel direction by P
    let qDirSuper = null;
    if (Math.hypot(qSuperWrapped[0], qSuperWrapped[1], qSuperWrapped[2]) < 1e-6) {
      const nb = (qi + 1 < nQ) ? parentQPoints[qi + 1] : parentQPoints[qi - 1];
      if (nb) {
        const dParent = (qi + 1 < nQ)
          ? [nb[0] - Q_PC[0], nb[1] - Q_PC[1], nb[2] - Q_PC[2]]
          : [Q_PC[0] - nb[0], Q_PC[1] - nb[1], Q_PC[2] - nb[2]];
        const dSuper = matVec3(P, dParent);
        const norm = Math.hypot(dSuper[0], dSuper[1], dSuper[2]);
        if (norm > 1e-9) qDirSuper = [dSuper[0] / norm, dSuper[1] / norm, dSuper[2] / norm];
      }
    }

    const { energiesMev, eigvecs: evRow } = solveDynamicalMatrixAtQ(payload, qSuperWrapped, qDirSuper);
    const wRow = computeModeSpectralWeights(
      evRow,
      qSuperUnwrapped,
      qSuperWrapped,
      superCell.fracPositions,
      relation
    );

    bands[qi] = energiesMev;
    weights[qi] = wRow;
    eigvecs[qi] = evRow;
    qPointsSuper[qi] = qSuperWrapped;
  }

  return {
    bands,
    weights,
    eigvecs,
    qPoints: parentQPoints,
    qPointsSuper,
    kpathMeta: parentPathSource.kpathMeta,
    parentStructure: parentPathSource.baseStructure,
    superStructure: superModel.baseStructure,
    relation,
  };
}

/**
 * Evaluate the broadened 2D unfolded phonon spectral function:
 *
 *   A(Q_k, E) = sum_{ν = 1}^{3 N_SC} W_{Q_k, ν} * L_η(E - E_{Q_k, ν})
 *
 * where `L_η(ΔE) = (η / π) / (ΔE^2 + η^2)` is a normalized Lorentzian of half-width
 * `eta` (FWHM = 2η). Directly comparable to single-crystal inelastic neutron or
 * X-ray scattering intensity maps along high-symmetry reciprocal directions!
 */
export function computeSpectralFunctionGrid(bands, weights, options = {}) {
  if (!bands?.length || !weights?.length) return null;
  const nQ = bands.length;
  const nModes = bands[0]?.length || 0;

  let minE = 0, maxE = 1;
  for (let q = 0; q < nQ; q++) {
    for (let m = 0; m < nModes; m++) {
      const e = bands[q][m];
      if (!Number.isFinite(e)) continue;
      if (e < minE) minE = e;
      if (e > maxE) maxE = e;
    }
  }
  const span = Math.max(1, maxE - minE);
  const yMin = Number.isFinite(options.eMin) ? options.eMin : (minE < 0 ? minE * 1.08 : 0);
  const yMax = Number.isFinite(options.eMax) ? options.eMax : maxE * 1.06;
  const nEnergyBins = options.nEnergyBins || 220;
  const fwhm = options.fwhm || Math.max(0.25, span * 0.018);
  const eta = 0.5 * fwhm;
  const eta2 = eta * eta;
  const invPiEta = eta / Math.PI;

  const intensity = new Float32Array(nQ * nEnergyBins);
  const dE = (yMax - yMin) / Math.max(1, nEnergyBins - 1);
  let maxIntensity = 0;

  for (let q = 0; q < nQ; q++) {
    const eRow = bands[q];
    const wRow = weights[q];
    const colOff = q * nEnergyBins;
    for (let m = 0; m < nModes; m++) {
      const w = wRow[m];
      if (!(w > 1e-5)) continue;
      const e0 = eRow[m];
      if (!Number.isFinite(e0)) continue;
      // Only evaluate bins within ±12*eta of e0 for speed (Lorentzian tail < 0.7% beyond 12*eta)
      const binLo = Math.max(0, Math.floor((e0 - 12 * eta - yMin) / dE));
      const binHi = Math.min(nEnergyBins - 1, Math.ceil((e0 + 12 * eta - yMin) / dE));
      for (let b = binLo; b <= binHi; b++) {
        const diff = (yMin + b * dE) - e0;
        intensity[colOff + b] += w * (invPiEta / (diff * diff + eta2));
      }
    }
    for (let b = 0; b < nEnergyBins; b++) {
      if (intensity[colOff + b] > maxIntensity) maxIntensity = intensity[colOff + b];
    }
  }

  return {
    intensity,
    nQ,
    nEnergyBins,
    yMin,
    yMax,
    fwhm,
    maxIntensity,
  };
}

/**
 * Fallback helper when ONLY a distorted supercell `_mlip_dm.json` is loaded (with
 * no separate `Parent` candidate in memory): attempts to detect a smaller, higher-
 * symmetry parent primitive cell directly from the supercell's own average geometry
 * at a looser tolerance (`tolFrac`, default ~0.06 fractional / ~0.35 Å), and builds
 * its Setyawan–Curtarolo high-symmetry k-path (`buildBZModel`).
 */
export function inferIntrinsicParentPath(superModel, options = {}) {
  const superCell = extractCellFromBaseStructure(superModel?.baseStructure);
  if (!superCell || superCell.elements.length < 2) return null;

  const A_SC = superCell.lattice;
  const atomsCart = superCell.fracPositions.map((f, i) => ({
    pos: vecMat3(f, A_SC),
    element: superCell.elements[i],
  }));

  // Test common subcell reductions along axes (1×1×2, 2×1×1, 1×2×1, 2×2×1, 2×2×2)
  // plus Bravais centering reductions of A_SC itself.
  const candidateTransforms = [
    [[0.5, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[1, 0, 0], [0, 0.5, 0], [0, 0, 1]],
    [[1, 0, 0], [0, 1, 0], [0, 0, 0.5]],
    [[0.5, 0.5, 0], [-0.5, 0.5, 0], [0, 0, 1]],
    [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]],
    [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
    // Rhombohedral FCC primitive inside 45°-rotated tetragonal (e.g. P-42₁m -> F-43m primitive!)
    [[0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]], // wait: in 45°-rotated TET: a_tet = (a+b)/2, b_tet = (-a+b)/2, c_tet = c =>
    [[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5]],
    [[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [0, -1, 0]],
  ];

  let bestParent = null;
  for (const Tsub of candidateTransforms) {
    const A_cand = matMul3(Tsub, A_SC);
    const relabeled = relabelAtoms(atomsCart, A_cand, { tol: options.tolFrac ?? 0.08 });
    if (relabeled.error || relabeled.issues.length > 0) continue;
    if (relabeled.nCells < 2 || relabeled.nBasis >= superCell.elements.length) continue;
    if (relabeled.maxResidual > (options.maxResidualAng ?? 0.45)) continue;

    if (!bestParent || relabeled.nBasis < bestParent.relabeled.nBasis ||
       (relabeled.nBasis === bestParent.relabeled.nBasis && relabeled.maxResidual < bestParent.relabeled.maxResidual)) {
      bestParent = { A_cand, relabeled };
    }
  }

  if (!bestParent) return null;

  const { A_cand, relabeled } = bestParent;
  const bravais = analyzeBravais(A_cand);
  // Build standardized primitive BZ path for A_cand
  const bzModel = buildBZModel(bravais);
  if (!bzModel?.path?.length) return null;

  const nPtsPerSeg = options.nPointsPerSegment || 35;
  const qPoints = [];
  const hsymIndex = {};
  const segSizes = [];

  bzModel.path.forEach(([startKey, endKey], segIdx) => {
    const p0 = bzModel.points[startKey].frac;
    const p1 = bzModel.points[endKey].frac;
    const isCont = segIdx > 0 && bzModel.path[segIdx - 1][1] === startKey;
    const startStep = isCont ? 1 : 0;
    if (!isCont) hsymIndex[qPoints.length] = bzModel.points[startKey].display;
    let count = 0;
    for (let s = startStep; s < nPtsPerSeg; s++) {
      const t = s / (nPtsPerSeg - 1);
      qPoints.push([
        p0[0] + (p1[0] - p0[0]) * t,
        p0[1] + (p1[1] - p0[1]) * t,
        p0[2] + (p1[2] - p0[2]) * t,
      ]);
      count++;
    }
    segSizes.push(count);
    hsymIndex[qPoints.length - 1] = bzModel.points[endKey].display;
  });

  // Express parent basis in A_prim (= bravais.A_prim) so qPoints (in B_prim) match!
  const A_prim = bravais.A_prim;
  const relabelPrim = relabelAtoms(atomsCart, A_prim, { tol: options.tolFrac ?? 0.08 });
  const chosenA = (!relabelPrim.error && relabelPrim.issues.length === 0) ? A_prim : A_cand;
  const chosenBasis = (!relabelPrim.error && relabelPrim.issues.length === 0) ? relabelPrim.basis : relabeled.basis;

  const atomDic = {};
  chosenBasis.forEach((b, idx) => { (atomDic[b.element] || (atomDic[b.element] = [])).push(idx); });
  const baseStructure = {
    atomDic,
    dim: [1, 1, 1],
    v1: chosenA[0],
    v2: chosenA[1],
    v3: chosenA[2],
    uniqueRN: chosenBasis.map((_, i) => i),
    atomType: chosenBasis.map((_, i) => i),
    hsym_xyz: Float64Array.from(chosenBasis.flatMap(b => b.frac)),
    cellIdx: new Float64Array(chosenBasis.length * 3),
    bandRecip: reciprocalLattice(chosenA),
  };

  return {
    label: `Inferred Parent (${bravais.code}, ${chosenBasis.length} sites)`,
    qPoints,
    kpathMeta: { segSizes, hsymIndex },
    baseStructure,
    distances: pathDistances(qPoints, baseStructure.bandRecip, segSizes),
  };
}
