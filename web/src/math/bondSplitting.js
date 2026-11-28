// web/src/math/bondSplitting.js
//
// Crystallographic Coordination & Bond-Length Splitting Analysis
// ==============================================================
// In Reverse Monte Carlo (RMC) total-scattering refinements, local symmetry breaking
// (such as Jahn–Teller elongations, ferroelectric off-centering, lone-pair stereochemical
// activity, Peierls dimerization, or octahedral tilts) manifests directly in real space
// as the splitting of a degenerate parent coordination shell into distinct short/medium/long
// sub-shells — the exact feature resolved in the pair distribution function G(r).
//
// Physical Quantities Computed Here
// ---------------------------------
// Given an ensemble-aligned & subgroup-symmetrized structure `aligned` (with cell matrix `A`
// and `symmetrizedBasis` carrying each site's `frac` and `dispCart = r_local - r_parent` [Å]):
//
//   1. Exact Undistorted Reference Coordinates in the Subgroup Cell:
//      Because `r_local,s = r_parent,s + dispCart_s` holds identically for both Γ (`1×1×1`)
//      and zone-boundary (`1×1×2`, `2×2×1`, `2×2×2`) supercells, subtracting `dispCart_s`
//      recovers the exact parent reference coordinates `r_parent,s` inside `A`!
//
//   2. First-Shell Bond Splitting (`parentShells -> localSubshells`):
//      For each unique central-site orbit `(elCenter, elNeighbor)`, we locate all first-shell
//      periodic neighbors `(j, imageShift)` within `1.25 * d_min` (capped at `3.8 Å`), evaluate
//      both `d_parent` and `d_local` on the *identical* topological bond list, and compute:
//        * Peak-to-peak splitting:   Δd = d_max - d_min  [Å]
//        * Mean bond length change:  δ<d> = <d_local> - <d_parent>  [Å]
//        * Baur distortion index (Baur, Acta Cryst. B 30, 1195, 1974):
//              D_Baur = (1 / (N_coord * <d_local>)) * sum_{k=1}^{N_coord} |d_k - <d_local>|
//
//   3. Exaggerated 3D Visualization Structure (`buildExaggeratedStructure`):
//      Constructs a `CrystalViewer`-ready structure with atomic positions
//          r_vis,s(η) = r_parent,s + η * dispCart_s
//      and Cartesian arrow vectors `staticVectors` so subtle `0.03–0.10 Å` distortions
//      can be inspected visually at `×1`, `×5`, `×10`, or `×25` magnification.

import { COVALENT_R } from '../constants.js';

const wrap01 = (x) => ((x % 1) + 1) % 1;

function invert3x3(m) {
  const det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  if (Math.abs(det) < 1e-14) return null;
  const id = 1 / det;
  return [
    [(m[1][1] * m[2][2] - m[1][2] * m[2][1]) * id, (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * id, (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * id],
    [(m[1][2] * m[2][0] - m[1][0] * m[2][2]) * id, (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * id, (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * id],
    [(m[1][0] * m[2][1] - m[1][1] * m[2][0]) * id, (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * id, (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * id],
  ];
}

function fracToCart(f, A) {
  return [
    f[0] * A[0][0] + f[1] * A[1][0] + f[2] * A[2][0],
    f[0] * A[0][1] + f[1] * A[1][1] + f[2] * A[2][1],
    f[0] * A[0][2] + f[1] * A[1][2] + f[2] * A[2][2],
  ];
}

function cartToFrac(r, Ainv) {
  // Since r = f * A (row-vector convention), f = r * A^(-1)
  return [
    r[0] * Ainv[0][0] + r[1] * Ainv[1][0] + r[2] * Ainv[2][0],
    r[0] * Ainv[0][1] + r[1] * Ainv[1][1] + r[2] * Ainv[2][1],
    r[0] * Ainv[0][2] + r[1] * Ainv[1][2] + r[2] * Ainv[2][2],
  ];
}

/**
 * Cluster sorted distances `vals` into degenerate multiplicity groups within tolerance `tol` [Å].
 * Returns `[{ distance, count }, ...]`.
 */
function groupDistances(vals, tol = 0.0045) {
  if (!vals.length) return [];
  const sorted = [...vals].sort((a, b) => a - b);
  const groups = [];
  let curSum = sorted[0];
  let curCnt = 1;
  let curAnchor = sorted[0];
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i] - curAnchor <= tol) {
      curSum += sorted[i];
      curCnt += 1;
    } else {
      groups.push({ distance: curSum / curCnt, count: curCnt });
      curSum = sorted[i];
      curCnt = 1;
      curAnchor = sorted[i];
    }
  }
  groups.push({ distance: curSum / curCnt, count: curCnt });
  return groups;
}

export function formatShellGroups(groups, digits = 3) {
  if (!groups?.length) return '—';
  return groups.map(g => `${g.count}×${g.distance.toFixed(digits)}`).join(' + ');
}

/**
 * Compute Parent-vs-Local first-shell coordination bond splitting and site off-centering
 * from an `aligned` subgroup result (`scResult.aligned`).
 */
export function computeBondSplitting(aligned, opts = {}) {
  if (!aligned?.A || !aligned?.symmetrizedBasis?.length) {
    return { bondRows: [], siteRows: [], maxSplittingAng: 0 };
  }
  const { maxBondCutoff = 3.85, shellRatio = 1.24 } = opts;
  const A = aligned.A;
  const sites = aligned.symmetrizedBasis;
  const nSites = sites.length;

  // Reconstruct continuous Cartesian positions for both local and undistorted parent sites
  const rLocal = new Array(nSites);
  const rParent = new Array(nSites);
  for (let i = 0; i < nSites; i++) {
    const rl = fracToCart(sites[i].frac, A);
    const u = sites[i].dispCart || [0, 0, 0];
    rLocal[i] = rl;
    rParent[i] = [rl[0] - u[0], rl[1] - u[1], rl[2] - u[2]];
  }

  // Determine periodic image search range so small cells (< 4 Å) still capture full 1st shell
  const cellLens = A.map(v => Math.hypot(v[0], v[1], v[2]));
  const nImg = cellLens.map(L => Math.min(3, Math.max(1, Math.ceil(maxBondCutoff / Math.max(L, 1.0)))));

  const uniqueElements = [...new Set(sites.map(s => s.el))];
  const isMonoElement = uniqueElements.length === 1;

  // Group per-site shifts by (element, rounded |u|) so equivalent Wyckoff orbits merge cleanly
  const siteGroups = new Map();
  for (let i = 0; i < nSites; i++) {
    const s = sites[i];
    const mag = s.dispNorm ?? Math.hypot(...(s.dispCart || [0, 0, 0]));
    const u = s.dispCart || [0, 0, 0];
    const key = `${s.el}:${mag.toFixed(3)}`;
    const prev = siteGroups.get(key);
    if (!prev) {
      siteGroups.set(key, {
        el: s.el,
        count: 1,
        shiftAng: mag,
        seAng: s.dispSE || 0,
        exampleVec: u,
      });
    } else {
      prev.count += 1;
    }
  }
  const siteRows = [...siteGroups.values()].sort((a, b) => b.shiftAng - a.shiftAng);

  // Build coordination environments per central site i and neighbor element elB
  const envMap = new Map();
  for (let i = 0; i < nSites; i++) {
    const elA = sites[i].el;
    const byNeighborEl = new Map();

    for (let j = 0; j < nSites; j++) {
      const elB = sites[j].el;
      if (!isMonoElement && elA === elB) continue;

      for (let ix = -nImg[0]; ix <= nImg[0]; ix++) {
        for (let iy = -nImg[1]; iy <= nImg[1]; iy++) {
          for (let iz = -nImg[2]; iz <= nImg[2]; iz++) {
            if (i === j && ix === 0 && iy === 0 && iz === 0) continue;
            const shiftCart = fracToCart([ix, iy, iz], A);

            const dxP = rParent[j][0] + shiftCart[0] - rParent[i][0];
            const dyP = rParent[j][1] + shiftCart[1] - rParent[i][1];
            const dzP = rParent[j][2] + shiftCart[2] - rParent[i][2];
            const dP = Math.hypot(dxP, dyP, dzP);
            if (dP < 0.6 || dP > maxBondCutoff) continue;

            const dxL = rLocal[j][0] + shiftCart[0] - rLocal[i][0];
            const dyL = rLocal[j][1] + shiftCart[1] - rLocal[i][1];
            const dzL = rLocal[j][2] + shiftCart[2] - rLocal[i][2];
            const dL = Math.hypot(dxL, dyL, dzL);

            let arr = byNeighborEl.get(elB);
            if (!arr) { arr = []; byNeighborEl.set(elB, arr); }
            arr.push({ dP, dL });
          }
        }
      }
    }

    for (const [elB, pairs] of byNeighborEl.entries()) {
      if (!pairs.length) continue;
      // Avoid duplicate B–A mirror rows when A–B is already reported (keep cation/heavier or covalent-plausible first)
      if (elA > elB) {
        const covCut = ((COVALENT_R[elA] || 1.2) + (COVALENT_R[elB] || 1.2)) * 1.45;
        const dMinCheck = Math.min(...pairs.map(p => p.dP));
        if (dMinCheck > covCut) continue;
      }

      const dMinP = Math.min(...pairs.map(p => p.dP));
      const covSum = (COVALENT_R[elA] || 1.2) + (COVALENT_R[elB] || 1.2);
      if (dMinP > Math.max(3.4, covSum * 1.48)) continue;

      const firstShell = pairs.filter(p => p.dP <= dMinP * shellRatio);
      if (!firstShell.length) continue;

      const pGroups = groupDistances(firstShell.map(p => p.dP), 0.005);
      const lGroups = groupDistances(firstShell.map(p => p.dL), 0.0045);

      const dLocals = firstShell.map(p => p.dL);
      const dParents = firstShell.map(p => p.dP);
      const meanLocal = dLocals.reduce((a, b) => a + b, 0) / dLocals.length;
      const meanParent = dParents.reduce((a, b) => a + b, 0) / dParents.length;
      const dMinL = Math.min(...dLocals);
      const dMaxL = Math.max(...dLocals);
      const splittingAng = Math.max(0, dMaxL - dMinL);
      const baurIndex = meanLocal > 1e-6
        ? dLocals.reduce((s, d) => s + Math.abs(d - meanLocal), 0) / (dLocals.length * meanLocal)
        : 0;

      // Deduplicate symmetry-equivalent central sites sharing the same coordination signature
      const pairLabel = elA <= elB ? `${elA}–${elB}` : `${elA}–${elB}`;
      const sigKey = `${elA}-${elB}:${formatShellGroups(pGroups, 3)}->${formatShellGroups(lGroups, 3)}`;
      // Also skip exact reverse B-A row if an identical coordination count & distance set exists
      const canonicalPair = [elA, elB].sort().join('–');
      const existingReverse = [...envMap.values()].find(
        r => r.canonicalPair === canonicalPair
          && r.coordNum === firstShell.length
          && Math.abs(r.splittingAng - splittingAng) < 1e-3
          && Math.abs(r.meanLocal - meanLocal) < 1e-3
      );
      if (existingReverse) {
        if (existingReverse.centerEl === elA) existingReverse.siteMultiplicity += 1;
        continue;
      }

      const prev = envMap.get(sigKey);
      if (prev) {
        prev.siteMultiplicity += 1;
      } else {
        envMap.set(sigKey, {
          pairLabel,
          canonicalPair,
          centerEl: elA,
          neighborEl: elB,
          siteMultiplicity: 1,
          coordNum: firstShell.length,
          parentGroups: pGroups,
          localGroups: lGroups,
          parentSummary: formatShellGroups(pGroups, 3),
          localSummary: formatShellGroups(lGroups, 3),
          meanParent,
          meanLocal,
          splittingAng,
          baurIndex,
        });
      }
    }
  }

  const bondRows = [...envMap.values()].sort((a, b) => b.splittingAng - a.splittingAng);
  const maxSplittingAng = bondRows.length ? bondRows[0].splittingAng : 0;
  return { bondRows, siteRows, maxSplittingAng };
}

/**
 * Build a `CrystalViewer` baseStructure + Cartesian `staticVectors` for an `aligned`
 * local subgroup at visual exaggeration factor `exaggeration` (e.g. 1, 5, 10, 25).
 *
 * At `exaggeration = 1`, atomic positions equal the exact physical local coordinates.
 * At `exaggeration = η > 1`, displacements from the ideal parent Wyckoff positions are
 * magnified `r_vis(η) = r_parent + η * dispCart` so subtle `0.04 Å` tilts and polar
 * off-centerings become immediately legible in 3D!
 */
export function buildExaggeratedAlignedStructure(aligned, exaggeration = 1) {
  if (!aligned?.A || !aligned?.symmetrizedBasis?.length) return null;
  const A = aligned.A;
  const Ainv = invert3x3(A);
  const sites = aligned.symmetrizedBasis;
  const nSites = sites.length;

  const dynAtomDic = {};
  const uniqueRN = new Array(nSites);
  const hsym_xyz = new Float64Array(nSites * 3);
  const staticVectors = new Float64Array(nSites * 3);

  for (let i = 0; i < nSites; i++) {
    const s = sites[i];
    const rn = i + 1; // Unique row per site in the aligned block cell so each site gets its own arrow!
    uniqueRN[i] = rn;
    if (!dynAtomDic[s.el]) dynAtomDic[s.el] = [];
    dynAtomDic[s.el].push(rn);

    const u = s.dispCart || [0, 0, 0];
    staticVectors[i * 3] = u[0];
    staticVectors[i * 3 + 1] = u[1];
    staticVectors[i * 3 + 2] = u[2];

    if (Ainv && Math.abs(exaggeration - 1) > 1e-6) {
      const rLoc = fracToCart(s.frac, A);
      const rPar = [rLoc[0] - u[0], rLoc[1] - u[1], rLoc[2] - u[2]];
      const rExag = [
        rPar[0] + exaggeration * u[0],
        rPar[1] + exaggeration * u[1],
        rPar[2] + exaggeration * u[2],
      ];
      const fExag = cartToFrac(rExag, Ainv);
      hsym_xyz[i * 3] = wrap01(fExag[0]);
      hsym_xyz[i * 3 + 1] = wrap01(fExag[1]);
      hsym_xyz[i * 3 + 2] = wrap01(fExag[2]);
    } else {
      hsym_xyz[i * 3] = s.frac[0];
      hsym_xyz[i * 3 + 1] = s.frac[1];
      hsym_xyz[i * 3 + 2] = s.frac[2];
    }
  }

  return {
    baseStructure: {
      v1: A[0],
      v2: A[1],
      v3: A[2],
      dim: [1, 1, 1],
      atomDic: dynAtomDic,
      uniqueRN,
      atomType: uniqueRN.slice(),
      hsym_xyz,
      cellIdx: new Float64Array(nSites * 3),
    },
    staticVectors,
  };
}
