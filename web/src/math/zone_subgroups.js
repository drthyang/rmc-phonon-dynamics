// web/src/math/zone_subgroups.js
//
// Zone-boundary multi-cell subgroup discovery, coset domain unmixing, and
// non-symmorphic space-group classification for commensurate supercell blocks:
//   - X-star (1D cell doubling: 1×1×2, 1×2×1, 2×1×1)
//   - M-star (2D cell doubling: 2×2×1, 2×1×2, 1×2×2)
//   - R-star (3D cell doubling: 2×2×2)
// as well as centering-breaking primitive-BZ zone-boundary subgroups inside
// centered (F, I, C, A, B) 1×1×1 conventional cells.
//
// Key mathematical guarantee (zero Γ-leakage):
//   Every multi-cell block vector U ∈ R^(V_blk * D) is projected onto its pure
//   non-Γ zone-boundary complement U_ZB = U - U_Γ (subtracting the sub-cell mean
//   per basis site) for R², Excess ΔF, and neighbor-held-out Spatial CV ΔF scoring.
//   Thus a pure 1×1×1 (Γ) ferrodistortive crystal has U_ZB = thermal noise only,
//   giving Excess ΔF ≈ 0 and CV ΔF ≈ 1.0 with zero false-positive leakage into
//   1×1×2 / 2×2×1 / 2×2×2 subgroups, while full block domain realignment recovers
//   both the primary zone-boundary mode and any coupled secondary Γ strain/shift.

import { findSpaceGroupOps, classifyOperations, classifyRotation } from './symmetry.js';
import { applyGammaCell, removeCellDrift } from './supercell_symmetry.js';

const wrap01 = (x) => ((x % 1) + 1) % 1;
const wrapHalf = (x) => x - Math.round(x);
const mod = (a, m) => ((a % m) + m) % m;

function inv3(A) {
  const [[a, b, c], [d, e, f], [g, h, i]] = A;
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  const inv = 1 / det;
  return [
    [(e * i - f * h) * inv, (c * h - b * i) * inv, (b * f - c * e) * inv],
    [(f * g - d * i) * inv, (a * i - c * g) * inv, (c * d - a * f) * inv],
    [(d * h - e * g) * inv, (b * g - a * h) * inv, (a * e - b * d) * inv],
  ];
}

function mat3Mul(A, B) {
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      C[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j];
  return C;
}

function isIdentR(R) {
  return R[0][0] === 1 && R[1][1] === 1 && R[2][2] === 1 &&
         R[0][1] === 0 && R[0][2] === 0 && R[1][0] === 0 &&
         R[1][2] === 0 && R[2][0] === 0 && R[2][1] === 0;
}

function opKey48(R, t) {
  const r = R.flat().join(',');
  const q = t.map(x => Math.round(wrap01(x) * 48) % 48).join(',');
  return `${r}|${q}`;
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeGaussian(rng) {
  let spare = null;
  return () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    let u = 0, v = 0;
    while (u <= 1e-12) u = rng();
    while (v <= 1e-12) v = rng();
    const mag = Math.sqrt(-2 * Math.log(u));
    spare = mag * Math.sin(2 * Math.PI * v);
    return mag * Math.cos(2 * Math.PI * v);
  };
}

/**
 * Refine symmorphic H-M classification from `classifyOperations` to distinguish
 * key non-symmorphic screw/glide space groups (e.g. P-42₁m #113 vs P-42m #111,
 * P4/mbm #127, P4/nmm #129, I4/mcm #140, Pnma #62, Imma #74, Fm-3c #226, R-3c #167).
 */
export function refineSpaceGroupSymbol(ops, cls) {
  const { centering, pointGroup } = cls;
  let symbol = cls.spaceGroup;
  let number = cls.spaceGroupNumber;

  // Intrinsic screw/glide component of {R | t} is (R + I) t / 2 for 2-fold/mirror!
  // Why? Under origin shift x -> x + o, t -> t + (R - I) o. Since (R + I)(R - I) = R^2 - I = 0
  // for any involution (R^2 = I, i.e. 2-fold rotations, mirrors, inversion), the vector
  // s(R, t) = 0.5 * (R + I) t (mod 1) is STRICTLY ORIGIN-INVARIANT!
  const invScrew = (R, t) => [
    wrap01(0.5 * ((R[0][0] + 1) * t[0] + R[0][1] * t[1] + R[0][2] * t[2])),
    wrap01(0.5 * (R[1][0] * t[0] + (R[1][1] + 1) * t[1] + R[1][2] * t[2])),
    wrap01(0.5 * (R[2][0] * t[0] + R[2][1] * t[1] + (R[2][2] + 1) * t[2])),
  ];
  const isNonZeroVec = (v) => Math.abs(wrapHalf(v[0])) > 0.08 || Math.abs(wrapHalf(v[1])) > 0.08 || Math.abs(wrapHalf(v[2])) > 0.08;

  // Collect involutions (2-fold axes and mirrors)
  const twoFolds = [];
  const mirrors = [];
  for (const op of ops) {
    const kind = classifyRotation(op.R);
    if (kind === '2') twoFolds.push(op);
    else if (kind === 'm') mirrors.push(op);
  }

  // 1. Tetragonal -42m family: distinguish P-42m (#111), P-42c (#112), P-42₁m (#113), P-42₁c (#114),
  //    P-4m2 (#115), P-4c2 (#116), P-4b2 (#117), P-4n2 (#118), I-42m (#121), I-42d (#122)
  if (pointGroup === '-42m') {
    const minus4 = ops.find(o => classifyRotation(o.R) === '-4');
    let cAxis = 2;
    if (minus4) {
      const R2 = mat3Mul(minus4.R, minus4.R);
      if (R2[0][0] === 1) cAxis = 0;
      else if (R2[1][1] === 1) cAxis = 1;
      else cAxis = 2;
    }
    if (centering === 'P') {
      const perp2 = twoFolds.filter(o => o.R[0][0] === 1 || o.R[1][1] === 1 || o.R[2][2] === 1);
      const xyAxis2 = perp2.filter(o => o.R[cAxis][cAxis] === -1);
      if (xyAxis2.length > 0) {
        // Secondary <100> has 2 or 2₁; tertiary <110> has m or c-glide (note: in P-42₁m, one <110> mirror
        // acquires an in-plane 1/2(a+b) shift from 2₁, whereas P-42c / P-42₁c have c-axis glide s[cAxis] = 1/2!)
        const hasScrew21 = xyAxis2.some(o => isNonZeroVec(invScrew(o.R, o.t)));
        const hasCGlide = mirrors.length > 0 && mirrors.every(o => Math.abs(wrapHalf(invScrew(o.R, o.t)[cAxis])) > 0.08);
        if (hasScrew21 && !hasCGlide) { symbol = 'P-42₁m'; number = 113; }
        else if (hasScrew21 && hasCGlide) { symbol = 'P-42₁c'; number = 114; }
        else if (!hasScrew21 && hasCGlide) { symbol = 'P-42c'; number = 112; }
        else { symbol = 'P-42m'; number = 111; }
      } else {
        // Secondary <100> has mirrors -> P-4m2 (#115) / P-4c2 (#116) / P-4b2 (#117) / P-4n2 (#118)
        const hasCGlide = mirrors.some(o => Math.abs(wrapHalf(invScrew(o.R, o.t)[cAxis])) > 0.08);
        const hasInPlane = mirrors.some(o => {
          const s = invScrew(o.R, o.t);
          return [0, 1, 2].some(ax => ax !== cAxis && Math.abs(wrapHalf(s[ax])) > 0.08);
        });
        if (hasCGlide && hasInPlane) { symbol = 'P-4n2'; number = 118; }
        else if (hasCGlide) { symbol = 'P-4c2'; number = 116; }
        else if (hasInPlane) { symbol = 'P-4b2'; number = 117; }
        else { symbol = 'P-4m2'; number = 115; }
      }
    } else if (centering === 'I') {
      const hasGlide = mirrors.every(o => {
        const s = invScrew(o.R, o.t);
        const sI = [wrapHalf(s[0] - 0.5), wrapHalf(s[1] - 0.5), wrapHalf(s[2] - 0.5)];
        return isNonZeroVec(s) && (Math.abs(sI[0]) > 0.08 || Math.abs(sI[1]) > 0.08 || Math.abs(sI[2]) > 0.08);
      });
      symbol = hasGlide ? 'I-42d' : 'I-42m';
      number = hasGlide ? 122 : 121;
    } else if (centering === 'C' || centering === 'A' || centering === 'B') {
      // Base-centered perpendicular to cAxis = 45°-rotated primitive P tetragonal setting!
      const hasMirrorCGlide = mirrors.some(o => Math.abs(wrapHalf(invScrew(o.R, o.t)[cAxis])) > 0.08);
      const hasMirrorQuarter = mirrors.some(o => {
        const s = invScrew(o.R, o.t);
        return [0, 1, 2].some(ax => ax !== cAxis && Math.abs(wrapHalf(2 * s[ax])) > 0.15);
      });
      // Check if mirrors lie on parent axial planes (diagonal in P -> P-42m/P-42₁m) vs diagonal planes (axial in P -> P-4m2/P-4n2)
      const mirrorsAreParentAxial = mirrors.some(o => o.R[0][0] === -1 && o.R[1][1] === 1 && o.R[2][2] === 1 ||
                                                      o.R[0][0] === 1 && o.R[1][1] === -1 && o.R[2][2] === 1 ||
                                                      o.R[0][0] === 1 && o.R[1][1] === 1 && o.R[2][2] === -1);
      if (!mirrorsAreParentAxial) {
        if (hasMirrorCGlide && hasMirrorQuarter) { symbol = 'P-4n2'; number = 118; }
        else if (hasMirrorCGlide) { symbol = 'P-4c2'; number = 116; }
        else if (hasMirrorQuarter) { symbol = 'P-4b2'; number = 117; }
        else { symbol = 'P-4m2'; number = 115; }
      } else {
        const has21Quarter = twoFolds.some(o => {
          if (o.R[cAxis][cAxis] !== -1) return false;
          const s = invScrew(o.R, o.t);
          return [0, 1, 2].some(ax => ax !== cAxis && Math.abs(wrapHalf(2 * s[ax])) > 0.15);
        });
        if (has21Quarter && !hasMirrorCGlide) { symbol = 'P-42₁m'; number = 113; }
        else if (has21Quarter && hasMirrorCGlide) { symbol = 'P-42₁c'; number = 114; }
        else if (!has21Quarter && hasMirrorCGlide) { symbol = 'P-42c'; number = 112; }
        else { symbol = 'P-42m'; number = 111; }
      }
    }
  }

  // 2. Tetragonal 4/mmm family (e.g. P4/mbm #127, P4/nmm #129, I4/mcm #140)
  else if (pointGroup === '4/mmm') {
    if (centering === 'P') {
      const nGlides = mirrors.filter(o => isNonZeroVec(invScrew(o.R, o.t))).length;
      if (nGlides === 2) { symbol = 'P4/mbm'; number = 127; }
      else if (nGlides === 3) { symbol = 'P4/nmm'; number = 129; }
      else if (nGlides >= 4) { symbol = 'P4/mcc'; number = 124; }
    } else if (centering === 'C' || centering === 'A' || centering === 'B') {
      // 2x2x1 C-centered 4/mmm is standard P4/mbm (#127) when axial mirrors glide, else P4/mmm (#123)
      const cenVec = centering === 'C' ? [0.5, 0.5, 0] : centering === 'A' ? [0, 0.5, 0.5] : [0.5, 0, 0.5];
      const nonCGlides = mirrors.filter(o => {
        const s = invScrew(o.R, o.t);
        const sC = [wrapHalf(s[0] - cenVec[0]), wrapHalf(s[1] - cenVec[1]), wrapHalf(s[2] - cenVec[2])];
        return isNonZeroVec(s) && (Math.abs(sC[0]) > 0.08 || Math.abs(sC[1]) > 0.08 || Math.abs(sC[2]) > 0.08);
      }).length;
      if (nonCGlides > 0) { symbol = 'P4/mbm'; number = 127; }
      else { symbol = 'P4/mmm'; number = 123; }
    } else if (centering === 'F' || centering === 'I') {
      // 2x2x2 F4/mmm = standard I4/mcm (#140, a⁰a⁰c⁻ octahedral tilt!) vs I4/mmm (#139)
      const cenVecs = centering === 'F'
        ? [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
        : [[0, 0, 0], [0.5, 0.5, 0.5]];
      const hasTrueGlide = mirrors.some(o => {
        const s = invScrew(o.R, o.t);
        return !cenVecs.some(c =>
          Math.abs(wrapHalf(s[0] - c[0])) < 0.08 &&
          Math.abs(wrapHalf(s[1] - c[1])) < 0.08 &&
          Math.abs(wrapHalf(s[2] - c[2])) < 0.08
        );
      });
      if (centering === 'F') {
        symbol = hasTrueGlide ? 'I4/mcm' : 'I4/mmm';
        number = hasTrueGlide ? 140 : 139;
      } else if (hasTrueGlide) {
        symbol = 'I4/mcm'; number = 140;
      }
    }
  }

  // 3. Orthorhombic mmm family (Pnma #62, Imma #74, Cmcm #63)
  else if (pointGroup === 'mmm') {
    if (centering === 'P') {
      const gl = mirrors.filter(o => isNonZeroVec(invScrew(o.R, o.t))).length;
      if (gl === 2) { symbol = 'Pnma'; number = 62; }
      else if (gl === 3) { symbol = 'Pbca'; number = 61; }
      else if (gl === 1) { symbol = 'Pmma'; number = 51; }
    } else if (centering === 'F' || centering === 'I') {
      const cenVecs = centering === 'F'
        ? [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
        : [[0, 0, 0], [0.5, 0.5, 0.5]];
      const gl = mirrors.filter(o => {
        const s = invScrew(o.R, o.t);
        return !cenVecs.some(c =>
          Math.abs(wrapHalf(s[0] - c[0])) < 0.08 &&
          Math.abs(wrapHalf(s[1] - c[1])) < 0.08 &&
          Math.abs(wrapHalf(s[2] - c[2])) < 0.08
        );
      }).length;
      if (gl > 0) { symbol = 'Imma'; number = 74; }
      else if (centering === 'F') { symbol = 'Immm'; number = 71; }
    }
  }

  // 4. Trigonal -3m on 2x2x2 F-lattice (R-3c #167 a⁻a⁻a⁻ tilt vs R-3m #166)
  else if (pointGroup === '-3m' && centering === 'F') {
    const cenVecs = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]];
    const hasCGlide = mirrors.some(o => {
      const s = invScrew(o.R, o.t);
      return !cenVecs.some(c =>
        Math.abs(wrapHalf(s[0] - c[0])) < 0.08 &&
        Math.abs(wrapHalf(s[1] - c[1])) < 0.08 &&
        Math.abs(wrapHalf(s[2] - c[2])) < 0.08
      );
    });
    symbol = hasCGlide ? 'R-3c' : 'R-3m';
    number = hasCGlide ? 167 : 166;
  }

  // 5. Cubic m-3m on 2x2x2 F-lattice (Fm-3c #226 vs Fm-3m #225)
  else if (pointGroup === 'm-3m' && centering === 'F') {
    const cenVecs = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]];
    const diagMirrors = mirrors.filter(o => o.R[0][0] === 0 || o.R[1][1] === 0 || o.R[2][2] === 0);
    const hasCGlide = diagMirrors.some(o => {
      const s = invScrew(o.R, o.t);
      return !cenVecs.some(c =>
        Math.abs(wrapHalf(s[0] - c[0])) < 0.08 &&
        Math.abs(wrapHalf(s[1] - c[1])) < 0.08 &&
        Math.abs(wrapHalf(s[2] - c[2])) < 0.08
      );
    });
    if (hasCGlide) { symbol = 'Fm-3c'; number = 226; }
  }

  return { symbol, number };
}

/**
 * Analytically lift 1×1×1 parent operations `reps111` onto a commensurate
 * supercell block `cellMult = [M1, M2, M3]` in < 0.5 ms.
 */
export function buildBlockGroup(reps111, nBasis, cellMult) {
  const [M1, M2, M3] = cellMult;
  const vol = M1 * M2 * M3;
  const M = [M1, M2, M3];
  const nBlockBasis = vol * nBasis;
  const blockReps = [];

  for (let gi = 0; gi < reps111.length; gi++) {
    const g = reps111[gi];
    const R = g.R;
    let ok = true;
    const Rblk = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < 3 && ok; i++) {
      for (let j = 0; j < 3 && ok; j++) {
        const num = R[i][j] * M[j];
        if (num % M[i] !== 0) ok = false;
        else Rblk[i][j] = num / M[i];
      }
    }
    if (!ok) continue;

    for (let dx = 0; dx < M1; dx++) {
      for (let dy = 0; dy < M2; dy++) {
        for (let dz = 0; dz < M3; dz++) {
          const perm = new Int32Array(nBlockBasis);
          // Also record per-subcell destination for O(1) reuse of already-rotated unit-cell gU!
          // Note: shift[s] can depend on basis site `s` (for non-symmorphic or centered ops),
          // so `perm` is exact at the site level!
          for (let mx = 0; mx < M1; mx++) {
            for (let my = 0; my < M2; my++) {
              for (let mz = 0; mz < M3; mz++) {
                const subIdx = (mx * M2 + my) * M3 + mz;
                for (let s = 0; s < nBasis; s++) {
                  const sh = g.shift[s];
                  const nx = mod(R[0][0] * mx + R[0][1] * my + R[0][2] * mz + sh[0] + dx, M1);
                  const ny = mod(R[1][0] * mx + R[1][1] * my + R[1][2] * mz + sh[1] + dy, M2);
                  const nz = mod(R[2][0] * mx + R[2][1] * my + R[2][2] * mz + sh[2] + dz, M3);
                  const dstSub = (nx * M2 + ny) * M3 + nz;
                  perm[subIdx * nBasis + s] = dstSub * nBasis + g.perm[s];
                }
              }
            }
          }
          const tBlk = [
            wrap01((g.t[0] + dx) / M1),
            wrap01((g.t[1] + dy) / M2),
            wrap01((g.t[2] + dz) / M3),
          ];
          const tUnit = [g.t[0] + dx, g.t[1] + dy, g.t[2] + dz];
          blockReps.push({
            R: Rblk,
            t: tBlk,
            tUnit,
            perm,
            Rcart: g.Rcart,
            g0Idx: gi,
            delta: [dx, dy, dz],
          });
        }
      }
    }
  }
  return { blockReps, nBlockBasis };
}

/**
 * Enumerate genuine isotropy subgroups of `blockReps` that break the sub-cell
 * translations along every doubled axis of `cellMult` (or break Bravais centering
 * when `cellMult = [1, 1, 1]` and `centeringBreakingOnly = true`).
 */
export function enumerateZoneIsotropySubgroups(reps111, nBasis, cellMult, kStarVec, opts = {}) {
  const { centeringBreakingOnly = false } = opts;
  const [M1, M2, M3] = cellMult;
  const vol = M1 * M2 * M3;
  const { blockReps, nBlockBasis } = buildBlockGroup(reps111, nBasis, cellMult);
  const nOps = blockReps.length;
  if (nOps === 0) return { blockReps, nBlockBasis, families: [] };

  const keyToIdx = new Map(blockReps.map((r, i) => [opKey48(r.R, r.t), i]));
  const mul = Array.from({ length: nOps }, () => new Int32Array(nOps).fill(-1));
  for (let i = 0; i < nOps; i++) {
    const a = blockReps[i];
    for (let j = 0; j < nOps; j++) {
      const b = blockReps[j];
      const R = mat3Mul(a.R, b.R);
      const t = [
        a.R[0][0] * b.t[0] + a.R[0][1] * b.t[1] + a.R[0][2] * b.t[2] + a.t[0],
        a.R[1][0] * b.t[0] + a.R[1][1] * b.t[1] + a.R[1][2] * b.t[2] + a.t[1],
        a.R[2][0] * b.t[0] + a.R[2][1] * b.t[1] + a.R[2][2] * b.t[2] + a.t[2],
      ];
      const k = keyToIdx.get(opKey48(R, t));
      if (k !== undefined) mul[i][j] = k;
    }
  }

  const idIdx = blockReps.findIndex(r => isIdentR(r.R) && Math.hypot(...r.t) < 1e-3);
  if (idIdx < 0) return { blockReps, nBlockBasis, families: [] };

  // Pure translations inside blockReps
  const allPureTrans = [];
  const subCellTrans = []; // integer unit-cell translations (g0.t == 0)
  for (let i = 0; i < nOps; i++) {
    const r = blockReps[i];
    if (!isIdentR(r.R)) continue;
    allPureTrans.push(i);
    if (Math.hypot(...reps111[r.g0Idx].t) < 1e-3) subCellTrans.push(i);
  }

  // Axis shifts that MUST be broken so the subgroup truly requires `cellMult`
  const axisShiftIdx = [];
  if (M1 > 1) axisShiftIdx.push(subCellTrans.find(i => blockReps[i].delta[0] === 1 && blockReps[i].delta[1] === 0 && blockReps[i].delta[2] === 0) ?? -1);
  if (M2 > 1) axisShiftIdx.push(subCellTrans.find(i => blockReps[i].delta[0] === 0 && blockReps[i].delta[1] === 1 && blockReps[i].delta[2] === 0) ?? -1);
  if (M3 > 1) axisShiftIdx.push(subCellTrans.find(i => blockReps[i].delta[0] === 0 && blockReps[i].delta[1] === 0 && blockReps[i].delta[2] === 1) ?? -1);

  // Wavevector kernel seeds for fast pruning on vol >= 4 (2×2×1, 2×2×2)
  const kernelSeeds = [idIdx];
  if (vol >= 4 && kStarVec) {
    for (const ti of subCellTrans) {
      const d = blockReps[ti].delta;
      const dot = kStarVec[0] * d[0] + kStarVec[1] * d[1] + kStarVec[2] * d[2];
      if (Math.abs(wrapHalf(dot)) < 1e-4) kernelSeeds.push(ti);
    }
    if (nOps > 256) {
      for (const ti of allPureTrans) {
        const u = blockReps[ti].tUnit;
        const dot = kStarVec[0] * u[0] + kStarVec[1] * u[1] + kStarVec[2] * u[2];
        if (Math.abs(wrapHalf(dot)) < 1e-4) kernelSeeds.push(ti);
      }
    }
  }

  function closeSet(seeds) {
    const inSet = new Uint8Array(nOps);
    const q = [];
    for (const s of seeds) if (s >= 0 && !inSet[s]) { inSet[s] = 1; q.push(s); }
    for (let h = 0; h < q.length; h++) {
      const a = q[h];
      for (let b = 0; b < nOps; b++) {
        if (!inSet[b]) continue;
        const ab = mul[a][b], ba = mul[b][a];
        if (ab >= 0 && !inSet[ab]) { inSet[ab] = 1; q.push(ab); }
        if (ba >= 0 && !inSet[ba]) { inSet[ba] = 1; q.push(ba); }
      }
    }
    const out = [];
    for (let i = 0; i < nOps; i++) if (inSet[i]) out.push(i);
    return out;
  }

  const base = closeSet(kernelSeeds);
  const cosetReps = [];
  const seenCoset = new Uint8Array(nOps);
  for (let g = 0; g < nOps; g++) {
    if (seenCoset[g]) continue;
    cosetReps.push(g);
    for (const b of base) if (mul[g][b] >= 0) seenCoset[mul[g][b]] = 1;
  }

  const found = new Map();
  const addSub = (arr) => {
    if (vol > 1 && axisShiftIdx.some(ai => ai >= 0 && arr.includes(ai))) return false;
    if (centeringBreakingOnly && allPureTrans.every(ti => arr.includes(ti))) return false;
    const k = arr.join(',');
    if (found.has(k)) return false;
    found.set(k, arr);
    return true;
  };
  addSub(base);

  const bfs = [base];
  const depths = [0];
  const maxDepth = nOps >= 192 ? 2 : 99;
  for (let idx = 0; idx < bfs.length; idx++) {
    const cur = bfs[idx];
    const d = depths[idx];
    if (d >= maxDepth) continue;
    const curSet = new Set(cur);
    for (const g of cosetReps) {
      if (curSet.has(g)) continue;
      const next = closeSet([...cur, g]);
      if (addSub(next)) { bfs.push(next); depths.push(d + 1); }
    }
  }

  // Ensure centrosymmetric mmm / m-3 / -3m 3-generator groups are included when maxDepth = 2
  if (nOps >= 192) {
    const invOps = [];
    for (let i = 0; i < nOps; i++) {
      if (classifyRotation(blockReps[i].R) === '-1') invOps.push(i);
    }
    const snapshot = [...found.values()];
    for (const arr of snapshot) {
      if (arr.length < base.length * 2) continue;
      for (const invOp of invOps) {
        if (!arr.includes(invOp)) addSub(closeSet([...arr, invOp]));
      }
    }
  }

  // Inverse table for conjugacy classes under G_block
  const inv = new Int32Array(nOps).fill(-1);
  for (let i = 0; i < nOps; i++) {
    for (let j = 0; j < nOps; j++) {
      if (mul[i][j] === idIdx) { inv[i] = j; break; }
    }
  }

  // Precompute per-operation trace Tr(Γ(g_i)) and drift trace Tr(R_cart(g_i)) ONCE (O(nOps * nBlockBasis))
  // so computing internal DoF = (Σ Tr(Γ(h)) - Σ Tr(R_cart(h))) / |H| for every subgroup H takes O(|H|) scalar ops!
  const opSiteTrace = new Float64Array(nOps);
  const opDriftTrace = new Float64Array(nOps);
  for (let i = 0; i < nOps; i++) {
    const { perm, Rcart } = blockReps[i];
    const trR = Rcart[0][0] + Rcart[1][1] + Rcart[2][2];
    opDriftTrace[i] = trR;
    let fixed = 0;
    for (let s = 0; s < nBlockBasis; s++) if (perm[s] === s) fixed++;
    opSiteTrace[i] = fixed * trR;
  }
  const calcInternalDoF = (opArr) => {
    let st = 0, dt = 0;
    for (let i = 0; i < opArr.length; i++) {
      const oi = opArr[i];
      st += opSiteTrace[oi];
      dt += opDriftTrace[oi];
    }
    return Math.round((st - dt) / Math.max(1, opArr.length));
  };

  // Minimum order filter keeps crystallographic subgroups while pruning noisy P1/P-1 combinatorial tails on huge cells
  const minOrd = vol === 1 ? 4 : (nOps >= 256 ? Math.max(8, Math.floor(nOps / 24)) : 4);

  // Compute total DoF and pure Zone-Boundary DoF: dofZB(H) = Tr(P_H) - Tr(P_<H, T_subcells>)
  const metaMap = new Map();
  for (const [k, ops] of found.entries()) {
    if (ops.length < minOrd) continue;
    const totDoF = calcInternalDoF(ops);
    let zbDoF = totDoF;
    if (vol > 1) {
      const hGamma = closeSet([...ops, ...subCellTrans]);
      const gammaDoF = calcInternalDoF(hGamma);
      zbDoF = Math.max(0, totDoF - gammaDoF);
    } else if (centeringBreakingOnly) {
      const hCen = closeSet([...ops, ...allPureTrans]);
      const cenDoF = calcInternalDoF(hCen);
      zbDoF = Math.max(0, totDoF - cenDoF);
    }
    metaMap.set(k, { totDoF, zbDoF });
  }

  const visited = new Set();
  const rawFamilies = [];
  for (const [k, opIndices] of found.entries()) {
    if (visited.has(k) || opIndices.length < minOrd) continue;

    const conjToAlign = new Map();
    conjToAlign.set(k, idIdx);
    const normalizerOps = [];
    for (let g = 0; g < nOps; g++) {
      const gi = inv[g];
      if (gi < 0) continue;
      const conj = opIndices.map(h => mul[mul[g][h]][gi]).sort((a, b) => a - b);
      const ck = conj.join(',');
      if (ck === k) normalizerOps.push(g);
      if (!conjToAlign.has(ck)) conjToAlign.set(ck, gi);
      visited.add(ck);
    }

    const meta = metaMap.get(k);
    if (!meta || meta.zbDoF <= 0) continue;

    // Isotropy filter: drop H if a proper supergroup K ⊃ H in `found` has the exact same totDoF!
    let isIsotropy = true;
    for (const [ok, other] of found.entries()) {
      if (other.length <= opIndices.length) continue;
      const om = metaMap.get(ok);
      if (!om || om.totDoF !== meta.totDoF) continue;
      if (opIndices.every(x => other.includes(x))) { isIsotropy = false; break; }
    }
    if (!isIsotropy) continue;

    const variants = [];
    for (const [ck, alignOpIdx] of conjToAlign.entries()) {
      const arr = found.get(ck);
      if (arr) variants.push({ opIndices: arr, ops: arr.map(i => blockReps[i]), alignOpIdx });
    }
    if (!variants.length) continue;

    // Normalizer coset reps N(H_0)/H_0 (automatically includes antiphase half-lattice shifts!)
    const h0Set = new Set(variants[0].opIndices);
    const normCosetReps = [];
    const seenNorm = new Set();
    for (const n of normalizerOps) {
      if (seenNorm.has(n)) continue;
      normCosetReps.push(n);
      for (const h of h0Set) seenNorm.add(mul[n][h]);
    }
    // Cap normalizer coset search at 16 for speed on very low-order subgroups
    const cappedNormCosets = normCosetReps.slice(0, 16);

    const cls = classifyOperations(variants[0].ops, 0.02);
    const refined = refineSpaceGroupSymbol(variants[0].ops, cls);
    const cellLabel = cellMult.join('×');

    rawFamilies.push({
      id: `${cellLabel}|${refined.symbol}|${variants[0].ops.length}|${variants.length}|${k}`,
      spaceGroup: refined.symbol,
      rawSpaceGroup: cls.spaceGroup,
      spaceGroupNumber: refined.number,
      pointGroup: cls.pointGroup,
      centering: cls.centering,
      cellMult,
      cellLabel,
      order: variants[0].ops.length,
      nVariants: variants.length,
      dof: meta.totDoF,
      zbDoF: meta.zbDoF,
      normCosetReps: cappedNormCosets,
      variants,
    });
  }

  rawFamilies.sort((a, b) => b.order - a.order);
  return { blockReps, nBlockBasis, families: rawFamilies };
}

function domainClusterStatsBlock(domGrid, B1, B2, B3) {
  const nBlocks = B1 * B2 * B3;
  const visited = new Uint8Array(nBlocks);
  const queue = new Int32Array(nBlocks);
  let nClusters = 0, sumSize = 0, sumSize2 = 0, maxCluster = 0;

  for (let start = 0; start < nBlocks; start++) {
    if (visited[start]) continue;
    const d0 = domGrid[start];
    visited[start] = 1;
    let head = 0, tail = 0;
    queue[tail++] = start;
    while (head < tail) {
      const cur = queue[head++];
      const z = cur % B3;
      const rem = (cur - z) / B3;
      const y = rem % B2;
      const x = (rem - y) / B2;
      const nbs = [
        (((x + 1) % B1) * B2 + y) * B3 + z,
        (((x + B1 - 1) % B1) * B2 + y) * B3 + z,
        (x * B2 + ((y + 1) % B2)) * B3 + z,
        (x * B2 + ((y + B2 - 1) % B2)) * B3 + z,
        (x * B2 + y) * B3 + ((z + 1) % B3),
        (x * B2 + y) * B3 + ((z + B3 - 1) % B3),
      ];
      for (let k = 0; k < 6; k++) {
        const nb = nbs[k];
        if (!visited[nb] && domGrid[nb] === d0) { visited[nb] = 1; queue[tail++] = nb; }
      }
    }
    const sz = tail;
    nClusters++;
    sumSize += sz;
    sumSize2 += sz * sz;
    if (sz > maxCluster) maxCluster = sz;
  }
  return {
    meanClusterCells: sumSize / Math.max(1, nClusters),
    weightMeanClusterCells: sumSize2 / Math.max(1, sumSize),
    maxClusterCells: maxCluster,
  };
}

function fitBlockCorrLength(radialCorr) {
  if (!radialCorr.length) return 0;
  const c1 = radialCorr[0];
  if (!(c1 > 0.015)) return 0;
  const pts = [];
  for (let i = 0; i < radialCorr.length; i++) {
    const c = radialCorr[i];
    if (c > 0.015) pts.push([i + 1, Math.log(c), c]);
    else break;
  }
  if (pts.length === 1) return c1 < 0.999 ? -1 / Math.log(c1) : radialCorr.length;
  let sw = 0, sr = 0, sy = 0, srr = 0, sry = 0;
  for (const [r, y, w] of pts) {
    sw += w; sr += w * r; sy += w * y; srr += w * r * r; sry += w * r * y;
  }
  const det = sw * srr - sr * sr;
  if (Math.abs(det) < 1e-12) return c1 < 0.999 ? -1 / Math.log(c1) : 0;
  const slope = (sw * sry - sr * sy) / det;
  if (slope >= -1e-4) return radialCorr.length;
  return Math.min(radialCorr.length * 2, -1 / slope);
}

/**
 * Streaming Zone-Boundary Multi-Cell Block Accumulator (`1×1×2`, `2×2×1`, `2×2×2`, etc.).
 */
export class ZoneBlockAccumulator {
  constructor(dim, A111, idealParentBasis111, reps111, parentSGName, cellMult, zoneStar, kStarVec, opts = {}) {
    const { nNullSamples = 96, maxSavedDomainGrids = 8, centeringBreakingOnly = false } = opts;
    this.dim = dim;
    this.N1 = dim[0]; this.N2 = dim[1]; this.N3 = dim[2];
    this.cellMult = cellMult;
    this.M1 = cellMult[0]; this.M2 = cellMult[1]; this.M3 = cellMult[2];
    this.vol = this.M1 * this.M2 * this.M3;
    this.centeringBreakingOnly = centeringBreakingOnly;
    this.B1 = this.N1 / this.M1;
    this.B2 = this.N2 / this.M2;
    this.B3 = this.N3 / this.M3;
    this.nBlocks = this.B1 * this.B2 * this.B3;
    this.zoneStar = zoneStar;
    this.cellLabel = centeringBreakingOnly && this.vol === 1 ? '1×1×1 (P)' : cellMult.join('×');
    this.parentSGName = parentSGName;
    this.maxSavedDomainGrids = maxSavedDomainGrids;

    this.nBasis111 = idealParentBasis111.length;
    this.D111 = 3 * this.nBasis111;
    this.nBlockBasis = this.vol * this.nBasis111;
    this.Dblk = 3 * this.nBlockBasis;

    this.Ablock = [
      A111[0].map(v => v * this.M1),
      A111[1].map(v => v * this.M2),
      A111[2].map(v => v * this.M3),
    ];
    this.AblockInv = inv3(this.Ablock);
    this.aBlockAvg = (
      Math.hypot(...this.Ablock[0]) +
      Math.hypot(...this.Ablock[1]) +
      Math.hypot(...this.Ablock[2])
    ) / 3;

    // Build ideal parent basis tiled onto [M1, M2, M3] in exact subIdx=(mx*M2+my)*M3+mz order
    this.idealBlockBasis = [];
    let rn = 1;
    for (let mx = 0; mx < this.M1; mx++) {
      for (let my = 0; my < this.M2; my++) {
        for (let mz = 0; mz < this.M3; mz++) {
          for (let s = 0; s < this.nBasis111; s++) {
            const b = idealParentBasis111[s];
            this.idealBlockBasis.push({
              rn: rn++,
              el: b.el,
              frac: [
                (mx + b.frac[0]) / this.M1,
                (my + b.frac[1]) / this.M2,
                (mz + b.frac[2]) / this.M3,
              ],
            });
          }
        }
      }
    }

    const enumRes = enumerateZoneIsotropySubgroups(reps111, this.nBasis111, cellMult, kStarVec, { centeringBreakingOnly });
    this.reps111 = reps111;
    this.blockReps = enumRes.blockReps;
    this._pureCenOps = this.vol === 1 && centeringBreakingOnly
      ? this.blockReps.filter(r => isIdentR(r.R))
      : null;

    // Dimension of pure non-Γ zone-boundary subspace U_ZB = U - U_Γ
    if (this.vol > 1) {
      this.Dzb = (this.vol - 1) * this.D111;
    } else if (this._pureCenOps && this._pureCenOps.length > 1) {
      this.Dzb = Math.round(this.D111 * (1 - 1 / this._pureCenOps.length));
    } else {
      this.Dzb = 0;
    }

    this.maxLag = Math.max(1, Math.min(4, Math.floor(Math.min(this.B1, this.B2, this.B3) / 2)));

    this.fams = enumRes.families
      .filter(f => f.zbDoF > 0 && f.zbDoF < this.Dzb)
      .map(fam => {
        const nDomains = fam.nVariants > 1
          ? fam.nVariants
          : Math.min(8, Math.max(1, fam.normCosetReps.length));
        const domainKind = fam.nVariants > 1 ? 'twin' : (nDomains > 1 ? 'antiphase' : 'single');
        return {
          ...fam,
          cellLabel: this.cellLabel,
          id: `${this.cellLabel}|${fam.spaceGroup}|${fam.order}|${fam.nVariants}|${fam.zbDoF}|${fam.variants[0].opIndices.join(',')}`,
          nDomains,
          domainKind,
          nullR2: fam.zbDoF / Math.max(1, this.Dzb),
          nullBestR2: 0,
          nullBreakingFStat: 0,
          sumMeanR2: 0,
          sumBestR2: 0,
          sumCvR2: 0,
          domainCounts: new Float64Array(nDomains),
          nBlockSamples: 0,
          lagMatches: new Float64Array(this.maxLag),
          lagPairs: new Float64Array(this.maxLag),
          sumMeanCluster: 0,
          sumWeightCluster: 0,
          sumMaxCluster: 0,
          nCfgStats: 0,
          meanExcessDF: 0, m2ExcessDF: 0,
          meanCvDF: 0, m2CvDF: 0,
          meanRawDF: 0, m2RawDF: 0,
          meanNnMatch: 0, m2NnMatch: 0,
          winCount: 0,
          excessDFValues: [],
          varFracMean: new Float64Array(nDomains),
          varFracM2: new Float64Array(nDomains),
          refBrk: new Float64Array(this.Dblk),
          hasRefBrk: false,
          sumAligned: new Float64Array(this.Dblk),
          sumFolded: new Float64Array(this.Dblk),
          cosetCounts: new Float64Array(Math.max(1, fam.normCosetReps.length)),
          nAlignedBlocks: 0,
          cfgAlignMean: new Float64Array(this.Dblk),
          cfgAlignM2: new Float64Array(this.Dblk),
          nCfgAlign: 0,
          savedGrids: [],
        };
      });

    // Scratch buffers
    this._uFull = new Float64Array(this.Dblk);
    this._uZB = new Float64Array(this.Dblk);
    this._subMean = new Float64Array(this.D111);
    this._subGU = Array.from({ length: this.vol }, () =>
      Array.from({ length: reps111.length }, () => new Float64Array(this.D111))
    );
    this._gUzb = Array.from({ length: this.blockReps.length }, () => new Float64Array(this.Dblk));
    this._proj = new Float64Array(this.Dblk);
    this._alignedBlk = new Float64Array(this.Dblk);
    this._brkVec = new Float64Array(this.Dblk);
    this._candBrk = new Float64Array(this.Dblk);
    this._tmp = new Float64Array(this.Dblk);

    this._blkDomain = this.fams.map(() => new Int32Array(this.nBlocks));
    this._blkVarR2 = this.fams.map(f => new Float32Array(this.nBlocks * f.nVariants));
    this._blkValid = new Uint8Array(this.nBlocks);
    this._cfgAlignSum = this.fams.map(() => new Float64Array(this.Dblk));
    this.nConfigs = 0;

    if (this.fams.length > 0) this._calibrateNull(nNullSamples);
  }

  /** Subtract Γ sub-cell average (or Bravais centering average when vol === 1) so outZB lives purely in R^Dzb. */
  _extractZB(uBlk, outZB) {
    const { vol, D111 } = this;
    if (vol === 1 && this._pureCenOps) {
      const ops = this._pureCenOps;
      const mean = this._subMean;
      mean.fill(0);
      for (let k = 0; k < ops.length; k++) {
        const perm = ops[k].perm;
        for (let s = 0; s < this.nBasis111; s++) {
          const dst = perm[s] * 3;
          const src = s * 3;
          mean[dst]     += uBlk[src];
          mean[dst + 1] += uBlk[src + 1];
          mean[dst + 2] += uBlk[src + 2];
        }
      }
      const invC = 1 / ops.length;
      for (let d = 0; d < D111; d++) outZB[d] = uBlk[d] - mean[d] * invC;
      return;
    }
    const mean = this._subMean;
    mean.fill(0);
    for (let m = 0; m < vol; m++) {
      const off = m * D111;
      for (let d = 0; d < D111; d++) mean[d] += uBlk[off + d];
    }
    const invV = 1 / vol;
    for (let d = 0; d < D111; d++) mean[d] *= invV;
    for (let m = 0; m < vol; m++) {
      const off = m * D111;
      for (let d = 0; d < D111; d++) outZB[off + d] = uBlk[off + d] - mean[d];
    }
  }

  /**
   * Populate `this._gUzb[opIdx] = Γ_block(op) uZB` for all block ops by rotating each
   * sub-cell with `reps111` ONCE (`vol * |G0|` small matvecs) and permuting!
   */
  _computeBlockGU(uZB) {
    const { vol, nBasis111, D111, reps111, blockReps } = this;
    for (let m = 0; m < vol; m++) {
      const subSlice = uZB.subarray(m * D111, (m + 1) * D111);
      const guRow = this._subGU[m];
      for (let gi = 0; gi < reps111.length; gi++) {
        applyGammaCell(reps111[gi], subSlice, nBasis111, guRow[gi]);
      }
    }
    for (let oi = 0; oi < blockReps.length; oi++) {
      const op = blockReps[oi];
      const perm = op.perm;
      const gi = op.g0Idx;
      const gPerm = reps111[gi].perm;
      const out = this._gUzb[oi];
      for (let m = 0; m < vol; m++) {
        const rotSub = this._subGU[m][gi];
        const mBase = m * nBasis111;
        for (let s = 0; s < nBasis111; s++) {
          const dst = perm[mBase + s] * 3;
          const src = gPerm[s] * 3;
          out[dst]     = rotSub[src];
          out[dst + 1] = rotSub[src + 1];
          out[dst + 2] = rotSub[src + 2];
        }
      }
    }
  }

  _calibrateNull(nSamples) {
    if (nSamples <= 0 || this.Dzb <= 0 || !this.fams.length) return;
    const rng = mulberry32(20261126 ^ (this.nBlockBasis * 2246822519));
    const randn = makeGaussian(rng);
    const { Dblk, Dzb } = this;
    const uRaw = new Float64Array(Dblk);
    const uZB = this._uZB;
    const proj = this._proj;

    // Simulate finite-block null configs (matching addConfig's per-config ratio + symbol-group branch selection)
    const blocksPerCfg = Math.max(1, Math.min(this.nBlocks, 16));
    const nNullCfgs = Math.max(40, Math.ceil(Math.max(nSamples, 256) / blocksPerCfg));
    const totalBlocks = nNullCfgs * blocksPerCfg;

    const sumAllBest = new Float64Array(this.fams.length);
    const cfgBest = new Float64Array(this.fams.length);
    const cfgRawDF = new Float64Array(this.fams.length);
    const sumRawDF = new Float64Array(this.fams.length);

    // Group indices by spaceGroup symbol so multi-origin Wyckoff branches calibrate their group-max selection bias
    const symGroups = new Map();
    for (let fi = 0; fi < this.fams.length; fi++) {
      const sg = this.fams[fi].spaceGroup;
      if (!symGroups.has(sg)) symGroups.set(sg, []);
      symGroups.get(sg).push(fi);
    }
    const sumGroupMaxDF = new Map([...symGroups.keys()].map(sg => [sg, 0]));

    for (let mc = 0; mc < nNullCfgs; mc++) {
      cfgBest.fill(0);
      for (let b = 0; b < blocksPerCfg; b++) {
        for (let d = 0; d < Dblk; d++) uRaw[d] = randn();
        this._extractZB(uRaw, uZB);
        let norm2 = 0;
        for (let d = 0; d < Dblk; d++) norm2 += uZB[d] * uZB[d];
        if (norm2 < 1e-18) norm2 = 1;

        this._computeBlockGU(uZB);

        for (let fi = 0; fi < this.fams.length; fi++) {
          const fam = this.fams[fi];
          let bestR2 = 0;
          for (let vi = 0; vi < fam.nVariants; vi++) {
            const ops = fam.variants[vi].opIndices;
            const invH = 1 / ops.length;
            proj.fill(0);
            for (let k = 0; k < ops.length; k++) {
              const gu = this._gUzb[ops[k]];
              for (let d = 0; d < Dblk; d++) proj[d] += gu[d];
            }
            let p2 = 0;
            for (let d = 0; d < Dblk; d++) { const v = proj[d] * invH; p2 += v * v; }
            const r2 = Math.min(1, p2 / norm2);
            if (r2 > bestR2) bestR2 = r2;
          }
          cfgBest[fi] += bestR2;
          sumAllBest[fi] += bestR2;
        }
      }

      const invB = 1 / blocksPerCfg;
      for (let fi = 0; fi < this.fams.length; fi++) {
        const fam = this.fams[fi];
        const meanR2 = cfgBest[fi] * invB;
        const remDoF = Math.max(1, Dzb - fam.zbDoF);
        const df = (meanR2 / fam.zbDoF) / Math.max(1e-12, (1 - meanR2) / remDoF);
        cfgRawDF[fi] = df;
        sumRawDF[fi] += df;
      }

      for (const [sg, idxs] of symGroups.entries()) {
        let maxEx = -Infinity;
        for (const fi of idxs) if (cfgRawDF[fi] > maxEx) maxEx = cfgRawDF[fi];
        sumGroupMaxDF.set(sg, sumGroupMaxDF.get(sg) + maxEx);
      }
    }

    const invTot = 1 / totalBlocks;
    const invCfg = 1 / nNullCfgs;
    // Compute extra branch-max penalty per symbol group when multiple Wyckoff branches exist
    const groupBranchBonus = new Map();
    for (const [sg, idxs] of symGroups.entries()) {
      if (idxs.length <= 1) { groupBranchBonus.set(sg, 0); continue; }
      // Calibrate expected excess of best-branch-of-M-configs: ~ (E[max_b rawDF] - max_b E[rawDF]) / sqrt(8)
      const meanOfMax = sumGroupMaxDF.get(sg) * invCfg;
      const maxOfMean = Math.max(...idxs.map(fi => sumRawDF[fi] * invCfg));
      groupBranchBonus.set(sg, Math.max(0, (meanOfMax - maxOfMean) * 0.45));
    }

    for (let fi = 0; fi < this.fams.length; fi++) {
      const fam = this.fams[fi];
      const bonus = groupBranchBonus.get(fam.spaceGroup) || 0;
      if (fam.nVariants === 1) {
        // Exact analytical Beta(d/2, (D-d)/2) expectation + 2nd-order finite-block Jensen correction (zero MC noise!)
        fam.nullBestR2 = fam.zbDoF / Dzb;
        fam.nullBreakingFStat = 1 + 2 / Math.max(4, blocksPerCfg * Dzb) + bonus;
      } else {
        fam.nullBestR2 = sumAllBest[fi] * invTot;
        fam.nullBreakingFStat = sumRawDF[fi] * invCfg + bonus;
      }
    }
  }

  addConfig(uSuper, cfgLabel) {
    if (!this.fams.length) return;
    const { N2, N3, M1, M2, M3, B1, B2, B3, nBlocks, nBlockBasis, D111, Dblk, Dzb } = this;
    this.nConfigs++;

    const uFull = this._uFull;
    const uZB = this._uZB;
    const proj = this._proj;
    const alignedBlk = this._alignedBlk;
    const brkVec = this._brkVec;
    const candBrk = this._candBrk;
    const tmp = this._tmp;

    const cfgSumBest = new Float64Array(this.fams.length);
    const cfgDomCounts = this.fams.map(f => new Float64Array(f.nDomains));
    for (let fi = 0; fi < this.fams.length; fi++) this._cfgAlignSum[fi].fill(0);
    this._blkValid.fill(0);
    const lockPolarity = this.nConfigs === 1;
    const bestBrkNorm2 = lockPolarity ? new Float64Array(this.fams.length).fill(-1) : null;

    // Sample both primary [0,0,0] and dual half-shifted [M1>>1, M2>>1, M3>>1] block origins when vol > 1
    // so the estimator is invariant to unit-cell origin shifts relative to antiphase domain walls!
    const originShifts = this.vol > 1
      ? [[0, 0, 0], [M1 >> 1, M2 >> 1, M3 >> 1]]
      : [[0, 0, 0]];
    let nTotalBlockSamples = 0;
    let nTotalCvBlocks = 0;
    const cfgSumCvByFam = new Float64Array(this.fams.length);
    const savedPrimaryDom = this.fams.map(() => new Int32Array(nBlocks));
    let nPrimaryValid = 0;

    for (let si = 0; si < originShifts.length; si++) {
      const [ox, oy, oz] = originShifts[si];
      const isPrimaryGrid = si === 0;
      this._blkValid.fill(0);
      let shiftValid = 0;

      for (let bx = 0; bx < B1; bx++) {
        for (let by = 0; by < B2; by++) {
          for (let bz = 0; bz < B3; bz++) {
            const bIdx = (bx * B2 + by) * B3 + bz;
            // Gather subcells into uFull (with periodic wrap for dual origin shift)
            for (let mx = 0; mx < M1; mx++) {
              const cx = (bx * M1 + mx + ox) % this.N1;
              for (let my = 0; my < M2; my++) {
                const cy = (by * M2 + my + oy) % N2;
                for (let mz = 0; mz < M3; mz++) {
                  const cz = (bz * M3 + mz + oz) % N3;
                  const c = (cx * N2 + cy) * N3 + cz;
                  const subIdx = (mx * M2 + my) * M3 + mz;
                  uFull.set(uSuper.subarray(c * D111, (c + 1) * D111), subIdx * D111);
                }
              }
            }

            // Remove uniform block COM drift from uFull (for full structure alignment)
            removeCellDrift(uFull, nBlockBasis, tmp);
            uFull.set(tmp);

            // Extract pure non-Γ zone-boundary part U_ZB = U - U_Γ
            this._extractZB(uFull, uZB);
            let norm2 = 0;
            for (let d = 0; d < Dblk; d++) norm2 += uZB[d] * uZB[d];
            if (norm2 < 1e-18) continue;

            this._blkValid[bIdx] = 1;
            shiftValid++;
            if (isPrimaryGrid) nPrimaryValid++;
            nTotalBlockSamples++;
            this._computeBlockGU(uZB);

            for (let fi = 0; fi < this.fams.length; fi++) {
              const fam = this.fams[fi];
              const varR2Buf = this._blkVarR2[fi];
              const vOff = bIdx * fam.nVariants;
              let sumVarR2 = 0, bestR2 = -1, bestV = 0;

              for (let vi = 0; vi < fam.nVariants; vi++) {
                const ops = fam.variants[vi].opIndices;
                const invH = 1 / ops.length;
                proj.fill(0);
                for (let k = 0; k < ops.length; k++) {
                  const gu = this._gUzb[ops[k]];
                  for (let d = 0; d < Dblk; d++) proj[d] += gu[d];
                }
                let p2 = 0;
                for (let d = 0; d < Dblk; d++) { const v = proj[d] * invH; p2 += v * v; }
                const r2 = Math.min(1, p2 / norm2);
                varR2Buf[vOff + vi] = r2;
                sumVarR2 += r2;
                if (r2 > bestR2) { bestR2 = r2; bestV = vi; }
              }

              fam.sumMeanR2 += sumVarR2 / fam.nVariants;
              fam.sumBestR2 += bestR2;
              fam.nBlockSamples += 1;
              cfgSumBest[fi] += bestR2;

              // Align ZB breaking vector first to resolve antiphase/polarity normalizer coset
              const alignOp = fam.variants[bestV].alignOpIdx;
              const normCosets = fam.normCosetReps;
              const h0Ops = fam.variants[0].opIndices;
              const invH0 = 1 / h0Ops.length;

              const alignedZB = this._gUzb[alignOp];
              brkVec.fill(0);
              for (let k = 0; k < h0Ops.length; k++) {
                applyGammaCell(this.blockReps[h0Ops[k]], alignedZB, nBlockBasis, tmp);
                for (let d = 0; d < Dblk; d++) brkVec[d] += tmp[d];
              }
              let b2 = 0;
              for (let d = 0; d < Dblk; d++) {
                brkVec[d] *= invH0;
                b2 += brkVec[d] * brkVec[d];
              }

              if (!fam.hasRefBrk && b2 > 1e-18) {
                fam.refBrk.set(brkVec);
                fam.hasRefBrk = true;
                if (bestBrkNorm2) bestBrkNorm2[fi] = b2;
              } else if (lockPolarity && bestBrkNorm2 && b2 > bestBrkNorm2[fi] * 2.5) {
                let sgnDot = 0;
                for (let d = 0; d < Dblk; d++) sgnDot += brkVec[d] * fam.refBrk[d];
                if (sgnDot >= 0) {
                  fam.refBrk.set(brkVec);
                  bestBrkNorm2[fi] = b2;
                }
              }

              let bestCosetIdx = 0;
              if (normCosets.length > 1 && fam.hasRefBrk) {
                let bestDot = -Infinity;
                for (let ni = 0; ni < normCosets.length; ni++) {
                  applyGammaCell(this.blockReps[normCosets[ni]], brkVec, nBlockBasis, candBrk);
                  let dot = 0;
                  for (let d = 0; d < Dblk; d++) dot += candBrk[d] * fam.refBrk[d];
                  if (dot > bestDot) { bestDot = dot; bestCosetIdx = ni; }
                }
              }

              const effDom = fam.nVariants > 1 ? bestV : (bestCosetIdx % fam.nDomains);
              this._blkDomain[fi][bIdx] = effDom;
              if (isPrimaryGrid) savedPrimaryDom[fi][bIdx] = effDom;
              fam.domainCounts[effDom] += 1;
              cfgDomCounts[fi][effDom] += 1;

              // Apply winning variant + antiphase normalizer coset to FULL block displacement uFull
              applyGammaCell(this.blockReps[alignOp], uFull, nBlockBasis, alignedBlk);
              if (normCosets.length > 1 && fam.hasRefBrk) {
                applyGammaCell(this.blockReps[normCosets[bestCosetIdx]], alignedBlk, nBlockBasis, tmp);
                alignedBlk.set(tmp);
              }

              fam.cosetCounts[bestCosetIdx] += 1;
              for (let d = 0; d < Dblk; d++) {
                fam.sumAligned[d] += alignedBlk[d];
                fam.sumFolded[d] += uFull[d];
                this._cfgAlignSum[fi][d] += alignedBlk[d];
              }
              fam.nAlignedBlocks += 1;
            }
          }
        }
      }

      // Within this shift `si`, blocks are 100% disjoint (share zero atoms!) -> run neighbor-voted Spatial CV!
      if (shiftValid > 0 && nBlocks >= 4) {
        nTotalCvBlocks += shiftValid;
        for (let fi = 0; fi < this.fams.length; fi++) {
          const fam = this.fams[fi];
          const nV = fam.nVariants;
          if (nV <= 1) continue;
          const dom = this._blkDomain[fi];
          const varR2Buf = this._blkVarR2[fi];
          const voteBuf = new Float64Array(nV);
          for (let x = 0; x < B1; x++) {
            const xp = (x + 1) % B1, xm = (x + B1 - 1) % B1;
            for (let y = 0; y < B2; y++) {
              const yp = (y + 1) % B2, ym = (y + B2 - 1) % B2;
              for (let z = 0; z < B3; z++) {
                const bIdx = (x * B2 + y) * B3 + z;
                if (!this._blkValid[bIdx]) continue;
                const zp = (z + 1) % B3, zm = (z + B3 - 1) % B3;
                const nbs = [
                  (xp * B2 + y) * B3 + z, (xm * B2 + y) * B3 + z,
                  (x * B2 + yp) * B3 + z, (x * B2 + ym) * B3 + z,
                  (x * B2 + y) * B3 + zp, (x * B2 + y) * B3 + zm,
                ];
                voteBuf.fill(0);
                for (let k = 0; k < 6; k++) {
                  const nb = nbs[k];
                  if (nb !== bIdx && this._blkValid[nb]) voteBuf[dom[nb]] += 1;
                }
                let bestNbV = 0, bestVote = -1;
                for (let vi = 0; vi < nV; vi++) {
                  if (voteBuf[vi] > bestVote) { bestVote = voteBuf[vi]; bestNbV = vi; }
                }
                cfgSumCvByFam[fi] += varR2Buf[bIdx * nV + bestNbV];
              }
            }
          }
        }
      }
    }

    if (nPrimaryValid === 0 || nTotalBlockSamples === 0) return null;

    // Finalize per-config domain correlation & Welford stats
    let bestCfgIdx = -1, bestCfgExcess = -Infinity;
    for (let fi = 0; fi < this.fams.length; fi++) {
      const fam = this.fams[fi];
      const dom = savedPrimaryDom[fi];
      const nV = fam.nVariants;
      const nDom = fam.nDomains;

      const cfgSumCvR2 = (nV > 1 && nBlocks >= 4) ? cfgSumCvByFam[fi] : cfgSumBest[fi];
      fam.sumCvR2 += cfgSumCvR2;

      let cfgNnMatch = null;
      if (nDom > 1 && nBlocks >= 4) {
        for (let r = 1; r <= this.maxLag; r++) {
          let matches = 0, pairs = 0;
          for (let x = 0; x < B1; x++) {
            const xr = (x + r) % B1;
            for (let y = 0; y < B2; y++) {
              const yr = (y + r) % B2;
              for (let z = 0; z < B3; z++) {
                const zr = (z + r) % B3;
                const c = (x * B2 + y) * B3 + z;
                const d0 = dom[c];
                if (B1 > 1 && dom[(xr * B2 + y) * B3 + z] === d0) matches++;
                if (B2 > 1 && dom[(x * B2 + yr) * B3 + z] === d0) matches++;
                if (B3 > 1 && dom[(x * B2 + y) * B3 + zr] === d0) matches++;
                pairs += (B1 > 1 ? 1 : 0) + (B2 > 1 ? 1 : 0) + (B3 > 1 ? 1 : 0);
              }
            }
          }
          fam.lagMatches[r - 1] += matches;
          fam.lagPairs[r - 1] += Math.max(1, pairs);
          if (r === 1) cfgNnMatch = matches / Math.max(1, pairs);
        }
        const cl = domainClusterStatsBlock(dom, B1, B2, B3);
        fam.sumMeanCluster += cl.meanClusterCells * this.vol;
        fam.sumWeightCluster += cl.weightMeanClusterCells * this.vol;
        fam.sumMaxCluster += cl.maxClusterCells * this.vol;
      }

      // Upsample block twin/antiphase domain grid onto [N1, N2, N3] for 3D DomainCube3D & 2D Slice
      if (fam.savedGrids.length < this.maxSavedDomainGrids && nDom > 1) {
        const upGrid = new Int32Array(this.N1 * this.N2 * this.N3);
        for (let bx = 0; bx < B1; bx++) {
          for (let by = 0; by < B2; by++) {
            for (let bz = 0; bz < B3; bz++) {
              const dVal = dom[(bx * B2 + by) * B3 + bz];
              for (let mx = 0; mx < M1; mx++) {
                for (let my = 0; my < M2; my++) {
                  for (let mz = 0; mz < M3; mz++) {
                    const c = (((bx * M1 + mx) * this.N2) + (by * M2 + my)) * this.N3 + (bz * M3 + mz);
                    upGrid[c] = dVal;
                  }
                }
              }
            }
          }
        }
        fam.savedGrids.push({ configName: cfgLabel, grid: upGrid });
      }

      const cfgBestR2 = cfgSumBest[fi] / nTotalBlockSamples;
      const cfgCvR2 = (nV > 1 && nBlocks >= 4) ? (cfgSumCvR2 / Math.max(1, nTotalCvBlocks)) : cfgBestR2;
      const remDoF = Math.max(1, Dzb - fam.zbDoF);
      const rawDF = (cfgBestR2 / fam.zbDoF) / Math.max(1e-12, (1 - cfgBestR2) / remDoF);
      const excessDF = rawDF - fam.nullBreakingFStat;
      const cvDF = nV > 1
        ? (cfgCvR2 / fam.zbDoF) / Math.max(1e-12, (1 - cfgCvR2) / remDoF)
        : rawDF;

      fam.nCfgStats++;
      fam.excessDFValues.push(excessDF);
      const k = fam.nCfgStats;
      const d1 = rawDF - fam.meanRawDF; fam.meanRawDF += d1 / k; fam.m2RawDF += d1 * (rawDF - fam.meanRawDF);
      const d2 = excessDF - fam.meanExcessDF; fam.meanExcessDF += d2 / k; fam.m2ExcessDF += d2 * (excessDF - fam.meanExcessDF);
      const d3 = cvDF - fam.meanCvDF; fam.meanCvDF += d3 / k; fam.m2CvDF += d3 * (cvDF - fam.meanCvDF);
      if (cfgNnMatch !== null) {
        const d4 = cfgNnMatch - fam.meanNnMatch; fam.meanNnMatch += d4 / k; fam.m2NnMatch += d4 * (cfgNnMatch - fam.meanNnMatch);
      }
      for (let di = 0; di < nDom; di++) {
        const vf = cfgDomCounts[fi][di] / nTotalBlockSamples;
        const dv = vf - fam.varFracMean[di];
        fam.varFracMean[di] += dv / k;
        fam.varFracM2[di] += dv * (vf - fam.varFracMean[di]);
      }

      if (excessDF > bestCfgExcess) { bestCfgExcess = excessDF; bestCfgIdx = fi; }

      fam.nCfgAlign++;
      const ka = fam.nCfgAlign;
      const invB = 1 / nTotalBlockSamples;
      const cSum = this._cfgAlignSum[fi];
      for (let d = 0; d < Dblk; d++) {
        const x = cSum[d] * invB;
        const delta = x - fam.cfgAlignMean[d];
        fam.cfgAlignMean[d] += delta / ka;
        fam.cfgAlignM2[d] += delta * (x - fam.cfgAlignMean[d]);
      }
    }

    return bestCfgIdx >= 0 ? { bestFam: this.fams[bestCfgIdx], bestExcessDF: bestCfgExcess, nBlockBasis: this.nBlockBasis } : null;
  }

  _buildAlignedForFamily(fam) {
    const { Dblk, nBlockBasis, Ablock, AblockInv } = this;
    const invN = 1 / Math.max(1, fam.nAlignedBlocks);
    const meanAligned = new Float64Array(Dblk);
    const meanFolded = new Float64Array(Dblk);
    for (let d = 0; d < Dblk; d++) {
      meanAligned[d] = fam.sumAligned[d] * invN;
      meanFolded[d] = fam.sumFolded[d] * invN;
    }

    const h0Ops = fam.variants[0].opIndices;
    const symAligned = new Float64Array(Dblk);
    const tmp = this._tmp;
    for (let k = 0; k < h0Ops.length; k++) {
      applyGammaCell(this.blockReps[h0Ops[k]], meanAligned, nBlockBasis, tmp);
      for (let d = 0; d < Dblk; d++) symAligned[d] += tmp[d];
    }
    const invH = 1 / h0Ops.length;
    for (let d = 0; d < Dblk; d++) symAligned[d] *= invH;

    const nCfg = fam.nCfgAlign;
    const seVec = new Float64Array(Dblk);
    if (nCfg >= 2) {
      for (let d = 0; d < Dblk; d++) {
        const varCfg = fam.cfgAlignM2[d] / (nCfg - 1);
        seVec[d] = Math.sqrt(Math.max(0, varCfg) / nCfg);
      }
    }

    const toBasis = (uVec, baseSites) => baseSites.map((b, s) => {
      const ux = uVec[s * 3], uy = uVec[s * 3 + 1], uz = uVec[s * 3 + 2];
      const df = [
        ux * AblockInv[0][0] + uy * AblockInv[1][0] + uz * AblockInv[2][0],
        ux * AblockInv[0][1] + uy * AblockInv[1][1] + uz * AblockInv[2][1],
        ux * AblockInv[0][2] + uy * AblockInv[1][2] + uz * AblockInv[2][2],
      ];
      const seNorm = Math.hypot(seVec[s * 3], seVec[s * 3 + 1], seVec[s * 3 + 2]);
      return {
        rn: b.rn,
        el: b.el,
        frac: [wrap01(b.frac[0] + df[0]), wrap01(b.frac[1] + df[1]), wrap01(b.frac[2] + df[2])],
        dispCart: [ux, uy, uz],
        dispNorm: Math.hypot(ux, uy, uz),
        dispSE: seNorm,
      };
    });

    const alignedBasis = toBasis(meanAligned, this.idealBlockBasis);
    const symmetrizedBasis = toBasis(symAligned, this.idealBlockBasis);
    const foldedBasis = toBasis(meanFolded, this.idealBlockBasis);

    const rmsVec = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0) / Math.max(1, nBlockBasis));
    const alignedRms = rmsVec(meanAligned);
    const symRms = rmsVec(symAligned);
    const foldedRms = rmsVec(meanFolded);
    const rmsSE = rmsVec(seVec);
    const maxShift = Math.max(0, ...symmetrizedBasis.map(s => s.dispNorm));

    // Verify space group (and refine non-symmorphic screw/glide symbol!)
    const rawVer = findSpaceGroupOps(Ablock, symmetrizedBasis, 1e-3);
    const refVer = refineSpaceGroupSymbol(rawVer.ops, rawVer);

    const elements = [...new Set(this.idealBlockBasis.map(b => b.el))];
    const elementStats = elements.map(el => {
      const idxs = this.idealBlockBasis.map((b, i) => (b.el === el ? i : -1)).filter(i => i >= 0);
      const rmsOf = (arr) => Math.sqrt(idxs.reduce((s, i) => s + arr[i].dispNorm ** 2, 0) / Math.max(1, idxs.length));
      const maxOf = (arr) => Math.max(0, ...idxs.map(i => arr[i].dispNorm));
      return {
        el,
        count: idxs.length,
        alignedRms: rmsOf(alignedBasis),
        symRms: rmsOf(symmetrizedBasis),
        foldedRms: rmsOf(foldedBasis),
        maxShift: maxOf(symmetrizedBasis),
      };
    });

    return {
      id: fam.id,
      cellMult: this.cellMult,
      cellLabel: this.cellLabel,
      zoneStar: this.zoneStar,
      targetSpaceGroup: fam.spaceGroup,
      targetSpaceGroupNumber: fam.spaceGroupNumber,
      targetPointGroup: fam.pointGroup,
      parentSpaceGroup: this.parentSGName,
      verifiedSpaceGroup: symRms > 1e-4 ? refVer.symbol : fam.spaceGroup,
      verifiedSpaceGroupNumber: symRms > 1e-4 ? refVer.number : fam.spaceGroupNumber,
      verifiedOrder: rawVer.order,
      nVariants: fam.nVariants,
      nNormalizerCosets: Math.max(1, fam.normCosetReps.length),
      cosetFractions: Array.from(fam.cosetCounts).map(x => x / Math.max(1, fam.nAlignedBlocks)),
      nSamples: fam.nAlignedBlocks,
      nConfigs: this.nConfigs,
      A: Ablock,
      alignedRms,
      symRms,
      rmsSE,
      foldedRms,
      maxShift,
      recoveryGain: foldedRms > 1e-9 ? symRms / foldedRms : Infinity,
      elementStats,
      alignedBasis,
      symmetrizedBasis,
      foldedBasis,
    };
  }

  finalize() {
    if (this.nConfigs === 0 || !this.fams.length) return { subgroups: [], alignedById: {} };
    const { Dzb, nConfigs } = this;
    const alignedById = {};

    const allSubgroups = this.fams.map(fam => {
      const n = Math.max(1, fam.nBlockSamples);
      const meanR2 = fam.sumMeanR2 / n;
      const bestDomainR2 = fam.sumBestR2 / n;
      const cvDomainR2 = fam.sumCvR2 / n;
      const remDoF = Math.max(1, Dzb - fam.zbDoF);

      const fStat = (bestDomainR2 / fam.zbDoF) / Math.max(1e-12, (1 - bestDomainR2) / remDoF);
      const breakingFStat = fStat;
      const pooledExcessDF = breakingFStat - fam.nullBreakingFStat;
      const cvBreakingFStat = (cvDomainR2 / fam.zbDoF) / Math.max(1e-12, (1 - cvDomainR2) / remDoF);

      const k = fam.nCfgStats;
      const excessDeltaF = k >= 2 ? fam.meanExcessDF : pooledExcessDF;
      const excessDeltaFSE = k >= 2 ? Math.sqrt(Math.max(0, fam.m2ExcessDF / (k - 1)) / k) : null;
      const cvDeltaFSE = k >= 2 ? Math.sqrt(Math.max(0, fam.m2CvDF / (k - 1)) / k) : null;
      const zScore = excessDeltaFSE && excessDeltaFSE > 1e-9 ? fam.meanExcessDF / excessDeltaFSE : null;
      const winRate = k > 0 ? fam.winCount / k : 0;
      let medianExcessDeltaF = excessDeltaF;
      if (fam.excessDFValues.length > 0) {
        const sArr = [...fam.excessDFValues].sort((a, b) => a - b);
        const mid = sArr.length >> 1;
        medianExcessDeltaF = sArr.length % 2 ? sArr[mid] : 0.5 * (sArr[mid - 1] + sArr[mid]);
      }

      const domainFractions = Array.from(fam.domainCounts).map(x => x / n);
      const domainFractionSE = k >= 2
        ? Array.from(fam.varFracM2).map(m2 => Math.sqrt(Math.max(0, m2 / (k - 1)) / k))
        : null;

      const nDom = fam.nDomains;
      const randomMatchRate = 1 / Math.max(1, nDom);
      const radialMatchRate = [];
      const radialCorr = [];
      if (nDom > 1) {
        for (let r = 0; r < this.maxLag; r++) {
          const pMatch = fam.lagPairs[r] > 0 ? fam.lagMatches[r] / fam.lagPairs[r] : randomMatchRate;
          radialMatchRate.push(pMatch);
          radialCorr.push((pMatch - randomMatchRate) / Math.max(1e-12, 1 - randomMatchRate));
        }
      }
      const neighborMatchRate = radialMatchRate.length ? radialMatchRate[0] : null;
      const neighborMatchSE = k >= 2 && nDom > 1
        ? Math.sqrt(Math.max(0, fam.m2NnMatch / (k - 1)) / k)
        : null;
      const corrLengthBlocks = fitBlockCorrLength(radialCorr);
      const corrLengthAng = corrLengthBlocks * this.aBlockAvg;

      alignedById[fam.id] = this._buildAlignedForFamily(fam);

      return {
        id: fam.id,
        cellMult: this.cellMult,
        cellLabel: this.cellLabel,
        zoneStar: this.zoneStar,
        isZoneBoundary: true,
        spaceGroup: fam.spaceGroup,
        spaceGroupNumber: fam.spaceGroupNumber,
        pointGroup: fam.pointGroup,
        centering: fam.centering,
        order: fam.order,
        nVariants: fam.nVariants,
        nDomains: nDom,
        domainKind: fam.domainKind,
        dof: fam.dof,
        breakingDoF: fam.zbDoF,
        nullR2: fam.nullR2,
        nullBestR2: fam.nullBestR2,
        nullBreakingFStat: fam.nullBreakingFStat,
        meanR2,
        bestDomainR2,
        cvDomainR2,
        cvBreakingR2: cvDomainR2,
        fStat,
        breakingFStat,
        excessDeltaF,
        medianExcessDeltaF,
        excessDeltaFSE,
        cvBreakingFStat,
        cvDeltaFSE,
        zScore,
        winRate,
        domainFractions,
        domainFractionSE,
        neighborMatchRate,
        neighborMatchSE,
        neighborExcess: neighborMatchRate !== null ? neighborMatchRate - randomMatchRate : null,
        radialCorr,
        corrLengthCells: corrLengthBlocks * Math.cbrt(this.vol),
        corrLengthAng,
        meanClusterCells: nDom > 1 ? fam.sumMeanCluster / nConfigs : null,
        weightMeanClusterCells: nDom > 1 ? fam.sumWeightCluster / nConfigs : null,
        maxClusterCells: nDom > 1 ? fam.sumMaxCluster / nConfigs : null,
        domainGrid: fam.savedGrids.length ? Array.from(fam.savedGrids[0].grid) : null,
        savedGrids: fam.savedGrids.map(sg => ({ configName: sg.configName, grid: Array.from(sg.grid) })),
      };
    });

    // Deduplicate origin-shifted branches that share the same (cellLabel, spaceGroup), keeping the strongest branch
    const bestBySymbol = new Map();
    for (const s of allSubgroups) {
      const key = `${s.cellLabel}|${s.spaceGroup}`;
      const prev = bestBySymbol.get(key);
      const curScore = nConfigs >= 3 ? s.medianExcessDeltaF : s.excessDeltaF;
      const prevScore = prev ? (nConfigs >= 3 ? prev.medianExcessDeltaF : prev.excessDeltaF) : -Infinity;
      if (!prev || curScore > prevScore) bestBySymbol.set(key, s);
    }

    return {
      subgroups: [...bestBySymbol.values()],
      alignedById,
    };
  }
}

/**
 * Determine the commensurate zone-boundary channels ([M1, M2, M3], star, kStarVec)
 * supported by `dim = [N1, N2, N3]` and initial FFT power `fftRe/Im` on Config #1.
 */
export function selectZoneChannels(dim, A111, fftPower1, nBasis111, opts = {}) {
  const { maxVolume = 8, maxBlockAtoms = 240 } = opts;
  const [N1, N2, N3] = dim;
  const e1 = Math.hypot(...A111[0]), e2 = Math.hypot(...A111[1]), e3 = Math.hypot(...A111[2]);
  const isCubicMetric = Math.abs(e1 - e2) / e1 < 0.02 && Math.abs(e1 - e3) / e1 < 0.02;

  let medianP = 0;
  if (fftPower1 && fftPower1.length > 0) {
    const sorted = Array.from(fftPower1).sort((a, b) => a - b);
    medianP = sorted[sorted.length >> 1] || 0;
  }

  // Sum exact zone-boundary voxel + immediate ±1 diffuse satellite halo so antiphase
  // domain walls (which split k_ZB into k_ZB ± Δq) still steer to the right axis!
  const haloPowerAtHalf = (hx, hy, hz) => {
    if (!fftPower1) return 0;
    if ((hx && N1 % 2 !== 0) || (hy && N2 % 2 !== 0) || (hz && N3 % 2 !== 0)) return -1;
    const kx0 = hx ? N1 / 2 : 0;
    const ky0 = hy ? N2 / 2 : 0;
    const kz0 = hz ? N3 / 2 : 0;
    let sum = 0;
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dz = -1; dz <= 1; dz++) {
          if (Math.abs(dx) + Math.abs(dy) + Math.abs(dz) > 1) continue;
          const kx = (kx0 + dx + N1) % N1;
          const ky = (ky0 + dy + N2) % N2;
          const kz = (kz0 + dz + N3) % N3;
          if (kx === 0 && ky === 0 && kz === 0) continue;
          const w = (dx === 0 && dy === 0 && dz === 0) ? 1.0 : 0.5;
          sum += w * (fftPower1[(kx * N2 + ky) * N3 + kz] || 0);
        }
      }
    }
    return sum;
  };

  const channels = [];

  // 1. X-star (1D doubling, vol = 2): [1,1,2], [1,2,1], [2,1,1]
  if (maxVolume >= 2 && 2 * nBasis111 <= maxBlockAtoms) {
    const xCands = [
      { cellMult: [1, 1, 2], kStar: [0, 0, 0.5], p: haloPowerAtHalf(0, 0, 1), ok: N3 % 2 === 0 && (N1 * N2 * (N3 / 2)) >= 2 },
      { cellMult: [1, 2, 1], kStar: [0, 0.5, 0], p: haloPowerAtHalf(0, 1, 0), ok: N2 % 2 === 0 && (N1 * (N2 / 2) * N3) >= 2 },
      { cellMult: [2, 1, 1], kStar: [0.5, 0, 0], p: haloPowerAtHalf(1, 0, 0), ok: N1 % 2 === 0 && ((N1 / 2) * N2 * N3) >= 2 },
    ].filter(c => c.ok);

    if (xCands.length > 0) {
      if (isCubicMetric) {
        const def = xCands.find(c => c.cellMult[2] === 2) || xCands[0];
        const best = [...xCands].sort((a, b) => b.p - a.p)[0];
        const chosen = (best.p > def.p * 1.6 && best.p > medianP * 6) ? best : def;
        channels.push({ cellMult: chosen.cellMult, zoneStar: 'X', kStar: chosen.kStar });
      } else {
        const cDoub = xCands.find(c => c.cellMult[2] === 2);
        const aDoub = xCands.find(c => c.cellMult[0] === 2) || xCands.find(c => c.cellMult[1] === 2);
        if (cDoub) channels.push({ cellMult: cDoub.cellMult, zoneStar: 'X', kStar: cDoub.kStar });
        if (aDoub && (!cDoub || Math.abs(e1 - e3) / e1 >= 0.02)) {
          channels.push({ cellMult: aDoub.cellMult, zoneStar: 'X', kStar: aDoub.kStar });
        }
      }
    }
  }

  // 2. M-star (2D doubling, vol = 4): [2,2,1], [2,1,2], [1,2,2]
  if (maxVolume >= 4 && 4 * nBasis111 <= maxBlockAtoms) {
    const mCands = [
      { cellMult: [2, 2, 1], kStar: [0.5, 0.5, 0], p: haloPowerAtHalf(1, 1, 0), ok: N1 % 2 === 0 && N2 % 2 === 0 && ((N1 / 2) * (N2 / 2) * N3) >= 4 },
      { cellMult: [2, 1, 2], kStar: [0.5, 0, 0.5], p: haloPowerAtHalf(1, 0, 1), ok: N1 % 2 === 0 && N3 % 2 === 0 && ((N1 / 2) * N2 * (N3 / 2)) >= 4 },
      { cellMult: [1, 2, 2], kStar: [0, 0.5, 0.5], p: haloPowerAtHalf(0, 1, 1), ok: N2 % 2 === 0 && N3 % 2 === 0 && (N1 * (N2 / 2) * (N3 / 2)) >= 4 },
    ].filter(c => c.ok);

    if (mCands.length > 0) {
      const def = mCands.find(c => c.cellMult[2] === 1) || mCands[0];
      const best = [...mCands].sort((a, b) => b.p - a.p)[0];
      const chosen = (best.p > def.p * 1.6 && best.p > medianP * 6) ? best : def;
      channels.push({ cellMult: chosen.cellMult, zoneStar: 'M', kStar: chosen.kStar });
    }
  }

  // 3. R-star (3D doubling, vol = 8): [2,2,2]
  if (maxVolume >= 8 && 8 * nBasis111 <= maxBlockAtoms && N1 % 2 === 0 && N2 % 2 === 0 && N3 % 2 === 0 && ((N1 / 2) * (N2 / 2) * (N3 / 2)) >= 4) {
    channels.push({ cellMult: [2, 2, 2], zoneStar: 'R', kStar: [0.5, 0.5, 0.5] });
  }

  return channels;
}
