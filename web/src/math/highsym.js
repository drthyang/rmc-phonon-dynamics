// web/src/math/highsym.js
//
// Standard high-symmetry points (Setyawan & Curtarolo 2010, same convention as
// seekpath) in the PRIMITIVE reciprocal basis, plus suggested band paths, for
// the common Bravais lattices. Combined with the primitive cell (bravais.js) and
// the Wigner-Seitz BZ (brillouin.js) this reproduces seekpath's k-path tool for
// these lattices: a primitive Brillouin zone with the correct labeled points.

import { brillouinZone } from './brillouin.js';
import { primToConv, fracToCart } from './bravais.js';

const T = 1 / 3, U2 = 2 / 3;

// label -> primitive-reciprocal fractional coords; path = ordered [from,to] pairs.
const TABLES = {
  CUB: {
    pts: { GAMMA: [0, 0, 0], X: [0, 0.5, 0], M: [0.5, 0.5, 0], R: [0.5, 0.5, 0.5] },
    path: [['GAMMA', 'X'], ['X', 'M'], ['M', 'GAMMA'], ['GAMMA', 'R'], ['R', 'X'], ['M', 'R']],
  },
  FCC: {
    pts: { GAMMA: [0, 0, 0], X: [0.5, 0, 0.5], L: [0.5, 0.5, 0.5], W: [0.5, 0.25, 0.75], K: [0.375, 0.375, 0.75], U: [0.625, 0.25, 0.625] },
    path: [['GAMMA', 'X'], ['X', 'W'], ['W', 'K'], ['K', 'GAMMA'], ['GAMMA', 'L'], ['L', 'U'], ['U', 'W'], ['W', 'L'], ['L', 'K'], ['U', 'X']],
  },
  BCC: {
    pts: { GAMMA: [0, 0, 0], H: [0.5, -0.5, 0.5], P: [0.25, 0.25, 0.25], N: [0, 0, 0.5] },
    path: [['GAMMA', 'H'], ['H', 'N'], ['N', 'GAMMA'], ['GAMMA', 'P'], ['P', 'H'], ['P', 'N']],
  },
  TET: {
    pts: { GAMMA: [0, 0, 0], X: [0, 0.5, 0], M: [0.5, 0.5, 0], Z: [0, 0, 0.5], R: [0, 0.5, 0.5], A: [0.5, 0.5, 0.5] },
    path: [['GAMMA', 'X'], ['X', 'M'], ['M', 'GAMMA'], ['GAMMA', 'Z'], ['Z', 'R'], ['R', 'A'], ['A', 'Z'], ['X', 'R'], ['M', 'A']],
  },
  ORC: {
    pts: { GAMMA: [0, 0, 0], X: [0.5, 0, 0], S: [0.5, 0.5, 0], Y: [0, 0.5, 0], Z: [0, 0, 0.5], U: [0.5, 0, 0.5], R: [0.5, 0.5, 0.5], TT: [0, 0.5, 0.5] },
    path: [['GAMMA', 'X'], ['X', 'S'], ['S', 'Y'], ['Y', 'GAMMA'], ['GAMMA', 'Z'], ['Z', 'U'], ['U', 'R'], ['R', 'TT'], ['TT', 'Z'], ['Y', 'TT'], ['U', 'X'], ['S', 'R']],
  },
  HEX: {
    pts: { GAMMA: [0, 0, 0], A: [0, 0, 0.5], K: [T, T, 0], H: [T, T, 0.5], M: [0.5, 0, 0], L: [0.5, 0, 0.5] },
    path: [['GAMMA', 'M'], ['M', 'K'], ['K', 'GAMMA'], ['GAMMA', 'A'], ['A', 'L'], ['L', 'H'], ['H', 'A'], ['L', 'M'], ['K', 'H']],
  },
};
// Fallbacks for lattices without a dedicated table yet (display + a basic path).
TABLES.BCT = TABLES.TET; TABLES.ORCF = TABLES.ORC; TABLES.ORCI = TABLES.ORC; TABLES.ORCC = TABLES.ORC;
TABLES.RHL = TABLES.HEX; TABLES.MCL = TABLES.ORC; TABLES.MCLC = TABLES.ORC; TABLES.TRI = TABLES.CUB;
void U2;

const DISPLAY = { GAMMA: 'Γ', TT: 'T' };
export const displayLabel = (l) => DISPLAY[l] || l;

export function hasTable(code) { return code === 'CUB' || code === 'FCC' || code === 'BCC' || code === 'TET' || code === 'ORC' || code === 'HEX'; }

/**
 * Assemble the BZ model for a Bravais analysis (from bravais.analyzeBravais):
 *   points : { label: { frac (primitive), fracConv (conventional, for calc),
 *                       cart (primitive cartesian, for display), display } }
 *   path   : [[fromLabel, toLabel], ...]   (seekpath-style suggested path)
 *   bz     : { vertices, faces, edges } primitive Wigner-Seitz cell (cartesian)
 */
export function buildBZModel(bravais) {
  const tbl = TABLES[bravais.code] || TABLES.CUB;
  const points = {};
  for (const [label, frac] of Object.entries(tbl.pts)) {
    points[label] = {
      frac,
      fracConv: primToConv(frac, bravais.T),
      cart: fracToCart(frac, bravais.B_prim),
      display: displayLabel(label),
    };
  }
  return { code: bravais.code, points, path: tbl.path, bz: brillouinZone(bravais.B_prim) };
}
