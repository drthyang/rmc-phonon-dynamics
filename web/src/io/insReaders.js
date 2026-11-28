// web/src/io/insReaders.js
//
// Experimental & Calculated Inelastic Neutron Scattering (INS) File Readers
// =========================================================================
// Directly parses reduced spectrometer outputs from SNS (SEQUOIA, CNCS, ARCS, VISION),
// ISIS (LET, MARI, MERLIN, TOSCA), and ILL/NCNR (Mantid, MSlice, Horace, DAVE) without
// requiring manual spreadsheet conversion:
//
//   1. 1D GDOS / S(E) (`parseExpIns1D`):
//      * 2- or 3-column CSV / whitespace `.xy`, `.xye`, `.dat`, `.txt` (`E, I [, sigma]`)
//      * Mantid `SaveAscii` 4-column spectrum tables (`SpecIdx, E, I, sigma`)
//      * DAVE 1D ASCII files with `#` metadata headers and automatic unit conversion
//        (`ueV` / `micro-eV` -> `meV`, `cm^-1` -> `meV`, `THz` -> `meV`)
//      * Automatic filtering of Mantid/DAVE masked-detector sentinels (`<= -1e20`, `NaN`).
//
//   2. 2D Powder S(|Q|, E) (`parseExpIns2D`):
//      * Mantid / MSlice / Horace `.spe` (`SaveSPE`) ASCII format (`### Phi/Q Grid`,
//        `### Energy Grid`, `### S(Phi,w)`, `### Errors`, with `-1e30` kinematic masks
//        preserved as `NaN` so unmeasured pixels never bias Rw fits!)
//      * DAVE 2D ASCII format (`# X-axis` / `# Y-axis` / `# Data` blocks)
//      * Dense 2D matrix CSV/TXT (`[label, Q0, Q1, ...]`)
//      * 3- or 4-column `(Q, E, I [, sigma])` / `(E, Q, I)` triplet tables (Mantid `SaveAscii`).

import { THZ_TO_MEV } from '../constants.js';

const CM1_TO_MEV = 0.12398419843320026;
const MANTID_MASK_THRESHOLD = -1e19; // Mantid SaveSPE / SaveAscii uses -1e30 for masked detector bins

function splitRow(line) {
  const trimmed = line.trim();
  if (!trimmed) return [];
  if (trimmed.includes(',')) return trimmed.split(',').map(s => s.trim());
  return trimmed.split(/\s+/);
}

function detectEnergyScaleToMev(headerLines) {
  const joined = headerLines.join(' \n ');
  if (/\b(uev|µev|micro[-\s]?ev)\b/i.test(joined)) return 1e-3;
  if (/\b(cm\^-1|cm-1|wavenumber)\b/i.test(joined)) return CM1_TO_MEV;
  if (/\bthz\b/i.test(joined) && !/\bmev\b/i.test(joined)) return THZ_TO_MEV;
  return 1.0;
}

/**
 * Parse a 1D experimental INS spectrum (`.xye`, `.xy`, `.dat`, `.csv`, Mantid `SaveAscii`, DAVE 1D).
 * Returns `{ name, energy, intensity, sigma, format }`.
 */
export function parseExpIns1D(text, name = 'experimental_1d.xye') {
  const rawLines = text.split(/\r?\n/);
  const commentLines = [];
  const dataTokens = [];

  for (const raw of rawLines) {
    const line = raw.trim();
    if (!line) continue;
    if (line.startsWith('#') || line.startsWith('%') || line.startsWith('//') || line.startsWith('!')) {
      commentLines.push(line);
      continue;
    }
    const parts = splitRow(line);
    if (parts.length < 2) continue;
    const nums = parts.map(Number);
    if (!Number.isFinite(nums[0]) || !Number.isFinite(nums[1])) {
      commentLines.push(line);
      continue;
    }
    dataTokens.push(nums);
  }

  if (dataTokens.length < 3) {
    throw new Error(`Could not parse ≥3 numeric (E, I) points from "${name}".`);
  }

  // Detect Mantid SaveAscii 4-column format where column 0 is constant integer spectrum ID (e.g. `1, E, I, E_err`)
  const isFourColSpecIdx = dataTokens.every(r => r.length >= 4 && Number.isInteger(r[0]) && r[0] === dataTokens[0][0]);
  const eScale = detectEnergyScaleToMev(commentLines);

  const rows = [];
  for (const nums of dataTokens) {
    const eRaw = isFourColSpecIdx ? nums[1] : nums[0];
    const yRaw = isFourColSpecIdx ? nums[2] : nums[1];
    const sRaw = isFourColSpecIdx ? nums[3] : (nums.length >= 3 ? nums[2] : NaN);
    if (!Number.isFinite(eRaw) || !Number.isFinite(yRaw)) continue;
    if (yRaw <= MANTID_MASK_THRESHOLD) continue; // skip masked Mantid bins (-1e30)
    const e = eRaw * eScale;
    const sig = Number.isFinite(sRaw) && sRaw > 0 && sRaw < 1e19 ? sRaw : null;
    rows.push([e, yRaw, sig]);
  }

  if (rows.length < 3) {
    throw new Error(`All points in "${name}" were masked or non-numeric.`);
  }

  rows.sort((a, b) => a[0] - b[0]);
  const hasSigma = rows.every(r => r[2] !== null);
  return {
    name,
    format: name.toLowerCase().endsWith('.xye') ? 'xye' : (isFourColSpecIdx ? 'mantid-ascii' : '1d-table'),
    energy: rows.map(r => r[0]),
    intensity: rows.map(r => r[1]),
    sigma: hasSigma ? rows.map(r => r[2]) : null,
  };
}

/**
 * Parse an rmcph-mlip `_ndos.csv` file:
 *   energy_meV,total_dos,neutron_gdos,gdos_<El1>,gdos_<El2>,...
 */
export function parseNeutronDosCsv(text, name = 'mlip_ndos.csv') {
  const lines = text.split(/\r?\n/).filter(l => l.trim() && !l.trim().startsWith('#'));
  if (lines.length < 4) throw new Error(`"${name}" is too short to be a neutron DOS CSV.`);
  const header = splitRow(lines[0]).map(h => h.toLowerCase());
  if (!header.includes('energy_mev') || !header.includes('neutron_gdos')) {
    throw new Error(`"${name}" does not have energy_meV / neutron_gdos columns.`);
  }
  const idxE = header.indexOf('energy_mev');
  const idxTot = header.indexOf('total_dos');
  const idxGdos = header.indexOf('neutron_gdos');
  const elCols = [];
  const rawHeader = splitRow(lines[0]);
  for (let i = 0; i < rawHeader.length; i++) {
    if (rawHeader[i].toLowerCase().startsWith('gdos_')) {
      elCols.push({ el: rawHeader[i].slice(5), idx: i });
    }
  }

  const energy_mev = [];
  const total_dos = [];
  const neutron_gdos = [];
  const element_gdos = {};
  for (const c of elCols) element_gdos[c.el] = [];

  for (let i = 1; i < lines.length; i++) {
    const parts = splitRow(lines[i]);
    const e = Number(parts[idxE]);
    const g = Number(parts[idxGdos]);
    if (!Number.isFinite(e) || !Number.isFinite(g)) continue;
    energy_mev.push(e);
    total_dos.push(idxTot >= 0 && Number.isFinite(Number(parts[idxTot])) ? Number(parts[idxTot]) : g);
    neutron_gdos.push(g);
    for (const c of elCols) {
      const v = Number(parts[c.idx]);
      element_gdos[c.el].push(Number.isFinite(v) ? v : 0);
    }
  }
  if (energy_mev.length < 3) throw new Error(`No valid rows parsed from "${name}".`);

  return {
    name,
    energy_mev,
    total_dos,
    neutron_gdos,
    element_gdos,
    fwhm_mev: 0,
  };
}

function edgesToCenters(arr, targetLen) {
  if (arr.length === targetLen) return arr.slice();
  if (arr.length === targetLen + 1) {
    const out = new Array(targetLen);
    for (let i = 0; i < targetLen; i++) out[i] = 0.5 * (arr[i] + arr[i + 1]);
    return out;
  }
  return arr.slice(0, targetLen);
}

/**
 * Parse Mantid / MSlice / Horace ASCII `.spe` (`SaveSPE`) 2D S(|Q|, E) files.
 *
 * Format specification:
 *   Line 0:  `<nQ> <nE>`
 *   Section: `### Phi Grid` or `### Q Grid` -> `nQ` or `nQ + 1` floats
 *   Section: `### Energy Grid`              -> `nE + 1` (edges) or `nE` (centers) floats
 *   Repeated `nQ` times:
 *     `### S(Phi,w)` or `### S(Q,w)` -> `nE` intensity floats (`-1e30` = kinematic mask -> `NaN`)
 *     `### Errors`                   -> `nE` uncertainty floats
 */
export function parseMantidSpe(text, name = 'experiment.spe') {
  const lines = text.split(/\r?\n/);
  let cursor = 0;
  while (cursor < lines.length && !lines[cursor].trim()) cursor++;
  if (cursor >= lines.length) throw new Error(`Empty .spe file "${name}".`);

  const headNums = lines[cursor].trim().split(/\s+/).map(Number);
  const nQ = Math.round(headNums[0]);
  const nE = Math.round(headNums[1]);
  if (!(nQ >= 1 && nE >= 2)) {
    throw new Error(`Invalid .spe header dimensions in "${name}".`);
  }
  cursor++;

  const readFloatsUntilMarker = () => {
    const vals = [];
    while (cursor < lines.length) {
      const t = lines[cursor].trim();
      if (t.startsWith('###')) break;
      if (t) {
        // Fortran fixed-width floats can occasionally abut (e.g. `-1.000E+30-1.000E+30`)
        const spaced = t.replace(/(\d)([+-]\d+\.\d)/g, '$1 $2');
        for (const tok of spaced.split(/\s+/)) {
          const v = Number(tok);
          if (Number.isFinite(v)) vals.push(v);
        }
      }
      cursor++;
    }
    return vals;
  };

  let rawQ = null;
  let rawE = null;
  const qSlices = [];
  const qErrors = [];

  while (cursor < lines.length) {
    const marker = lines[cursor].trim();
    if (!marker.startsWith('###')) {
      cursor++;
      continue;
    }
    const lower = marker.toLowerCase();
    cursor++;
    const block = readFloatsUntilMarker();
    if (lower.includes('phi grid') || lower.includes('q grid')) {
      rawQ = block;
    } else if (lower.includes('energy grid')) {
      rawE = block;
    } else if (lower.includes('s(phi,w)') || lower.includes('s(q,w)') || lower.includes('s(|q|,w)')) {
      qSlices.push(block);
    } else if (lower.includes('error')) {
      qErrors.push(block);
    }
  }

  if (!rawQ || !rawE || qSlices.length < nQ) {
    throw new Error(`Incomplete .spe sections in "${name}" (expected ${nQ} S(Q,w) blocks, got ${qSlices.length}).`);
  }

  const q_grid = edgesToCenters(rawQ, nQ);
  const e_grid = edgesToCenters(rawE, nE);

  // Transpose from SPE's [qi][ei] into viewer's standard `sqe[ei][qi]`, preserving kinematic masks as NaN
  const sqe = Array.from({ length: nE }, () => new Array(nQ).fill(NaN));
  const sigma = qErrors.length >= nQ ? Array.from({ length: nE }, () => new Array(nQ).fill(NaN)) : null;

  for (let qi = 0; qi < nQ; qi++) {
    const sRow = qSlices[qi];
    const eRow = qErrors[qi];
    for (let ei = 0; ei < Math.min(nE, sRow.length); ei++) {
      const v = sRow[ei];
      if (Number.isFinite(v) && v > MANTID_MASK_THRESHOLD) {
        sqe[ei][qi] = v;
        if (sigma && eRow && Number.isFinite(eRow[ei]) && eRow[ei] >= 0 && eRow[ei] < 1e19) {
          sigma[ei][qi] = eRow[ei];
        }
      }
    }
  }

  return { name, format: 'spe', q_grid, e_grid, sqe, sigma };
}

/**
 * Parse NCNR DAVE 2D ASCII format (`# X-axis` / `# Y-axis` / `# Data` blocks).
 */
function tryParseDave2D(text, name) {
  if (!/#\s*(x-axis|y-axis|dimensions)/i.test(text)) return null;
  const lines = text.split(/\r?\n/);
  let section = null;
  const xVals = [];
  const yVals = [];
  const dataRows = [];
  let xIsQ = false;

  for (const raw of lines) {
    const t = raw.trim();
    if (!t) continue;
    if (t.startsWith('#')) {
      const low = t.toLowerCase();
      if (low.includes('x-axis') || low.startsWith('# x:')) {
        section = 'x';
        if (low.includes('wavevector') || low.includes('|q|') || low.includes('a^-1') || low.includes('angstrom')) xIsQ = true;
      } else if (low.includes('y-axis') || low.startsWith('# y:')) {
        section = 'y';
      } else if (low.includes('data') || low.includes('intensity') || low.includes('z-axis')) {
        section = 'data';
      } else if (low.includes('error')) {
        section = 'error';
      }
      continue;
    }
    if (section === 'error') break;
    const nums = splitRow(t).map(Number).filter(Number.isFinite);
    if (!nums.length) continue;
    if (section === 'x') xVals.push(...nums);
    else if (section === 'y') yVals.push(...nums);
    else if (section === 'data') dataRows.push(nums);
  }

  if (xVals.length < 2 || yVals.length < 2 || dataRows.length < 2) return null;
  // Build standard `sqe[ei][qi]`
  if (xIsQ && dataRows.length === yVals.length && dataRows[0].length === xVals.length) {
    const sqe = dataRows.map(r => r.map(v => (v <= MANTID_MASK_THRESHOLD ? NaN : v)));
    return { name, format: 'dave-2d', q_grid: xVals, e_grid: yVals, sqe };
  }
  if (!xIsQ && dataRows.length === yVals.length && dataRows[0].length === xVals.length) {
    // X = Energy (cols), Y = Q (rows) -> transpose to sqe[ei][qi]
    const nE = xVals.length, nQ = yVals.length;
    const sqe = Array.from({ length: nE }, (_, ei) =>
      Array.from({ length: nQ }, (__, qi) => {
        const v = dataRows[qi][ei];
        return v <= MANTID_MASK_THRESHOLD ? NaN : v;
      })
    );
    return { name, format: 'dave-2d', q_grid: yVals, e_grid: xVals, sqe };
  }
  return null;
}

/**
 * Parse a 2D experimental S(|Q|, E) file across Mantid `.spe`, DAVE 2D ASCII,
 * dense matrix CSV/TXT, or 3/4-column `(Q, E, I [, sigma])` tables.
 */
export function parseExpIns2D(text, name = 'experimental_2d.csv') {
  const trimmedStart = text.trimStart();
  if (name.toLowerCase().endsWith('.spe') || /^\d+\s+\d+\s*\r?\n\s*###\s*(phi|q)\s+grid/i.test(trimmedStart)) {
    return parseMantidSpe(text, name);
  }

  const dave = tryParseDave2D(text, name);
  if (dave) return dave;

  const lines = text.split(/\r?\n/).filter(l => {
    const t = l.trim();
    return t && !t.startsWith('#') && !t.startsWith('%') && !t.startsWith('//');
  });
  if (lines.length < 3) throw new Error(`Could not parse 2D S(|Q|, E) from "${name}".`);

  const first = splitRow(lines[0]);

  // Format (A): Matrix with >= 4 columns and numeric Q header in cols 1..N
  if (first.length >= 4 && first.slice(1).every(v => Number.isFinite(Number(v)))) {
    const q_grid = first.slice(1).map(Number);
    const e_grid = [];
    const sqe = [];
    for (let i = 1; i < lines.length; i++) {
      const parts = splitRow(lines[i]);
      if (parts.length < q_grid.length + 1) continue;
      const e = Number(parts[0]);
      if (!Number.isFinite(e)) continue;
      const row = new Array(q_grid.length);
      for (let j = 0; j < q_grid.length; j++) {
        const v = Number(parts[j + 1]);
        row[j] = Number.isFinite(v) && v > MANTID_MASK_THRESHOLD ? v : NaN;
      }
      e_grid.push(e);
      sqe.push(row);
    }
    if (e_grid.length >= 2 && q_grid.length >= 2) {
      return { name, format: 'matrix-2d', q_grid, e_grid, sqe };
    }
  }

  // Format (B): 3- or 4-column (Q, E, I [, sigma]) or (E, Q, I)
  const firstIsHeader = !Number.isFinite(Number(first[0])) || !Number.isFinite(Number(first[1]));
  const swapQE = firstIsHeader && /^e/i.test(first[0]) && /^q/i.test(first[1]);
  const startRow = firstIsHeader ? 1 : 0;

  const qSet = new Set();
  const eSet = new Set();
  const pts = [];
  for (let i = startRow; i < lines.length; i++) {
    const parts = splitRow(lines[i]);
    if (parts.length < 3) continue;
    const a = Number(parts[0]), b = Number(parts[1]), val = Number(parts[2]);
    if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(val)) continue;
    const q = swapQE ? b : a;
    const e = swapQE ? a : b;
    const qk = Math.round(q * 1e5) / 1e5;
    const ek = Math.round(e * 1e5) / 1e5;
    qSet.add(qk);
    eSet.add(ek);
    pts.push([qk, ek, val <= MANTID_MASK_THRESHOLD ? NaN : val]);
  }
  if (pts.length < 4 || qSet.size < 2 || eSet.size < 2) {
    throw new Error(`Could not recognize "${name}" as a Mantid .spe, DAVE 2D, matrix, or (Q, E, I) table.`);
  }

  const q_grid = [...qSet].sort((a, b) => a - b);
  const e_grid = [...eSet].sort((a, b) => a - b);
  const qIdx = new Map(q_grid.map((v, i) => [v, i]));
  const eIdx = new Map(e_grid.map((v, i) => [v, i]));
  const sqe = Array.from({ length: e_grid.length }, () => new Array(q_grid.length).fill(NaN));
  for (const [q, e, val] of pts) {
    sqe[eIdx.get(e)][qIdx.get(q)] = val;
  }
  return { name, format: 'triplets-2d', q_grid, e_grid, sqe };
}
