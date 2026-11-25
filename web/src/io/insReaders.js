// web/src/io/insReaders.js
//
// Parsers for experimental 1D INS (GDOS / S(E)), 2D INS S(|Q|, E),
// and rmcph-mlip neutron-weighted DOS CSV files.

function splitRow(line) {
  const trimmed = line.trim();
  if (!trimmed) return [];
  if (trimmed.includes(',')) return trimmed.split(',').map(s => s.trim());
  return trimmed.split(/\s+/);
}

/**
 * Parse a 2- or 3-column 1D experimental INS file: E_meV, I [, sigma].
 * Skips comments (#, %, //) and header rows.
 */
export function parseExpIns1D(text, name = 'experimental_1d.dat') {
  const lines = text.split(/\r?\n/);
  const rows = [];
  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith('#') || line.startsWith('%') || line.startsWith('//')) continue;
    const parts = splitRow(line);
    if (parts.length < 2) continue;
    const e = Number(parts[0]);
    const y = Number(parts[1]);
    const s = parts.length >= 3 ? Number(parts[2]) : NaN;
    if (!Number.isFinite(e) || !Number.isFinite(y)) continue;
    rows.push([e, y, Number.isFinite(s) && s > 0 ? s : null]);
  }
  if (rows.length < 3) {
    throw new Error(`Could not parse ≥3 numeric (E, I) points from "${name}".`);
  }
  rows.sort((a, b) => a[0] - b[0]);
  const hasSigma = rows.every(r => r[2] !== null);
  return {
    name,
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

/**
 * Parse a 2D experimental S(|Q|, E) file.
 * Supports two common tabular formats:
 *   (A) Matrix CSV/TXT: first row = [label, Q0, Q1, ...], subsequent rows = [E_i, S(Q0,Ei), S(Q1,Ei), ...]
 *   (B) 3-column triplet table: Q, E, I (auto-detects E, Q, I if header starts with E)
 */
export function parseExpIns2D(text, name = 'experimental_2d.csv') {
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
        row[j] = Number.isFinite(v) ? v : 0;
      }
      e_grid.push(e);
      sqe.push(row);
    }
    if (e_grid.length >= 2 && q_grid.length >= 2) {
      return { name, q_grid, e_grid, sqe };
    }
  }

  // Format (B): 3-column (Q, E, I) or (E, Q, I)
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
    pts.push([qk, ek, val]);
  }
  if (pts.length < 4 || qSet.size < 2 || eSet.size < 2) {
    throw new Error(`Could not recognize "${name}" as a 2D S(|Q|, E) matrix or 3-column (Q, E, I) table.`);
  }

  const q_grid = [...qSet].sort((a, b) => a - b);
  const e_grid = [...eSet].sort((a, b) => a - b);
  const qIdx = new Map(q_grid.map((v, i) => [v, i]));
  const eIdx = new Map(e_grid.map((v, i) => [v, i]));
  const sqe = Array.from({ length: e_grid.length }, () => new Array(q_grid.length).fill(0));
  for (const [q, e, val] of pts) {
    sqe[eIdx.get(e)][qIdx.get(q)] = val;
  }
  return { name, q_grid, e_grid, sqe };
}
