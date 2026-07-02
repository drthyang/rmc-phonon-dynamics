// web/src/io/sqgr.js
//
// RMCProfile fit-quality reader (comprehensive port of the rmc-toolkits readers,
// github.com/drthyang/rmc-toolkits — rmc_toolkits/parsers.py + plots.py).
//
// Reads the per-configuration RMCProfile output CSVs from a folder and groups
// them by config number N and by DATA CHANNEL (probe × space):
//
//   X-ray   S(Q)/F(Q) : {stem}_{N}_XFQ{b}.csv   (or {stem}_{N}_FQ{b}.csv, the
//                        same X-ray reciprocal data written at lower precision)
//   X-ray   G(r)       : {stem}_{N}_FT_XFQ{b}.csv
//   Neutron S(Q)       : {stem}_{N}_SQ{b}.csv
//   Neutron G(r)       : {stem}_{N}_PDF{b}.csv
//   Bragg              : {stem}_{N}_bragg.csv     (Q- or ToF-axis, header-detected)
//   partials (no fit)  : {stem}_{N}_PDFpartials.csv, {stem}_{N}_FQ{b}partials.csv
//
// Each fit channel is data (measured) vs calc (RMC model); the weighted R-factor
// is Rw = sqrt(Σ(exp−calc)²/Σexp²)·100  (calc = column 2, exp = column 3).
// b is a bank index (1 by default); datasets with >1 bank produce one channel
// each. Previously only the two X-ray channels were read — neutron S(Q)/G(r) and
// Bragg are now first-class and appear automatically when a dataset has them.

/* ── channel table ──────────────────────────────────────────────────────────
 * Order = display order. `prio` breaks ties within a channel key: the X-ray
 * reciprocal is emitted as both _XFQ (full precision) and _FQ (7 sig figs) — the
 * same data — so _XFQ (prio 0) wins over _FQ (prio 1). The capture groups are
 * (stem, config, bank); bank defaults to 1 when the group is empty.
 * `_FT_XFQ`, `…partials` etc. cannot match the plain _XFQ/_FQ/_PDF patterns
 * (verified: an intervening token or missing digits blocks them). */
const CHANNELS = [
  { key: 'xgr', order: 1, prio: 0, group: 'X-ray',   quantity: 'G(r)', name: 'Pair distribution', sym: 'G(r)',
    xlabel: 'r (Å)', ylabel: 'G(r) (Å⁻²)', re: /^(.+?)_(\d+)_FT_XFQ(\d*)\.csv$/i },
  { key: 'xsq', order: 0, prio: 0, group: 'X-ray',   quantity: 'S(Q)', name: 'Structure factor', sym: 'F(Q)',
    xlabel: 'Q (Å⁻¹)', ylabel: 'F(Q)', re: /^(.+?)_(\d+)_XFQ(\d*)\.csv$/i },
  { key: 'xsq', order: 0, prio: 1, group: 'X-ray',   quantity: 'S(Q)', name: 'Structure factor', sym: 'F(Q)',
    xlabel: 'Q (Å⁻¹)', ylabel: 'F(Q)', re: /^(.+?)_(\d+)_FQ(\d*)\.csv$/i },
  { key: 'nsq', order: 2, prio: 0, group: 'Neutron', quantity: 'S(Q)', name: 'Structure factor', sym: 'S(Q)',
    xlabel: 'Q (Å⁻¹)', ylabel: 'S(Q)', re: /^(.+?)_(\d+)_SQ(\d*)\.csv$/i },
  { key: 'ngr', order: 3, prio: 0, group: 'Neutron', quantity: 'G(r)', name: 'Pair distribution', sym: 'G(r)',
    xlabel: 'r (Å)', ylabel: 'G(r) (Å⁻²)', re: /^(.+?)_(\d+)_PDF(\d+)\.csv$/i },
  { key: 'bragg', order: 4, prio: 0, group: 'Bragg', quantity: 'Bragg', name: 'Bragg profile', sym: '',
    xlabel: 'Q (Å⁻¹)', ylabel: 'I', re: /^(.+?)_(\d+)_bragg\.csv$/i },
];

// Short label for the R-value breakdown (bar tooltip / selected-config readout).
const GROUP_SHORT = { 'X-ray': 'X', 'Neutron': 'N', 'Bragg': '' };
export function channelTag(def, bank = 1) {
  const g = GROUP_SHORT[def.group];
  const base = def.group === 'Bragg' ? 'Bragg' : `${g} ${def.quantity}`;
  return bank > 1 ? `${base}·${bank}` : base;
}
export function channelTitle(def, bank = 1) {
  const base = def.group === 'Bragg' ? 'Bragg profile' : `${def.group} · ${def.name}`;
  return bank > 1 ? `${base} (bank ${bank})` : base;
}

/** RMCProfile time-of-flight Bragg CSVs label column 0 "Flight time (µs)"; those
 *  use a ToF (µs) axis. Anything else (e.g. "Q or theta") stays a Q axis. */
const isTofHeader = (h) => /tof|flight|time/i.test(h || '');

/** Parse a whitespace-padded CSV into { headers:[...], cols:[[...], ...] }. */
export function parseCsv(text) {
  const lines = text.split('\n').filter(l => l.trim().length);
  if (!lines.length) return { headers: [], cols: [] };
  const headers = lines[0].split(',').map(h => h.trim());
  const cols = headers.map(() => []);
  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].split(',');
    for (let c = 0; c < headers.length; c++) {
      const v = parseFloat((parts[c] || '').trim());
      cols[c].push(Number.isFinite(v) ? v : NaN);
    }
  }
  return { headers, cols };
}

/** Rw = sqrt(Σ(exp-calc)²/Σexp²)·100, with calc=col[1], exp=col[2]. */
export function rwFromCols(cols) {
  if (cols.length < 3) return null;
  const calc = cols[1], exp = cols[2];
  let ss = 0, se = 0;
  const n = Math.min(calc.length, exp.length);
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(calc[i]) || !Number.isFinite(exp[i])) continue;
    const d = exp[i] - calc[i];
    ss += d * d; se += exp[i] * exp[i];
  }
  if (n === 0 || se <= 0) return null;
  return Math.sqrt(ss / se) * 100;
}

async function readParsed(handle) {
  if (!handle) return null;
  try { return parseCsv(await (await handle.getFile()).text()); }
  catch { return null; }
}
async function readCols(handle) { const p = await readParsed(handle); return p ? p.cols : null; }

/**
 * Scan the directory once, grouping file handles by config number and channel.
 * Returns [{ config, stem, channels: { slot: { def, bank, handle } } }] sorted by
 * config, keeping only configs that have at least one fit channel. `slot` is the
 * channel key plus a #bank suffix when bank > 1, so multi-bank data stays distinct.
 */
export async function listSqgrConfigs(dirHandle) {
  const byCfg = new Map();
  for await (const entry of dirHandle.values()) {
    if (entry.kind !== 'file') continue;
    for (const def of CHANNELS) {
      const m = entry.name.match(def.re);
      if (!m) continue;
      const cfg = +m[2], bank = m[3] ? +m[3] : 1;
      if (!byCfg.has(cfg)) byCfg.set(cfg, { config: cfg, stem: m[1], channels: {} });
      const rec = byCfg.get(cfg);
      const slot = def.key + (bank > 1 ? `#${bank}` : '');
      const cur = rec.channels[slot];
      if (!cur || def.prio < cur.def.prio) rec.channels[slot] = { def, bank, handle: entry };
      break;   // first matching pattern classifies this file
    }
  }
  return [...byCfg.values()]
    .filter(e => Object.keys(e.channels).length)
    .sort((a, b) => a.config - b.config);
}

/** Present channel slots for a config, in display order (channel order, then bank). */
export function presentSlots(entry) {
  return Object.keys(entry.channels).sort((a, b) => {
    const A = entry.channels[a], B = entry.channels[b];
    return A.def.order - B.def.order || A.bank - B.bank;
  });
}

/**
 * Per-config R-value summary across every present channel:
 *   { mean, parts: [{ tag, rw, group, quantity, bank }] }
 * `mean` (the bar height) is the average of the available channel Rw values.
 * Reads only what Rw needs; a single unreadable CSV is skipped, not fatal.
 */
export async function configRw(entry) {
  const slots = presentSlots(entry);
  const parts = [];
  for (const s of slots) {
    const { def, bank, handle } = entry.channels[s];
    const cols = await readCols(handle);
    const rw = cols ? rwFromCols(cols) : null;
    if (rw != null) parts.push({ slot: s, tag: channelTag(def, bank), rw, group: def.group, quantity: def.quantity, bank });
  }
  const mean = parts.length ? parts.reduce((a, p) => a + p.rw, 0) / parts.length : 0;
  return { mean, parts };
}

/**
 * Full curves for one config: one entry per present channel, in display order.
 *   [{ slot, def, bank, title, tag, xlabel, ylabel, x, rmc, expt, rw }]
 * The Bragg x-axis label switches to ToF (µs) when the file header is time-of-flight.
 */
export async function getConfigChannels(entry) {
  const out = [];
  for (const s of presentSlots(entry)) {
    const { def, bank, handle } = entry.channels[s];
    const parsed = await readParsed(handle);
    if (!parsed || parsed.cols.length < 3) continue;
    const { headers, cols } = parsed;
    const xlabel = (def.key === 'bragg' && isTofHeader(headers[0])) ? 'ToF (µs)' : def.xlabel;
    out.push({
      slot: s, def, bank, title: channelTitle(def, bank), tag: channelTag(def, bank),
      xlabel, ylabel: def.ylabel, sym: def.sym,
      x: cols[0], rmc: cols[1], expt: cols[2], rw: rwFromCols(cols),
    });
  }
  return out;
}
