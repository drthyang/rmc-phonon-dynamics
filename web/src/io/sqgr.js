// web/src/io/sqgr.js
//
// RMCProfile fit-quality reader (port of rmcph_gui backend/api/sqgr.py).
// Reads per-config RMCProfile output CSVs from a folder:
//   {stem}_{N}_XFQ1.csv       — X-ray F(Q):  Q, F(Q)_RMC, F(Q)_Expt
//   {stem}_{N}_FT_XFQ1.csv    — X-ray G(r):  r, G(r)_RMC, G(r)_Expt
//   {stem}_{N}_PDFpartials.csv — G(r) partial pairs: r, A-A, A-B, ...
// and computes the weighted R-factor Rw = sqrt(Σ(exp-calc)²/Σexp²)·100.

const XFQ1_RE = /^(.+)_(\d+)_XFQ1\.csv$/i;
const FT_RE = /^(.+)_(\d+)_FT_XFQ1\.csv$/i;
const PART_RE = /^(.+)_(\d+)_PDFpartials\.csv$/i;

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

/** Scan the directory once and group file handles by config number. */
export async function listSqgrConfigs(dirHandle) {
  const byCfg = new Map();
  const get = (cfg, stem) => {
    if (!byCfg.has(cfg)) byCfg.set(cfg, { config: cfg, stem });
    return byCfg.get(cfg);
  };
  for await (const entry of dirHandle.values()) {
    if (entry.kind !== 'file') continue;
    const name = entry.name;
    if (/_FT_XFQ1\.csv$/i.test(name)) { const m = name.match(FT_RE); if (m) get(+m[2], m[1]).ft = entry; continue; }
    if (PART_RE.test(name)) { const m = name.match(PART_RE); if (m) get(+m[2], m[1]).partials = entry; continue; }
    if (XFQ1_RE.test(name)) { const m = name.match(XFQ1_RE); if (m) { const e = get(+m[2], m[1]); e.xfq = entry; e.stem = m[1]; } }
  }
  return [...byCfg.values()].filter(e => e.xfq).sort((a, b) => a.config - b.config);
}

async function readCols(handle) {
  if (!handle) return null;
  const file = await handle.getFile();
  return parseCsv(await file.text()).cols;
}

/** Per-config Rw for F(Q) and G(r), read in parallel batches with progress. */
export async function computeRwSummary(entries, onProgress) {
  const points = new Array(entries.length);
  const batch = 24;
  let done = 0;
  for (let i = 0; i < entries.length; i += batch) {
    const slice = entries.slice(i, i + batch);
    await Promise.all(slice.map(async (e, k) => {
      const [xfqCols, ftCols] = await Promise.all([readCols(e.xfq), readCols(e.ft)]);
      points[i + k] = {
        config: e.config,
        xfq: xfqCols ? rwFromCols(xfqCols) : null,
        xpdf: ftCols ? rwFromCols(ftCols) : null,
      };
      done++;
    }));
    if (onProgress) onProgress(done, entries.length);
  }
  return points;
}

/** Full F(Q), G(r) and partials curves for one config entry. */
export async function getSqgrData(entry) {
  const out = {};
  const xfq = await readCols(entry.xfq);
  if (xfq && xfq.length >= 3) out.xfq = { x: xfq[0], rmc: xfq[1], expt: xfq[2] };
  const ft = await readCols(entry.ft);
  if (ft && ft.length >= 3) out.xpdf = { x: ft[0], rmc: ft[1], expt: ft[2] };
  if (entry.partials) {
    const file = await entry.partials.getFile();
    const { headers, cols } = parseCsv(await file.text());
    out.partials = { x: cols[0], pairs: headers.slice(1).map((h, i) => ({ name: h, y: cols[i + 1] })).filter(p => p.y) };
  }
  return out;
}
