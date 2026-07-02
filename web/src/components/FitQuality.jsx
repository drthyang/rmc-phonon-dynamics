import React, { useState, useEffect } from 'react';
import { listSqgrConfigs, getConfigChannels, configRw } from '../io/sqgr';
import SciChart from './SciChart';

/**
 * Per-config R-value for the bar chart, averaged over every present data channel
 * (X-ray / neutron S(Q) & G(r), Bragg). Each bar's height is the mean Rw; the
 * per-channel breakdown is kept for the tooltip. Resilient: a single unreadable
 * CSV is skipped, not fatal.
 */
async function computeBars(ents, onProgress, isCancelled) {
  const bars = new Array(ents.length).fill(null);
  const batch = 24; let done = 0;
  for (let i = 0; i < ents.length; i += batch) {
    if (isCancelled()) return null;
    await Promise.all(ents.slice(i, i + batch).map(async (e, k) => {
      const { mean, parts } = await configRw(e);
      bars[i + k] = { rw: mean, parts };
      done++;
    }));
    onProgress(done, ents.length);
  }
  return bars;
}

// Compact "X S(Q) 3.2 · N G(r) 4.1" breakdown from a bar's per-channel parts.
const breakdown = (parts) => (parts || []).map(p => `${p.tag} ${p.rw.toFixed(1)}`).join(' · ');

/**
 * Fit-quality card (Cobalt redesign). Reads the selected RMCProfile output
 * folder, computes Rw per configuration, and drives the assessment:
 *   • an R-value bar chart (click a bar / type a config number to inspect),
 *   • one panel per available data channel — X-ray and neutron structure
 *     factor S(Q) & pair distribution G(r), plus Bragg — each as data
 *     (measured) vs fit (RMC model). Panels appear only for channels the
 *     dataset actually contains.
 * Computing Rw over the whole ensemble is heavy, so it runs on demand.
 */
const DIM = 'var(--dim)', FAINT = 'var(--faint)', INK = 'var(--ink)', ACCENT = 'var(--accent)', ACCENTINK = 'var(--accentInk)';

export default function FitQuality({ dirHandle, onFlagged, excludeBad, onExcludeChange }) {
  const [fitDir, setFitDir] = useState(null);    // folder scanned for RMCProfile fit CSVs
  const [entries, setEntries] = useState([]);
  const [bars, setBars] = useState([]);          // [{ rw }] aligned to entries
  const [sel, setSel] = useState(0);             // index into entries
  const [detail, setDetail] = useState(null);
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState(null);
  const [sigma, setSigma] = useState(2);         // flag configs worse than mean + sigma·std

  // Adopt the dataset folder by default; the user can point elsewhere (the
  // RMCProfile *_XFQ1.csv outputs sometimes live in a separate folder).
  useEffect(() => { setFitDir(dirHandle || null); }, [dirHandle]);

  // Reset, then auto-run the (heavy) fit check whenever the scanned folder
  // changes, so the R-value overview appears without an extra click.
  useEffect(() => {
    setEntries([]); setBars([]); setDetail(null); setSel(0); setError(null); setProgress(null);
    if (fitDir) { let cancelled = false; runCheck(fitDir, () => cancelled); return () => { cancelled = true; }; }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fitDir]);

  const pickFitDir = async () => {
    try { setFitDir(await window.showDirectoryPicker({ mode: 'read' })); }
    catch { /* user dismissed the picker */ }
  };

  const runCheck = async (dir = fitDir, isCancelled = () => false) => {
    if (!dir) { setError('Select a dataset folder first.'); return; }
    try {
      setError(null); setDetail(null);
      const ents = await listSqgrConfigs(dir);
      if (isCancelled()) return;
      setEntries(ents);
      if (!ents.length) { setError(`No RMCProfile fit CSVs (S(Q), G(r), Bragg) in “${dir.name}”. Pick the folder that holds the fit outputs.`); return; }
      setProgress({ done: 0, total: ents.length });
      const bs = await computeBars(ents, (done, total) => setProgress({ done, total }), isCancelled);
      if (!bs || isCancelled()) return;
      setBars(bs); setProgress(null);
      // Default to the best (lowest-Rw) configuration.
      let best = 0; for (let i = 1; i < bs.length; i++) if (bs[i].rw < bs[best].rw) best = i;
      selectConfig(best, ents);
    } catch (e) { console.error(e); setError(e.message); setProgress(null); }
  };

  const selectConfig = async (idx, ents = entries) => {
    if (idx < 0 || idx >= ents.length) return;
    setSel(idx);
    try { setDetail({ idx, channels: await getConfigChannels(ents[idx]) }); } catch (e) { setError(e.message); }
  };

  const nConfigs = entries.length;
  const maxRw = bars.reduce((m, b) => Math.max(m, b.rw), 0) || 1;
  const meanRw = bars.length ? bars.reduce((s, b) => s + b.rw, 0) / bars.length : 0;
  const stdRw = bars.length ? Math.sqrt(bars.reduce((s, b) => s + (b.rw - meanRw) ** 2, 0) / bars.length) : 0;
  const threshold = meanRw + sigma * stdRw;   // configs with Rw above this are flagged "bad"
  const worseCount = bars.reduce((n, b) => n + (b.rw > threshold ? 1 : 0), 0);
  const selBar = bars[sel];
  const selRw = selBar?.rw;
  // Configs above the mean + sigma·std threshold are flagged red; the rest teal.
  // When "exclude" is on, flagged configs are dropped from the run → show them
  // as transparent dimgray to make the exclusion visible.
  const rwColor = (rw) => {
    if (rw <= threshold) return 'hsla(168, 52%, 48%, 0.5)';
    if (excludeBad) return 'rgba(105, 105, 105, 0.32)';
    const t = Math.max(0, Math.min(1, (rw - threshold) / ((maxRw - threshold) || 1)));
    return `hsla(${Math.round(22 * (1 - t) + 6 * t)}, 82%, ${Math.round(56 - t * 10)}%, ${0.7 + t * 0.3})`;
  };
  // Pixel height of a bar (and the threshold guide line) within the 104px body.
  const barH = (rw) => Math.max(2, Math.round(6 + (rw / maxRw) * 92));

  // Report the flagged config numbers (+ sigma) up so the Run step can exclude them.
  useEffect(() => {
    if (!onFlagged) return;
    const cfgs = [];
    for (let i = 0; i < bars.length; i++) if (bars[i].rw > threshold) cfgs.push(entries[i].config);
    onFlagged(cfgs, sigma);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bars, sigma]);

  const title = (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
      <span style={{ font: "600 13px 'Space Grotesk'", letterSpacing: '.01em', color: INK }}>Fit quality</span>
      <span style={{ font: "11px 'Space Mono'", color: DIM }}>how well the model reproduces the measured data</span>
      <button onClick={pickFitDir} className="rnr-btn" title="scan a different folder for the RMCProfile fit outputs"
        style={{ marginLeft: 'auto', font: "11px 'Space Mono'", color: ACCENTINK, background: 'var(--soft)', border: 'none', borderRadius: 6, padding: '4px 10px', cursor: 'pointer' }}>
        {fitDir ? `folder: ${fitDir.name} · change` : 'choose fit folder…'}
      </button>
    </div>
  );

  // Pre-compute state: invite the (heavy) check.
  if (!bars.length) {
    return (
      <div className="rnr-card" style={{ padding: 18 }}>
        {title}
        <div style={{ background: 'var(--inset)', border: '1px solid var(--border)', borderRadius: 9, padding: '26px 16px', textAlign: 'center' }}>
          {progress ? (
            <>
              <div style={{ font: "12px 'Space Mono'", color: DIM, marginBottom: 10 }}>Computing Rw… {progress.done}/{progress.total}</div>
              <div style={{ height: 8, borderRadius: 5, background: 'var(--inset2)', overflow: 'hidden', maxWidth: 320, margin: '0 auto' }}>
                <div style={{ height: '100%', background: ACCENT, borderRadius: 5, width: `${100 * progress.done / progress.total}%`, transition: 'width .3s' }} />
              </div>
            </>
          ) : (
            <>
              <div style={{ font: "12px 'Spline Sans'", color: FAINT, marginBottom: 12, lineHeight: 1.6 }}>
                {error
                  ? <span style={{ color: 'var(--warnInk)' }}>{error}</span>
                  : fitDir
                    ? <>Scanning <b style={{ color: DIM }}>{fitDir.name}</b> for RMCProfile S(Q)/G(r)/Bragg outputs…</>
                    : 'Select a dataset folder, or choose the folder that holds the RMCProfile fit-output CSVs.'}
              </div>
              <div style={{ display: 'flex', gap: 8, justifyContent: 'center' }}>
                {fitDir && (
                  <button onClick={() => runCheck()} className="rnr-btn"
                    style={{ background: ACCENT, color: '#fff', border: 'none', borderRadius: 8, padding: '9px 18px', font: "600 13px 'Space Grotesk'", cursor: 'pointer' }}>
                    {error ? 'Retry' : 'Check fit quality'}
                  </button>
                )}
                <button onClick={pickFitDir} className="rnr-btn"
                  style={{ background: 'var(--inset2)', color: DIM, border: `1px solid var(--border)`, borderRadius: 8, padding: '9px 18px', font: "600 13px 'Space Grotesk'", cursor: 'pointer' }}>
                  Choose fit-output folder…
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    );
  }

  const channels = detail?.channels || [];
  return (
    <div className="rnr-card" style={{ padding: 18 }}>
      {title}

      {/* R-value overview */}
      <div style={{ marginBottom: 14 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8, gap: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, font: "11px 'Space Mono'", color: DIM, whiteSpace: 'nowrap' }}>
            <span>R-value per configuration <span style={{ color: FAINT }}>(click a bar to inspect)</span></span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>flag &gt;
              <select value={sigma} onChange={e => setSigma(+e.target.value)} title="flag configs whose Rw exceeds mean + Nσ"
                style={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 6, padding: '2px 4px', font: "12px 'Space Mono'", color: ACCENTINK, cursor: 'pointer' }}>
                {[1, 2, 3, 4, 5].map(v => <option key={v} value={v}>{v}</option>)}
              </select>σ
            </span>
            {onExcludeChange && (
              <label title="exclude the flagged configs from the calculation" style={{ display: 'flex', alignItems: 'center', gap: 5, cursor: 'pointer', color: excludeBad ? ACCENTINK : DIM }}>
                <input type="checkbox" checked={!!excludeBad} onChange={e => onExcludeChange(e.target.checked)} /> exclude
              </label>
            )}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, font: "11px 'Space Mono'", color: DIM }}>
            <span>config</span>
            <input type="number" min={1} max={nConfigs} value={sel + 1}
              onChange={(e) => { const v = parseInt(e.target.value); if (!isNaN(v)) selectConfig(Math.max(0, Math.min(nConfigs - 1, v - 1))); }}
              style={{ width: 74, boxSizing: 'border-box', background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 6, padding: '4px 8px', font: "13px 'Space Mono'", color: ACCENTINK, textAlign: 'center' }} />
            <span>/ {nConfigs}</span>
            {selRw != null && <span style={{ color: ACCENTINK, fontWeight: 700, marginLeft: 2 }}>Rw {selRw.toFixed(1)}%</span>}
            {selBar?.parts?.length > 0 &&
              <span style={{ color: FAINT, marginLeft: 2 }}>({breakdown(selBar.parts)})</span>}
          </div>
        </div>
        {/* Histogram framed like a plot: heat-mapped bars on a baseline, with a
            y-scale hint. Adaptive gap so hundreds of bars never collapse to 0px. */}
        <div style={{ position: 'relative', background: 'var(--inset)', border: `1px solid var(--border)`, borderRadius: 9, padding: '12px 12px 0' }}>
          <span style={{ position: 'absolute', top: 7, left: 12, font: "10px 'Space Mono'", color: FAINT }}>Rw {maxRw.toFixed(1)}%</span>
          <span style={{ position: 'absolute', top: 7, right: 12, font: "10px 'Space Mono'", color: FAINT }}>
            <span style={{ color: 'hsl(12,82%,50%)' }}>●</span> {worseCount} of {nConfigs} &gt; {sigma}σ
          </span>
          <div style={{ position: 'relative', display: 'flex', alignItems: 'flex-end', gap: nConfigs > 150 ? 1 : nConfigs > 60 ? 2 : 3, height: 104, overflow: 'hidden', borderBottom: `1.5px solid var(--dim)` }}>
            {/* threshold guide (mean + Nσ): bars rising above it are flagged */}
            <div style={{ position: 'absolute', left: 0, right: 0, bottom: Math.min(barH(threshold), 101), borderTop: '1px dashed var(--warnInk)', pointerEvents: 'none', zIndex: 2 }}>
              <span style={{ position: 'absolute', right: 0, top: -12, font: "9px 'Space Mono'", color: 'var(--warnInk)' }}>{sigma}σ · {threshold.toFixed(1)}%</span>
            </div>
            {bars.map((b, i) => {
              const selected = i === sel;
              const bd = breakdown(b.parts);
              return (
                <div key={i} className="rnr-bar" onClick={() => selectConfig(i)} title={`config ${entries[i].config} · Rw ${b.rw.toFixed(1)}%${bd ? ` (${bd})` : ''}`}
                  style={{ flex: 1, minWidth: 0, cursor: 'pointer', borderRadius: '2px 2px 0 0', height: barH(b.rw), background: selected ? 'var(--accent)' : rwColor(b.rw), outline: selected ? '1.5px solid var(--accentInk)' : 'none', outlineOffset: 0, zIndex: selected ? 3 : 0, position: 'relative' }} />
              );
            })}
          </div>
        </div>
      </div>

      {/* One panel per available data channel (X-ray / neutron S(Q) & G(r), Bragg).
          Flex-wrap: two per row on a wide card, stacking as channels increase. */}
      {channels.length === 0
        ? <Empty />
        : (
          <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
            {channels.map((ch) => {
              const f = buildFitSeries(ch.x, ch.expt, ch.rmc);
              return (
                <div key={ch.slot} style={{ flex: '1 1 340px', minWidth: 0, background: 'var(--inset)', border: '1px solid var(--border)', borderRadius: 9, padding: '13px 15px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                    <GroupChip group={ch.def.group} />
                    <span style={{ font: "600 13px 'Space Grotesk'", color: INK }}>{ch.def.name}</span>
                    {ch.sym && <span style={{ color: FAINT, font: "400 11px 'Space Mono'" }}>{ch.sym}</span>}
                    {ch.bank > 1 && <span style={{ color: FAINT, font: "400 10px 'Space Mono'" }}>bank {ch.bank}</span>}
                    {ch.rw != null && <span style={{ marginLeft: 'auto', color: ACCENTINK, font: "700 11px 'Space Mono'" }}>Rw {ch.rw.toFixed(1)}%</span>}
                  </div>
                  <SciChart xLabel={ch.xlabel} yLabel={ch.ylabel} height={300} series={f.series} baselines={f.baselines} resetKey={`${sel}-${ch.slot}`} />
                </div>
              );
            })}
          </div>
        )}
    </div>
  );
}

// Colored probe chip: X-ray blue, Neutron purple, Bragg amber.
const CHIP = {
  'X-ray': { bg: 'rgba(47,109,240,0.14)', fg: '#2f6df0' },
  'Neutron': { bg: 'rgba(140,92,246,0.16)', fg: '#7c4dd6' },
  'Bragg': { bg: 'rgba(224,150,40,0.18)', fg: '#b57414' },
};
function GroupChip({ group }) {
  const c = CHIP[group] || CHIP['X-ray'];
  return <span style={{ font: "700 9.5px 'Space Mono'", letterSpacing: '.06em', color: c.fg, background: c.bg, borderRadius: 5, padding: '2px 7px', textTransform: 'uppercase', flex: 'none' }}>{group}</span>;
}

/**
 * Build the RMC-style overlay for a fit panel: measured data as blue open
 * circles, the model fit as a red line, and the difference (data − fit) as a
 * green line offset below the data, with a dashed-grey baseline at its zero.
 */
function buildFitSeries(x, dataArr, fitArr) {
  let lo = Infinity, hi = -Infinity, damp = 0;
  const n = x.length, diff = new Array(n);
  for (let i = 0; i < n; i++) {
    const e = dataArr[i], r = fitArr[i];
    if (Number.isFinite(e)) { if (e < lo) lo = e; if (e > hi) hi = e; }
    if (Number.isFinite(r)) { if (r < lo) lo = r; if (r > hi) hi = r; }
    const d = (Number.isFinite(e) && Number.isFinite(r)) ? e - r : 0;
    diff[i] = d; const ad = Math.abs(d); if (ad > damp) damp = ad;
  }
  if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
  const span = (hi - lo) || 1;
  const base = lo - span * 0.12 - damp;   // diff zero sits a gap below the data
  return {
    series: [
      { name: 'data', color: '#2f6df0', marker: true, points: x.map((xi, i) => [xi, dataArr[i]]) },
      { name: 'fit', color: '#e0564b', width: 1.8, points: x.map((xi, i) => [xi, fitArr[i]]) },
      { name: 'diff', color: '#1f9d57', width: 1.3, points: x.map((xi, i) => [xi, base + diff[i]]), vy: diff },
    ],
    baselines: [base],
  };
}

function Empty() {
  return <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--faint)', font: "12px 'Spline Sans'" }}>no curve for this config</div>;
}
