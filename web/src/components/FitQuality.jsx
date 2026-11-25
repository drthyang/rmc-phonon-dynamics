import React, { useState, useEffect } from 'react';
import { listSqgrConfigs, getConfigChannels, configRw } from '../io/sqgr';
import SciChart from './SciChart';

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

const breakdown = (parts) => (parts || []).map(p => `${p.tag} ${p.rw.toFixed(1)}`).join(' · ');

export default function FitQuality({ dirHandle, onFlagged, excludeBad, onExcludeChange }) {
  const [fitDir, setFitDir] = useState(null);
  const [entries, setEntries] = useState([]);
  const [bars, setBars] = useState([]);
  const [sel, setSel] = useState(0);
  const [detail, setDetail] = useState(null);
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState(null);
  const [sigma, setSigma] = useState(2);

  useEffect(() => { setFitDir(dirHandle || null); }, [dirHandle]);

  useEffect(() => {
    setEntries([]); setBars([]); setDetail(null); setSel(0); setError(null); setProgress(null);
    if (fitDir) {
      let cancelled = false;
      runCheck(fitDir, () => cancelled);
      return () => { cancelled = true; };
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fitDir]);

  const pickFitDir = async () => {
    try { setFitDir(await window.showDirectoryPicker({ mode: 'read' })); }
    catch { /* user dismissed picker */ }
  };

  const runCheck = async (dir = fitDir, isCancelled = () => false) => {
    if (!dir) { setError('Select a dataset folder first.'); return; }
    try {
      setError(null); setDetail(null);
      const ents = await listSqgrConfigs(dir);
      if (isCancelled()) return;
      setEntries(ents);
      if (!ents.length) {
        setError(`No RMCProfile fit CSVs (S(Q), G(r), Bragg) in “${dir.name}”.`);
        return;
      }
      setProgress({ done: 0, total: ents.length });
      const bs = await computeBars(ents, (done, total) => setProgress({ done, total }), isCancelled);
      if (!bs || isCancelled()) return;
      setBars(bs); setProgress(null);
      let best = 0;
      for (let i = 1; i < bs.length; i++) if (bs[i].rw < bs[best].rw) best = i;
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
  const threshold = meanRw + sigma * stdRw;
  const worseCount = bars.reduce((n, b) => n + (b.rw > threshold ? 1 : 0), 0);
  const selBar = bars[sel];
  const selRw = selBar?.rw;

  const rwColor = (rw) => {
    if (rw <= threshold) return 'hsla(168, 48%, 45%, 0.62)';
    if (excludeBad) return 'rgba(154, 149, 141, 0.35)';
    const t = Math.max(0, Math.min(1, (rw - threshold) / ((maxRw - threshold) || 1)));
    return `hsla(${Math.round(18 * (1 - t) + 4 * t)}, 78%, ${Math.round(54 - t * 10)}%, ${0.75 + t * 0.25})`;
  };
  const barH = (rw) => Math.max(3, Math.round(6 + (rw / maxRw) * 92));

  useEffect(() => {
    if (!onFlagged) return;
    const cfgs = [];
    for (let i = 0; i < bars.length && i < entries.length; i++) if (bars[i].rw > threshold) cfgs.push(entries[i].config);
    onFlagged(cfgs, sigma);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bars, entries, sigma]);

  const header = (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', marginBottom: 12 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, flexWrap: 'wrap' }}>
        <span style={{ fontWeight: 600, fontSize: 'var(--fz-title)' }}>Fit quality</span>
        <span style={{ fontSize: 'var(--fz-small)', color: 'var(--secondary)' }}>
          agreement between RMC configurations and measured scattering data
        </span>
      </div>
      <button type="button" onClick={pickFitDir} className="m-btn subtle sm">
        {fitDir ? `${fitDir.name}/ · change` : 'Choose fit folder…'}
      </button>
    </div>
  );

  if (!bars.length) {
    return (
      <div className="m-frame" style={{ padding: 16 }}>
        {header}
        <div className="m-rule-box" style={{ padding: '20px 16px', textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
          {progress ? (
            <>
              <div className="tabular-nums" style={{ fontSize: 'var(--fz-small)', fontFamily: 'var(--font-mono)', color: 'var(--secondary)' }}>
                Computing Rw… {progress.done}/{progress.total}
              </div>
              <div style={{ height: 6, borderRadius: 3, background: 'var(--muted2)', overflow: 'hidden', width: '100%', maxWidth: 300 }}>
                <div style={{ height: '100%', background: 'var(--ink)', width: `${100 * progress.done / progress.total}%`, transition: 'width 0.2s' }} />
              </div>
            </>
          ) : (
            <>
              <div style={{ fontSize: 'var(--fz-control)', color: error ? 'var(--warn-ink)' : 'var(--secondary)' }}>
                {error
                  ? error
                  : fitDir
                    ? <>Scanning <b style={{ color: 'var(--ink)' }}>{fitDir.name}</b> for RMCProfile S(Q), G(r), and Bragg CSVs…</>
                    : 'Open an ensemble folder above, or point to the folder holding RMCProfile fit CSVs.'}
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                {fitDir && (
                  <button type="button" onClick={() => runCheck()} className="m-btn sm">
                    {error ? 'Retry' : 'Check fit quality'}
                  </button>
                )}
                <button type="button" onClick={pickFitDir} className="m-btn subtle sm">
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
    <div className="m-frame" style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 14 }}>
      {header}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, flexWrap: 'wrap', fontSize: 'var(--fz-small)', color: 'var(--secondary)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <span>R-value per config <span style={{ color: 'var(--faint)' }}>(click bar to inspect)</span></span>
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
              <span>flag &gt;</span>
              <select
                value={sigma}
                onChange={e => setSigma(+e.target.value)}
                className="m-select mono"
                style={{ height: 24, padding: '0 6px' }}
              >
                {[1, 2, 3, 4, 5].map(v => <option key={v} value={v}>{v}</option>)}
              </select>
              <span>σ</span>
            </label>
            {onExcludeChange && (
              <label style={{ display: 'inline-flex', alignItems: 'center', gap: 5, cursor: 'pointer', color: excludeBad ? 'var(--ink)' : 'var(--secondary)', fontWeight: excludeBad ? 600 : 400 }}>
                <input type="checkbox" checked={!!excludeBad} onChange={e => onExcludeChange(e.target.checked)} />
                exclude flagged
              </label>
            )}
          </div>

          <div className="tabular-nums" style={{ display: 'inline-flex', alignItems: 'center', gap: 8, fontFamily: 'var(--font-mono)' }}>
            <span>config</span>
            <input
              type="number"
              min={1}
              max={nConfigs}
              value={sel + 1}
              onChange={(e) => {
                const v = parseInt(e.target.value, 10);
                if (!isNaN(v)) selectConfig(Math.max(0, Math.min(nConfigs - 1, v - 1)));
              }}
              className="m-input mono"
              style={{ width: 64, height: 26, textAlign: 'center', fontWeight: 600 }}
            />
            <span>/ {nConfigs}</span>
            {selRw != null && <b style={{ color: 'var(--ink)' }}>Rw {selRw.toFixed(1)}%</b>}
            {selBar?.parts?.length > 0 && <span style={{ color: 'var(--faint)' }}>({breakdown(selBar.parts)})</span>}
          </div>
        </div>

        <div style={{ position: 'relative', background: 'var(--muted)', border: '1px solid var(--line)', borderRadius: 6, padding: '24px 12px 0' }}>
          <span className="tabular-nums" style={{ position: 'absolute', top: 6, left: 12, fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--faint)' }}>
            max {maxRw.toFixed(1)}% · mean {meanRw.toFixed(1)}%
          </span>
          <span className="tabular-nums" style={{ position: 'absolute', top: 6, right: 12, fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--faint)' }}>
            <span style={{ color: 'var(--warn-ink)' }}>●</span> {worseCount} of {nConfigs} &gt; {sigma}σ
          </span>

          <div style={{ position: 'relative', display: 'flex', alignItems: 'flex-end', gap: nConfigs > 150 ? 1 : nConfigs > 60 ? 2 : 3, height: 102, overflow: 'hidden', borderBottom: '1px solid var(--ink)' }}>
            <div style={{ position: 'absolute', left: 0, right: 0, bottom: Math.min(barH(threshold), 99), borderTop: '1px dashed var(--warn-ink)', pointerEvents: 'none', zIndex: 2 }}>
              <span style={{ position: 'absolute', right: 0, top: -13, fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--warn-ink)' }}>
                {sigma}σ · {threshold.toFixed(1)}%
              </span>
            </div>
            {bars.map((b, i) => {
              const selected = i === sel;
              const bd = breakdown(b.parts);
              return (
                <div
                  key={i}
                  className="rnr-bar"
                  onClick={() => selectConfig(i)}
                  title={`config ${entries[i].config} · Rw ${b.rw.toFixed(1)}%${bd ? ` (${bd})` : ''}`}
                  style={{
                    flex: 1,
                    minWidth: 0,
                    cursor: 'pointer',
                    borderRadius: '2px 2px 0 0',
                    height: barH(b.rw),
                    background: selected ? 'var(--ink)' : rwColor(b.rw),
                    zIndex: selected ? 3 : 0,
                    position: 'relative',
                  }}
                />
              );
            })}
          </div>
        </div>
      </div>

      {channels.length === 0 ? (
        <div style={{ height: 240, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--faint)', fontSize: 'var(--fz-control)' }}>
          No curves for this configuration.
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: 14 }}>
          {channels.map((ch) => {
            const f = buildFitSeries(ch.x, ch.expt, ch.rmc);
            return (
              <div key={ch.slot} className="m-frame" style={{ padding: '12px 14px' }}>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 6 }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, fontWeight: 600, padding: '1px 6px', borderRadius: 4, background: 'var(--muted)', color: 'var(--ink)' }}>
                    {ch.def.group}
                  </span>
                  <span style={{ fontWeight: 600, fontSize: 'var(--fz-control)' }}>{ch.def.name}</span>
                  {ch.sym && <span style={{ color: 'var(--faint)', fontFamily: 'var(--font-mono)', fontSize: 11 }}>{ch.sym}</span>}
                  {ch.bank > 1 && <span style={{ color: 'var(--faint)', fontFamily: 'var(--font-mono)', fontSize: 11 }}>bank {ch.bank}</span>}
                  {ch.rw != null && (
                    <span className="tabular-nums" style={{ marginLeft: 'auto', fontFamily: 'var(--font-mono)', fontSize: 11.5, fontWeight: 600, color: 'var(--ink)' }}>
                      Rw {ch.rw.toFixed(1)}%
                    </span>
                  )}
                </div>
                <SciChart xLabel={ch.xlabel} yLabel={ch.ylabel} height={280} series={f.series} baselines={f.baselines} resetKey={`${sel}-${ch.slot}`} />
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function buildFitSeries(x, dataArr, fitArr) {
  let lo = Infinity, hi = -Infinity, damp = 0;
  const n = x.length, diff = new Array(n);
  for (let i = 0; i < n; i++) {
    const e = dataArr[i], r = fitArr[i];
    if (Number.isFinite(e)) { if (e < lo) lo = e; if (e > hi) hi = e; }
    if (Number.isFinite(r)) { if (r < lo) lo = r; if (r > hi) hi = r; }
    const d = (Number.isFinite(e) && Number.isFinite(r)) ? e - r : 0;
    diff[i] = d;
    const ad = Math.abs(d);
    if (ad > damp) damp = ad;
  }
  if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
  const span = (hi - lo) || 1;
  const base = lo - span * 0.12 - damp;
  return {
    series: [
      { name: 'data', color: '#2563eb', marker: true, points: x.map((xi, i) => [xi, dataArr[i]]) },
      { name: 'fit', color: '#dc2626', width: 1.8, points: x.map((xi, i) => [xi, fitArr[i]]) },
      { name: 'diff', color: '#15803d', width: 1.3, points: x.map((xi, i) => [xi, base + diff[i]]), vy: diff },
    ],
    baselines: [base],
  };
}
