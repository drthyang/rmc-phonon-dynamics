import React, { useState, useEffect } from 'react';
import { Activity } from 'lucide-react';
import { listSqgrConfigs, computeRwSummary, getSqgrData } from '../io/sqgr';
import LineChart from './LineChart';

const C_EXPT = '#22d3ee', C_RMC = '#f59e0b', C_DIFF = '#a78bfa';
const PAIR_COLORS = ['#60a5fa', '#f87171', '#34d399', '#fbbf24', '#c084fc', '#f472b6', '#2dd4bf', '#fb923c'];

/**
 * RMCProfile fit-quality check (port of rmcph_gui sqgr view). Operates on a
 * directory of RMCProfile output CSVs — by default the dataset folder already
 * selected in the runner (dirHandle prop) so it's a pre-run sanity check.
 * Computing Rw over the whole ensemble is heavy, so it runs on demand.
 */
export default function FitQuality({ dirHandle }) {
  const [entries, setEntries] = useState([]);
  const [rwPoints, setRwPoints] = useState([]);
  const [progress, setProgress] = useState(null);
  const [selConfig, setSelConfig] = useState(null);
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState(null);

  // Reset when the folder changes (don't auto-compute — it's expensive).
  useEffect(() => { setEntries([]); setRwPoints([]); setDetail(null); setSelConfig(null); setError(null); }, [dirHandle]);

  const runCheck = async () => {
    if (!dirHandle) { setError('Select a dataset folder first.'); return; }
    try {
      setError(null); setDetail(null); setSelConfig(null);
      const ents = await listSqgrConfigs(dirHandle);
      setEntries(ents);
      if (!ents.length) { setError('No *_XFQ1.csv RMCProfile files in this folder.'); return; }
      setProgress({ done: 0, total: ents.length });
      const pts = await computeRwSummary(ents, (done, total) => setProgress({ done, total }));
      setRwPoints(pts); setProgress(null);
      selectConfig(ents[ents.length - 1].config, ents);
    } catch (e) { console.error(e); setError(e.message); setProgress(null); }
  };

  const selectConfig = async (config, ents = entries) => {
    setSelConfig(config);
    const entry = ents.find(e => e.config === config);
    if (!entry) return;
    try { setDetail({ config, ...(await getSqgrData(entry)) }); } catch (e) { setError(e.message); }
  };

  const zip = (x, y) => x.map((xi, i) => [xi, y[i]]);
  const diff = (x, a, b) => x.map((xi, i) => [xi, a[i] - b[i]]);

  const overview = [];
  if (rwPoints.some(p => p.xfq != null)) overview.push({ name: 'Rw F(Q)', color: C_RMC, width: 1.6, points: rwPoints.filter(p => p.xfq != null).map(p => [p.config, p.xfq]) });
  if (rwPoints.some(p => p.xpdf != null)) overview.push({ name: 'Rw G(r)', color: C_EXPT, width: 1.6, points: rwPoints.filter(p => p.xpdf != null).map(p => [p.config, p.xpdf]) });

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3 flex-wrap">
        <div className="flex items-center gap-2 text-gray-200"><Activity className="w-5 h-5 text-indigo-400" /><h2 className="text-lg font-medium">Fit quality (pre-run check)</h2></div>
        <button onClick={runCheck} disabled={!dirHandle || !!progress}
          className={`px-4 py-1.5 rounded-lg text-sm font-medium ${(!dirHandle || progress) ? 'bg-white/10 text-gray-500' : 'bg-indigo-600 hover:bg-indigo-500'}`}>
          {progress ? 'Computing…' : 'Check fit quality'}
        </button>
        {!dirHandle && <span className="text-xs text-gray-500">select a dataset folder first</span>}
        {entries.length > 0 && !progress && <span className="text-xs text-gray-400">{entries.length} configs · click a point</span>}
      </div>

      {progress && (
        <div>
          <div className="text-xs text-gray-400 mb-1">Computing Rw… {progress.done}/{progress.total}</div>
          <div className="w-full bg-white/10 h-2 rounded-full overflow-hidden"><div className="h-full bg-blue-500" style={{ width: `${100 * progress.done / progress.total}%` }} /></div>
        </div>
      )}
      {error && <div className="text-amber-300 text-sm bg-amber-500/10 border border-amber-500/20 rounded px-3 py-2">{error}</div>}

      {overview.length > 0 && (
        <div className="bg-black/30 rounded-xl border border-white/5 p-3">
          <div className="text-sm text-gray-300 mb-1">Rw vs configuration {selConfig != null && <span className="text-gray-500">(selected: {selConfig})</span>}</div>
          <LineChart series={overview} xLabel="configuration" yLabel="Rw (%)" height={240} markerSeries={0} onPick={(_, x) => selectConfig(Math.round(x))} />
        </div>
      )}

      {detail && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
          {detail.xfq && <Panel title={`X-ray F(Q) — config ${detail.config}`}>
            <LineChart xLabel="Q (Å⁻¹)" yLabel="F(Q)" series={[
              { name: 'Expt', color: C_EXPT, points: zip(detail.xfq.x, detail.xfq.expt) },
              { name: 'RMC', color: C_RMC, points: zip(detail.xfq.x, detail.xfq.rmc) },
              { name: 'Diff', color: C_DIFF, dashed: true, points: diff(detail.xfq.x, detail.xfq.expt, detail.xfq.rmc) },
            ]} />
          </Panel>}
          {detail.xpdf && <Panel title={`X-ray G(r) — config ${detail.config}`}>
            <LineChart xLabel="r (Å)" yLabel="G(r)" series={[
              { name: 'Expt', color: C_EXPT, points: zip(detail.xpdf.x, detail.xpdf.expt) },
              { name: 'RMC', color: C_RMC, points: zip(detail.xpdf.x, detail.xpdf.rmc) },
              { name: 'Diff', color: C_DIFF, dashed: true, points: diff(detail.xpdf.x, detail.xpdf.expt, detail.xpdf.rmc) },
            ]} />
          </Panel>}
          {detail.partials && <Panel title={`G(r) partials — config ${detail.config}`} wide>
            <LineChart xLabel="r (Å)" yLabel="g(r)" series={detail.partials.pairs.map((p, i) => ({ name: p.name, color: PAIR_COLORS[i % PAIR_COLORS.length], points: zip(detail.partials.x, p.y) }))} />
          </Panel>}
        </div>
      )}
    </div>
  );
}

function Panel({ title, children, wide }) {
  return (
    <div className={`bg-black/30 rounded-xl border border-white/5 p-3 ${wide ? 'xl:col-span-2' : ''}`}>
      <div className="text-sm text-gray-300 mb-1">{title}</div>
      {children}
    </div>
  );
}
