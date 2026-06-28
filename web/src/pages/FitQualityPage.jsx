import React, { useState } from 'react';
import { FolderOpen, Activity } from 'lucide-react';
import { listSqgrConfigs, computeRwSummary, getSqgrData } from '../io/sqgr';
import LineChart from '../components/LineChart';

const C_EXPT = '#22d3ee', C_RMC = '#f59e0b', C_DIFF = '#a78bfa';
const PAIR_COLORS = ['#60a5fa', '#f87171', '#34d399', '#fbbf24', '#c084fc', '#f472b6', '#2dd4bf', '#fb923c'];

/**
 * RMCProfile fit-quality preview (port of rmcph_gui sqgr view). Pick a folder of
 * RMCProfile output CSVs; see Rw-vs-configuration for X-ray F(Q) and G(r), then
 * click a configuration to inspect observed / RMC / difference curves + partials.
 */
export default function FitQualityPage() {
  const [entries, setEntries] = useState([]);
  const [rwPoints, setRwPoints] = useState([]);
  const [progress, setProgress] = useState(null);
  const [selConfig, setSelConfig] = useState(null);
  const [detail, setDetail] = useState(null);
  const [dirName, setDirName] = useState(null);
  const [error, setError] = useState(null);

  const pickFolder = async () => {
    try {
      const dh = await window.showDirectoryPicker({ mode: 'read' });
      setDirName(dh.name); setError(null); setDetail(null); setSelConfig(null); setRwPoints([]);
      const ents = await listSqgrConfigs(dh);
      setEntries(ents);
      if (!ents.length) { setError('No *_XFQ1.csv RMCProfile files found in this folder.'); return; }
      setProgress({ done: 0, total: ents.length });
      const pts = await computeRwSummary(ents, (done, total) => setProgress({ done, total }));
      setRwPoints(pts);
      setProgress(null);
      selectConfig(ents[ents.length - 1].config, ents); // default: last config
    } catch (e) { console.error(e); setError(e.message); }
  };

  const selectConfig = async (config, ents = entries) => {
    setSelConfig(config);
    const entry = ents.find(e => e.config === config);
    if (!entry) return;
    try { setDetail({ config, ...(await getSqgrData(entry)) }); }
    catch (e) { setError(e.message); }
  };

  const zip = (x, y) => x.map((xi, i) => [xi, y[i]]);
  const diff = (x, a, b) => x.map((xi, i) => [xi, a[i] - b[i]]);

  const overviewSeries = [];
  if (rwPoints.some(p => p.xfq != null)) overviewSeries.push({ name: 'Rw F(Q)', color: C_RMC, width: 1.6, points: rwPoints.filter(p => p.xfq != null).map(p => [p.config, p.xfq]) });
  if (rwPoints.some(p => p.xpdf != null)) overviewSeries.push({ name: 'Rw G(r)', color: C_EXPT, width: 1.6, points: rwPoints.filter(p => p.xpdf != null).map(p => [p.config, p.xpdf]) });

  return (
    <div className="flex flex-col gap-5">
      <div className="glass-panel rounded-2xl p-6">
        <div className="flex items-center gap-3 mb-4 text-gray-200"><Activity className="w-5 h-5 text-indigo-400" /><h2 className="text-lg font-medium">RMCProfile fit quality</h2></div>
        <button onClick={pickFolder} className="flex items-center gap-2 bg-white/5 hover:bg-white/10 border border-white/10 py-2.5 px-4 rounded-xl font-medium">
          <FolderOpen className="w-5 h-5" />{dirName ? `Folder: ${dirName}` : 'Select RMCProfile output folder'}
        </button>
        {progress && (
          <div className="mt-4">
            <div className="text-xs text-gray-400 mb-1">Computing Rw… {progress.done}/{progress.total}</div>
            <div className="w-full bg-white/10 h-2 rounded-full overflow-hidden"><div className="h-full bg-blue-500" style={{ width: `${100 * progress.done / progress.total}%` }} /></div>
          </div>
        )}
        {error && <div className="mt-3 text-amber-300 text-sm bg-amber-500/10 border border-amber-500/20 rounded px-3 py-2">{error}</div>}
        {entries.length > 0 && !progress && <div className="mt-3 text-xs text-gray-400">{entries.length} configurations · click a point to inspect</div>}
      </div>

      {overviewSeries.length > 0 && (
        <div className="glass-panel rounded-2xl p-4">
          <div className="text-sm text-gray-300 mb-1">Rw vs configuration {selConfig != null && <span className="text-gray-500">(selected: {selConfig})</span>}</div>
          <LineChart series={overviewSeries} xLabel="configuration" yLabel="Rw (%)" height={260} markerSeries={0}
            onPick={(_, x) => selectConfig(Math.round(x))} />
        </div>
      )}

      {detail && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
          {detail.xfq && (
            <Panel title={`X-ray F(Q) — config ${detail.config}`}>
              <LineChart xLabel="Q (Å⁻¹)" yLabel="F(Q)" series={[
                { name: 'Expt', color: C_EXPT, points: zip(detail.xfq.x, detail.xfq.expt) },
                { name: 'RMC', color: C_RMC, points: zip(detail.xfq.x, detail.xfq.rmc) },
                { name: 'Diff', color: C_DIFF, dashed: true, points: diff(detail.xfq.x, detail.xfq.expt, detail.xfq.rmc) },
              ]} />
            </Panel>
          )}
          {detail.xpdf && (
            <Panel title={`X-ray G(r) — config ${detail.config}`}>
              <LineChart xLabel="r (Å)" yLabel="G(r)" series={[
                { name: 'Expt', color: C_EXPT, points: zip(detail.xpdf.x, detail.xpdf.expt) },
                { name: 'RMC', color: C_RMC, points: zip(detail.xpdf.x, detail.xpdf.rmc) },
                { name: 'Diff', color: C_DIFF, dashed: true, points: diff(detail.xpdf.x, detail.xpdf.expt, detail.xpdf.rmc) },
              ]} />
            </Panel>
          )}
          {detail.partials && (
            <Panel title={`G(r) partials — config ${detail.config}`} wide>
              <LineChart xLabel="r (Å)" yLabel="g(r)" series={detail.partials.pairs.map((p, i) => ({ name: p.name, color: PAIR_COLORS[i % PAIR_COLORS.length], points: zip(detail.partials.x, p.y) }))} />
            </Panel>
          )}
        </div>
      )}
    </div>
  );
}

function Panel({ title, children, wide }) {
  return (
    <div className={`glass-panel rounded-2xl p-4 ${wide ? 'xl:col-span-2' : ''}`}>
      <div className="text-sm text-gray-300 mb-1">{title}</div>
      {children}
    </div>
  );
}
