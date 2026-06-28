import React, { useState, useMemo } from 'react';
import { BarChart3 } from 'lucide-react';
import { phononDOS } from '../math/dos';
import LineChart from './LineChart';

/**
 * Phonon DOS over a uniform q-grid (proper PhDOS, vs the band-path approximation
 * in the viewer). Runs the WebGPU S(k)->eigh over an n³ Γ-centered grid via the
 * pipeline (reusing the parsed-config cache), then Gaussian-broadens all the
 * eigenvalues. Grid eigenvalues are cached so σ / energy range re-histogram
 * instantly without recomputing.
 */
export default function PhononDOS({ pipeline, files, family, baseStructure, temperature, referenceHandle }) {
  const [gridN, setGridN] = useState(6);
  const [sigma, setSigma] = useState(1.0);
  const [energies, setEnergies] = useState(null);     // Float64Array of grid eigenvalues
  const [info, setInfo] = useState(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('');

  const Emax = useMemo(() => {
    if (!energies) return 50;
    const v = Array.from(energies).filter(x => Number.isFinite(x) && x > 0).sort((a, b) => a - b);
    if (!v.length) return 50;
    return Math.max(5, Math.ceil(v[Math.floor(v.length * 0.99)] * 1.1));
  }, [energies]);

  const dos = useMemo(() => energies ? phononDOS(energies, { sigma, Emin: 0, Emax, nE: 400 }) : null, [energies, sigma, Emax]);

  const compute = async () => {
    if (!pipeline || !files?.length || !baseStructure) { setStatus('Select a dataset first.'); return; }
    setBusy(true); setStatus('');
    pipeline.onProgress = (p, t) => { setProgress(p); setStatus(t); };
    try {
      const res = await pipeline.computeDOSGrid(files, family, baseStructure, gridN, temperature, 50, { referenceHandle });
      setEnergies(res.energies);
      setInfo({ gridN: res.gridN, nq: res.nq, nModes: res.nModes });
      setStatus(`Done: ${res.nq} q-points × ${res.nModes} modes.`);
    } catch (e) {
      setStatus(e.message === 'cancelled' ? 'Cancelled.' : 'Error: ' + e.message);
    } finally { setBusy(false); }
  };

  const exportCsv = () => {
    if (!dos) return;
    let csv = 'E_meV,g(E)\n';
    for (let i = 0; i < dos.E.length; i++) csv += `${dos.E[i].toFixed(5)},${dos.dos[i].toExponential(6)}\n`;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = 'phonon_dos.csv'; document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(a.href);
  };

  return (
    <div>
      <div className="flex items-center gap-3 flex-wrap mb-3">
        <div className="flex items-center gap-2 text-gray-200"><BarChart3 className="w-5 h-5 text-emerald-400" /><h2 className="text-lg font-medium">Phonon DOS (q-grid)</h2></div>
        <label className="text-xs text-gray-400 flex items-center gap-1">grid n³
          <input type="number" min={2} max={16} value={gridN} onChange={e => setGridN(Math.max(2, Math.min(16, parseInt(e.target.value) || 2)))} className="w-14 bg-white/5 border border-white/10 rounded px-2 py-1" />
        </label>
        <span className="text-[11px] text-gray-500">{gridN ** 3} q-points</span>
        <label className="text-xs text-gray-400 flex items-center gap-1">σ (meV)
          <input type="number" min={0.1} step={0.1} value={sigma} onChange={e => setSigma(Math.max(0.1, parseFloat(e.target.value) || 0.1))} className="w-16 bg-white/5 border border-white/10 rounded px-2 py-1" />
        </label>
        {busy
          ? <button onClick={() => { pipeline?.cancel(); setStatus('Cancelling…'); }} className="px-4 py-1.5 rounded-lg text-sm bg-red-600/80 hover:bg-red-600">Cancel</button>
          : <button onClick={compute} disabled={!files?.length} className={`px-4 py-1.5 rounded-lg text-sm font-medium ${files?.length ? 'bg-emerald-600 hover:bg-emerald-500' : 'bg-white/10 text-gray-500'}`}>Compute DOS</button>}
        {dos && <button onClick={exportCsv} className="px-3 py-1.5 rounded-lg text-sm bg-white/10 hover:bg-white/20 border border-white/10">Export CSV</button>}
      </div>

      {busy && (
        <div className="mb-3">
          <div className="text-xs text-gray-400 mb-1">{status} {Math.round(progress)}%</div>
          <div className="w-full bg-white/10 h-2 rounded-full overflow-hidden"><div className="h-full bg-emerald-500" style={{ width: `${progress}%` }} /></div>
        </div>
      )}
      {!busy && status && <div className="text-xs text-gray-400 mb-2">{status}{info ? ` (${info.gridN}³ grid)` : ''}</div>}

      {dos && (
        <div className="bg-black/30 rounded-xl border border-white/5 p-3">
          <LineChart xLabel="Energy (meV)" yLabel="g(E)" height={280}
            series={[{ name: 'phonon DOS', color: '#34d399', width: 1.6, points: Array.from(dos.E, (e, i) => [e, dos.dos[i]]) }]} />
        </div>
      )}
    </div>
  );
}
