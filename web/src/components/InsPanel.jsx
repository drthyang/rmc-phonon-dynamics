import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Waves } from 'lucide-react';
import { buildYData } from '../compute/ins';
import { downloadString } from '../io/writers';

/**
 * Minimal INS (simulated inelastic neutron scattering) panel: powder S(|Q|,E)
 * heatmap + phonon DOS. Computation runs in io/sqeworker.js. UI is intentionally
 * basic — enough to run and validate the calculation.
 */
export default function InsPanel({ results, temperature }) {
  const workerRef = useRef(null);
  const canvasRef = useRef(null);
  const [running, setRunning] = useState(false);
  const [out, setOut] = useState(null);
  const [error, setError] = useState(null);

  // Default energy window from the computed band range.
  const maxE = useMemo(() => {
    let m = 1;
    for (const row of results.bands) for (const v of row) if (isFinite(v)) m = Math.max(m, Math.abs(v));
    return Math.ceil(m * 1.15);
  }, [results]);

  const [params, setParams] = useState(() => ({
    T: temperature ?? 5, Emin: 0, Emax: maxE, sigma: Math.max(0.5, maxE / 40),
    nE: 128, nQbins: 128, Ei: 0,
  }));

  useEffect(() => {
    setParams(p => ({ ...p, Emax: maxE, sigma: Math.max(0.5, maxE / 40) }));
  }, [maxE]);

  useEffect(() => {
    workerRef.current = new Worker(new URL('../io/sqeworker.js', import.meta.url), { type: 'module' });
    workerRef.current.onmessage = (e) => {
      setRunning(false);
      if (e.data.success) { setOut(e.data); setError(null); }
      else setError(e.data.error || 'INS computation failed');
    };
    return () => workerRef.current?.terminate();
  }, []);

  const run = () => {
    setRunning(true);
    setError(null);
    try {
      const ydata = buildYData(results);
      workerRef.current.postMessage({ ydata, params });
    } catch (err) {
      setRunning(false);
      setError(err.message);
    }
  };

  // Draw S(Q,E) heatmap.
  useEffect(() => {
    if (!out?.powResult || !canvasRef.current) return;
    const { S, nX, nE, Smax } = out.powResult;
    const cv = canvasRef.current;
    cv.width = nX; cv.height = nE;
    const ctx = cv.getContext('2d');
    const img = ctx.createImageData(nX, nE);
    const inv = Smax > 0 ? 1 / Smax : 0;
    for (let qi = 0; qi < nX; qi++) {
      for (let ei = 0; ei < nE; ei++) {
        const v = Math.sqrt(Math.max(0, S[qi * nE + ei] * inv)); // sqrt for contrast
        // Energy increases upward => flip row.
        const px = ((nE - 1 - ei) * nX + qi) * 4;
        img.data[px] = Math.min(255, 30 + v * 80);
        img.data[px + 1] = Math.min(255, v * 200);
        img.data[px + 2] = Math.min(255, 60 + v * 195);
        img.data[px + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [out]);

  const exportCsv = () => {
    if (!out?.powResult) return;
    const { S, nX, nE, Eaxis, xMax } = out.powResult;
    const dQ = xMax / nX;
    let csv = 'Q_invA,E_meV,S\n';
    for (let qi = 0; qi < nX; qi++) for (let ei = 0; ei < nE; ei++) {
      csv += `${((qi + 0.5) * dQ).toFixed(5)},${Eaxis[ei].toFixed(5)},${S[qi * nE + ei].toExponential(6)}\n`;
    }
    downloadString(csv, 'sqe.csv');
  };

  const numField = (label, key, step = 1) => (
    <div>
      <label className="text-[10px] text-gray-400 uppercase tracking-wider block">{label}</label>
      <input type="number" step={step} value={params[key]}
        onChange={e => setParams(p => ({ ...p, [key]: parseFloat(e.target.value) }))}
        className="w-full bg-white/5 border border-white/10 rounded px-2 py-1 text-sm" />
    </div>
  );

  return (
    <div>
      <div className="flex items-center gap-3 mb-4 text-gray-200">
        <Waves className="w-5 h-5 text-cyan-400" />
        <h2 className="text-lg font-medium">Simulated INS — S(|Q|,E) &amp; DOS</h2>
      </div>

      <div className="grid grid-cols-3 sm:grid-cols-6 gap-2 mb-4">
        {numField('T (K)', 'T')}
        {numField('E min', 'Emin')}
        {numField('E max', 'Emax')}
        {numField('σ (meV)', 'sigma', 0.1)}
        {numField('nE', 'nE')}
        {numField('nQ', 'nQbins')}
      </div>

      <div className="flex gap-2 mb-4">
        <button onClick={run} disabled={running}
          className={`px-4 py-2 rounded-lg text-sm font-medium ${running ? 'bg-cyan-700/40' : 'bg-cyan-600 hover:bg-cyan-500'}`}>
          {running ? 'Computing…' : 'Run INS'}
        </button>
        {out?.powResult && (
          <button onClick={exportCsv} className="px-4 py-2 rounded-lg text-sm bg-white/10 hover:bg-white/20 border border-white/10">
            Export S(Q,E) CSV
          </button>
        )}
      </div>

      {error && <div className="text-red-400 text-sm mb-3">{error}</div>}

      {out?.powResult && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="lg:col-span-2">
            <div className="text-xs text-gray-400 mb-1">S(|Q|,E) — x: |Q| (0–{out.powResult.xMax.toFixed(2)} Å⁻¹), y: E (0–{params.Emax} meV)</div>
            <canvas ref={canvasRef} className="w-full rounded-lg border border-white/10"
              style={{ imageRendering: 'pixelated', aspectRatio: '1.4', background: '#06070a' }} />
          </div>
          <div>
            <div className="text-xs text-gray-400 mb-1">Phonon DOS</div>
            <DosPlot dosResult={out.dosResult} />
          </div>
        </div>
      )}
    </div>
  );
}

function DosPlot({ dosResult }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!dosResult || !ref.current) return;
    const { dos, nE, dosMax } = dosResult;
    const cv = ref.current;
    const W = 260, H = 200;
    cv.width = W; cv.height = H;
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = '#22d3ee';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < nE; i++) {
      const x = 4 + (dos[i] / (dosMax || 1)) * (W - 8);  // DOS horizontal
      const y = H - (i / (nE - 1)) * (H - 4) - 2;         // energy vertical
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }, [dosResult]);
  return <canvas ref={ref} className="w-full rounded-lg border border-white/10" style={{ background: '#06070a' }} />;
}
