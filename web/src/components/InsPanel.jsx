import React, { useEffect, useMemo, useRef, useState } from 'react';
import { buildInsData } from '../compute/ins';
import { downloadString } from '../io/writers';

/* ── Cobalt theme tokens ───────────────────────────────────────────────── */
const INK = 'var(--ink)', DIM = 'var(--dim)', FAINT = 'var(--faint)';
const ACCENT = 'var(--accent)', BORDER = 'var(--border)', INSET = 'var(--inset)';
const cardStyle = { background: 'var(--card)', border: `1px solid ${BORDER}`, borderRadius: 10 };
const insetInput = { width: '100%', boxSizing: 'border-box', background: INSET, border: `1px solid ${BORDER}`, borderRadius: 7, padding: '8px 10px', font: "13px 'Space Mono'", color: INK };

/**
 * INS (simulated inelastic neutron scattering) panel: powder S(|Q|,E) heatmap +
 * phonon DOS. Computation runs in io/sqeworker.js; the colormap + kinematic-mask
 * logic is unchanged — only the surrounding chrome was reskinned to the light
 * theme (the heatmap canvas stays dark for contrast, as designed).
 */
export default function InsPanel({ results, temperature }) {
  const workerRef = useRef(null);
  const canvasRef = useRef(null);
  const [running, setRunning] = useState(false);
  const [out, setOut] = useState(null);
  const [error, setError] = useState(null);

  // Default energy window from the BULK of the spectrum (90th percentile of
  // positive energies) so a few near-zero-eigenvalue outlier modes don't cram
  // the real bands into the bottom.
  const maxE = useMemo(() => {
    const vals = [];
    for (const row of results.bands) for (const v of row) if (isFinite(v) && v > 0) vals.push(v);
    if (!vals.length) return 50;
    vals.sort((a, b) => a - b);
    const p90 = vals[Math.min(vals.length - 1, Math.floor(vals.length * 0.90))];
    return Math.max(5, Math.ceil(p90 * 1.1));
  }, [results]);

  // Eᵢ (incident energy) defaults to just above the band top so the kinematic
  // (energy-conservation) cutoff frames the spectrum.
  const [params, setParams] = useState(() => ({
    T: temperature ?? 5, Emin: 0, Emax: maxE, sigma: Math.max(0.3, maxE / 100),
    nE: 160, nQbins: 140, Ei: Math.max(5, Math.ceil(maxE * 1.25)),
  }));
  const [cmap, setCmap] = useState('viridis');
  const [logScale, setLogScale] = useState(true);

  useEffect(() => {
    setParams(p => ({ ...p, Emax: maxE, sigma: Math.max(0.3, maxE / 100), Ei: Math.max(5, Math.ceil(maxE * 1.25)) }));
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
      const { data, transfer } = buildInsData(results);
      workerRef.current.postMessage({ data, params }, transfer);
    } catch (err) {
      setRunning(false);
      setError(err.message);
    }
  };

  // Draw S(Q,E) heatmap, masking the kinematically inaccessible region for a
  // direct-geometry spectrometer with incident energy Eᵢ:
  //   kᵢ=√(Eᵢ/c), k_f=√((Eᵢ−E)/c), c=ħ²/2mₙ; accessible iff E≤Eᵢ and
  //   |kᵢ−k_f| ≤ Q ≤ kᵢ+k_f. This produces the parabolic energy-conservation cutoff.
  useEffect(() => {
    if (!out?.powResult || !canvasRef.current) return;
    const { S, nX, nE, Smax, Eaxis, xMax } = out.powResult;
    const cv = canvasRef.current;
    cv.width = nX; cv.height = nE;
    const ctx = cv.getContext('2d');
    const img = ctx.createImageData(nX, nE);
    const inv = Smax > 0 ? 1 / Smax : 0;
    const logK = 1 / Math.log1p(1000);
    const HBAR2_2MN = 2.0723;
    const dQ = xMax / nX;
    const Ei = params.Ei;
    const ki = Ei > 0 ? Math.sqrt(Ei / HBAR2_2MN) : 0;
    const BG = [11, 14, 22];   // matches the dark canvas background (#0b0e16)

    for (let ei = 0; ei < nE; ei++) {
      const E = Eaxis[ei];
      let qlo = -1, qhi = Infinity;
      if (Ei > 0) {
        if (E > Ei) { qlo = 1; qhi = -1; }       // above Eᵢ → fully inaccessible
        else { const kf = Math.sqrt(Math.max(0, (Ei - E)) / HBAR2_2MN); qlo = Math.abs(ki - kf); qhi = ki + kf; }
      }
      for (let qi = 0; qi < nX; qi++) {
        const px = ((nE - 1 - ei) * nX + qi) * 4;
        const Q = (qi + 0.5) * dQ;
        let r, g, b;
        if (Ei > 0 && (Q < qlo || Q > qhi)) { [r, g, b] = BG; }
        else {
          const raw = Math.max(0, S[qi * nE + ei] * inv);
          const v = logScale ? Math.log1p(raw * 1000) * logK : Math.sqrt(raw);
          [r, g, b] = colormap(v, cmap);
        }
        img.data[px] = r; img.data[px + 1] = g; img.data[px + 2] = b; img.data[px + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [out, cmap, logScale, params.Ei]);

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

  const fields = [
    ['T (K)', 'T', 1], ['E min', 'Emin', 1], ['E max', 'Emax', 1], ['σ (meV)', 'sigma', 0.1],
    ['Eᵢ (meV)', 'Ei', 1], ['nE', 'nE', 1], ['nQ', 'nQbins', 1],
  ];

  return (
    <div style={{ ...cardStyle, padding: 20 }}>
      {/* header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16, flexWrap: 'wrap' }}>
        <span style={{ width: 24, height: 24, borderRadius: 6, background: ACCENT, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2"><path d="M2 12c2-4 4-4 6 0s4 4 6 0 4-4 6 0" /></svg>
        </span>
        <span style={{ font: "600 15px 'Space Grotesk'", color: INK }}>Simulated INS · S(|Q|,E) &amp; phonon DOS</span>
        <span style={{ font: "11px 'Space Mono'", color: FAINT }}>powder-averaged from eigenvectors</span>
      </div>

      {/* params */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginBottom: 16 }}>
        {fields.map(([label, key, step]) => (
          <div key={key} style={{ flex: 1, minWidth: 84 }}>
            <div style={{ font: "10px 'Space Mono'", color: FAINT, marginBottom: 5 }}>{label}</div>
            <input type="number" step={step} value={params[key]}
              onChange={e => setParams(p => ({ ...p, [key]: parseFloat(e.target.value) }))} style={insetInput} />
          </div>
        ))}
      </div>

      {/* actions */}
      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 12, marginBottom: 18 }}>
        <button onClick={run} disabled={running} className="rnr-btn"
          style={{ background: running ? 'var(--inset2)' : ACCENT, color: running ? DIM : '#fff', border: 'none', borderRadius: 8, padding: '9px 20px', font: "700 13px 'Space Grotesk'", cursor: running ? 'default' : 'pointer' }}>
          {running ? 'Computing…' : 'Run INS'}
        </button>
        <label style={{ display: 'flex', alignItems: 'center', gap: 7, font: "11px 'Space Mono'", color: DIM }}>colormap
          <select value={cmap} onChange={e => setCmap(e.target.value)}
            style={{ background: INSET, border: `1px solid ${BORDER}`, borderRadius: 6, padding: '6px 9px', font: "12px 'Space Mono'", color: INK, cursor: 'pointer' }}>
            <option value="viridis">viridis</option>
            <option value="magma">magma</option>
            <option value="gray">gray</option>
          </select>
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, font: "11px 'Space Mono'", color: DIM, cursor: 'pointer' }}>
          <input type="checkbox" checked={logScale} onChange={e => setLogScale(e.target.checked)} /> log scale
        </label>
        <span style={{ font: "10.5px 'Space Mono'", color: FAINT }}>Eᵢ = 0 ⇒ direct (full Q range)</span>
        {out?.powResult && (
          <button onClick={exportCsv} className="rnr-btn"
            style={{ marginLeft: 'auto', background: INSET, border: `1px solid ${BORDER}`, borderRadius: 7, padding: '8px 14px', font: "600 12px 'Space Grotesk'", color: INK, cursor: 'pointer' }}>
            Export S(Q,E) CSV
          </button>
        )}
      </div>

      {error && <div style={{ color: 'var(--warnInk)', font: "13px 'Space Mono'", marginBottom: 12 }}>{error}</div>}

      {out?.powResult ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 20 }}>
          {/* heatmap */}
          <div style={{ gridColumn: 'span 2', minWidth: 0 }}>
            <div style={{ font: "11px 'Space Mono'", color: DIM, marginBottom: 8 }}>S(|Q|,E) — powder-averaged simulated INS</div>
            <div style={{ display: 'flex', gap: 10 }}>
              {/* E axis */}
              <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', padding: '2px 0', font: "10px 'Space Mono'", color: FAINT, textAlign: 'right' }}>
                {ticks(params.Emin, params.Emax, 5).slice().reverse().map((t, i) => <span key={i}>{t.v.toFixed(0)}</span>)}
              </div>
              {/* canvas */}
              <div style={{ flex: 1, minWidth: 0 }}>
                <canvas ref={canvasRef} style={{ width: '100%', aspectRatio: '1.6', display: 'block', border: `1px solid ${BORDER}`, borderRadius: 8, imageRendering: 'pixelated', background: '#0b0e16' }} />
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 5, font: "10px 'Space Mono'", color: FAINT }}>
                  {ticks(0, out.powResult.xMax, 5).map((t, i) => <span key={i}>{t.v.toFixed(1)}</span>)}
                </div>
                <div style={{ textAlign: 'center', font: "10px 'Space Mono'", color: DIM, marginTop: 2 }}>|Q| (Å⁻¹)</div>
              </div>
              {/* colorbar */}
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', paddingTop: 2 }}>
                <div style={{ width: 13, height: 200, borderRadius: 4, border: `1px solid ${BORDER}`, background: COLORBAR[cmap] }} />
                <span style={{ font: "10px 'Space Mono'", color: FAINT, marginTop: 5 }}>S ↑</span>
              </div>
            </div>
            <div style={{ font: "10px 'Space Mono'", color: FAINT, marginTop: 6 }}>E (meV) — vertical axis</div>
          </div>
          {/* DOS */}
          <div style={{ minWidth: 0 }}>
            <div style={{ font: "11px 'Space Mono'", color: DIM, marginBottom: 8 }}>Phonon DOS</div>
            <DosPlot dosResult={out.dosResult} Emin={params.Emin} Emax={params.Emax} />
            <div style={{ textAlign: 'center', font: "10px 'Space Mono'", color: DIM, marginTop: 6 }}>g(E) → &nbsp;·&nbsp; energy vertical</div>
          </div>
        </div>
      ) : (
        <div style={{ font: "12px 'Spline Sans'", color: FAINT }}>Press <b style={{ color: DIM }}>Run INS</b> to compute the powder S(|Q|,E) map and phonon DOS.</div>
      )}
    </div>
  );
}

function ticks(min, max, n) {
  const arr = [];
  for (let i = 0; i <= n; i++) { const frac = i / n; arr.push({ frac, v: min + frac * (max - min) }); }
  return arr;
}

// Multi-stop colormaps (RGB) for the heatmap pixels.
const CMAPS = {
  viridis: [[11, 14, 22], [30, 58, 138], [34, 211, 238], [254, 249, 195]],
  magma: [[5, 3, 10], [80, 18, 123], [221, 73, 104], [252, 253, 191]],
  gray: [[14, 14, 18], [90, 90, 96], [180, 180, 186], [245, 245, 245]],
};
// CSS gradients (bottom→top) for the colorbar, derived from the same stops.
const grad = (st) => `linear-gradient(to top, ${st.map(c => `rgb(${c[0]},${c[1]},${c[2]})`).join(', ')})`;
const COLORBAR = { viridis: grad(CMAPS.viridis), magma: grad(CMAPS.magma), gray: grad(CMAPS.gray) };

function colormap(t, name = 'viridis') {
  const stops = CMAPS[name] || CMAPS.viridis;
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const a = stops[i], b = stops[i + 1];
  return [Math.round(a[0] + f * (b[0] - a[0])), Math.round(a[1] + f * (b[1] - a[1])), Math.round(a[2] + f * (b[2] - a[2]))];
}

function DosPlot({ dosResult, Emin = 0, Emax = 1 }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!dosResult || !ref.current) return;
    const { dos, nE, dosMax } = dosResult;
    const cv = ref.current;
    const PL = 30, PB = 6, W = 260, H = 236;
    cv.width = W; cv.height = H;
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, W, H);
    const plotW = W - PL - 8, plotH = H - PB - 6;
    // E-axis gridlines + ticks (energy vertical)
    ctx.font = "10px 'Space Mono', monospace"; ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const e = Emin + (i / 4) * (Emax - Emin);
      const y = H - PB - (i / 4) * plotH;
      ctx.strokeStyle = '#e3e7ef'; ctx.beginPath(); ctx.moveTo(PL, y); ctx.lineTo(W - 8, y); ctx.stroke();
      ctx.fillStyle = '#9aa1b2'; ctx.fillText(e.toFixed(0), PL - 4, y + 3);
    }
    // g(E) curve (DOS horizontal, energy vertical) with faint fill.
    const xy = (i) => [PL + (dos[i] / (dosMax || 1)) * plotW, H - PB - (i / (nE - 1)) * plotH];
    ctx.beginPath();
    ctx.moveTo(PL, H - PB);
    for (let i = 0; i < nE; i++) { const [x, y] = xy(i); ctx.lineTo(x, y); }
    ctx.lineTo(PL, H - PB - plotH);
    ctx.closePath();
    ctx.fillStyle = 'rgba(47,109,240,0.12)'; ctx.fill();
    ctx.strokeStyle = '#2f6df0'; ctx.lineWidth = 1.6; ctx.beginPath();
    for (let i = 0; i < nE; i++) { const [x, y] = xy(i); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
    ctx.stroke();
  }, [dosResult, Emin, Emax]);
  return <canvas ref={ref} style={{ width: '100%', aspectRatio: '1.1', display: 'block', border: '1px solid var(--border)', borderRadius: 8, background: 'var(--inset)' }} />;
}
