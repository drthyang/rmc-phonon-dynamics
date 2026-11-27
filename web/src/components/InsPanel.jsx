import React, { useEffect, useId, useMemo, useRef, useState } from 'react';
import { buildInsData } from '../compute/ins';
import { CHOPPER_PRESETS, chopperResolutionFwhmAtE } from '../math/insCompare';
import { downloadString } from '../io/writers';
import { CMAP_NAMES, colormap, cmapGradient, intensityTransform } from './sqeColormaps';

const HBAR2_2MN = 2.0721248;
const BG = [11, 14, 22];
const DOS_STROKE = '#1b1a18', DOS_FILL = 'rgba(27,26,24,0.10)';

function qRange(E, Ei, ki) {
  if (Ei <= 0) return [-1, Infinity];
  if (E > Ei) return [1, -1];
  const kf = Math.sqrt(Math.max(0, Ei - E) / HBAR2_2MN);
  return [Math.abs(ki - kf), ki + kf];
}

/**
 * Simulated INS panel in MOTIF's clean museum-editorial aesthetic.
 * Supports coherent + incoherent 1-phonon scattering, Sjolander multiphonon (n >= 2),
 * and direct-geometry chopper resolution ΔE(E; E_i).
 */
export default function InsPanel({ results, temperature }) {
  const uid = useId();
  const workerRef = useRef(null);
  const canvasRef = useRef(null);
  const [running, setRunning] = useState(false);
  const [out, setOut] = useState(null);
  const [error, setError] = useState(null);

  const maxE = useMemo(() => {
    const vals = [];
    for (const row of results.bands) for (const v of row) if (isFinite(v) && v > 0) vals.push(v);
    if (!vals.length) return 50;
    vals.sort((a, b) => a - b);
    const p90 = vals[Math.min(vals.length - 1, Math.floor(vals.length * 0.90))];
    return Math.max(5, Math.ceil(p90 * 1.15));
  }, [results]);

  const [params, setParams] = useState(() => ({
    T: temperature ?? 5,
    Emin: 0,
    Emax: maxE,
    sigma: Math.max(0.3, maxE / 100),
    nE: 160,
    nQbins: 140,
    Ei: Math.max(5, Math.ceil(maxE * 1.25)),
    instrument: 'SEQUOIA',
    scatteringMode: 'total',
    includeMultiphonon: true,
    useFullMesh: true,
  }));
  const [cmap, setCmap] = useState('viridis');
  const [scale, setScale] = useState('log');
  const [contrast, setContrast] = useState(1);
  const [showModeDos, setShowModeDos] = useState(false);
  const [cursor, setCursor] = useState(null);

  useEffect(() => {
    setOut(null); setError(null); setCursor(null);
    setParams(p => ({
      ...p,
      T: temperature ?? p.T,
      Emax: maxE,
      sigma: Math.max(0.3, maxE / 100),
      Ei: Math.max(5, Math.ceil(maxE * 1.25)),
    }));
  }, [results, maxE, temperature]);

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
    const bad = ['T', 'Emin', 'Emax', 'sigma', 'Ei', 'nE', 'nQbins'].find(k => !Number.isFinite(params[k]));
    if (bad) { setError('Please enter a valid number for every field.'); return; }
    if (!(params.Emax > params.Emin)) { setError('E max must be greater than E min.'); return; }
    if (!(params.sigma > 0)) { setError('σ must be greater than 0.'); return; }
    if (params.nE < 2 || params.nQbins < 1) { setError('nE must be ≥ 2 and nQ ≥ 1.'); return; }
    setRunning(true); setError(null);
    try {
      const { data, transfer } = buildInsData(results, {
        temperatureK: params.T,
        useFullMesh: params.useFullMesh,
      });
      workerRef.current.postMessage({ data, params }, transfer);
    } catch (err) {
      setRunning(false); setError(err.message);
    }
  };

  const spec = useMemo(() => {
    if (!out?.powResult) return null;
    const { S, nX, nE, xMax, Ei, Eaxis } = out.powResult;
    const dQ = xMax / nX;
    const ki = Ei > 0 ? Math.sqrt(Ei / HBAR2_2MN) : 0;
    const g = new Float64Array(nE);
    for (let ei = 0; ei < nE; ei++) {
      const [qlo, qhi] = qRange(Eaxis[ei], Ei, ki);
      if (qlo > qhi) continue;
      let s = 0;
      for (let qi = 0; qi < nX; qi++) {
        const Q = (qi + 0.5) * dQ;
        if (Q < qlo || Q > qhi) continue;
        s += S[qi * nE + ei];
      }
      g[ei] = s * dQ;
    }
    let gmax = 0;
    for (let i = 0; i < nE; i++) if (g[i] > gmax) gmax = g[i];
    return { g, gmax, nE };
  }, [out]);

  useEffect(() => {
    if (!out?.powResult || !canvasRef.current) return;
    const { S, nX, nE, Smax, Eaxis, xMax, Ei } = out.powResult;
    const cv = canvasRef.current;
    cv.width = nX; cv.height = nE;
    const ctx = cv.getContext('2d');
    const img = ctx.createImageData(nX, nE);
    const inv = Smax > 0 ? 1 / Smax : 0;
    const dQ = xMax / nX;
    const ki = Ei > 0 ? Math.sqrt(Ei / HBAR2_2MN) : 0;

    for (let ei = 0; ei < nE; ei++) {
      const [qlo, qhi] = qRange(Eaxis[ei], Ei, ki);
      for (let qi = 0; qi < nX; qi++) {
        const px = ((nE - 1 - ei) * nX + qi) * 4;
        const Q = (qi + 0.5) * dQ;
        let r, g, b;
        if (Q < qlo || Q > qhi) { [r, g, b] = BG; }
        else {
          const raw = Math.max(0, S[qi * nE + ei] * inv);
          [r, g, b] = colormap(intensityTransform(raw, scale, contrast), cmap);
        }
        img.data[px] = r; img.data[px + 1] = g; img.data[px + 2] = b; img.data[px + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [out, cmap, scale, contrast]);

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

  const exportDos = () => {
    if (!spec || !out?.powResult) return;
    const { Eaxis } = out.powResult;
    let csv = 'E_meV,integrated_intensity\n';
    for (let i = 0; i < spec.nE; i++) csv += `${Eaxis[i].toFixed(5)},${spec.g[i].toExponential(6)}\n`;
    downloadString(csv, 'integrated_intensity.csv');
  };

  const fields = [
    ['T (K)', 'T', 1], ['E min', 'Emin', 1], ['E max', 'Emax', 1], ['σ (meV)', 'sigma', 0.1],
    ['Eᵢ (meV)', 'Ei', 1], ['nE', 'nE', 1], ['nQ', 'nQbins', 1],
  ];
  const scaleLabel = scale === 'log' ? 'log' : scale === 'sqrt' ? '√' : 'linear';

  const pr = out?.powResult;
  const Emin = pr ? pr.Eaxis[0] : 0;
  const Emax = pr ? pr.Eaxis[pr.nE - 1] : 1;
  const eTicks = pr ? energyTicks(Emin, Emax, pr.nE, 5) : [];

  const chopperHint = useMemo(() => {
    if (params.instrument === 'constant' || !(params.Ei > 0)) return null;
    const f0 = chopperResolutionFwhmAtE(0, params.Ei, params.instrument);
    const eHi = Math.max(5, Math.round(0.8 * params.Ei));
    const fHi = chopperResolutionFwhmAtE(eHi, params.Ei, params.instrument);
    return `FWHM: ${f0.toFixed(2)} meV (E=0) → ${fHi.toFixed(2)} meV (E=${eHi})`;
  }, [params.instrument, params.Ei]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, flexWrap: 'wrap' }}>
          <h2 style={{ margin: 0, fontSize: 17, fontWeight: 700, letterSpacing: '-0.02em' }}>
            Simulated INS · S(|Q|, E)
          </h2>
          <span style={{ fontSize: 'var(--fz-small)', color: 'var(--secondary)' }}>
            coherent + incoherent powder average, Debye–Waller &amp; multiphonon
          </span>
        </div>
        <span id={`${uid}-chopper-note`} style={{ fontSize: 'var(--fz-small)', fontFamily: 'var(--font-mono)', color: 'var(--faint)' }}>
          {chopperHint || 'Eᵢ = 0 ⇒ full |Q| range (constant σ)'}
        </span>
      </div>

      {/* Numeric & Spectrometer Physics Parameters */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(105px, 1fr))', gap: 10 }}>
        {fields.map(([label, key, step]) => (
          <div key={key} style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <label htmlFor={`${uid}-${key}`} style={{ fontSize: 'var(--fz-small)', color: 'var(--secondary)' }}>
              {label}
            </label>
            <input
              id={`${uid}-${key}`}
              type="number"
              step={step}
              value={Number.isFinite(params[key]) ? params[key] : ''}
              onChange={e => setParams(p => ({ ...p, [key]: parseFloat(e.target.value) }))}
              className="m-input mono tabular-nums"
              style={{ width: '100%' }}
            />
          </div>
        ))}

        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <label htmlFor={`${uid}-inst`} style={{ fontSize: 'var(--fz-small)', color: 'var(--secondary)' }}>
            Resolution
          </label>
          <select
            id={`${uid}-inst`}
            value={params.instrument}
            onChange={e => setParams(p => ({ ...p, instrument: e.target.value }))}
            aria-describedby={`${uid}-chopper-note`}
            className="m-select mono"
            style={{ width: '100%', height: 32 }}
          >
            {Object.values(CHOPPER_PRESETS).map(prOpt => (
              <option key={prOpt.key} value={prOpt.key}>{prOpt.shortLabel}</option>
            ))}
          </select>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <label htmlFor={`${uid}-scatt`} style={{ fontSize: 'var(--fz-small)', color: 'var(--secondary)' }}>
            Cross-section
          </label>
          <select
            id={`${uid}-scatt`}
            value={params.scatteringMode}
            onChange={e => setParams(p => ({ ...p, scatteringMode: e.target.value }))}
            className="m-select mono"
            style={{ width: '100%', height: 32 }}
          >
            <option value="total">Coh + Inc</option>
            <option value="coherent">Coherent only</option>
            <option value="incoherent_approx">Incoh approx</option>
          </select>
        </div>
      </div>

      {/* Actions & Display Bar */}
      <div className="m-rule-box" style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 14 }}>
        <button type="button" onClick={run} disabled={running} className="m-btn primary">
          {running ? 'Computing…' : 'Run INS'}
        </button>

        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 'var(--fz-small)', color: 'var(--secondary)' }}>
          <label htmlFor={`${uid}-cmap`}>colormap</label>
          <select id={`${uid}-cmap`} value={cmap} onChange={e => setCmap(e.target.value)} className="m-select mono" style={{ height: 28 }}>
            {CMAP_NAMES.map(n => <option key={n} value={n}>{n}</option>)}
          </select>
        </span>

        <div className="m-tools" role="group" aria-label="Intensity scale">
          <span style={{ fontSize: 'var(--fz-small)', color: 'var(--faint)', marginRight: 4 }}>scale</span>
          {[['linear', 'Lin'], ['sqrt', '√'], ['log', 'Log']].map(([val, lab]) => (
            <button
              key={val}
              type="button"
              aria-pressed={scale === val}
              className={scale === val ? 'on' : 'dim'}
              onClick={() => setScale(val)}
            >
              {lab}
            </button>
          ))}
        </div>

        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8, fontSize: 'var(--fz-small)', color: 'var(--secondary)' }}>
          <label htmlFor={`${uid}-contrast`}>contrast</label>
          <input id={`${uid}-contrast`} type="range" min={0.4} max={3} step={0.1} value={contrast} onChange={e => setContrast(parseFloat(e.target.value))} className="m-range" style={{ width: 88 }} />
          <span className="tabular-nums" style={{ fontFamily: 'var(--font-mono)', color: 'var(--ink)', minWidth: 28 }}>×{contrast.toFixed(1)}</span>
        </span>

        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 'var(--fz-small)', color: 'var(--ink)' }}>
          <input
            id={`${uid}-multi`}
            type="checkbox"
            checked={params.includeMultiphonon}
            onChange={e => setParams(p => ({ ...p, includeMultiphonon: e.target.checked }))}
          />
          <label htmlFor={`${uid}-multi`} style={{ cursor: 'pointer' }}>multiphonon (n≥2)</label>
        </span>

        {results.dynamicalMatrixPayload && (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 'var(--fz-small)', color: 'var(--ink)' }}>
            <input
              id={`${uid}-mesh`}
              type="checkbox"
              checked={params.useFullMesh}
              onChange={e => setParams(p => ({ ...p, useFullMesh: e.target.checked }))}
            />
            <label htmlFor={`${uid}-mesh`} style={{ cursor: 'pointer' }}>3D BZ mesh</label>
          </span>
        )}

        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 'var(--fz-small)', color: 'var(--ink)' }}>
          <input id={`${uid}-modedos`} type="checkbox" checked={showModeDos} onChange={e => setShowModeDos(e.target.checked)} />
          <label htmlFor={`${uid}-modedos`} style={{ cursor: 'pointer' }}>mode DOS</label>
        </span>

        {pr && (
          <div style={{ marginLeft: 'auto', display: 'inline-flex', gap: 6 }}>
            <button type="button" onClick={exportDos} className="m-btn sm">Save g(E) CSV</button>
            <button type="button" onClick={exportCsv} className="m-btn sm">Save S(Q,E) CSV</button>
          </div>
        )}
      </div>

      {error && <div className="m-notice bad" style={{ margin: 0 }}>{error}</div>}

      {pr ? (
        <div className="m-frame" style={{ padding: 16 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 'var(--fz-small)', fontFamily: 'var(--font-mono)', color: 'var(--secondary)', marginBottom: 10, flexWrap: 'wrap', gap: 8 }}>
            <span>
              S(|Q|, E) · {scaleLabel} · {pr.scatteringMode === 'coherent' ? 'Coherent 1-ph' : pr.scatteringMode === 'incoherent_approx' ? 'Incoh approx' : 'Coh + Inc'}
              {pr.includeMultiphonon ? ' + Multi' : ''}
              {pr.instrument && pr.instrument !== 'constant' ? ` · ${pr.instrument}` : ''}
              {cursor && (
                <b className="tabular-nums" style={{ marginLeft: 12, color: 'var(--ink)' }}>
                  |Q| {cursor.Q.toFixed(2)} Å⁻¹ · E {cursor.E.toFixed(1)} meV · {cursor.masked ? 'outside kinematic range' : `S ${cursor.S.toExponential(2)}`}
                </b>
              )}
            </span>
            <span style={{ color: 'var(--faint)' }}>∫S(|Q|, E) d|Q| (barn · meV⁻¹ · Å⁻¹, norm.)</span>
          </div>

          <div className="sqe-fig">
            <div className="sqe-grid">
              <div style={{ gridArea: 'ticks', position: 'relative', minWidth: 34, height: '100%' }}>
                {eTicks.map((t, i) => (
                  <span key={i} style={{ position: 'absolute', right: 5, top: `${t.frac * 100}%`, transform: 'translateY(-50%)', fontFamily: 'var(--font-mono)', fontSize: 10.5, color: 'var(--faint)', whiteSpace: 'nowrap' }}>
                    {t.label}
                  </span>
                ))}
                <span style={{ position: 'absolute', left: -2, top: '50%', transform: 'translateY(-50%) rotate(-90deg)', transformOrigin: 'left center', fontFamily: 'var(--font-mono)', fontSize: 10.5, color: 'var(--secondary)' }}>
                  E (meV)
                </span>
              </div>

              <canvas
                ref={canvasRef}
                onMouseMove={(e) => {
                  const rect = e.currentTarget.getBoundingClientRect();
                  if (!rect.width || !rect.height) return;
                  const fx = Math.max(0, Math.min(0.9999, (e.clientX - rect.left) / rect.width));
                  const fy = Math.max(0, Math.min(0.9999, (e.clientY - rect.top) / rect.height));
                  const qi = Math.floor(fx * pr.nX);
                  const ei = pr.nE - 1 - Math.floor(fy * pr.nE);
                  const dQ = pr.xMax / pr.nX;
                  const Q = (qi + 0.5) * dQ;
                  const E = pr.Eaxis[ei];
                  const ki = pr.Ei > 0 ? Math.sqrt(pr.Ei / HBAR2_2MN) : 0;
                  const [qlo, qhi] = qRange(E, pr.Ei, ki);
                  setCursor({ Q, E, S: pr.S[qi * pr.nE + ei], masked: Q < qlo || Q > qhi });
                }}
                onMouseLeave={() => setCursor(null)}
                style={{ gridArea: 'heat', width: '100%', height: '100%', minWidth: 0, minHeight: 0, display: 'block', border: '1px solid var(--line)', borderRadius: 6, imageRendering: 'pixelated', background: '#0b0e16', cursor: 'crosshair' }}
              />

              <div className="sqe-bar" style={{ gridArea: 'bar', display: 'flex', flexDirection: 'column', alignItems: 'center', minHeight: 0 }}>
                <div style={{ flex: 1, width: 13, borderRadius: 4, border: '1px solid var(--line)', background: cmapGradient(cmap) }} />
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--faint)', marginTop: 5 }}>S↑</span>
              </div>

              <div className="sqe-curve" style={{ gridArea: 'curve', minWidth: 0, minHeight: 0 }}>
                <DosPlot spec={spec} Emin={Emin} Emax={Emax} modeDos={showModeDos ? out.dosResult : null} />
              </div>

              <div style={{ gridArea: 'qlab' }}>
                <div className="tabular-nums" style={{ display: 'flex', justifyContent: 'space-between', marginTop: 5, fontFamily: 'var(--font-mono)', fontSize: 10.5, color: 'var(--faint)' }}>
                  {ticks(0, pr.xMax, 5).map((t, i) => <span key={i}>{t.v.toFixed(1)}</span>)}
                </div>
                <div style={{ textAlign: 'center', fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--secondary)', marginTop: 2 }}>|Q| (Å⁻¹)</div>
              </div>
              <div style={{ gridArea: 'glab', textAlign: 'center', fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--secondary)', marginTop: 5, alignSelf: 'start' }}>
                intensity →
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="m-rule-box" style={{ padding: '28px 18px', textAlign: 'center', color: 'var(--secondary)', fontSize: 'var(--fz-control)' }}>
          Click <b style={{ color: 'var(--ink)' }}>Run INS</b> to compute the coherent + incoherent powder S(|Q|, E) heatmap and integrated intensity curve.
        </div>
      )}
    </div>
  );
}

function ticks(min, max, n) {
  const arr = [];
  for (let i = 0; i <= n; i++) { const frac = i / n; arr.push({ frac, v: min + frac * (max - min) }); }
  return arr;
}

function energyTicks(Emin, Emax, nE, n) {
  const dE = (Emax - Emin) / (nE - 1);
  const Etop = Emax + dE / 2, span = nE * dE || 1;
  const out = [];
  for (let i = 0; i <= n; i++) {
    const v = Emin + (i / n) * (Emax - Emin);
    out.push({ label: v.toFixed(0), frac: (Etop - v) / span });
  }
  return out;
}

function DosPlot({ spec, Emin, Emax, modeDos }) {
  const ref = useRef(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv || !spec || !(isFinite(Emin) && isFinite(Emax) && Emax > Emin)) {
      if (cv) cv.getContext('2d')?.clearRect(0, 0, cv.width, cv.height);
      return;
    }
    const { g, gmax, nE } = spec;
    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = cv.getBoundingClientRect();
      const W = Math.max(1, Math.round(rect.width * dpr));
      const H = Math.max(1, Math.round(rect.height * dpr));
      if (cv.width !== W) cv.width = W;
      if (cv.height !== H) cv.height = H;
      const ctx = cv.getContext('2d');
      ctx.clearRect(0, 0, W, H);
      const dE = (Emax - Emin) / (nE - 1);
      const Etop = Emax + dE / 2, span = nE * dE || 1;
      const yOf = (E) => ((Etop - E) / span) * H;
      const gm = gmax || 1;

      if (modeDos && modeDos.dos) {
        const mm = modeDos.dosMax || 1;
        ctx.strokeStyle = 'rgba(95,91,85,0.65)';
        ctx.setLineDash([4 * dpr, 3 * dpr]);
        ctx.lineWidth = 1.1 * dpr;
        ctx.beginPath();
        for (let i = 0; i < nE; i++) {
          const x = (modeDos.dos[i] / mm) * W, y = yOf(Emin + i * dE);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }

      ctx.beginPath();
      ctx.moveTo(0, yOf(Emin));
      for (let i = 0; i < nE; i++) ctx.lineTo((g[i] / gm) * W, yOf(Emin + i * dE));
      ctx.lineTo(0, yOf(Emax));
      ctx.closePath();
      ctx.fillStyle = DOS_FILL;
      ctx.fill();
      ctx.strokeStyle = DOS_STROKE;
      ctx.lineWidth = 1.6 * dpr;
      ctx.beginPath();
      for (let i = 0; i < nE; i++) {
        const x = (g[i] / gm) * W, y = yOf(Emin + i * dE);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(cv);
    return () => ro.disconnect();
  }, [spec, Emin, Emax, modeDos]);

  return (
    <canvas
      ref={ref}
      style={{ width: '100%', height: '100%', display: 'block', border: '1px solid var(--line)', borderRadius: 6, background: 'var(--surface)' }}
    />
  );
}
