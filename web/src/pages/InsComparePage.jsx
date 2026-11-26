import React, { useState, useMemo, useEffect, useRef } from 'react';
import { parseExpIns1D, parseExpIns2D, parseNeutronDosCsv } from '../io/insReaders';
import { modelFromText } from '../io/phonopyDM';
import { phononDOS } from '../math/dos';
import {
  gaussianBroaden1D,
  fitAndScore1D,
  compare2DSqe,
  integrateOverQ,
  integrateOverE,
} from '../math/insCompare';
import { CMAP_NAMES, colormap, intensityTransform } from '../components/sqeColormaps';
import { DEFAULT_COLORS } from '../constants';
import LineChart from '../components/LineChart';

function HeatmapCanvas({ grid, qGrid, eGrid, cmap = 'viridis', scale = 'log', contrast = 1, diverging = false, title, subtitle }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv || !grid?.length || !grid[0]?.length) return;
    const nE = grid.length;
    const nQ = grid[0].length;
    cv.width = nQ;
    cv.height = nE;
    const ctx = cv.getContext('2d');
    const img = ctx.createImageData(nQ, nE);

    let vmax = 0;
    for (let ei = 0; ei < nE; ei++) {
      for (let qi = 0; qi < nQ; qi++) {
        const v = diverging ? Math.abs(grid[ei][qi]) : Math.max(0, grid[ei][qi]);
        if (Number.isFinite(v) && v > vmax) vmax = v;
      }
    }
    const inv = vmax > 1e-18 ? 1 / vmax : 0;

    for (let ei = 0; ei < nE; ei++) {
      for (let qi = 0; qi < nQ; qi++) {
        const px = ((nE - 1 - ei) * nQ + qi) * 4;
        let r, g, b;
        if (diverging) {
          const t = Math.max(-1, Math.min(1, grid[ei][qi] * inv * contrast));
          if (t >= 0) {
            r = Math.round(255);
            g = Math.round(255 * (1 - t * 0.82));
            b = Math.round(255 * (1 - t * 0.85));
          } else {
            const a = -t;
            r = Math.round(255 * (1 - a * 0.82));
            g = Math.round(255 * (1 - a * 0.65));
            b = Math.round(255);
          }
        } else {
          const raw = Math.max(0, (grid[ei][qi] || 0) * inv);
          [r, g, b] = colormap(intensityTransform(raw, scale, contrast), cmap);
        }
        img.data[px] = r;
        img.data[px + 1] = g;
        img.data[px + 2] = b;
        img.data[px + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [grid, cmap, scale, contrast, diverging]);

  const qMin = qGrid?.[0] ?? 0, qMax = qGrid?.[qGrid.length - 1] ?? 1;
  const eMin = eGrid?.[0] ?? 0, eMax = eGrid?.[eGrid.length - 1] ?? 1;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6, minWidth: 0 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', fontSize: 12 }}>
        <b style={{ color: 'var(--ink)' }}>{title}</b>
        {subtitle && <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--secondary)' }}>{subtitle}</span>}
      </div>
      <div style={{ position: 'relative', border: '1px solid var(--line)', borderRadius: 6, overflow: 'hidden', background: '#0b0e16', aspectRatio: '4 / 3' }}>
        <canvas ref={canvasRef} style={{ width: '100%', height: '100%', display: 'block', imageRendering: 'pixelated' }} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--font-mono)', fontSize: 10.5, color: 'var(--faint)' }}>
        <span>|Q| {qMin.toFixed(1)}–{qMax.toFixed(1)} Å⁻¹</span>
        <span>E {eMin.toFixed(1)}–{eMax.toFixed(1)} meV</span>
      </div>
    </div>
  );
}

/**
 * Extract a 1D neutron-weighted GDOS (or fallback unweighted DOS from bands) for a candidate.
 */
function candidateDosOf(cand) {
  const nd = cand.model?.neutronDos;
  if (nd?.energy_mev?.length && nd?.neutron_gdos?.length) {
    return {
      energy: nd.energy_mev,
      gdos: nd.neutron_gdos,
      total: nd.total_dos || nd.neutron_gdos,
      elementGdos: nd.element_gdos || {},
      isNeutronWeighted: true,
      baseFwhm: nd.fwhm_mev || 0,
    };
  }
  // Fallback: compute unweighted DOS from bands if only band.yaml / old dm.json was loaded
  const bands = cand.model?.bands;
  if (bands?.length) {
    const flat = [];
    for (const row of bands) for (const v of row) if (Number.isFinite(v)) flat.push(v);
    if (flat.length) {
      const eMax = Math.max(5, Math.ceil(Math.max(...flat) * 1.1));
      const d = phononDOS(Float64Array.from(flat), { sigma: 0.6, Emin: 0, Emax: eMax, nE: 300 });
      return {
        energy: Array.from(d.E),
        gdos: Array.from(d.dos),
        total: Array.from(d.dos),
        elementGdos: {},
        isNeutronWeighted: false,
        baseFwhm: 1.4,
      };
    }
  }
  return null;
}

export default function InsComparePage({ candidates = [], onLoadModelFile, onPinDosCandidate }) {
  const [mode, setMode] = useState('1d'); // '1d' | '2d'
  const [exp1D, setExp1D] = useState(null);
  const [exp2D, setExp2D] = useState(null);
  const [err, setErr] = useState(null);

  // 1D comparison controls
  const [extraFwhm, setExtraFwhm] = useState(0);
  const [fitBg, setFitBg] = useState(false);
  const [eWinMin, setEWinMin] = useState('');
  const [eWinMax, setEWinMax] = useState('');
  const [showPartials, setShowPartials] = useState(true);
  const [partialCandId, setPartialCandId] = useState(null);
  const [hiddenIds, setHiddenIds] = useState([]);

  // 2D comparison controls
  const [selected2DId, setSelected2DId] = useState(null);
  const [cmap, setCmap] = useState('viridis');
  const [scale2D, setScale2D] = useState('log');
  const [contrast, setContrast] = useState(1);
  const [qSliceRange, setQSliceRange] = useState([1.0, 5.0]);
  const [eSliceRange, setESliceRange] = useState([5.0, 15.0]);

  const dosCandidates = useMemo(() => {
    return candidates
      .map(c => ({ ...c, dosData: candidateDosOf(c) }))
      .filter(c => c.dosData !== null);
  }, [candidates]);

  const ins2DCandidates = useMemo(
    () => candidates.filter(c => c.model?.powderIns?.sqe?.length),
    [candidates]
  );

  const activePartialCand = useMemo(() => {
    if (!dosCandidates.length) return null;
    return dosCandidates.find(c => c.id === partialCandId) || dosCandidates[0];
  }, [dosCandidates, partialCandId]);

  const active2DCand = useMemo(() => {
    if (!ins2DCandidates.length) return null;
    return ins2DCandidates.find(c => c.id === selected2DId) || ins2DCandidates[0];
  }, [ins2DCandidates, selected2DId]);

  const handleLoadExp1D = async (file) => {
    if (!file) return;
    try {
      setErr(null);
      const parsed = parseExpIns1D(await file.text(), file.name);
      setExp1D(parsed);
      setMode('1d');
    } catch (e) { setErr(e.message); }
  };

  const handleLoadExp2D = async (file) => {
    if (!file) return;
    try {
      setErr(null);
      const parsed = parseExpIns2D(await file.text(), file.name);
      setExp2D(parsed);
      setMode('2d');
    } catch (e) { setErr(e.message); }
  };

  const handleLoadMlipFile = async (file) => {
    if (!file) return;
    try {
      setErr(null);
      const text = await file.text();
      if (file.name.toLowerCase().endsWith('.csv')) {
        const nd = parseNeutronDosCsv(text, file.name);
        onPinDosCandidate?.(nd, file.name);
      } else {
        const m = modelFromText(text);
        onLoadModelFile?.(m, file.name, { stayOnPage: true });
      }
    } catch (e) { setErr(e.message); }
  };

  // Compute broadened 1D spectra + experimental fit scores
  const scored1D = useMemo(() => {
    const eLo = eWinMin === '' ? -Infinity : parseFloat(eWinMin);
    const eHi = eWinMax === '' ? Infinity : parseFloat(eWinMax);
    return dosCandidates.map(c => {
      const broadened = gaussianBroaden1D(c.dosData.energy, c.dosData.gdos, extraFwhm);
      let fit = null;
      if (exp1D) {
        fit = fitAndScore1D(exp1D.energy, exp1D.intensity, exp1D.sigma, c.dosData.energy, broadened, {
          eMin: eLo, eMax: eHi, fitBackground: fitBg,
        });
      }
      const scale = fit ? fit.scale : 1;
      const offset = fit ? fit.offset : 0;
      const points = c.dosData.energy.map((e, i) => [e, broadened[i] * scale + offset]);
      return { ...c, broadened, fit, points, scale, offset };
    });
  }, [dosCandidates, extraFwhm, exp1D, eWinMin, eWinMax, fitBg]);

  const ranked1D = useMemo(() => {
    return [...scored1D].sort((a, b) => (a.fit?.rw ?? Infinity) - (b.fit?.rw ?? Infinity));
  }, [scored1D]);

  // Series for main 1D LineChart
  const main1DSeries = useMemo(() => {
    const out = [];
    if (exp1D) {
      out.push({
        name: `Exp · ${exp1D.name}`,
        points: exp1D.energy.map((e, i) => [e, exp1D.intensity[i]]),
        color: '#1b1a18',
        width: 2.2,
      });
    }
    for (const s of scored1D) {
      if (hiddenIds.includes(s.id)) continue;
      out.push({
        name: s.label + (s.dosData.isNeutronWeighted ? '' : ' (unweighted)'),
        points: s.points,
        color: s.color,
        width: 1.9,
      });
    }
    return out;
  }, [exp1D, scored1D, hiddenIds]);

  const residualSeries = useMemo(() => {
    if (!exp1D) return [];
    const out = [];
    for (const s of scored1D) {
      if (hiddenIds.includes(s.id) || !s.fit?.expE?.length) continue;
      out.push({
        name: `${s.label} residual`,
        points: s.fit.expE.map((e, i) => [e, s.fit.residual[i]]),
        color: s.color,
        width: 1.5,
      });
    }
    return out;
  }, [exp1D, scored1D, hiddenIds]);

  const partialSeries = useMemo(() => {
    if (!showPartials || !activePartialCand) return [];
    const scored = scored1D.find(s => s.id === activePartialCand.id);
    if (!scored) return [];
    const elGdos = scored.dosData.elementGdos || {};
    const els = Object.keys(elGdos);
    if (!els.length) return [];
    return els.map(el => {
      const b = gaussianBroaden1D(scored.dosData.energy, elGdos[el], extraFwhm);
      return {
        name: `${el} partial`,
        points: scored.dosData.energy.map((e, i) => [e, b[i] * scored.scale]),
        color: DEFAULT_COLORS[el] || '#6b7280',
        width: 1.6,
        dashed: true,
      };
    });
  }, [showPartials, activePartialCand, scored1D, extraFwhm]);

  // 2D comparison computation
  const cmp2D = useMemo(() => {
    if (!active2DCand?.model?.powderIns || !exp2D) return null;
    return compare2DSqe(active2DCand.model.powderIns, exp2D);
  }, [active2DCand, exp2D]);

  const slice1DQSeries = useMemo(() => {
    const out = [];
    if (cmp2D) {
      const expCut = integrateOverQ(cmp2D.expGrid, cmp2D.q_grid, qSliceRange[0], qSliceRange[1]);
      const calcCut = integrateOverQ(cmp2D.calcGrid, cmp2D.q_grid, qSliceRange[0], qSliceRange[1]);
      out.push({ name: 'Exp S(E)', points: cmp2D.e_grid.map((e, i) => [e, expCut[i]]), color: '#1b1a18', width: 2 });
      out.push({ name: `${active2DCand.label} S(E)`, points: cmp2D.e_grid.map((e, i) => [e, calcCut[i]]), color: active2DCand.color, width: 1.8 });
    } else if (active2DCand?.model?.powderIns) {
      const p = active2DCand.model.powderIns;
      const calcCut = integrateOverQ(p.sqe, p.q_grid, qSliceRange[0], qSliceRange[1]);
      out.push({ name: `${active2DCand.label} S(E)`, points: p.e_grid.map((e, i) => [e, calcCut[i]]), color: active2DCand.color, width: 1.8 });
    }
    return out;
  }, [cmp2D, active2DCand, qSliceRange]);

  const slice1DESeries = useMemo(() => {
    const out = [];
    if (cmp2D) {
      const expCut = integrateOverE(cmp2D.expGrid, cmp2D.e_grid, eSliceRange[0], eSliceRange[1]);
      const calcCut = integrateOverE(cmp2D.calcGrid, cmp2D.e_grid, eSliceRange[0], eSliceRange[1]);
      out.push({ name: 'Exp S(|Q|)', points: cmp2D.q_grid.map((q, i) => [q, expCut[i]]), color: '#1b1a18', width: 2 });
      out.push({ name: `${active2DCand.label} S(|Q|)`, points: cmp2D.q_grid.map((q, i) => [q, calcCut[i]]), color: active2DCand.color, width: 1.8 });
    } else if (active2DCand?.model?.powderIns) {
      const p = active2DCand.model.powderIns;
      const calcCut = integrateOverE(p.sqe, p.e_grid, eSliceRange[0], eSliceRange[1]);
      out.push({ name: `${active2DCand.label} S(|Q|)`, points: p.q_grid.map((q, i) => [q, calcCut[i]]), color: active2DCand.color, width: 1.8 });
    }
    return out;
  }, [cmp2D, active2DCand, eSliceRange]);

  const toggleHide = (id) => setHiddenIds(prev => (prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]));

  // ── Empty landing state when neither candidates nor experiment are loaded ──
  if (dosCandidates.length === 0 && !exp1D && !exp2D) {
    return (
      <div className="m-landing">
        <div className="m-landing-card">
          <h1 className="m-landing-title">
            Validate local symmetry against experimental INS.
          </h1>
          <p className="m-landing-lead">
            Load one or more MLIP outputs (<code style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9em' }}>_mlip_dm.json</code> or <code style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9em' }}>_ndos.csv</code>) for your symmetry candidates, then overlay experimental 1D GDOS / S(E) or 2D S(|Q|, E) data to rank which local structure explains the measured phonon dynamics.
          </p>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            <label className="m-button primary lg">
              Load MLIP _dm.json / _ndos.csv…
              <input type="file" accept=".json,.yaml,.yml,.csv" onChange={e => handleLoadMlipFile(e.target.files?.[0])} style={{ display: 'none' }} />
            </label>
            <label className="m-button lg">
              Load Exp 1D INS (E, I)…
              <input type="file" accept=".csv,.txt,.dat,.xy,.xye" onChange={e => handleLoadExp1D(e.target.files?.[0])} style={{ display: 'none' }} />
            </label>
            <label className="m-button lg">
              Load Exp 2D S(|Q|, E)…
              <input type="file" accept=".csv,.txt,.dat" onChange={e => handleLoadExp2D(e.target.files?.[0])} style={{ display: 'none' }} />
            </label>
          </div>
          {err && <div className="m-notice bad" style={{ margin: '16px 0 0' }}>{err}</div>}
        </div>
      </div>
    );
  }

  const hasBothSecondary1D = residualSeries.length > 0 && showPartials && partialSeries.length > 0;

  return (
    <div className="m-work">
      {/* ════════ LEFT STAGE ════════ */}
      <div className="m-stage">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', paddingBottom: 10, borderBottom: '1px solid var(--line)', flex: 'none' }}>
          <div className="m-tools" role="tablist" aria-label="INS comparison mode">
            <button type="button" role="tab" aria-selected={mode === '1d'} className={mode === '1d' ? 'on' : 'dim'} onClick={() => setMode('1d')}>
              1D GDOS / S(E)
            </button>
            <button type="button" role="tab" aria-selected={mode === '2d'} className={mode === '2d' ? 'on' : 'dim'} onClick={() => setMode('2d')}>
              2D S(|Q|, E)
            </button>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <label className="m-button sm">
              + MLIP JSON / CSV
              <input type="file" accept=".json,.yaml,.yml,.csv" onChange={e => handleLoadMlipFile(e.target.files?.[0])} style={{ display: 'none' }} />
            </label>
            {mode === '1d' ? (
              <label className="m-button primary sm">
                {exp1D ? 'Replace Exp 1D…' : '+ Exp 1D INS…'}
                <input type="file" accept=".csv,.txt,.dat,.xy,.xye" onChange={e => handleLoadExp1D(e.target.files?.[0])} style={{ display: 'none' }} />
              </label>
            ) : (
              <label className="m-button primary sm">
                {exp2D ? 'Replace Exp 2D…' : '+ Exp 2D S(|Q|,E)…'}
                <input type="file" accept=".csv,.txt,.dat" onChange={e => handleLoadExp2D(e.target.files?.[0])} style={{ display: 'none' }} />
              </label>
            )}
          </div>
        </div>

        {err && <div className="m-notice bad" style={{ margin: 0 }}>{err}</div>}

        {/* ── MODE 1: 1D GDOS / S(E) ── */}
        {mode === '1d' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {exp1D && ranked1D[0]?.fit && Number.isFinite(ranked1D[0].fit.rw) && (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', padding: '10px 14px', borderRadius: 8, background: 'var(--note-bg)', border: '1px solid var(--line)', color: 'var(--note-ink)', fontSize: 13 }}>
                <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ width: 9, height: 9, borderRadius: '50%', background: ranked1D[0].color }} />
                  <span>
                    Best match: <b>{ranked1D[0].label}</b> · <b style={{ fontFamily: 'var(--font-mono)' }}>Rw {(100 * ranked1D[0].fit.rw).toFixed(1)}%</b>
                    {Number.isFinite(ranked1D[0].fit.cosine) && ` (cos ${ranked1D[0].fit.cosine.toFixed(3)})`}
                  </span>
                </div>
                {ranked1D[1]?.fit && Number.isFinite(ranked1D[1].fit.rw) && (
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11.5 }}>
                    vs. {ranked1D[1].label}: Rw {(100 * ranked1D[1].fit.rw).toFixed(1)}%
                  </span>
                )}
              </div>
            )}
            <div className="m-frame" style={{ padding: '12px 16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 6 }}>
                <span style={{ fontWeight: 600, fontSize: 14 }}>
                  Neutron-weighted GDOS vs. Experiment
                </span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11.5, color: 'var(--faint)' }}>
                  {exp1D ? `${exp1D.name} (${exp1D.energy.length} pts)` : 'No experimental 1D curve loaded'}
                </span>
              </div>
              <LineChart
                series={main1DSeries}
                xLabel="Energy transfer (meV)"
                yLabel="Intensity / GDOS (arb.)"
                height={320}
              />
            </div>

            {(residualSeries.length > 0 || (showPartials && partialSeries.length > 0)) && (
              <div className={`ins-1d-secondary ${hasBothSecondary1D ? 'has-both' : ''}`}>
                {residualSeries.length > 0 && (
                  <div className="m-frame" style={{ padding: '10px 16px' }}>
                    <div style={{ fontSize: 'var(--fz-small)', fontWeight: 600, color: 'var(--secondary)', marginBottom: 4 }}>
                      Residuals (Exp − Fit)
                    </div>
                    <LineChart
                      series={residualSeries}
                      xLabel="Energy transfer (meV)"
                      yLabel="ΔI"
                      height={180}
                    />
                  </div>
                )}

                {showPartials && partialSeries.length > 0 && (
                  <div className="m-frame" style={{ padding: '10px 16px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: 6 }}>
                      <span style={{ fontSize: 'var(--fz-small)', fontWeight: 600, color: 'var(--secondary)' }}>
                        Per-element partial GDOS · {activePartialCand?.label}
                      </span>
                      {dosCandidates.length > 1 && (
                        <select
                          value={activePartialCand?.id || ''}
                          onChange={e => setPartialCandId(e.target.value)}
                          className="m-input sm"
                        >
                          {dosCandidates.map(c => <option key={c.id} value={c.id}>{c.label}</option>)}
                        </select>
                      )}
                    </div>
                    <LineChart
                      series={partialSeries}
                      xLabel="Energy transfer (meV)"
                      yLabel="Element GDOS"
                      height={190}
                    />
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ── MODE 2: 2D S(|Q|, E) ── */}
        {mode === '2d' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {!active2DCand && !exp2D ? (
              <div className="m-frame" style={{ padding: '32px 20px', textAlign: 'center', color: 'var(--secondary)' }}>
                Load an MLIP <code style={{ fontFamily: 'var(--font-mono)' }}>_mlip_dm.json</code> (with embedded <code style={{ fontFamily: 'var(--font-mono)' }}>powder_ins</code>) and/or an experimental 2D <code style={{ fontFamily: 'var(--font-mono)' }}>S(|Q|, E)</code> CSV to compare 2D neutron spectra.
              </div>
            ) : (
              <>
                <div className="m-frame" style={{ padding: '14px 16px' }}>
                  <div className={`ins-heatmap-grid ${cmp2D ? 'is-trio' : ''}`}>
                    {cmp2D ? (
                      <>
                        <HeatmapCanvas
                          grid={cmp2D.calcGrid}
                          qGrid={cmp2D.q_grid}
                          eGrid={cmp2D.e_grid}
                          cmap={cmap}
                          scale={scale2D}
                          contrast={contrast}
                          title={`MLIP · ${active2DCand.label}`}
                          subtitle={`scale ${cmp2D.scale.toFixed(2)}`}
                        />
                        <HeatmapCanvas
                          grid={cmp2D.expGrid}
                          qGrid={cmp2D.q_grid}
                          eGrid={cmp2D.e_grid}
                          cmap={cmap}
                          scale={scale2D}
                          contrast={contrast}
                          title={`Experiment · ${exp2D.name}`}
                          subtitle={`Rw = ${Number.isFinite(cmp2D.rw) ? (100 * cmp2D.rw).toFixed(1) + '%' : '—'}`}
                        />
                        <HeatmapCanvas
                          grid={cmp2D.diffGrid}
                          qGrid={cmp2D.q_grid}
                          eGrid={cmp2D.e_grid}
                          contrast={contrast}
                          diverging
                          title="Difference (MLIP − Exp)"
                          subtitle={`cos = ${Number.isFinite(cmp2D.cosine) ? cmp2D.cosine.toFixed(3) : '—'}`}
                        />
                      </>
                    ) : active2DCand ? (
                      <HeatmapCanvas
                        grid={active2DCand.model.powderIns.sqe}
                        qGrid={active2DCand.model.powderIns.q_grid}
                        eGrid={active2DCand.model.powderIns.e_grid}
                        cmap={cmap}
                        scale={scale2D}
                        contrast={contrast}
                        title={`MLIP Powder INS · ${active2DCand.label}`}
                        subtitle={`T = ${active2DCand.model.powderIns.temperature_k} K`}
                      />
                    ) : (
                      <HeatmapCanvas
                        grid={exp2D.sqe}
                        qGrid={exp2D.q_grid}
                        eGrid={exp2D.e_grid}
                        cmap={cmap}
                        scale={scale2D}
                        contrast={contrast}
                        title={`Experiment · ${exp2D.name}`}
                      />
                    )}
                  </div>
                </div>

                {/* Interactive 1D Q-integrated and E-cut slices */}
                {(slice1DQSeries.length > 0 || slice1DESeries.length > 0) && (
                  <div className="ins-slice-grid">
                    <div className="m-frame" style={{ padding: '10px 14px' }}>
                      <div style={{ fontSize: 'var(--fz-small)', fontWeight: 600, color: 'var(--secondary)', marginBottom: 4 }}>
                        Q-integrated S(E) · |Q| ∈ [{qSliceRange[0]}, {qSliceRange[1]}] Å⁻¹
                      </div>
                      <LineChart series={slice1DQSeries} xLabel="Energy (meV)" yLabel="S(E)" height={190} />
                    </div>
                    <div className="m-frame" style={{ padding: '10px 14px' }}>
                      <div style={{ fontSize: 'var(--fz-small)', fontWeight: 600, color: 'var(--secondary)', marginBottom: 4 }}>
                        Constant-E cut S(|Q|) · E ∈ [{eSliceRange[0]}, {eSliceRange[1]}] meV
                      </div>
                      <LineChart series={slice1DESeries} xLabel="|Q| (Å⁻¹)" yLabel="S(|Q|)" height={190} />
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>

      {/* ════════ RIGHT INSPECTOR ════════ */}
      <aside className="m-side ins-side" aria-label="INS comparison controls">
        {mode === '1d' ? (
          <>
            {/* Scoreboard */}
            <section>
              <div style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--faint)', marginBottom: 8 }}>
                Candidate Scoreboard
              </div>
              {ranked1D.length === 0 ? (
                <div style={{ fontSize: 12, color: 'var(--faint)' }}>No MLIP candidates loaded yet.</div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {ranked1D.map((s, idx) => {
                    const hidden = hiddenIds.includes(s.id);
                    return (
                      <div key={s.id} style={{ padding: '8px 10px', borderRadius: 6, border: '1px solid var(--line)', background: idx === 0 && exp1D ? 'var(--note-bg)' : 'var(--surface)', opacity: hidden ? 0.45 : 1 }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6 }}>
                          <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 12.5, fontWeight: 600, cursor: 'pointer' }}>
                            <input type="checkbox" checked={!hidden} onChange={() => toggleHide(s.id)} />
                            <span style={{ width: 8, height: 8, borderRadius: '50%', background: s.color }} />
                            <span>{s.label}</span>
                          </label>
                          {s.fit && Number.isFinite(s.fit.rw) && (
                            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11.5, fontWeight: 700 }}>
                              Rw {(100 * s.fit.rw).toFixed(1)}%
                            </span>
                          )}
                        </div>
                        {s.fit && (
                          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--secondary)', marginTop: 4, display: 'flex', gap: 10 }}>
                            <span>cos {Number.isFinite(s.fit.cosine) ? s.fit.cosine.toFixed(3) : '—'}</span>
                            <span>χ² {Number.isFinite(s.fit.chi2) ? s.fit.chi2.toExponential(2) : '—'}</span>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </section>

            {/* Broadening & fit window */}
            <section style={{ paddingTop: 14, borderTop: '1px solid var(--line)', display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--faint)' }}>
                Resolution &amp; Fit
              </div>
              <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12 }}>
                <span style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--secondary)' }}>
                  <span>Extra Gaussian FWHM</span>
                  <b className="tabular-nums" style={{ fontFamily: 'var(--font-mono)', color: 'var(--ink)' }}>{extraFwhm.toFixed(1)} meV</b>
                </span>
                <input type="range" min={0} max={8} step={0.1} value={extraFwhm} onChange={e => setExtraFwhm(parseFloat(e.target.value))} className="m-range" />
              </label>

              <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--secondary)' }}>
                <span>Fit E</span>
                <input type="number" placeholder="min" value={eWinMin} onChange={e => setEWinMin(e.target.value)} className="m-input mono" style={{ width: 62, height: 26, textAlign: 'center' }} />
                <span>–</span>
                <input type="number" placeholder="max" value={eWinMax} onChange={e => setEWinMax(e.target.value)} className="m-input mono" style={{ width: 62, height: 26, textAlign: 'center' }} />
                <span>meV</span>
              </div>

              <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 12.5, cursor: 'pointer' }}>
                <input type="checkbox" checked={fitBg} onChange={e => setFitBg(e.target.checked)} />
                <span>Fit flat background offset</span>
              </label>

              <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 12.5, cursor: 'pointer' }}>
                <input type="checkbox" checked={showPartials} onChange={e => setShowPartials(e.target.checked)} />
                <span>Show per-element partial GDOS</span>
              </label>
            </section>
          </>
        ) : (
          /* 2D Inspector */
          <>
            <section style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--faint)' }}>
                2D Candidate &amp; Display
              </div>
              {ins2DCandidates.length > 0 && (
                <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12 }}>
                  <span style={{ color: 'var(--secondary)' }}>Candidate</span>
                  <select value={active2DCand?.id || ''} onChange={e => setSelected2DId(e.target.value)} className="m-input" style={{ height: 28 }}>
                    {ins2DCandidates.map(c => <option key={c.id} value={c.id}>{c.label}</option>)}
                  </select>
                </label>
              )}

              <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12 }}>
                <span style={{ color: 'var(--secondary)' }}>Colormap</span>
                <select value={cmap} onChange={e => setCmap(e.target.value)} className="m-input" style={{ height: 28 }}>
                  {CMAP_NAMES.map(name => <option key={name} value={name}>{name}</option>)}
                </select>
              </label>

              <div className="m-tools" role="group">
                {['log', 'sqrt', 'linear'].map(s => (
                  <button key={s} type="button" className={scale2D === s ? 'on' : 'dim'} onClick={() => setScale2D(s)}>{s}</button>
                ))}
              </div>

              <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12 }}>
                <span style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--secondary)' }}>
                  <span>Contrast</span>
                  <b className="tabular-nums" style={{ fontFamily: 'var(--font-mono)', color: 'var(--ink)' }}>{contrast.toFixed(1)}×</b>
                </span>
                <input type="range" min={0.4} max={4} step={0.1} value={contrast} onChange={e => setContrast(parseFloat(e.target.value))} className="m-range" />
              </label>
            </section>

            <section style={{ paddingTop: 14, borderTop: '1px solid var(--line)', display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--faint)' }}>
                1D Slice Windows
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--secondary)' }}>
                <span>|Q| cut</span>
                <input type="number" step="0.5" value={qSliceRange[0]} onChange={e => setQSliceRange([+e.target.value, qSliceRange[1]])} className="m-input mono" style={{ width: 56, height: 26, textAlign: 'center' }} />
                <span>–</span>
                <input type="number" step="0.5" value={qSliceRange[1]} onChange={e => setQSliceRange([qSliceRange[0], +e.target.value])} className="m-input mono" style={{ width: 56, height: 26, textAlign: 'center' }} />
                <span>Å⁻¹</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--secondary)' }}>
                <span>E cut</span>
                <input type="number" step="1" value={eSliceRange[0]} onChange={e => setESliceRange([+e.target.value, eSliceRange[1]])} className="m-input mono" style={{ width: 56, height: 26, textAlign: 'center' }} />
                <span>–</span>
                <input type="number" step="1" value={eSliceRange[1]} onChange={e => setESliceRange([eSliceRange[0], +e.target.value])} className="m-input mono" style={{ width: 56, height: 26, textAlign: 'center' }} />
                <span>meV</span>
              </div>
            </section>
          </>
        )}
      </aside>
    </div>
  );
}
