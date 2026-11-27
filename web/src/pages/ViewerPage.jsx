import React, { useState, useMemo, useEffect } from 'react';
import BandStructurePlot from '../components/BandStructurePlot';
import CrystalViewer from '../components/CrystalViewer';
import { DEFAULT_COLORS, COVALENT_R, THZ_TO_MEV } from '../constants';
import ModeInspector from '../components/ModeInspector';
import InsPanel from '../components/InsPanel';
import { generatePhonopyBandYaml, generateBandJson, downloadString } from '../io/writers';
import { modelFromText } from '../io/phonopyDM';
import {
  extractCellFromBaseStructure,
  findSupercellRelation,
  unfoldModelOntoParentPath,
  inferIntrinsicParentPath,
} from '../math/unfolding';

const THZ_PER_MEV = 1 / THZ_TO_MEV;

/**
 * Viewer page — MOTIF two-pane studio layout:
 *   Left stage: quiet toolbar + side-by-side Band Structure & 3D Mode (or S(Q,E))
 *   Right inspector column (behind 1px hairline): Motion | Visual | Species | Info
 */
export default function ViewerPage({
  model,
  candidates = [],
  activeCandidateId = null,
  onSelectCandidate,
  onRemoveCandidate,
  onLoadModel,
  onGoToIns,
}) {
  const [tab, setTab] = useState('modes');           // 'modes' | 'sqe'
  const [sideTab, setSideTab] = useState('motion');  // 'motion' | 'visual' | 'species' | 'info'
  const [hiddenOverlays, setHiddenOverlays] = useState([]);
  const [selK, setSelK] = useState(0);
  const [selM, setSelM] = useState(0);

  const [eMin, setEMin] = useState('');
  const [eMax, setEMax] = useState('');

  // 3D controls
  const [nx, setNx] = useState(2), [ny, setNy] = useState(2), [nz, setNz] = useState(1);
  const supercell = useMemo(() => [nx, ny, nz], [nx, ny, nz]);
  const [amplitude, setAmplitude] = useState(1.5);
  const [speed, setSpeed] = useState(0.03);
  const [playing, setPlaying] = useState(true);
  const [showVectors, setShowVectors] = useState(false);
  const [vectorScale, setVectorScale] = useState(2.0);
  const [vectorColor, setVectorColor] = useState('#dc2626');
  const [showCell, setShowCell] = useState(true);
  const [atomScale, setAtomScale] = useState(1);
  const [camNonce, setCamNonce] = useState(null);
  const [bandReset, setBandReset] = useState(0);

  // Appearance
  const [displayStyle, setDisplayStyle] = useState('ballstick');
  const [showBonds, setShowBonds] = useState(true);
  const [bondCutoff, setBondCutoff] = useState(3.0);
  const [bondThickness, setBondThickness] = useState(0.06);
  const [bondRules, setBondRules] = useState({});
  const [shading, setShading] = useState(true);
  const [shadingStrength, setShadingStrength] = useState(0.5);
  const [elementColors, setElementColors] = useState({});
  const [elementRadii, setElementRadii] = useState({});
  const [recording, setRecording] = useState(false);
  const [gifSignal, setGifSignal] = useState(0);

  const [thz, setThz] = useState(false);
  const [zoneMode, setZoneMode] = useState('unfolded');     // 'unfolded' | 'native'
  const [spectralStyle, setSpectralStyle] = useState('both'); // 'both' | 'heatmap' | 'weighted'
  const [loadErr, setLoadErr] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  const phononCandidates = useMemo(() => candidates.filter(c => c.model?.bands), [candidates]);
  const toggleOverlay = (id) => setHiddenOverlays(prev => (prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]));

  // Identify a Parent Brillouin-Zone reference for unfolding the active supercell/subgroup model:
  //   1. Prefer an explicit `kind === 'parent'` candidate (or smallest compatible unit-cell candidate).
  //   2. Otherwise, infer an intrinsic higher-symmetry parent cell from the supercell geometry itself.
  const parentReference = useMemo(() => {
    if (!model?.dynamicalMatrixPayload || !model?.baseStructure) return null;
    const activeCell = extractCellFromBaseStructure(model.baseStructure);
    if (!activeCell) return null;

    const otherCands = phononCandidates
      .filter(c => c.id !== activeCandidateId && c.model?.qPoints?.length && c.model?.baseStructure)
      .sort((a, b) => {
        const pa = a.kind === 'parent' ? 0 : 1;
        const pb = b.kind === 'parent' ? 0 : 1;
        if (pa !== pb) return pa - pb;
        return (a.model.baseStructure.uniqueRN?.length || 999) - (b.model.baseStructure.uniqueRN?.length || 999);
      });

    for (const cand of otherCands) {
      const pCell = extractCellFromBaseStructure(cand.model.baseStructure);
      const rel = findSupercellRelation(pCell, activeCell);
      if (rel && (rel.multiplicity > 1 || cand.kind === 'parent')) {
        return {
          label: cand.spaceGroup || cand.label || 'Parent',
          qPoints: cand.model.qPoints,
          kpathMeta: cand.model.kpathMeta,
          baseStructure: cand.model.baseStructure,
          relation: rel,
          candidateId: cand.id,
        };
      }
    }

    const inferred = inferIntrinsicParentPath(model);
    if (inferred) {
      const pCell = extractCellFromBaseStructure(inferred.baseStructure);
      const rel = findSupercellRelation(pCell, activeCell);
      if (rel && rel.multiplicity > 1) {
        return { ...inferred, relation: rel, candidateId: null };
      }
    }
    return null;
  }, [model, phononCandidates, activeCandidateId]);

  const unfoldedDispersion = useMemo(() => {
    if (!parentReference || !model?.dynamicalMatrixPayload) return null;
    return unfoldModelOntoParentPath(model, parentReference, { relation: parentReference.relation });
  }, [model, parentReference]);

  const isUnfoldedActive = Boolean(zoneMode === 'unfolded' && unfoldedDispersion);

  const elements = model ? Object.keys(model.baseStructure.atomDic) : [];
  const colorOf = (el) => elementColors[el] || DEFAULT_COLORS[el] || '#cccccc';

  const unit = thz ? 'THz' : 'meV';
  const escale = thz ? THZ_PER_MEV : 1;
  const fmtE = (e) => Number.isFinite(e) ? `${(e * escale).toFixed(3)} ${unit}` : '—';

  const activeRawBands = isUnfoldedActive ? unfoldedDispersion.bands : model?.bands;
  const activeQPoints = isUnfoldedActive ? unfoldedDispersion.qPoints : model?.qPoints;
  const activePlotStructure = isUnfoldedActive ? unfoldedDispersion.parentStructure : model?.baseStructure;
  const activeKpathMeta = isUnfoldedActive ? unfoldedDispersion.kpathMeta : model?.kpathMeta;
  const activeWeights = isUnfoldedActive ? unfoldedDispersion.weights : null;
  const activeEigvecs = isUnfoldedActive ? unfoldedDispersion.eigvecs : model?.eigvecs;
  const modeInspectorResults = useMemo(() => (
    model ? {
      bands: activeRawBands,
      qPoints: activeQPoints,
      eigvecs: activeEigvecs,
      baseStructure: model.baseStructure,
    } : null
  ), [model, activeRawBands, activeQPoints, activeEigvecs]);

  useEffect(() => {
    if (!activeRawBands?.length) { setSelK(0); setSelM(0); return; }
    setSelK(k => Math.max(0, Math.min(activeRawBands.length - 1, k)));
    setSelM(m => Math.max(0, Math.min((activeRawBands[0]?.length || 1) - 1, m)));
  }, [activeRawBands]);

  const displayBands = useMemo(
    () => !activeRawBands ? null : (escale === 1 ? activeRawBands : activeRawBands.map(row => row.map(v => v * escale))),
    [activeRawBands, escale]
  );

  const parseLoadedFile = async (file) => {
    if (!file) return;
    try {
      const m = modelFromText(await file.text());
      setLoadErr(null); setSelK(0); setSelM(0);
      onLoadModel(m, file.name);
    } catch (err) { setLoadErr(err.message); }
  };

  const overlayModels = useMemo(() => {
    if (!model || !activePlotStructure || !activeQPoints) return [];
    const targetPathSource = {
      qPoints: activeQPoints,
      kpathMeta: activeKpathMeta,
      baseStructure: activePlotStructure,
    };
    const targetCell = extractCellFromBaseStructure(activePlotStructure);

    return phononCandidates
      .filter(c => c.id !== activeCandidateId && !hiddenOverlays.includes(c.id))
      .map(c => {
        // Case A: This overlay IS the Parent reference of our currently unfolded Local model -> exact path match!
        if (isUnfoldedActive && parentReference?.candidateId === c.id) {
          return {
            id: c.id,
            label: c.label,
            color: c.color,
            bands: escale === 1 ? c.model.bands : c.model.bands.map(row => row.map(v => v * escale)),
            qPoints: c.model.qPoints,
            baseStructure: c.model.baseStructure,
            kpathMeta: c.model.kpathMeta,
            exactParentPath: true,
          };
        }
        // Case B: Active model is Parent (or unfolded onto Parent) and this overlay is a Local supercell with DM ->
        //         unfold the overlay directly onto `targetPathSource` so both share the exact Parent Brillouin Zone!
        if (c.model.dynamicalMatrixPayload && targetCell) {
          const ovCell = extractCellFromBaseStructure(c.model.baseStructure);
          const rel = findSupercellRelation(targetCell, ovCell);
          if (rel && (rel.multiplicity > 1 || c.kind === 'local-subgroup')) {
            const ovUnf = unfoldModelOntoParentPath(c.model, targetPathSource, { relation: rel });
            if (ovUnf) {
              return {
                id: c.id,
                label: `${c.label} (unfolded ×${rel.multiplicity})`,
                color: c.color,
                bands: escale === 1 ? ovUnf.bands : ovUnf.bands.map(row => row.map(v => v * escale)),
                weights: ovUnf.weights,
                qPoints: ovUnf.qPoints,
                baseStructure: ovUnf.parentStructure,
                kpathMeta: ovUnf.kpathMeta,
                exactParentPath: true,
              };
            }
          }
        }
        return {
          id: c.id,
          label: c.label,
          color: c.color,
          bands: escale === 1 ? c.model.bands : c.model.bands.map(row => row.map(v => v * escale)),
          qPoints: c.model.qPoints,
          baseStructure: c.model.baseStructure,
          kpathMeta: c.model.kpathMeta,
        };
      });
  }, [phononCandidates, activeCandidateId, hiddenOverlays, escale, model, activePlotStructure, activeQPoints, activeKpathMeta, isUnfoldedActive, parentReference]);

  const loadFile = (e) => parseLoadedFile(e.target.files?.[0]);

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    parseLoadedFile(e.dataTransfer?.files?.[0]);
  };

  // ── MOTIF-style editorial empty state ───────────────────────────────────
  if (!model) {
    return (
      <div
        className={`m-landing ${dragOver ? 'is-drag' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
      >
        <div className="m-landing-card">
          <h1 className="m-landing-title">
            Phonon bands, 3D modes &amp; simulated INS.
          </h1>
          <p className="m-landing-lead">
            Pin symmetry candidates &amp; run MLIP from the <b style={{ color: 'var(--ink)' }}>Symmetry</b> tab, or drop a saved{' '}
            <code style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9em', background: 'var(--muted)', padding: '1px 5px', borderRadius: 4 }}>_mlip_dm.json</code>{' '}
            or{' '}
            <code style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9em', background: 'var(--muted)', padding: '1px 5px', borderRadius: 4 }}>band.yaml</code>{' '}
            file anywhere on this page.
          </p>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <label className="m-button primary lg">
              Open band.yaml / .json…
              <input type="file" accept=".yaml,.yml,.json" onChange={loadFile} style={{ display: 'none' }} />
            </label>
          </div>
          {loadErr && (
            <div className="m-notice bad" style={{ margin: '16px 0 0' }}>{loadErr}</div>
          )}
        </div>
      </div>
    );
  }

  const eig = activeEigvecs?.[selK]?.[selM] || null;
  const qPointFor3D = isUnfoldedActive
    ? unfoldedDispersion.qPointsSuper?.[selK]
    : model.qPoints?.[selK];
  const nModes = activeRawBands?.[0]?.length || 0;
  const nK = activeRawBands?.length || 0;
  const energy = activeRawBands?.[selK]?.[selM];
  const selectedWeight = isUnfoldedActive ? unfoldedDispersion.weights?.[selK]?.[selM] : null;

  const exportYaml = () => downloadString(generatePhonopyBandYaml(model.baseStructure, model.qPoints, model.bands, model.eigvecs, model.kpathMeta), 'band_gpu.yaml', 'text/yaml');
  const exportJson = () => downloadString(generateBandJson(model.baseStructure, model.qPoints, model.bands, model.eigvecs, model.kpathMeta), 'band_gpu.json', 'application/json');
  const resetCam = () => setCamNonce('reset:' + Math.random());

  return (
    <div className={tab === 'modes' ? 'm-work' : 'm-work alone'}>
      {/* ════════ LEFT STAGE ════════ */}
      <div className="m-stage">
        {/* Top stage bar: view switch + file actions */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 14, flexWrap: 'wrap', paddingBottom: 10, borderBottom: '1px solid var(--line)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
            <div className="m-tools" role="tablist" aria-label="Viewer mode">
              <button type="button" role="tab" aria-selected={tab === 'modes'} className={tab === 'modes' ? 'on' : 'dim'} onClick={() => setTab('modes')}>
                Bands + Mode
              </button>
              <button type="button" role="tab" aria-selected={tab === 'sqe'} className={tab === 'sqe' ? 'on' : 'dim'} onClick={() => setTab('sqe')}>
                S(Q,E)
              </button>
            </div>

            {tab === 'modes' && (
              <div className="tabular-nums" style={{ display: 'inline-flex', alignItems: 'center', gap: 7, fontSize: 'var(--fz-small)', color: 'var(--secondary)' }}>
                <span>k</span>
                <input
                  type="number"
                  min={1}
                  max={nK}
                  value={selK + 1}
                  aria-label="k-point index"
                  onChange={e => setSelK(Math.max(0, Math.min(nK - 1, (parseInt(e.target.value, 10) || 1) - 1)))}
                  className="m-input mono"
                  style={{ width: 54, height: 26, textAlign: 'center', fontWeight: 600 }}
                />
                <span>band</span>
                <input
                  type="number"
                  min={1}
                  max={nModes}
                  value={selM + 1}
                  aria-label="band index"
                  onChange={e => setSelM(Math.max(0, Math.min(nModes - 1, (parseInt(e.target.value, 10) || 1) - 1)))}
                  className="m-input mono"
                  style={{ width: 48, height: 26, textAlign: 'center', fontWeight: 600 }}
                />
                {Number.isFinite(energy) && (
                  <b style={{ fontFamily: 'var(--font-mono)', color: energy < 0 ? 'var(--warn-ink)' : 'var(--ink)' }}>
                    {fmtE(energy)}
                  </b>
                )}
              </div>
            )}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <div className="m-tools" role="group" aria-label="Energy unit">
              <button type="button" className={!thz ? 'on' : 'dim'} onClick={() => setThz(false)}>meV</button>
              <button type="button" className={thz ? 'on' : 'dim'} onClick={() => setThz(true)}>THz</button>
            </div>
            <label className="m-button sm">
              Open…
              <input type="file" accept=".yaml,.yml,.json" onChange={loadFile} style={{ display: 'none' }} />
            </label>
            {model.eigvecs && (
              <>
                <button type="button" onClick={exportYaml} className="m-btn sm">Save YAML</button>
                <button type="button" onClick={exportJson} className="m-btn sm">Save JSON</button>
              </>
            )}
          </div>
        </div>

        {loadErr && <div className="m-notice bad" style={{ margin: 0 }}>{loadErr}</div>}

        {/* Multi-candidate switcher & band-overlay bar */}
        {phononCandidates.length > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', padding: '6px 10px', borderRadius: 6, background: 'var(--muted)', border: '1px solid var(--line)', fontSize: 12 }}>
            <span style={{ color: 'var(--secondary)', fontWeight: 600 }}>Candidates:</span>
            {phononCandidates.map(c => {
              const isActive = c.id === activeCandidateId;
              const isOverlaid = !hiddenOverlays.includes(c.id);
              return (
                <div
                  key={c.id}
                  style={{
                    display: 'inline-flex', alignItems: 'center', gap: 6, padding: '2px 8px',
                    borderRadius: 5,
                    background: isActive ? 'var(--surface)' : 'transparent',
                    border: `1px solid ${isActive ? 'var(--ink)' : 'var(--line)'}`,
                  }}
                >
                  <button
                    type="button"
                    onClick={() => onSelectCandidate?.(c.id)}
                    style={{ display: 'inline-flex', alignItems: 'center', gap: 6, background: 'none', border: 'none', padding: 0, cursor: 'pointer', color: 'var(--ink)', fontWeight: isActive ? 600 : 400, fontSize: 12 }}
                  >
                    <span style={{ width: 8, height: 8, borderRadius: '50%', background: c.color }} />
                    <span>{c.label}</span>
                  </button>
                  {!isActive && (
                    <label style={{ display: 'inline-flex', alignItems: 'center', gap: 3, fontSize: 11, color: 'var(--secondary)', cursor: 'pointer', marginLeft: 4 }}>
                      <input type="checkbox" checked={isOverlaid} onChange={() => toggleOverlay(c.id)} />
                      overlay
                    </label>
                  )}
                  {phononCandidates.length > 1 && onRemoveCandidate && (
                    <button
                      type="button"
                      onClick={() => onRemoveCandidate(c.id)}
                      style={{ background: 'none', border: 'none', padding: '0 2px', cursor: 'pointer', color: 'var(--faint)', fontSize: 13 }}
                      title="Remove candidate"
                    >
                      ×
                    </button>
                  )}
                </div>
              );
            })}
            {onGoToIns && (
              <button type="button" className="m-btn sm active" style={{ marginLeft: 'auto' }} onClick={onGoToIns}>
                3 · Compare with INS →
              </button>
            )}
          </div>
        )}

        {/* ── TAB 1: BANDS + 3D MODE ── */}
        {tab === 'modes' && (
          <>
            {/* Quiet MOTIF-style figure toolbar */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
              <div className="m-tools">
                <span style={{ fontSize: 'var(--fz-small)', color: 'var(--faint)', marginRight: 4 }}>along</span>
                {['x', 'y', 'z'].map(ax => (
                  <button key={ax} type="button" onClick={() => setCamNonce(ax + Math.random())} title={`View along ${ax}`}>
                    {ax}
                  </button>
                ))}
                <button type="button" onClick={resetCam} style={{ marginLeft: 6 }}>Reset view</button>
                <button type="button" onClick={() => setPlaying(p => !p)} style={{ marginLeft: 6 }}>
                  {playing ? 'Pause' : 'Play'}
                </button>
              </div>

              <div style={{ display: 'inline-flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', fontSize: 'var(--fz-small)', color: 'var(--secondary)' }}>
                {unfoldedDispersion && (
                  <div className="m-tools" role="group" aria-label="Brillouin zone representation" style={{ gap: 6 }}>
                    <button
                      type="button"
                      aria-pressed={zoneMode === 'unfolded'}
                      className={zoneMode === 'unfolded' ? 'on' : 'dim'}
                      onClick={() => { setZoneMode('unfolded'); setBandReset(n => n + 1); }}
                      title={`Project ${unfoldedDispersion.relation.nSuperAtoms}-atom supercell modes onto the ${unfoldedDispersion.relation.nParentAtoms}-atom parent Brillouin zone (×${unfoldedDispersion.relation.multiplicity})`}
                    >
                      Unfolded BZ (×{unfoldedDispersion.relation.multiplicity})
                    </button>
                    <button
                      type="button"
                      aria-pressed={zoneMode === 'native'}
                      className={zoneMode === 'native' ? 'on' : 'dim'}
                      onClick={() => { setZoneMode('native'); setBandReset(n => n + 1); }}
                      title="Plot folded branches in the supercell's own smaller Brillouin zone"
                    >
                      Supercell BZ
                    </button>
                  </div>
                )}

                {isUnfoldedActive && (
                  <div className="m-tools" role="group" aria-label="Spectral display style" style={{ gap: 5 }}>
                    {[
                      ['both', 'A(Q,E) + Modes'],
                      ['heatmap', 'A(Q,E) Map'],
                      ['weighted', 'Weighted Modes'],
                    ].map(([k, lbl]) => (
                      <button
                        key={k}
                        type="button"
                        aria-pressed={spectralStyle === k}
                        className={spectralStyle === k ? 'on' : 'dim'}
                        onClick={() => setSpectralStyle(k)}
                      >
                        {lbl}
                      </button>
                    ))}
                  </div>
                )}

                <span>E range</span>
                <input
                  type="number"
                  placeholder="min"
                  value={eMin}
                  onChange={e => setEMin(e.target.value)}
                  className="m-input mono"
                  style={{ width: 56, height: 26, textAlign: 'center' }}
                />
                <span>–</span>
                <input
                  type="number"
                  placeholder="max"
                  value={eMax}
                  onChange={e => setEMax(e.target.value)}
                  className="m-input mono"
                  style={{ width: 56, height: 26, textAlign: 'center' }}
                />
                <button
                  type="button"
                  onClick={() => { setEMin(''); setEMax(''); setBandReset(n => n + 1); }}
                  className="m-btn subtle sm"
                >
                  Reset
                </button>
              </div>
            </div>

            {/* Side-by-side Figures (adapts via .ph-fig-grid container query & layout mode) */}
            <div className="ph-fig-grid">
              {/* Band Structure */}
              <div className="m-frame" style={{ padding: '10px 12px', display: 'flex', flexDirection: 'column' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 8, flexWrap: 'wrap', marginBottom: 4 }}>
                  <span style={{ fontWeight: 600, fontSize: 'var(--fz-control)' }}>
                    {isUnfoldedActive
                      ? `Unfolded spectral dispersion A(Q,E) · ×${unfoldedDispersion.relation.multiplicity} → ${parentReference?.label || 'Parent BZ'}`
                      : 'Phonon band structure'}
                  </span>
                  <span style={{ fontSize: 'var(--fz-small)', color: 'var(--faint)' }}>
                    {isUnfoldedActive
                      ? `RMS static shift ${unfoldedDispersion.relation.rmsDispAng.toFixed(3)} Å · click mode to animate`
                      : 'click a point to inspect'}
                  </span>
                </div>
                <BandStructurePlot
                  bands={displayBands}
                  weights={activeWeights}
                  spectralStyle={spectralStyle}
                  qPoints={activeQPoints}
                  baseStructure={activePlotStructure}
                  kpathMeta={activeKpathMeta}
                  selected={{ k: selK, m: selM }}
                  unit={unit}
                  resetSignal={bandReset}
                  eMin={eMin === '' ? undefined : parseFloat(eMin)}
                  eMax={eMax === '' ? undefined : parseFloat(eMax)}
                  overlays={overlayModels}
                  onPick={(k, m) => { setSelK(k); setSelM(m); }}
                />
              </div>

              {/* 3D Mode Displacement */}
              <div className="m-frame" style={{ display: 'flex', flexDirection: 'column' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 14px', borderBottom: '1px solid var(--line)' }}>
                  <span style={{ fontWeight: 600, fontSize: 'var(--fz-control)' }}>Mode displacement</span>
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8, fontFamily: 'var(--font-mono)', fontSize: 'var(--fz-small)', color: 'var(--faint)' }}>
                    {recording && (
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, color: 'var(--warn-ink)', fontWeight: 700 }}>
                        <span style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--warn-ink)', animation: 'blip 1s infinite' }} />
                        REC
                      </span>
                    )}
                    <span>{nx}×{ny}×{nz} cell</span>
                  </span>
                </div>

                <div className="ph-mode-canvas">
                  {activeEigvecs ? (
                    <CrystalViewer
                      baseStructure={model.baseStructure}
                      eigenvector={eig}
                      qPoint={qPointFor3D}
                      isPlaying={playing}
                      amplitude={amplitude}
                      speed={speed}
                      supercell={supercell}
                      showVectors={showVectors}
                      showCell={showCell}
                      atomScale={atomScale}
                      cameraAxis={camNonce}
                      elementColors={elementColors}
                      elementRadii={elementRadii}
                      displayStyle={displayStyle}
                      showBonds={showBonds}
                      bondCutoff={bondCutoff}
                      bondThickness={bondThickness}
                      bondRules={bondRules}
                      shading={shading}
                      shadingStrength={shadingStrength}
                      recording={recording}
                      gifSignal={gifSignal}
                      vectorScale={vectorScale}
                      vectorColor={vectorColor}
                    />
                  ) : (
                    <span style={{ color: 'var(--faint)', fontSize: 'var(--fz-control)' }}>
                      Loaded file has no eigenvectors — 3D modes unavailable.
                    </span>
                  )}
                </div>

                {activeEigvecs && (
                  <ModeInspector
                    results={modeInspectorResults}
                    selectedK={selK}
                    selectedMode={selM}
                    unit={unit}
                    scale={escale}
                    colors={elementColors}
                    spectralWeight={selectedWeight}
                  />
                )}

                <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', padding: '8px 14px', borderTop: '1px solid var(--line)', fontSize: 'var(--fz-small)', color: 'var(--secondary)' }}>
                  {elements.map(el => (
                    <span key={el} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                      <span style={{ width: 10, height: 10, borderRadius: '50%', background: colorOf(el), boxShadow: 'inset 0 0 0 1px rgba(0,0,0,0.16)' }} />
                      <b style={{ color: 'var(--ink)' }}>{el}</b>
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </>
        )}

        {/* ── TAB 2: S(Q,E) ── */}
        {tab === 'sqe' && (
          model.eigvecs ? (
            <InsPanel results={model} temperature={model.temperature} />
          ) : (
            <div className="m-rule-box" style={{ color: 'var(--secondary)' }}>
              Loaded file has no eigenvectors — simulated S(Q,E) requires eigenvectors for the dynamical structure factor.
            </div>
          )
        )}
      </div>

      {/* ════════ RIGHT INSPECTOR COLUMN (MOTIF .side) ════════ */}
      {tab === 'modes' && (
        <aside className="m-side" aria-label="Mode controls">
          <div className="m-tabs" role="tablist">
            {[
              ['motion', 'Motion'],
              ['visual', 'Visual'],
              ['species', 'Species'],
              ['info', 'Info'],
            ].map(([id, label]) => (
              <button
                key={id}
                type="button"
                role="tab"
                aria-selected={sideTab === id}
                className={sideTab === id ? 'on' : ''}
                onClick={() => setSideTab(id)}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="m-tab-body">
            {/* ── MOTION TAB ── */}
            {sideTab === 'motion' && (
              <>
                <details open className="m-section">
                  <summary><span>Oscillation</span></summary>
                  <div className="m-section-body">
                    <SliderRow label="amplitude" value={amplitude} valueLabel={`×${amplitude.toFixed(1)}`} min={0} max={10} step={0.1} onChange={setAmplitude} />
                    <SliderRow label="speed" value={speed} valueLabel={speed.toFixed(2)} min={0.01} max={0.3} step={0.01} onChange={setSpeed} />
                    <div className="m-row" style={{ marginTop: 12 }}>
                      <button type="button" onClick={() => setPlaying(p => !p)} className={`m-btn ${playing ? '' : 'primary'}`} style={{ flex: 1 }}>
                        {playing ? 'Pause' : 'Play'}
                      </button>
                      <button
                        type="button"
                        onClick={() => setRecording(r => !r)}
                        className={`m-btn ${recording ? 'danger' : ''}`}
                        title="Record WebM video"
                      >
                        {recording ? 'Stop' : 'WebM'}
                      </button>
                      <button
                        type="button"
                        onClick={() => setGifSignal(s => s + 1)}
                        className="m-btn"
                        title="Capture ~50 frames as an animated GIF"
                      >
                        GIF
                      </button>
                    </div>
                  </div>
                </details>

                <details open className="m-section">
                  <summary><span>Displacement vectors</span></summary>
                  <div className="m-section-body">
                    <div className="m-row">
                      <label style={{ display: 'inline-flex', alignItems: 'center', gap: 8, cursor: 'pointer', flex: 1 }}>
                        <input type="checkbox" checked={showVectors} onChange={e => setShowVectors(e.target.checked)} />
                        Show eigenvector arrows
                      </label>
                      {showVectors && (
                        <input
                          type="color"
                          value={vectorColor}
                          onChange={e => setVectorColor(e.target.value)}
                          title="Vector colour"
                          style={{ width: 26, height: 26, padding: 2, border: '1px solid var(--control)', borderRadius: '50%', background: 'var(--surface)', cursor: 'pointer' }}
                        />
                      )}
                    </div>
                    {showVectors && (
                      <SliderRow label="vector size" value={vectorScale} valueLabel={`×${vectorScale.toFixed(1)}`} min={0.2} max={6} step={0.1} onChange={setVectorScale} />
                    )}
                  </div>
                </details>
              </>
            )}

            {/* ── VISUAL TAB ── */}
            {sideTab === 'visual' && (
              <>
                <details open className="m-section">
                  <summary><span>Supercell</span></summary>
                  <div className="m-section-body">
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 8 }}>
                      {[['nx', nx, setNx], ['ny', ny, setNy], ['nz', nz, setNz]].map(([l, v, set]) => (
                        <label key={l} style={{ display: 'flex', flexDirection: 'column', gap: 3, fontSize: 'var(--fz-small)', color: 'var(--secondary)' }}>
                          <span>{l}</span>
                          <input
                            type="number"
                            min={1}
                            max={6}
                            value={v}
                            onChange={e => set(Math.max(1, Math.min(6, parseInt(e.target.value, 10) || 1)))}
                            className="m-input mono"
                            style={{ width: '100%', textAlign: 'center' }}
                          />
                        </label>
                      ))}
                    </div>
                    <div className="m-row" style={{ marginTop: 10 }}>
                      <label style={{ display: 'inline-flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                        <input type="checkbox" checked={showCell} onChange={e => setShowCell(e.target.checked)} />
                        Show unit-cell box
                      </label>
                    </div>
                  </div>
                </details>

                <details open className="m-section">
                  <summary><span>Atoms &amp; style</span></summary>
                  <div className="m-section-body">
                    <div className="m-row">
                      <span className="m-lab">style</span>
                      <select value={displayStyle} onChange={e => setDisplayStyle(e.target.value)} className="m-select" style={{ flex: 1 }}>
                        <option value="ballstick">Ball &amp; stick</option>
                        <option value="spacefill">Spacefill</option>
                        <option value="wireframe">Wireframe</option>
                      </select>
                    </div>
                    <SliderRow label="atom size" value={atomScale} valueLabel={`×${atomScale.toFixed(1)}`} min={0.3} max={3} step={0.1} onChange={setAtomScale} />
                    <div className="m-row">
                      <label style={{ display: 'inline-flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                        <input type="checkbox" checked={shading} onChange={e => setShading(e.target.checked)} />
                        Directional shading
                      </label>
                    </div>
                    {shading && (
                      <SliderRow label="relief" value={shadingStrength} valueLabel={shadingStrength.toFixed(2)} min={0} max={1} step={0.05} onChange={setShadingStrength} />
                    )}
                  </div>
                </details>

                <details open className="m-section">
                  <summary><span>Bonds</span></summary>
                  <div className="m-section-body">
                    <div className="m-row">
                      <label style={{ display: 'inline-flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                        <input type="checkbox" checked={showBonds} onChange={e => setShowBonds(e.target.checked)} />
                        Show bonds
                      </label>
                    </div>
                    {showBonds && (
                      <>
                        <SliderRow label="cutoff" value={bondCutoff} valueLabel={`${bondCutoff.toFixed(2)} Å`} min={1.5} max={4.5} step={0.05} onChange={setBondCutoff} />
                        <SliderRow label="thickness" value={bondThickness} valueLabel={bondThickness.toFixed(2)} min={0.02} max={0.3} step={0.01} onChange={setBondThickness} />
                        {elements.length > 1 && (
                          <div style={{ marginTop: 10 }}>
                            <div style={{ fontSize: 'var(--fz-small)', color: 'var(--secondary)', marginBottom: 6 }}>Per-pair cutoffs (Å)</div>
                            {elementPairs(elements).map(([a, b]) => {
                              const key = [a, b].sort().join('-');
                              return (
                                <div key={key} className="m-row" style={{ margin: '4px 0' }}>
                                  <span className="m-lab" style={{ fontFamily: 'var(--font-mono)' }}>{a}–{b}</span>
                                  <input
                                    type="number"
                                    step={0.05}
                                    min={0}
                                    placeholder="auto"
                                    value={bondRules[key] ?? ''}
                                    onChange={e => setBondRules(r => {
                                      const n = { ...r };
                                      if (e.target.value === '') delete n[key];
                                      else n[key] = parseFloat(e.target.value);
                                      return n;
                                    })}
                                    className="m-input mono"
                                    style={{ width: 84 }}
                                  />
                                </div>
                              );
                            })}
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </details>
              </>
            )}

            {/* ── SPECIES TAB ── */}
            {sideTab === 'species' && (
              <details open className="m-section">
                <summary><span>Atom types</span></summary>
                <div className="m-section-body">
                  {elements.map(el => (
                    <div key={el} className="m-row">
                      <span style={{ width: 28, fontWeight: 600, fontFamily: 'var(--font-mono)' }}>{el}</span>
                      <input
                        type="color"
                        value={colorOf(el)}
                        onChange={e => setElementColors(p => ({ ...p, [el]: e.target.value }))}
                        title={`${el} colour`}
                        style={{ width: 26, height: 26, padding: 2, border: '1px solid var(--control)', borderRadius: '50%', background: 'var(--surface)', cursor: 'pointer', flex: 'none' }}
                      />
                      <input
                        type="number"
                        min={0.1}
                        max={3}
                        step={0.05}
                        value={elementRadii[el] ?? (COVALENT_R[el] || 1.0)}
                        title="Covalent radius (Å)"
                        onChange={e => {
                          const raw = e.target.value;
                          setElementRadii(r => {
                            const next = { ...r };
                            const v = parseFloat(raw);
                            if (raw === '' || !Number.isFinite(v) || v <= 0) delete next[el];
                            else next[el] = v;
                            return next;
                          });
                        }}
                        className="m-input mono"
                        style={{ flex: 1, minWidth: 0 }}
                      />
                      <span style={{ fontSize: 'var(--fz-small)', color: 'var(--faint)' }}>Å</span>
                    </div>
                  ))}
                  <div style={{ marginTop: 12 }}>
                    <button
                      type="button"
                      onClick={() => { setElementColors({}); setElementRadii({}); }}
                      className="m-btn subtle sm"
                    >
                      Reset colours &amp; radii
                    </button>
                  </div>
                </div>
              </details>
            )}

            {/* ── INFO TAB ── */}
            {sideTab === 'info' && <StructureTables model={model} />}
          </div>
        </aside>
      )}
    </div>
  );
}

/* ── Helpers ─────────────────────────────────────────────────────────────── */

function elementPairs(elements) {
  const out = [];
  for (let i = 0; i < elements.length; i++) for (let j = i; j < elements.length; j++) out.push([elements[i], elements[j]]);
  return out;
}

function SliderRow({ label, value, valueLabel, min, max, step, onChange }) {
  return (
    <div className="m-row" style={{ flexWrap: 'nowrap' }}>
      <span className="m-lab">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        className="m-range"
        style={{ flex: 1, minWidth: 0 }}
      />
      <span className="tabular-nums" style={{ width: 48, textAlign: 'right', fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--ink)' }}>
        {valueLabel}
      </span>
    </div>
  );
}

function StructureTables({ model }) {
  const bs = model.baseStructure;
  const A = [bs.v1, bs.v2, bs.v3];
  const rev = {};
  for (const [el, idxs] of Object.entries(bs.atomDic)) idxs.forEach(i => { rev[i] = el; });
  const sites = (bs.uniqueRN || []).map((rn, r) => ({
    el: rev[rn] || '?', rn,
    pos: [bs.hsym_xyz[r * 3], bs.hsym_xyz[r * 3 + 1], bs.hsym_xyz[r * 3 + 2]],
  }));
  return (
    <>
      <details open className="m-section">
        <summary><span>Lattice (Å)</span></summary>
        <div className="m-section-body">
          <table className="tabular-nums" style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--font-mono)', fontSize: 12 }}>
            <tbody>
              {A.map((v, i) => (
                <tr key={i}>
                  <td style={{ padding: '5px 8px 5px 0', color: 'var(--secondary)', borderBottom: i < 2 ? '1px solid var(--line)' : 'none' }}>{'abc'[i]}</td>
                  {v.map((x, j) => (
                    <td key={j} style={{ padding: '5px 0', textAlign: 'right', color: 'var(--ink)', borderBottom: i < 2 ? '1px solid var(--line)' : 'none' }}>
                      {x.toFixed(3)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>

      <details open className="m-section">
        <summary><span>Atom positions ({sites.length})</span></summary>
        <div className="m-section-body">
          <table className="tabular-nums" style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--font-mono)', fontSize: 11.5 }}>
            <thead>
              <tr style={{ color: 'var(--faint)', borderBottom: '1px solid var(--line)' }}>
                <th style={{ textAlign: 'left', fontWeight: 500, paddingBottom: 4 }}>el</th>
                <th style={{ textAlign: 'right', fontWeight: 500, paddingBottom: 4 }}>x</th>
                <th style={{ textAlign: 'right', fontWeight: 500, paddingBottom: 4 }}>y</th>
                <th style={{ textAlign: 'right', fontWeight: 500, paddingBottom: 4 }}>z</th>
              </tr>
            </thead>
            <tbody>
              {sites.map((s, i) => (
                <tr key={i} style={{ borderBottom: i < sites.length - 1 ? '1px solid var(--line)' : 'none' }}>
                  <td style={{ padding: '4px 0', fontWeight: 600, color: 'var(--ink)' }}>{s.el}</td>
                  {s.pos.map((x, j) => (
                    <td key={j} style={{ padding: '4px 0', textAlign: 'right', color: 'var(--secondary)' }}>{x.toFixed(4)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </>
  );
}
