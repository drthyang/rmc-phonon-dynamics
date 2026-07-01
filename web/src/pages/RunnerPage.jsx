import React, { useState, useMemo, useRef, useEffect } from 'react';
import { listConfigs, readBaseStructure, findStructureFile, listRmc6f } from '../io/readers';
import { conventionalLattice, buildKPathFromSegments } from '../math/reciprocal';
import { analyzeBravais } from '../math/bravais';
import { IDENT, det3, vecMat3, buildCellLabeling } from '../math/cells';
import { findSpaceGroupOps } from '../math/symmetry';
import { buildConventionalBZModel, displayLabel } from '../math/highsym';
import { phononDOS } from '../math/dos';
import { DEFAULT_COLORS } from '../constants';
import { modelFromText } from '../io/phonopyDM';
import BrillouinZoneViewer from '../components/BrillouinZoneViewer';
import CrystalViewer from '../components/CrystalViewer';
import FitQuality from '../components/FitQuality';
import SciChart from '../components/SciChart';

/* ── style tokens ─────────────────────────────────────────────────────── */
const INK = 'var(--ink)', DIM = 'var(--dim)', FAINT = 'var(--faint)';
const ACCENT = 'var(--accent)', ACCENTINK = 'var(--accentInk)', BORDER = 'var(--border)';
const cardTitle = { font: "600 13px 'Space Grotesk'", letterSpacing: '.01em', color: INK };
const eyebrow = { font: "10px 'Space Mono'", letterSpacing: '.16em', color: FAINT };
const stepBtn = { border: 'none', background: 'transparent', cursor: 'pointer', width: 24, height: 16, display: 'flex', alignItems: 'center', justifyContent: 'center', color: DIM, fontSize: 9, lineHeight: 1 };

const SUB = { '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉' };
const sub = (n) => String(n).split('').map(c => SUB[c] || c).join('');

/**
 * Runner page (Cobalt redesign) — three numbered groups:
 *   1 · Data & assessment   (folder, crystal preview, fit quality)
 *   2 · Reciprocal space & k-path   (Brillouin zone, editable segments)
 *   3 · Run calculation   (displacement reference, parameters, launch + log)
 *
 * The compute/io/math layers are untouched; this only reshapes the UI around
 * the same data shapes and the live PhononPipeline.
 */
export default function RunnerPage({ pipeline, ready, onResults, onLoadResult }) {
  const [dirHandle, setDirHandle] = useState(null);
  const [filesList, setFilesList] = useState([]);
  const [configFamily, setConfigFamily] = useState(null);
  const [rmc6fList, setRmc6fList] = useState([]);
  const [structureName, setStructureName] = useState(null);
  const [baseStructure, setBaseStructure] = useState(null);

  const [refMode, setRefMode] = useState('average');   // 'average' | 'file'  (reference SOURCE)
  const [referenceMode, setReferenceMode] = useState('per-atom'); // 'per-atom' | 'symmetrized' (cell reference)
  const [cellType, setCellType] = useState('conventional'); // 'conventional' | 'primitive' | 'custom'
  const [customN, setCustomN] = useState([1, 1, 1]);   // P = diag(n) for a custom supercell
  const [symTol, setSymTol] = useState(0.5);           // Å tolerance for symmetry detection (loose — basis is a single config)
  const [refName, setRefName] = useState('');

  const [temperature, setTemperature] = useState(5);
  const [degenerateTol, setDegenerateTol] = useState(5e-3);
  const [density, setDensity] = useState(20);           // pts / Å⁻¹

  const [bzSegments, setBzSegments] = useState([]);     // [{from,to}]
  const [pointsConv, setPointsConv] = useState({});     // label -> conventional fractional
  const [segNpoints, setSegNpoints] = useState({});     // {segIndex: npoints override}

  const [runDos, setRunDos] = useState(false);
  const [dosN, setDosN] = useState(10);                 // q-grid is N × N × N
  const [flaggedConfigs, setFlaggedConfigs] = useState([]); // config #s flagged by fit quality
  const [flagSigma, setFlagSigma] = useState(2);
  const [excludeBad, setExcludeBad] = useState(false);  // drop flagged configs from the run

  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressText, setProgressText] = useState('');
  const [logLines, setLogLines] = useState([]);
  const [dosEnergies, setDosEnergies] = useState(null);

  const lastMsg = useRef('');
  const logEl = useRef(null);
  const cell111 = useMemo(() => [1, 1, 1], []);
  const pushLog = (t) => { if (t && t !== lastMsg.current) { lastMsg.current = t; setLogLines(l => [...l, t]); } };
  useEffect(() => { if (logEl.current) logEl.current.scrollTop = logEl.current.scrollHeight; }, [logLines]);

  // Bravais / BZ — memoized for STABLE identity (keeps the three.js cameras).
  const bravais = useMemo(
    () => (baseStructure?.v1 && baseStructure.basis
      ? analyzeBravais(conventionalLattice(baseStructure.v1, baseStructure.v2, baseStructure.v3, baseStructure.dim), baseStructure.basis)
      : null),
    [baseStructure]
  );
  // Cell-framework default (Phase 1): compute over the CONVENTIONAL cell, so the
  // k-path uses conventional high-symmetry points (X at ½). This fixes the
  // spurious Γ→X mirror symmetry the primitive seekpath path produced.
  const bzModel = useMemo(() => (bravais ? buildConventionalBZModel(bravais) : null), [bravais]);

  // Computation cell: P = I (conventional), M (primitive, unfolded) or diag(n)
  // (custom supercell). The path is still picked on the conventional BZ; the
  // pipeline maps q → P·q internally and groups S(k) by that cell's basis sites.
  const compP = useMemo(() => {
    if (cellType === 'custom') return [[customN[0], 0, 0], [0, customN[1], 0], [0, 0, customN[2]]];
    if (cellType === 'primitive') return bravais?.M || IDENT;
    return IDENT;
  }, [cellType, customN, bravais]);
  const isCentered = !!bravais && bravais.centering !== 'P';
  const nConvBasis = baseStructure?.basis?.length || 0;
  // Actual basis-site count for the chosen cell. Custom supercells always multiply
  // (n₁n₂n₃ × conventional). Conventional/primitive are SUB-cells, so we relabel
  // the reference basis to get the TRUE fold — a primitive cell only reduces to
  // ¼ (FCC) when the average positions still respect the centering; a disorder-
  // broken RMC average may not fold, and the hint must show that honestly.
  const cellInfo = useMemo(() => {
    if (!bravais || nConvBasis === 0) return { nBasis: 0, ideal: 0 };
    const idealMult = cellType === 'custom' ? customN[0] * customN[1] * customN[2] : Math.abs(det3(compP));
    const ideal = Math.max(1, Math.round(nConvBasis * idealMult));
    if (cellType === 'custom') return { nBasis: ideal, ideal, residual: 0 };
    const b = baseStructure.basis;
    const avgPos = b.map(s => vecMat3(s.frac, bravais.A_conv));
    const lab = buildCellLabeling(avgPos, b.map(s => s.rn), b.map(() => 1), bravais.A_conv, compP, { tol: 0.08 });
    return { nBasis: lab.nBasis, ideal, residual: lab.maxResidual || 0 };
  }, [bravais, baseStructure, nConvBasis, cellType, customN, compP]);
  const nBasis = cellInfo.nBasis;
  const nBranches = 3 * nBasis;

  // Detected symmetry of the reference structure (pure-JS, offline). The basis is
  // one representative config per site (not the ensemble mean), so its symmetry is
  // tolerance-dependent — hence an adjustable tol: trace how the space-group order
  // grows as you loosen it. Report-only (does not drive folding yet).
  const symInfo = useMemo(() => {
    if (!bravais || !baseStructure?.basis) return null;
    const basis = baseStructure.basis.map(s => ({ el: s.el, frac: s.frac }));
    return findSpaceGroupOps(bravais.A_conv, basis, symTol);
  }, [bravais, baseStructure, symTol]);
  const primitiveNoFold = cellType === 'primitive' && cellInfo.ideal > 0 && nBasis > cellInfo.ideal * 1.5;
  // How much symmetry the fold imposes (RMS Å of folded sites from the symmetrized
  // site). Only meaningful when the cell actually folds (primitive).
  const foldsSites = nBasis < nConvBasis;
  const residual = cellInfo.residual || 0;
  const residualHigh = foldsSites && residual > 0.3;

  const previewStruct = useMemo(() => {
    if (!baseStructure?.basis) return null;
    const b = baseStructure.basis;
    return {
      v1: baseStructure.v1, v2: baseStructure.v2, v3: baseStructure.v3, dim: baseStructure.dim,
      atomDic: baseStructure.atomDic,
      uniqueRN: b.map(x => x.rn), atomType: b.map(x => x.rn),
      hsym_xyz: Float64Array.from(b.flatMap(x => x.frac)),
      cellIdx: new Float64Array(b.length * 3),
    };
  }, [baseStructure]);

  // Crystallographic readout (unit cell from supercell vectors / dim).
  const readout = useMemo(() => {
    if (!baseStructure?.v1) return null;
    const { v1, v2, v3, dim, atomDic } = baseStructure;
    const norm = (v) => Math.hypot(v[0], v[1], v[2]);
    const ang = (u, v) => Math.acos(Math.max(-1, Math.min(1, (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (norm(u) * norm(v))))) * 180 / Math.PI;
    const a = norm(v1) / (dim?.[0] || 1), b = norm(v2) / (dim?.[1] || 1), c = norm(v3) / (dim?.[2] || 1);
    const counts = Object.entries(atomDic).map(([el, arr]) => [el, arr.length]);
    const sites = counts.reduce((s, [, n]) => s + n, 0);
    const formula = counts.map(([el, n]) => n > 1 ? `${el}${sub(n)}` : el).join(' ');
    return {
      formula, sites,
      a: a.toFixed(3), b: b.toFixed(3), c: c.toFixed(3),
      al: ang(v2, v3).toFixed(0), be: ang(v1, v3).toFixed(0), ga: ang(v1, v2).toFixed(0),
      supercell: dim ? dim.join(' × ') : '—',
    };
  }, [baseStructure]);

  const loadStructure = async (handle, name) => {
    const info = await readBaseStructure(handle);
    setBaseStructure(info); setStructureName(name);
  };

  const handleSelectFolder = async () => {
    try {
      const dh = await window.showDirectoryPicker({ mode: 'read' });
      setDirHandle(dh);
      const { files, family } = await listConfigs(dh);
      setFilesList(files); setConfigFamily(family);
      setRmc6fList(await listRmc6f(dh));
      if (family === 'rmc6f' && files.length > 0) await loadStructure(files[0], files[0].name);
      else if (family === 'frac') {
        const sh = await findStructureFile(dh);
        if (!sh) { setProgressText('Frac configs found but no .rmc6f structure file in this folder.'); setBaseStructure(null); return; }
        await loadStructure(sh, sh.name);
      } else setBaseStructure(null);
    } catch (err) { console.error(err); }
  };

  const onStructureChange = async (name) => {
    const item = rmc6fList.find(r => r.name === name);
    if (item) await loadStructure(item.handle, name);
  };

  // Per-segment reciprocal-space length (Å⁻¹) → density-driven npoints.
  const segLen = (s) => {
    const a = bzModel?.points?.[s.from]?.cart, b = bzModel?.points?.[s.to]?.cart;
    if (!a || !b) return 1;
    return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  };
  const defN = (s) => Math.max(2, Math.round(density * segLen(s)));
  const segments = bzSegments.map((s, i) => ({ from: s.from, to: s.to, npoints: segNpoints[i] ?? defN(s) }));
  const totalK = segments.reduce((a, s) => a + Math.max(2, s.npoints), 0);

  const setSegN = (i, n) => setSegNpoints(m => ({ ...m, [i]: Math.max(2, n) }));
  const onPathChange = (segs, conv) => { setBzSegments(segs); setPointsConv(conv); setSegNpoints({}); };

  // Map a structure-file handle → its config number (…_N.rmc6f / …N.txt), so the
  // fit-quality-flagged configs can be excluded from the run.
  const configNumOf = (h) => { const m = (h?.name || '').match(/(\d+)\.(?:rmc6f|txt)$/i); return m ? +m[1] : null; };
  const flaggedSet = useMemo(() => new Set(flaggedConfigs), [flaggedConfigs]);
  const flaggedInRun = flaggedSet.size ? filesList.filter(f => flaggedSet.has(configNumOf(f))).length : 0;

  const run = async () => {
    if (!filesList.length || !baseStructure) return;
    if (!pipeline) { setProgressText('Compute engine still initializing — try again in a moment.'); return; }
    if (segments.length < 1) { setProgressText('Build a k-path first (click ≥2 high-symmetry points).'); return; }
    const runFiles = (excludeBad && flaggedSet.size)
      ? filesList.filter(f => !flaggedSet.has(configNumOf(f))) : filesList;
    if (!runFiles.length) { setProgressText('Every configuration is flagged/excluded — raise the σ threshold or turn off exclusion.'); return; }
    setIsProcessing(true); setProgress(0); setDosEnergies(null);
    lastMsg.current = '';
    const excluded = filesList.length - runFiles.length;
    setLogLines([`▶ Run started · T = ${temperature} K · degen tol ${degenerateTol}`
      + (excluded ? ` · excluding ${excluded} configs flagged > ${flagSigma}σ (${runFiles.length} used)` : '')]);
    setProgressText('Starting…');
    try {
      const { qFrac, segSizes, hsymIndex } = buildKPathFromSegments(pointsConv, segments);
      const hsymDisplay = {};
      for (const [k, v] of Object.entries(hsymIndex)) hsymDisplay[k] = displayLabel(v);
      const kpathMeta = { qFrac, segSizes, hsymIndex: hsymDisplay };

      let referenceHandle = null;
      if (refMode === 'file') {
        const item = rmc6fList.find(r => r.name === refName);
        if (!item) { setProgressText('Select an equilibrium .rmc6f for the file reference.'); return; }
        referenceHandle = item.handle;
      }

      pipeline.onProgress = (p, t) => { setProgress(p); setProgressText(t); pushLog(t); };
      const res = await pipeline.runCalculation(runFiles, configFamily, baseStructure, qFrac, temperature, 50, { referenceHandle, degenerateTol, referenceMode, computationCell: { P: compP } });

      let dos = null;
      if (runDos) {
        pushLog(`Computing phonon DOS · q-grid ${dosN}³…`);
        try {
          const d = await pipeline.computeDOSGrid(runFiles, configFamily, baseStructure, dosN, temperature, 50, { referenceHandle, referenceMode, computationCell: { P: compP } });
          setDosEnergies(d.energies);
          dos = { energies: d.energies, gridN: d.gridN };
          pushLog(`Phonon DOS · ${d.nq} q-points × ${d.nModes} modes`);
        } catch (e) { if (e.message === 'cancelled') throw e; pushLog('DOS failed: ' + e.message); }
      }
      pushLog('Done — opening viewer…');
      setProgressText('Done — opening viewer…');
      onResults(res, kpathMeta, dos);
    } catch (e) {
      if (e.message === 'cancelled') { setProgressText('Cancelled.'); pushLog('■ Cancelled by user'); }
      else { console.error(e); setProgressText('Error: ' + e.message); pushLog('Error: ' + e.message); }
    } finally {
      setIsProcessing(false);
    }
  };

  const dosCurve = useMemo(() => {
    if (!dosEnergies) return null;
    const v = Array.from(dosEnergies).filter(x => Number.isFinite(x) && x > 0).sort((a, b) => a - b);
    const Emax = v.length ? Math.max(5, Math.ceil(v[Math.floor(v.length * 0.99)] * 1.1)) : 50;
    const d = phononDOS(dosEnergies, { sigma: 1.0, Emin: 0, Emax, nE: 400 });
    return Array.from(d.E, (e, i) => [e, d.dos[i]]);
  }, [dosEnergies]);

  const canRun = ready && filesList.length > 0 && baseStructure && segments.length > 0 && !isProcessing;

  return (
    <main style={{ maxWidth: 1320, margin: '0 auto', padding: 22, display: 'flex', flexDirection: 'column', gap: 28 }}>

      {/* ════════ GROUP 1 · DATA & ASSESSMENT ════════ */}
      <section>
        <GroupHeader n="1" title="Data & assessment" desc="Inspect the configuration & fit before calculating" />

        <div style={{ display: 'flex', gap: 14, marginBottom: 14, alignItems: 'stretch' }}>
          {/* data folder */}
          <div className="rnr-card" style={{ width: 340, padding: 18, display: 'flex', flexDirection: 'column' }}>
            <div style={{ ...cardTitle, marginBottom: 13 }}>Data folder</div>
            <button onClick={handleSelectFolder} className="rnr-btn"
              style={{ width: '100%', background: ACCENT, color: '#fff', border: 'none', borderRadius: 8, padding: 11, font: "600 14px 'Space Grotesk'", cursor: 'pointer' }}>
              {dirHandle ? 'Change directory' : 'Select directory'}
            </button>
            <label className="rnr-btn" style={{ width: '100%', marginTop: 8, background: 'var(--bg)', border: `1px solid ${BORDER}`, color: DIM, borderRadius: 8, padding: 9, font: "500 12px 'Spline Sans'", cursor: 'pointer', textAlign: 'center', boxSizing: 'border-box' }}>
              Load saved result (.yaml / .json)
              <input type="file" accept=".yaml,.yml,.json" style={{ display: 'none' }} onChange={async (e) => {
                const file = e.target.files?.[0]; if (!file) return;
                try { onLoadResult(modelFromText(await file.text())); }
                catch (err) { setProgressText('Load failed: ' + err.message); }
              }} />
            </label>

            {rmc6fList.length > 1 && (
              <select value={structureName || ''} onChange={e => onStructureChange(e.target.value)}
                style={{ marginTop: 8, width: '100%', background: 'var(--inset)', border: `1px solid ${BORDER}`, borderRadius: 7, padding: '8px 10px', font: "12px 'Space Mono'", color: INK, cursor: 'pointer' }}>
                {rmc6fList.map(r => <option key={r.name} value={r.name}>{r.name}</option>)}
              </select>
            )}

            <div style={{ marginTop: 14, background: 'var(--inset)', borderRadius: 8, padding: 12, font: "12px/2.05 'Space Mono'", color: DIM }}>
              <Row k="dir" v={dirHandle ? dirHandle.name + '/' : '—'} />
              <Row k="configs" v={filesList.length ? `${filesList.length} · ${configFamily}` : '—'} vColor={ACCENTINK} />
              <Row k="formula" v={readout ? `${readout.formula} · ${readout.sites} sites` : '—'} />
              {readout && <>
                <div style={{ marginTop: 7, paddingTop: 7, borderTop: `1px solid ${BORDER}` }}>
                  <Row k="a, b, c (Å)" v={`${readout.a}, ${readout.b}, ${readout.c}`} />
                </div>
                <Row k="α, β, γ" v={`${readout.al}°, ${readout.be}°, ${readout.ga}°`} />
                <Row k="supercell" v={readout.supercell} />
              </>}
            </div>
          </div>

          {/* crystal structure */}
          <div className="rnr-card" style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden', position: 'relative' }}>
            <div style={{ position: 'absolute', top: 14, left: 16, zIndex: 2, ...cardTitle }}>Crystal structure</div>
            <span style={{ position: 'absolute', top: 17, right: 16, zIndex: 2, font: "10px 'Space Mono'", color: FAINT }}>1×1×1 cell</span>
            <div style={{ flex: 1, minHeight: 236, background: 'var(--inset)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              {previewStruct
                ? <CrystalViewer baseStructure={previewStruct} eigenvector={null} isPlaying={false} supercell={cell111} showCell showBonds />
                : <span style={{ color: FAINT, font: "12px 'Spline Sans'" }}>Select a dataset to preview the structure.</span>}
            </div>
            <div style={{ display: 'flex', gap: 16, padding: '10px 16px', borderTop: `1px solid ${BORDER}`, font: "11px 'Space Mono'", color: DIM, flexWrap: 'wrap' }}>
              {previewStruct
                ? Object.keys(baseStructure.atomDic).map(el => (
                    <span key={el}><span style={{ color: DEFAULT_COLORS[el] || '#cccccc' }}>●</span> {el}</span>
                  ))
                : <span style={{ color: FAINT }}>elements appear once a dataset is loaded</span>}
            </div>
          </div>
        </div>

        {/* fit quality */}
        <FitQuality dirHandle={dirHandle} onFlagged={(cfgs, sig) => { setFlaggedConfigs(cfgs); setFlagSigma(sig); }}
          excludeBad={excludeBad} onExcludeChange={setExcludeBad} />
      </section>

      {/* ════════ GROUP 2 · RECIPROCAL SPACE & k-PATH ════════ */}
      <section>
        <GroupHeader n="2" title="Reciprocal space & k-path" desc="Click high-symmetry points to build the path" />
        <div style={{ display: 'flex', gap: 14, height: 420 }}>
          {/* k-path segments (left, 340 — aligns with the other left-column cards) */}
          <div className="rnr-card" style={{ width: 340, padding: 18, display: 'flex', flexDirection: 'column' }}>
            <div style={{ ...cardTitle, marginBottom: 14 }}>k-path segments</div>
            <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 10, paddingRight: 4 }}>
              {segments.length === 0 ? (
                <div style={{ padding: '16px 8px', textAlign: 'center', color: FAINT, font: "12px 'Spline Sans'" }}>
                  No path defined — press <b style={{ color: ACCENTINK, fontWeight: 600 }}>Default path</b> or click points on the zone.
                </div>
              ) : segments.map((s, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10, background: 'var(--inset)', borderRadius: 8, padding: '8px 12px', flex: 'none' }}>
                  <span style={{ flex: 1, color: INK, font: "700 14px 'Noto Sans', sans-serif", letterSpacing: '.02em' }}>{displayLabel(s.from)} → {displayLabel(s.to)}</span>
                  <span style={{ color: FAINT, font: "11px 'Space Mono'" }}>npoints</span>
                  <Stepper value={s.npoints} onInc={() => setSegN(i, s.npoints + 1)} onDec={() => setSegN(i, s.npoints - 1)} />
                </div>
              ))}
            </div>
            <div style={{ paddingTop: 14, marginTop: 14, borderTop: `1px solid ${BORDER}`, display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', rowGap: 8 }}>
              <span style={{ font: "11px 'Space Mono'", color: FAINT }}>density</span>
              <Stepper width={38} value={density} onInc={() => { setDensity(d => d + 1); setSegNpoints({}); }} onDec={() => { setDensity(d => Math.max(1, d - 1)); setSegNpoints({}); }} />
              <span style={{ font: "11px 'Space Mono'", color: FAINT }}>pts/Å⁻¹</span>
              <span style={{ marginLeft: 'auto', font: "11px 'Space Mono'", color: FAINT }}>
                {segments.length} seg · <span style={{ color: ACCENTINK, fontWeight: 700 }}>{totalK} k-pts</span>
              </span>
            </div>
          </div>

          {/* Brillouin zone (right, flex — aligns with Crystal & Run) */}
          <BrillouinZoneViewer bzModel={bzModel} system={bravais?.system} onPathChange={onPathChange} />
        </div>
      </section>

      {/* ════════ GROUP 3 · RUN ════════ */}
      <section>
        <GroupHeader n="3" title="Run calculation" />
        <div style={{ display: 'flex', gap: 14, alignItems: 'stretch' }}>
          {/* displacement reference */}
          <div className="rnr-card" style={{ width: 340, display: 'flex', flexDirection: 'column', padding: 18 }}>
            <div style={{ ...cardTitle, marginBottom: 12 }}>Displacement reference</div>
            <Radio checked={refMode === 'average'} onClick={() => setRefMode('average')}>
              Ensemble average <span style={{ font: "11px 'Space Mono'", color: FAINT }}>default</span>
            </Radio>
            <Radio checked={refMode === 'file'} onClick={() => setRefMode('file')}>Equilibrium .rmc6f file</Radio>
            {refMode === 'file' && (
              <div style={{ marginTop: 10, paddingLeft: 26 }}>
                <select value={refName} onChange={e => setRefName(e.target.value)}
                  style={{ width: '100%', background: 'var(--inset)', border: `1px solid ${BORDER}`, borderRadius: 7, padding: '8px 10px', font: "12px 'Space Mono'", color: INK, cursor: 'pointer' }}>
                  <option value="">(select a file)</option>
                  {rmc6fList.map(r => <option key={r.name} value={r.name}>{r.name}</option>)}
                </select>
                {refName && <div style={{ marginTop: 8, font: "11px 'Space Mono'", color: FAINT }}>selected <span style={{ color: ACCENTINK }}>{refName}</span></div>}
              </div>
            )}
            <div style={{ marginTop: 14, paddingTop: 14, borderTop: `1px solid ${BORDER}` }}>
              <div style={{ font: "600 11px 'Space Grotesk'", letterSpacing: '.02em', color: DIM, marginBottom: 8 }}>REFERENCE SITE</div>
              <Radio checked={referenceMode === 'per-atom'} onClick={() => setReferenceMode('per-atom')}>
                Per-atom <span style={{ font: "11px 'Space Mono'", color: FAINT }}>default</span>
              </Radio>
              <Radio checked={referenceMode === 'symmetrized'} onClick={() => setReferenceMode('symmetrized')}>Symmetrized site</Radio>
            </div>
            <div style={{ marginTop: 'auto', paddingTop: 14, font: "11px/1.7 'Spline Sans'", color: FAINT }}>
              Sets the equilibrium positions r₀ for the displacement field u = r − r₀ that builds the dynamical matrix.
              {referenceMode === 'symmetrized'
                ? ' Symmetrized: r₀ is the cell’s shared basis-site average (imposes the cell symmetry).'
                : ' Per-atom: each atom about its own ensemble mean.'}
            </div>
          </div>

          {/* run */}
          <div className="rnr-card" style={{ flex: 1, minWidth: 0, padding: 18 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
              <span style={cardTitle}>Run</span>
              {bravais && (
                <span style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8, font: "11px 'Space Mono'", color: DIM }}>
                  <span>Bravais <span style={{ color: ACCENTINK, fontWeight: 700 }}>{bravais.code} {bravais.system}</span></span>
                  {symInfo && (
                    <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}
                      title={`Space-group operations of the reference structure detected at ${symTol.toFixed(2)} Å tolerance: ${symInfo.nSpace} ops (point group order ${symInfo.nPoint}), holding to ${symInfo.maxResidual.toFixed(3)} Å RMS. The basis is a single representative config, so loosen the tolerance to trace the underlying symmetry.`}>
                      <span style={{ color: 'var(--faint)' }}>·</span>
                      <span style={{ color: symInfo.nSpace > 1 ? ACCENTINK : 'var(--warnInk)', fontWeight: 700 }}>{symInfo.nSpace}</span>
                      <span>sym-ops @</span>
                      <Stepper width={34} value={symTol.toFixed(2)}
                        onInc={() => setSymTol(t => Math.min(1.5, +(t + 0.05).toFixed(2)))}
                        onDec={() => setSymTol(t => Math.max(0.05, +(t - 0.05).toFixed(2)))} />
                      <span>Å</span>
                    </span>
                  )}
                </span>
              )}
            </div>

            <div style={{ ...eyebrow, marginBottom: 9 }}>PARAMETERS</div>
            <div style={{ display: 'grid', gridTemplateColumns: '150px 150px', gap: 12, marginBottom: 16 }}>
              <Field label="T (K)" value={temperature} onChange={v => setTemperature(v)} step={1} />
              <Field label="degen tol" value={degenerateTol} onChange={v => setDegenerateTol(v)} step={0.001} />
            </div>

            <div style={{ ...eyebrow, marginBottom: 9 }}>COMPUTATION CELL</div>
            <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 16, flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', border: `1px solid ${BORDER}`, borderRadius: 8, overflow: 'hidden' }}>
                {[['conventional', 'Conventional'], ...(isCentered ? [['primitive', 'Primitive']] : []), ['custom', 'Custom supercell']].map(([t, lbl]) => (
                  <button key={t} onClick={() => setCellType(t)} className="rnr-btn"
                    title={t === 'primitive' ? 'Unfolded dispersion in the primitive cell' : undefined}
                    style={{ background: cellType === t ? ACCENT : 'transparent', color: cellType === t ? '#fff' : DIM, border: 'none', padding: '8px 13px', font: "600 12px 'Space Grotesk'", cursor: 'pointer', borderRight: t === 'custom' ? 'none' : `1px solid ${BORDER}` }}>{lbl}</button>
                ))}
              </div>
              {cellType === 'custom' && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  {[0, 1, 2].map(i => (
                    <Stepper key={i} width={28} value={customN[i]}
                      onInc={() => setCustomN(n => n.map((x, j) => j === i ? Math.min(8, x + 1) : x))}
                      onDec={() => setCustomN(n => n.map((x, j) => j === i ? Math.max(1, x - 1) : x))} />
                  ))}
                  <span style={{ font: "11px 'Space Mono'", color: FAINT }}>× conv.</span>
                </div>
              )}
              {nConvBasis > 0 && (
                <span style={{ marginLeft: 'auto', font: "11px 'Space Mono'", color: (nBranches > 600 || primitiveNoFold || residualHigh) ? 'var(--warnInk)' : FAINT }}
                  title={primitiveNoFold
                    ? `The average positions do not fold to the ideal ${cellInfo.ideal} primitive sites — this ensemble average has broken the ideal centering.`
                    : (foldsSites ? `Folded sites sit ${residual.toFixed(3)} Å (RMS) from their symmetrized position — how much symmetry this cell imposes.${residualHigh ? ' Large: the data may not support this symmetry.' : ''}` : undefined)}>
                  {nBasis} sites · {nBranches} branches{cellType === 'primitive' ? (primitiveNoFold ? ' · avg not centered ⚠' : ' · unfolded') : ''}{foldsSites && !primitiveNoFold ? ` · ⌀${residual.toFixed(2)} Å${residualHigh ? ' ⚠' : ''}` : ''}{nBranches > 600 ? ' ⚠' : ''}
                </span>
              )}
            </div>

            <div style={{ ...eyebrow, marginBottom: 9 }}>OPTIONS</div>
            <label onClick={() => setRunDos(v => !v)}
              style={{ display: 'flex', alignItems: 'center', gap: 11, background: 'var(--inset)', border: `1px solid ${BORDER}`, borderRadius: 8, padding: '11px 13px', cursor: 'pointer', marginBottom: 16 }}>
              <span style={{ width: 18, height: 18, borderRadius: 5, flex: 'none', display: 'flex', alignItems: 'center', justifyContent: 'center', background: runDos ? ACCENT : 'transparent', border: `2px solid ${runDos ? ACCENT : 'var(--bar)'}` }}>
                {runDos && <span style={{ color: '#fff', font: "700 12px 'Space Grotesk'", lineHeight: 1 }}>✓</span>}
              </span>
              <span style={{ font: "13px 'Spline Sans'", color: INK }}>Run Phonon DOS</span>
              <span style={{ marginLeft: 'auto', font: "11px 'Space Mono'", color: FAINT }}>q-grid</span>
              <span onClick={e => e.stopPropagation()} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <Stepper width={34} value={dosN} onInc={() => setDosN(n => Math.min(40, n + 1))} onDec={() => setDosN(n => Math.max(2, n - 1))} />
                <span style={{ font: "12.5px 'Space Mono'", color: FAINT }}>³ = {dosN ** 3} pts</span>
              </span>
            </label>

            {/* launch */}
            <div style={{ borderTop: `1px solid ${BORDER}`, paddingTop: 15 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                <div style={{ font: "600 13px 'Space Grotesk'", color: INK, whiteSpace: 'nowrap' }}>
                  {segments.length} segments · <span style={{ color: ACCENTINK }}>{totalK} k-points</span>
                  {excludeBad && flaggedInRun > 0 && <span style={{ font: "600 11px 'Space Mono'", color: 'var(--warnInk)' }}> · {flaggedInRun} configs excluded</span>}
                </div>
                <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
                  {isProcessing
                    ? <button onClick={() => { pipeline?.cancel(); setProgressText('Cancelling…'); }} className="rnr-btn"
                        style={{ background: '#e0564b', color: '#fff', border: 'none', borderRadius: 9, padding: '13px 22px', font: "700 14px 'Space Grotesk'", cursor: 'pointer' }}>■ Cancel</button>
                    : <button onClick={run} disabled={!canRun} className="rnr-btn"
                        style={{ background: canRun ? ACCENT : 'var(--inset2)', color: canRun ? '#fff' : FAINT, border: 'none', borderRadius: 9, padding: '13px 28px', font: "700 15px 'Space Grotesk'", cursor: canRun ? 'pointer' : 'default' }}>▶ Run phonon bands</button>}
                </div>
              </div>
              <div style={{ marginTop: 14, display: 'flex', alignItems: 'center', gap: 12 }}>
                <span style={{ flex: 1, minWidth: 0, font: "11px 'Space Mono'", color: DIM, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{progressText || 'idle · ready to compute'}</span>
                <span style={{ font: "11px 'Space Mono'", color: ACCENTINK, fontWeight: 700 }}>{Math.round(progress)}%</span>
              </div>
              <div style={{ marginTop: 6, height: 8, borderRadius: 5, background: 'var(--inset2)', overflow: 'hidden' }}>
                <div style={{ height: '100%', borderRadius: 5, background: ACCENT, width: `${progress}%`, transition: 'width .4s ease' }} />
              </div>
            </div>
          </div>
        </div>

        {/* log console — full width beneath the controls so messages have room to breathe */}
        <div className="rnr-card" style={{ marginTop: 14, padding: '14px 16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
            <span style={cardTitle}>Log console</span>
            <span style={{ font: "11px 'Space Mono'", color: FAINT }}>run output &amp; progress messages</span>
            {isProcessing && <span style={{ marginLeft: 'auto', font: "11px 'Space Mono'", color: ACCENTINK, fontWeight: 700 }}>running · {Math.round(progress)}%</span>}
          </div>
          <div ref={logEl} style={{ background: '#0f1623', border: '1px solid #1c2740', borderRadius: 9, padding: '12px 14px', height: 150, overflowY: 'auto', font: "11.5px/1.7 'Space Mono'", color: '#9fb3d1' }}>
            {logLines.length === 0
              ? <div style={{ color: '#4a6b8a' }}>› console output will appear here…</div>
              : logLines.map((t, i) => <div key={i} style={{ whiteSpace: 'pre-wrap' }}><span style={{ color: '#4a6b8a' }}>›</span> {t}</div>)}
          </div>
        </div>

        {/* phonon DOS result (when computed as part of the run) */}
        {dosCurve && (
          <div className="rnr-card" style={{ marginTop: 14, padding: 18 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
              <span style={cardTitle}>Phonon DOS</span>
              <span style={{ font: "11px 'Space Mono'", color: DIM }}>{dosN}³ q-grid · Gaussian-broadened</span>
            </div>
            <SciChart xLabel="Energy (meV)" yLabel="g(E)" height={220} series={[{ name: 'g(E)', color: ACCENT, width: 1.8, points: dosCurve }]} />
          </div>
        )}
      </section>
    </main>
  );
}

/* ── small building blocks ─────────────────────────────────────────────── */
function GroupHeader({ n, title, desc }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14 }}>
      <span style={{ width: 22, height: 22, borderRadius: 6, background: ACCENT, color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', font: "700 12px 'Space Grotesk'" }}>{n}</span>
      <span style={{ font: "600 15px 'Space Grotesk'", letterSpacing: '-.01em', color: INK }}>{title}</span>
      {desc && <span style={{ font: "12px 'Spline Sans'", color: DIM }}>{desc}</span>}
      <div style={{ flex: 1, height: 1, background: BORDER, marginLeft: 6 }} />
    </div>
  );
}

function Row({ k, v, vColor }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
      <span>{k}</span><span style={{ color: vColor || INK, textAlign: 'right' }}>{v}</span>
    </div>
  );
}

function Stepper({ value, onInc, onDec, width = 42 }) {
  return (
    <div style={{ display: 'flex', alignItems: 'stretch', background: 'var(--card)', border: `1px solid ${BORDER}`, borderRadius: 7, overflow: 'hidden' }}>
      <span style={{ padding: '6px 0', width, textAlign: 'center', font: "13px 'Space Mono'", color: ACCENTINK }}>{value}</span>
      <div style={{ display: 'flex', flexDirection: 'column', borderLeft: `1px solid ${BORDER}` }}>
        <button className="rnr-step" onClick={onInc} title="increase" style={stepBtn}>▲</button>
        <button className="rnr-step" onClick={onDec} title="decrease" style={{ ...stepBtn, borderTop: `1px solid ${BORDER}` }}>▼</button>
      </div>
    </div>
  );
}

function Field({ label, value, onChange, step }) {
  return (
    <div>
      <div style={{ font: "10.5px 'Space Mono'", color: FAINT, marginBottom: 5 }}>{label}</div>
      <input type="number" step={step} value={value} onChange={e => onChange(parseFloat(e.target.value))}
        style={{ width: '100%', boxSizing: 'border-box', background: 'var(--inset)', border: `1px solid ${BORDER}`, borderRadius: 7, padding: '9px 11px', font: "13px 'Space Mono'", color: INK }} />
    </div>
  );
}

function Radio({ checked, onClick, children }) {
  return (
    <label onClick={onClick} style={{ display: 'flex', alignItems: 'center', gap: 10, font: "13px 'Spline Sans'", color: INK, padding: '9px 10px', borderRadius: 8, background: checked ? 'var(--soft)' : 'transparent', cursor: 'pointer', marginTop: 2 }}>
      <span style={{ width: 16, height: 16, flex: 'none', borderRadius: '50%', border: `2px solid ${checked ? ACCENT : 'var(--bar)'}`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        {checked && <span style={{ width: 7, height: 7, borderRadius: '50%', background: ACCENT }} />}
      </span>
      {children}
    </label>
  );
}
