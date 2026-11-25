import React, { useState, useMemo, useRef, useEffect } from 'react';
import { listConfigs, readBaseStructure, findStructureFile, listRmc6f } from '../io/readers';
import { conventionalLattice, buildKPathFromSegments } from '../math/reciprocal';
import { analyzeBravais } from '../math/bravais';
import { IDENT, det3, matMul3, vecMat3, buildCellLabeling } from '../math/cells';
import { findSpaceGroupOps, siteOrbits, symmetryLadder, wyckoffLetter } from '../math/symmetry';
import { buildConventionalBZModel, buildBZModel, displayLabel } from '../math/highsym';
import { phononDOS } from '../math/dos';
import { DEFAULT_COLORS } from '../constants';
import { modelFromText } from '../io/phonopyDM';
import { downloadString } from '../io/writers';
import BrillouinZoneViewer from '../components/BrillouinZoneViewer';
import CrystalViewer from '../components/CrystalViewer';
import FitQuality from '../components/FitQuality';
import SciChart from '../components/SciChart';
import SupercellSymmetryPanel from '../components/SupercellSymmetryPanel';

const SUB = { '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉' };
const sub = (n) => String(n).split('').map(c => SUB[c] || c).join('');

/**
 * Runner page — MOTIF single-screen two-pane studio:
 *   • Empty state: calm editorial landing ("Open ensemble folder…" / "Load band.yaml")
 *   • Loaded state:
 *       Left stage:  Crystal · Fit quality · Brillouin zone (· Phonon DOS)
 *       Right panel: 1. Data · 2. Cell · 3. Path · 4. Run + collapsible Log + pinned Run bar
 */
export default function RunnerPage({ pipeline, ready, onResults, onLoadResult }) {
  const [dirHandle, setDirHandle] = useState(null);
  const [filesList, setFilesList] = useState([]);
  const [configFamily, setConfigFamily] = useState(null);
  const [rmc6fList, setRmc6fList] = useState([]);
  const [structureName, setStructureName] = useState(null);
  const [baseStructure, setBaseStructure] = useState(null);

  const [stageView, setStageView] = useState('crystal');  // 'crystal' | 'fit' | 'bz' | 'dos'
  const [sideTab, setSideTab] = useState('data');         // 'data' | 'cell' | 'path' | 'run'
  const [logOpen, setLogOpen] = useState(false);

  const [refMode, setRefMode] = useState('average');
  const [referenceMode, setReferenceMode] = useState('per-atom');
  const [imposeSymmetry, setImposeSymmetry] = useState(false);
  const [baseCell, setBaseCell] = useState('unit');
  const [customN, setCustomN] = useState([1, 1, 1]);
  const [fold, setFold] = useState('conventional');
  const [symTol, setSymTol] = useState(0.5);
  const [tolMax, setTolMax] = useState(1.0);
  const [tolSteps, setTolSteps] = useState(20);
  const [fineTol, setFineTol] = useState(false);
  const [avgBasis, setAvgBasis] = useState(null);
  const [symBusy, setSymBusy] = useState(false);
  const [scSymResult, setScSymResult] = useState(null);
  const [scSymBusy, setScSymBusy] = useState(false);
  const [scSymErr, setScSymErr] = useState(null);
  const [scSymSample, setScSymSample] = useState(20);
  const [refName, setRefName] = useState('');

  const [temperature, setTemperature] = useState(5);
  const [degenerateTol, setDegenerateTol] = useState(5e-3);
  const [density, setDensity] = useState(20);

  const [bzSegments, setBzSegments] = useState([]);
  const [pointsConv, setPointsConv] = useState({});
  const [segNpoints, setSegNpoints] = useState({});

  const [runDos, setRunDos] = useState(false);
  const [dosN, setDosN] = useState(10);
  const [flaggedConfigs, setFlaggedConfigs] = useState([]);
  const [flagSigma, setFlagSigma] = useState(2);
  const [excludeBad, setExcludeBad] = useState(false);

  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressText, setProgressText] = useState('');
  const [logLines, setLogLines] = useState([]);
  const [dosEnergies, setDosEnergies] = useState(null);
  const [copiedLog, setCopiedLog] = useState(false);
  const [loadErr, setLoadErr] = useState(null);

  const [helpOpen, setHelpOpen] = useState({});
  const toggleHelp = (k) => setHelpOpen(h => ({ ...h, [k]: !h[k] }));

  const lastMsg = useRef('');
  const logEl = useRef(null);
  const cell111 = useMemo(() => [1, 1, 1], []);
  const pushLog = (t) => { if (t && t !== lastMsg.current) { lastMsg.current = t; setLogLines(l => [...l, t]); } };
  useEffect(() => { if (logEl.current) logEl.current.scrollTop = logEl.current.scrollHeight; }, [logLines, logOpen]);

  const bravais = useMemo(
    () => (baseStructure?.v1 && baseStructure.basis
      ? analyzeBravais(conventionalLattice(baseStructure.v1, baseStructure.v2, baseStructure.v3, baseStructure.dim), baseStructure.basis)
      : null),
    [baseStructure]
  );

  const nConvBasis = baseStructure?.basis?.length || 0;
  const Pbase = useMemo(() => (baseCell === 'custom'
    ? [[customN[0], 0, 0], [0, customN[1], 0], [0, 0, customN[2]]] : IDENT), [baseCell, customN]);
  const Lbase = useMemo(() => (bravais ? matMul3(Pbase, bravais.A_conv) : null), [bravais, Pbase]);

  const baseBasis = useMemo(() => {
    const src = avgBasis || baseStructure?.basis;
    if (!src) return null;
    const n = baseCell === 'custom' ? customN : [1, 1, 1];
    const out = [];
    for (let i = 0; i < n[0]; i++) for (let j = 0; j < n[1]; j++) for (let k = 0; k < n[2]; k++)
      for (const s of src) out.push({ el: s.el, frac: [(i + s.frac[0]) / n[0], (j + s.frac[1]) / n[1], (k + s.frac[2]) / n[2]] });
    return out;
  }, [baseStructure, avgBasis, baseCell, customN]);

  const bravaisBase = useMemo(() => {
    if (!Lbase || !baseBasis) return bravais;
    try { return analyzeBravais(Lbase, baseBasis); } catch { return bravais; }
  }, [Lbase, baseBasis, bravais]);

  const symInfo = useMemo(() => {
    if (!Lbase || !baseBasis) return null;
    const sg = findSpaceGroupOps(Lbase, baseBasis, symTol);
    const orbits = siteOrbits(Lbase, baseBasis, sg.ops, symTol);
    const wyckoff = orbits.map(o => {
      const L = wyckoffLetter(sg.spaceGroupNumber, sg.centering, o.size, o.site, o.rep);
      return { el: o.element, label: `${o.size}${L || ` (${o.site})`}`, site: o.site, mult: o.size };
    });
    return { ...sg, orbits, wyckoff, onAverage: !!avgBasis };
  }, [Lbase, baseBasis, symTol, avgBasis]);

  const symLadder = useMemo(() => (Lbase && baseBasis ? symmetryLadder(Lbase, baseBasis, tolMax) : []), [Lbase, baseBasis, tolMax]);

  useEffect(() => { setSymTol(t => Math.min(t, tolMax)); }, [tolMax]);

  const primitiveAvail = !!bravaisBase && bravaisBase.centering !== 'P';
  const effFold = (fold === 'primitive' && primitiveAvail) ? 'primitive' : 'conventional';

  const compP = useMemo(() => (effFold === 'primitive' && bravaisBase
    ? matMul3(bravaisBase.M, Pbase) : Pbase), [effFold, bravaisBase, Pbase]);

  const bzModel = useMemo(() => {
    if (!bravaisBase) return null;
    return effFold === 'primitive' ? buildBZModel(bravaisBase) : buildConventionalBZModel(bravaisBase);
  }, [bravaisBase, effFold]);

  const compOrbits = useMemo(() => {
    if (!symInfo) return [];
    if (effFold !== 'primitive' || !Lbase || !baseBasis || !bravaisBase) return symInfo.wyckoff;
    try {
      const avgPos = baseBasis.map(s => vecMat3(s.frac, Lbase));
      const lab = buildCellLabeling(avgPos, baseBasis.map(s => s.el), baseBasis.map(() => 1), Lbase, bravaisBase.M, { tol: 0.08 });
      if (!lab.nBasis) return symInfo.wyckoff;
      const tauBasis = lab.tauFracLabel.map((frac, t) => ({ frac, el: lab.tauElement[t] }));
      const sg = findSpaceGroupOps(lab.L, tauBasis, symTol);
      const orbs = siteOrbits(lab.L, tauBasis, sg.ops, symTol);
      return orbs.map(o => {
        const L = wyckoffLetter(sg.spaceGroupNumber, sg.centering, o.size, o.site, o.rep);
        return { el: o.element, label: `${o.size}${L || ` (${o.site})`}`, site: o.site, mult: o.size };
      });
    } catch { return symInfo.wyckoff; }
  }, [symInfo, effFold, Lbase, baseBasis, bravaisBase, symTol]);

  const cellInfo = useMemo(() => {
    if (!bravais || nConvBasis === 0) return { nBasis: 0, ideal: 0, residual: 0 };
    const ideal = Math.max(1, Math.round(nConvBasis * Math.abs(det3(compP))));
    if (effFold === 'primitive' && baseCell === 'unit') {
      const b = baseStructure.basis;
      const avgPos = b.map(s => vecMat3(s.frac, bravais.A_conv));
      const lab = buildCellLabeling(avgPos, b.map(s => s.rn), b.map(() => 1), bravais.A_conv, compP, { tol: 0.08 });
      return { nBasis: lab.nBasis, ideal, residual: lab.maxResidual || 0 };
    }
    return { nBasis: ideal, ideal, residual: 0 };
  }, [bravais, baseStructure, nConvBasis, effFold, baseCell, compP]);

  const nBasis = cellInfo.nBasis;
  const nBranches = 3 * nBasis;
  const primitiveNoFold = effFold === 'primitive' && cellInfo.ideal > 0 && nBasis > cellInfo.ideal * 1.5;
  const foldsSites = nBasis < nConvBasis;
  const residual = cellInfo.residual || 0;
  const residualHigh = foldsSites && residual > 0.3;

  useEffect(() => { setAvgBasis(null); }, [baseStructure]);

  const detectOnAverage = async () => {
    if (!pipeline || !filesList.length || !configFamily || !baseStructure) return;
    setSymBusy(true);
    try {
      const basis = await pipeline.computeAverageBasis(filesList, configFamily, baseStructure, 60);
      setAvgBasis(basis);
      setSymTol(0.15);
    } catch (e) {
      pushLog(`Symmetry-on-average failed: ${e.message}`);
    } finally { setSymBusy(false); }
  };

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
      formula, sites, counts,
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
      setLoadErr(null);
      setFlaggedConfigs([]);
      setDosEnergies(null);
      setScSymResult(null);
      setScSymErr(null);
      setRefName('');
      const { files, family } = await listConfigs(dh);
      setFilesList(files); setConfigFamily(family);
      setRmc6fList(await listRmc6f(dh));
      if (family === 'rmc6f' && files.length > 0) {
        await loadStructure(files[0], files[0].name);
        setStageView('crystal');
      } else if (family === 'frac') {
        const sh = await findStructureFile(dh);
        if (!sh) { setProgressText('Frac configs found but no .rmc6f structure file in this folder.'); setBaseStructure(null); return; }
        await loadStructure(sh, sh.name);
        setStageView('crystal');
      } else {
        setBaseStructure(null);
        setLoadErr(`No .rmc6f or Frac*.txt configurations found in “${dh.name}”.`);
      }
    } catch (err) {
      if (err?.name !== 'AbortError') console.error(err);
    }
  };

  const loadBandFile = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      setLoadErr(null);
      onLoadResult(modelFromText(await file.text()));
    } catch (err) {
      setLoadErr('Load failed: ' + err.message);
    }
  };

  const onStructureChange = async (name) => {
    const item = rmc6fList.find(r => r.name === name);
    if (item) await loadStructure(item.handle, name);
  };

  const runSupercellSymmetry = async () => {
    if (!pipeline || !baseStructure || !filesList.length) return;
    setScSymBusy(true);
    setScSymErr(null);
    try {
      const res = await pipeline.analyzeSupercellSymmetry(filesList, configFamily, baseStructure, {
        sampleSize: scSymSample,
      });
      setScSymResult(res);
    } catch (err) {
      console.error(err);
      setScSymErr(err.message || String(err));
    } finally {
      setScSymBusy(false);
    }
  };

  const alignSubgroup = (subgroupSpec) => {
    if (!pipeline || !scSymResult) return;
    try {
      const aligned = pipeline.alignSupercellSubgroup(subgroupSpec);
      setScSymResult(prev => (prev ? { ...prev, aligned } : prev));
    } catch (err) {
      setScSymErr(err.message || String(err));
    }
  };

  const downloadAlignedCIF = (aligned, projected = true) => {
    if (!pipeline || !aligned) return;
    const cif = pipeline.buildAlignedCIF(aligned, { projected });
    const sgSlug = (aligned.targetSpaceGroup || 'aligned').replace(/[^A-Za-z0-9_-]+/g, '');
    const suffix = projected ? 'projected' : 'raw';
    downloadString(cif, `aligned_${sgSlug}_${suffix}.cif`, 'chemical/x-cif');
  };

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
    setIsProcessing(true); setProgress(0); setDosEnergies(null); setLogOpen(true);
    lastMsg.current = '';
    const excluded = filesList.length - runFiles.length;
    setLogLines([`▶ Run started · T = ${temperature} K · degen tol ${degenerateTol}`
      + ` · r₀ ${referenceMode === 'symmetrized' ? 'ideal site' : 'per-atom'} · S(k) symmetrize ${imposeSymmetry ? 'on' : 'off'}`
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
      const res = await pipeline.runCalculation(runFiles, configFamily, baseStructure, qFrac, temperature, 50, { referenceHandle, degenerateTol, referenceMode, computationCell: { P: compP }, symmetrize: imposeSymmetry, symTol });

      let dos = null;
      if (runDos) {
        pushLog(`Computing phonon DOS · q-grid ${dosN}³…`);
        try {
          const d = await pipeline.computeDOSGrid(runFiles, configFamily, baseStructure, dosN, temperature, 50, { referenceHandle, referenceMode, computationCell: { P: compP }, symmetrize: imposeSymmetry, symTol });
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

  const copyLog = (e) => {
    e.stopPropagation();
    if (!logLines.length) return;
    navigator.clipboard?.writeText(logLines.join('\n'));
    setCopiedLog(true);
    setTimeout(() => setCopiedLog(false), 1600);
  };

  const selectSideTab = (t) => {
    setSideTab(t);
    if (t === 'data') setStageView('crystal');
    if (t === 'path') setStageView('bz');
  };

  // ── Calm MOTIF editorial landing when no folder is open ──────────────────
  if (!dirHandle) {
    return (
      <div
        style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'var(--page)',
          padding: '48px 24px',
        }}
      >
        <div style={{ width: 'min(640px, 100%)', textAlign: 'left', lineHeight: 1.55 }}>
          <h1 style={{ fontSize: 36, fontWeight: 700, letterSpacing: '-0.035em', lineHeight: 1.08, margin: '0 0 16px', color: 'var(--ink)' }}>
            Phonons from RMC total-scattering ensembles.
          </h1>
          <p style={{ margin: '0 0 10px', color: 'var(--secondary)', fontSize: 15 }}>
            Extract phonon band structures, animated 3D modes, and simulated inelastic neutron spectra directly from your RMCProfile displacement covariance — computed locally on WebGPU.
          </p>
          <p style={{ margin: '0 0 24px', color: 'var(--faint)', fontSize: 13 }}>
            Your files never leave your computer. Open a folder of numbered <code style={{ fontFamily: 'var(--font-mono)' }}>.rmc6f</code> or <code style={{ fontFamily: 'var(--font-mono)' }}>Frac*.txt</code> configurations to begin.
          </p>

          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <button type="button" onClick={handleSelectFolder} className="m-btn primary lg">
              Open ensemble folder…
            </button>
            <label className="m-button lg">
              Load band.yaml / .json…
              <input type="file" accept=".yaml,.yml,.json" onChange={loadBandFile} style={{ display: 'none' }} />
            </label>
          </div>

          {loadErr && (
            <div className="m-notice bad" style={{ margin: '18px 0 0' }}>{loadErr}</div>
          )}
        </div>
      </div>
    );
  }

  // ── Loaded State: Two-Pane MOTIF Studio ──────────────────────────────────
  return (
    <div className="m-work">
      {/* ════════ LEFT STAGE (Visual Workspace) ════════ */}
      <div className="m-stage">
        {/* Stage Header Bar */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 14, flexWrap: 'wrap', paddingBottom: 10, borderBottom: '1px solid var(--line)', flex: 'none' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
            <div className="m-tools" role="tablist" aria-label="Stage view">
              <button type="button" role="tab" aria-selected={stageView === 'crystal'} className={stageView === 'crystal' ? 'on' : 'dim'} onClick={() => setStageView('crystal')}>
                Crystal
              </button>
              <button type="button" role="tab" aria-selected={stageView === 'fit'} className={stageView === 'fit' ? 'on' : 'dim'} onClick={() => setStageView('fit')}>
                Fit quality
                {flaggedInRun > 0 && (
                  <span style={{ fontSize: 10.5, fontFamily: 'var(--font-mono)', padding: '1px 5px', borderRadius: 4, background: 'var(--warn-bg)', color: 'var(--warn-ink)' }}>
                    {flaggedInRun} &gt; {flagSigma}σ
                  </span>
                )}
              </button>
              <button type="button" role="tab" aria-selected={stageView === 'bz'} className={stageView === 'bz' ? 'on' : 'dim'} onClick={() => setStageView('bz')}>
                Brillouin zone
              </button>
              {dosCurve && (
                <button type="button" role="tab" aria-selected={stageView === 'dos'} className={stageView === 'dos' ? 'on' : 'dim'} onClick={() => setStageView('dos')}>
                  Phonon DOS
                </button>
              )}
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--secondary)' }}>
              {dirHandle.name}/
            </span>
            <button type="button" onClick={handleSelectFolder} className="m-btn sm">Change folder…</button>
            <label className="m-button subtle sm">
              Load band.yaml
              <input type="file" accept=".yaml,.yml,.json" onChange={loadBandFile} style={{ display: 'none' }} />
            </label>
          </div>
        </div>

        {loadErr && <div className="m-notice bad" style={{ margin: 0 }}>{loadErr}</div>}

        {/* Stage View 1: Crystal Structure (kept mounted) */}
        <div
          className="m-frame"
          style={{ display: stageView === 'crystal' ? 'flex' : 'none', flexDirection: 'column', flex: 1, minHeight: 380 }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 14px', borderBottom: '1px solid var(--line)' }}>
            <span style={{ fontWeight: 600 }}>
              {readout ? `${readout.formula} · unit cell` : 'Crystal structure'}
            </span>
            <span style={{ fontSize: 'var(--fz-small)', color: 'var(--faint)', fontFamily: 'var(--font-mono)' }}>
              {readout ? `1×1×1 (${readout.sites} sites)` : '1×1×1 cell'}
            </span>
          </div>
          <div style={{ flex: 1, minHeight: 320, background: 'var(--surface)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {previewStruct ? (
              <CrystalViewer baseStructure={previewStruct} eigenvector={null} isPlaying={false} supercell={cell111} showCell showBonds />
            ) : (
              <span style={{ color: 'var(--faint)', fontSize: 'var(--fz-control)' }}>
                No valid crystal structure loaded from this folder.
              </span>
            )}
          </div>
          <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', padding: '8px 14px', borderTop: '1px solid var(--line)', fontSize: 'var(--fz-small)', color: 'var(--secondary)' }}>
            {readout ? (
              readout.counts.map(([el, count]) => (
                <span key={el} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ width: 10, height: 10, borderRadius: '50%', background: DEFAULT_COLORS[el] || '#cccccc', boxShadow: 'inset 0 0 0 1px rgba(0,0,0,0.18)', flex: 'none' }} />
                  <b style={{ color: 'var(--ink)', fontWeight: 600 }}>{el}</b>
                  <span style={{ color: 'var(--faint)', fontFamily: 'var(--font-mono)' }}>×{count}</span>
                </span>
              ))
            ) : (
              <span style={{ color: 'var(--faint)' }}>Species legend</span>
            )}
          </div>
        </div>

        {/* Stage View 2: Fit Quality (kept mounted so Rw scan persists across tab switches) */}
        <div style={{ display: stageView === 'fit' ? 'block' : 'none' }}>
          <FitQuality
            dirHandle={dirHandle}
            onFlagged={(cfgs, sig) => { setFlaggedConfigs(cfgs); setFlagSigma(sig); }}
            excludeBad={excludeBad}
            onExcludeChange={setExcludeBad}
          />
        </div>

        {/* Stage View 3: Brillouin Zone (kept mounted so k-path initializes immediately & persists) */}
        <div style={{ display: stageView === 'bz' ? 'flex' : 'none', flexDirection: 'column', flex: 1, minHeight: 380 }}>
          <BrillouinZoneViewer bzModel={bzModel} system={bravais?.system} onPathChange={onPathChange} />
        </div>

        {/* Stage View 4: Phonon DOS (when computed) */}
        {dosCurve && (
          <div className="m-frame" style={{ display: stageView === 'dos' ? 'block' : 'none', padding: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
              <span style={{ fontWeight: 600 }}>Phonon Density of States</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--fz-small)', color: 'var(--secondary)' }}>
                {dosN}³ q-grid · Gaussian-broadened
              </span>
            </div>
            <SciChart xLabel="Energy (meV)" yLabel="g(E)" height={340} series={[{ name: 'g(E)', color: '#1b1a18', width: 1.8, points: dosCurve }]} />
          </div>
        )}
      </div>

      {/* ════════ RIGHT CONTROL COLUMN (MOTIF .m-side) ════════ */}
      <aside className="m-side" aria-label="Calculation setup">
        {/* 4-step underlined tabs */}
        <div className="m-tabs" role="tablist">
          {[
            ['data', '1. Data'],
            ['cell', '2. Cell'],
            ['path', '3. Path'],
            ['run', '4. Run'],
          ].map(([id, label]) => (
            <button
              key={id}
              type="button"
              role="tab"
              aria-selected={sideTab === id}
              className={sideTab === id ? 'on' : ''}
              onClick={() => selectSideTab(id)}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Scrollable tab body */}
        <div className="m-tab-body">

          {/* ── TAB 1: DATA ── */}
          {sideTab === 'data' && (
            <>
              <details open className="m-section">
                <summary><span>Ensemble summary</span></summary>
                <div className="m-section-body">
                  {rmc6fList.length > 1 && (
                    <div className="m-row" style={{ marginBottom: 10 }}>
                      <span className="m-lab" style={{ width: 68 }}>structure</span>
                      <select
                        value={structureName || ''}
                        onChange={e => onStructureChange(e.target.value)}
                        className="m-select mono"
                        style={{ flex: 1, minWidth: 0 }}
                      >
                        {rmc6fList.map(r => <option key={r.name} value={r.name}>{r.name}</option>)}
                      </select>
                    </div>
                  )}

                  <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 'var(--fz-control)' }}>
                    <tbody>
                      <InfoRow k="folder" v={`${dirHandle.name}/`} mono />
                      <InfoRow k="configs" v={filesList.length ? `${filesList.length} (${configFamily})` : '—'} strong={!!filesList.length} mono />
                      <InfoRow k="formula" v={readout ? `${readout.formula} · ${readout.sites} sites` : '—'} />
                      <InfoRow k="a, b, c (Å)" v={readout ? `${readout.a}, ${readout.b}, ${readout.c}` : '—'} mono />
                      <InfoRow k="α, β, γ" v={readout ? `${readout.al}°, ${readout.be}°, ${readout.ga}°` : '—'} mono />
                      <InfoRow k="supercell" v={readout ? readout.supercell : '—'} mono last />
                    </tbody>
                  </table>
                </div>
              </details>

              <details open className="m-section">
                <summary><span>Fit quality filter</span></summary>
                <div className="m-section-body">
                  <p style={{ margin: '0 0 10px', fontSize: 'var(--fz-small)', color: 'var(--secondary)', lineHeight: 1.5 }}>
                    {flaggedInRun > 0
                      ? `${flaggedInRun} of ${filesList.length} configurations exceed ${flagSigma}σ in Rw.`
                      : 'Inspect S(Q), G(r), and Bragg agreement across configurations on the Fit quality stage.'}
                  </p>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, flexWrap: 'wrap' }}>
                    <label style={{ display: 'inline-flex', alignItems: 'center', gap: 7, cursor: 'pointer', fontSize: 'var(--fz-control)' }}>
                      <input type="checkbox" checked={excludeBad} onChange={e => setExcludeBad(e.target.checked)} disabled={!flaggedInRun} />
                      Exclude flagged configs
                    </label>
                    <button type="button" onClick={() => setStageView('fit')} className="m-btn subtle sm">
                      View fit plots →
                    </button>
                  </div>
                </div>
              </details>
            </>
          )}

          {/* ── TAB 2: CELL & SYMMETRY ── */}
          {sideTab === 'cell' && (
            <>
              <details open className="m-section">
                <summary><span>Base cell &amp; fold</span></summary>
                <div className="m-section-body" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  <div className="m-row" style={{ margin: 0 }}>
                    <span className="m-lab" style={{ width: 64 }}>cell</span>
                    <div style={{ display: 'inline-flex', gap: 6 }}>
                      <button type="button" onClick={() => setBaseCell('unit')} className={`m-btn sm ${baseCell === 'unit' ? 'active' : 'subtle'}`}>Unit</button>
                      <button type="button" onClick={() => setBaseCell('custom')} className={`m-btn sm ${baseCell === 'custom' ? 'active' : 'subtle'}`}>Supercell</button>
                    </div>
                  </div>

                  {baseCell === 'custom' && (
                    <div className="m-row" style={{ margin: 0 }}>
                      <span className="m-lab" style={{ width: 64 }}>size</span>
                      <div style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                        {[0, 1, 2].map(i => (
                          <Stepper
                            key={i}
                            value={customN[i]}
                            onInc={() => setCustomN(n => n.map((x, j) => j === i ? Math.min(8, x + 1) : x))}
                            onDec={() => setCustomN(n => n.map((x, j) => j === i ? Math.max(1, x - 1) : x))}
                          />
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="m-row" style={{ margin: 0 }}>
                    <span className="m-lab" style={{ width: 64 }}>fold</span>
                    <div style={{ display: 'inline-flex', gap: 6 }}>
                      <button type="button" onClick={() => setFold('conventional')} className={`m-btn sm ${effFold === 'conventional' ? 'active' : 'subtle'}`}>
                        Conventional
                      </button>
                      <button
                        type="button"
                        disabled={!primitiveAvail}
                        onClick={() => primitiveAvail && setFold('primitive')}
                        className={`m-btn sm ${effFold === 'primitive' ? 'active' : 'subtle'}`}
                        title={primitiveAvail ? `Fold by ${bravaisBase?.centering}-centering` : 'P-centered — primitive equals conventional'}
                      >
                        Primitive
                      </button>
                    </div>
                  </div>

                  {nConvBasis > 0 && (
                    <div
                      className="tabular-nums"
                      style={{
                        fontSize: 'var(--fz-small)',
                        fontFamily: 'var(--font-mono)',
                        color: (nBranches > 600 || primitiveNoFold || residualHigh) ? 'var(--warn-ink)' : 'var(--secondary)',
                      }}
                    >
                      {nBasis} sites · <b style={{ color: 'var(--ink)' }}>{nBranches}</b> branches
                      {primitiveNoFold ? ' · avg not centered ⚠' : ''}
                      {foldsSites && !primitiveNoFold ? ` · RMS ${residual.toFixed(2)} Å${residualHigh ? ' ⚠' : ''}` : ''}
                      {nBranches > 600 ? ' ⚠' : ''}
                    </div>
                  )}
                </div>
              </details>

              <details open className="m-section">
                <summary>
                  <span>Detected symmetry</span>
                  <button
                    type="button"
                    className={`m-info ${helpOpen.sym ? 'on' : ''}`}
                    onClick={(e) => { e.preventDefault(); toggleHelp('sym'); }}
                    aria-label="About symmetry detection"
                  >?</button>
                </summary>
                <div className="m-section-body" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {helpOpen.sym && (
                    <p className="m-explain" style={{ margin: 0 }}>
                      Operations that map every site onto an equivalent one within tolerance (Å). Click a segment on the ladder to select that space group.
                    </p>
                  )}

                  {symInfo ? (
                    <>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, flexWrap: 'wrap' }}>
                        <div style={{ display: 'inline-flex', alignItems: 'baseline', gap: 7 }}>
                          <span style={{ fontFamily: 'var(--font-sym)', fontSize: 18, fontWeight: 700, color: 'var(--ink)' }}>
                            {symInfo.spaceGroup}
                          </span>
                          {symInfo.spaceGroupNumber && (
                            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11.5, color: 'var(--faint)' }}>
                              #{symInfo.spaceGroupNumber}
                            </span>
                          )}
                        </div>
                        <button
                          type="button"
                          onClick={detectOnAverage}
                          disabled={symBusy || !filesList.length}
                          className={`m-btn sm ${symInfo.onAverage ? 'active' : ''}`}
                          title="Detect symmetry on ensemble average for tighter tolerance"
                        >
                          {symBusy ? 'Averaging…' : symInfo.onAverage ? '✓ Ensemble avg' : 'Detect on avg'}
                        </button>
                      </div>

                      <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
                        {symInfo.wyckoff.map((w, i) => (
                          <span
                            key={i}
                            title={`${w.el}: site symmetry ${w.site}`}
                            style={{ padding: '2px 7px', borderRadius: 4, background: 'var(--muted)', fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--secondary)' }}
                          >
                            <b style={{ color: 'var(--ink)' }}>{w.el}</b> {w.label}
                          </span>
                        ))}
                      </div>

                      {/* Symmetry ladder */}
                      {symLadder.length > 0 && (() => {
                        const ladderMax = symLadder[symLadder.length - 1].to || 1;
                        const maxOps = Math.max(...symLadder.map(b => b.nSpace));
                        return (
                          <div>
                            <div style={{ display: 'flex', height: 28, borderRadius: 6, overflow: 'hidden', border: '1px solid var(--control)' }}>
                              {symLadder.map((b, i) => {
                                const w = Math.max(5, (b.to - b.from) / ladderMax * 100);
                                const level = maxOps > 1 ? Math.log(b.nSpace) / Math.log(maxOps) : 0;
                                const active = i === symLadder.length - 1
                                  ? (symTol >= b.from && symTol <= b.to)
                                  : (symTol >= b.from && symTol < b.to);
                                return (
                                  <button
                                    type="button"
                                    key={i}
                                    onClick={() => setSymTol(Math.max(0.05, Math.min(tolMax, +(((b.from + b.to) / 2)).toFixed(2))))}
                                    title={`${b.spaceGroup} · ${b.from.toFixed(2)}–${b.to.toFixed(2)} Å · ${b.nSpace} ops`}
                                    style={{
                                      width: `${w}%`,
                                      minWidth: 0,
                                      height: '100%',
                                      border: 0,
                                      padding: '0 3px',
                                      background: active ? 'var(--accent)' : `hsl(38 ${10 + level * 12}% ${94 - level * 20}%)`,
                                      borderRight: i < symLadder.length - 1 ? '1px solid var(--line)' : 'none',
                                      boxShadow: active ? 'inset 0 0 0 1.5px var(--ink)' : 'none',
                                      cursor: 'pointer',
                                      display: 'flex',
                                      alignItems: 'center',
                                      justifyContent: 'center',
                                    }}
                                  >
                                    <span style={{ fontFamily: 'var(--font-sym)', fontSize: 10.5, fontWeight: 700, color: 'var(--ink)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                      {b.spaceGroup}
                                    </span>
                                  </button>
                                );
                              })}
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, fontSize: 10.5, fontFamily: 'var(--font-mono)', color: 'var(--faint)' }}>
                              <span>0</span>
                              <span>tol <b style={{ color: 'var(--ink)' }}>{symTol.toFixed(2)} Å</b></span>
                              <span>{ladderMax.toFixed(1)} Å</span>
                            </div>

                            <div style={{ marginTop: 6 }}>
                              <button type="button" onClick={() => setFineTol(v => !v)} className="m-btn subtle sm">
                                {fineTol ? '▾ Hide fine-tune' : '▸ Fine-tune tolerance'}
                              </button>
                              {fineTol && (
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 8 }}>
                                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11.5 }}>
                                    <span style={{ color: 'var(--secondary)' }}>max</span>
                                    <input
                                      type="number"
                                      step={0.1}
                                      min={0.2}
                                      max={3}
                                      value={tolMax}
                                      onChange={e => { const v = parseFloat(e.target.value); if (Number.isFinite(v)) setTolMax(Math.max(0.2, Math.min(3, v))); }}
                                      className="m-input mono"
                                      style={{ width: 54, height: 24, textAlign: 'center' }}
                                    />
                                    <span style={{ color: 'var(--secondary)' }}>steps</span>
                                    <input
                                      type="number"
                                      step={1}
                                      min={5}
                                      max={200}
                                      value={tolSteps}
                                      onChange={e => { const v = parseInt(e.target.value, 10); if (Number.isFinite(v)) setTolSteps(Math.max(5, Math.min(200, v))); }}
                                      className="m-input mono"
                                      style={{ width: 52, height: 24, textAlign: 'center' }}
                                    />
                                  </div>
                                  <input
                                    type="range"
                                    min={+(tolMax / tolSteps).toFixed(4)}
                                    max={tolMax}
                                    step={+(tolMax / tolSteps).toFixed(4)}
                                    value={Math.min(symTol, tolMax)}
                                    onChange={e => setSymTol(parseFloat(e.target.value))}
                                    className="m-range"
                                  />
                                </div>
                              )}
                            </div>
                          </div>
                        );
                      })()}

                      {/* Impose symmetry on S(k) */}
                      <div className="m-rule-box" style={{ background: imposeSymmetry ? 'var(--note-bg)' : 'var(--muted)', padding: '10px 12px' }}>
                        <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontWeight: 600 }}>
                          <input type="checkbox" checked={imposeSymmetry} onChange={e => setImposeSymmetry(e.target.checked)} />
                          Impose symmetry on S(k)
                        </label>
                        <div style={{ marginTop: 4, fontSize: 11.5, fontFamily: 'var(--font-mono)', color: 'var(--secondary)' }}>
                          pool {nBasis} sites → {compOrbits.length} orbit{compOrbits.length === 1 ? '' : 's'}
                        </div>
                        {imposeSymmetry && (
                          <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap', marginTop: 6 }}>
                            {compOrbits.map((w, i) => (
                              <span key={i} style={{ padding: '1px 6px', borderRadius: 4, background: 'var(--surface)', border: '1px solid var(--line)', fontFamily: 'var(--font-mono)', fontSize: 11 }}>
                                <b>{w.el}</b> {w.label} ×{w.mult}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    </>
                  ) : (
                    <span style={{ color: 'var(--faint)', fontSize: 'var(--fz-small)' }}>No structure available.</span>
                  )}

                  <SupercellSymmetryPanel
                    result={scSymResult}
                    busy={scSymBusy}
                    error={scSymErr}
                    canRun={!!pipeline && !!baseStructure && filesList.length > 0}
                    sampleSize={scSymSample}
                    onSampleSizeChange={setScSymSample}
                    onRun={runSupercellSymmetry}
                    onAlignSubgroup={alignSubgroup}
                    onDownloadCIF={downloadAlignedCIF}
                    helpOpen={!!helpOpen.scsym}
                    onToggleHelp={() => toggleHelp('scsym')}
                  />
                </div>
              </details>
            </>
          )}

          {/* ── TAB 3: k-PATH ── */}
          {sideTab === 'path' && (
            <details open className="m-section">
              <summary><span>k-path segments</span></summary>
              <div className="m-section-body" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, paddingBottom: 8, borderBottom: '1px solid var(--line)' }}>
                  <span style={{ fontSize: 'var(--fz-small)', color: 'var(--secondary)' }}>Density (pts/Å⁻¹)</span>
                  <Stepper
                    value={density}
                    onInc={() => { setDensity(d => d + 1); setSegNpoints({}); }}
                    onDec={() => { setDensity(d => Math.max(1, d - 1)); setSegNpoints({}); }}
                  />
                </div>

                {segments.length === 0 ? (
                  <p style={{ margin: '8px 0', color: 'var(--faint)', fontSize: 'var(--fz-small)' }}>
                    No path defined — click points on the Brillouin zone or press <b>Default</b> below it.
                  </p>
                ) : segments.map((s, i) => (
                  <div
                    key={i}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                      padding: '4px 0',
                      borderBottom: i < segments.length - 1 ? '1px solid var(--line)' : 'none',
                    }}
                  >
                    <span style={{ flex: 1, fontFamily: 'var(--font-sym)', fontWeight: 700, fontSize: 14 }}>
                      {displayLabel(s.from)} <span style={{ color: 'var(--faint)', fontWeight: 400 }}>→</span> {displayLabel(s.to)}
                    </span>
                    <Stepper value={s.npoints} onInc={() => setSegN(i, s.npoints + 1)} onDec={() => setSegN(i, s.npoints - 1)} />
                  </div>
                ))}

                <div style={{ paddingTop: 6, fontFamily: 'var(--font-mono)', fontSize: 11.5, color: 'var(--secondary)', display: 'flex', justifyContent: 'space-between' }}>
                  <span>{segments.length} segments</span>
                  <b style={{ color: 'var(--ink)' }}>{totalK} k-points</b>
                </div>
              </div>
            </details>
          )}

          {/* ── TAB 4: RUN SETTINGS ── */}
          {sideTab === 'run' && (
            <>
              <details open className="m-section">
                <summary>
                  <span>Displacement reference</span>
                  <button
                    type="button"
                    className={`m-info ${helpOpen.ref ? 'on' : ''}`}
                    onClick={(e) => { e.preventDefault(); toggleHelp('ref'); }}
                    aria-label="About displacement reference"
                  >?</button>
                </summary>
                <div className="m-section-body" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {helpOpen.ref && (
                    <p className="m-explain" style={{ margin: 0 }}>
                      Sets r₀ in u = r − r₀. Per-atom mean captures thermal motion only; ideal site includes static disorder.
                    </p>
                  )}

                  <div className="m-rule-box" style={{ display: 'flex', flexDirection: 'column', gap: 7, padding: '10px 12px' }}>
                    <span style={{ fontSize: 'var(--fz-small)', fontWeight: 600, color: 'var(--secondary)' }}>Equilibrium site r₀</span>
                    <RadioOption checked={referenceMode === 'per-atom'} onChange={() => setReferenceMode('per-atom')} note="default">
                      Per-atom mean
                    </RadioOption>
                    <RadioOption checked={referenceMode === 'symmetrized'} onChange={() => setReferenceMode('symmetrized')}>
                      Ideal site (symmetrized)
                    </RadioOption>
                  </div>

                  <div className="m-rule-box" style={{ display: 'flex', flexDirection: 'column', gap: 7, padding: '10px 12px' }}>
                    <span style={{ fontSize: 'var(--fz-small)', fontWeight: 600, color: 'var(--secondary)' }}>Average source</span>
                    <RadioOption checked={refMode === 'average'} onChange={() => setRefMode('average')} note="default">
                      Ensemble average
                    </RadioOption>
                    <RadioOption checked={refMode === 'file'} onChange={() => setRefMode('file')}>
                      Equilibrium .rmc6f file
                    </RadioOption>
                    {refMode === 'file' && (
                      <select value={refName} onChange={e => setRefName(e.target.value)} className="m-select mono" style={{ width: '100%' }}>
                        <option value="">(select an .rmc6f file)</option>
                        {rmc6fList.map(r => <option key={r.name} value={r.name}>{r.name}</option>)}
                      </select>
                    )}
                  </div>
                </div>
              </details>

              <details open className="m-section">
                <summary><span>Parameters &amp; DOS</span></summary>
                <div className="m-section-body" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                    <Field label="Temperature T (K)" value={temperature} onChange={v => setTemperature(v)} step={1} />
                    <Field label="Degeneracy tol" value={degenerateTol} onChange={v => setDegenerateTol(v)} step={0.001} />
                  </div>

                  <div className="m-rule-box" style={{ padding: '10px 12px', display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <label style={{ display: 'inline-flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontWeight: 600 }}>
                      <input type="checkbox" checked={runDos} onChange={e => setRunDos(e.target.checked)} />
                      Compute Phonon DOS
                    </label>
                    {runDos && (
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 'var(--fz-small)', fontFamily: 'var(--font-mono)', color: 'var(--secondary)' }}>
                        <span>q-grid</span>
                        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                          <Stepper value={dosN} onInc={() => setDosN(n => Math.min(40, n + 1))} onDec={() => setDosN(n => Math.max(2, n - 1))} />
                          <span className="tabular-nums">³ = {dosN ** 3}</span>
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </details>
            </>
          )}
        </div>

        {/* ── Collapsible Log Drawer (MOTIF .m-drawer pattern) ── */}
        <div className="m-drawer">
          <div className="m-drawer-head" onClick={() => setLogOpen(o => !o)} role="button" tabIndex={0} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setLogOpen(o => !o); } }}>
            <span>Log {logLines.length > 0 ? `(${logLines.length})` : ''}</span>
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
              {logOpen && logLines.length > 0 && (
                <>
                  <button type="button" onClick={copyLog} className="m-btn subtle sm" style={{ height: 22, fontSize: 11 }}>{copiedLog ? 'Copied' : 'Copy'}</button>
                  <button type="button" onClick={(e) => { e.stopPropagation(); setLogLines([]); }} className="m-btn subtle sm" style={{ height: 22, fontSize: 11 }}>Clear</button>
                </>
              )}
              <span style={{ color: 'var(--secondary)' }}>{logOpen ? '▾' : '▴'}</span>
            </div>
          </div>
          {logOpen && (
            <div
              ref={logEl}
              style={{
                background: 'var(--muted)',
                borderTop: '1px solid var(--line)',
                padding: '8px 14px',
                height: 130,
                overflowY: 'auto',
                fontFamily: 'var(--font-mono)',
                fontSize: 11.5,
                lineHeight: 1.6,
                color: 'var(--ink)',
              }}
            >
              {logLines.length === 0 ? (
                <div style={{ color: 'var(--faint)' }}>› console output will appear here…</div>
              ) : logLines.map((t, i) => (
                <div key={i} style={{ whiteSpace: 'pre-wrap' }}>
                  <span style={{ color: 'var(--faint)' }}>›</span> {t}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* ── Pinned Run Dock (always 1 click away from any tab) ── */}
        <div className="m-run-dock">
          <div className="tabular-nums" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 8, fontSize: 11.5, fontFamily: 'var(--font-mono)', color: 'var(--secondary)' }}>
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {symInfo ? symInfo.spaceGroup : '—'} · <b style={{ color: 'var(--ink)' }}>{nBranches}</b> br · <b style={{ color: 'var(--ink)' }}>{totalK}</b> k-pts
              {excludeBad && flaggedInRun > 0 ? ` · −${flaggedInRun}` : ''}
            </span>
            {(isProcessing || progress > 0) && (
              <b style={{ color: 'var(--ink)' }}>{Math.round(progress)}%</b>
            )}
          </div>

          {(isProcessing || progress > 0) && (
            <div style={{ height: 5, borderRadius: 3, background: 'var(--muted2)', overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${progress}%`, background: 'var(--ink)', transition: 'width 0.3s ease' }} />
            </div>
          )}

          {progressText && (
            <div style={{ fontSize: 11.5, fontFamily: 'var(--font-mono)', color: 'var(--secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {progressText}
            </div>
          )}

          {isProcessing ? (
            <button
              type="button"
              onClick={() => { pipeline?.cancel(); setProgressText('Cancelling…'); }}
              className="m-btn danger lg"
              style={{ width: '100%' }}
            >
              ■ Cancel calculation
            </button>
          ) : (
            <button
              type="button"
              onClick={run}
              disabled={!canRun}
              className="m-btn primary lg"
              style={{ width: '100%' }}
            >
              ▶ Run phonon bands
            </button>
          )}
        </div>
      </aside>
    </div>
  );
}

/* ── Building blocks ─────────────────────────────────────────────────────── */

function InfoRow({ k, v, mono, strong, last }) {
  return (
    <tr>
      <td style={{ padding: '6px 8px 6px 0', color: 'var(--secondary)', borderBottom: last ? 'none' : '1px solid var(--line)', width: '42%' }}>{k}</td>
      <td
        className="tabular-nums"
        style={{
          padding: '6px 0',
          textAlign: 'right',
          borderBottom: last ? 'none' : '1px solid var(--line)',
          fontFamily: mono ? 'var(--font-mono)' : 'inherit',
          fontWeight: strong ? 600 : 400,
          color: 'var(--ink)',
          overflowWrap: 'anywhere',
        }}
      >
        {v}
      </td>
    </tr>
  );
}

function Stepper({ value, onInc, onDec }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', border: '1px solid var(--control)', borderRadius: 'var(--radius)', overflow: 'hidden', background: 'var(--surface)' }}>
      <button
        type="button"
        onClick={onDec}
        aria-label="Decrease"
        style={{ width: 24, height: 26, border: 0, background: 'none', color: 'var(--ink)', cursor: 'pointer', fontWeight: 600 }}
      >−</button>
      <span
        className="tabular-nums"
        style={{ minWidth: 30, textAlign: 'center', fontFamily: 'var(--font-mono)', fontSize: 12.5, fontWeight: 600, borderLeft: '1px solid var(--line)', borderRight: '1px solid var(--line)', padding: '0 4px', lineHeight: '26px' }}
      >
        {value}
      </span>
      <button
        type="button"
        onClick={onInc}
        aria-label="Increase"
        style={{ width: 24, height: 26, border: 0, background: 'none', color: 'var(--ink)', cursor: 'pointer', fontWeight: 600 }}
      >+</button>
    </span>
  );
}

function Field({ label, value, onChange, step }) {
  const [draft, setDraft] = useState(String(value));
  useEffect(() => {
    setDraft(prev => (parseFloat(prev) === value ? prev : String(value)));
  }, [value]);
  return (
    <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <span style={{ fontSize: 'var(--fz-small)', color: 'var(--secondary)' }}>{label}</span>
      <input
        type="number"
        step={step}
        value={draft}
        onChange={e => {
          const s = e.target.value;
          setDraft(s);
          const v = parseFloat(s);
          if (Number.isFinite(v)) onChange(v);
        }}
        onBlur={() => setDraft(String(value))}
        className="m-input mono tabular-nums"
        style={{ width: '100%' }}
      />
    </label>
  );
}

function RadioOption({ checked, onChange, note, children }) {
  return (
    <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: 'var(--fz-control)', color: 'var(--ink)' }}>
      <input type="radio" checked={checked} onChange={onChange} style={{ accentColor: 'var(--ink)' }} />
      <span style={{ fontWeight: checked ? 600 : 400 }}>{children}</span>
      {note && <span style={{ fontSize: 11, color: 'var(--faint)', fontFamily: 'var(--font-mono)' }}>{note}</span>}
    </label>
  );
}
