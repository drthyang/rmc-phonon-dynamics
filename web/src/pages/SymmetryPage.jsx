import React, { useState, useMemo, useCallback, useRef } from 'react';
import {
  listConfigs,
  findStructureFile,
  readBaseStructure,
  generateUnitCellCIF,
} from '../io/readers';
import {
  analyzeSupercellFiles,
  alignSubgroupFromResult,
  buildAlignedCifText,
} from '../compute/supercellAnalyzer';
import { conventionalLattice } from '../math/reciprocal';
import { findSpaceGroupOps, siteOrbits, symmetryLadder, wyckoffLetter } from '../math/symmetry';
import { DEFAULT_COLORS } from '../constants';
import { modelFromText } from '../io/phonopyDM';
import { downloadString } from '../io/writers';
import CrystalViewer from '../components/CrystalViewer';
import LineChart from '../components/LineChart';

const SUB = { '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉' };
const sub = (n) => String(n).split('').map(c => SUB[c] || c).join('');

const DOMAIN_PALETTE = [
  '#c95d3a', '#2b6cb0', '#2f855a', '#b7791f',
  '#7b4b94', '#1d8a8a', '#b83280', '#5a67d8',
];

/**
 * Step 1: Guided Multi-Config Supercell Symmetry & Domain Studio.
 * Scales seamlessly from 1 .rmc6f file to 500+ RMC ensemble configs in O(1) memory:
 *   • Top Action Strip: Parent vs. Selected Local Subgroup + ensemble count selector + 1-click "Compute Phonons"
 *   • Left Card:  3D Crystal (Parent / Local toggle) OR 2D Domain Map + Radial C_dom(r) & ξ
 *   • Right Card: Local Subgroups (null-calibrated Excess ΔF ± SE, Spatial CV ΔF, Win %, ξ) | k-Space Modes | Convergence | Average Ladder
 */
export default function SymmetryPage({
  candidates,
  onRemoveCandidate,
  onLoadModelFile,
  onGoToPage,
  mlipServer,
  onRunMlipBatch,
  onRetryCandidate,
}) {
  const [sourceLabel, setSourceLabel] = useState(null);
  const [filesPool, setFilesPool] = useState([]);
  const [configFamily, setConfigFamily] = useState('rmc6f');
  const [baseStructure, setBaseStructure] = useState(null);
  const [sampleSize, setSampleSize] = useState(25);

  const [visualView, setVisualView] = useState('crystal');     // 'crystal' | 'domains'
  const [crystalMode, setCrystalMode] = useState('aligned');   // 'parent' | 'aligned'
  const [analysisTab, setAnalysisTab] = useState('subgroups'); // 'subgroups' | 'kspace' | 'convergence' | 'average'
  const [sliceZ, setSliceZ] = useState(0);
  const [domainCfgIdx, setDomainCfgIdx] = useState(0);

  const [symTol, setSymTol] = useState(0.25);
  const [scResult, setScResult] = useState(null);
  const [scBusy, setScBusy] = useState(false);
  const [scProgress, setScProgress] = useState({ done: 0, total: 0, text: '' });
  const [scErr, setScErr] = useState(null);
  const [loadErr, setLoadErr] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [copiedCli, setCopiedCli] = useState(false);
  const abortRef = useRef(null);

  // What to compute in Step 2 (both checked by default!)
  const [includeParent, setIncludeParent] = useState(true);
  const [includeLocal, setIncludeLocal] = useState(true);
  const [mlipModel, setMlipModel] = useState('mace-mp-medium');
  const [mlipSupercell, setMlipSupercell] = useState('2 2 2');
  const [mlipTemp, setMlipTemp] = useState(250);
  const [mlipRelax, setMlipRelax] = useState(true);
  const [localSymprec, setLocalSymprec] = useState(0.002);
  const [showMlipOpts, setShowMlipOpts] = useState(false);

  const cell111 = useMemo(() => [1, 1, 1], []);

  const Aconv = useMemo(
    () => (baseStructure?.v1 ? conventionalLattice(baseStructure.v1, baseStructure.v2, baseStructure.v3, baseStructure.dim) : null),
    [baseStructure]
  );

  const activeParentBasis = useMemo(
    () => scResult?.idealParentBasis || scResult?.refBasis || baseStructure?.basis || null,
    [scResult, baseStructure]
  );

  const symInfo = useMemo(() => {
    if (!Aconv || !activeParentBasis) return null;
    const sg = findSpaceGroupOps(Aconv, activeParentBasis, symTol);
    const orbits = siteOrbits(Aconv, activeParentBasis, sg.ops, symTol);
    const wyckoff = orbits.map(o => {
      const L = wyckoffLetter(sg.spaceGroupNumber, sg.centering, o.size, o.site, o.rep);
      return { el: o.element, label: `${o.size}${L || ` (${o.site})`}`, site: o.site, mult: o.size };
    });
    return { ...sg, orbits, wyckoff };
  }, [Aconv, activeParentBasis, symTol]);

  const symLadder = useMemo(
    () => (Aconv && activeParentBasis ? symmetryLadder(Aconv, activeParentBasis, 1.0) : []),
    [Aconv, activeParentBasis]
  );

  const readout = useMemo(() => {
    if (!baseStructure?.v1) return null;
    const { v1, v2, v3, dim, atomDic } = baseStructure;
    const norm = (v) => Math.hypot(v[0], v[1], v[2]);
    const a = norm(v1) / (dim?.[0] || 1), b = norm(v2) / (dim?.[1] || 1), c = norm(v3) / (dim?.[2] || 1);
    const counts = Object.entries(atomDic).map(([el, arr]) => [el, arr.length]);
    const sites = counts.reduce((s, [, n]) => s + n, 0);
    const nCells = (dim?.[0] || 1) * (dim?.[1] || 1) * (dim?.[2] || 1);
    const formula = counts.map(([el, n]) => n > 1 ? `${el}${sub(n)}` : el).join(' ');
    return {
      formula, sites, totalAtoms: sites * nCells, nCells, counts,
      a: a.toFixed(3), b: b.toFixed(3), c: c.toFixed(3),
      supercell: dim ? dim.join('×') : '—',
    };
  }, [baseStructure]);

  const sortedSubgroups = useMemo(() => {
    if (!scResult?.local?.subgroups?.length) return [];
    const multi = (scResult.nConfigs || 1) >= 3;
    return [...scResult.local.subgroups]
      .filter(s => s.breakingDoF > 0 && s.dof < scResult.local.internalDoF)
      .sort((a, b) => {
        const sa = multi ? a.medianExcessDeltaF : a.excessDeltaF;
        const sb = multi ? b.medianExcessDeltaF : b.excessDeltaF;
        return (sb - sa) || (b.cvBreakingFStat - a.cvBreakingFStat);
      });
  }, [scResult]);

  const activeSubgroup = useMemo(() => {
    if (!sortedSubgroups.length) return null;
    const targetSg = scResult?.aligned?.targetSpaceGroup;
    return sortedSubgroups.find(s => s.spaceGroup === targetSg) || sortedSubgroups[0];
  }, [sortedSubgroups, scResult]);

  const previewStruct = useMemo(() => {
    if (!baseStructure?.basis) return null;
    const useAligned = crystalMode === 'aligned' && scResult?.aligned?.symmetrizedBasis;
    const b = useAligned ? scResult.aligned.symmetrizedBasis : (activeParentBasis || baseStructure.basis);
    return {
      v1: baseStructure.v1, v2: baseStructure.v2, v3: baseStructure.v3, dim: baseStructure.dim,
      atomDic: baseStructure.atomDic,
      uniqueRN: b.map((x, i) => x.rn ?? i),
      atomType: b.map((x, i) => x.rn ?? i),
      hsym_xyz: Float64Array.from(b.flatMap(x => x.frac)),
      cellIdx: new Float64Array(b.length * 3),
    };
  }, [baseStructure, activeParentBasis, crystalMode, scResult]);

  const runAnalysis = useCallback(async (files, family, base, nSample, tol) => {
    if (!files?.length || !base) return;
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setScBusy(true);
    setScErr(null);
    setDomainCfgIdx(0);
    setScProgress({ done: 0, total: Math.min(nSample, files.length), text: 'Starting…' });
    await new Promise(r => setTimeout(r, 16));
    try {
      const res = await analyzeSupercellFiles(files, family, base, {
        sampleSize: nSample,
        parentTol: tol,
        signal: ctrl.signal,
        onProgress: (done, total, text) => setScProgress({ done, total, text }),
      });
      setScResult(res);
      setCrystalMode(res.aligned ? 'aligned' : 'parent');
    } catch (err) {
      if (err?.message !== 'Cancelled') {
        console.error(err);
        setScErr(err.message || String(err));
      }
    } finally {
      if (abortRef.current === ctrl) setScBusy(false);
    }
  }, []);

  const handleFilesSelected = async (fileList) => {
    const arr = Array.from(fileList || []);
    if (!arr.length) return;
    if (arr.length === 1) {
      const lower = arr[0].name.toLowerCase();
      if (lower.endsWith('.json') || lower.endsWith('.yaml') || lower.endsWith('.yml')) {
        try {
          setLoadErr(null);
          onLoadModelFile(modelFromText(await arr[0].text()), arr[0].name);
          return;
        } catch (err) {
          setLoadErr(`Could not parse ${arr[0].name}: ${err.message}`);
          return;
        }
      }
    }
    const rmcFiles = arr
      .filter(f => f.name.toLowerCase().endsWith('.rmc6f') && !f.name.toLowerCase().includes('average'))
      .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
    if (!rmcFiles.length) {
      setLoadErr('Please select one or more .rmc6f supercell files (or a saved _dm.json / band.yaml).');
      return;
    }
    try {
      setLoadErr(null); setScErr(null); setScResult(null);
      const base = await readBaseStructure(rmcFiles[0]);
      const nRun = Math.min(rmcFiles.length, 50);
      setBaseStructure(base);
      setFilesPool(rmcFiles);
      setConfigFamily('rmc6f');
      setSampleSize(nRun);
      setSourceLabel(rmcFiles.length === 1 ? rmcFiles[0].name : `${rmcFiles.length} .rmc6f files`);
      await runAnalysis(rmcFiles, 'rmc6f', base, nRun, symTol);
    } catch (err) {
      console.error(err);
      setLoadErr(err.message || 'Failed to parse .rmc6f file.');
    }
  };

  const handleSelectFolder = async () => {
    if (typeof window.showDirectoryPicker !== 'function') {
      setLoadErr('Folder picker requires Chrome or Edge. You can still select multiple .rmc6f files in any browser.');
      return;
    }
    try {
      const dh = await window.showDirectoryPicker({ mode: 'read' });
      setLoadErr(null); setScErr(null); setScResult(null);
      const { files, family } = await listConfigs(dh);
      if (!files.length) { setLoadErr(`No .rmc6f or Frac*.txt files found in "${dh.name}".`); return; }
      const structHandle = family === 'rmc6f' ? files[0] : await findStructureFile(dh);
      if (!structHandle) { setLoadErr(`No companion .rmc6f found in "${dh.name}".`); return; }
      const base = await readBaseStructure(structHandle);
      const nRun = Math.min(files.length, 25);
      setBaseStructure(base);
      setFilesPool(files);
      setConfigFamily(family);
      setSampleSize(nRun);
      setSourceLabel(`${dh.name}/ (${files.length} configs)`);
      await runAnalysis(files, family, base, nRun, symTol);
    } catch (err) {
      if (err?.name !== 'AbortError') setLoadErr(err.message || String(err));
    }
  };

  const handleChangeSampleSize = (n) => {
    setSampleSize(n);
    if (filesPool.length && baseStructure) {
      runAnalysis(filesPool, configFamily, baseStructure, n, symTol);
    }
  };

  const selectSubgroup = (sg) => {
    if (!scResult) return;
    const aligned = alignSubgroupFromResult(scResult, sg);
    if (aligned) {
      setScResult(prev => (prev ? { ...prev, aligned } : prev));
      setDomainCfgIdx(0);
    }
  };

  const buildSelectedSpecs = () => {
    const specs = [];
    const stem = (sourceLabel || 'rmc').replace(/\.(rmc6f|cif)$/i, '').replace(/[^A-Za-z0-9_-]+/g, '_');
    if (includeParent && Aconv && activeParentBasis) {
      const sg = symInfo?.spaceGroup || 'P1';
      specs.push({
        label: `Parent · ${sg}`,
        spaceGroup: sg,
        kind: 'parent',
        symprec: 'auto',
        filename: `${stem}_parent_${sg.replace(/[^A-Za-z0-9_-]+/g, '')}.cif`,
        cifText: generateUnitCellCIF(Aconv, activeParentBasis, {
          title: `${stem}_parent_${sg}`,
          spaceGroup: sg,
          comment: `Parent average unit cell from ${sourceLabel} (tol=${symTol.toFixed(2)} A).`,
        }),
      });
    }
    if (includeLocal && scResult?.aligned) {
      const sg = scResult.aligned.verifiedSpaceGroup || scResult.aligned.targetSpaceGroup || 'local';
      specs.push({
        label: `Local · ${sg}${scResult.nConfigs > 1 ? ` (${scResult.nConfigs} cfg)` : ''}`,
        spaceGroup: sg,
        kind: 'local-subgroup',
        symprec: localSymprec,
        filename: `${stem}_local_${sg.replace(/[^A-Za-z0-9_-]+/g, '')}.cif`,
        cifText: buildAlignedCifText(scResult.aligned, { projected: true, title: `${stem}_${sg}` }),
      });
    }
    return specs;
  };

  const mlipParams = () => ({
    model: mlipModel,
    supercell: mlipSupercell.trim().split(/\s+/).map(Number),
    temperature: mlipTemp,
    relax: mlipRelax,
  });

  const handleComputePhonons = () => {
    const specs = buildSelectedSpecs();
    if (!specs.length) return;
    if (mlipServer?.connected) {
      onRunMlipBatch(specs, mlipParams());
    } else {
      for (const s of specs) downloadString(s.cifText, s.filename, 'chemical/x-cif');
    }
  };

  const handleCopyCli = () => {
    const specs = buildSelectedSpecs();
    const relaxFlag = mlipRelax ? '' : ' --no-relax';
    const cmds = specs.map(s =>
      `uv run rmcph-mlip run ${s.filename} --model ${mlipModel} --supercell ${mlipSupercell} -T ${mlipTemp} --symprec ${s.symprec ?? 'auto'}${relaxFlag} --out results/${s.filename.replace(/\.cif$/i, '')}`
    ).join('\n');
    navigator.clipboard?.writeText(cmds);
    setCopiedCli(true);
    setTimeout(() => setCopiedCli(false), 1800);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer?.files?.length) handleFilesSelected(e.dataTransfer.files);
  };

  // Convergence & Radial C_dom(r) chart series
  const convergenceSeries = useMemo(() => {
    const conv = scResult?.convergence || [];
    if (conv.length < 2) return [];
    return [
      { name: 'Excess ΔF (over null)', points: conv.map(p => [p.n, p.excessDeltaF]), color: '#2563eb', width: 2 },
      { name: 'Spatial CV ΔF (bias-free)', points: conv.map(p => [p.n, p.cvDeltaF]), color: '#157347', width: 1.8, dashed: true },
    ];
  }, [scResult]);

  const radialSeries = useMemo(() => {
    if (!activeSubgroup?.radialCorr?.length) return [];
    const pts = [[0, 1.0], ...activeSubgroup.radialCorr.map((c, i) => [i + 1, c])];
    return [{ name: `${activeSubgroup.spaceGroup} C_dom(r)`, points: pts, color: '#c95d3a', width: 2 }];
  }, [activeSubgroup]);

  // ── Clean Editorial Landing State ─────────────────────────────────────────
  if (!baseStructure) {
    return (
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        style={{
          flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
          background: dragOver ? 'var(--muted)' : 'var(--page)', padding: '48px 24px',
        }}
      >
        <div style={{ width: 'min(680px, 100%)', textAlign: 'left', lineHeight: 1.55 }}>
          <h1 style={{ fontSize: 36, fontWeight: 700, letterSpacing: '-0.035em', lineHeight: 1.08, margin: '0 0 14px', color: 'var(--ink)' }}>
            Modern phonons from RMC — start with supercell symmetry.
          </h1>
          <p style={{ margin: '0 0 22px', color: 'var(--secondary)', fontSize: 15 }}>
            Drop <b style={{ color: 'var(--ink)' }}>1 or many <code style={{ fontFamily: 'var(--font-mono)', fontSize: 13.5 }}>.rmc6f</code> supercells</b> (or a full ensemble folder) to stream null-calibrated local symmetry breaking, spatial cross-validated domains, and correlation length ξ — then run MLIP phonons and validate directly against INS.
          </p>

          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', marginBottom: 24 }}>
            <label className="m-button primary lg">
              Open .rmc6f file(s)…
              <input type="file" accept=".rmc6f" multiple onChange={e => handleFilesSelected(e.target.files)} style={{ display: 'none' }} />
            </label>
            {typeof window !== 'undefined' && typeof window.showDirectoryPicker === 'function' && (
              <button type="button" onClick={handleSelectFolder} className="m-btn lg">
                Open ensemble folder (10–500+)…
              </button>
            )}
            <label className="m-button subtle lg">
              Load saved _dm.json / band.yaml…
              <input type="file" accept=".json,.yaml,.yml" onChange={e => handleFilesSelected(e.target.files)} style={{ display: 'none' }} />
            </label>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 12, paddingTop: 18, borderTop: '1px solid var(--line)', fontSize: 12.5, color: 'var(--secondary)' }}>
            <div><b style={{ color: 'var(--ink)' }}>1 · Ensemble Symmetry</b><br />Stream 1–500+ configs in O(1) RAM with null calibration &amp; ξ(r).</div>
            <div><b style={{ color: 'var(--ink)' }}>2 · MLIP Phonons</b><br />Compare Parent vs. ensemble-averaged Local bands in seconds.</div>
            <div><b style={{ color: 'var(--ink)' }}>3 · INS Compare</b><br />Rank models quantitatively against 1D GDOS &amp; 2D S(|Q|,E).</div>
          </div>

          {loadErr && <div className="m-notice bad" style={{ margin: '18px 0 0' }}>{loadErr}</div>}
        </div>
      </div>
    );
  }

  // ── Guided Single-Screen Workspace ────────────────────────────────────────
  const parentSg = symInfo?.spaceGroup || 'P1';
  const hasRealDisplacements = (scResult?.aligned?.symRms || 0) > 1e-4;
  const localSg = hasRealDisplacements
    ? (scResult?.aligned?.verifiedSpaceGroup || activeSubgroup?.spaceGroup || null)
    : (activeSubgroup?.spaceGroup || null);
  const selectedCount = (includeParent ? 1 : 0) + (includeLocal && localSg ? 1 : 0);
  const anyRunning = candidates.some(c => c.mlipRunning);
  const readyModels = candidates.filter(c => c.model?.bands);
  const isMulti = (scResult?.nConfigs || 1) >= 2;
  const maxExcessDF = Math.max(0.2, ...sortedSubgroups.map(s => Math.max(0, isMulti ? s.medianExcessDeltaF : s.excessDeltaF)));
  const maxSnr = Math.max(1, ...(scResult?.spectrum?.topK || []).map(k => k.snr || 0));

  const samplePresets = [10, 25, 50, 100].filter(n => n < filesPool.length);
  const activeGridObj = activeSubgroup?.savedGrids?.[Math.min(domainCfgIdx, (activeSubgroup?.savedGrids?.length || 1) - 1)] || null;
  const activeGrid = activeGridObj?.grid || activeSubgroup?.domainGrid || null;

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
      style={{
        flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', gap: 14,
        padding: '14px var(--edge)', overflow: 'hidden', background: 'var(--page)',
      }}
    >
      {/* ════════ TOP GUIDED ACTION STRIP ════════ */}
      <div className="m-frame" style={{ flex: 'none', padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: 10, background: 'var(--surface)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 14, flexWrap: 'wrap' }}>
          {/* File + Ensemble identity */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <b style={{ fontSize: 16 }}>{readout?.formula || 'Supercell'}</b>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--secondary)' }}>
              {sourceLabel}
            </span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11.5, color: 'var(--faint)' }}>
              {readout?.supercell}
              {scResult ? ` · ${scResult.nConfigs} cfg (${(scResult.nConfigs * (readout?.nCells || 1)).toLocaleString()} cells)` : ''}
            </span>
            {scResult?.skippedConfigs?.length > 0 && (
              <span
                title={scResult.skippedConfigs.map(s => `${s.name}: ${s.reason}`).join('\n')}
                style={{ fontFamily: 'var(--font-mono)', fontSize: 11, padding: '1px 6px', borderRadius: 4, background: 'var(--muted)', color: 'var(--secondary)' }}
              >
                skipped {scResult.skippedConfigs.length} unconverged
              </span>
            )}
            <label className="m-button subtle sm">
              Open files…
              <input type="file" accept=".rmc6f" multiple onChange={e => handleFilesSelected(e.target.files)} style={{ display: 'none' }} />
            </label>
            {typeof window !== 'undefined' && typeof window.showDirectoryPicker === 'function' && (
              <button type="button" onClick={handleSelectFolder} className="m-btn subtle sm">Folder…</button>
            )}
          </div>

          {/* Primary Step 1 -> Step 2 Action */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6, padding: '5px 10px', borderRadius: 6, border: '1px solid var(--line)', background: includeParent ? 'var(--muted)' : 'transparent', fontSize: 12.5, cursor: 'pointer' }}>
              <input type="checkbox" checked={includeParent} onChange={e => setIncludeParent(e.target.checked)} />
              <span>Parent <b>{parentSg}</b></span>
            </label>

            {localSg && (
              <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6, padding: '5px 10px', borderRadius: 6, border: '1px solid var(--line)', background: includeLocal ? 'var(--note-bg)' : 'transparent', fontSize: 12.5, cursor: 'pointer' }}>
                <input type="checkbox" checked={includeLocal} onChange={e => setIncludeLocal(e.target.checked)} />
                <span>Local <b>{localSg}</b></span>
                {activeSubgroup && (
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--note-ink)' }}>
                    {activeSubgroup.zScore !== null
                      ? `${activeSubgroup.zScore.toFixed(1)}σ`
                      : `+ΔF ${activeSubgroup.excessDeltaF.toFixed(2)}`}
                  </span>
                )}
              </label>
            )}

            <button type="button" className="m-btn sm subtle" onClick={() => setShowMlipOpts(v => !v)}>
              {mlipModel.replace('mace-mp-', 'MACE-')} · {mlipSupercell.replace(/\s+/g, '×')} · {mlipTemp}K ▾
            </button>

            {mlipServer?.connected ? (
              <button type="button" className="m-btn primary" disabled={selectedCount === 0 || anyRunning || scBusy} onClick={handleComputePhonons}>
                {anyRunning ? 'Computing phonons…' : `▶ Compute Phonons (${selectedCount})`}
              </button>
            ) : (
              <>
                <button type="button" className="m-btn primary" disabled={selectedCount === 0} onClick={handleComputePhonons}>
                  Download CIF{selectedCount > 1 ? 's' : ''}
                </button>
                <button type="button" className="m-btn" disabled={selectedCount === 0} onClick={handleCopyCli}>
                  {copiedCli ? 'Copied CLI ✓' : 'Copy MLIP CLI'}
                </button>
              </>
            )}
          </div>
        </div>

        {/* Multi-config ensemble depth selector + live streaming progress */}
        {(filesPool.length > 1 || scBusy) && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', paddingTop: 8, borderTop: '1px solid var(--line)', fontSize: 12 }}>
            {filesPool.length > 1 && (
              <>
                <span style={{ color: 'var(--secondary)' }}>Ensemble depth:</span>
                {samplePresets.map(n => (
                  <button key={n} type="button" disabled={scBusy} className={`m-btn sm ${sampleSize === n ? 'active' : ''}`} onClick={() => handleChangeSampleSize(n)}>
                    {n}
                  </button>
                ))}
                <button
                  type="button"
                  disabled={scBusy}
                  className={`m-btn sm ${sampleSize >= filesPool.length ? 'active' : ''}`}
                  onClick={() => handleChangeSampleSize(filesPool.length)}
                >
                  All ({filesPool.length})
                </button>
              </>
            )}

            {scBusy && (
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8, marginLeft: 'auto' }}>
                <div style={{ width: 120, height: 6, borderRadius: 3, background: 'var(--muted)', overflow: 'hidden' }}>
                  <div style={{ width: `${Math.round(100 * scProgress.done / Math.max(1, scProgress.total))}%`, height: '100%', background: 'var(--fig-accent)', transition: 'width 0.1s' }} />
                </div>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11.5, color: 'var(--secondary)' }}>
                  {scProgress.text}
                </span>
                <button type="button" className="m-btn sm" onClick={() => abortRef.current?.abort()}>Cancel</button>
              </div>
            )}
          </div>
        )}

        {/* Collapsible MLIP Settings Row */}
        {showMlipOpts && (
          <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', alignItems: 'center', paddingTop: 8, borderTop: '1px solid var(--line)', fontSize: 12 }}>
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
              <span style={{ color: 'var(--secondary)' }}>MLIP Model</span>
              <select value={mlipModel} onChange={e => setMlipModel(e.target.value)} className="m-input" style={{ height: 26, padding: '0 8px' }}>
                <option value="mace-mp-medium">mace-mp-medium (local)</option>
                <option value="mace-mp-small">mace-mp-small</option>
                <option value="mace-mp-large">mace-mp-large</option>
              </select>
            </label>
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
              <span style={{ color: 'var(--secondary)' }}>Phonopy supercell</span>
              <input value={mlipSupercell} onChange={e => setMlipSupercell(e.target.value)} className="m-input mono" style={{ width: 68, height: 26, textAlign: 'center' }} />
            </label>
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
              <span style={{ color: 'var(--secondary)' }}>T (K)</span>
              <input type="number" value={mlipTemp} onChange={e => setMlipTemp(Number(e.target.value) || 5)} className="m-input mono" style={{ width: 62, height: 26, textAlign: 'center' }} />
            </label>
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }} title="Tight tolerance keeps the local subgroup instead of snapping back to the parent symmetry">
              <span style={{ color: 'var(--secondary)' }}>Local symprec (Å)</span>
              <input type="number" step="0.001" min="0.0001" max="0.2" value={localSymprec} onChange={e => setLocalSymprec(Math.max(0.0001, parseFloat(e.target.value) || 0.002))} className="m-input mono" style={{ width: 68, height: 26, textAlign: 'center' }} />
            </label>
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6, cursor: 'pointer' }} title="Uncheck to evaluate MLIP phonons directly at the RMC local geometry without BFGS relaxation">
              <input type="checkbox" checked={mlipRelax} onChange={e => setMlipRelax(e.target.checked)} />
              <span>Relax atoms</span>
            </label>
            <button type="button" className="m-btn sm" onClick={handleCopyCli} style={{ marginLeft: 'auto' }}>
              {copiedCli ? 'Copied CLI ✓' : 'Copy CLI command'}
            </button>
          </div>
        )}

        {/* Live Job / Results Strip */}
        {candidates.length > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', paddingTop: 8, borderTop: '1px solid var(--line)', fontSize: 12 }}>
            {candidates.map(c => (
              <div
                key={c.id}
                style={{
                  display: 'inline-flex', alignItems: 'center', gap: 6, padding: '3px 9px', borderRadius: 5,
                  border: `1px solid ${c.mlipError ? 'var(--warn-ink)' : 'var(--line)'}`,
                  background: c.mlipError ? 'var(--warn-bg)' : c.model ? 'var(--note-bg)' : 'var(--muted)',
                }}
              >
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: c.color }} />
                <b>{c.label}</b>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: c.mlipError ? 'var(--warn-ink)' : 'var(--secondary)' }}>
                  {c.mlipRunning ? (c.mlipStatus || 'Running…') : c.mlipError ? c.mlipError : c.model ? (c.mlipStatus || 'Ready ✓') : 'Pinned'}
                </span>
                {c.mlipError && mlipServer?.connected && (
                  <button type="button" className="m-btn sm" style={{ height: 20, padding: '0 6px', fontSize: 10.5 }} onClick={() => onRetryCandidate(c, mlipParams())}>
                    Retry
                  </button>
                )}
                <button type="button" onClick={() => onRemoveCandidate(c.id)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--faint)', padding: '0 2px' }}>×</button>
              </div>
            ))}
            {readyModels.length > 0 && (
              <div style={{ marginLeft: 'auto', display: 'inline-flex', gap: 6 }}>
                <button type="button" className="m-btn sm active" onClick={() => onGoToPage('phonons')}>
                  2 · View Phonons →
                </button>
                <button type="button" className="m-btn sm" onClick={() => onGoToPage('ins')}>
                  3 · Compare INS →
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {(loadErr || scErr) && <div className="m-notice bad" style={{ margin: 0 }}>{loadErr || scErr}</div>}

      {/* ════════ MAIN 2-COLUMN WORKSPACE ════════ */}
      <div style={{ flex: 1, minHeight: 0, display: 'grid', gridTemplateColumns: 'minmax(300px, 42%) minmax(380px, 1fr)', gap: 14 }}>
        {/* ── LEFT COLUMN: 3D Crystal OR 2D Domain Map + Radial C_dom(r) ── */}
        <div className="m-frame" style={{ display: 'flex', flexDirection: 'column', minHeight: 0, overflow: 'hidden' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px', borderBottom: '1px solid var(--line)', gap: 8, flexWrap: 'wrap' }}>
            <div className="m-tools" role="tablist">
              <button type="button" className={visualView === 'crystal' ? 'on' : 'dim'} onClick={() => setVisualView('crystal')}>
                3D Crystal
              </button>
              <button type="button" className={visualView === 'domains' ? 'on' : 'dim'} onClick={() => setVisualView('domains')}>
                Domain Map &amp; ξ
              </button>
            </div>

            {visualView === 'crystal' && (
              <div style={{ display: 'inline-flex', gap: 4 }}>
                <button type="button" className={`m-btn sm ${crystalMode === 'parent' ? 'active' : ''}`} onClick={() => setCrystalMode('parent')}>
                  Parent ({parentSg})
                </button>
                <button type="button" className={`m-btn sm ${crystalMode === 'aligned' ? 'active' : ''}`} disabled={!scResult?.aligned} onClick={() => setCrystalMode('aligned')}>
                  Local ({localSg || '—'})
                </button>
              </div>
            )}
          </div>

          {/* 3D Crystal View */}
          <div style={{ display: visualView === 'crystal' ? 'flex' : 'none', flex: 1, flexDirection: 'column', minHeight: 0 }}>
            <div style={{ flex: 1, minHeight: 260, background: 'var(--surface)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              {previewStruct && (
                <CrystalViewer baseStructure={previewStruct} eigenvector={null} isPlaying={false} supercell={cell111} showCell showBonds />
              )}
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10, flexWrap: 'wrap', padding: '8px 14px', borderTop: '1px solid var(--line)', fontSize: 12, color: 'var(--secondary)' }}>
              <div style={{ display: 'inline-flex', gap: 12, flexWrap: 'wrap' }}>
                {readout?.counts.map(([el, count]) => (
                  <span key={el} style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
                    <span style={{ width: 9, height: 9, borderRadius: '50%', background: DEFAULT_COLORS[el] || '#ccc', boxShadow: 'inset 0 0 0 1px rgba(0,0,0,0.2)' }} />
                    <b style={{ color: 'var(--ink)' }}>{el}</b>×{count}
                  </span>
                ))}
              </div>
              {crystalMode === 'aligned' && scResult?.aligned && (
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--note-ink)' }}>
                  RMS {scResult.aligned.symRms.toFixed(4)}
                  {scResult.aligned.rmsSE > 0 ? ` ± ${scResult.aligned.rmsSE.toFixed(4)}` : ''} Å
                </span>
              )}
            </div>
          </div>

          {/* 2D Domain Map + Radial C_dom(r) View */}
          {visualView === 'domains' && (
            <div style={{ flex: 1, overflow: 'auto', padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 12 }}>
              {activeGrid && activeSubgroup ? (
                <>
                  {/* Domain physics summary strip */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, padding: '8px 10px', borderRadius: 6, background: 'var(--muted)', border: '1px solid var(--line)', fontSize: 11.5, fontFamily: 'var(--font-mono)' }}>
                    <div>
                      <span style={{ color: 'var(--secondary)' }}>Corr. length ξ</span><br />
                      <b>{activeSubgroup.corrLengthCells.toFixed(2)} cells</b>{' '}
                      <span style={{ whiteSpace: 'nowrap' }}>({activeSubgroup.corrLengthAng.toFixed(1)} Å)</span>
                    </div>
                    <div>
                      <span style={{ color: 'var(--secondary)' }}>3D Cluster avg</span><br />
                      <b>{activeSubgroup.meanClusterCells?.toFixed(1) ?? '—'} cells</b>
                    </div>
                    <div>
                      <span style={{ color: 'var(--secondary)' }}>NN same-domain</span><br />
                      <b>{(100 * (activeSubgroup.neighborMatchRate || 0)).toFixed(1)}%</b>
                      {activeSubgroup.neighborMatchSE !== null && ` ±${(100 * activeSubgroup.neighborMatchSE).toFixed(1)}%`}
                    </div>
                  </div>

                  {/* Controls: config slice + z-layer */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10, flexWrap: 'wrap', fontSize: 12 }}>
                    {activeSubgroup.savedGrids?.length > 1 ? (
                      <select
                        value={domainCfgIdx}
                        onChange={e => setDomainCfgIdx(parseInt(e.target.value, 10))}
                        className="m-input"
                        style={{ height: 26, fontSize: 11.5, padding: '0 6px' }}
                      >
                        {activeSubgroup.savedGrids.map((sg, i) => (
                          <option key={i} value={i}>{sg.configName}</option>
                        ))}
                      </select>
                    ) : (
                      <span><b>{activeSubgroup.spaceGroup}</b> · {activeSubgroup.nVariants} variants</span>
                    )}
                    <label style={{ display: 'inline-flex', alignItems: 'center', gap: 8, fontFamily: 'var(--font-mono)', color: 'var(--secondary)' }}>
                      <span>z = <b style={{ color: 'var(--ink)' }}>{sliceZ}</b>/{scResult.dim[2] - 1}</span>
                      <input type="range" min={0} max={Math.max(0, scResult.dim[2] - 1)} value={Math.min(sliceZ, scResult.dim[2] - 1)} onChange={e => setSliceZ(parseInt(e.target.value, 10))} className="m-range" style={{ width: 96 }} />
                    </label>
                  </div>

                  {/* 2D z-slice grid + ensemble fractions */}
                  <div style={{ display: 'flex', gap: 14, alignItems: 'flex-start', flexWrap: 'wrap' }}>
                    <div
                      style={{
                        display: 'grid',
                        gridTemplateColumns: `repeat(${scResult.dim[1]}, 24px)`,
                        gap: 3, padding: 8, borderRadius: 6, background: 'var(--muted)', border: '1px solid var(--line)',
                      }}
                    >
                      {Array.from({ length: scResult.dim[0] * scResult.dim[1] }, (_, idx) => {
                        const x = Math.floor(idx / scResult.dim[1]);
                        const y = idx % scResult.dim[1];
                        const z = Math.min(sliceZ, scResult.dim[2] - 1);
                        const c = (x * scResult.dim[1] + y) * scResult.dim[2] + z;
                        const dom = activeGrid[c] ?? 0;
                        return (
                          <div
                            key={idx}
                            title={`Cell (${x},${y},${z}) → Domain ${dom + 1}`}
                            style={{
                              width: 24, height: 24, borderRadius: 4,
                              background: DOMAIN_PALETTE[dom % DOMAIN_PALETTE.length],
                              color: '#fff', fontSize: 10, fontFamily: 'var(--font-mono)', fontWeight: 700,
                              display: 'flex', alignItems: 'center', justifyContent: 'center',
                            }}
                          >
                            {dom + 1}
                          </div>
                        );
                      })}
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: 4, fontFamily: 'var(--font-mono)', fontSize: 11.5 }}>
                      <span style={{ color: 'var(--secondary)', fontWeight: 600 }}>Ensemble fractions</span>
                      {activeSubgroup.domainFractions.map((f, i) => (
                        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <span style={{ width: 10, height: 10, borderRadius: 3, background: DOMAIN_PALETTE[i % DOMAIN_PALETTE.length] }} />
                          <span>
                            D{i + 1}: {(100 * f).toFixed(1)}%
                            {activeSubgroup.domainFractionSE ? ` ±${(100 * activeSubgroup.domainFractionSE[i]).toFixed(1)}%` : ''}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Radial Domain Autocorrelation C_dom(r) */}
                  {radialSeries.length > 0 && (
                    <div style={{ paddingTop: 6, borderTop: '1px solid var(--line)' }}>
                      <div style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--secondary)', marginBottom: 4 }}>
                        Radial Domain Autocorrelation C_dom(r)
                      </div>
                      <LineChart series={radialSeries} xLabel="Lag r (unit cells)" yLabel="C_dom(r)" height={140} />
                    </div>
                  )}
                </>
              ) : (
                <div style={{ color: 'var(--faint)', fontSize: 13 }}>
                  Select a multi-variant local subgroup on the right to view its 2D domain orientation map and correlation length ξ.
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── RIGHT COLUMN: Flat 4-Tab Ensemble Symmetry Analysis Card ── */}
        <div className="m-frame" style={{ display: 'flex', flexDirection: 'column', minHeight: 0, overflow: 'hidden' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px', borderBottom: '1px solid var(--line)', gap: 8, flexWrap: 'wrap' }}>
            <div className="m-tools" role="tablist">
              <button type="button" className={analysisTab === 'subgroups' ? 'on' : 'dim'} onClick={() => setAnalysisTab('subgroups')}>
                Local Subgroups
              </button>
              <button type="button" className={analysisTab === 'kspace' ? 'on' : 'dim'} onClick={() => setAnalysisTab('kspace')}>
                k-Space Modes
              </button>
              {isMulti && (
                <button type="button" className={analysisTab === 'convergence' ? 'on' : 'dim'} onClick={() => setAnalysisTab('convergence')}>
                  Convergence ({scResult.nConfigs})
                </button>
              )}
              <button type="button" className={analysisTab === 'average' ? 'on' : 'dim'} onClick={() => setAnalysisTab('average')}>
                Average Ladder
              </button>
            </div>
            <span style={{ fontSize: 11.5, color: 'var(--faint)' }}>
              {analysisTab === 'subgroups' ? 'Null-calibrated · click row to select' : ''}
            </span>
          </div>

          <div style={{ flex: 1, overflow: 'auto', padding: '12px 16px' }}>
            {scBusy && !scResult ? (
              <div style={{ padding: 24, color: 'var(--secondary)', fontSize: 13 }}>
                Streaming supercell configurations… ({scProgress.text})
              </div>
            ) : analysisTab === 'subgroups' ? (
              sortedSubgroups.length > 0 ? (
                <>
                  {!hasRealDisplacements && (
                    <div style={{ marginBottom: 10, padding: '8px 10px', borderRadius: 6, background: 'var(--muted)', fontSize: 12, color: 'var(--secondary)' }}>
                      Zero atomic displacements detected (this looks like an ideal/initial <code style={{ fontFamily: 'var(--font-mono)' }}>.rmc6f</code>). Load converged RMC configurations to measure local symmetry breaking.
                    </div>
                  )}
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--line)', textAlign: 'left', color: 'var(--faint)', fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                        <th style={{ padding: '6px 6px' }}>Subgroup</th>
                        <th style={{ padding: '6px 6px' }}>Vars</th>
                        <th style={{ padding: '6px 6px' }} title="Bias-corrected ΔF minus exact isotropic-noise null expectation">Excess ΔF {isMulti ? '(±SE)' : ''}</th>
                        <th style={{ padding: '6px 6px', textAlign: 'right' }} title="Spatial neighbor cross-validated ΔF (zero within-cell selection bias)">CV ΔF</th>
                        {isMulti && <th style={{ padding: '6px 6px', textAlign: 'right' }} title="Fraction of RMC configs ranking this subgroup #1">Win %</th>}
                        <th style={{ padding: '6px 6px', textAlign: 'right' }}>ξ (Å)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sortedSubgroups.slice(0, 12).map((sg, i) => {
                        const isSel = activeSubgroup?.spaceGroup === sg.spaceGroup;
                        const scoreVal = isMulti ? sg.medianExcessDeltaF : sg.excessDeltaF;
                        return (
                          <tr
                            key={sg.id || i}
                            onClick={() => selectSubgroup(sg)}
                            style={{
                              cursor: 'pointer',
                              borderBottom: '1px solid var(--line)',
                              background: isSel ? 'var(--note-bg)' : 'transparent',
                            }}
                          >
                            <td style={{ padding: '8px 6px', fontWeight: isSel ? 700 : 500, fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap' }}>
                              {isSel && <span style={{ color: 'var(--ok-ink)', marginRight: 5 }}>●</span>}
                              {sg.spaceGroup}
                            </td>
                            <td style={{ padding: '8px 6px', fontFamily: 'var(--font-mono)', color: 'var(--secondary)' }}>
                              {sg.nVariants}
                            </td>
                            <td style={{ padding: '8px 6px' }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                <div style={{ flex: 1, height: 6, borderRadius: 3, background: 'var(--muted)', overflow: 'hidden', minWidth: 48 }}>
                                  <div style={{ width: `${Math.max(0, Math.min(100, (scoreVal / maxExcessDF) * 100))}%`, height: '100%', background: isSel ? 'var(--ok-ink)' : 'var(--fig-accent)' }} />
                                </div>
                                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11.5, whiteSpace: 'nowrap' }}>
                                  {scoreVal >= 0 ? '+' : ''}{scoreVal.toFixed(2)}
                                  {sg.excessDeltaFSE !== null && (
                                    <span style={{ color: 'var(--faint)', fontSize: 10.5 }}> ±{sg.excessDeltaFSE.toFixed(2)}</span>
                                  )}
                                </span>
                              </div>
                            </td>
                            <td style={{ padding: '8px 6px', textAlign: 'right', fontFamily: 'var(--font-mono)', color: sg.cvBreakingFStat > 1.15 ? 'var(--ok-ink)' : 'var(--secondary)', fontWeight: sg.cvBreakingFStat > 1.15 ? 600 : 400 }}>
                              {Number.isFinite(sg.cvBreakingFStat) ? sg.cvBreakingFStat.toFixed(2) : '—'}
                            </td>
                            {isMulti && (
                              <td style={{ padding: '8px 6px', textAlign: 'right', fontFamily: 'var(--font-mono)', fontWeight: sg.winRate > 0.5 ? 700 : 400 }}>
                                {(100 * sg.winRate).toFixed(0)}%
                              </td>
                            )}
                            <td style={{ padding: '8px 6px', textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--secondary)' }}>
                              {sg.nVariants > 1 && sg.corrLengthAng > 0.1 ? `${sg.corrLengthAng.toFixed(1)}` : '—'}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </>
              ) : (
                <div style={{ color: 'var(--faint)', fontSize: 13 }}>No symmetry-breaking local subgroups detected above the parent cell.</div>
              )
            ) : analysisTab === 'kspace' ? (
              scResult?.spectrum?.topK?.length ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  {/* High-symmetry zone-boundary summary (Γ, X, M, R) */}
                  {scResult.spectrum.highSymStars && (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8, padding: '8px 10px', borderRadius: 6, background: 'var(--muted)', border: '1px solid var(--line)', fontFamily: 'var(--font-mono)', fontSize: 11.5 }}>
                      <div><span style={{ color: 'var(--secondary)' }}>Γ (0,0,0)</span><br /><b>{scResult.spectrum.highSymStars.gamma?.toFixed(2)}×</b></div>
                      <div><span style={{ color: 'var(--secondary)' }}>X (½,0,0)</span><br /><b>{scResult.spectrum.highSymStars.X?.toFixed(2) ?? '—'}×</b></div>
                      <div><span style={{ color: 'var(--secondary)' }}>M (½,½,0)</span><br /><b>{scResult.spectrum.highSymStars.M?.toFixed(2) ?? '—'}×</b></div>
                      <div><span style={{ color: 'var(--secondary)' }}>R (½,½,½)</span><br /><b>{scResult.spectrum.highSymStars.R?.toFixed(2) ?? '—'}×</b></div>
                    </div>
                  )}
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--line)', textAlign: 'left', color: 'var(--faint)', fontSize: 11, textTransform: 'uppercase' }}>
                        <th style={{ padding: '6px 8px' }}>k-vector (rlu)</th>
                        <th style={{ padding: '6px 8px' }}>SNR</th>
                        <th style={{ padding: '6px 8px', textAlign: 'right' }}>P(k) (Å²)</th>
                        {isMulti && <th style={{ padding: '6px 8px', textAlign: 'right' }}>Coherent %</th>}
                      </tr>
                    </thead>
                    <tbody>
                      {scResult.spectrum.topK.slice(0, 12).map((k, i) => (
                        <tr key={i} style={{ borderBottom: '1px solid var(--line)' }}>
                          <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>
                            ({k.kFrac.map(v => v.toFixed(3)).join(', ')})
                          </td>
                          <td style={{ padding: '6px 8px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                              <div style={{ flex: 1, height: 6, borderRadius: 3, background: 'var(--muted)', overflow: 'hidden' }}>
                                <div style={{ width: `${Math.min(100, (k.snr / maxSnr) * 100)}%`, height: '100%', background: 'var(--fig-accent)' }} />
                              </div>
                              <span style={{ fontFamily: 'var(--font-mono)', width: 42, textAlign: 'right' }}>{k.snr.toFixed(1)}×</span>
                            </div>
                          </td>
                          <td style={{ padding: '6px 8px', textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--secondary)' }}>
                            {Number.isFinite(k.power) ? k.power.toFixed(3) : '—'}
                          </td>
                          {isMulti && (
                            <td style={{ padding: '6px 8px', textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--secondary)' }}>
                              {k.power > 1e-12 ? `${(100 * k.coherentPower / k.power).toFixed(0)}%` : '0%'}
                            </td>
                          )}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div style={{ color: 'var(--faint)', fontSize: 13 }}>No k-space spectrum available.</div>
              )
            ) : analysisTab === 'convergence' ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                <div style={{ fontSize: 12.5, color: 'var(--secondary)' }}>
                  Tracking how the leading local subgroup&apos;s bias-corrected <b>Excess ΔF</b> and bias-free <b>Spatial CV ΔF</b> stabilize as more RMC configurations are streamed.
                </div>
                {convergenceSeries.length > 0 && (
                  <LineChart series={convergenceSeries} xLabel="Number of RMC configs (M)" yLabel="Symmetry-breaking ΔF" height={240} />
                )}
              </div>
            ) : (
              /* Average Symmetry Ladder */
              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 12.5 }}>
                  <span style={{ color: 'var(--secondary)' }}>Tolerance</span>
                  <input type="range" min={0.02} max={1.0} step={0.01} value={symTol} onChange={e => setSymTol(parseFloat(e.target.value))} className="m-range" style={{ flex: 1 }} />
                  <b style={{ fontFamily: 'var(--font-mono)', width: 54, textAlign: 'right' }}>{symTol.toFixed(2)} Å</b>
                </label>

                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {symLadder.map((b, i) => (
                    <button
                      key={i}
                      type="button"
                      className={`m-btn sm ${parentSg === b.spaceGroup ? 'active' : ''}`}
                      onClick={() => setSymTol(Math.min(1.0, Math.max(0.02, (b.from + b.to) * 0.5)))}
                      style={{ fontFamily: 'var(--font-mono)' }}
                    >
                      {b.spaceGroup} <span style={{ color: 'var(--faint)', fontSize: 10.5 }}>[{b.from.toFixed(2)}–{b.to.toFixed(2)}Å]</span>
                    </button>
                  ))}
                </div>

                {symInfo?.wyckoff?.length > 0 && (
                  <div style={{ fontSize: 12, color: 'var(--secondary)' }}>
                    <div style={{ fontWeight: 600, marginBottom: 6, color: 'var(--ink)' }}>Wyckoff sites ({parentSg})</div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                      {symInfo.wyckoff.map((w, i) => (
                        <span key={i} style={{ fontFamily: 'var(--font-mono)', padding: '2px 8px', borderRadius: 4, background: 'var(--muted)', border: '1px solid var(--line)' }}>
                          <b>{w.el}</b> {w.label}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
