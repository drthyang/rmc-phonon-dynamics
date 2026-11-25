import React, { useState, useMemo, useCallback } from 'react';
import {
  listConfigs,
  findStructureFile,
  readBaseStructure,
  generateUnitCellCIF,
} from '../io/readers';
import {
  analyzeSupercellFiles,
  alignSubgroupFromContext,
  buildAlignedCifText,
} from '../compute/supercellAnalyzer';
import { conventionalLattice } from '../math/reciprocal';
import { findSpaceGroupOps, siteOrbits, symmetryLadder, wyckoffLetter } from '../math/symmetry';
import { DEFAULT_COLORS } from '../constants';
import { modelFromText } from '../io/phonopyDM';
import { downloadString } from '../io/writers';
import CrystalViewer from '../components/CrystalViewer';

const SUB = { '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉' };
const sub = (n) => String(n).split('').map(c => SUB[c] || c).join('');

const DOMAIN_PALETTE = [
  '#c95d3a', '#2b6cb0', '#2f855a', '#b7791f',
  '#7b4b94', '#1d8a8a', '#b83280', '#5a67d8',
];

/**
 * Step 1: Guided Supercell Symmetry & MLIP Launch Studio.
 * Single-screen layout — no nested tab soup:
 *   • Top Action Strip: Parent vs. Selected Local Subgroup checkboxes + 1-click "Compute Phonons" (or Download CIF / Copy CLI)
 *   • Left Card:  3D Crystal (Parent / Local toggle) OR 2D Domain Map
 *   • Right Card: Local Subgroups (click any row to pick your Local candidate) | k-Space Modes | Average Ladder
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
  const [baseStructure, setBaseStructure] = useState(null);

  const [visualView, setVisualView] = useState('crystal');    // 'crystal' | 'domains'
  const [crystalMode, setCrystalMode] = useState('aligned');  // 'parent' | 'aligned'
  const [analysisTab, setAnalysisTab] = useState('subgroups'); // 'subgroups' | 'kspace' | 'average'
  const [sliceZ, setSliceZ] = useState(0);

  const [symTol, setSymTol] = useState(0.25);
  const [scResult, setScResult] = useState(null);
  const [scBusy, setScBusy] = useState(false);
  const [scErr, setScErr] = useState(null);
  const [loadErr, setLoadErr] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [copiedCli, setCopiedCli] = useState(false);

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
    () => scResult?.refBasis || baseStructure?.basis || null,
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
    return [...scResult.local.subgroups]
      .filter(s => s.breakingDoF > 0 && s.dof < scResult.local.internalDoF)
      .sort((a, b) => b.breakingFStat - a.breakingFStat);
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
    setScBusy(true);
    setScErr(null);
    await new Promise(r => setTimeout(r, 16));
    try {
      const res = await analyzeSupercellFiles(files, family, base, { sampleSize: nSample, parentTol: tol });
      setScResult(res);
      setCrystalMode(res.aligned ? 'aligned' : 'parent');
    } catch (err) {
      console.error(err);
      setScErr(err.message || String(err));
    } finally {
      setScBusy(false);
    }
  }, []);

  const handleSingleFile = async (file) => {
    if (!file) return;
    const lower = file.name.toLowerCase();
    if (lower.endsWith('.json') || lower.endsWith('.yaml') || lower.endsWith('.yml')) {
      try {
        setLoadErr(null);
        onLoadModelFile(modelFromText(await file.text()), file.name);
        return;
      } catch (err) {
        setLoadErr(`Could not parse ${file.name}: ${err.message}`);
        return;
      }
    }
    try {
      setLoadErr(null); setScErr(null); setScResult(null);
      const base = await readBaseStructure(file);
      setBaseStructure(base);
      setSourceLabel(file.name);
      await runAnalysis([file], 'rmc6f', base, 1, symTol);
    } catch (err) {
      console.error(err);
      setLoadErr(err.message || 'Failed to parse .rmc6f file.');
    }
  };

  const handleSelectFolder = async () => {
    if (typeof window.showDirectoryPicker !== 'function') {
      setLoadErr('Folder picker requires Chrome or Edge. You can still drop a single .rmc6f file in any browser.');
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
      setBaseStructure(base);
      setSourceLabel(`${dh.name}/ (${files.length} configs)`);
      await runAnalysis(files, family, base, Math.min(10, files.length), symTol);
    } catch (err) {
      if (err?.name !== 'AbortError') setLoadErr(err.message || String(err));
    }
  };

  const selectSubgroup = (sg) => {
    if (!scResult?.ctx) return;
    try {
      const aligned = alignSubgroupFromContext(scResult.ctx, sg);
      setScResult(prev => (prev ? { ...prev, aligned } : prev));
    } catch (err) {
      setScErr(err.message || String(err));
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
        label: `Local · ${sg}`,
        spaceGroup: sg,
        kind: 'local-subgroup',
        // Tight symprec preserves the local symmetry-breaking distortion instead of snapping back to the parent group
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
    const f = e.dataTransfer?.files?.[0];
    if (f) handleSingleFile(f);
  };

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
        <div style={{ width: 'min(660px, 100%)', textAlign: 'left', lineHeight: 1.55 }}>
          <h1 style={{ fontSize: 36, fontWeight: 700, letterSpacing: '-0.035em', lineHeight: 1.08, margin: '0 0 14px', color: 'var(--ink)' }}>
            Modern phonons from RMC — start with one supercell.
          </h1>
          <p style={{ margin: '0 0 22px', color: 'var(--secondary)', fontSize: 15 }}>
            Drop a <b style={{ color: 'var(--ink)' }}>single <code style={{ fontFamily: 'var(--font-mono)', fontSize: 13.5 }}>.rmc6f</code> supercell</b> to detect hidden local symmetry breaking and twin domains, compute MLIP phonons for both parent and local structures, and validate directly against experimental INS.
          </p>

          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', marginBottom: 24 }}>
            <label className="m-button primary lg">
              Open .rmc6f supercell…
              <input type="file" accept=".rmc6f" onChange={e => handleSingleFile(e.target.files?.[0])} style={{ display: 'none' }} />
            </label>
            {typeof window !== 'undefined' && typeof window.showDirectoryPicker === 'function' && (
              <button type="button" onClick={handleSelectFolder} className="m-btn lg">
                Open folder…
              </button>
            )}
            <label className="m-button subtle lg">
              Load saved _dm.json / band.yaml…
              <input type="file" accept=".json,.yaml,.yml" onChange={e => handleSingleFile(e.target.files?.[0])} style={{ display: 'none' }} />
            </label>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 12, paddingTop: 18, borderTop: '1px solid var(--line)', fontSize: 12.5, color: 'var(--secondary)' }}>
            <div><b style={{ color: 'var(--ink)' }}>1 · Symmetry</b><br />Find local subgroups &amp; domains from 1 supercell.</div>
            <div><b style={{ color: 'var(--ink)' }}>2 · MLIP Phonons</b><br />Compare Parent vs. Local bands in seconds.</div>
            <div><b style={{ color: 'var(--ink)' }}>3 · INS Compare</b><br />Rank models against 1D GDOS &amp; 2D S(|Q|,E).</div>
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
  const maxDeltaF = Math.max(1, ...sortedSubgroups.map(s => s.breakingFStat || 0));
  const maxSnr = Math.max(1, ...(scResult?.spectrum?.topK || []).map(k => k.snr || 0));

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
      <div
        className="m-frame"
        style={{
          flex: 'none', padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: 10,
          background: 'var(--surface)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 14, flexWrap: 'wrap' }}>
          {/* File + Structure identity */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <b style={{ fontSize: 16 }}>{readout?.formula || 'Supercell'}</b>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--secondary)' }}>
                  {sourceLabel}
                </span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11.5, color: 'var(--faint)' }}>
                  {readout?.supercell} ({readout?.totalAtoms?.toLocaleString()} atoms)
                </span>
              </div>
            </div>
            <label className="m-button subtle sm">
              Change file…
              <input type="file" accept=".rmc6f" onChange={e => handleSingleFile(e.target.files?.[0])} style={{ display: 'none' }} />
            </label>
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
                    ΔF={activeSubgroup.breakingFStat.toFixed(1)}
                  </span>
                )}
              </label>
            )}

            <button
              type="button"
              className="m-btn sm subtle"
              onClick={() => setShowMlipOpts(v => !v)}
              title="MLIP model, supercell, and temperature settings"
            >
              {mlipModel.replace('mace-mp-', 'MACE-')} · {mlipSupercell.replace(/\s+/g, '×')} · {mlipTemp}K ▾
            </button>

            {mlipServer?.connected ? (
              <button
                type="button"
                className="m-btn primary"
                disabled={selectedCount === 0 || anyRunning || scBusy}
                onClick={handleComputePhonons}
              >
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

        {/* Live Job / Results Strip (only visible once you've launched or loaded candidates) */}
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
      <div style={{ flex: 1, minHeight: 0, display: 'grid', gridTemplateColumns: 'minmax(300px, 42%) minmax(360px, 1fr)', gap: 14 }}>
        {/* ── LEFT COLUMN: 3D Crystal OR 2D Domain Map ── */}
        <div className="m-frame" style={{ display: 'flex', flexDirection: 'column', minHeight: 0, overflow: 'hidden' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px', borderBottom: '1px solid var(--line)', gap: 8, flexWrap: 'wrap' }}>
            <div className="m-tools" role="tablist">
              <button type="button" className={visualView === 'crystal' ? 'on' : 'dim'} onClick={() => setVisualView('crystal')}>
                3D Crystal
              </button>
              <button type="button" className={visualView === 'domains' ? 'on' : 'dim'} onClick={() => setVisualView('domains')}>
                2D Domain Map
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
                  RMS shift {scResult.aligned.symRms.toFixed(3)} Å
                </span>
              )}
            </div>
          </div>

          {/* 2D Domain Map View */}
          {visualView === 'domains' && (
            <div style={{ flex: 1, overflow: 'auto', padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 12 }}>
              {activeSubgroup?.domainGrid ? (
                <>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10, flexWrap: 'wrap', fontSize: 12 }}>
                    <span>
                      <b>{activeSubgroup.spaceGroup}</b> · {activeSubgroup.nVariants} twin orientations
                    </span>
                    <label style={{ display: 'inline-flex', alignItems: 'center', gap: 8, fontFamily: 'var(--font-mono)', color: 'var(--secondary)' }}>
                      <span>z = <b style={{ color: 'var(--ink)' }}>{sliceZ}</b>/{scResult.dim[2] - 1}</span>
                      <input type="range" min={0} max={Math.max(0, scResult.dim[2] - 1)} value={Math.min(sliceZ, scResult.dim[2] - 1)} onChange={e => setSliceZ(parseInt(e.target.value, 10))} className="m-range" style={{ width: 100 }} />
                    </label>
                  </div>

                  <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start', flexWrap: 'wrap' }}>
                    <div
                      style={{
                        display: 'grid',
                        gridTemplateColumns: `repeat(${scResult.dim[1]}, 26px)`,
                        gap: 3, padding: 8, borderRadius: 6, background: 'var(--muted)', border: '1px solid var(--line)',
                      }}
                    >
                      {Array.from({ length: scResult.dim[0] * scResult.dim[1] }, (_, idx) => {
                        const x = Math.floor(idx / scResult.dim[1]);
                        const y = idx % scResult.dim[1];
                        const z = Math.min(sliceZ, scResult.dim[2] - 1);
                        const c = (x * scResult.dim[1] + y) * scResult.dim[2] + z;
                        const dom = activeSubgroup.domainGrid[c] ?? 0;
                        return (
                          <div
                            key={idx}
                            title={`Cell (${x},${y},${z}) → Domain ${dom + 1}`}
                            style={{
                              width: 26, height: 26, borderRadius: 4,
                              background: DOMAIN_PALETTE[dom % DOMAIN_PALETTE.length],
                              color: '#fff', fontSize: 10.5, fontFamily: 'var(--font-mono)', fontWeight: 700,
                              display: 'flex', alignItems: 'center', justifyContent: 'center',
                            }}
                          >
                            {dom + 1}
                          </div>
                        );
                      })}
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: 5, fontFamily: 'var(--font-mono)', fontSize: 11.5 }}>
                      {activeSubgroup.domainFractions.map((f, i) => (
                        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <span style={{ width: 10, height: 10, borderRadius: 3, background: DOMAIN_PALETTE[i % DOMAIN_PALETTE.length] }} />
                          <span>D{i + 1}: {(100 * f).toFixed(1)}%</span>
                        </div>
                      ))}
                      {activeSubgroup.neighborMatchRate !== null && (
                        <div style={{ marginTop: 6, color: 'var(--secondary)' }}>
                          NN match: <b style={{ color: 'var(--ink)' }}>{(100 * activeSubgroup.neighborMatchRate).toFixed(1)}%</b>
                        </div>
                      )}
                    </div>
                  </div>
                </>
              ) : (
                <div style={{ color: 'var(--faint)', fontSize: 13 }}>
                  Select a multi-variant local subgroup on the right to view its 2D domain orientation map.
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── RIGHT COLUMN: Flat 3-Tab Symmetry Analysis Card ── */}
        <div className="m-frame" style={{ display: 'flex', flexDirection: 'column', minHeight: 0, overflow: 'hidden' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px', borderBottom: '1px solid var(--line)', gap: 8, flexWrap: 'wrap' }}>
            <div className="m-tools" role="tablist">
              <button type="button" className={analysisTab === 'subgroups' ? 'on' : 'dim'} onClick={() => setAnalysisTab('subgroups')}>
                Local Subgroups (ΔF)
              </button>
              <button type="button" className={analysisTab === 'kspace' ? 'on' : 'dim'} onClick={() => setAnalysisTab('kspace')}>
                k-Space Modes
              </button>
              <button type="button" className={analysisTab === 'average' ? 'on' : 'dim'} onClick={() => setAnalysisTab('average')}>
                Average Ladder
              </button>
            </div>
            <span style={{ fontSize: 11.5, color: 'var(--faint)' }}>
              {analysisTab === 'subgroups' ? 'Click a row to choose Local candidate' : ''}
            </span>
          </div>

          <div style={{ flex: 1, overflow: 'auto', padding: '12px 16px' }}>
            {scBusy ? (
              <div style={{ padding: 24, color: 'var(--secondary)', fontSize: 13 }}>
                Analyzing supercell displacements…
              </div>
            ) : analysisTab === 'subgroups' ? (
              sortedSubgroups.length > 0 ? (
                <>
                  {!hasRealDisplacements && (
                    <div style={{ marginBottom: 10, padding: '8px 10px', borderRadius: 6, background: 'var(--muted)', fontSize: 12, color: 'var(--secondary)' }}>
                      Zero atomic displacements detected (this looks like an ideal/initial <code style={{ fontFamily: 'var(--font-mono)' }}>.rmc6f</code>). Load a converged RMC configuration to measure local symmetry breaking ΔF.
                    </div>
                  )}
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12.5 }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--line)', textAlign: 'left', color: 'var(--faint)', fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                      <th style={{ padding: '6px 8px' }}>Subgroup</th>
                      <th style={{ padding: '6px 8px' }}>Variants</th>
                      <th style={{ padding: '6px 8px' }}>ΔF (symmetry breaking)</th>
                      <th style={{ padding: '6px 8px', textAlign: 'right' }}>R²</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedSubgroups.slice(0, 12).map((sg, i) => {
                      const isSel = activeSubgroup?.spaceGroup === sg.spaceGroup;
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
                          <td style={{ padding: '8px 8px', fontWeight: isSel ? 700 : 500, fontFamily: 'var(--font-mono)' }}>
                            {isSel && <span style={{ color: 'var(--ok-ink)', marginRight: 6 }}>●</span>}
                            {sg.spaceGroup}
                          </td>
                          <td style={{ padding: '8px 8px', fontFamily: 'var(--font-mono)', color: 'var(--secondary)' }}>
                            {sg.nVariants}
                          </td>
                          <td style={{ padding: '8px 8px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                              <div style={{ flex: 1, height: 7, borderRadius: 4, background: 'var(--muted)', overflow: 'hidden', minWidth: 60 }}>
                                <div style={{ width: `${Math.min(100, (sg.breakingFStat / maxDeltaF) * 100)}%`, height: '100%', background: isSel ? 'var(--ok-ink)' : 'var(--fig-accent)' }} />
                              </div>
                              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11.5, width: 42, textAlign: 'right' }}>
                                {sg.breakingFStat.toFixed(2)}
                              </span>
                            </div>
                          </td>
                          <td style={{ padding: '8px 8px', textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--secondary)' }}>
                            {Number.isFinite(sg.bestDomainR2) ? `${(100 * sg.bestDomainR2).toFixed(1)}%` : '—'}
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
                  <div style={{ padding: '8px 12px', borderRadius: 6, background: 'var(--muted)', fontSize: 12, fontFamily: 'var(--font-mono)' }}>
                    Mode-filtered minimal supercell: <b>{scResult.minimal.miniDim.join('×')}</b> ({scResult.minimal.nActiveK} active coherent k-modes)
                  </div>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--line)', textAlign: 'left', color: 'var(--faint)', fontSize: 11, textTransform: 'uppercase' }}>
                        <th style={{ padding: '6px 8px' }}>k-vector (rlu)</th>
                        <th style={{ padding: '6px 8px' }}>SNR</th>
                        <th style={{ padding: '6px 8px', textAlign: 'right' }}>P(k) (Å²)</th>
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
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div style={{ color: 'var(--faint)', fontSize: 13 }}>No k-space spectrum available.</div>
              )
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
