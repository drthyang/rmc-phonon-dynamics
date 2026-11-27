import React, { useState, useMemo, useCallback, useRef, useEffect, useId } from 'react';
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
import { FALLBACK_MLIP_MODELS, shortModelName } from '../io/mlipClient';
import { downloadString } from '../io/writers';
import CrystalViewer from '../components/CrystalViewer';
import DomainCube3D from '../components/DomainCube3D';
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

  const [domainView, setDomainView] = useState('both');        // 'cube' | 'slice' | 'both'
  const [sliceMode3D, setSliceMode3D] = useState('all');       // 'all' | 'slice' | 'cutaway'
  const [soloDomain, setSoloDomain] = useState(null);          // null or domain index 0..V-1
  const [showCorrChart, setShowCorrChart] = useState(false);
  const [crystalMode, setCrystalMode] = useState('aligned');   // 'parent' | 'aligned'
  const [analysisTab, setAnalysisTab] = useState('subgroups'); // 'subgroups' | 'kspace' | 'convergence' | 'average'
  const [cellFilter, setCellFilter] = useState('all');         // 'all' | '1×1×1' | '1×1×1 (P)' | '1×1×2' | '2×2×1' | '2×2×2'
  const [sliceZ, setSliceZ] = useState(0);
  const [domainCfgIdx, setDomainCfgIdx] = useState(0);
  const cellToolbarRef = useRef(null);
  const subgroupTableId = useId();

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
  const [extraModels, setExtraModels] = useState([]);
  const [mlipSupercell, setMlipSupercell] = useState('2 2 2');
  const [mlipTemp, setMlipTemp] = useState(250);
  const [relaxMode, setRelaxMode] = useState('fix_symmetry'); // 'fix_symmetry' | 'frozen' | 'unconstrained'
  const [localSymprec, setLocalSymprec] = useState(0.002);
  const [showMlipOpts, setShowMlipOpts] = useState(false);

  const availableModels = mlipServer?.models?.length ? mlipServer.models : FALLBACK_MLIP_MODELS;
  const activeModelList = useMemo(
    () => [mlipModel, ...extraModels.filter(m => m !== mlipModel)],
    [mlipModel, extraModels]
  );
  const toggleExtraModel = (id) => {
    setExtraModels(prev => (prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]));
  };

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
      .filter(s => s.isZoneBoundary ? s.breakingDoF > 0 : (s.breakingDoF > 0 && s.dof < scResult.local.internalDoF))
      .sort((a, b) => {
        const sa = multi ? a.medianExcessDeltaF : a.excessDeltaF;
        const sb = multi ? b.medianExcessDeltaF : b.excessDeltaF;
        return (sb - sa) || (b.cvBreakingFStat - a.cvBreakingFStat);
      });
  }, [scResult]);

  const cellGroups = useMemo(() => {
    if (!sortedSubgroups.length) return [];
    const ORDER = ['1×1×1', '1×1×1 (P)', '1×1×2', '1×2×1', '2×1×1', '2×2×1', '2×1×2', '1×2×2', '2×2×2'];
    const byCell = new Map();
    for (const s of sortedSubgroups) {
      const key = s.cellLabel || '1×1×1';
      const cur = byCell.get(key);
      if (!cur) byCell.set(key, { key, star: s.zoneStar || 'Γ', count: 1 });
      else cur.count += 1;
    }
    return [...byCell.values()].sort((a, b) => {
      const ia = ORDER.indexOf(a.key), ib = ORDER.indexOf(b.key);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
    });
  }, [sortedSubgroups]);

  const filteredSubgroups = useMemo(
    () => (cellFilter === 'all' ? sortedSubgroups : sortedSubgroups.filter(s => (s.cellLabel || '1×1×1') === cellFilter)),
    [sortedSubgroups, cellFilter]
  );

  // Roving tabindex + ArrowRight/ArrowLeft/Home/End fallback for browsers without native HTML `focusgroup`
  useEffect(() => {
    const bar = cellToolbarRef.current;
    if (!bar || (typeof HTMLElement !== 'undefined' && 'focusGroup' in HTMLElement.prototype)) return undefined;
    const getItems = () => Array.from(bar.querySelectorAll('button'));
    const initItems = getItems();
    const pressedIdx = Math.max(0, initItems.findIndex(b => b.getAttribute('aria-pressed') === 'true'));
    initItems.forEach((b, i) => { b.tabIndex = i === pressedIdx ? 0 : -1; });

    const onFocusIn = ({ target }) => {
      const items = getItems();
      if (items.includes(target)) items.forEach(b => { b.tabIndex = b === target ? 0 : -1; });
    };
    const onKeyDown = (e) => {
      const items = getItems();
      const cur = items.indexOf(document.activeElement);
      if (cur < 0 || !items.length) return;
      const rtl = getComputedStyle(bar).direction === 'rtl';
      let next = -1;
      if (e.key === 'ArrowRight') next = (cur + (rtl ? -1 : 1) + items.length) % items.length;
      else if (e.key === 'ArrowLeft') next = (cur + (rtl ? 1 : -1) + items.length) % items.length;
      else if (e.key === 'Home') next = 0;
      else if (e.key === 'End') next = items.length - 1;
      if (next >= 0) {
        e.preventDefault();
        items.forEach((b, i) => { b.tabIndex = i === next ? 0 : -1; });
        items[next].focus();
      }
    };
    bar.addEventListener('focusin', onFocusIn);
    bar.addEventListener('keydown', onKeyDown);
    return () => {
      bar.removeEventListener('focusin', onFocusIn);
      bar.removeEventListener('keydown', onKeyDown);
    };
  }, [analysisTab, cellGroups.length, cellFilter]);

  const activeSubgroup = useMemo(() => {
    if (!sortedSubgroups.length) return null;
    const targetId = scResult?.aligned?.id;
    if (targetId) {
      const byId = sortedSubgroups.find(s => s.id === targetId);
      if (byId) return byId;
    }
    const targetSg = scResult?.aligned?.targetSpaceGroup;
    return sortedSubgroups.find(s => s.spaceGroup === targetSg) || sortedSubgroups[0];
  }, [sortedSubgroups, scResult]);

  const previewStruct = useMemo(() => {
    if (!baseStructure?.basis) return null;
    const useAligned = crystalMode === 'aligned' && scResult?.aligned?.symmetrizedBasis;
    const b = useAligned ? scResult.aligned.symmetrizedBasis : (activeParentBasis || baseStructure.basis);
    if (useAligned && scResult?.aligned?.A) {
      const Ablk = scResult.aligned.A;
      const dynAtomDic = {};
      b.forEach((site, idx) => {
        const rn = site.rn ?? (idx + 1);
        if (!dynAtomDic[site.el]) dynAtomDic[site.el] = [];
        dynAtomDic[site.el].push(rn);
      });
      return {
        v1: Ablk[0], v2: Ablk[1], v3: Ablk[2], dim: [1, 1, 1],
        atomDic: dynAtomDic,
        uniqueRN: b.map((x, i) => x.rn ?? (i + 1)),
        atomType: b.map((x, i) => x.rn ?? (i + 1)),
        hsym_xyz: Float64Array.from(b.flatMap(x => x.frac)),
        cellIdx: new Float64Array(b.length * 3),
      };
    }
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
      setLoadErr(null); setScErr(null); setScResult(null); setCellFilter('all');
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
      setLoadErr(null); setScErr(null); setScResult(null); setCellFilter('all');
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
      setSoloDomain(null);
    }
  };

  const handleSelectCellFilter = (key) => {
    setCellFilter(key);
    if (key !== 'all') {
      const currentCell = activeSubgroup?.cellLabel || '1×1×1';
      if (currentCell !== key) {
        const topInCell = sortedSubgroups.find(s => (s.cellLabel || '1×1×1') === key);
        if (topInCell) selectSubgroup(topInCell);
      }
    }
  };

  const parseSupercellVec = () => {
    const raw = mlipSupercell.trim().split(/\s+/).map(n => Math.max(1, parseInt(n, 10) || 1));
    return [raw[0] || 2, raw[1] || raw[0] || 2, raw[2] || raw[0] || 2];
  };

  const buildSelectedSpecs = () => {
    const specs = [];
    const stem = (sourceLabel || 'rmc').replace(/\.(rmc6f|cif)$/i, '').replace(/[^A-Za-z0-9_-]+/g, '_');
    const baseSc = parseSupercellVec();
    if (includeParent && Aconv && activeParentBasis) {
      const sg = symInfo?.spaceGroup || 'P1';
      specs.push({
        label: `Parent · ${sg}`,
        spaceGroup: sg,
        kind: 'parent',
        symprec: 'auto',
        supercell: baseSc,
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
      const cm = scResult.aligned.cellMult || [1, 1, 1];
      const cellLbl = scResult.aligned.cellLabel || cm.join('×');
      const isMultiCell = cm[0] * cm[1] * cm[2] > 1;
      const scaledSc = [
        Math.max(1, Math.round(baseSc[0] / cm[0])),
        Math.max(1, Math.round(baseSc[1] / cm[1])),
        Math.max(1, Math.round(baseSc[2] / cm[2])),
      ];
      const cellTag = cellLbl !== '1×1×1' ? `${cellLbl} · ` : '';
      const fileCellTag = isMultiCell ? `${cm.join('x')}_` : '';
      specs.push({
        label: `Local · ${cellTag}${sg}${scResult.nConfigs > 1 ? ` (${scResult.nConfigs} cfg)` : ''}`,
        spaceGroup: sg,
        cellMult: cm,
        cellLabel: cellLbl,
        kind: 'local-subgroup',
        symprec: localSymprec,
        supercell: scaledSc,
        filename: `${stem}_local_${fileCellTag}${sg.replace(/[^A-Za-z0-9_-]+/g, '')}.cif`,
        cifText: buildAlignedCifText(scResult.aligned, { projected: true, title: `${stem}_${fileCellTag}${sg}` }),
      });
    }
    return specs;
  };

  const mlipParams = () => ({
    model: mlipModel,
    models: activeModelList,
    supercell: parseSupercellVec(),
    temperature: mlipTemp,
    relax: relaxMode !== 'frozen',
    relaxMode,
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
    const cmds = [];
    for (const modId of activeModelList) {
      for (const s of specs) {
        const scStr = (s.supercell || parseSupercellVec()).join(' ');
        const tag = activeModelList.length > 1 ? `_${modId}` : '';
        cmds.push(
          `uv run rmcph-mlip run ${s.filename} --model ${modId} --supercell ${scStr} -T ${mlipTemp} --symprec ${s.symprec ?? 'auto'} --relax-mode ${relaxMode} --out results/${s.filename.replace(/\.cif$/i, '')}${tag}`
        );
      }
    }
    navigator.clipboard?.writeText(cmds.join('\n'));
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
        className={`m-landing ${dragOver ? 'is-drag' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
      >
        <div className="m-landing-card">
          <h1 className="m-landing-title">
            Modern phonons from RMC — start with supercell symmetry.
          </h1>
          <p className="m-landing-lead">
            Drop <b style={{ color: 'var(--ink)' }}>1 or many <code style={{ fontFamily: 'var(--font-mono)', fontSize: '0.92em' }}>.rmc6f</code> supercells</b> (or a full ensemble folder) to stream null-calibrated local symmetry breaking, spatial cross-validated domains, and correlation length ξ — then run MLIP phonons and validate directly against INS.
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

          <div className="sym-landing-steps">
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
  const localCellLabel = scResult?.aligned?.cellLabel || activeSubgroup?.cellLabel || '1×1×1';
  const localCellPrefix = localCellLabel !== '1×1×1' ? `${localCellLabel} · ` : '';
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
      className="sym-workspace"
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
    >
      {/* ════════ TOP GUIDED ACTION STRIP ════════ */}
      <div className="m-frame sym-strip">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 14, flexWrap: 'wrap' }}>
          {/* File + Ensemble identity */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', minWidth: 0 }}>
            <b style={{ fontSize: 'var(--fz-title)' }}>{readout?.formula || 'Supercell'}</b>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--fz-small)', color: 'var(--secondary)' }}>
              {sourceLabel}
            </span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--fz-small)', color: 'var(--faint)' }}>
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
                <span>Local <b>{localCellPrefix}{localSg}</b></span>
                {activeSubgroup && (
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--note-ink)' }}>
                    {activeSubgroup.zScore !== null
                      ? `${activeSubgroup.zScore.toFixed(1)}σ`
                      : `+ΔF ${activeSubgroup.excessDeltaF.toFixed(2)}`}
                  </span>
                )}
              </label>
            )}

            <button
              type="button"
              className="m-btn sm subtle"
              aria-expanded={showMlipOpts}
              onClick={() => setShowMlipOpts(v => !v)}
            >
              {shortModelName(mlipModel)}{activeModelList.length > 1 ? `+${activeModelList.length - 1}` : ''} · {relaxMode === 'fix_symmetry' ? 'FixSym' : relaxMode === 'frozen' ? 'Frozen' : 'Free'} · {mlipSupercell.replace(/\s+/g, '×')} · {mlipTemp}K ▾
            </button>

            {mlipServer?.connected ? (
              <button type="button" className="m-btn primary" disabled={selectedCount === 0 || anyRunning || scBusy} onClick={handleComputePhonons}>
                {anyRunning ? 'Computing phonons…' : `▶ Compute Phonons (${selectedCount * activeModelList.length})`}
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

        {/* Collapsible MLIP & FixSymmetry Settings Row */}
        {showMlipOpts && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, paddingTop: 8, borderTop: '1px solid var(--line)', fontSize: 12 }}>
            <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', alignItems: 'center' }}>
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                <label htmlFor={`${subgroupTableId}-mlip-model`} style={{ color: 'var(--secondary)' }}>Primary uMLIP</label>
                <select
                  id={`${subgroupTableId}-mlip-model`}
                  value={mlipModel}
                  onChange={e => setMlipModel(e.target.value)}
                  className="m-input"
                  style={{ height: 26, padding: '0 8px' }}
                >
                  {availableModels.map(m => (
                    <option key={m.id} value={m.id} disabled={mlipServer?.connected && !m.installed}>
                      {m.label}{m.cached_locally ? ' · cached' : (!m.installed ? ` · (${m.install_hint})` : '')}
                    </option>
                  ))}
                </select>
              </span>

              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                <label htmlFor={`${subgroupTableId}-relax-mode`} style={{ color: 'var(--secondary)' }}>Relaxation</label>
                <select
                  id={`${subgroupTableId}-relax-mode`}
                  value={relaxMode}
                  onChange={e => setRelaxMode(e.target.value)}
                  className="m-input"
                  style={{ height: 26, padding: '0 8px' }}
                >
                  <option value="fix_symmetry">FixSymmetry (lock space group)</option>
                  <option value="frozen">Frozen RMC geometry (0 steps)</option>
                  <option value="unconstrained">Unconstrained BFGS (allow P1)</option>
                </select>
              </span>

              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                <label htmlFor={`${subgroupTableId}-sc`} style={{ color: 'var(--secondary)' }}>Phonopy supercell</label>
                <input id={`${subgroupTableId}-sc`} value={mlipSupercell} onChange={e => setMlipSupercell(e.target.value)} className="m-input mono" style={{ width: 68, height: 26, textAlign: 'center' }} />
              </span>

              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                <label htmlFor={`${subgroupTableId}-temp`} style={{ color: 'var(--secondary)' }}>T (K)</label>
                <input id={`${subgroupTableId}-temp`} type="number" value={mlipTemp} onChange={e => setMlipTemp(Number(e.target.value) || 5)} className="m-input mono" style={{ width: 62, height: 26, textAlign: 'center' }} />
              </span>

              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                <label htmlFor={`${subgroupTableId}-symprec`} style={{ color: 'var(--secondary)' }}>Local symprec (Å)</label>
                <input id={`${subgroupTableId}-symprec`} type="number" step="0.001" min="0.0001" max="0.2" value={localSymprec} onChange={e => setLocalSymprec(Math.max(0.0001, parseFloat(e.target.value) || 0.002))} className="m-input mono" style={{ width: 68, height: 26, textAlign: 'center' }} />
              </span>

              <button type="button" className="m-btn sm" onClick={handleCopyCli} style={{ marginLeft: 'auto' }}>
                {copiedCli ? 'Copied CLI ✓' : 'Copy CLI command'}
              </button>
            </div>

            {/* Multi-uMLIP Consensus Strip */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
              <span style={{ color: 'var(--faint)', fontSize: 11.5 }}>Multi-uMLIP consensus (+compare):</span>
              {availableModels
                .filter(m => m.id !== mlipModel && (!mlipServer?.connected || m.installed))
                .slice(0, 5)
                .map(m => {
                  const active = extraModels.includes(m.id);
                  return (
                    <button
                      key={m.id}
                      type="button"
                      aria-pressed={active}
                      className={`m-btn sm ${active ? 'active' : 'subtle'}`}
                      style={{ height: 22, fontSize: 11, padding: '0 8px' }}
                      onClick={() => toggleExtraModel(m.id)}
                    >
                      {active ? '✓ ' : '+ '}{m.short} ({m.functional})
                    </button>
                  );
                })}
            </div>
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

      {/* ════════ MULTI-LAYOUT WORKSPACE: Stack | Deck | Split | Panoramic ════════ */}
      <div className="sym-grid">
        {/* ── CARD 1: 3D Crystal Structure ── */}
        <section className="m-frame sym-crystal" aria-label="3D Crystal Structure">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '7px 12px', borderBottom: '1px solid var(--line)', gap: 8, flexWrap: 'wrap' }}>
            <span style={{ fontWeight: 600, fontSize: 'var(--fz-control)' }}>3D Crystal Structure</span>
            <div style={{ display: 'inline-flex', gap: 4 }}>
              <button type="button" className={`m-btn sm ${crystalMode === 'parent' ? 'active' : ''}`} onClick={() => setCrystalMode('parent')}>
                Parent ({parentSg})
              </button>
              <button type="button" className={`m-btn sm ${crystalMode === 'aligned' ? 'active' : ''}`} disabled={!scResult?.aligned} onClick={() => setCrystalMode('aligned')}>
                Local ({localSg ? `${localCellPrefix}${localSg}` : '—'})
              </button>
            </div>
          </div>
          <div style={{ flex: 1, minHeight: 170, background: 'var(--surface)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {previewStruct && (
              <CrystalViewer baseStructure={previewStruct} eigenvector={null} isPlaying={false} supercell={cell111} showCell showBonds />
            )}
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, flexWrap: 'wrap', padding: '6px 12px', borderTop: '1px solid var(--line)', fontSize: 'var(--fz-small)', color: 'var(--secondary)' }}>
            <div style={{ display: 'inline-flex', gap: 10, flexWrap: 'wrap' }}>
              {(crystalMode === 'aligned' && scResult?.aligned?.elementStats
                ? scResult.aligned.elementStats.map(es => [es.el, es.count])
                : readout?.counts || []
              ).map(([el, count]) => (
                <span key={el} style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                  <span style={{ width: 8, height: 8, borderRadius: '50%', background: DEFAULT_COLORS[el] || '#ccc', boxShadow: 'inset 0 0 0 1px rgba(0,0,0,0.2)' }} />
                  <b style={{ color: 'var(--ink)' }}>{el}</b>×{count}
                </span>
              ))}
            </div>
            {crystalMode === 'aligned' && scResult?.aligned && (
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--fz-small)', color: 'var(--note-ink)' }}>
                RMS {scResult.aligned.symRms.toFixed(4)}
                {scResult.aligned.rmsSE > 0 ? ` ± ${scResult.aligned.rmsSE.toFixed(4)}` : ''} Å
              </span>
            )}
          </div>
        </section>

        {/* ── CARD 2: 3D Domain Cube + 2D Domain Map ── */}
        <section className="m-frame sym-domains" aria-label="Supercell Domains">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '7px 12px', borderBottom: '1px solid var(--line)', gap: 8, flexWrap: 'wrap' }}>
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8, minWidth: 0, flexWrap: 'wrap' }}>
              <span style={{ fontWeight: 600, fontSize: 'var(--fz-control)' }}>
                Domains {activeSubgroup ? `· ${activeSubgroup.cellLabel && activeSubgroup.cellLabel !== '1×1×1' ? `${activeSubgroup.cellLabel} ` : ''}${activeSubgroup.spaceGroup}` : ''}
              </span>
              <div className="m-tools" role="tablist" aria-label="Domain view" style={{ gap: 6 }}>
                <button type="button" role="tab" aria-selected={domainView === 'both'} className={domainView === 'both' ? 'on' : 'dim'} onClick={() => setDomainView('both')}>
                  3D + 2D
                </button>
                <button type="button" role="tab" aria-selected={domainView === 'cube'} className={domainView === 'cube' ? 'on' : 'dim'} onClick={() => setDomainView('cube')}>
                  3D Cube
                </button>
                <button type="button" role="tab" aria-selected={domainView === 'slice'} className={domainView === 'slice' ? 'on' : 'dim'} onClick={() => setDomainView('slice')}>
                  2D Slice
                </button>
              </div>
            </div>

            <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
              {activeSubgroup?.savedGrids?.length > 1 && (
                <select
                  value={domainCfgIdx}
                  onChange={e => setDomainCfgIdx(parseInt(e.target.value, 10))}
                  className="m-input sm"
                  aria-label="Domain configuration"
                >
                  {activeSubgroup.savedGrids.map((sg, i) => (
                    <option key={i} value={i}>{sg.configName}</option>
                  ))}
                </select>
              )}
              {radialSeries.length > 0 && (
                <button
                  type="button"
                  className={`m-btn sm ${showCorrChart ? 'active' : 'subtle'}`}
                  onClick={() => setShowCorrChart(v => !v)}
                >
                  C(r)
                </button>
              )}
            </div>
          </div>

          {activeGrid && activeSubgroup ? (
            <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', overflow: 'auto', overscrollBehavior: 'contain', padding: '8px 12px', gap: 8 }}>
              {/* Compact controls: z-layer + 3D slice mode + physics metrics */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, flexWrap: 'wrap', fontSize: 'var(--fz-small)', fontFamily: 'var(--font-mono)' }}>
                <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', color: 'var(--secondary)' }}>
                  <span>z=<b style={{ color: 'var(--ink)' }}>{sliceZ}</b>/{scResult.dim[2] - 1}</span>
                  <input
                    type="range"
                    min={0}
                    max={Math.max(0, scResult.dim[2] - 1)}
                    value={Math.min(sliceZ, scResult.dim[2] - 1)}
                    onChange={e => setSliceZ(parseInt(e.target.value, 10))}
                    className="m-range"
                    aria-label="z slice index"
                    style={{ width: 80 }}
                  />
                  {domainView !== 'slice' && (
                    <div style={{ display: 'inline-flex', gap: 3 }}>
                      {[['all', 'Full'], ['slice', 'Slice'], ['cutaway', 'Cut']].map(([k, lbl]) => (
                        <button
                          key={k}
                          type="button"
                          className={`m-btn sm ${sliceMode3D === k ? 'active' : ''}`}
                          onClick={() => setSliceMode3D(k)}
                        >
                          {lbl}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
                <div style={{ display: 'inline-flex', gap: 10, flexWrap: 'wrap', color: 'var(--secondary)' }}>
                  <span>ξ=<b style={{ color: 'var(--ink)' }}>{activeSubgroup.corrLengthAng.toFixed(1)}Å</b></span>
                  <span>clust=<b style={{ color: 'var(--ink)' }}>{activeSubgroup.meanClusterCells?.toFixed(1) ?? '—'}</b></span>
                  <span>NN=<b style={{ color: 'var(--ink)' }}>{(100 * (activeSubgroup.neighborMatchRate || 0)).toFixed(1)}%</b></span>
                </div>
              </div>

              {/* Main 3D Cube + 2D Slice visual row */}
              <div className={`sym-domain-split ${domainView === 'both' ? 'is-both' : ''}`}>
                {domainView !== 'slice' && (
                  <div className="sym-cube-box">
                    <DomainCube3D
                      grid={activeGrid}
                      dim={scResult.dim}
                      palette={DOMAIN_PALETTE}
                      activeDomain={soloDomain}
                      sliceZ={Math.min(sliceZ, scResult.dim[2] - 1)}
                      sliceMode={sliceMode3D}
                    />
                  </div>
                )}

                {domainView !== 'cube' && (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4, maxWidth: '100%', overflowX: 'auto' }}>
                    <div
                      className="sym-slice-grid"
                      style={{
                        gridTemplateColumns: `repeat(${scResult.dim[1]}, var(--cell-sz))`,
                      }}
                    >
                      {Array.from({ length: scResult.dim[0] * scResult.dim[1] }, (_, idx) => {
                        const x = Math.floor(idx / scResult.dim[1]);
                        const y = idx % scResult.dim[1];
                        const z = Math.min(sliceZ, scResult.dim[2] - 1);
                        const c = (x * scResult.dim[1] + y) * scResult.dim[2] + z;
                        const dom = activeGrid[c] ?? 0;
                        const dimmed = soloDomain !== null && dom !== soloDomain;
                        return (
                          <div
                            key={idx}
                            className="sym-slice-cell"
                            title={`Cell (${x},${y},${z}) → Domain D${dom + 1}`}
                            style={{
                              background: DOMAIN_PALETTE[dom % DOMAIN_PALETTE.length],
                              opacity: dimmed ? 0.16 : 1,
                            }}
                          >
                            {dom + 1}
                          </div>
                        );
                      })}
                    </div>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--fz-small)', color: 'var(--faint)' }}>
                      2D slice z={Math.min(sliceZ, scResult.dim[2] - 1)}
                    </span>
                  </div>
                )}
              </div>

              {/* Clickable domain legend chips (click to solo/isolate in 3D Cube & 2D Slice!) */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', fontFamily: 'var(--font-mono)', fontSize: 'var(--fz-small)' }}>
                {activeSubgroup.domainFractions.map((f, i) => {
                  const isSolo = soloDomain === i;
                  return (
                    <button
                      key={i}
                      type="button"
                      onClick={() => setSoloDomain(cur => (cur === i ? null : i))}
                      title="Click to isolate this twin domain in 3D & 2D"
                      style={{
                        display: 'inline-flex', alignItems: 'center', gap: 5,
                        padding: '3px 7px', borderRadius: 4, cursor: 'pointer',
                        border: `1px solid ${isSolo ? 'var(--ink)' : 'var(--line)'}`,
                        background: isSolo ? 'var(--muted)' : 'transparent',
                        opacity: soloDomain !== null && !isSolo ? 0.45 : 1,
                        fontFamily: 'var(--font-mono)', fontSize: 'var(--fz-small)', color: 'var(--ink)',
                      }}
                    >
                      <span style={{ width: 9, height: 9, borderRadius: 2, background: DOMAIN_PALETTE[i % DOMAIN_PALETTE.length] }} />
                      <span>
                        {activeSubgroup.domainKind === 'antiphase' ? 'A' : 'D'}{i + 1} {(100 * f).toFixed(1)}%
                        {activeSubgroup.domainFractionSE ? `±${(100 * activeSubgroup.domainFractionSE[i]).toFixed(1)}` : ''}
                      </span>
                    </button>
                  );
                })}
                {soloDomain !== null && (
                  <button type="button" className="m-btn sm subtle" onClick={() => setSoloDomain(null)}>
                    Show all
                  </button>
                )}
              </div>

              {/* Optional Radial Autocorrelation C_dom(r) chart */}
              {showCorrChart && radialSeries.length > 0 && (
                <div style={{ paddingTop: 6, borderTop: '1px solid var(--line)' }}>
                  <LineChart series={radialSeries} xLabel="Lag r (unit cells)" yLabel="C_dom(r)" height={135} />
                </div>
              )}
            </div>
          ) : (
            <div style={{ padding: 16, color: 'var(--faint)', fontSize: 'var(--fz-control)' }}>
              Select a multi-variant local subgroup in the Symmetry Analysis panel to visualize its 3D domain cube and 2D slice map.
            </div>
          )}
        </section>

        {/* ── CARD 3: Flat 4-Tab Ensemble Symmetry Analysis Card ── */}
        <section className="m-frame sym-analysis" aria-label="Symmetry Analysis">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 14px', borderBottom: '1px solid var(--line)', gap: 8, flexWrap: 'wrap' }}>
            <div className="m-tools" role="tablist" aria-label="Symmetry analysis views">
              <button type="button" role="tab" aria-selected={analysisTab === 'subgroups'} className={analysisTab === 'subgroups' ? 'on' : 'dim'} onClick={() => setAnalysisTab('subgroups')}>
                Local Subgroups
              </button>
              <button type="button" role="tab" aria-selected={analysisTab === 'kspace'} className={analysisTab === 'kspace' ? 'on' : 'dim'} onClick={() => setAnalysisTab('kspace')}>
                k-Space Modes
              </button>
              {isMulti && (
                <button type="button" role="tab" aria-selected={analysisTab === 'convergence'} className={analysisTab === 'convergence' ? 'on' : 'dim'} onClick={() => setAnalysisTab('convergence')}>
                  Convergence ({scResult.nConfigs})
                </button>
              )}
              <button type="button" role="tab" aria-selected={analysisTab === 'average'} className={analysisTab === 'average' ? 'on' : 'dim'} onClick={() => setAnalysisTab('average')}>
                Average Ladder
              </button>
            </div>
            <span style={{ fontSize: 'var(--fz-small)', color: 'var(--faint)' }}>
              {analysisTab === 'subgroups' ? 'Null-calibrated · click row to select' : ''}
            </span>
          </div>

          <div className="sym-analysis-body">
            {scBusy && !scResult ? (
              <div style={{ padding: 24, color: 'var(--secondary)', fontSize: 'var(--fz-control)' }}>
                Streaming supercell configurations… ({scProgress.text})
              </div>
            ) : analysisTab === 'subgroups' ? (
              sortedSubgroups.length > 0 ? (
                <>
                  {!hasRealDisplacements && (
                    <div style={{ marginBottom: 10, padding: '8px 10px', borderRadius: 6, background: 'var(--muted)', fontSize: 'var(--fz-small)', color: 'var(--secondary)' }}>
                      Zero atomic displacements detected (this looks like an ideal/initial <code style={{ fontFamily: 'var(--font-mono)' }}>.rmc6f</code>). Load converged RMC configurations to measure local symmetry breaking.
                    </div>
                  )}

                  {cellGroups.length > 1 && (
                    <div
                      ref={cellToolbarRef}
                      role="toolbar"
                      aria-label="Filter candidate subgroups by supercell"
                      aria-controls={subgroupTableId}
                      {...({ focusgroup: 'toolbar wrap' })}
                      className="sym-cell-toolbar"
                    >
                      <button
                        type="button"
                        className="sym-cell-pill"
                        aria-pressed={cellFilter === 'all'}
                        onClick={() => handleSelectCellFilter('all')}
                      >
                        <span>All</span>
                        <span className="sym-cell-pill-count">{sortedSubgroups.length}</span>
                      </button>
                      {cellGroups.map(cg => (
                        <button
                          key={cg.key}
                          type="button"
                          className="sym-cell-pill"
                          aria-pressed={cellFilter === cg.key}
                          onClick={() => handleSelectCellFilter(cg.key)}
                        >
                          <span>{cg.key === '1×1×1 (P)' ? '1×1×1·P' : cg.key} · {cg.star === 'X (prim)' ? 'X' : cg.star}</span>
                          <span className="sym-cell-pill-count">{cg.count}</span>
                        </button>
                      ))}
                    </div>
                  )}

                  <div id={subgroupTableId} className="m-table-wrap" tabIndex={0} role="region" aria-label="Candidate local subgroups">
                    <table>
                      <thead>
                        <tr style={{ borderBottom: '1px solid var(--line)', textAlign: 'left', color: 'var(--faint)', fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                          <th scope="col" style={{ padding: '6px 6px' }}>Subgroup</th>
                          <th scope="col" style={{ padding: '6px 6px' }} title="Orientational twin variants (and antiphase coset domains when single-variant)">Vars</th>
                          <th scope="col" style={{ padding: '6px 6px' }} title="Bias-corrected ΔF minus exact isotropic-noise null expectation (evaluated on pure non-Γ zone-boundary subspace for multi-cell channels)">Excess ΔF {isMulti ? '(±SE)' : ''}</th>
                          <th scope="col" style={{ padding: '6px 6px', textAlign: 'right' }} title="Spatial neighbor cross-validated ΔF (zero within-cell selection bias)">CV ΔF</th>
                          {isMulti && <th scope="col" style={{ padding: '6px 6px', textAlign: 'right' }} title="Fraction of RMC configs ranking this subgroup #1 across all channels">Win %</th>}
                          <th scope="col" style={{ padding: '6px 6px', textAlign: 'right' }}>ξ (Å)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredSubgroups.slice(0, 14).map((sg, i) => {
                          const isSel = activeSubgroup?.id ? activeSubgroup.id === sg.id : activeSubgroup?.spaceGroup === sg.spaceGroup;
                          const scoreVal = isMulti ? sg.medianExcessDeltaF : sg.excessDeltaF;
                          const cellLbl = sg.cellLabel || '1×1×1';
                          const isZb = Boolean(sg.isZoneBoundary || cellLbl !== '1×1×1');
                          const nDom = sg.nDomains || sg.nVariants;
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
                                <span>{sg.spaceGroup}</span>
                                <span className={`sym-cell-badge ${isZb ? 'is-zb' : ''}`} title={`Cell: ${cellLbl} (${sg.zoneStar || 'Γ'}-star)`}>
                                  {cellLbl === '1×1×1 (P)' ? '1×1×1·P' : cellLbl}
                                </span>
                              </td>
                              <td
                                style={{ padding: '8px 6px', fontFamily: 'var(--font-mono)', color: 'var(--secondary)', whiteSpace: 'nowrap' }}
                                title={sg.domainKind === 'antiphase' ? `1 orientational variant × ${nDom} antiphase domains` : `${sg.nVariants} orientational twin variants`}
                              >
                                {sg.nVariants === 1 && nDom > 1 ? `1×${nDom}a` : sg.nVariants}
                              </td>
                              <td style={{ padding: '8px 6px' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                  <div style={{ flex: 1, height: 6, borderRadius: 3, background: 'var(--muted)', overflow: 'hidden', minWidth: 48 }}>
                                    <div style={{ width: `${Math.max(0, Math.min(100, (scoreVal / maxExcessDF) * 100))}%`, height: '100%', background: isSel ? 'var(--ok-ink)' : 'var(--fig-accent)' }} />
                                  </div>
                                  <span style={{ fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap' }}>
                                    {scoreVal >= 0 ? '+' : ''}{scoreVal.toFixed(2)}
                                    {sg.excessDeltaFSE !== null && (
                                      <span style={{ color: 'var(--faint)', fontSize: '0.9em' }}> ±{sg.excessDeltaFSE.toFixed(2)}</span>
                                    )}
                                  </span>
                                </div>
                              </td>
                              <td style={{ padding: '8px 6px', textAlign: 'right', fontFamily: 'var(--font-mono)', color: sg.cvBreakingFStat > 1.15 ? 'var(--ok-ink)' : 'var(--secondary)', fontWeight: sg.cvBreakingFStat > 1.15 ? 600 : 400, whiteSpace: 'nowrap' }}>
                                {Number.isFinite(sg.cvBreakingFStat) ? sg.cvBreakingFStat.toFixed(2) : '—'}
                              </td>
                              {isMulti && (
                                <td style={{ padding: '8px 6px', textAlign: 'right', fontFamily: 'var(--font-mono)', fontWeight: sg.winRate > 0.5 ? 700 : 400, whiteSpace: 'nowrap' }}>
                                  {(100 * sg.winRate).toFixed(0)}%
                                </td>
                              )}
                              <td style={{ padding: '8px 6px', textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--secondary)', whiteSpace: 'nowrap' }}>
                                {nDom > 1 && sg.corrLengthAng > 0.1 ? `${sg.corrLengthAng.toFixed(1)}` : '—'}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </>
              ) : (
                <div style={{ color: 'var(--faint)', fontSize: 'var(--fz-control)' }}>No symmetry-breaking local subgroups detected above the parent cell.</div>
              )
            ) : analysisTab === 'kspace' ? (
              scResult?.spectrum?.topK?.length ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  {/* High-symmetry zone-boundary summary (Γ, X, M, R) */}
                  {scResult.spectrum.highSymStars && (
                    <div className="sym-kstars">
                      <div><span style={{ color: 'var(--secondary)' }}>Γ (0,0,0)</span><br /><b>{scResult.spectrum.highSymStars.gamma?.toFixed(2)}×</b></div>
                      <div><span style={{ color: 'var(--secondary)' }}>X (½,0,0)</span><br /><b>{scResult.spectrum.highSymStars.X?.toFixed(2) ?? '—'}×</b></div>
                      <div><span style={{ color: 'var(--secondary)' }}>M (½,½,0)</span><br /><b>{scResult.spectrum.highSymStars.M?.toFixed(2) ?? '—'}×</b></div>
                      <div><span style={{ color: 'var(--secondary)' }}>R (½,½,½)</span><br /><b>{scResult.spectrum.highSymStars.R?.toFixed(2) ?? '—'}×</b></div>
                    </div>
                  )}
                  <div className="m-table-wrap" tabIndex={0} role="region" aria-label="Supercell k-space displacement power">
                    <table>
                      <thead>
                        <tr style={{ borderBottom: '1px solid var(--line)', textAlign: 'left', color: 'var(--faint)', fontSize: 10.5, textTransform: 'uppercase' }}>
                          <th style={{ padding: '6px 8px' }}>k-vector (rlu)</th>
                          <th style={{ padding: '6px 8px' }}>SNR</th>
                          <th style={{ padding: '6px 8px', textAlign: 'right' }}>P(k) (Å²)</th>
                          {isMulti && <th style={{ padding: '6px 8px', textAlign: 'right' }}>Coherent %</th>}
                        </tr>
                      </thead>
                      <tbody>
                        {scResult.spectrum.topK.slice(0, 12).map((k, i) => (
                          <tr key={i} style={{ borderBottom: '1px solid var(--line)' }}>
                            <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap' }}>
                              ({k.kFrac.map(v => v.toFixed(3)).join(', ')})
                            </td>
                            <td style={{ padding: '6px 8px', minWidth: 110 }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                <div style={{ flex: 1, height: 6, borderRadius: 3, background: 'var(--muted)', overflow: 'hidden' }}>
                                  <div style={{ width: `${Math.min(100, (k.snr / maxSnr) * 100)}%`, height: '100%', background: 'var(--fig-accent)' }} />
                                </div>
                                <span style={{ fontFamily: 'var(--font-mono)', width: 42, textAlign: 'right' }}>{k.snr.toFixed(1)}×</span>
                              </div>
                            </td>
                            <td style={{ padding: '6px 8px', textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--secondary)', whiteSpace: 'nowrap' }}>
                              {Number.isFinite(k.power) ? k.power.toFixed(3) : '—'}
                            </td>
                            {isMulti && (
                              <td style={{ padding: '6px 8px', textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--secondary)', whiteSpace: 'nowrap' }}>
                                {k.power > 1e-12 ? `${(100 * k.coherentPower / k.power).toFixed(0)}%` : '0%'}
                              </td>
                            )}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : (
                <div style={{ color: 'var(--faint)', fontSize: 'var(--fz-control)' }}>No k-space spectrum available.</div>
              )
            ) : analysisTab === 'convergence' ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                <div style={{ fontSize: 'var(--fz-control)', color: 'var(--secondary)' }}>
                  Tracking how the leading local subgroup&apos;s bias-corrected <b>Excess ΔF</b> and bias-free <b>Spatial CV ΔF</b> stabilize as more RMC configurations are streamed.
                </div>
                {convergenceSeries.length > 0 && (
                  <LineChart series={convergenceSeries} xLabel="Number of RMC configs (M)" yLabel="Symmetry-breaking ΔF" height={240} />
                )}
              </div>
            ) : (
              /* Average Symmetry Ladder */
              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 'var(--fz-control)' }}>
                  <span style={{ color: 'var(--secondary)' }}>Tolerance</span>
                  <input type="range" min={0.02} max={1.0} step={0.01} value={symTol} onChange={e => setSymTol(parseFloat(e.target.value))} className="m-range" style={{ flex: 1 }} />
                  <b style={{ fontFamily: 'var(--font-mono)', width: 58, textAlign: 'right' }}>{symTol.toFixed(2)} Å</b>
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
                      {b.spaceGroup} <span style={{ color: 'var(--faint)', fontSize: '0.88em' }}>[{b.from.toFixed(2)}–{b.to.toFixed(2)}Å]</span>
                    </button>
                  ))}
                </div>

                {symInfo?.wyckoff?.length > 0 && (
                  <div style={{ fontSize: 'var(--fz-small)', color: 'var(--secondary)' }}>
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
        </section>
      </div>
    </div>
  );
}
