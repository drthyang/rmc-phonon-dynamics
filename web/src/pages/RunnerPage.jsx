import React, { useState, useMemo } from 'react';
import { FolderOpen, Play, Settings, Database, Network } from 'lucide-react';
import { listConfigs, readBaseStructure, findStructureFile, listRmc6f } from '../io/readers';
import { conventionalLattice, buildKPathFromSegments } from '../math/reciprocal';
import { analyzeBravais } from '../math/bravais';
import { buildBZModel, displayLabel } from '../math/highsym';
import { fromBandText } from '../io/viewermodel';
import BrillouinZoneViewer from '../components/BrillouinZoneViewer';
import CrystalViewer from '../components/CrystalViewer';
import DatasetInspector from '../components/DatasetInspector';
import FitQuality from '../components/FitQuality';
import PhononDOS from '../components/PhononDOS';
import { Upload } from 'lucide-react';

/**
 * Runner page — the full data → structure → k-path → run workflow, mirroring the
 * legacy rmcph_gui 4-step runner (folder, structure override, displacement
 * reference, editable k-path segments, T, degenerate tolerance).
 */
export default function RunnerPage({ pipeline, onResults, onLoadResult }) {
  const [dirHandle, setDirHandle] = useState(null);
  const [filesList, setFilesList] = useState([]);
  const [configFamily, setConfigFamily] = useState(null);
  const [rmc6fList, setRmc6fList] = useState([]);
  const [structureName, setStructureName] = useState(null);
  const [baseStructure, setBaseStructure] = useState(null);

  const [refMode, setRefMode] = useState('average');     // 'average' | 'file'
  const [refName, setRefName] = useState('');

  const [temperature, setTemperature] = useState(5);
  const [degenerateTol, setDegenerateTol] = useState(5e-3);
  const [density, setDensity] = useState(20);            // default points/segment

  const [bzSegments, setBzSegments] = useState([]);      // [{from,to}] label pairs
  const [pointsConv, setPointsConv] = useState({});      // label -> conventional fractional
  const [segNpoints, setSegNpoints] = useState({});      // {segIndex: npoints} overrides

  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressText, setProgressText] = useState('');

  // Bravais analysis (seekpath-style): primitive cell + BZ + standard points.
  // Memoized so bzModel keeps a STABLE identity across re-renders — otherwise the
  // BZ viewer's build effect re-runs every render and resets the camera (you
  // couldn't orbit the zone).
  const bravais = useMemo(
    () => (baseStructure?.v1 && baseStructure.basis
      ? analyzeBravais(conventionalLattice(baseStructure.v1, baseStructure.v2, baseStructure.v3, baseStructure.dim), baseStructure.basis)
      : null),
    [baseStructure]
  );
  const bzModel = useMemo(() => (bravais ? buildBZModel(bravais) : null), [bravais]);

  // Static crystal-structure preview (basis sites of one conventional cell),
  // available before running. Memoized so CrystalViewer doesn't rebuild/reset.
  const previewStruct = useMemo(() => {
    if (!baseStructure?.basis) return null;
    const b = baseStructure.basis;
    return {
      v1: baseStructure.v1, v2: baseStructure.v2, v3: baseStructure.v3, dim: baseStructure.dim,
      atomDic: baseStructure.atomDic,
      uniqueRN: b.map(x => x.rn),
      atomType: b.map(x => x.rn),
      hsym_xyz: Float64Array.from(b.flatMap(x => x.frac)),
      cellIdx: new Float64Array(b.length * 3),
    };
  }, [baseStructure]);

  const loadStructure = async (handle, name) => {
    const info = await readBaseStructure(handle);
    setBaseStructure(info);
    setStructureName(name);
  };

  const handleSelectFolder = async () => {
    try {
      const dh = await window.showDirectoryPicker({ mode: 'read' });
      setDirHandle(dh);
      const { files, family } = await listConfigs(dh);
      setFilesList(files); setConfigFamily(family);
      const rlist = await listRmc6f(dh);
      setRmc6fList(rlist);

      if (family === 'rmc6f' && files.length > 0) {
        await loadStructure(files[0], files[0].name);
      } else if (family === 'frac') {
        const sh = await findStructureFile(dh);
        if (!sh) { setProgressText('Frac configs found but no .rmc6f structure file in this folder.'); setBaseStructure(null); return; }
        await loadStructure(sh, sh.name);
      } else { setBaseStructure(null); }
    } catch (err) { console.error(err); }
  };

  const onStructureChange = async (name) => {
    const item = rmc6fList.find(r => r.name === name);
    if (item) await loadStructure(item.handle, name);
  };

  // Segments from the BZ path, with optional per-segment npoints overrides.
  const segments = bzSegments.map((s, i) => ({ from: s.from, to: s.to, npoints: segNpoints[i] ?? density }));
  const totalK = segments.reduce((a, s) => a + Math.max(2, s.npoints), 0);

  const run = async () => {
    if (!filesList.length || !baseStructure) return;
    if (!pipeline) { setProgressText('Compute engine still initializing — try again in a moment.'); return; }
    if (segments.length < 1) { setProgressText('Build a k-path first (click ≥2 high-symmetry points).'); return; }
    setIsProcessing(true); setProgress(0); setProgressText('Starting…');
    try {
      // pointsConv maps labels -> CONVENTIONAL fractional reciprocal coords, which
      // is what the calculation needs (the supercell tiles the conventional cell).
      const { qFrac, segSizes, hsymIndex } = buildKPathFromSegments(pointsConv, segments);
      // Convert internal labels (GAMMA, …) to display (Γ) for plot axis labels.
      const hsymDisplay = {};
      for (const [k, v] of Object.entries(hsymIndex)) hsymDisplay[k] = displayLabel(v);
      const kpathMeta = { qFrac, segSizes, hsymIndex: hsymDisplay };

      let referenceHandle = null;
      if (refMode === 'file') {
        const item = rmc6fList.find(r => r.name === refName);
        if (!item) { setProgressText('Select an equilibrium .rmc6f for the file reference.'); setIsProcessing(false); return; }
        referenceHandle = item.handle;
      }

      pipeline.onProgress = (p, t) => { setProgress(p); setProgressText(t); };
      const res = await pipeline.runCalculation(
        filesList, configFamily, baseStructure, qFrac, temperature, 50,
        { referenceHandle, degenerateTol }
      );
      setProgressText('Done — opening viewer…');
      onResults(res, kpathMeta);
    } catch (e) {
      if (e.message === 'cancelled') setProgressText('Cancelled.');
      else { console.error(e); setProgressText('Error: ' + e.message); }
      setIsProcessing(false);
    }
  };

  return (
    <div className="flex flex-col gap-6">
    <div className="grid grid-cols-12 gap-6">
      {/* Left: data + reference + settings */}
      <div className="col-span-12 lg:col-span-4 flex flex-col gap-6">
        <div className="glass-panel rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-4 text-gray-200"><Database className="w-5 h-5 text-indigo-400" /><h2 className="text-lg font-medium">1 · Data folder</h2></div>
          <button onClick={handleSelectFolder} className="w-full flex items-center justify-center gap-2 bg-white/5 hover:bg-white/10 border border-white/10 py-3 px-4 rounded-xl font-medium">
            <FolderOpen className="w-5 h-5" />{dirHandle ? 'Change Directory' : 'Select Directory'}
          </button>

          <label className="mt-2 w-full flex items-center justify-center gap-2 bg-white/5 hover:bg-white/10 border border-white/10 py-2 px-4 rounded-xl text-sm cursor-pointer">
            <Upload className="w-4 h-4" />Load saved result (.yaml/.json)
            <input type="file" accept=".yaml,.yml,.json" className="hidden" onChange={async (e) => {
              const file = e.target.files?.[0]; if (!file) return;
              try { const m = fromBandText(await file.text()); onLoadResult(m); }
              catch (err) { setProgressText('Load failed: ' + err.message); }
            }} />
          </label>

          {rmc6fList.length > 1 && (
            <div className="mt-4">
              <label className="text-xs text-gray-400 uppercase tracking-wider block mb-1">Structure file</label>
              <select value={structureName || ''} onChange={e => onStructureChange(e.target.value)} className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm">
                {rmc6fList.map(r => <option key={r.name} value={r.name}>{r.name}</option>)}
              </select>
            </div>
          )}

          <DatasetInspector directoryName={dirHandle?.name} filesList={filesList} configFamily={configFamily} baseStructure={baseStructure} />
        </div>

        {/* Displacement reference */}
        <div className={`glass-panel rounded-2xl p-6 ${baseStructure ? '' : 'opacity-50 pointer-events-none'}`}>
          <div className="flex items-center gap-3 mb-4 text-gray-200"><Settings className="w-5 h-5 text-indigo-400" /><h2 className="text-lg font-medium">Displacement reference (hsym)</h2></div>
          <label className="flex items-center gap-2 text-sm mb-2">
            <input type="radio" name="refmode" checked={refMode === 'average'} onChange={() => setRefMode('average')} />
            Average of all configurations <span className="text-gray-500 text-xs">(default)</span>
          </label>
          <label className="flex items-center gap-2 text-sm">
            <input type="radio" name="refmode" checked={refMode === 'file'} onChange={() => setRefMode('file')} />
            Equilibrium .rmc6f file:
          </label>
          {refMode === 'file' && (
            <select value={refName} onChange={e => setRefName(e.target.value)} className="mt-2 w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm">
              <option value="">(select a file)</option>
              {rmc6fList.map(r => <option key={r.name} value={r.name}>{r.name}</option>)}
            </select>
          )}
        </div>

        {/* Run settings */}
        <div className={`glass-panel rounded-2xl p-6 ${baseStructure ? '' : 'opacity-50 pointer-events-none'}`}>
          <div className="flex items-center gap-3 mb-4 text-gray-200"><Play className="w-5 h-5 text-indigo-400" /><h2 className="text-lg font-medium">4 · Run</h2></div>
          {bravais && <div className="text-xs text-gray-400 mb-3">Bravais: <span className="text-indigo-300 font-mono">{bravais.code}</span> ({bravais.system}, {bravais.centering}-centered)</div>}
          <div className="grid grid-cols-3 gap-2 mb-4">
            <Num label="T (K)" value={temperature} step={1} onChange={setTemperature} />
            <Num label="degen tol" value={degenerateTol} step={0.001} onChange={setDegenerateTol} />
            <Num label="pts/seg" value={density} step={1} onChange={v => setDensity(Math.max(2, Math.round(v)))} />
          </div>
          <div className="text-xs text-gray-500 mb-3">{segments.length} segment(s) · {totalK} k-points</div>
          {isProcessing ? (
            <div className="flex gap-2">
              <button disabled className="flex-1 flex items-center justify-center gap-2 py-3 px-4 rounded-xl font-medium bg-blue-600/50">
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />Processing…
              </button>
              <button onClick={() => { pipeline?.cancel(); setProgressText('Cancelling…'); }}
                className="px-4 py-3 rounded-xl font-medium bg-red-600/80 hover:bg-red-600">Cancel</button>
            </div>
          ) : (
            <button onClick={run}
              className="w-full flex items-center justify-center gap-2 py-3 px-4 rounded-xl font-medium bg-blue-600 hover:bg-blue-500 shadow-lg shadow-blue-500/25">
              <Play className="w-5 h-5" />Run phonon bands
            </button>
          )}
          {progressText && (
            <div className="mt-4">
              <div className="flex justify-between text-xs text-gray-400 mb-1"><span className="truncate mr-3">{progressText}</span><span>{Math.round(progress)}%</span></div>
              <div className="w-full bg-white/10 h-2 rounded-full overflow-hidden"><div className="h-full bg-blue-500" style={{ width: `${progress}%` }} /></div>
            </div>
          )}
        </div>
      </div>

      {/* Right: BZ + k-path editor */}
      <div className="col-span-12 lg:col-span-8 flex flex-col gap-6">
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <div className={`glass-panel rounded-2xl h-[420px] relative ${baseStructure ? '' : 'opacity-50 pointer-events-none'}`}>
            <div className="absolute top-2 left-3 z-10 text-xs text-gray-400 pointer-events-none">2 · Crystal structure</div>
            {previewStruct
              ? <CrystalViewer baseStructure={previewStruct} eigenvector={null} isPlaying={false} supercell={[1, 1, 1]} showCell={true} showBonds={true} />
              : <div className="h-full flex items-center justify-center text-gray-500 text-sm">Select a dataset to preview the structure.</div>}
          </div>
          <div className={`glass-panel rounded-2xl h-[420px] ${baseStructure ? '' : 'opacity-50 pointer-events-none'}`}>
            <BrillouinZoneViewer bzModel={bzModel} system={bravais?.system}
              onPathChange={(segs, conv) => { setBzSegments(segs); setPointsConv(conv); setSegNpoints({}); }} />
          </div>
        </div>

        <div className={`glass-panel rounded-2xl p-6 ${baseStructure ? '' : 'opacity-50 pointer-events-none'}`}>
          <div className="flex items-center gap-3 mb-4 text-gray-200"><Network className="w-5 h-5 text-amber-500" /><h2 className="text-lg font-medium">3 · k-path segments</h2></div>
          {segments.length === 0 ? (
            <p className="text-gray-500 text-sm">Click high-symmetry points above to build a path.</p>
          ) : (
            <div className="space-y-2">
              {segments.map((s, i) => (
                <div key={i} className="flex items-center gap-3 text-sm">
                  <span className="font-mono w-20">{displayLabel(s.from)} → {displayLabel(s.to)}</span>
                  <span className="text-gray-500 text-xs">npoints</span>
                  <input type="number" min={2} value={s.npoints}
                    onChange={e => setSegNpoints(m => ({ ...m, [i]: Math.max(2, parseInt(e.target.value) || 2) }))}
                    className="w-20 bg-white/5 border border-white/10 rounded px-2 py-1 text-sm" />
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>

    {baseStructure && filesList.length > 0 && (
      <div className="glass-panel rounded-2xl p-6">
        <PhononDOS pipeline={pipeline} files={filesList} family={configFamily} baseStructure={baseStructure}
          temperature={temperature}
          referenceHandle={refMode === 'file' ? rmc6fList.find(r => r.name === refName)?.handle : null} />
      </div>
    )}

    {dirHandle && (
      <div className="glass-panel rounded-2xl p-6">
        <FitQuality dirHandle={dirHandle} />
      </div>
    )}
    </div>
  );
}

function Num({ label, value, step, onChange }) {
  return (
    <div>
      <label className="text-[10px] text-gray-400 uppercase tracking-wider block mb-1">{label}</label>
      <input type="number" step={step} value={value} onChange={e => onChange(parseFloat(e.target.value))}
        className="w-full bg-white/5 border border-white/10 rounded px-2 py-1 text-sm" />
    </div>
  );
}
