import React, { useState } from 'react';
import { FolderOpen, Play, Settings, Database, Network } from 'lucide-react';
import { listConfigs, readBaseStructure, findStructureFile, listRmc6f } from '../io/readers';
import { detectSystem, highSymmetryPoints, buildKPathFromSegments } from '../math/reciprocal';
import BrillouinZoneViewer from '../components/BrillouinZoneViewer';
import DatasetInspector from '../components/DatasetInspector';

/**
 * Runner page — the full data → structure → k-path → run workflow, mirroring the
 * legacy rmcph_gui 4-step runner (folder, structure override, displacement
 * reference, editable k-path segments, T, degenerate tolerance).
 */
export default function RunnerPage({ pipeline, onResults }) {
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

  const [selectedPath, setSelectedPath] = useState([]);
  const [bzPoints, setBzPoints] = useState({});
  const [segNpoints, setSegNpoints] = useState({});      // {segIndex: npoints} overrides

  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressText, setProgressText] = useState('');

  const crystalInfo = baseStructure?.v1 ? detectSystem(baseStructure.v1, baseStructure.v2, baseStructure.v3, baseStructure.dim) : null;
  const symSet = crystalInfo ? highSymmetryPoints(crystalInfo.system) : null;

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

  // Segments derived from the clicked path, with optional per-segment overrides.
  const segments = [];
  for (let i = 0; i < selectedPath.length - 1; i++) {
    segments.push({ from: selectedPath[i], to: selectedPath[i + 1], npoints: segNpoints[i] ?? density });
  }
  const totalK = segments.reduce((a, s) => a + Math.max(2, s.npoints), 0);

  const run = async () => {
    if (!filesList.length || !baseStructure) return;
    if (!pipeline) { setProgressText('Compute engine still initializing — try again in a moment.'); return; }
    if (segments.length < 1) { setProgressText('Build a k-path first (click ≥2 high-symmetry points).'); return; }
    setIsProcessing(true); setProgress(0); setProgressText('Starting…');
    try {
      const { qFrac, segSizes, hsymIndex } = buildKPathFromSegments(bzPoints, segments);
      const kpathMeta = { qFrac, segSizes, hsymIndex, pathLabels: selectedPath };

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
      console.error(e); setProgressText('Error: ' + e.message); setIsProcessing(false);
    }
  };

  return (
    <div className="grid grid-cols-12 gap-6">
      {/* Left: data + reference + settings */}
      <div className="col-span-12 lg:col-span-4 flex flex-col gap-6">
        <div className="glass-panel rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-4 text-gray-200"><Database className="w-5 h-5 text-indigo-400" /><h2 className="text-lg font-medium">1 · Data folder</h2></div>
          <button onClick={handleSelectFolder} className="w-full flex items-center justify-center gap-2 bg-white/5 hover:bg-white/10 border border-white/10 py-3 px-4 rounded-xl font-medium">
            <FolderOpen className="w-5 h-5" />{dirHandle ? 'Change Directory' : 'Select Directory'}
          </button>

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
          {crystalInfo && <div className="text-xs text-gray-400 mb-3">System: <span className="text-indigo-300 font-mono">{crystalInfo.system}</span> · a,b,c = {crystalInfo.a.toFixed(2)}, {crystalInfo.b.toFixed(2)}, {crystalInfo.c.toFixed(2)} Å</div>}
          <div className="grid grid-cols-3 gap-2 mb-4">
            <Num label="T (K)" value={temperature} step={1} onChange={setTemperature} />
            <Num label="degen tol" value={degenerateTol} step={0.001} onChange={setDegenerateTol} />
            <Num label="pts/seg" value={density} step={1} onChange={v => setDensity(Math.max(2, Math.round(v)))} />
          </div>
          <div className="text-xs text-gray-500 mb-3">{segments.length} segment(s) · {totalK} k-points</div>
          <button onClick={run} disabled={isProcessing}
            className={`w-full flex items-center justify-center gap-2 py-3 px-4 rounded-xl font-medium ${isProcessing ? 'bg-blue-600/50' : 'bg-blue-600 hover:bg-blue-500 shadow-lg shadow-blue-500/25'}`}>
            {isProcessing ? <><div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />Processing…</> : <><Play className="w-5 h-5" />Run phonon bands</>}
          </button>
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
        <div className={`glass-panel rounded-2xl h-[420px] ${baseStructure ? '' : 'opacity-50 pointer-events-none'}`}>
          <BrillouinZoneViewer symSet={symSet} system={crystalInfo?.system}
            onPathChange={(path, points) => { setSelectedPath(path); setBzPoints(points); setSegNpoints({}); }} />
        </div>

        <div className={`glass-panel rounded-2xl p-6 ${baseStructure ? '' : 'opacity-50 pointer-events-none'}`}>
          <div className="flex items-center gap-3 mb-4 text-gray-200"><Network className="w-5 h-5 text-amber-500" /><h2 className="text-lg font-medium">3 · k-path segments</h2></div>
          {segments.length === 0 ? (
            <p className="text-gray-500 text-sm">Click high-symmetry points above to build a path.</p>
          ) : (
            <div className="space-y-2">
              {segments.map((s, i) => (
                <div key={i} className="flex items-center gap-3 text-sm">
                  <span className="font-mono w-20">{s.from} → {s.to}</span>
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
