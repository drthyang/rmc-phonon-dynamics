import React, { useState, useMemo } from 'react';
import { Upload, Download, Box, Waves, Play, Pause, Circle } from 'lucide-react';
import BandStructurePlot from '../components/BandStructurePlot';
import CrystalViewer from '../components/CrystalViewer';
import { DEFAULT_COLORS, COVALENT_R } from '../constants';
import ModeInspector from '../components/ModeInspector';
import InsPanel from '../components/InsPanel';
import { generatePhonopyBandYaml, generateBandJson, downloadString } from '../io/writers';
import { exportVASP } from '../io/vaspexport';
import { fromBandText } from '../io/viewermodel';

/**
 * Viewer page — tabbed band structure / 3D mode / S(Q,E), with the full control
 * set ported from the legacy viz viewer. Works on a unit-cell "model" coming
 * either from the runner hand-off or a loaded band.yaml/.json.
 */
export default function ViewerPage({ model, onLoadModel }) {
  const [tab, setTab] = useState('modes');
  const [selK, setSelK] = useState(0);
  const [selM, setSelM] = useState(0);

  // Band y-range
  const [eMin, setEMin] = useState('');
  const [eMax, setEMax] = useState('');

  // 3D controls
  const [nx, setNx] = useState(2), [ny, setNy] = useState(2), [nz, setNz] = useState(1);
  // Stable identity so CrystalViewer doesn't rebuild (and reset the camera) on
  // every unrelated re-render (e.g. moving the amplitude slider).
  const supercell = useMemo(() => [nx, ny, nz], [nx, ny, nz]);
  const [amplitude, setAmplitude] = useState(3);
  const [speed, setSpeed] = useState(0.08);
  const [playing, setPlaying] = useState(true);
  const [showVectors, setShowVectors] = useState(false);
  const [showCell, setShowCell] = useState(true);
  const [atomScale, setAtomScale] = useState(1);
  const [camNonce, setCamNonce] = useState(null);

  // Appearance
  const [displayStyle, setDisplayStyle] = useState('ballstick');
  const [showBonds, setShowBonds] = useState(true);
  const [bondScale, setBondScale] = useState(1.15);
  const [bondRules, setBondRules] = useState({}); // "A-B" -> cutoff Å override
  const [shading, setShading] = useState(true);
  const [elementColors, setElementColors] = useState({});
  const [elementRadii, setElementRadii] = useState({});
  const [recording, setRecording] = useState(false);
  const [gifSignal, setGifSignal] = useState(0);

  const [thz, setThz] = useState(false);
  const [loadErr, setLoadErr] = useState(null);

  const elements = model ? Object.keys(model.baseStructure.atomDic) : [];

  const loadFile = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const m = fromBandText(text, { thz });
      setLoadErr(null); setSelK(0); setSelM(0);
      onLoadModel(m);
    } catch (err) { setLoadErr(err.message); }
  };

  if (!model) {
    return (
      <div className="glass-panel rounded-2xl p-10 text-center">
        <p className="text-gray-300 mb-4">No data loaded. Run a calculation on the <b>Runner</b> page, or load a <code>band.yaml</code>/<code>.json</code>.</p>
        <FileLoad onLoad={loadFile} thz={thz} setThz={setThz} />
        {loadErr && <div className="text-red-400 text-sm mt-3">{loadErr}</div>}
      </div>
    );
  }

  const eig = model.eigvecs?.[selK]?.[selM] || null;
  const qPoint = model.qPoints?.[selK];
  const nModes = model.bands[0].length;
  const nK = model.bands.length;
  const energy = model.bands[selK]?.[selM];

  const exportYaml = () => downloadString(generatePhonopyBandYaml(model.baseStructure, model.qPoints, model.bands, model.eigvecs, model.kpathMeta), 'band_gpu.yaml', 'text/yaml');
  const exportJson = () => downloadString(generateBandJson(model.baseStructure, model.qPoints, model.bands, model.eigvecs, model.kpathMeta), 'band_gpu.json', 'application/json');

  return (
    <div className="flex flex-col gap-4">
      {/* Toolbar */}
      <div className="glass-panel rounded-2xl px-4 py-3 flex flex-wrap items-center gap-3">
        <div className="flex rounded-lg overflow-hidden border border-white/10">
          <TabBtn icon={<Box className="w-4 h-4" />} label="Bands + Mode" active={tab === 'modes'} onClick={() => setTab('modes')} />
          <TabBtn icon={<Waves className="w-4 h-4" />} label="S(Q,E)" active={tab === 'sqe'} onClick={() => setTab('sqe')} />
        </div>
        <div className="flex items-center gap-2 text-xs font-mono text-gray-300">
          <span className="text-gray-500">mode</span>
          k <input type="number" min={1} max={nK} value={selK + 1} onChange={e => setSelK(Math.max(0, Math.min(nK - 1, (parseInt(e.target.value) || 1) - 1)))} className="w-14 bg-white/5 border border-white/10 rounded px-1 py-0.5" />
          band <input type="number" min={1} max={nModes} value={selM + 1} onChange={e => setSelM(Math.max(0, Math.min(nModes - 1, (parseInt(e.target.value) || 1) - 1)))} className="w-14 bg-white/5 border border-white/10 rounded px-1 py-0.5" />
          {Number.isFinite(energy) && <span className={energy < 0 ? 'text-red-400' : 'text-blue-300'}>{energy.toFixed(3)} meV</span>}
        </div>
        <div className="flex-1" />
        <FileLoad onLoad={loadFile} thz={thz} setThz={setThz} compact />
        {model.eigvecs && (
          <>
            <button onClick={exportYaml} className="flex items-center gap-1 text-xs bg-white/10 hover:bg-white/20 px-3 py-1.5 rounded border border-white/10"><Download className="w-3.5 h-3.5" />band.yaml</button>
            <button onClick={exportJson} className="flex items-center gap-1 text-xs bg-white/10 hover:bg-white/20 px-3 py-1.5 rounded border border-white/10"><Download className="w-3.5 h-3.5" />band.json</button>
            <button onClick={() => exportVASP(model, selK, selM)} title="POSCAR (eq + displaced mode) + INCAR + KPOINTS" className="flex items-center gap-1 text-xs bg-white/10 hover:bg-white/20 px-3 py-1.5 rounded border border-white/10"><Download className="w-3.5 h-3.5" />VASP</button>
          </>
        )}
      </div>

      {loadErr && <div className="text-red-400 text-sm">{loadErr}</div>}

      {/* Bands + Mode — side by side; click a band point to see the mode */}
      {tab === 'modes' && (
        <>
          <div className="grid grid-cols-12 gap-4">
            <div className="col-span-12 xl:col-span-6 glass-panel rounded-2xl h-[520px] relative">
              <div className="absolute top-3 right-3 z-20 flex items-center gap-2 text-xs text-gray-400">
                E-range
                <input type="number" placeholder="min" value={eMin} onChange={e => setEMin(e.target.value)} className="w-16 bg-white/5 border border-white/10 rounded px-1 py-0.5" />
                <input type="number" placeholder="max" value={eMax} onChange={e => setEMax(e.target.value)} className="w-16 bg-white/5 border border-white/10 rounded px-1 py-0.5" />
              </div>
              <BandStructurePlot bands={model.bands} qPoints={model.qPoints} baseStructure={model.baseStructure}
                kpathMeta={model.kpathMeta} selected={{ k: selK, m: selM }}
                eMin={eMin === '' ? undefined : parseFloat(eMin)} eMax={eMax === '' ? undefined : parseFloat(eMax)}
                onPick={(k, m) => { setSelK(k); setSelM(m); }} />
            </div>
            <div className="col-span-12 xl:col-span-6 glass-panel rounded-2xl h-[520px] relative">
              {model.eigvecs ? (
                <>
                  <CrystalViewer baseStructure={model.baseStructure} eigenvector={eig} qPoint={qPoint}
                    isPlaying={playing} amplitude={amplitude} speed={speed}
                    supercell={supercell} showVectors={showVectors} showCell={showCell} atomScale={atomScale}
                    cameraAxis={camNonce ? camNonce[0] : null}
                    elementColors={elementColors} elementRadii={elementRadii} displayStyle={displayStyle}
                    showBonds={showBonds} bondScale={bondScale} bondRules={bondRules} shading={shading} recording={recording} gifSignal={gifSignal} />
                  <div className="absolute bottom-3 left-3"><ModeInspector results={model} selectedK={selK} selectedMode={selM} /></div>
                </>
              ) : <div className="h-full flex items-center justify-center text-gray-500 text-sm">Loaded file has no eigenvectors — 3D modes unavailable.</div>}
            </div>
          </div>

          {/* Controls */}
          <div className="glass-panel rounded-2xl p-4 grid grid-cols-1 md:grid-cols-4 gap-5 text-sm">
            <div className="space-y-3">
              <div>
                <div className="text-xs text-gray-400 uppercase tracking-wider mb-1">Supercell</div>
                <div className="flex gap-2">
                  {[['nx', nx, setNx], ['ny', ny, setNy], ['nz', nz, setNz]].map(([l, v, set]) => (
                    <input key={l} type="number" min={1} max={6} value={v} onChange={e => set(Math.max(1, Math.min(6, parseInt(e.target.value) || 1)))} className="w-full bg-white/5 border border-white/10 rounded px-2 py-1" title={l} />
                  ))}
                </div>
              </div>
              <Slider label={`Amplitude ${amplitude.toFixed(1)}`} min={0} max={10} step={0.1} value={amplitude} onChange={setAmplitude} />
              <Slider label={`Speed ${speed.toFixed(2)}`} min={0.01} max={0.3} step={0.01} value={speed} onChange={setSpeed} />
              <div className="flex items-center gap-2">
                <button onClick={() => setPlaying(p => !p)} className="flex items-center gap-1 bg-white/10 hover:bg-white/20 px-3 py-1.5 rounded border border-white/10 text-xs">
                  {playing ? <><Pause className="w-3.5 h-3.5" />Pause</> : <><Play className="w-3.5 h-3.5" />Play</>}
                </button>
                <button onClick={() => setRecording(r => !r)} className={`flex items-center gap-1 px-3 py-1.5 rounded border text-xs ${recording ? 'bg-red-600 border-red-500' : 'bg-white/10 hover:bg-white/20 border-white/10'}`}>
                  <Circle className="w-3 h-3" />{recording ? 'Stop' : 'WebM'}
                </button>
                <button onClick={() => setGifSignal(s => s + 1)} title="Capture ~50 frames as an animated GIF" className="flex items-center gap-1 px-3 py-1.5 rounded border text-xs bg-white/10 hover:bg-white/20 border-white/10">GIF</button>
              </div>
              <div>
                <div className="text-xs text-gray-400 uppercase tracking-wider mb-1">Camera</div>
                <div className="flex gap-2">
                  {['x', 'y', 'z'].map(ax => <button key={ax} onClick={() => setCamNonce(ax + Math.random())} className="flex-1 bg-white/10 hover:bg-white/20 rounded py-1 text-xs uppercase">{ax}</button>)}
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <div className="text-xs text-gray-400 uppercase tracking-wider">Appearance</div>
              <label className="block text-xs">style
                <select value={displayStyle} onChange={e => setDisplayStyle(e.target.value)} className="w-full mt-1 bg-white/5 border border-white/10 rounded px-2 py-1">
                  <option value="ballstick">ball &amp; stick</option>
                  <option value="spacefill">spacefill</option>
                  <option value="wireframe">wireframe</option>
                </select>
              </label>
              <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={showBonds} onChange={e => setShowBonds(e.target.checked)} /> bonds</label>
              <Slider label={`bond cutoff ×${bondScale.toFixed(2)}`} min={0.6} max={1.8} step={0.05} value={bondScale} onChange={setBondScale} />
              {showBonds && elements.length > 0 && (
                <details className="text-xs">
                  <summary className="cursor-pointer text-gray-400">per-pair cutoffs (Å)</summary>
                  <div className="mt-1 space-y-1">
                    {elementPairs(elements).map(([a, b]) => {
                      const key = [a, b].sort().join('-');
                      return (
                        <div key={key} className="flex items-center gap-2">
                          <span className="font-mono w-12">{a}–{b}</span>
                          <input type="number" step={0.05} min={0} placeholder="auto" value={bondRules[key] ?? ''}
                            onChange={e => setBondRules(r => { const n = { ...r }; if (e.target.value === '') delete n[key]; else n[key] = parseFloat(e.target.value); return n; })}
                            className="w-16 bg-white/5 border border-white/10 rounded px-1 py-0.5" />
                        </div>
                      );
                    })}
                  </div>
                </details>
              )}
              <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={shading} onChange={e => setShading(e.target.checked)} /> shading</label>
              <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={showVectors} onChange={e => setShowVectors(e.target.checked)} /> displacement vectors</label>
              <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={showCell} onChange={e => setShowCell(e.target.checked)} /> show cell</label>
              <Slider label={`atom size ×${atomScale.toFixed(1)}`} min={0.3} max={3} step={0.1} value={atomScale} onChange={setAtomScale} />
            </div>

            <div className="max-h-[260px] overflow-y-auto custom-scrollbar">
              <div className="text-xs text-gray-400 uppercase tracking-wider mb-2">Atom types</div>
              {elements.map(el => (
                <div key={el} className="flex items-center gap-2 mb-1.5 text-xs">
                  <span className="w-8 font-mono">{el}</span>
                  <input type="color" value={elementColors[el] || DEFAULT_COLORS[el] || '#cccccc'}
                    onChange={e => setElementColors(c => ({ ...c, [el]: e.target.value }))} className="w-7 h-6 bg-transparent border border-white/10 rounded" />
                  <input type="number" min={0.1} max={3} step={0.05} value={elementRadii[el] ?? (COVALENT_R[el] || 1.0)}
                    onChange={e => setElementRadii(r => ({ ...r, [el]: parseFloat(e.target.value) }))} className="w-16 bg-white/5 border border-white/10 rounded px-1 py-0.5" title="radius (Å)" />
                </div>
              ))}
              <button onClick={() => { setElementColors({}); setElementRadii({}); }} className="text-xs text-gray-400 hover:text-gray-200 mt-1">reset types</button>
            </div>

            <div className="max-h-[260px] overflow-y-auto custom-scrollbar">
              <StructureTables model={model} />
            </div>
          </div>
        </>
      )}

      {/* S(Q,E) tab */}
      {tab === 'sqe' && (
        <div className="glass-panel rounded-2xl p-6">
          {model.eigvecs ? <InsPanel results={model} temperature={model.temperature} />
            : <div className="text-gray-500 text-sm">Loaded file has no eigenvectors — S(Q,E) unavailable. Eigenvectors are required for the structure factor.</div>}
        </div>
      )}
    </div>
  );
}

function elementPairs(elements) {
  const out = [];
  for (let i = 0; i < elements.length; i++) for (let j = i; j < elements.length; j++) out.push([elements[i], elements[j]]);
  return out;
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
    <div className="border-t border-white/10 pt-3 space-y-2">
      <details>
        <summary className="text-xs text-gray-400 uppercase tracking-wider cursor-pointer">Lattice (Å)</summary>
        <table className="mt-1 text-[10px] font-mono w-full">
          <tbody>
            {A.map((v, i) => (
              <tr key={i}><td className="text-gray-500 pr-2">{'abc'[i]}</td>{v.map((x, j) => <td key={j} className="text-right tabular-nums">{x.toFixed(3)}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </details>
      <details>
        <summary className="text-xs text-gray-400 uppercase tracking-wider cursor-pointer">Atom positions ({sites.length})</summary>
        <table className="mt-1 text-[10px] font-mono w-full">
          <thead><tr className="text-gray-500"><td>el</td><td className="text-right">x</td><td className="text-right">y</td><td className="text-right">z</td></tr></thead>
          <tbody>
            {sites.map((s, i) => (
              <tr key={i}><td>{s.el}</td>{s.pos.map((x, j) => <td key={j} className="text-right tabular-nums">{x.toFixed(4)}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}

function TabBtn({ icon, label, active, onClick }) {
  return (
    <button onClick={onClick} className={`flex items-center gap-1.5 px-3 py-1.5 text-sm ${active ? 'bg-blue-600 text-white' : 'bg-white/5 text-gray-300 hover:bg-white/10'}`}>
      {icon}{label}
    </button>
  );
}

function Slider({ label, min, max, step, value, onChange }) {
  return (
    <div>
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={e => onChange(parseFloat(e.target.value))} className="w-full" />
    </div>
  );
}

function FileLoad({ onLoad, thz, setThz, compact }) {
  return (
    <div className="flex items-center gap-2">
      <label className={`flex items-center gap-1 cursor-pointer ${compact ? 'text-xs bg-white/10 hover:bg-white/20 px-3 py-1.5 rounded border border-white/10' : 'bg-blue-600 hover:bg-blue-500 px-4 py-2 rounded-lg'}`}>
        <Upload className="w-3.5 h-3.5" />Load band.yaml/json
        <input type="file" accept=".yaml,.yml,.json" onChange={onLoad} className="hidden" />
      </label>
      <label className="text-xs text-gray-400 flex items-center gap-1"><input type="checkbox" checked={thz} onChange={e => setThz(e.target.checked)} /> THz</label>
    </div>
  );
}
