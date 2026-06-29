import React, { useState, useMemo } from 'react';
import BandStructurePlot from '../components/BandStructurePlot';
import CrystalViewer from '../components/CrystalViewer';
import { DEFAULT_COLORS, COVALENT_R } from '../constants';
import ModeInspector from '../components/ModeInspector';
import InsPanel from '../components/InsPanel';
import { generatePhonopyBandYaml, generateBandJson, downloadString } from '../io/writers';
import { modelFromText } from '../io/phonopyDM';

/* ── Cobalt theme tokens ───────────────────────────────────────────────── */
const INK = 'var(--ink)', DIM = 'var(--dim)', FAINT = 'var(--faint)';
const ACCENT = 'var(--accent)', ACCENTINK = 'var(--accentInk)', BORDER = 'var(--border)';
const INSET = 'var(--inset)', INSET2 = 'var(--inset2)', SOFT = 'var(--soft)';
const WARN = '#e0564b', WARNINK = 'var(--warnInk)';

const cardStyle = { background: 'var(--card)', border: `1px solid ${BORDER}`, borderRadius: 10 };
const cardTitle = { font: "600 13px 'Space Grotesk'", color: INK };
const chip = { display: 'flex', alignItems: 'center', gap: 6, background: INSET, border: `1px solid ${BORDER}`, borderRadius: 7, padding: '7px 12px', font: "600 12px 'Space Grotesk'", color: INK, cursor: 'pointer' };
const numBox = { background: INSET, border: `1px solid ${BORDER}`, borderRadius: 6, padding: '5px 8px', font: "13px 'Space Mono'", color: ACCENTINK, textAlign: 'center' };
const THZ_PER_MEV = 0.2418;

const energyColorOf = (e) => (e < 0 ? WARNINK : ACCENTINK);

/**
 * Viewer page (Cobalt redesign) — tabbed band structure / 3D mode / S(Q,E),
 * with the full control set. Works on a unit-cell "model" coming either from the
 * runner hand-off or a loaded band.yaml/.json. The compute/io/math layers are
 * untouched; this only reshapes the UI around the same data shapes.
 */
export default function ViewerPage({ model, onLoadModel }) {
  const [tab, setTab] = useState('modes');
  const [selK, setSelK] = useState(0);
  const [selM, setSelM] = useState(0);

  // Band y-range (in the currently displayed unit)
  const [eMin, setEMin] = useState('');
  const [eMax, setEMax] = useState('');

  // 3D controls
  const [nx, setNx] = useState(2), [ny, setNy] = useState(2), [nz, setNz] = useState(1);
  // Stable identity so CrystalViewer doesn't rebuild (and reset the camera) on
  // every unrelated re-render (e.g. moving the amplitude slider).
  const supercell = useMemo(() => [nx, ny, nz], [nx, ny, nz]);
  const [amplitude, setAmplitude] = useState(1.5);
  const [speed, setSpeed] = useState(0.03);
  const [playing, setPlaying] = useState(true);
  const [showVectors, setShowVectors] = useState(false);
  const [vectorScale, setVectorScale] = useState(2.0);
  const [vectorColor, setVectorColor] = useState('#e0564b');
  const [showCell, setShowCell] = useState(true);
  const [atomScale, setAtomScale] = useState(1);
  const [camNonce, setCamNonce] = useState(null);
  const [bandReset, setBandReset] = useState(0);   // bumps to reset the band-plot zoom

  // Appearance
  const [displayStyle, setDisplayStyle] = useState('ballstick');
  const [showBonds, setShowBonds] = useState(true);
  const [bondCutoff, setBondCutoff] = useState(3.0);   // absolute bond cutoff (Å)
  const [bondThickness, setBondThickness] = useState(0.06); // bond cylinder radius (Å)
  const [bondRules, setBondRules] = useState({}); // "A-B" -> cutoff Å override
  const [shading, setShading] = useState(true);
  const [shadingStrength, setShadingStrength] = useState(0.5);
  const [elementColors, setElementColors] = useState({});
  const [elementRadii, setElementRadii] = useState({});
  const [recording, setRecording] = useState(false);
  const [gifSignal, setGifSignal] = useState(0);

  const [thz, setThz] = useState(false);
  const [loadErr, setLoadErr] = useState(null);

  const elements = model ? Object.keys(model.baseStructure.atomDic) : [];
  const colorOf = (el) => elementColors[el] || DEFAULT_COLORS[el] || '#cccccc';

  // Display-unit conversion (live, meV ↔ THz). The model always stores meV.
  const unit = thz ? 'THz' : 'meV';
  const escale = thz ? THZ_PER_MEV : 1;
  const fmtE = (e) => Number.isFinite(e) ? `${(e * escale).toFixed(3)} ${unit}` : '—';

  // Band data scaled into the displayed unit for the plot. Declared before any
  // early return so the hook order stays stable when `model` is null.
  const displayBands = useMemo(
    () => !model ? null : (escale === 1 ? model.bands : model.bands.map(row => row.map(v => v * escale))),
    [model, escale]
  );

  const loadFile = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const m = modelFromText(await file.text());
      setLoadErr(null); setSelK(0); setSelM(0);
      onLoadModel(m);
    } catch (err) { setLoadErr(err.message); }
  };

  // ── empty state ────────────────────────────────────────────────────────
  if (!model) {
    return (
      <div style={{ ...cardStyle, padding: 40, textAlign: 'center' }}>
        <p style={{ font: "14px 'Spline Sans'", color: DIM, marginBottom: 18, lineHeight: 1.6 }}>
          No data loaded. Run a calculation on the <b style={{ color: INK }}>Runner</b> page,
          or load a <code style={{ font: "12px 'Space Mono'", color: ACCENTINK }}>band.yaml</code> / <code style={{ font: "12px 'Space Mono'", color: ACCENTINK }}>.json</code>.
        </p>
        <div style={{ display: 'flex', justifyContent: 'center', gap: 12, alignItems: 'center' }}>
          <label style={{ ...chip, background: ACCENT, color: '#fff', border: 'none', padding: '10px 18px', font: "600 13px 'Space Grotesk'" }}>
            <IconUpload /> Load band.yaml / .json
            <input type="file" accept=".yaml,.yml,.json" onChange={loadFile} style={{ display: 'none' }} />
          </label>
        </div>
        {loadErr && <div style={{ color: WARNINK, font: "13px 'Space Mono'", marginTop: 14 }}>{loadErr}</div>}
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

  const resetCam = () => setCamNonce('reset:' + Math.random());   // recentre + fit, keep the supercell

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

      {/* ════════ TOOLBAR ════════ */}
      <div style={{ ...cardStyle, padding: '11px 14px', display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 14 }}>
        {/* tab switch */}
        <div style={{ display: 'flex', background: INSET2, borderRadius: 8, padding: 3 }}>
          <SegTab active={tab === 'modes'} onClick={() => setTab('modes')} icon={<IconBox />}>Bands + Mode</SegTab>
          <SegTab active={tab === 'sqe'} onClick={() => setTab('sqe')} icon={<IconBands />}>S(Q,E)</SegTab>
        </div>

        {/* mode selector — only relevant to the Bands + Mode tab */}
        {tab === 'modes' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 9, font: "12px 'Space Mono'", color: DIM }}>
            <span style={{ color: FAINT }}>mode</span>
            <span>k</span>
            <input type="number" min={1} max={nK} value={selK + 1}
              onChange={e => setSelK(Math.max(0, Math.min(nK - 1, (parseInt(e.target.value) || 1) - 1)))}
              style={{ ...numBox, width: 54 }} />
            <span>band</span>
            <input type="number" min={1} max={nModes} value={selM + 1}
              onChange={e => setSelM(Math.max(0, Math.min(nModes - 1, (parseInt(e.target.value) || 1) - 1)))}
              style={{ ...numBox, width: 48 }} />
            {Number.isFinite(energy) && <span style={{ fontWeight: 700, color: energyColorOf(energy) }}>{fmtE(energy)}</span>}
          </div>
        )}

        <div style={{ flex: 1 }} />

        {/* load + THz */}
        <label style={{ ...chip, color: DIM }}>
          <IconUpload /> Load band.yaml / .json
          <input type="file" accept=".yaml,.yml,.json" onChange={loadFile} style={{ display: 'none' }} />
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, font: "11px 'Space Mono'", color: FAINT, cursor: 'pointer' }}>
          <input type="checkbox" checked={thz} onChange={e => setThz(e.target.checked)} /> THz
        </label>

        {model.eigvecs && (
          <>
            <div style={{ width: 1, height: 24, background: BORDER }} />
            <button onClick={exportYaml} className="rnr-btn" style={chip}><IconDownload />band.yaml</button>
            <button onClick={exportJson} className="rnr-btn" style={chip}><IconDownload />band.json</button>
          </>
        )}
      </div>

      {loadErr && <div style={{ color: WARNINK, font: "13px 'Space Mono'" }}>{loadErr}</div>}

      {/* ════════ TAB 1 · BANDS + MODE ════════ */}
      {tab === 'modes' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: 16 }}>

            {/* Card A — band structure */}
            <div style={{ ...cardStyle, padding: '14px 16px', position: 'relative' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6, gap: 10 }}>
                <span style={cardTitle}>Phonon band structure</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: 7, font: "11px 'Space Mono'", color: FAINT }}>
                  E-range
                  <input type="number" placeholder="min" value={eMin} onChange={e => setEMin(e.target.value)} style={{ ...numBox, width: 50, color: INK }} />
                  <input type="number" placeholder="max" value={eMax} onChange={e => setEMax(e.target.value)} style={{ ...numBox, width: 50, color: INK }} />
                  <button onClick={() => { setEMin(''); setEMax(''); setBandReset(n => n + 1); }} title="Reset zoom & energy range" className="rnr-btn"
                    style={{ display: 'flex', alignItems: 'center', gap: 5, background: INSET, border: `1px solid ${BORDER}`, borderRadius: 6, padding: '4px 9px', font: "600 11px 'Space Grotesk'", color: DIM, cursor: 'pointer' }}>
                    <IconReset />Reset
                  </button>
                </div>
              </div>
              <BandStructurePlot bands={displayBands} qPoints={model.qPoints} baseStructure={model.baseStructure}
                kpathMeta={model.kpathMeta} selected={{ k: selK, m: selM }} unit={unit} resetSignal={bandReset}
                eMin={eMin === '' ? undefined : parseFloat(eMin)} eMax={eMax === '' ? undefined : parseFloat(eMax)}
                onPick={(k, m) => { setSelK(k); setSelM(m); }} />
            </div>

            {/* Card B — 3D mode displacement */}
            <div style={{ ...cardStyle, overflow: 'hidden', position: 'relative', display: 'flex', flexDirection: 'column' }}>
              <div style={{ position: 'absolute', top: 14, left: 16, zIndex: 3, ...cardTitle }}>Mode displacement</div>
              <div style={{ position: 'absolute', top: 14, right: 16, zIndex: 3, display: 'flex', alignItems: 'center', gap: 8, font: "10px 'Space Mono'", color: FAINT }}>
                {recording && <span style={{ display: 'flex', alignItems: 'center', gap: 5, color: WARNINK }}><span style={{ width: 7, height: 7, borderRadius: '50%', background: WARN, animation: 'blip 1s infinite' }} />REC</span>}
                <span>{nx}×{ny}×{nz} cell</span>
                <button onClick={resetCam} title="Recentre & fit the camera" className="rnr-btn"
                  style={{ display: 'flex', alignItems: 'center', gap: 5, background: 'var(--card)', border: `1px solid ${BORDER}`, borderRadius: 6, padding: '4px 9px', font: "600 11px 'Space Grotesk'", color: DIM, cursor: 'pointer' }}>
                  <IconReset />Reset
                </button>
              </div>

              <div style={{ flex: 1, minHeight: 440, background: INSET, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                {model.eigvecs
                  ? <CrystalViewer baseStructure={model.baseStructure} eigenvector={eig} qPoint={qPoint}
                      isPlaying={playing} amplitude={amplitude} speed={speed}
                      supercell={supercell} showVectors={showVectors} showCell={showCell} atomScale={atomScale}
                      cameraAxis={camNonce}
                      elementColors={elementColors} elementRadii={elementRadii} displayStyle={displayStyle}
                      showBonds={showBonds} bondCutoff={bondCutoff} bondThickness={bondThickness} bondRules={bondRules} shading={shading} shadingStrength={shadingStrength}
                      recording={recording} gifSignal={gifSignal} vectorScale={vectorScale} vectorColor={vectorColor} />
                  : <span style={{ color: FAINT, font: "12px 'Spline Sans'" }}>Loaded file has no eigenvectors — 3D modes unavailable.</span>}
              </div>

              {/* mode-inspector footer (in-flow; never covers the atoms) */}
              {model.eigvecs && <ModeInspector results={model} selectedK={selK} selectedMode={selM} unit={unit} scale={escale} colors={elementColors} />}

              {/* legend */}
              <div style={{ display: 'flex', gap: 16, padding: '10px 16px', borderTop: `1px solid ${BORDER}`, font: "11px 'Space Mono'", color: DIM, flexWrap: 'wrap' }}>
                {elements.map(el => <span key={el}><span style={{ color: colorOf(el) }}>●</span> {el}</span>)}
                <span style={{ marginLeft: 'auto', color: FAINT }}>click a band point to inspect a mode</span>
              </div>
            </div>
          </div>

          {/* ── controls card ── */}
          <div style={{ ...cardStyle, padding: 18, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 24 }}>

            {/* col 1 · MOTION */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <ColHead>MOTION</ColHead>
              <Slider label="amplitude" value={amplitude} valueLabel={`×${amplitude.toFixed(1)}`} min={0} max={10} step={0.1} onChange={setAmplitude} />
              <Slider label="speed" value={speed} valueLabel={speed.toFixed(2)} min={0.01} max={0.3} step={0.01} onChange={setSpeed} />
              <div>
                <FieldLabel>playback</FieldLabel>
                <div style={{ display: 'flex', gap: 7 }}>
                  <button onClick={() => setPlaying(p => !p)} className="rnr-btn" style={{ ...chip, flex: 1, justifyContent: 'center' }}>{playing ? 'Pause' : 'Play'}</button>
                  <button onClick={() => setRecording(r => !r)} className="rnr-btn"
                    style={{ ...chip, background: recording ? WARN : INSET, borderColor: recording ? WARN : BORDER, color: recording ? '#fff' : INK }}>
                    {recording ? 'Stop' : 'WebM'}
                  </button>
                  <button onClick={() => setGifSignal(s => s + 1)} title="Capture ~50 frames as an animated GIF" className="rnr-btn" style={chip}>GIF</button>
                </div>
              </div>
            </div>

            {/* col 2 · VIEW */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <ColHead>VIEW</ColHead>
              <div>
                <FieldLabel>supercell</FieldLabel>
                <div style={{ display: 'flex', gap: 8 }}>
                  {[['nx', nx, setNx], ['ny', ny, setNy], ['nz', nz, setNz]].map(([l, v, set]) => (
                    <input key={l} type="number" min={1} max={6} value={v} title={l}
                      onChange={e => set(Math.max(1, Math.min(6, parseInt(e.target.value) || 1)))}
                      style={{ ...numBox, width: '100%', boxSizing: 'border-box', color: INK, padding: 7 }} />
                  ))}
                </div>
              </div>
              <div>
                <FieldLabel>camera axis</FieldLabel>
                <div style={{ display: 'flex', gap: 8 }}>
                  {['x', 'y', 'z'].map(ax => (
                    <button key={ax} onClick={() => setCamNonce(ax + Math.random())} className="rnr-btn"
                      style={{ flex: 1, background: INSET, border: `1px solid ${BORDER}`, borderRadius: 7, padding: 7, font: "600 12px 'Space Mono'", color: DIM, textTransform: 'uppercase', cursor: 'pointer' }}>{ax}</button>
                  ))}
                </div>
              </div>
              <Slider label="atom size" value={atomScale} valueLabel={`×${atomScale.toFixed(1)}`} min={0.3} max={3} step={0.1} onChange={setAtomScale} />
              <Check checked={showCell} onChange={setShowCell}>show box</Check>
            </div>

            {/* col 3 · RENDERING */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <ColHead>RENDERING</ColHead>
              <label style={{ font: "11px 'Space Mono'", color: DIM }}>representation
                <select value={displayStyle} onChange={e => setDisplayStyle(e.target.value)}
                  style={{ width: '100%', marginTop: 5, boxSizing: 'border-box', background: INSET, border: `1px solid ${BORDER}`, borderRadius: 7, padding: '8px 10px', font: "12px 'Spline Sans'", color: INK, cursor: 'pointer' }}>
                  <option value="ballstick">ball &amp; stick</option>
                  <option value="spacefill">spacefill</option>
                  <option value="wireframe">wireframe</option>
                </select>
              </label>
              {/* primary toggles stay visible; secondary sliders fold below */}
              <Check checked={showBonds} onChange={setShowBonds}>bonds</Check>
              {showBonds && <Slider label="bond cutoff" value={bondCutoff} valueLabel={`${bondCutoff.toFixed(2)} Å`} min={1.5} max={4.5} step={0.05} onChange={setBondCutoff} />}
              <Check checked={shading} onChange={setShading}>shading</Check>
              <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
                <Check checked={showVectors} onChange={setShowVectors}>displacement vectors</Check>
                {showVectors && <span style={{ marginLeft: 'auto' }}><ColorPicker value={vectorColor} onChange={setVectorColor} /></span>}
              </div>
              {showVectors && <Slider label="vector size" value={vectorScale} valueLabel={`×${vectorScale.toFixed(1)}`} min={0.2} max={6} step={0.1} onChange={setVectorScale} />}

              {(showBonds || shading) && (
                <Foldout summary="fine-tuning">
                  {showBonds && <Slider label="bond thickness" value={bondThickness} valueLabel={bondThickness.toFixed(2)} min={0.02} max={0.3} step={0.01} onChange={setBondThickness} />}
                  {shading && <Slider label="shading strength" value={shadingStrength} valueLabel={shadingStrength.toFixed(2)} min={0} max={1} step={0.05} onChange={setShadingStrength} />}
                  {showBonds && elements.length > 1 && (
                    <div>
                      <FieldLabel>per-pair cutoffs (Å)</FieldLabel>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                        {elementPairs(elements).map(([a, b]) => {
                          const key = [a, b].sort().join('-');
                          return (
                            <div key={key} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                              <span style={{ width: 44, font: "11px 'Space Mono'", color: DIM }}>{a}–{b}</span>
                              <input type="number" step={0.05} min={0} placeholder="auto" value={bondRules[key] ?? ''}
                                onChange={e => setBondRules(r => { const n = { ...r }; if (e.target.value === '') delete n[key]; else n[key] = parseFloat(e.target.value); return n; })}
                                style={{ ...numBox, width: 64, color: INK }} />
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </Foldout>
              )}
            </div>

            {/* col 4 · ATOM TYPES + STRUCTURE */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <ColHead>ATOM TYPES</ColHead>
                <button onClick={() => { setElementColors({}); setElementRadii({}); }}
                  style={{ background: 'none', border: 'none', cursor: 'pointer', font: "11px 'Space Mono'", color: FAINT, textDecoration: 'underline' }}>reset</button>
              </div>
              {elements.map(el => (
                <div key={el} style={{ display: 'flex', alignItems: 'center', gap: 9, font: "12px 'Space Mono'", color: INK }}>
                  <span style={{ width: 24 }}>{el}</span>
                  <ColorPicker value={colorOf(el)} onChange={c => setElementColors(p => ({ ...p, [el]: c }))} />
                  <input type="number" min={0.1} max={3} step={0.05} value={elementRadii[el] ?? (COVALENT_R[el] || 1.0)} title="radius (Å)"
                    onChange={e => setElementRadii(r => ({ ...r, [el]: parseFloat(e.target.value) }))}
                    style={{ ...numBox, flex: 1, color: INK, textAlign: 'left', padding: '6px 8px' }} />
                  <span style={{ font: "10px 'Space Mono'", color: FAINT }}>Å</span>
                </div>
              ))}
              <div style={{ marginTop: 6, paddingTop: 12, borderTop: `1px solid ${BORDER}`, display: 'flex', flexDirection: 'column', gap: 6, maxHeight: 170, overflowY: 'auto' }}>
                <div style={{ font: "10px 'Space Mono'", letterSpacing: '.16em', color: FAINT }}>STRUCTURE</div>
                <StructureTables model={model} />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ════════ TAB 2 · S(Q,E) ════════ */}
      {tab === 'sqe' && (
        model.eigvecs
          ? <InsPanel results={model} temperature={model.temperature} />
          : <div style={{ ...cardStyle, padding: 24, color: FAINT, font: "13px 'Spline Sans'" }}>
              Loaded file has no eigenvectors — S(Q,E) unavailable. Eigenvectors are required for the structure factor.
            </div>
      )}
    </div>
  );
}

/* ── small building blocks ─────────────────────────────────────────────── */
function elementPairs(elements) {
  const out = [];
  for (let i = 0; i < elements.length; i++) for (let j = i; j < elements.length; j++) out.push([elements[i], elements[j]]);
  return out;
}

function SegTab({ active, onClick, icon, children }) {
  return (
    <button onClick={onClick} className="rnr-btn"
      style={{ display: 'flex', alignItems: 'center', gap: 7, padding: '7px 15px', border: 'none', borderRadius: 6, cursor: 'pointer', font: "600 13px 'Space Grotesk'", background: active ? ACCENT : 'transparent', color: active ? '#fff' : DIM }}>
      {icon}{children}
    </button>
  );
}

function ColHead({ children }) {
  return <div style={{ font: "10px 'Space Mono'", letterSpacing: '.16em', color: ACCENTINK }}>{children}</div>;
}
function FieldLabel({ children }) {
  return <div style={{ font: "10.5px 'Space Mono'", color: FAINT, marginBottom: 6 }}>{children}</div>;
}
function Slider({ label, value, valueLabel, min, max, step, onChange }) {
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', font: "11px 'Space Mono'", color: DIM, marginBottom: 5 }}>
        <span>{label}</span><span style={{ color: ACCENTINK }}>{valueLabel}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={e => onChange(parseFloat(e.target.value))}
        style={{ width: '100%', accentColor: ACCENT }} />
    </div>
  );
}
// Fixed, easy-to-scan colour palette (includes the element + vector defaults).
const PALETTE = ['#e0564b', '#e06a3b', '#fde047', '#f5b301', '#13a07f', '#10b981', '#2f6df0', '#5b677a', '#8b5cf6', '#ec4899', '#111827', '#ffffff'];
// A colour chip that reveals the palette popover only when clicked.
function ColorPicker({ value, onChange }) {
  const [open, setOpen] = useState(false);
  const cur = (value || '').toLowerCase();
  return (
    <div style={{ position: 'relative', display: 'inline-flex' }}
      tabIndex={0} onBlur={(e) => { if (!e.currentTarget.contains(e.relatedTarget)) setOpen(false); }}>
      <button onClick={() => setOpen(o => !o)} title="choose colour" className="rnr-btn"
        style={{ width: 18, height: 18, borderRadius: 5, background: value, border: `1px solid ${BORDER}`, cursor: 'pointer', padding: 0, flex: 'none' }} />
      {open && (
        <div style={{ position: 'absolute', top: '125%', left: 0, zIndex: 20, background: 'var(--card)', border: `1px solid ${BORDER}`, borderRadius: 8, padding: 6, boxShadow: '0 6px 18px rgba(16,24,38,0.14)' }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, width: 116 }}>
            {PALETTE.map(c => {
              const on = c.toLowerCase() === cur;
              return (
                <button key={c} onClick={() => { onChange(c); setOpen(false); }} title={c} className="rnr-btn"
                  style={{ width: 16, height: 16, borderRadius: 4, background: c, cursor: 'pointer', padding: 0,
                    border: on ? `2px solid ${INK}` : `1px solid ${BORDER}`, boxShadow: on ? '0 0 0 1.5px #fff inset' : 'none' }} />
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
// Collapsible group for secondary/fine-tuning sliders — keeps the control panel
// compact (closed by default; expand to reveal).
function Foldout({ summary, open = false, children }) {
  return (
    <details open={open} style={{ font: "11px 'Space Mono'", color: DIM }}>
      <summary style={{ cursor: 'pointer', color: ACCENTINK, userSelect: 'none' }}>{summary}</summary>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginTop: 10, paddingLeft: 10, borderLeft: `2px solid ${BORDER}` }}>
        {children}
      </div>
    </details>
  );
}
function Check({ checked, onChange, children }) {
  return (
    <label style={{ display: 'flex', alignItems: 'center', gap: 9, font: "12px 'Spline Sans'", color: INK, cursor: 'pointer' }}>
      <input type="checkbox" checked={checked} onChange={e => onChange(e.target.checked)} /> {children}
    </label>
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
  const td = { textAlign: 'right', color: INK };
  return (
    <>
      <details open>
        <summary style={{ font: "11px 'Space Mono'", color: DIM, cursor: 'pointer', marginBottom: 6 }}>Lattice (Å)</summary>
        <table style={{ width: '100%', font: "11px 'Space Mono'", borderCollapse: 'collapse' }}>
          <tbody>
            {A.map((v, i) => (
              <tr key={i}><td style={{ color: FAINT, paddingRight: 8 }}>{'abc'[i]}</td>{v.map((x, j) => <td key={j} style={td}>{x.toFixed(3)}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </details>
      <details>
        <summary style={{ font: "11px 'Space Mono'", color: DIM, cursor: 'pointer', margin: '6px 0' }}>Atom positions ({sites.length})</summary>
        <table style={{ width: '100%', font: "10.5px 'Space Mono'", borderCollapse: 'collapse' }}>
          <tbody>
            <tr style={{ color: FAINT }}><td>el</td><td style={{ textAlign: 'right' }}>x</td><td style={{ textAlign: 'right' }}>y</td><td style={{ textAlign: 'right' }}>z</td></tr>
            {sites.map((s, i) => (
              <tr key={i}><td style={{ color: INK }}>{s.el}</td>{s.pos.map((x, j) => <td key={j} style={{ textAlign: 'right', color: DIM }}>{x.toFixed(4)}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </details>
    </>
  );
}

/* ── inline icons (Lucide-style 24-grid, stroke currentColor) ───────────── */
const svgProps = { width: 14, height: 14, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2, strokeLinecap: 'round', strokeLinejoin: 'round' };
function IconBands() { return <svg {...svgProps}><path d="M4 3v16a2 2 0 0 0 2 2h16" /><path d="M7 16c1.8-7 3.4-7 5 0s3.2 4 5-6" /></svg>; }
function IconBox() { return <svg {...svgProps}><path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z" /><path d="m3.3 7 8.7 5 8.7-5" /><path d="M12 22V12" /></svg>; }
function IconUpload() { return <svg {...svgProps}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><path d="M17 8l-5-5-5 5" /><path d="M12 3v12" /></svg>; }
function IconDownload() { return <svg {...svgProps} width={13} height={13}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><path d="M7 10l5 5 5-5" /><path d="M12 15V3" /></svg>; }
function IconReset() { return <svg {...svgProps} width={12} height={12}><path d="M3 12a9 9 0 1 0 3-6.7L3 8" /><path d="M3 3v5h5" /></svg>; }
