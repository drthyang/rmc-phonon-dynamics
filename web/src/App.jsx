import { useState, useRef, useEffect } from 'react';
import { PhononPipeline } from './compute/pipeline';
import { fromResults } from './io/viewermodel';
import { generatePhonopyBandYaml, downloadString } from './io/writers';
import RunnerPage from './pages/RunnerPage';
import ViewerPage from './pages/ViewerPage';

export default function App() {
  const [page, setPage] = useState('runner');     // 'runner' | 'viewer'
  const [model, setModel] = useState(null);        // unified viewer model
  const [ready, setReady] = useState(false);
  const pipelineRef = useRef(null);

  useEffect(() => {
    const p = new PhononPipeline(() => {});
    pipelineRef.current = p;
    p.initWorkers(4).then(() => setReady(true)).catch(e => { console.error('Init failed:', e); setReady(true); });
  }, []);

  const onResults = (results, kpathMeta, dos = null) => {
    const m = fromResults(results, kpathMeta);
    if (dos) m.dos = dos;   // phonon DOS computed alongside the run (optional)
    setModel(m);
    setPage('viewer');
    // Auto-save the band.yaml after a successful run.
    try {
      const yaml = generatePhonopyBandYaml(m.baseStructure, m.qPoints, m.bands, m.eigvecs, m.kpathMeta);
      downloadString(yaml, 'band_gpu.yaml', 'text/yaml');
    } catch (e) { console.error('band.yaml auto-save failed:', e); }
  };

  const loadModel = (m) => { setModel(m); setPage('viewer'); };

  const pill = (active) => ({
    padding: '6px 16px', borderRadius: 7, font: "600 13px 'Space Grotesk'",
    cursor: 'pointer', border: 'none', transition: 'color .15s, background .15s, box-shadow .15s',
    background: active ? 'var(--accent)' : 'transparent',
    color: active ? '#fff' : 'var(--dim)',
    boxShadow: active ? '0 2px 6px rgba(47,109,240,0.30)' : 'none',
  });

  return (
    <div className="rnr" style={{ minHeight: '100vh', background: 'var(--bg)', color: 'var(--ink)', fontFamily: "'Spline Sans', sans-serif" }}>
      {/* ── top nav ─────────────────────────────────────────────── */}
      <header style={{ height: 64, display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 24px', background: 'rgba(255,255,255,0.85)', backdropFilter: 'blur(12px) saturate(1.4)', WebkitBackdropFilter: 'blur(12px) saturate(1.4)', borderBottom: '1px solid var(--border)', boxShadow: '0 1px 3px rgba(16,24,38,0.05)', position: 'sticky', top: 0, zIndex: 50 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div style={{ width: 34, height: 34, borderRadius: 9, background: 'linear-gradient(135deg, #5a8bf7 0%, #2f6df0 55%, #2257cf 100%)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 4px 12px rgba(47,109,240,0.35), inset 0 1px 0 rgba(255,255,255,0.3)' }}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
              <g style={{ transformBox: 'fill-box', transformOrigin: 'center', animation: 'spin 9s linear infinite' }} stroke="#fff" strokeWidth="1.5" opacity="0.95">
                <ellipse cx="12" cy="12" rx="10" ry="4.1" />
                <ellipse cx="12" cy="12" rx="10" ry="4.1" transform="rotate(60 12 12)" />
                <ellipse cx="12" cy="12" rx="10" ry="4.1" transform="rotate(120 12 12)" />
              </g>
              <circle cx="12" cy="12" r="2.1" fill="#fff" />
            </svg>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.1 }}>
            <span style={{ font: "700 17px 'Space Grotesk'", letterSpacing: '-.02em', color: 'var(--ink)' }}>RMC Phonon Dynamics</span>
            <span style={{ font: "10px 'Space Mono'", letterSpacing: '.03em', color: 'var(--faint)', marginTop: 1 }}>WebGPU phonon-band calculator</span>
          </div>
          <div style={{ display: 'flex', gap: 3, marginLeft: 12, background: 'var(--inset2)', borderRadius: 9, padding: 3, border: '1px solid var(--border)' }}>
            <button onClick={() => setPage('runner')} style={pill(page === 'runner')}>Runner</button>
            <button onClick={() => setPage('viewer')} style={pill(page === 'viewer')}>Viewer</button>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '7px 13px', borderRadius: 8, background: 'var(--soft)', border: '1px solid rgba(47,109,240,0.16)' }}>
          <div style={{ width: 7, height: 7, borderRadius: '50%', background: ready ? 'var(--accent)' : 'var(--faint)', boxShadow: ready ? '0 0 0 3px rgba(47,109,240,0.15)' : 'none', animation: ready ? 'blip 2s infinite' : 'none' }} />
          <span style={{ font: "600 12px 'Space Mono'", color: ready ? 'var(--accentInk)' : 'var(--faint)' }}>{ready ? 'WebGPU ready' : 'initializing…'}</span>
        </div>
      </header>

      {/* Both pages stay mounted; we toggle visibility so switching tabs never
          remounts a page (Runner keeps its loaded dataset, k-path, run state, and
          the live three.js canvases instead of refreshing). */}
      <div style={{ display: page === 'runner' ? 'block' : 'none' }}>
        <RunnerPage pipeline={pipelineRef.current} ready={ready} onResults={onResults} onLoadResult={loadModel} />
      </div>
      <div style={{ display: page === 'viewer' ? 'block' : 'none' }}>
        <main className="max-w-[1400px] w-full mx-auto p-6"><ViewerPage model={model} onLoadModel={setModel} /></main>
      </div>
    </div>
  );
}
