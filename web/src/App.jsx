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

  // The app requires WebGPU + the File System Access API, which today only
  // ship together in Chromium browsers. Detect what's missing so we can tell
  // non-Chromium users (Safari/Firefox) to switch rather than hit a broken app.
  const missingFeatures = [
    typeof navigator !== 'undefined' && navigator.gpu ? null : 'WebGPU',
    typeof window !== 'undefined' && typeof window.showDirectoryPicker === 'function' ? null : 'folder access (File System Access API)',
  ].filter(Boolean);

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

  // Tab pills sit on the cobalt header: active = solid white with accent text,
  // inactive = translucent white text.
  const pill = (active) => ({
    padding: '6px 16px', borderRadius: 7, font: "600 13px 'Space Grotesk'",
    cursor: 'pointer', border: 'none', transition: 'color .15s, background .15s, box-shadow .15s',
    background: active ? '#fff' : 'transparent',
    color: active ? 'var(--accentInk)' : 'rgba(255,255,255,0.82)',
    boxShadow: active ? '0 2px 6px rgba(16,24,38,0.18)' : 'none',
  });

  return (
    <div className="rnr" style={{ minHeight: '100vh', background: 'var(--bg)', color: 'var(--ink)', fontFamily: "'Spline Sans', sans-serif" }}>
      {/* ── top nav ─────────────────────────────────────────────── */}
      <header style={{ height: 64, display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 24px', background: 'linear-gradient(120deg, #1f50c4 0%, #2f6df0 58%, #4884f6 100%)', borderBottom: '1px solid rgba(255,255,255,0.14)', boxShadow: '0 2px 12px rgba(31,80,196,0.28)', position: 'sticky', top: 0, zIndex: 50 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div style={{ width: 34, height: 34, borderRadius: 9, background: 'rgba(255,255,255,0.16)', border: '1px solid rgba(255,255,255,0.30)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.25)' }}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
              <g stroke="#fff" strokeWidth="1.5" opacity="0.95">
                <ellipse cx="12" cy="12" rx="10" ry="4.1" />
                <ellipse cx="12" cy="12" rx="10" ry="4.1" transform="rotate(60 12 12)" />
                <ellipse cx="12" cy="12" rx="10" ry="4.1" transform="rotate(120 12 12)" />
              </g>
              <circle cx="12" cy="12" r="2.1" fill="#fff" />
            </svg>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.1 }}>
            <span style={{ font: "700 17px 'Space Grotesk'", letterSpacing: '-.02em', color: '#fff' }}>RMC Phonon Dynamics</span>
            <span style={{ font: "10px 'Space Mono'", letterSpacing: '.03em', color: 'rgba(255,255,255,0.72)', marginTop: 1 }}>phonons from RMC total-scattering ensembles</span>
          </div>
          <div style={{ display: 'flex', gap: 3, marginLeft: 12, background: 'rgba(255,255,255,0.14)', borderRadius: 9, padding: 3, border: '1px solid rgba(255,255,255,0.20)' }}>
            <button onClick={() => setPage('runner')} style={pill(page === 'runner')}>Runner</button>
            <button onClick={() => setPage('viewer')} style={pill(page === 'viewer')}>Viewer</button>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {page === 'viewer' && model && (
            <span style={{ font: "11px 'Space Mono'", color: 'rgba(255,255,255,0.72)' }}>source {model.source || 'file'}</span>
          )}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '7px 13px', borderRadius: 8, background: 'rgba(255,255,255,0.16)', border: '1px solid rgba(255,255,255,0.22)' }}>
            <div style={{ width: 7, height: 7, borderRadius: '50%', background: ready ? '#fff' : 'rgba(255,255,255,0.5)', boxShadow: ready ? '0 0 0 3px rgba(255,255,255,0.22)' : 'none', animation: ready ? 'blip 2s infinite' : 'none' }} />
            <span style={{ font: "600 12px 'Space Mono'", color: ready ? '#fff' : 'rgba(255,255,255,0.7)' }}>
              {page === 'viewer' && model
                ? `${model.bands[0].length} modes · ${model.bands.length} k-pts`
                : ready ? 'WebGPU ready' : 'initializing…'}
            </span>
          </div>
        </div>
      </header>

      {/* Unsupported-browser notice (Safari/Firefox lack WebGPU and/or folder access). */}
      {missingFeatures.length > 0 && (
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12, padding: '12px 24px', background: 'rgba(240,102,59,0.10)', borderBottom: '1px solid rgba(240,102,59,0.30)', color: 'var(--warnInk)' }}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flex: 'none', marginTop: 1 }}>
            <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" /><path d="M12 9v4" /><path d="M12 17h.01" />
          </svg>
          <div style={{ font: "13px/1.55 'Spline Sans'" }}>
            <span style={{ fontWeight: 600 }}>Unsupported browser.</span>{' '}
            Please use <b>Chrome or Edge</b> — this app needs {missingFeatures.join(' and ')}.
          </div>
        </div>
      )}

      {/* Both pages stay mounted; we toggle visibility so switching tabs never
          remounts a page (Runner keeps its loaded dataset, k-path, run state, and
          the live three.js canvases instead of refreshing). */}
      <div style={{ display: page === 'runner' ? 'block' : 'none' }}>
        <RunnerPage pipeline={pipelineRef.current} ready={ready} onResults={onResults} onLoadResult={loadModel} />
      </div>
      <div style={{ display: page === 'viewer' ? 'block' : 'none' }}>
        <main style={{ maxWidth: 1320, margin: '0 auto', padding: 22 }}><ViewerPage model={model} onLoadModel={setModel} /></main>
      </div>

      {/* Footer — copyright + license, shown under whichever page is active. */}
      <footer style={{ borderTop: '1px solid var(--border)', padding: '16px 24px', display: 'flex', justifyContent: 'center' }}>
        <span style={{ font: "11px 'Space Mono'", color: 'var(--faint)' }}>
          © 2026 Tsung-Han Yang ·{' '}
          <a href="https://github.com/drthyang/rmc-phonon-dynamics/blob/main/LICENSE" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--accentInk)', textDecoration: 'none' }}>MIT License</a> ·{' '}
          <a href="https://github.com/drthyang/rmc-phonon-dynamics" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--accentInk)', textDecoration: 'none' }}>source</a>
        </span>
      </footer>
    </div>
  );
}
