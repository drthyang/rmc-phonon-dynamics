import { useState, useRef, useEffect } from 'react';
import { PhononPipeline } from './compute/pipeline';
import { fromResults } from './io/viewermodel';
import { generatePhonopyBandYaml, downloadString } from './io/writers';
import ThemeCue from './components/ThemeCue';
import RunnerPage from './pages/RunnerPage';
import ViewerPage from './pages/ViewerPage';

export default function App() {
  const [page, setPage] = useState('runner');     // 'runner' | 'viewer'
  const [model, setModel] = useState(null);       // unified viewer model
  const [ready, setReady] = useState(false);
  const pipelineRef = useRef(null);

  const missingFeatures = [
    typeof navigator !== 'undefined' && navigator.gpu ? null : 'WebGPU',
    typeof window !== 'undefined' && typeof window.showDirectoryPicker === 'function' ? null : 'folder access (File System Access API)',
  ].filter(Boolean);

  useEffect(() => {
    const p = new PhononPipeline(() => {});
    pipelineRef.current = p;
    p.initWorkers(4)
      .then(() => setReady(true))
      .catch(e => { console.error('Init failed:', e); setReady(true); });
  }, []);

  const onResults = (results, kpathMeta, dos = null) => {
    const m = fromResults(results, kpathMeta);
    if (dos) m.dos = dos;
    setModel(m);
    setPage('viewer');
    try {
      const yaml = generatePhonopyBandYaml(m.baseStructure, m.qPoints, m.bands, m.eigvecs, m.kpathMeta);
      downloadString(yaml, 'band_gpu.yaml', 'text/yaml');
    } catch (e) { console.error('band.yaml auto-save failed:', e); }
  };

  const loadModel = (m) => { setModel(m); setPage('viewer'); };

  const nModes = model?.bands?.[0]?.length || 0;
  const nKpts = model?.bands?.length || 0;

  return (
    <div className="m-shell">
      {/* ── Topbar: bold tight wordmark, underlined view tabs, ThemeCue + status ── */}
      <header className="m-topbar">
        <div className="m-brand">
          <span className="m-mark">RMC PHONON</span>
          <span className="m-submark">lattice dynamics from total-scattering ensembles</span>
        </div>

        <nav className="m-nav" aria-label="Primary views">
          <button
            type="button"
            className={page === 'runner' ? 'on' : ''}
            onClick={() => setPage('runner')}
          >
            Runner
          </button>
          <button
            type="button"
            className={page === 'viewer' ? 'on' : ''}
            onClick={() => setPage('viewer')}
          >
            Viewer
            {model && (
              <span
                className="tabular-nums"
                style={{
                  fontSize: 11,
                  fontWeight: 600,
                  padding: '1px 6px',
                  borderRadius: 4,
                  background: 'var(--accent)',
                  color: 'var(--ink)',
                  border: '1px solid var(--ink)',
                  lineHeight: 1.3,
                }}
              >
                {nModes}
              </span>
            )}
          </button>
        </nav>

        <div className="m-actions">
          <ThemeCue />
          <span className="m-status" title={ready ? 'WebGPU compute engine ready' : 'Initializing WebGPU workers'}>
            <span className={ready ? 'm-status-dot ready' : 'm-status-dot'} />
            {page === 'viewer' && model && nModes > 0
              ? `${nModes} modes · ${nKpts} k-pts`
              : ready ? 'WebGPU ready' : 'Initializing…'}
          </span>
        </div>
      </header>

      {/* ── Unsupported browser notice ─────────────────────────────────────── */}
      {missingFeatures.length > 0 && (
        <div className="m-notice bad" role="alert">
          <span>
            <strong>Unsupported browser.</strong> Please open in <strong>Chrome or Edge</strong> — this tool requires {missingFeatures.join(' and ')}.
          </span>
        </div>
      )}

      {/* ── Pages stay mounted to preserve datasets, k-paths & WebGL canvases ─ */}
      <div style={{ display: page === 'runner' ? 'flex' : 'none', flexDirection: 'column', flex: 1, minHeight: 0 }}>
        <RunnerPage pipeline={pipelineRef.current} ready={ready} onResults={onResults} onLoadResult={loadModel} />
      </div>
      <div style={{ display: page === 'viewer' ? 'flex' : 'none', flexDirection: 'column', flex: 1, minHeight: 0 }}>
        <ViewerPage model={model} onLoadModel={setModel} />
      </div>

      {/* ── Quiet editorial footer ─────────────────────────────────────────── */}
      <footer className="m-foot">
        <span>
          RMC Phonon Dynamics · © 2026 Tsung-Han Yang ·{' '}
          <a href="https://github.com/drthyang/rmc-phonon-dynamics/blob/main/LICENSE" target="_blank" rel="noopener noreferrer">
            AGPL-3.0
          </a>
        </span>
        <span style={{ display: 'inline-flex', gap: 14 }}>
          <a href="https://github.com/drthyang/rmc-phonon-dynamics" target="_blank" rel="noopener noreferrer">
            Source code
          </a>
        </span>
      </footer>
    </div>
  );
}
